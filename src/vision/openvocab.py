"""A detector you tell what to look for, in words.

Phase 15.3. Stock YOLO knows COCO's eighty classes -- people, vehicles,
household objects -- and has never heard of a cable tray. On a fit-out it
returns `book`, `keyboard`, `skis`: confident guesses at materials it has no
vocabulary for.

YOLO-World takes the class list as TEXT at inference time. Hand it
"suspended ceiling grid", "cable tray", "electrical conduit" and it looks for
those, with no training and no labelled data. That is the only way to get
useful classes out of this system before a dataset exists -- and building the
dataset is months of somebody's attention.

WHAT IT IS NOT

    It is not as good as a model trained on real examples of YOUR sites, and it
    is slower. Its purpose is to be useful immediately and to generate
    PRE-LABELS: a mostly-right box a person corrects in seconds rather than
    draws in a minute. Those corrections become the training set, and the
    trained model then replaces this one for the per-walk work, where speed
    matters and a fixed class list is fine.

    So this is a bootstrap, deliberately. It earns its place by making the
    labelling faster, not by being the final answer.

PROMPTS ARE NOT CLASS NAMES

    "ceiling grid" and "suspended ceiling grid tee bar" find different things.
    The prompt is a knob, and `FIT_OUT_PROMPTS` is a starting position rather
    than a decision -- the bench exists so it can be turned while looking at
    real photographs.
"""

from __future__ import annotations

import numpy as np

from vision.detect import DetectionError, RawDetection, ensure_weights

# Small and quick. The larger `-l` and `-x` variants are more accurate and
# several times slower; on a bench where somebody is iterating on prompts, the
# round trip matters more than the last few points of recall.
DEFAULT_WEIGHTS = "yolov8s-worldv2.pt"

# A starting vocabulary for interior fit-out, grouped the way the trades are.
# Deliberately wordy: an open-vocabulary model is matching a phrase against what
# it sees, and "metal cable tray on ceiling" grounds better than "tray".
FIT_OUT_PROMPTS: dict[str, list[str]] = {
    "furniture": [
        "office desk workstation",
        "office chair",
        "storage cabinet",
    ],
    "ceilings": [
        "suspended ceiling grid",
        "ceiling tile panel",
        "exposed ceiling services",
    ],
    "partitions": [
        "metal stud wall framing",
        "plasterboard drywall sheet",
        "unfinished brick wall",
    ],
    "mep": [
        "metal cable tray",
        "electrical conduit pipe",
        "ventilation duct",
        "ceiling light fitting",
        "electrical socket outlet",
        "sprinkler head",
    ],
    "site": [
        "construction worker",
        "hard hat",
        "ladder",
        "scaffolding",
        "power tool",
        "wooden plank",
        "cement bag",
    ],
}


def default_prompts() -> list[str]:
    return [phrase for group in FIT_OUT_PROMPTS.values() for phrase in group]


class OpenVocabProvider:
    """YOLO-World, told what to look for."""

    # Opt in to cross-label merging (see `merge_duplicates`). This model's
    # labels are whichever of thirty prompts scored highest in one view, and
    # neighbouring views routinely disagree -- "office chair" here, "office
    # desk workstation" there -- so same-label-only merging reports one object
    # once per phrase it resembled. Two near-identical boxes (IoU at or above
    # this) are one object whatever they were called; the labels then vote.
    # High on purpose: at ordinary overlap levels different labels still mean
    # different things, socket-in-front-of-cable-tray being the case to protect.
    cross_label_iou = 0.55

    def __init__(
        self,
        prompts: list[str] | None = None,
        weights: str = DEFAULT_WEIGHTS,
        device: str | None = None,
        confidence: float = 0.05,
    ) -> None:
        # A LOW DEFAULT THRESHOLD, on purpose. Open-vocabulary scores are not
        # comparable with a trained detector's -- a correct cable tray commonly
        # lands around 0.1 where COCO would say 0.9. Filtering at 0.25 throws
        # away most of what this model gets right.
        self.prompts = prompts or default_prompts()
        self.weights = weights
        self.confidence = confidence
        self._device = device
        self._model = None

    @property
    def name(self) -> str:
        return f"world:{self.weights}({len(self.prompts)} prompts)"

    def _load(self):
        if self._model is not None:
            return self._model
        try:
            from ultralytics import YOLOWorld
        except ImportError as exc:  # pragma: no cover - optional install
            raise DetectionError(
                "ultralytics is not installed, so open-vocabulary detection is "
                "unavailable."
            ) from exc

        local = ensure_weights(self.weights)
        try:
            model = YOLOWorld(str(local))
            model.set_classes(self.prompts)
        except Exception as exc:
            raise DetectionError(f"could not load {local}: {exc}") from exc

        if self._device is None:
            import torch

            self._device = "cuda" if torch.cuda.is_available() else "cpu"

        self._model = model
        return model

    @property
    def device(self) -> str:
        self._load()
        return self._device or "cpu"

    def detect(self, images: list[np.ndarray]) -> list[list[RawDetection]]:
        if not images:
            return []
        model = self._load()

        # RGB -> BGR, same as the COCO provider: ultralytics reads numpy input
        # the way OpenCV hands it over, and our panoramas are RGB throughout.
        batch = [image[..., ::-1] for image in images]

        try:
            results = model.predict(
                batch, device=self._device, conf=self.confidence, verbose=False
            )
        except Exception as exc:
            raise DetectionError(f"{self.name} failed: {exc}") from exc

        out: list[list[RawDetection]] = []
        for result in results:
            found: list[RawDetection] = []
            boxes = getattr(result, "boxes", None)
            if boxes is not None and len(boxes):
                names = result.names
                xyxy = boxes.xyxy.cpu().numpy()
                scores = boxes.conf.cpu().numpy()
                classes = boxes.cls.cpu().numpy().astype(int)
                for (x0, y0, x1, y1), score, index in zip(
                    xyxy, scores, classes, strict=True
                ):
                    found.append(
                        RawDetection(
                            label=str(names.get(int(index), index)),
                            confidence=float(score),
                            x0=float(x0), y0=float(y0), x1=float(x1), y1=float(y1),
                        )
                    )
            out.append(found)
        return out
