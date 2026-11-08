"""A YOLO detector behind the `DetectionProvider` seam.

The import of `ultralytics` is LAZY, and deliberately so. It drags in torch --
some three gigabytes with the CUDA runtime -- and neither the API process nor
the test suite has any business paying for that. Only the worker that actually
runs a frame through a model needs it, and `scripts/test.ps1` stays fast because
everything in `detect.py` is exercised through a fake provider instead.

WHAT THIS MODEL ACTUALLY KNOWS, WHICH IS NOT MUCH

    Stock YOLO weights are trained on COCO: eighty classes of people, vehicles,
    animals and household objects. It has never heard of a ceiling grid, cable
    tray, or workstation, and on a fit-out site it will mostly find `person` and
    `chair` and hallucinate the odd `bench`.

    That is expected at 15.2 and is not what this step is for. This proves the
    PLUMBING -- weights resolve inside the project, the GPU is used, sixteen
    views go through a model in one batch, boxes come back as directions on the
    sphere. Useful classes arrive in 15.3 with an open-vocabulary model, which
    plugs into the same seam.
"""

from __future__ import annotations

import numpy as np

from vision.detect import DetectionError, RawDetection, ensure_weights

DEFAULT_WEIGHTS = "yolo11n.pt"


class YoloProvider:
    """Ultralytics YOLO, batched over the views of one panorama."""

    def __init__(
        self,
        weights: str = DEFAULT_WEIGHTS,
        device: str | None = None,
        confidence: float = 0.25,
        url: str | None = None,
    ) -> None:
        self.weights = weights
        self.confidence = confidence
        self.url = url
        self._device = device
        self._model = None

    @property
    def name(self) -> str:
        return f"yolo:{self.weights}"

    # -- loading -----------------------------------------------------------

    def _load(self):
        if self._model is not None:
            return self._model

        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover - depends on the install
            raise DetectionError(
                "ultralytics is not installed. Run "
                "`uv pip install --torch-backend=auto ultralytics`."
            ) from exc

        local = ensure_weights(self.weights, self.url)

        try:
            self._model = YOLO(str(local))
        except Exception as exc:
            raise DetectionError(f"could not load {local}: {exc}") from exc

        if self._device is None:
            import torch

            self._device = "cuda" if torch.cuda.is_available() else "cpu"

        return self._model

    @property
    def device(self) -> str:
        self._load()
        return self._device or "cpu"

    # -- detection ---------------------------------------------------------

    def detect(self, images: list[np.ndarray]) -> list[list[RawDetection]]:
        if not images:
            return []

        model = self._load()

        # RGB -> BGR. Ultralytics reads numpy input as BGR, the way OpenCV
        # hands it over; our panoramas are RGB all the way from pyvips. Feeding
        # it the wrong order does not fail, it just quietly detects slightly
        # different things -- red and blue swapped is invisible in a bounding
        # box and very visible in the accuracy.
        batch = [image[..., ::-1] for image in images]

        try:
            results = model.predict(
                batch,
                device=self._device,
                conf=self.confidence,
                verbose=False,
            )
        except Exception as exc:
            raise DetectionError(f"{self.name} failed on {len(images)} views: {exc}") from exc

        out: list[list[RawDetection]] = []
        for result in results:
            found: list[RawDetection] = []
            boxes = getattr(result, "boxes", None)
            if boxes is not None and len(boxes):
                names = result.names
                xyxy = boxes.xyxy.cpu().numpy()
                confidences = boxes.conf.cpu().numpy()
                classes = boxes.cls.cpu().numpy().astype(int)
                for (x0, y0, x1, y1), score, index in zip(
                    xyxy, confidences, classes, strict=True
                ):
                    found.append(
                        RawDetection(
                            label=str(names.get(int(index), index)),
                            confidence=float(score),
                            x0=float(x0),
                            y0=float(y0),
                            x1=float(x1),
                            y1=float(y1),
                        )
                    )
            out.append(found)

        return out
