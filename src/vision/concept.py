"""Say what to find, get back MASKS -- the Grounded SAM 2 pipeline.

Phase 15.6. YOLO-World (openvocab.py) answers a text prompt with boxes, and a
box is the wrong shape for most of what a fit-out is made of: a box around
a long run of cable tray is almost all not cable tray, which is why tray never merged and
never measured. This module answers the same prompts with pixel masks.

TWO MODELS, ONE SEAM

    Grounding DINO  reads the prompt list and returns boxes -- an open-set
                    detector trained to GROUND phrases, so "metal cable tray"
                    lands far better than YOLO-World's embedding match.
    SAM 2           takes each box as a prompt and returns the pixels of the
                    object inside it. It knows nothing about words; that is
                    Grounding DINO's job. Composing them is the standard
                    "Grounded SAM" recipe.

    Both are plain transformers checkpoints: no compiled CUDA extensions
    (which is what makes the original Grounded-SAM repo a Windows fight), no
    gated weights, permissive licences (Apache-2.0 both).

WHY THE PROVIDER RETURNS PER-VIEW MASKS, NOT DETECTIONS

    Same reason `RawDetection` exists: a provider's job ends at "these pixels
    of the image you gave me". Converting view pixels to the sphere, merging
    the overlapping views and tracing outlines is `fuse_instances`' job, and
    keeping them apart means SAM 3 -- same masks-from-concepts contract, one
    model instead of two -- slots in as a second provider with zero changes
    downstream. That is the upgrade path, on purpose.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vision.detect import Detection, DetectionError
from vision.projection import (
    SphereBox,
    ViewSpec,
    column_to_lon_deg,
    row_to_lat_deg,
    wrap_lon_deg,
)

# Tiny, not base. Base cost 328 seconds for one panorama on the laptop GPU at
# fp32 -- a bench nobody will wait for. Tiny at fp16 is an order of magnitude
# quicker and the grounding quality difference on wordy fit-out phrases did
# not survive contact with real photographs.
GROUNDING_WEIGHTS = "IDEA-Research/grounding-dino-tiny"
SAM2_WEIGHTS = "facebook/sam2.1-hiera-small"

# Grounding DINO scores are calibrated far better than YOLO-World's -- a real
# object usually lands 0.35+ -- so the floor can mean something.
DEFAULT_BOX_THRESHOLD = 0.30
DEFAULT_TEXT_THRESHOLD = 0.25


def _snap_to_prompt(label: str, prompts: list[str]) -> str:
    """Map Grounding DINO's returned span back onto an actual prompt.

    The model reads the prompt list as one token stream, and a detection's
    label is whichever SPAN of it scored -- which can run across phrase
    boundaries, yielding labels like "ceiling grid ceiling panel ceiling"
    that exist in no prompt. Snap to the prompt sharing the most words; a
    label with no overlap at all is kept, since inventing a phrase the model
    did not say would be worse.
    """
    words = set(label.split())
    best, overlap = label, 0
    for prompt in prompts:
        shared = len(words & set(prompt.split()))
        if shared > overlap:
            best, overlap = prompt, shared
    return best


@dataclass(frozen=True, slots=True)
class InstanceMask:
    """One thing found in ONE view: a phrase, a score, and its pixels."""

    label: str
    confidence: float
    mask: np.ndarray  # bool, (view_h, view_w)
    # When the concept layer relabelled this instance, the phrasing the model
    # actually answered to. Empty means label IS what was asked.
    phrase: str = ""


class GroundedSam2Provider:
    """Grounding DINO for the words, SAM 2 for the pixels."""

    def __init__(
        self,
        prompts: list[str] | None = None,
        device: str | None = None,
        confidence: float = DEFAULT_BOX_THRESHOLD,
    ) -> None:
        from vision.openvocab import default_prompts

        self.prompts = [p.strip().lower() for p in (prompts or default_prompts()) if p.strip()]
        self.confidence = confidence
        self.weights = f"{GROUNDING_WEIGHTS} + {SAM2_WEIGHTS}"
        self._device = device
        self._grounder = None
        self._grounder_processor = None
        self._segmenter = None
        self._segmenter_processor = None

    @property
    def name(self) -> str:
        return f"grounded-sam2({len(self.prompts)} prompts)"

    def _load(self):
        if self._grounder is not None:
            return
        try:
            import torch
            from transformers import (
                AutoModelForZeroShotObjectDetection,
                AutoProcessor,
                Sam2Model,
                Sam2Processor,
            )
        except ImportError as exc:  # pragma: no cover - optional install
            raise DetectionError(
                "transformers is not installed, so Grounded SAM 2 is unavailable."
            ) from exc

        if self._device is None:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"

        try:
            self._grounder_processor = AutoProcessor.from_pretrained(GROUNDING_WEIGHTS)
            self._grounder = (
                AutoModelForZeroShotObjectDetection.from_pretrained(GROUNDING_WEIGHTS)
                .to(self._device)
                .eval()
            )
            self._segmenter_processor = Sam2Processor.from_pretrained(SAM2_WEIGHTS)
            self._segmenter = Sam2Model.from_pretrained(SAM2_WEIGHTS).to(self._device).eval()
        except Exception as exc:
            raise DetectionError(
                f"could not load Grounded SAM 2 ({GROUNDING_WEIGHTS} + {SAM2_WEIGHTS}): {exc}"
            ) from exc

    @property
    def device(self) -> str:
        self._load()
        return self._device or "cpu"

    def segment(self, images: list[np.ndarray]) -> list[list[InstanceMask]]:
        """One list of instances per image, in the order given."""
        if not images:
            return []
        import torch
        from PIL import Image

        self._load()
        assert self._grounder_processor is not None and self._segmenter_processor is not None

        # Grounding DINO's text convention: lowercase phrases joined by ". ".
        text = ". ".join(self.prompts) + "."
        pils = [Image.fromarray(image) for image in images]

        out: list[list[InstanceMask]] = []
        try:
            with torch.inference_mode():
                inputs = self._grounder_processor(
                    images=pils, text=[text] * len(pils), return_tensors="pt"
                ).to(self._device)
                # Autocast rather than half-precision WEIGHTS: Grounding DINO
                # mixes text and vision towers and casting the weights
                # themselves trips "expected Half but found Float" inside the
                # fusion layers. Autocast picks per-op precision and gets the
                # same speed-up without the crash. No-op on CPU.
                with torch.autocast(
                    self._device or "cpu", dtype=torch.float16, enabled=self._device == "cuda"
                ):
                    grounded = self._grounder(**inputs)
                results = self._grounder_processor.post_process_grounded_object_detection(
                    grounded,
                    inputs.input_ids,
                    threshold=self.confidence,
                    text_threshold=DEFAULT_TEXT_THRESHOLD,
                    target_sizes=[image.shape[:2] for image in images],
                )

                for _image, pil, found in zip(images, pils, results, strict=True):
                    boxes = found["boxes"].cpu()
                    scores = found["scores"].cpu().numpy()
                    labels = found.get("text_labels") or found.get("labels") or []
                    if len(boxes) == 0:
                        out.append([])
                        continue

                    sam_inputs = self._segmenter_processor(
                        images=pil,
                        input_boxes=[boxes.tolist()],
                        return_tensors="pt",
                    ).to(self._device)
                    sam_out = self._segmenter(**sam_inputs, multimask_output=False)
                    masks = self._segmenter_processor.post_process_masks(
                        sam_out.pred_masks.cpu(),
                        sam_inputs["original_sizes"].cpu(),
                    )[0]
                    # (n_boxes, 1, H, W) -> (n_boxes, H, W)
                    masks = masks.squeeze(1).numpy().astype(bool)

                    instances = [
                        InstanceMask(
                            label=_snap_to_prompt(str(label), self.prompts),
                            confidence=float(score),
                            mask=mask,
                        )
                        for label, score, mask in zip(labels, scores, masks, strict=True)
                        if mask.any()
                    ]
                    out.append(instances)
        except DetectionError:
            raise
        except Exception as exc:
            raise DetectionError(f"{self.name} failed: {exc}") from exc
        return out


# 3, not 3.1 -- checked, not preferred. facebook/sam3.1 ships only Meta's raw
# research checkpoint (sam3.1_multiplex.pt, no safetensors), which
# `Sam3Model.from_pretrained` cannot read; the only converted 3.1 repos are
# unofficial community quantisations. 3.1's gains are video multiplexing and
# real-time tracking, not per-image quality, so nothing is lost here. When
# Meta publishes a converted 3.1, upgrading is this one string.
SAM3_WEIGHTS = "facebook/sam3"


class Sam3Provider:
    """SAM 3: one model, concepts in, instance masks out.

    Same `segment` contract as `GroundedSam2Provider`, so everything
    downstream is shared. SAM 3 takes ONE concept phrase per forward pass, so
    a prompt list costs one batched pass over all views per prompt -- prompts
    are a budget here, not a free list. Its presence head means scores are
    calibrated and hard negatives ("sprinkler head" in a room with none)
    mostly stay silent, which is exactly YOLO-World's weakness.

    THE WEIGHTS ARE GATED. Meta requires accepting their licence on the
    model's Hugging Face page, and downloads need a logged-in token. The
    loader turns the resulting 401/403 into instructions rather than a
    traceback, because that failure is expected on every fresh machine.
    """

    def __init__(
        self,
        prompts: list[str] | None = None,
        weights: str = SAM3_WEIGHTS,
        device: str | None = None,
        confidence: float = 0.4,
    ) -> None:
        from vision.openvocab import default_prompts

        self.prompts = [p.strip().lower() for p in (prompts or default_prompts()) if p.strip()]
        self.confidence = confidence
        self.weights = weights
        self._device = device
        self._model = None
        self._processor = None

    @property
    def name(self) -> str:
        return f"sam3({len(self.prompts)} prompts)"

    def _load(self):
        if self._model is not None:
            return
        try:
            import torch
            from transformers import Sam3Model, Sam3Processor
        except ImportError as exc:  # pragma: no cover - optional install
            raise DetectionError(
                "transformers is not installed, so SAM 3 is unavailable."
            ) from exc

        if self._device is None:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            self._processor = Sam3Processor.from_pretrained(self.weights)
            self._model = Sam3Model.from_pretrained(self.weights).to(self._device).eval()
        except Exception as exc:
            message = str(exc)
            gated_words = ("gated", "401", "403", "restricted", "access")
            if any(word in message.lower() for word in gated_words):
                raise DetectionError(
                    "SAM 3's weights are gated by Meta. Request access at "
                    "https://huggingface.co/facebook/sam3.1 (sign in, accept the "
                    "licence), then run `hf auth login` in a project shell and "
                    "restart the API. Until then, Grounded SAM 2 answers the "
                    "same prompts."
                ) from exc
            raise DetectionError(f"could not load {self.weights}: {exc}") from exc

    @property
    def device(self) -> str:
        self._load()
        return self._device or "cpu"

    def segment(self, images: list[np.ndarray]) -> list[list[InstanceMask]]:
        if not images:
            return []
        import torch
        from PIL import Image

        self._load()
        assert self._processor is not None

        pils = [Image.fromarray(image) for image in images]
        out: list[list[InstanceMask]] = [[] for _ in images]
        # Small chunks, not one 16-view batch. SAM 3 is ~850M parameters and
        # its activations at full batch overflow a laptop GPU's 8 GB into
        # shared system memory, where "GPU inference" quietly becomes slower
        # than CPU. Four views at a time stays resident. Autocast for the
        # same reason as the grounder: half the memory, none of the dtype
        # fights.
        chunk = 4 if (self._device or "").startswith("cuda") else 2
        try:
            with torch.inference_mode():
                for prompt in self.prompts:
                    for start in range(0, len(pils), chunk):
                        batch = pils[start : start + chunk]
                        sizes = [
                            image.shape[:2] for image in images[start : start + chunk]
                        ]
                        inputs = self._processor(
                            images=batch, text=[prompt] * len(batch), return_tensors="pt"
                        ).to(self._device)
                        with torch.autocast(
                            self._device or "cpu",
                            dtype=torch.float16,
                            enabled=(self._device or "").startswith("cuda"),
                        ):
                            answer = self._model(**inputs)
                        results = self._processor.post_process_instance_segmentation(
                            answer,
                            threshold=self.confidence,
                            mask_threshold=0.5,
                            target_sizes=sizes,
                        )
                        for offset, found in enumerate(results):
                            index = start + offset
                            raw_masks, raw_scores = found["masks"], found["scores"]
                            if hasattr(raw_masks, "cpu"):
                                raw_masks = raw_masks.cpu()
                            if hasattr(raw_scores, "cpu"):
                                raw_scores = raw_scores.cpu()
                            masks = np.asarray(raw_masks).astype(bool)
                            scores = np.asarray(raw_scores, dtype=float)
                            for mask, score in zip(masks, scores, strict=True):
                                if mask.any():
                                    out[index].append(
                                        InstanceMask(
                                            label=prompt,
                                            confidence=float(score),
                                            mask=mask,
                                        )
                                    )
        except DetectionError:
            raise
        except Exception as exc:
            raise DetectionError(f"{self.name} failed: {exc}") from exc
        return out


# ---------------------------------------------------------------------------
# Putting per-view instances back together on the sphere
# ---------------------------------------------------------------------------

# Same-phrase duplicates across views merge at ordinary overlap; different
# phrases only when the masks are near-identical -- the same reasoning as
# `merge_duplicates`' cross_label_iou, but exact, because these are pixels.
SAME_LABEL_IOU = 0.45
CROSS_LABEL_IOU = 0.70
# Below this many canvas pixels an instance is projection noise, not a thing.
MIN_CANVAS_PIXELS = 12


def fuse_instances(
    instances_by_view: list[list[InstanceMask]],
    views: list[ViewSpec],
    canvas_width: int | None = None,
    seed_confidence: float | None = None,
    pipe_labels: frozenset[str] = frozenset(),
    max_pipe_thickness_deg: float = 1.7,
    stats: dict | None = None,
    second_look: frozenset[ViewSpec] = frozenset(),
) -> list[Detection]:
    """Per-view instance masks -> one detection per real object, with outlines.

    Every view pixel knows its direction, so each instance's mask becomes the
    set of panorama pixels it covers (src/vision/stitch.py), and "two views saw
    the same thing" is exact mask overlap there -- no rectangles, no pole
    trouble. Clusters (union-find, so A-B-C chains hold) fuse to the UNION of
    their masks, the label the members backed with the most summed confidence,
    an outline for every piece, and the length of the centre line.

    Overlap is measured WHERE BOTH VIEWS CAN SEE. A pipe crossing a view edge is
    seen as its left stretch in one view and its right stretch in the next;
    compared whole, the two masks barely overlap and the pipe stayed in pieces.
    Compared in the region both views cover, they coincide.

    `canvas_width` defaults to the views' own sharpness (`canvas_width_for`):
    2560 px for 640-px views, where the old fixed 1024 turned a thin pipe into
    a dotted line.

    GROWING, when `seed_confidence` is given (design note 15.16). The
    model sees more of a pipe than it is sure of: on the site photo, what
    scored 0.15-0.4 was largely the continuation of what scored above. So
    masks at or above `seed_confidence` are SEEDS, found directly, and cluster
    exactly as without growing. A weaker mask is kept only if it touches a
    seed -- or a mask already grown from one -- is the same kind of thing, and
    is no more than twice as thick as what it touches, pipe-thin. Every seed
    cluster grows at once, a step at a time, and a weak mask belongs to the
    first to reach it: letting weak masks JOIN clusters turned the site
    photo's ceiling into two giant webs. Weak masks connected to nothing are
    dropped. Every detection records how much of it was found directly and
    how much grown. Without a seed score every mask is found directly,
    exactly as before.

    THE PIPE-THIN CHECK, for the labels in `pipe_labels`: a mask of a pipe-like
    item thicker than `max_pipe_thickness_deg` (area over centre line, in
    degrees) is not pipe and is dropped before anything else. On the site
    photos "red pipe" also traced the broad red-painted vault bands; this
    removes most of them, but can also drop two pipe runs lying side by side
    (research experiment 05). `stats`, when given, gets the count.

    `second_look` names the views that were a second look along pipes; each
    detection counts its masks from them (`look_views`). `stats`, when given,
    also gets `pixels`: each returned detection's sure pixels, in the same
    order -- what the second look needs to find a pipe's ends -- and
    `grown_steps`: for each detection, every followed mask's pixels with the
    step of growing that reached it, which the research page draws stage by
    stage.
    """
    from vision.stitch import (
        canvas_width_for,
        dilate,
        inside_point,
        length_deg,
        outlines,
        overlap_where_both_look,
        paint,
        shared_count,
        thickness_px,
        touches,
    )

    if len(instances_by_view) != len(views):
        raise ValueError(
            f"{len(instances_by_view)} instance lists for {len(views)} views -- "
            "the provider must answer in the order it was asked"
        )

    width = canvas_width or canvas_width_for(views)
    height = width // 2

    # Each instance as the sorted panorama pixels it covers, with its view.
    painted: list[InstanceMask] = []
    pixels: list[np.ndarray] = []
    seen_by: list[ViewSpec] = []
    # Thickness measured once per mask: by the pipe-thin check here, reused
    # when following pipes below. Measuring it twice was half the thinning.
    thickness: dict[int, float] = {}
    too_thick = 0
    pixels_per_degree = width / 360.0
    for instances, spec in zip(instances_by_view, views, strict=True):
        for instance in instances:
            covered = paint(instance.mask, spec, width)
            if covered.size < MIN_CANVAS_PIXELS:
                continue
            measured = None
            if instance.label in pipe_labels:
                measured = thickness_px(covered, width)
                if measured / pixels_per_degree > max_pipe_thickness_deg:
                    too_thick += 1
                    continue
            if measured is not None:
                thickness[len(painted)] = measured
            painted.append(instance)
            pixels.append(covered)
            seen_by.append(spec)

    if stats is not None:
        stats["too_thick"] = too_thick
    if not painted:
        return []

    strong = [
        seed_confidence is None or instance.confidence >= seed_confidence
        for instance in painted
    ]
    growing = not all(strong)

    # Only pairs that share a coarse tile can overlap -- or, when growing,
    # touch -- so only those are checked. Growing tiles the pixels one step out,
    # so pieces that meet at a tile edge still find each other.
    tile = max(8, width // 128)
    tiles_wide = width // tile + 1
    by_tile: dict[int, list[int]] = {}
    # Each mask grown by a pixel once, reused for every "do these touch".
    reaches: list[np.ndarray] = []
    for index, covered in enumerate(pixels):
        reach = dilate(covered, width) if growing else covered
        reaches.append(reach)
        for key in np.unique((reach // width // tile) * tiles_wide + (reach % width) // tile):
            by_tile.setdefault(int(key), []).append(index)
    candidates = {
        (a, b)
        for members in by_tile.values()
        for i, a in enumerate(members)
        for b in members[i + 1 :]
    }

    parent = list(range(len(painted)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def thick(index: int) -> float:
        if index not in thickness:
            thickness[index] = thickness_px(pixels[index], width)
        return thickness[index]

    # Found directly: seeds cluster by overlap, exactly as without growing.
    touching: dict[int, list[int]] = {}
    for i, j in sorted(candidates):
        same_label = painted[i].label == painted[j].label
        if not (strong[i] and strong[j]):
            # Growing only ever runs along one kind of thing.
            if same_label and touches(pixels[i], pixels[j], width, reaches[i]):
                touching.setdefault(i, []).append(j)
                touching.setdefault(j, []).append(i)
            continue
        if same_label:
            measured = overlap_where_both_look(pixels[i], seen_by[i], pixels[j], seen_by[j], width)
            if measured is None:
                continue
            iou, contained = measured
            # Containment rescues the truncated case exactly as in the box
            # merge: a mask clipped by its view's edge sits inside the whole one.
            same = iou >= SAME_LABEL_IOU or contained >= 0.8
        else:
            # Different phrases only when the masks are near-identical as a
            # whole -- a socket in front of a tray is two things.
            shared = shared_count(pixels[i], pixels[j])
            same = shared / (pixels[i].size + pixels[j].size - shared) >= CROSS_LABEL_IOU
        if same:
            ra, rb = find(i), find(j)
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)

    clusters: dict[int, list[int]] = {}
    for index in range(len(painted)):
        if strong[index]:
            clusters.setdefault(find(index), []).append(index)

    # Grown: every seed cluster reaches out at once, one step at a time, and a
    # weak piece belongs to whichever gets there first -- the surer cluster on
    # a tie. A weak piece never joins two clusters into one: that turned the
    # site photo's ceiling into two giant webs.
    reached_at: dict[int, int] = {}  # weak mask -> the step that reached it
    if growing:
        surest = {
            root: max(painted[index].confidence for index in members)
            for root, members in clusters.items()
        }
        owner: dict[int, int] = {}
        frontier = sorted(
            (index for index in range(len(painted)) if strong[index]),
            key=lambda index: -surest[find(index)],
        )
        for index in frontier:
            owner[index] = find(index)
        step = 0
        while frontier:
            step += 1
            reached: list[int] = []
            for index in frontier:
                for neighbour in touching.get(index, ()):
                    if neighbour in owner or strong[neighbour]:
                        continue
                    # No thicker than a pipe continuing what it touches.
                    if thick(neighbour) > 2.0 * thick(index) + 2.0:
                        continue
                    owner[neighbour] = owner[index]
                    clusters[owner[index]].append(neighbour)
                    reached.append(neighbour)
                    reached_at[neighbour] = step
            frontier = reached

    out: list[Detection] = []
    found_pixels: list[np.ndarray] = []
    grown_steps: list[list[tuple[int, np.ndarray]]] = []
    for members in clusters.values():
        seeds = [index for index in members if strong[index]]
        if not seeds:
            # Grown from nothing: weak pieces that touch only each other.
            continue
        grown = [index for index in members if not strong[index]]
        grown_steps.append([(reached_at[index], pixels[index]) for index in grown])
        union = np.unique(np.concatenate([pixels[index] for index in members]))
        votes: dict[str, float] = {}
        phrase_votes: dict[str, float] = {}
        best = 0.0
        for index in members:
            instance = painted[index]
            votes[instance.label] = votes.get(instance.label, 0.0) + instance.confidence
            if strong[index]:
                best = max(best, instance.confidence)
        label = max(votes, key=lambda name: votes[name])
        # Which phrasings of the winning concept actually fired, strongest
        # first -- the attribution the scored bench and the writeups need.
        for index in members:
            instance = painted[index]
            if instance.label == label and instance.phrase:
                phrase_votes[instance.phrase] = (
                    phrase_votes.get(instance.phrase, 0.0) + instance.confidence
                )
        phrases = tuple(
            sorted(phrase_votes, key=lambda name: phrase_votes[name], reverse=True)
        )

        # The box around the union, seam-aware: rolled half a turn when it
        # touches both edges of the canvas.
        rows = union // width
        cols = union % width
        shift = width // 2 if (cols == 0).any() and (cols == width - 1).any() else 0
        rolled = (cols + shift) % width
        x0, x1 = int(rolled.min()), int(rolled.max())
        y0, y1 = int(rows.min()), int(rows.max())
        lat_top = float(row_to_lat_deg(float(y0), height))
        lat_bottom = float(row_to_lat_deg(float(y1), height))
        lat_centre = (lat_top + lat_bottom) / 2.0
        lon_left = float(column_to_lon_deg(float(x0 - shift), width))
        lon_span = (x1 - x0) * 360.0 / width
        lon_scale = max(np.cos(np.radians(lat_centre)), 1e-6)

        # A grown find is DRAWN as what the model was sure of -- exactly what it
        # looked like before growing -- and what was followed is kept apart, to
        # be shown for the one find a person selects. Drawing the whole union
        # made every outline the ragged edge of dozens of weak masks, and
        # doubled what the viewer had to draw.
        direct = union
        grown_only = np.empty(0, dtype=np.int64)
        if grown:
            direct = np.unique(np.concatenate([pixels[index] for index in seeds]))
            grown_all = np.unique(np.concatenate([pixels[index] for index in grown]))
            grown_only = np.setdiff1d(grown_all, direct, assume_unique=True)
        pieces = outlines(direct, width)
        found_pixels.append(direct)
        out.append(
            Detection(
                label=label,
                confidence=best,
                box=SphereBox(
                    lon_deg=float(wrap_lon_deg(lon_left + lon_span / 2.0)),
                    lat_deg=lat_centre,
                    lon_radius_deg=lon_span / 2.0 * float(lon_scale),
                    lat_radius_deg=(lat_top - lat_bottom) / 2.0,
                ),
                views=len(members),
                outline=pieces[0] if pieces else None,
                anchor=inside_point(direct, width),
                look_views=sum(1 for index in members if seen_by[index] in second_look),
                outlines=pieces,
                length_deg=length_deg(union, width),
                phrases=phrases,
                direct_views=len(seeds) if seed_confidence is not None else None,
                grown_views=len(grown),
                grown_share=float(grown_only.size) / float(union.size) if grown else 0.0,
                grown_outlines=(
                    outlines(grown_only, width, min_pixels=60, smooth_px=3)
                    if grown_only.size
                    else ()
                ),
                grown_scores=(
                    (
                        min(painted[index].confidence for index in grown),
                        max(painted[index].confidence for index in grown),
                    )
                    if grown
                    else None
                ),
            )
        )

    order = sorted(range(len(out)), key=lambda k: out[k].confidence, reverse=True)
    if stats is not None:
        stats["pixels"] = [found_pixels[k] for k in order]
        stats["grown_steps"] = [grown_steps[k] for k in order]
    return [out[k] for k in order]
