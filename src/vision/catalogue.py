"""The models the bench can run, and what each one actually knows.

Phase 15.3. Written down rather than hard-coded because the interesting
question is no longer "does detection work" -- it does -- but "which model, and
what is it good for". That question is answered by comparing them on the same
photograph, which needs them to be selectable.

WHAT IS NOT HERE, AND WHY

    There is no publicly trained model for interior fit-out. Searched for:
    interior, indoor, duct, electrical, conduit, scaffolding, concrete, rebar.
    What exists for construction is SITE SAFETY -- workers, hard hats, vests,
    machinery, cones -- because that is what the industry has funded, and it is
    a compliance problem rather than a progress-measurement one.

    So `construction` below is a real construction-trained model finding real
    construction things, and it still cannot tell you whether a ceiling grid is
    installed. That gap is the whole argument for phase 16: nobody is going to
    hand us a fit-out model, so the labelled set has to be built.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ModelEntry:
    """One selectable detector."""

    key: str
    label: str
    weights: str
    kind: str                       # "closed" or "open-vocabulary"
    licence: str
    origin: str
    good_for: str
    caveat: str
    url: str | None = None          # None means an ultralytics asset
    classes: list[str] = field(default_factory=list)
    # Where the bench's "keep scores" slider starts for this model. The mask
    # models' scores are calibrated differently: SAM 3 below 0.4 is mostly
    # guesswork on a plain view, Grounding DINO's own default is 0.3.
    default_confidence: float = 0.25


CATALOGUE: dict[str, ModelEntry] = {
    "coco": ModelEntry(
        key="coco",
        label="YOLO11 (COCO)",
        weights="yolo11n.pt",
        kind="closed",
        licence="AGPL-3.0",
        origin="Ultralytics, trained on COCO",
        good_for="People, vehicles, furniture. Confident and fast.",
        caveat=(
            "Eighty classes of everyday objects. On a fit-out it returns book, "
            "keyboard and skis -- confident guesses at materials it has no word "
            "for."
        ),
    ),
    "construction": ModelEntry(
        key="construction",
        label="Construction site safety",
        weights="construction-ppe.pt",
        url="https://huggingface.co/Hansung-Cho/yolov8-ppe-detection/resolve/main/best.pt",
        kind="closed",
        licence="MIT",
        origin="Hansung-Cho/yolov8-ppe-detection, trained on construction site imagery",
        good_for=(
            "Workers, hard hats, high-visibility vests, machinery, vehicles and "
            "cones -- and it flags the NEGATIVES, so a worker without a hard hat "
            "is its own class."
        ),
        caveat=(
            "Site safety, not progress. It knows nothing about ceilings, "
            "partitions or services. The nearest thing to a construction model "
            "that exists publicly."
        ),
        classes=[
            "Hardhat", "Mask", "NO-Hardhat", "NO-Mask", "NO-Safety Vest",
            "Person", "Safety Cone", "Safety Vest", "machinery", "vehicle",
        ],
    ),
    "surfaces": ModelEntry(
        key="surfaces",
        label="Surfaces (ceiling, wall, floor)",
        weights="openmmlab/upernet-convnext-small",
        kind="semantic",
        licence="MIT",
        origin="UperNet + ConvNeXt trained on ADE20K, 150 indoor scene classes",
        good_for=(
            "The Tier 1 surfaces -- ceiling, wall, floor, door, column, light "
            "fittings -- as AREAS with a coverage percentage, which is the "
            "shape a progress claim actually takes."
        ),
        caveat=(
            "It knows ceiling, not whether that ceiling is open grid, tiled or "
            "plasterboard -- and wall, not stud versus boarded. It answers "
            "WHERE the surfaces are, never what state they are in -- and state "
            "is what a progress claim rests on."
        ),
        classes=[
            "ceiling", "wall", "floor", "door", "windowpane", "column",
            "stairs", "railing", "light", "chandelier", "lamp", "cabinet",
            "desk", "chair", "table", "shelf", "radiator",
        ],
    ),
    "grounded": ModelEntry(
        key="grounded",
        label="Grounded SAM 2 (masks from words)",
        weights="grounding-dino-base + sam2.1-hiera-small",
        kind="concept",
        licence="Apache-2.0",
        origin="IDEA-Research Grounding DINO grounding the words, Meta SAM 2 drawing the pixels",
        good_for=(
            "Anything you can name, as PIXEL MASKS. Grounding DINO reads the "
            "prompt far better than YOLO-World, and a mask is the only honest "
            "shape for cable tray, ducting or a half-boarded wall."
        ),
        caveat=(
            "Two models per frame, so seconds rather than milliseconds. "
            "Phrases still matter: 'metal cable tray on ceiling' grounds "
            "better than 'tray'."
        ),
        default_confidence=0.3,
    ),
    "sam3": ModelEntry(
        key="sam3",
        label="SAM 3 (concept masks)",
        weights="facebook/sam3",
        kind="concept",
        licence="SAM licence (gated)",
        origin="Meta, promptable concept segmentation",
        good_for=(
            "The strongest open-vocabulary masks: every instance of a named "
            "concept, with calibrated scores and a presence head that stays "
            "quiet when the concept is not there."
        ),
        caveat=(
            "Weights are gated: accept the licence at "
            "huggingface.co/facebook/sam3.1 and `hf auth login` first. One "
            "forward pass per prompt, so keep the prompt list short."
        ),
        default_confidence=0.4,
    ),
    "sam3-trt": ModelEntry(
        key="sam3-trt",
        label="SAM 3 accelerated (TensorRT boxes)",
        weights="facebook/sam3 via DART TensorRT FP16 engines",
        kind="prompted",
        licence="SAM licence (gated); DART under the same SAM licence",
        origin=(
            "Meta's SAM 3, the identical checkpoint as above, split by DART "
            "(Turkcan 2026, arXiv 2603.11441) into two TensorRT engines with "
            "the text encoder cached"
        ),
        good_for=(
            "The same detections as SAM 3 -- same weights, same presence-gated "
            "scores -- as BOXES, roughly ten times faster, and pipelined across "
            "the views of a walk. The default for counting things."
        ),
        caveat=(
            "Boxes only: the mask decoder is what was skipped to get the speed, "
            "so outlines still need the plain SAM 3 entry. Needs engines built "
            "once per GPU with scripts\\build_dart_engines.ps1 -- they are "
            "specific to this GPU and TensorRT version and are never committed."
        ),
    ),
    "openvocab": ModelEntry(
        key="openvocab",
        label="YOLO-World (say what to find)",
        weights="yolov8s-worldv2.pt",
        kind="open-vocabulary",
        licence="AGPL-3.0",
        origin="Ultralytics YOLO-World",
        good_for=(
            "Anything you can name. Ask for 'metal cable tray' or 'suspended "
            "ceiling grid' with no training and no labels."
        ),
        caveat=(
            "Right words, low confidence on real construction interiors, "
            "far below its scores on things COCO knows. The largest variant is no better, so it is not a capacity problem. "
            "Useful for pre-labelling, not for counting."
        ),
    ),
}

# SAM 3 by decision, not accident: on real fit-out photographs it scored far
# above YOLO-World. The accelerated entry is the default: the scored bench put
# it within FP16 noise of the eager provider on every photograph at a
# seventeenth of the time. `sam3` stays as the reference and for masks; the YOLOs stay as the
# speed baseline and for comparison.
DEFAULT_KEY = "sam3-trt"


def entry(key: str) -> ModelEntry:
    found = CATALOGUE.get(key)
    if found is None:
        raise KeyError(
            f"{key!r} is not a model in the catalogue. "
            f"Choose one of: {', '.join(CATALOGUE)}"
        )
    return found
