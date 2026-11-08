"""The detection seam, and the part that puts one object back together again.

Everything downstream -- the stage, the schema, the viewer -- consumes
`Detection` and knows nothing about which model produced it. That is what lets
15.3 drop an open-vocabulary detector in beside the COCO one without touching
anything else, the same way `localization/base.py` lets phase 10 swap in visual
odometry.

WHY DE-DUPLICATION IS NOT OPTIONAL HERE

    `projection.default_views()` deliberately overlaps, because an object cut by
    one view's edge has to be whole in another. The direct consequence is that
    most objects are detected MORE THAN ONCE -- the same workstation, found in
    two or three views, arriving as two or three detections pointing the same
    way.

    So overlap buys correctness at the cost of duplicates, and this module is
    where that debt is paid. Without it, "how many workstations are installed"
    counts each one about twice, which is exactly the kind of wrong that looks
    plausible on a dashboard.

    The merge is on the SPHERE, in angles, not in the pixels of any one view --
    two views do not share a coordinate system, so there is nowhere else it
    could honestly happen.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from vision.projection import SphereBox, ViewSpec


class DetectionError(Exception):
    """The detector could not run at all. A message for the operator."""


@dataclass(frozen=True, slots=True)
class Detection:
    """One thing a model believes it saw, as a direction on the sphere.

    A direction and an angular size, with no distance -- see `SphereBox`. The
    same shape as a field note's anchor, deliberately.

    `views` records how many of the overlapping views found it. It survives the
    merge because it is the cheapest confidence signal available: something seen
    from three virtual cameras and missed by none is a different proposition
    from something one view found once, and no single score captures that.
    """

    label: str
    confidence: float
    box: SphereBox
    views: int = 1

    # How far this box sat from the nearest edge of the view it was found in.
    # Small means the object ran off the side of that view and was measured
    # truncated, so its direction and its size are both understated.
    #
    # This is what overlap is FOR. Several views see the same object; at least
    # one of them, by construction, sees it whole. Carrying the margin lets the
    # merge pick that one instead of an arbitrary duplicate.
    edge_margin_px: float = 0.0

    # The traced boundary of the detection, as (lon, lat) vertices, when the
    # provider knows the actual pixels rather than just a box -- surfaces do,
    # detectors don't (yet). A box is the right display for a chair; a wall
    # wrapping half the room drawn as a rectangle looks broken however correct
    # the detection is, and this is what lets the client draw the real shape.
    outline: tuple[tuple[float, float], ...] | None = None

    # EVERY piece's outline, largest first -- `outline` is the first of these,
    # kept for the callers that draw one shape. A pipe the photo breaks behind
    # a column is two pieces of one pipe; drawing only the longer hid the rest.
    outlines: tuple[tuple[tuple[float, float], ...], ...] = ()

    # How long the thing is along its centre line, in degrees of view, for a
    # mask. For pipe, conduit and tray, how much is installed is a length, not
    # a count. Degrees until the ceiling plan supplies distances.
    length_deg: float | None = None

    # HOW IT WAS FOUND, for a mask grown from confident pieces (concept.py,
    # `fuse_instances` with a seed score). `direct_views` view masks scored at
    # or above the seed and were found directly; `grown_views` scored below it
    # and were kept only because they touched those, pipe-thin, and continued
    # them. `grown_share` is the fraction of the pixels only the grown pieces
    # cover, `grown_outlines` their shape, `grown_scores` their lowest and
    # highest score. All empty when nothing was grown.
    direct_views: int | None = None
    grown_views: int = 0
    grown_share: float = 0.0
    grown_outlines: tuple[tuple[tuple[float, float], ...], ...] = ()
    grown_scores: tuple[float, float] | None = None

    # Where a pin for this find goes, (lon, lat): a point ON the mask -- the
    # one deepest inside its largest sure piece. The box centre of an L-shaped
    # pipe, or of one in pieces, is empty ceiling. None for boxes, whose
    # centre is inside them.
    anchor: tuple[float, float] | None = None

    # How many of its view masks came from a SECOND LOOK (src/vision/
    # second_look.py) rather than the ordinary views. Zero when there was none.
    look_views: int = 0

    # When the concept layer (concepts.py) translated a request into several
    # plain-language phrasings, the phrasings of the winning label that
    # actually fired, strongest first. Empty for everything else.
    phrases: tuple[str, ...] = ()

    @property
    def lon_deg(self) -> float:
        return self.box.lon_deg

    @property
    def lat_deg(self) -> float:
        return self.box.lat_deg


@dataclass(frozen=True, slots=True)
class RawDetection:
    """What a provider returns: a box in ONE view's pixels.

    Deliberately not a `Detection`. A provider's job ends at "there is a thing
    at these pixels of the image you gave me"; converting that to a direction is
    this module's job, and keeping the two apart means a provider cannot get the
    geometry wrong.
    """

    label: str
    confidence: float
    x0: float
    y0: float
    x1: float
    y1: float
    # See InstanceMask.phrase: the phrasing that fired, when the concept layer
    # relabelled this detection. Empty means label is what was asked.
    phrase: str = ""


@runtime_checkable
class DetectionProvider(Protocol):
    """Finds objects in ordinary flat images."""

    name: str

    def detect(self, images: list[np.ndarray]) -> list[list[RawDetection]]:
        """One list of detections per image, in the order given.

        Batched because that is how a GPU wants to be fed: sixteen views of one
        panorama in a single call rather than sixteen round trips.
        """
        ...


# ---------------------------------------------------------------------------
# Putting the views back together
# ---------------------------------------------------------------------------

def to_sphere(
    raw_by_view: list[list[RawDetection]], views: list[ViewSpec]
) -> list[Detection]:
    """Convert per-view pixel boxes into directions. No merging yet."""
    if len(raw_by_view) != len(views):
        raise ValueError(
            f"{len(raw_by_view)} result lists for {len(views)} views -- "
            "the provider must answer in the order it was asked"
        )

    from vision.projection import box_to_sphere

    out: list[Detection] = []
    for results, spec in zip(raw_by_view, views, strict=True):
        for item in results:
            out.append(
                Detection(
                    label=item.label,
                    confidence=item.confidence,
                    box=box_to_sphere(spec, item.x0, item.y0, item.x1, item.y1),
                    edge_margin_px=min(
                        item.x0,
                        item.y0,
                        spec.width - 1 - item.x1,
                        spec.height - 1 - item.y1,
                    ),
                    phrases=(item.phrase,) if item.phrase else (),
                )
            )
    return out


DEFAULT_IOU = 0.3
DEFAULT_CONTAINMENT = 0.7

# How large a MERGED box may get before the cluster is judged to have run away.
#
# Chaining is deliberate -- A overlaps B overlaps C is one object crossing three
# views, and refusing that was the fault clustering was introduced to fix. But
# single-linkage has no notion of "enough": a repeating linear feature gives
# every neighbouring pair a genuine overlap, and the component grows along the
# whole run. On the first real footage a corridor's conduit came back as ONE
# "pipe" 212 degrees wide and 112 tall -- wider than a full turn, taller than
# pole to pole, and a description of nothing.
#
# These bound the merge, not the detector. A single view is 90 degrees across,
# so an unmerged box cannot approach them; only a cluster can. When the limit
# stops a merge the run comes back as several pipes, which is a usable answer,
# where one pipe filling the room is not.
#
# 60 degrees of longitude radius is a third of the horizon in one box. Nothing
# a snag list cares about is that wide, and a surface that genuinely is arrives
# as one detection with an outline rather than as a cluster.
MAX_MERGED_LON_RADIUS_DEG = 60.0
MAX_MERGED_LAT_RADIUS_DEG = 45.0


def _union_box(boxes: list[SphereBox]) -> SphereBox:
    """The smallest angular box covering every box in the cluster.

    This is why merging heals a truncated detection instead of merely deleting
    it: the clipped box lies inside the intact one, so their union IS the
    intact box, whichever of them happened to sort first.

    Longitude offsets are measured relative to the first box and scaled by
    cos(latitude) before being combined with the radii, for the same reason
    `_overlap_area` does it: the radii are true angular widths and raw
    longitude differences are not.
    """
    import math

    from vision.projection import wrap_lon_deg

    first = boxes[0]
    if len(boxes) == 1:
        return first

    mean_lat = sum(b.lat_deg for b in boxes) / len(boxes)
    scale = max(math.cos(math.radians(mean_lat)), 1e-6)

    lon_lo = lon_hi = 0.0
    lat_lo = lat_hi = 0.0
    for box in boxes:
        offset = float(wrap_lon_deg(box.lon_deg - first.lon_deg)) * scale
        lon_lo = min(lon_lo, offset - box.lon_radius_deg)
        lon_hi = max(lon_hi, offset + box.lon_radius_deg)
        d_lat = box.lat_deg - first.lat_deg
        lat_lo = min(lat_lo, d_lat - box.lat_radius_deg)
        lat_hi = max(lat_hi, d_lat + box.lat_radius_deg)

    centre_lat = first.lat_deg + (lat_lo + lat_hi) / 2.0
    return SphereBox(
        lon_deg=float(wrap_lon_deg(first.lon_deg + (lon_lo + lon_hi) / 2.0 / scale)),
        lat_deg=max(-90.0, min(90.0, centre_lat)),
        # Clamped to shapes that can exist. A longitude radius above 180 is more
        # than a full turn and a latitude radius above 90 is taller than pole to
        # pole; the caller refuses merges long before this, so reaching either
        # means the cluster was already nonsense and the reported geometry
        # should at least not be impossible.
        lon_radius_deg=min(180.0, (lon_hi - lon_lo) / 2.0),
        lat_radius_deg=min(90.0, (lat_hi - lat_lo) / 2.0),
    )


def merge_duplicates(
    detections: list[Detection],
    iou_threshold: float = DEFAULT_IOU,
    containment_threshold: float = DEFAULT_CONTAINMENT,
    cross_label_iou: float | None = None,
) -> list[Detection]:
    """Collapse detections of the same object seen from several views.

    CLUSTER, THEN FUSE -- not greedy suppression. Detections are grouped into
    connected components of the "same object" relation, and each cluster is
    reported once. The previous version was single-pass NMS, and it had two
    faults a live bench made obvious:

    * A CHAIN BROKE. Views A, B, C of one object where A overlaps B and B
      overlaps C but A barely overlaps C: the greedy pass matched B into A and
      then C found nothing over threshold, so one object came back twice.
      Connected components make A-B-C one cluster by construction.

    * THE SURVIVOR'S GEOMETRY WAS ONE VIEW'S GUESS. Absorbing a duplicate only
      bumped a counter; the kept box was never revisited, so the reported box
      was whichever single view sorted first -- including its truncation. Now
      the cluster's geometry is the UNION of its members' boxes, which is what
      "several cameras each saw part or all of it" honestly adds up to.

    Two things make two reports the same object:

    * OVERLAP (IoU above `iou_threshold`). The ordinary case -- one object in
      two views, boxes largely on top of each other.

    * CONTAINMENT (the smaller box mostly inside the larger). The truncated
      case. A box clipped by a view's border is genuinely much smaller than the
      intact view of the same thing, so its IoU with the good one stays low --
      0.14 in the test -- while it sits wholly inside it.

    LABELS. Same-label pairs merge on either test. Different labels normally
    never merge: a socket in front of a cable tray is two real things in nearly
    one direction, and merging across labels deletes the smaller one, which is
    always the one a snag list is about. That rule is also what makes
    containment safe: two same-label objects genuinely nested inside one
    another is not a thing, whereas a tile inside a grid is two labels and
    never reaches the test.

    `cross_label_iou` is the exception, for providers whose labels are noisy.
    An open-vocabulary model prompted with thirty phrases routinely calls the
    same object "office chair" from one view and "office desk workstation" from
    the next, and same-label-only merging then reports both. When set, two
    detections whose boxes are NEAR-IDENTICAL (IoU at or above this much
    higher bar -- containment deliberately does not count) merge across
    labels, and the cluster's label is decided by summed confidence per label:
    the phrase the views agreed on most strongly wins. Closed-set models leave
    it None and keep the strict rule.

    The fused detection: voted label, best confidence, union geometry, the
    largest edge margin, and `views` = cluster size.
    """
    if not detections:
        return []

    ordered = sorted(
        detections, key=lambda d: (d.confidence, d.edge_margin_px), reverse=True
    )

    # Union-find over the "same object" relation.
    parent = list(range(len(ordered)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    # Cluster membership is tracked as we go, because whether two detections
    # belong together is not only about THEM: joining their clusters may produce
    # a box neither of them would have justified. Checking that needs the
    # members, and pairwise union-find alone does not have them.
    members: dict[int, list[int]] = {index: [index] for index in range(len(ordered))}

    for i in range(len(ordered)):
        for j in range(i + 1, len(ordered)):
            a, b = ordered[i], ordered[j]
            if a.label == b.label:
                same = (
                    a.box.iou(b.box) >= iou_threshold
                    or a.box.containment(b.box) >= containment_threshold
                )
            else:
                same = (
                    cross_label_iou is not None
                    and a.box.iou(b.box) >= cross_label_iou
                )
            if not same:
                continue

            ra, rb = find(i), find(j)
            if ra == rb:
                continue

            # WOULD the joined cluster still describe an object? A run of
            # conduit gives every neighbouring pair a real overlap, so the pair
            # test keeps saying yes while the component grows along the whole
            # corridor. The union is where that shows up, so the union is what
            # gets vetoed -- the pair stays unmerged and the run comes back as
            # several pipes.
            candidate = _union_box(
                [ordered[index].box for index in members[ra] + members[rb]]
            )
            if (
                candidate.lon_radius_deg > MAX_MERGED_LON_RADIUS_DEG
                or candidate.lat_radius_deg > MAX_MERGED_LAT_RADIUS_DEG
            ):
                continue

            root, absorbed = min(ra, rb), max(ra, rb)
            parent[absorbed] = root
            members[root] = members[root] + members[absorbed]
            del members[absorbed]

    clusters: dict[int, list[Detection]] = {}
    for index in range(len(ordered)):
        clusters.setdefault(find(index), []).append(ordered[index])

    merged: list[Detection] = []
    for root in sorted(clusters):
        members = clusters[root]  # already strongest-first
        votes: dict[str, float] = {}
        for member in members:
            votes[member.label] = votes.get(member.label, 0.0) + member.confidence
        label = max(votes, key=lambda name: votes[name])
        # Attribution survives the merge: the winning label's phrasings,
        # strongest first, duplicates folded.
        phrase_votes: dict[str, float] = {}
        for member in members:
            if member.label == label:
                for phrase in member.phrases:
                    phrase_votes[phrase] = (
                        phrase_votes.get(phrase, 0.0) + member.confidence
                    )
        merged.append(
            replace(
                members[0],
                label=label,
                box=_union_box([m.box for m in members]),
                edge_margin_px=max(m.edge_margin_px for m in members),
                views=len(members),
                phrases=tuple(
                    sorted(phrase_votes, key=lambda p: phrase_votes[p], reverse=True)
                ),
            )
        )
    return merged


# Per-view suppression, applied BEFORE anything reaches the sphere.
#
# WHY THE PROVIDER'S OWN NMS IS NOT ENOUGH
#
#     SAM 3 suppresses at IoU 0.7 -- DETR-style, deliberately permissive, and
#     right for a model whose job ends at "there is a thing here". A corridor is
#     full of near-parallel linear edges, so at 0.7 a single conduit run comes
#     back as roughly thirty boxes in ONE 90-degree view: the pipe, the shadow
#     beside it, the wall line above it, each overlapping its neighbour by less
#     than 70%.
#
#     The sphere merge then inherits that pile. It is built to join genuine
#     partial views of one object, and single-linkage over thirty adjacent boxes
#     joins all of them: measured, one "pipe" cluster with 465 members.
#
# So each view is thinned first, per label, at a tighter IoU. The merge then
# does the job it was designed for rather than absorbing noise, and every number
# downstream -- detections, objects, counts -- means what it says.
VIEW_NMS_IOU = 0.5


def _box_iou(a: RawDetection, b: RawDetection) -> float:
    """Overlap of two boxes in ONE view's pixels. Flat, not spherical -- these
    have not been projected yet, and that is the point: suppression belongs
    where the detector's own coordinates still are."""
    left = max(min(a.x0, a.x1), min(b.x0, b.x1))
    right = min(max(a.x0, a.x1), max(b.x0, b.x1))
    top = max(min(a.y0, a.y1), min(b.y0, b.y1))
    bottom = min(max(a.y0, a.y1), max(b.y0, b.y1))
    if right <= left or bottom <= top:
        return 0.0

    overlap = (right - left) * (bottom - top)
    area_a = abs(a.x1 - a.x0) * abs(a.y1 - a.y0)
    area_b = abs(b.x1 - b.x0) * abs(b.y1 - b.y0)
    union = area_a + area_b - overlap
    return overlap / union if union > 0 else 0.0


def suppress_in_view(
    detections: list[RawDetection], iou_threshold: float = VIEW_NMS_IOU
) -> list[RawDetection]:
    """Greedy non-maximum suppression within one view, per label.

    Per LABEL, like the sphere merge and for the same reason: a socket in front
    of a cable tray is two real things in one direction, and suppressing across
    labels would delete the smaller one -- always the one a snag list is about.

    Strongest first, and a box is kept unless it overlaps something already kept
    by more than the threshold. That is ordinary NMS; what matters here is only
    that it happens at all, before the geometry leaves the view.
    """
    kept: list[RawDetection] = []
    for candidate in sorted(detections, key=lambda d: d.confidence, reverse=True):
        if any(
            other.label == candidate.label
            and _box_iou(candidate, other) >= iou_threshold
            for other in kept
        ):
            continue
        kept.append(candidate)
    return kept


def views_to_objects(
    raw_by_view: list[list[RawDetection]],
    specs: list[ViewSpec],
    *,
    min_confidence: float,
    cross_label_iou: float | None = None,
    phrase_map: Mapping[str, str] | None = None,
    iou_threshold: float = DEFAULT_IOU,
    view_nms_iou: float = VIEW_NMS_IOU,
) -> tuple[list[Detection], list[Detection]]:
    """Boxes from every view -> (every detection on the sphere, merged objects).

    THE ONE RULE, used by the detection bench and by the visit pipeline alike.
    They used to carry a copy each, and the copies drifted: per-view thinning
    went into the pipeline on 16 Sept and never reached the bench, and the
    concept expansion lived in the bench and never reached the pipeline. So a
    setup tuned on one photo meant something else over a whole visit -- "cable
    tray" asked as the phrasings that fire on one, asked literally (which fires
    on nothing) on the other. One function, and the two cannot disagree.

    In order, each for a reason:

    1. Phrasings folded back under their concept (`phrase_map`), before
       anything compares labels -- so two phrasings of one concept thin and
       merge as one thing.
    2. Thinned per view: the sphere merge is a duplicate JOINER, not a
       suppressor, and handing it an unsuppressed pile is what produced a
       465-member "pipe".
    3. Projected onto the sphere and cut at `min_confidence`.
    4. Merged across views. A provider that knows its labels are unstable
       across views opts into cross-label merging via `cross_label_iou`.
    """
    if phrase_map:
        from vision.concepts import relabel_raw

        raw_by_view = relabel_raw(raw_by_view, dict(phrase_map))
    thinned = [suppress_in_view(view, view_nms_iou) for view in raw_by_view]
    on_sphere = [d for d in to_sphere(thinned, specs) if d.confidence >= min_confidence]
    merged = merge_duplicates(
        on_sphere, iou_threshold=iou_threshold, cross_label_iou=cross_label_iou
    )
    return on_sphere, merged


def detect_panorama(
    panorama: np.ndarray,
    provider: DetectionProvider,
    views: list[ViewSpec] | None = None,
    min_confidence: float = 0.25,
    iou_threshold: float = DEFAULT_IOU,
    view_nms_iou: float = VIEW_NMS_IOU,
    phrase_map: Mapping[str, str] | None = None,
) -> list[Detection]:
    """The whole of one frame: render the views, detect, merge.

    This is the function the pipeline stage calls, and the one a test can drive
    with a fake provider and no GPU in sight. `phrase_map` is the concept
    expansion's fold-back (src/vision/concepts.py), for prompted providers.
    """
    from vision.projection import default_views, render

    specs = views if views is not None else default_views()
    images = [render(panorama, spec) for spec in specs]
    _, merged = views_to_objects(
        provider.detect(images),
        specs,
        min_confidence=min_confidence,
        cross_label_iou=getattr(provider, "cross_label_iou", None),
        phrase_map=phrase_map,
        iou_threshold=iou_threshold,
        view_nms_iou=view_nms_iou,
    )
    return merged


# ---------------------------------------------------------------------------
# Where weights live
# ---------------------------------------------------------------------------

def ensure_weights(name: str, url: str | None = None) -> Path:
    """Get a weights file into the project directory, whatever it takes.

    TWO WAYS, because ultralytics shells out to curl and curl cannot complete a
    TLS handshake to github on this machine -- "schannel: SEC_E_INVALID_TOKEN".
    Python's own HTTPS fetches the identical file in seconds. Try the library's
    way, then ours, and report both failures if neither works.

    Downloaded to a `.part` and renamed on completion, so an interrupted fetch
    cannot leave a truncated `.pt` that then fails to load, confusingly, forever.

    Also the reason a bare name is never handed to `YOLO()`: that downloads into
    the working directory, which for the API process is the source tree. One
    appeared there the first time this ran.
    """
    directory = model_dir()
    directory.mkdir(parents=True, exist_ok=True)
    local = directory / name
    if local.exists():
        return local

    problems: list[str] = []

    # An explicit url means the weights are somebody else's -- a fine-tuned
    # model from Hugging Face rather than one of ultralytics' own assets -- so
    # its downloader has nothing to offer and is skipped.
    if url is None:
        try:
            from ultralytics.utils.downloads import attempt_download_asset

            attempt_download_asset(str(local))
        except Exception as exc:  # noqa: BLE001 - the fallback is the point
            problems.append(f"ultralytics: {exc}")

    if not local.exists():
        try:
            import urllib.request

            source = url
            if source is None:
                from ultralytics.utils.downloads import GITHUB_ASSETS_VERSION

                source = (
                    "https://github.com/ultralytics/assets/releases/download/"
                    f"{GITHUB_ASSETS_VERSION}/{name}"
                )
            partial = local.with_suffix(local.suffix + ".part")
            urllib.request.urlretrieve(source, partial)  # noqa: S310 - a fixed https URL
            partial.replace(local)
        except Exception as exc:  # noqa: BLE001
            problems.append(f"direct: {exc}")

    if not local.exists():
        raise DetectionError(
            f"could not download {name} to {local}. " + " | ".join(problems)
        )
    return local


def model_dir() -> Path:
    """The project-local directory for model weights.

    `MODEL_DIR`, if set. The fallback keeps this
    importable in a bare shell, but every real run has the variable -- weights
    are hundreds of megabytes and the default for all of these libraries is
    somewhere under the user profile on C:, which this project deliberately
    stays off.
    """
    configured = os.environ.get("MODEL_DIR")
    if configured:
        return Path(configured)
    return Path(__file__).resolve().parents[2] / ".cache" / "models"
