"""Putting per-view masks back onto the sphere, at the views' own sharpness.

`concept.fuse_instances` joins masks found in overlapping flat views into one
detection per real thing. This module is the geometry under it: which panorama
pixels each view covers, a mask as those pixels, where two views can both see,
and the shape and length of what they add up to.

WHAT WAS WRONG BEFORE (research experiment 02, "where the rest of the pipe
goes")

    * The canvas was 1024 px wide -- 2.8 px per degree, against 7 in a view
      and 33 in the photo. A pipe a few view-pixels wide landed on it as a
      dotted line.
    * Masks were painted FORWARD, view pixel to canvas pixel. That is only
      hole-free while the canvas is coarser than the view; any finer and it
      speckles, which is why the canvas could not simply be made bigger.
    * The outline kept a detection's largest connected piece and dropped the
      rest, so a pipe broken anywhere was drawn as its longest fragment.
    * Two views' masks of one pipe were compared as WHOLE masks. View A sees
      the left stretch, view B the right; they overlap only where both can
      look, so their overall overlap stays low and a pipe crossing a view
      edge -- most masks on the site photo -- stayed in pieces.

WHAT THIS DOES INSTEAD

    * The canvas matches the views: one canvas pixel per view pixel at the
      view's centre (`canvas_width_for`), 2560 px for the default 640-px views.
    * Painting is BACKWARD: every canvas pixel a view covers looks up the view
      pixel it came from (`footprint`). No canvas pixel inside a view is ever
      skipped, whatever the resolutions.
    * A mask is the sorted array of the canvas pixels it covers -- a few
      thousand integers, not a full canvas each -- so a finer canvas costs
      memory in proportion to what was found, not to the sphere.
    * Two views are compared where BOTH can see (`shared_region`). Inside
      that region their masks of one pipe coincide; outside it, one of them
      simply was not looking.
    * Every piece of a detection is outlined, and a mask's length is measured
      along its centre line (`length_deg`), because for pipe, conduit and tray
      how much is installed is a length, not a count.
"""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np

from vision.projection import (
    ViewSpec,
    column_to_lon_deg,
    direction,
    row_to_lat_deg,
    wrap_lon_deg,
)

# A canvas finer than this is not worth its memory: 8192 px is 23 px/deg,
# enough for 2048-px views at 90 degrees.
MAX_CANVAS_WIDTH = 8192

# Two masks are compared only where both views can see; below this many shared
# pixels there is nothing to compare, and the pair is left alone.
MIN_SHARED_PIXELS = 8

# A piece smaller than this, on the canvas, is not drawn as its own outline.
MIN_PIECE_PIXELS = 12

# Vertices per outline: enough for a pipe's bends, few enough to ship.
MAX_OUTLINE_POINTS = 160


def canvas_width_for(views: list[ViewSpec]) -> int:
    """A canvas as sharp as the sharpest view at its centre, and no sharper.

    640-px views at 90 degrees are 7.1 px/deg, so 2560 px round the sphere. An
    even width, because the canvas is width x width/2.
    """
    pixels_per_degree = max(spec.width / spec.fov_deg for spec in views)
    width = int(math.ceil(360.0 * pixels_per_degree / 2.0)) * 2
    return min(width, MAX_CANVAS_WIDTH)


@lru_cache(maxsize=256)
def footprint(spec: ViewSpec, width: int) -> tuple[np.ndarray, np.ndarray]:
    """Every canvas pixel this view covers, and the view pixel each one shows.

    Returns (canvas indices, view indices), both flat, the canvas indices
    sorted. BACKWARD: canvas pixel -> direction -> view pixel, nearest pixel,
    so the covered region has no holes however the resolutions compare.

    Cached: the same sixteen views are used for every photo of a run, so each
    footprint is worked out once per process.
    """
    height = width // 2
    # Only the rows the view can reach: the latitudes along its border, and
    # the pole when the view contains it.
    border = np.linspace(0.0, 1.0, 64)
    us = np.concatenate([border * (spec.width - 1), np.full(64, spec.width - 1.0),
                         border[::-1] * (spec.width - 1), np.zeros(64)])
    vs = np.concatenate([np.zeros(64), border * (spec.height - 1),
                         np.full(64, spec.height - 1.0), border[::-1] * (spec.height - 1)])
    _, lats = spec.pixel_to_lon_lat(us, vs)
    top, bottom = float(np.max(lats)), float(np.min(lats))
    if spec.lat_deg + spec.vertical_fov_deg / 2.0 >= 90.0 or spec.contains(0.0, 90.0):
        top = 90.0
    if spec.lat_deg - spec.vertical_fov_deg / 2.0 <= -90.0 or spec.contains(0.0, -90.0):
        bottom = -90.0
    row_first = max(0, int(math.floor((90.0 - top) / 180.0 * height)) - 1)
    row_last = min(height - 1, int(math.ceil((90.0 - bottom) / 180.0 * height)) + 1)

    rows = np.arange(row_first, row_last + 1, dtype=np.int64)
    # And only the columns: a view holding neither pole spans no more longitude
    # than its border does. Checking every column round the sphere was most of
    # the cost of a second look's two dozen new views on a sharp canvas.
    if top < 90.0 and bottom > -90.0:
        border_lons, _ = spec.pixel_to_lon_lat(us, vs)
        offsets = (np.asarray(border_lons) - spec.lon_deg + 180.0) % 360.0 - 180.0
        first = int(math.floor((spec.lon_deg + offsets.min() + 180.0) / 360.0 * width)) - 2
        count = min(width, int(math.ceil((offsets.max() - offsets.min()) / 360.0 * width)) + 5)
        cols = (first + np.arange(count, dtype=np.int64)) % width
    else:
        cols = np.arange(width, dtype=np.int64)
    grid_r, grid_c = np.meshgrid(rows, cols, indexing="ij")
    lon = np.asarray(column_to_lon_deg(grid_c.astype(np.float64), width))
    lat = np.asarray(row_to_lat_deg(grid_r.astype(np.float64), height))
    u, v, in_front = spec.lon_lat_to_pixel(lon, lat)
    ui = np.rint(u).astype(np.int64)
    vi = np.rint(v).astype(np.int64)
    inside = in_front & (ui >= 0) & (ui < spec.width) & (vi >= 0) & (vi < spec.height)

    canvas = (grid_r * width + grid_c)[inside]
    view = (vi * spec.width + ui)[inside]
    # Columns that wrap past the seam arrive out of order; everything downstream
    # relies on sorted canvas pixels.
    order = np.argsort(canvas, kind="stable")
    return canvas[order].astype(np.int64), view[order].astype(np.int64)


def paint(mask: np.ndarray, spec: ViewSpec, width: int) -> np.ndarray:
    """A view's boolean mask as the sorted canvas pixels it covers."""
    canvas, view = footprint(spec, width)
    return canvas[mask.ravel()[view]]


@lru_cache(maxsize=512)
def shared_region(a: ViewSpec, b: ViewSpec, width: int) -> np.ndarray:
    """The canvas pixels BOTH views can see, sorted. A view with itself is its
    whole footprint."""
    if a == b:
        return footprint(a, width)[0]
    return np.intersect1d(footprint(a, width)[0], footprint(b, width)[0], assume_unique=True)


def shared_count(a: np.ndarray, b: np.ndarray) -> int:
    """How many pixels two SORTED, duplicate-free pixel lists share.

    Looks the shorter list up in the longer one instead of merging and
    re-sorting both, which `np.intersect1d` does; with tens of thousands of
    comparisons per photo that re-sorting was a fifth of the stitching time.
    """
    if a.size == 0 or b.size == 0:
        return 0
    small, big = (a, b) if a.size <= b.size else (b, a)
    at = np.searchsorted(big, small)
    at[at == big.size] = 0
    return int(np.count_nonzero(big[at] == small))


def in_view(indices: np.ndarray, spec: ViewSpec, width: int) -> np.ndarray:
    """Which of these canvas pixels the view covers -- the same test
    `footprint` applies (in front, nearest pixel inside the view), asked of
    these pixels only."""
    height = width // 2
    lon = np.asarray(column_to_lon_deg((indices % width).astype(np.float64), width))
    lat = np.asarray(row_to_lat_deg((indices // width).astype(np.float64), height))
    u, v, in_front = spec.lon_lat_to_pixel(lon, lat)
    ui = np.rint(u)
    vi = np.rint(v)
    return in_front & (ui >= 0) & (ui < spec.width) & (vi >= 0) & (vi < spec.height)


def overlap_where_both_look(
    a: np.ndarray, spec_a: ViewSpec, b: np.ndarray, spec_b: ViewSpec, width: int
) -> tuple[float, float] | None:
    """(IoU, containment) of two masks, measured only where both views see.

    None when, in that region, the masks do not both have something to compare
    -- one view saw the thing and the other was simply not looking there,
    which is not evidence either way.
    """
    # A mask's pixels are all in its own view, so "where both look" is, for
    # each mask, its pixels the OTHER view can see -- asked of those few
    # thousand pixels directly, not by searching the millions both views
    # share. Same answer as intersecting with `shared_region`; that search
    # was most of the stitching time once views got sharp or numerous.
    a_in = a[in_view(a, spec_b, width)] if spec_a != spec_b else a
    b_in = b[in_view(b, spec_a, width)] if spec_a != spec_b else b
    if a_in.size < MIN_SHARED_PIXELS or b_in.size < MIN_SHARED_PIXELS:
        return None
    shared = shared_count(a_in, b_in)
    if shared == 0:
        return 0.0, 0.0
    return shared / (a_in.size + b_in.size - shared), shared / min(a_in.size, b_in.size)


# ---------------------------------------------------------------------------
# Shape and length of a union of pixels
# ---------------------------------------------------------------------------


def crop(indices: np.ndarray, width: int) -> tuple[np.ndarray, int, int, int]:
    """The pixels as a small boolean image: (image, first row, first column,
    seam shift). A region crossing the 360-degree seam is rolled half a turn
    first so it comes out in one piece."""
    rows = indices // width
    cols = indices % width
    shift = width // 2 if (cols.min() < width // 8 and cols.max() > width - width // 8) else 0
    cols = (cols + shift) % width
    r0, c0 = int(rows.min()), int(cols.min())
    image = np.zeros((int(rows.max()) - r0 + 1, int(cols.max()) - c0 + 1), dtype=np.uint8)
    image[rows - r0, cols - c0] = 1
    return image, r0, c0, shift


def inside_point(indices: np.ndarray, width: int) -> tuple[float, float]:
    """(lon, lat) of the pixel deepest inside the largest piece: always ON the
    mask, and in the middle of a pipe's width rather than at its edge."""
    import cv2

    image, r0, c0, shift = crop(indices, width)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(image, connectivity=8)
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA])) if count > 1 else 0
    piece = (labels == largest).astype(np.uint8)
    depth = cv2.distanceTransform(np.pad(piece, 1), cv2.DIST_L2, 3)[1:-1, 1:-1]
    row, col = np.unravel_index(int(np.argmax(depth)), depth.shape)
    return (
        float(wrap_lon_deg(column_to_lon_deg(float(col + c0 - shift), width))),
        float(row_to_lat_deg(float(row + r0), width // 2)),
    )


def outlines(
    indices: np.ndarray,
    width: int,
    min_pixels: int = MIN_PIECE_PIXELS,
    smooth_px: int = 0,
) -> tuple[tuple[tuple[float, float], ...], ...]:
    """Every piece's outline as (lon, lat) vertices, largest piece first.

    ALL pieces: a pipe that the photo breaks behind a column is two pieces of
    one pipe, and drawing only the longer one hid the other.

    Long outlines are SIMPLIFIED (Douglas-Peucker, tolerance raised until the
    piece fits MAX_OUTLINE_POINTS), not thinned by keeping every nth vertex:
    that zig-zagged the edge of a long pipe. `smooth_px` closes gaps of about
    that many canvas pixels first -- for the followed part of a grown find,
    which is dozens of ragged weak masks side by side.
    """
    import cv2

    image, r0, c0, shift = crop(indices, width)
    if smooth_px > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * smooth_px + 1, 2 * smooth_px + 1)
        )
        image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    height = width // 2
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pieces = []
    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        if cv2.contourArea(contour) < min_pixels and pieces:
            continue
        tolerance = 0.75
        points = cv2.approxPolyDP(contour, tolerance, True)
        while len(points) > MAX_OUTLINE_POINTS:
            tolerance *= 1.5
            points = cv2.approxPolyDP(contour, tolerance, True)
        points = points.reshape(-1, 2)
        if len(points) < 3:
            continue
        pieces.append(
            tuple(
                (
                    float(wrap_lon_deg(column_to_lon_deg(float(x + c0 - shift), width))),
                    float(row_to_lat_deg(float(y + r0), height)),
                )
                for x, y in points
            )
        )
    return tuple(pieces)


def dilate(indices: np.ndarray, width: int) -> np.ndarray:
    """The pixels plus every pixel touching them (8-neighbourhood), sorted.

    "Does this piece touch that one" is then an overlap test. Rows are clipped
    at the poles; at the 360-degree seam a column steps into the neighbouring
    row, which misplaces a touch by one pixel on a line nothing sits on.
    """
    height = width // 2
    offsets = [dy * width + dx for dy in (-1, 0, 1) for dx in (-1, 0, 1)]
    grown = np.concatenate([indices + offset for offset in offsets])
    grown = grown[(grown >= 0) & (grown < width * height)]
    return np.unique(grown)


def touches(
    a: np.ndarray, b: np.ndarray, width: int, a_dilated: np.ndarray | None = None
) -> bool:
    """Do two sets of pixels share a pixel or sit side by side? Pass `a` grown
    by a pixel (`dilate`) when it is already known: growing it again for every
    pair was most of the time following pipes took."""
    grown = dilate(a, width) if a_dilated is None else a_dilated
    return shared_count(grown, b) > 0


def thickness_px(indices: np.ndarray, width: int) -> float:
    """How thick a shape is, in canvas pixels: its area over its centre line.

    A pipe is long and thin, so a weak piece CONTINUING one has about the same
    thickness; a broad band of red-painted vault, which the model also scores
    weakly as "plumbing pipe", is many times thicker than the pipe beside it.

    Exact for anything pipe-width. A BROAD shape -- roughly 16 px or more,
    judged from its distance transform (a band W wide averages about W/4 from
    its edge) -- is thinned at reduced size: thinning peels one pixel a pass,
    so a wide band took dozens of passes, and such a shape only needs to be
    known as far too wide, not to the pixel. That was two thirds of the time
    once a second look multiplied the masks.
    """
    import cv2

    image, _, _, _ = crop(indices, width)
    depth = cv2.distanceTransform(np.pad(image, 1), cv2.DIST_L2, 3)
    rough = 4.0 * float(depth[depth > 0].mean()) if depth.any() else 0.0
    scale = max(1, int(rough // 8)) if rough >= 16.0 else 1
    if scale > 1:
        # Shrunk so that any covered pixel survives: a thin arm on a broad blob
        # must not vanish and shorten the centre line. The area is the real one.
        h, w = image.shape
        small = cv2.resize(
            image.astype(np.float32),
            (max(1, -(-w // scale)), max(1, -(-h // scale))),
            interpolation=cv2.INTER_AREA,
        ) > 0
        return float(indices.size) / max(scale * int(thin(small).sum()), 1)
    return float(indices.size) / max(int(thin(image).sum()), 1)


@lru_cache(maxsize=1)
def _peel_tables() -> tuple[np.ndarray, np.ndarray]:
    """Zhang-Suen's peel decision for each of the 256 neighbourhoods, one table
    per sub-pass. Bit k of the index is neighbour p(k+2): p2 north, then
    clockwise to p9 north-west."""
    tables = (np.zeros(256, dtype=bool), np.zeros(256, dtype=bool))
    for code in range(256):
        p = [(code >> k) & 1 for k in range(8)]
        p2, _, p4, _, p6, _, p8, _ = p
        ring = p + [p[0]]
        transitions = sum(1 for k in range(8) if ring[k] == 0 and ring[k + 1] == 1)
        common = 2 <= sum(p) <= 6 and transitions == 1
        tables[0][code] = common and p2 * p4 * p6 == 0 and p4 * p6 * p8 == 0
        tables[1][code] = common and p2 * p4 * p8 == 0 and p2 * p6 * p8 == 0
    return tables


def thin(image: np.ndarray) -> np.ndarray:
    """One-pixel-wide centre line of a binary image (Zhang-Suen thinning).

    The textbook algorithm: peel boundary pixels in two alternating sub-passes
    until nothing changes, never breaking a line in two. What is left runs
    down the middle of a pipe. Each pixel's eight neighbours are packed into a
    byte and the peel decision looked up (`_peel_tables`) -- the same rule,
    computed once for all 256 neighbourhoods instead of per pixel per pass,
    which made thinning a quarter of the stitching time.
    """
    img = np.pad(image.astype(np.uint8), 1)
    tables = _peel_tables()
    while True:
        changed = False
        for step in (0, 1):
            code = (
                img[:-2, 1:-1]
                | (img[:-2, 2:] << 1)
                | (img[1:-1, 2:] << 2)
                | (img[2:, 2:] << 3)
                | (img[2:, 1:-1] << 4)
                | (img[2:, :-2] << 5)
                | (img[1:-1, :-2] << 6)
                | (img[:-2, :-2] << 7)
            )
            centre = img[1:-1, 1:-1]
            peel = (centre == 1) & tables[step][code]
            if peel.any():
                centre[peel] = 0
                changed = True
        if not changed:
            return img[1:-1, 1:-1].astype(bool)


def length_deg(indices: np.ndarray, width: int) -> float:
    """How long the thing is, in degrees of view, along its centre line.

    Each pair of neighbouring centre-line pixels adds the true angle between
    them -- so a pipe near the ceiling, where the canvas stretches, is not
    counted long. Degrees, not metres: a length on the floor needs the camera
    height and the ceiling's, which the ceiling plan will supply.
    """
    import cv2

    image, r0, c0, shift = crop(indices, width)
    line = thin(image)
    ys, xs = np.nonzero(line)
    if len(ys) < 2:
        return 0.0
    height = width // 2

    # Thinning eats each end of a line back by about half its width, so a
    # thick-looking pipe measured short -- 37.5 of 40 degrees for a band 2.4
    # degrees thick. Each END of the centre line gets its local half-width back:
    # the distance from that end pixel to the edge of the shape.
    padded = np.pad(line.astype(np.uint8), 1)
    neighbours = sum(
        np.roll(np.roll(padded, dy, axis=0), dx, axis=1)
        for dy in (-1, 0, 1)
        for dx in (-1, 0, 1)
        if (dy, dx) != (0, 0)
    )[1:-1, 1:-1]
    ends = line & (neighbours == 1)
    half_widths = cv2.distanceTransform(image, cv2.DIST_L2, 3)[ends]
    total = float(half_widths.sum()) * 180.0 / height
    for dy, dx in ((0, 1), (1, -1), (1, 0), (1, 1)):
        ny, nx = ys + dy, xs + dx
        valid = (ny < line.shape[0]) & (nx >= 0) & (nx < line.shape[1])
        valid[valid] = line[ny[valid], nx[valid]]
        if not valid.any():
            continue
        lon_a = np.asarray(column_to_lon_deg((xs[valid] + c0 - shift).astype(np.float64), width))
        lat_a = np.asarray(row_to_lat_deg((ys[valid] + r0).astype(np.float64), height))
        lon_b = np.asarray(column_to_lon_deg((nx[valid] + c0 - shift).astype(np.float64), width))
        lat_b = np.asarray(row_to_lat_deg((ny[valid] + r0).astype(np.float64), height))
        total += float(np.sum(_angles_deg(lon_a, lat_a, lon_b, lat_b)))
    return total


def _angles_deg(lon_a, lat_a, lon_b, lat_b) -> np.ndarray:
    """Great-circle angles between arrays of directions -- the half-chord
    form `projection.angle_between_deg` uses, which stays exact for the tiny
    angles between neighbouring pixels, vectorised."""
    ax, ay, az = direction(lon_a, lat_a)
    bx, by, bz = direction(lon_b, lat_b)
    apart = np.sqrt((ax - bx) ** 2 + (ay - by) ** 2 + (az - bz) ** 2)
    together = np.sqrt((ax + bx) ** 2 + (ay + by) ** 2 + (az + bz) ** 2)
    return np.degrees(2.0 * np.arctan2(apart, together))


__all__ = [
    "canvas_width_for",
    "footprint",
    "length_deg",
    "outlines",
    "overlap_where_both_look",
    "paint",
    "shared_region",
    "thin",
]
