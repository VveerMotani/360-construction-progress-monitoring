"""A second look along pipes: new views where the first pass found pipe.

Research experiment 06 (research/experiments/2026-09-23-06-second-look).
Most sure pipe masks touch the edge of their view: the pipe runs
out of the frame, and the model saw only part of it. So after the ordinary
sixteen views, each pipe-like find above eye level gets new views -- one
centred on it and one just past each of its loose ends -- at 60 degrees and
the model's own 1008 px, and what they find is stitched in with the rest.

On the site photos this found more pipe for some extra model time a photo.
Rendering the ordinary views sharper, or zooming over the whole ceiling, did
not do the same: the gain comes from views that put the pipe in the middle,
not from pixels.

Pure geometry: which views to add. Running the model and stitching are the
caller's, as for the first pass.
"""

from __future__ import annotations

import math

import numpy as np

from vision.projection import (
    ViewSpec,
    angle_between_deg,
    column_to_lon_deg,
    row_to_lat_deg,
    wrap_lon_deg,
)

LOOK_FOV_DEG = 60.0
# SAM 3 resizes every view to 1008 x 1008; rendering at that size gives it the
# photo's detail instead of an upscaled 640.
LOOK_PX = 1008
# How far past a loose end the new view is centred, along the pipe.
BEYOND_DEG = 18.0
MAX_LOOKS = 24
# New views closer than this to one already chosen add nothing.
MIN_APART_DEG = 12.0
# Services run on the ceiling and high on the walls. A "pipe" whose middle is
# below eye level is a floor cable or a false find; its look is not worth a view.
MIN_LAT_DEG = 0.0
# The canvas the second look is stitched on: as sharp as 1008-px views at
# 90 degrees (11.4 px/deg). The 60-degree views are sharper still; stitching at
# their full 16.8 px/deg doubled the time for no measured gain.
CANVAS_WIDTH = 4096


def pipe_ends(
    indices: np.ndarray, width: int, max_ends: int = 4
) -> list[tuple[float, float, float, float]]:
    """Up to `max_ends` loose ends of a pipe's centre line, each as
    (end lon, end lat, lon, lat of the point BEYOND_DEG further along the pipe).

    Ends are centre-line pixels with one neighbour; the direction is from a
    point a dozen pixels back along the line to the end. Farthest-apart ends
    first, so a long pipe's two real ends beat the stubs of its branches.
    """
    from vision.stitch import crop, thin

    image, r0, c0, shift = crop(indices, width)
    line = thin(image.astype(bool))
    ys, xs = np.nonzero(line)
    if len(ys) < 8:
        return []
    on = set(zip(ys.tolist(), xs.tolist(), strict=True))
    steps = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dy or dx]

    def neighbours(y: int, x: int) -> list[tuple[int, int]]:
        return [(y + dy, x + dx) for dy, dx in steps if (y + dy, x + dx) in on]

    ends = sorted((y, x) for y, x in on if len(neighbours(y, x)) == 1)
    if not ends:
        return []
    points = np.array(ends, dtype=float)
    centre = points.mean(axis=0)
    chosen = [ends[int(np.argmax(np.hypot(*(points - centre).T)))]]
    while len(chosen) < min(max_ends, len(ends)):
        gaps = np.min([np.hypot(points[:, 0] - y, points[:, 1] - x) for y, x in chosen], axis=0)
        chosen.append(ends[int(np.argmax(gaps))])

    def lon_lat(row: float, col: float) -> tuple[float, float]:
        return (
            float(wrap_lon_deg(column_to_lon_deg(float(col + c0 - shift), width))),
            float(np.clip(row_to_lat_deg(float(row + r0), width // 2), -80.0, 80.0)),
        )

    reach = BEYOND_DEG * width / 360.0
    out = []
    for y, x in chosen:
        path, seen = [(y, x)], {(y, x)}
        while len(path) < 12:
            ahead = [p for p in neighbours(*path[-1]) if p not in seen]
            if not ahead:
                break
            seen.add(ahead[0])
            path.append(ahead[0])
        if len(path) < 4:
            continue
        back_y, back_x = path[-1]
        dy, dx = y - back_y, x - back_x
        norm = math.hypot(dy, dx) or 1.0
        out.append((*lon_lat(y, x), *lon_lat(y + dy / norm * reach, x + dx / norm * reach)))
    return out


def look_views(pipes: list[dict]) -> list[ViewSpec]:
    """The new views for a list of pipe finds, each
    {"length": deg, "centre": (lon, lat), "ends": [...pipe_ends...]}: one
    centred on each find above eye level and one past each of its ends, longest
    finds first, none closer than MIN_APART_DEG to another, at most MAX_LOOKS."""
    views: list[ViewSpec] = []

    def add(lon: float, lat: float) -> None:
        if len(views) >= MAX_LOOKS:
            return
        if any(angle_between_deg(v.lon_deg, v.lat_deg, lon, lat) < MIN_APART_DEG for v in views):
            return
        views.append(
            ViewSpec(lon_deg=lon, lat_deg=lat, fov_deg=LOOK_FOV_DEG, width=LOOK_PX, height=LOOK_PX)
        )

    for pipe in sorted(pipes, key=lambda p: -p["length"]):
        if pipe["centre"][1] < MIN_LAT_DEG:
            continue
        add(*pipe["centre"])
        for _, _, lon, lat in pipe["ends"]:
            if lat >= MIN_LAT_DEG - 10.0:
                add(lon, lat)
    return views
