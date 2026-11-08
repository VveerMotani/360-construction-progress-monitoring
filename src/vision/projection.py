"""Virtual cameras on the sphere: the layer that lets a detector see at all.

WHY THIS EXISTS

    An object detector is trained on ordinary photographs. Our panoramas are
    unrolled spheres, which are not that:

      * straight lines bend, and the bending depends on where in the frame the
        object sits;
      * anything near the ceiling or floor is smeared across a huge width, so a
        ceiling tile occupies a shape no camera has ever produced;
      * an object at the left edge continues at the RIGHT edge. Half a door at
        each end of the image, and a detector will either miss it or find two.

    Running a detector straight over an equirectangular frame therefore does not
    give slightly worse results. It gives results whose errors are a function of
    WHERE the object was, which is precisely the correlation that makes an
    accuracy number meaningless.

    So: resample the sphere into a set of ordinary rectilinear photographs --
    what a normal camera would have seen pointing that way -- detect in those,
    and convert each box back to a direction. That is the standard approach and
    it is the whole of this module.

THE ANGLE CONVENTION, AND WHY IT IS NOT THE METRIC ONE

    This module speaks IMAGE angles only:

        lon   degrees, 0 at the image's centre column, increasing to the RIGHT,
              wrapping at +/-180 (the left and right edges).
        lat   degrees, +90 at the top row, -90 at the bottom.

    That is exactly what `make_demo_panoramas.py` writes and what Photo Sphere
    Viewer reads, and `test_projection.py` pins it against both.

    It is deliberately NOT the metric heading that the rest of the system
    stores. Converting image longitude to a direction on the floor plan needs
    `north_offset_deg` (`metric = north_offset - lon`, settled in
    design note 7.x). Keeping that conversion out of here means the
    projection maths is proven on its own terms, and the offset lives at one
    boundary rather than in this file.

    Detections cross into metric headings at the point they are stored, next to
    the same conversion field notes already use.

GEOMETRY

    A direction is a unit vector, with +X to the right at lon 0, +Y up, +Z
    forward along lon 0:

        x = cos(lat) sin(lon)     y = sin(lat)     z = cos(lat) cos(lon)

    A view is a pinhole camera looking down its own +Z. Rotating view space to
    world space is a pitch about X followed by a yaw about Y; `_to_world` and
    `_to_view` are exact inverses and a test asserts it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

# ---------------------------------------------------------------------------
# The equirectangular convention
#
# One place, because a half-pixel or a sign here rotates every detection in the
# system and nothing would look obviously broken.
# ---------------------------------------------------------------------------

def column_to_lon_deg(x: np.ndarray | float, width: int) -> np.ndarray | float:
    """Image column -> longitude. Pixel centres, hence the +0.5."""
    return ((np.asarray(x, dtype=np.float64) + 0.5) / width - 0.5) * 360.0


def row_to_lat_deg(y: np.ndarray | float, height: int) -> np.ndarray | float:
    """Image row -> latitude. Row 0 is the ceiling, not the floor."""
    return (0.5 - (np.asarray(y, dtype=np.float64) + 0.5) / height) * 180.0


def lon_deg_to_column(lon: np.ndarray | float, width: int) -> np.ndarray | float:
    return (np.asarray(lon, dtype=np.float64) / 360.0 + 0.5) * width - 0.5


def lat_deg_to_row(lat: np.ndarray | float, height: int) -> np.ndarray | float:
    return (0.5 - np.asarray(lat, dtype=np.float64) / 180.0) * height - 0.5


def wrap_lon_deg(lon: np.ndarray | float) -> np.ndarray | float:
    """Fold a longitude into (-180, +180]."""
    return (np.asarray(lon, dtype=np.float64) + 180.0) % 360.0 - 180.0


def direction(lon_deg: np.ndarray | float, lat_deg: np.ndarray | float):
    """Unit vector for an image angle."""
    lon = np.radians(np.asarray(lon_deg, dtype=np.float64))
    lat = np.radians(np.asarray(lat_deg, dtype=np.float64))
    cos_lat = np.cos(lat)
    return cos_lat * np.sin(lon), np.sin(lat), cos_lat * np.cos(lon)


def angle_between_deg(
    lon_a: float, lat_a: float, lon_b: float, lat_b: float
) -> float:
    """Great-circle angle between two directions, in degrees.

    The honest distance between two points on a sphere. Comparing longitudes
    directly instead is the classic mistake: two directions 10 degrees apart in
    longitude are almost the same direction near the poles and quite different
    at the horizon.

    NOT `acos(a . b)`, which is the obvious formula and is useless here. Near
    zero the dot product is 1 - eps, and acos loses about half its significant
    digits: two IDENTICAL directions come back a millionth of a degree apart.
    That is harmless for a big angle and fatal for the small ones, which are the
    only ones anything asks about -- de-duplicating detections is entirely a
    question of whether two angles are nearly the same. The half-chord form via
    atan2 is exact at both ends.
    """
    ax, ay, az = direction(lon_a, lat_a)
    bx, by, bz = direction(lon_b, lat_b)
    apart = math.sqrt((ax - bx) ** 2 + (ay - by) ** 2 + (az - bz) ** 2)
    together = math.sqrt((ax + bx) ** 2 + (ay + by) ** 2 + (az + bz) ** 2)
    return math.degrees(2.0 * math.atan2(apart, together))


# ---------------------------------------------------------------------------
# One virtual camera
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ViewSpec:
    """A pinhole camera pointed at (lon, lat), rendering `width` x `height`.

    `fov_deg` is HORIZONTAL. The vertical field follows from the aspect ratio,
    so a square view is square in angle too.
    """

    lon_deg: float
    lat_deg: float
    fov_deg: float
    width: int
    height: int

    @property
    def focal_px(self) -> float:
        """Pixels per unit of tangent -- the pinhole's focal length."""
        return (self.width / 2.0) / math.tan(math.radians(self.fov_deg) / 2.0)

    @property
    def vertical_fov_deg(self) -> float:
        return 2.0 * math.degrees(math.atan((self.height / 2.0) / self.focal_px))

    # -- view space <-> world space ----------------------------------------

    def _to_world(self, x, y, z):
        lon = math.radians(self.lon_deg)
        lat = math.radians(self.lat_deg)
        cl, sl = math.cos(lat), math.sin(lat)
        cy, sy = math.cos(lon), math.sin(lon)
        y1 = y * cl + z * sl
        z1 = -y * sl + z * cl
        return x * cy + z1 * sy, y1, -x * sy + z1 * cy

    def _to_view(self, x, y, z):
        lon = math.radians(self.lon_deg)
        lat = math.radians(self.lat_deg)
        cl, sl = math.cos(lat), math.sin(lat)
        cy, sy = math.cos(lon), math.sin(lon)
        x1 = x * cy - z * sy
        z1 = x * sy + z * cy
        return x1, y * cl - z1 * sl, y * sl + z1 * cl

    # -- the two directions of the mapping ---------------------------------

    def pixel_to_lon_lat(self, u, v):
        """Where a pixel of this view is looking, in image angles.

        `u`, `v` may be arrays. Pixel centres, so (0, 0) is the middle of the
        top-left pixel and the view's centre falls at ((w-1)/2, (h-1)/2).
        """
        u = np.asarray(u, dtype=np.float64)
        v = np.asarray(v, dtype=np.float64)
        focal = self.focal_px
        x = (u + 0.5 - self.width / 2.0) / focal
        y = -(v + 0.5 - self.height / 2.0) / focal
        z = np.ones_like(x)

        norm = np.sqrt(x * x + y * y + z * z)
        wx, wy, wz = self._to_world(x / norm, y / norm, z / norm)
        return np.degrees(np.arctan2(wx, wz)), np.degrees(np.arcsin(np.clip(wy, -1.0, 1.0)))

    def lon_lat_to_pixel(self, lon_deg, lat_deg):
        """The inverse. Returns (u, v, in_front).

        `in_front` is False for directions BEHIND the camera. Those still
        produce finite pixel coordinates -- the projection has no idea it is
        looking backwards -- so anything using this must check it. Dropping the
        flag puts objects behind you into the frame, mirrored.
        """
        x, y, z = direction(lon_deg, lat_deg)
        vx, vy, vz = self._to_view(x, y, z)
        in_front = np.asarray(vz) > 1e-9
        safe = np.where(in_front, vz, 1.0)
        focal = self.focal_px
        u = np.asarray(vx) / safe * focal + self.width / 2.0 - 0.5
        v = -np.asarray(vy) / safe * focal + self.height / 2.0 - 0.5
        return u, v, in_front

    def contains(self, lon_deg: float, lat_deg: float, margin_px: float = 0.0) -> bool:
        """Is this direction inside the rendered image?"""
        u, v, in_front = self.lon_lat_to_pixel(lon_deg, lat_deg)
        if not bool(in_front):
            return False
        return bool(
            margin_px <= u <= self.width - 1 - margin_px
            and margin_px <= v <= self.height - 1 - margin_px
        )


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

class ViewSampler:
    """The pixel mapping for one view, computed once and reused per frame.

    WHY THIS IS NOT PREMATURE

        Rendering sixteen views of one panorama took 3.19 seconds. YOLO ran the
        same sixteen through a GPU in 0.12. Ninety-six per cent of a detection
        pass was spent preparing images, not looking at them -- twelve minutes
        for a two-hundred-frame walk, against forty seconds of actual model.

        The saving is free, because the mapping from a view's pixel to a
        panorama's pixel depends only on the view and the image SIZE, never on
        what the image contains. Every frame of a walk is the same size, so the
        trigonometry is identical every time. It was being redone two hundred
        times.

    What is cached is the four corner indices and the two blend weights per
    output pixel -- everything except the gather itself. Roughly 10 MB per view
    at 640 px, so the default set costs about 160 MB resident in the worker.
    """

    __slots__ = ("spec", "source", "_xa", "_xb", "_ya", "_yb", "_fx", "_fy")

    def __init__(self, spec: ViewSpec, source_width: int, source_height: int) -> None:
        self.spec = spec
        self.source = (source_height, source_width)

        us, vs = np.meshgrid(
            np.arange(spec.width, dtype=np.float64),
            np.arange(spec.height, dtype=np.float64),
        )
        lon, lat = spec.pixel_to_lon_lat(us, vs)

        xs = np.asarray(lon_deg_to_column(lon, source_width))
        ys = np.clip(
            np.asarray(lat_deg_to_row(lat, source_height)), 0.0, source_height - 1.0
        )

        x0 = np.floor(xs).astype(np.int64)
        y0 = np.floor(ys).astype(np.int64)

        # Wrap in x, clamp in y. A view straddling the image seam reads columns
        # from both ends; clamping there smears the last column across half the
        # output, which is the artefact this whole module exists to avoid.
        self._xa = (x0 % source_width).astype(np.int32)
        self._xb = ((x0 + 1) % source_width).astype(np.int32)
        self._ya = np.clip(y0, 0, source_height - 1).astype(np.int32)
        self._yb = np.clip(y0 + 1, 0, source_height - 1).astype(np.int32)
        self._fx = (xs - x0).astype(np.float32)[..., None]
        self._fy = (ys - y0).astype(np.float32)[..., None]

    def sample(self, panorama: np.ndarray) -> np.ndarray:
        if panorama.shape[:2] != self.source:
            raise ValueError(
                f"sampler built for {self.source[1]}x{self.source[0]}, "
                f"given {panorama.shape[1]}x{panorama.shape[0]}"
            )
        # Gather FIRST, widen after. Converting the panorama to float up front
        # costs a 100 MB temporary for an 8K frame, and it was being paid once
        # per view -- sixteen times for four hundred thousand pixels actually
        # read. Indexing uint8 and letting the multiply promote touches only
        # what is sampled, and numpy widens the corners for free.
        top = (
            panorama[self._ya, self._xa] * (1.0 - self._fx)
            + panorama[self._ya, self._xb] * self._fx
        )
        bottom = (
            panorama[self._yb, self._xa] * (1.0 - self._fx)
            + panorama[self._yb, self._xb] * self._fx
        )
        return top * (1.0 - self._fy) + bottom * self._fy


# Bounded so a worker cannot accumulate samplers for every size it ever sees.
# The default set is 16, and a run uses one panorama size, so 64 holds several
# view sets at once without the cache ever being the reason memory grows.
@lru_cache(maxsize=64)
def sampler_for(spec: ViewSpec, source_width: int, source_height: int) -> ViewSampler:
    return ViewSampler(spec, source_width, source_height)


def _sample_bilinear(image: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Bilinear sample of an (H, W, C) image, WRAPPING in x, clamping in y.

    The wrap is the point. A view straddling the image seam reads columns from
    both ends, and clamping there would smear the last column across half the
    output -- the exact artefact this whole module exists to avoid.
    """
    height, width = image.shape[:2]
    x0 = np.floor(xs).astype(np.int64)
    y0 = np.floor(ys).astype(np.int64)
    fx = (xs - x0)[..., None]
    fy = (ys - y0)[..., None]

    xa, xb = x0 % width, (x0 + 1) % width
    ya = np.clip(y0, 0, height - 1)
    yb = np.clip(y0 + 1, 0, height - 1)

    top = image[ya, xa] * (1.0 - fx) + image[ya, xb] * fx
    bottom = image[yb, xa] * (1.0 - fx) + image[yb, xb] * fx
    return top * (1.0 - fy) + bottom * fy


def render(panorama: np.ndarray, spec: ViewSpec) -> np.ndarray:
    """Resample an equirectangular frame into one flat view.

    `panorama` is (H, W, C) and is expected to be 2:1. Returns the same dtype.
    """
    if panorama.ndim != 3:
        raise ValueError(f"expected an (H, W, C) panorama, got shape {panorama.shape}")

    height, width = panorama.shape[:2]
    out = sampler_for(spec, width, height).sample(panorama)
    if np.issubdtype(panorama.dtype, np.integer):
        info = np.iinfo(panorama.dtype)
        return np.clip(np.rint(out), info.min, info.max).astype(panorama.dtype)
    return out.astype(panorama.dtype)


# ---------------------------------------------------------------------------
# Which views to render
# ---------------------------------------------------------------------------

# Three rings, deliberately overlapping.
#
# OVERLAP IS NOT WASTE. It is the only thing that saves an object cut by a view
# edge: whatever straddles a boundary in one view sits whole in its neighbour,
# and the duplicate is cheap to drop afterwards. Butt-jointed views (a plain
# cubemap) move the seam problem rather than solving it.
#
# The vertical spread is chosen for what this product looks at. A horizon ring
# alone would miss suspended ceiling grid and high-level cable tray, which is
# most of what "is this floor fitted out" actually means.
#
# 90 degree square views: 6 around the horizon covers 540 degrees of the 360
# available, and the +/-45 rings reach the poles because a square view centred
# at 45 spans latitude 0 to 90 at its centre column.
# The two caps are not decoration. A ring at +/-45 reaches the pole only at the
# centre column of each view, so straight up lands exactly ON an edge of four
# views and comfortably inside none of them -- the one direction guaranteed to
# be cut. `test_anything_cut_by_a_view_edge_is_whole_in_another` found that,
# with 552 stranded directions clustered at the poles. Straight up is where the
# ceiling grid is, so that is not a corner worth cutting.
_DEFAULT_RINGS: tuple[tuple[float, int, float], ...] = (
    (0.0, 6, 0.0),      # horizon: walls, furniture, doors
    (45.0, 4, 45.0),    # up: ceiling grid, tray, ducting
    (-45.0, 4, 45.0),   # down: flooring, skirting, furniture bases
    (90.0, 1, 0.0),     # zenith cap
    (-90.0, 1, 0.0),    # nadir cap -- mostly the operator's own hard hat
)

DEFAULT_FOV_DEG = 90.0
DEFAULT_VIEW_PX = 640

# Named ring layouts, so "how many views" is a choice with a name rather than a
# number nobody can reason about. Measured in docs/view_geometry.md, where the
# figure that matters is the WORST-CASE CLEARANCE -- the largest object
# guaranteed whole inside some single view, wherever on the sphere it sits:
#
#     default  16 views   8.2 deg clearance   the standard set
#     cube      6 views   0.0 deg             covers the sphere, guarantees nothing
#     rings    14 views   0.1 deg             the default without its pole caps
#     dense    22 views  16.5 deg             1.4x the cost, 2x the guarantee
#
# `cube` and `rings` are here to be MEASURED AGAINST, not used: they are the two
# arrangements that look reasonable and are not. Keeping them runnable is what
# makes that claim checkable on real frames rather than only on geometry.
LAYOUTS: dict[str, tuple[tuple[float, int, float], ...]] = {
    "default": _DEFAULT_RINGS,
    "cube": ((0.0, 4, 0.0), (90.0, 1, 0.0), (-90.0, 1, 0.0)),
    "rings": ((0.0, 6, 0.0), (45.0, 4, 45.0), (-45.0, 4, 45.0)),
    "dense": (
        (0.0, 8, 0.0), (50.0, 6, 30.0), (-50.0, 6, 30.0),
        (90.0, 1, 0.0), (-90.0, 1, 0.0),
    ),
}


def views_for(
    layout: str = "default",
    size: int = DEFAULT_VIEW_PX,
    fov_deg: float = DEFAULT_FOV_DEG,
) -> list[ViewSpec]:
    """Build a named view set. The three knobs a detection run can turn.

    Field of view and pixel size are the two questions docs/view_geometry.md
    left open and could not answer from geometry alone -- 100 degrees buys a
    26.2 deg guarantee for the same sixteen renders but costs 19% of the angular
    resolution, and 640 px already discards about 4x the detail at the horizon.
    Both are settled by measuring recall on real frames, which means both have
    to be reachable from a run rather than edited in the source.
    """
    rings = LAYOUTS.get(layout)
    if rings is None:
        raise ValueError(
            f"unknown view layout {layout!r}; expected one of {sorted(LAYOUTS)}"
        )

    views: list[ViewSpec] = []
    for lat_deg, count, offset_deg in rings:
        step = 360.0 / count
        for index in range(count):
            views.append(
                ViewSpec(
                    lon_deg=float(wrap_lon_deg(index * step + offset_deg)),
                    lat_deg=lat_deg,
                    fov_deg=fov_deg,
                    width=size,
                    height=size,
                )
            )
    return views


def default_views(
    size: int = DEFAULT_VIEW_PX, fov_deg: float = DEFAULT_FOV_DEG
) -> list[ViewSpec]:
    """The standard set of virtual cameras for one panorama.

    Sixteen views. The rings above and below are rotated half a step off the
    horizon ring so their vertical edges do not line up with it -- an object
    unlucky enough to sit on a boundary is then cut in at most one ring.
    """
    return views_for("default", size, fov_deg)


# ---------------------------------------------------------------------------
# Back from a box to a direction
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SphereBox:
    """Where a detection is, once it is off the flat view and back on the sphere.

    An angular extent rather than a rectangle, because a rectangle on a sphere
    is not a thing. `lon_deg`/`lat_deg` are the box centre; the radii are half
    its angular width and height as seen from the camera.

    This is the same shape as a field note's anchor -- a direction with no
    distance -- which is the whole reason detections can reuse the note pins,
    the mini map and the plan overlay unchanged. A 360 photograph carries no
    depth, for a person or for a detector.
    """

    lon_deg: float
    lat_deg: float
    lon_radius_deg: float
    lat_radius_deg: float

    @property
    def lon_half_span_deg(self) -> float:
        """The box's half-width as a LONGITUDE offset rather than an angle.

        `lon_radius_deg` is a true angular width -- how wide the thing looks
        from the camera -- and near the poles that is not the same as a span of
        longitude. Lines of longitude converge, so covering ten degrees of
        actual width at latitude 60 takes twenty degrees of longitude, and at
        latitude 80 it takes nearly sixty.

        Anything that DRAWS the box needs this; anything that measures or
        compares it wants the angular radius. Drawing with the angular radius
        squashes every ceiling and floor box towards the vertical, which is
        exactly how this was noticed.

        CAPPED AT HALF A TURN, because the conversion runs away at the poles --
        a modest 8 degree width becomes 2292 degrees of longitude at latitude
        89.8, and eight million at the pole itself. That is not a bug in the
        arithmetic; longitude genuinely stops meaning anything where the lines
        meet. 180 is the honest answer there: a box at the pole covers EVERY
        longitude, because it is a cap around the pole rather than a rectangle.

        Near the poles a centre-and-two-radii box is a poor description of the
        real shape whatever we do with it. Storing the traced outline is the
        proper fix.
        """
        import math

        span = self.lon_radius_deg / max(math.cos(math.radians(self.lat_deg)), 1e-6)
        return min(span, 180.0)

    def separation_deg(self, other: SphereBox) -> float:
        return angle_between_deg(
            self.lon_deg, self.lat_deg, other.lon_deg, other.lat_deg
        )

    def _overlap_area(self, other: SphereBox) -> float:
        """Angular area the two boxes share, in square degrees.

        Longitude differences are scaled by cos(latitude) before being compared
        with the radii. The radii are true angular widths -- `box_to_sphere`
        measured them along great circles -- while a difference of longitudes is
        not: sixty degrees of longitude near the ceiling is a few degrees of
        actual angle. Comparing the two unscaled is the mistake that makes every
        ceiling fixture look like several.
        """
        mean_lat = math.radians((self.lat_deg + other.lat_deg) / 2.0)
        scale = max(math.cos(mean_lat), 1e-6)
        centre_lon = float(wrap_lon_deg(other.lon_deg - self.lon_deg)) * scale

        width = min(self.lon_radius_deg, centre_lon + other.lon_radius_deg) - max(
            -self.lon_radius_deg, centre_lon - other.lon_radius_deg
        )
        centre_lat = other.lat_deg - self.lat_deg
        height = min(self.lat_radius_deg, centre_lat + other.lat_radius_deg) - max(
            -self.lat_radius_deg, centre_lat - other.lat_radius_deg
        )
        return max(0.0, width) * max(0.0, height)

    @property
    def area_deg2(self) -> float:
        return 4.0 * self.lon_radius_deg * self.lat_radius_deg

    def iou(self, other: SphereBox) -> float:
        """Intersection over union. Zero when they do not overlap at all."""
        shared = self._overlap_area(other)
        if shared <= 0.0:
            return 0.0
        return shared / (self.area_deg2 + other.area_deg2 - shared)

    def containment(self, other: SphereBox) -> float:
        """How much of the SMALLER box lies inside the larger one.

        IoU alone cannot recognise a truncated duplicate. A box clipped by a
        view's edge is genuinely much smaller than the intact view of the same
        object, so their IoU is low however well they line up -- 0.14 for the
        case in the tests -- while the small one sits entirely within the big
        one. That is the signature of one object seen twice, not of two objects.
        """
        smaller = min(self.area_deg2, other.area_deg2)
        if smaller <= 0.0:
            return 0.0
        return self._overlap_area(other) / smaller


def box_to_sphere(
    spec: ViewSpec, x0: float, y0: float, x1: float, y1: float
) -> SphereBox:
    """Convert a detector's pixel box in one view into an angular extent.

    The centre goes through the exact projection. The radii are measured from
    the box's own edge midpoints rather than its corners: a corner of a wide box
    is further from the centre than its edge is, and using corners inflates
    every box by a factor that grows towards the edge of the view.
    """
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    lon_c, lat_c = spec.pixel_to_lon_lat(cx, cy)
    lon_c, lat_c = float(lon_c), float(lat_c)

    lon_l, lat_l = spec.pixel_to_lon_lat(x0, cy)
    lon_r, lat_r = spec.pixel_to_lon_lat(x1, cy)
    lon_t, lat_t = spec.pixel_to_lon_lat(cx, y0)
    lon_b, lat_b = spec.pixel_to_lon_lat(cx, y1)

    horizontal = angle_between_deg(float(lon_l), float(lat_l), float(lon_r), float(lat_r))
    vertical = angle_between_deg(float(lon_t), float(lat_t), float(lon_b), float(lat_b))

    return SphereBox(
        lon_deg=float(wrap_lon_deg(lon_c)),
        lat_deg=lat_c,
        lon_radius_deg=horizontal / 2.0,
        lat_radius_deg=vertical / 2.0,
    )
