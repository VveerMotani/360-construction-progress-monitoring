"""Experiment 13: thin pipe centrelines from a tube filter on redness, scored.

    python research/tube_filter.py ridges      # ridge maps for the four traced photos (cached)
    python research/tube_filter.py score       # recall / precision against the hand traces

WHY

SAM 3 returns blobs: in one find, pipes, hose coils, paint and the space
between two runs, merged (experiment 12). The layout needs lines. A tube
filter -- the kind used to find blood vessels in scans -- answers at every
pixel "is this the middle of a long, thin, brighter-than-around band of about
this width?", at several widths, and the answer can be thinned to one-pixel
centrelines with the junctions left visible.

HOW

1. The panorama is cut into five flat views (up, and the four sides) of
   100 degrees, so a straight pipe is a straight band of steady width; the
   stretched panorama would smear it near the top.
2. The filter runs on redness (Lab a*, minus the concrete's, clipped), so grey
   concrete and black conduit give nothing.
3. At each width (sigma) the image's second derivatives say how sharply it
   curves across and along each point; a tube curves sharply across and
   hardly at all along (Frangi et al. 1998). The best width per pixel is kept,
   which also estimates the pipe's width.
4. Only the crest of each band is kept (non-maximum suppression across it),
   weak crests are kept only where joined to strong ones (hysteresis), short
   bits are dropped, and each view keeps only its own part of the sphere.
5. Crest pixels inside the flat-paint region of experiment 12 are dropped.

Scored like experiment 11, on the 1/4 canvas, within 20 full-resolution px:
recall = share of the sure traced pipe near a centreline; precision = share of
centreline near any trace. For SAM, precision is given both for its outline
(experiment 11's number) and for its centreline, which is the fair comparison.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paint_filter as pf  # noqa: E402
import score_traces as st  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "research" / "experiments" / "2026-10-07-13-tube-filter"
CACHE = ROOT / ".cache" / "tube_filter"
RUNS = pf.RUNS
SCALE = st.SCALE
TOL_C = st.TOL_PX // SCALE
MIN_LAT_DEG = st.MIN_LAT_DEG

FACE_PX = 2048
FACE_FOV = 100.0                     # a little over 90 so a pipe at a view's edge is still whole
SIGMAS = (1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 9.0, 13.0)  # face px; a band of width w peaks near sigma = 0.29 w
BETA = 0.5                           # how strongly a blob (round) is told apart from a tube (long)
C_RED = 5.0                          # a* units: how much curvature counts as "structure"
CONCRETE_A = 8.0                     # a* of the grey concrete, taken off before filtering
MIN_PIECE = 12                       # face px: shorter crest pieces are dropped


# ---------------------------------------------------------------------------
# flat views
# ---------------------------------------------------------------------------

def faces() -> list[tuple[str, np.ndarray, np.ndarray, np.ndarray]]:
    """(name, forward, right, up) for the up view and the four sides."""
    out = []
    z = np.array([0.0, 0.0, 1.0])
    out.append(("up", z, np.array([1.0, 0.0, 0.0]), np.array([0.0, -1.0, 0.0])))
    for k, az in enumerate((0, 90, 180, 270)):
        f = np.array([math.cos(math.radians(az)), math.sin(math.radians(az)), 0.0])
        r = np.cross(f, z)
        out.append((f"side{k}", f, r, np.cross(r, f)))
    return out


def face_dirs(f: np.ndarray, r: np.ndarray, u: np.ndarray) -> np.ndarray:
    k = math.tan(math.radians(FACE_FOV) / 2)
    g = (np.arange(FACE_PX) + 0.5) / FACE_PX * 2 - 1
    gx, gy = np.meshgrid(g * k, -g * k)
    d = f[None, None] + gx[..., None] * r + gy[..., None] * u
    return d / np.linalg.norm(d, axis=2, keepdims=True)


def lonlat(d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.degrees(np.arctan2(d[..., 1], d[..., 0])), np.degrees(np.arcsin(np.clip(d[..., 2], -1, 1)))


def own_region(d: np.ndarray, f: np.ndarray) -> np.ndarray:
    """Pixels whose direction belongs to this view's cube face (no double counting)."""
    axis = int(np.argmax(np.abs(f)))
    sign = np.sign(f[axis])
    a = np.abs(d)
    return (np.argmax(a, axis=2) == axis) & (np.sign(d[..., axis]) == sign)


# ---------------------------------------------------------------------------
# the tube filter
# ---------------------------------------------------------------------------

def _gauss_kernels(s: float):
    r = int(math.ceil(4 * s))
    x = np.arange(-r, r + 1, dtype=np.float32)
    g = np.exp(-x * x / (2 * s * s))
    g /= g.sum()
    g1 = -x / (s * s) * g
    g2 = (x * x / (s ** 4) - 1 / (s * s)) * g
    return g, g1, g2


def tubeness(img: np.ndarray):
    """Best response over SIGMAS, the sigma it came from, and the unit normal across the tube."""
    best = np.zeros_like(img)
    best_s = np.zeros_like(img)
    nx = np.zeros_like(img)
    ny = np.zeros_like(img)
    for s in SIGMAS:
        g, g1, g2 = _gauss_kernels(s)
        hxx = cv2.sepFilter2D(img, -1, g2, g) * s * s
        hyy = cv2.sepFilter2D(img, -1, g, g2) * s * s
        hxy = cv2.sepFilter2D(img, -1, g1, g1) * s * s
        tmp = np.sqrt((hxx - hyy) ** 2 + 4 * hxy ** 2)
        mu1 = (hxx + hyy + tmp) / 2
        mu2 = (hxx + hyy - tmp) / 2
        swap = np.abs(mu1) > np.abs(mu2)
        l1 = np.where(swap, mu2, mu1)   # small: along the tube
        l2 = np.where(swap, mu1, mu2)   # large: across it; negative for a bright tube
        rb = l1 / np.where(np.abs(l2) < 1e-6, 1e-6, l2)
        ss = np.sqrt(l1 ** 2 + l2 ** 2)
        v = np.exp(-rb ** 2 / (2 * BETA ** 2)) * (1 - np.exp(-ss ** 2 / (2 * C_RED ** 2)))
        v[l2 >= 0] = 0
        # eigenvector of l2 = direction across the tube
        ex, ey = hxy, l2 - hxx
        alt = np.abs(ex) + np.abs(ey) < 1e-9
        ex = np.where(alt, l2 - hyy, ex)
        ey = np.where(alt, hxy, ey)
        nrm = np.sqrt(ex ** 2 + ey ** 2) + 1e-12
        better = v > best
        best = np.where(better, v, best)
        best_s = np.where(better, s, best_s)
        nx = np.where(better, ex / nrm, nx)
        ny = np.where(better, ey / nrm, ny)
    return best, best_s, nx, ny


def crest(v: np.ndarray, nx: np.ndarray, ny: np.ndarray) -> np.ndarray:
    """Keep a pixel only where the response is highest across the tube."""
    h, w = v.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    a = cv2.remap(v, xx + nx, yy + ny, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    b = cv2.remap(v, xx - nx, yy - ny, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return np.where((v >= a) & (v >= b), v, 0)


def ridges(name: str) -> dict:
    """Crest pixels of every view, as sphere directions with response and width. Cached."""
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / f"{name[:-4]}.npz"
    if f.exists():
        return dict(np.load(f))
    im = cv2.imread(str(ROOT / "data" / "lab" / RUNS[name] / "image.jpg"))
    im = cv2.resize(im, (im.shape[1] // 2, im.shape[0] // 2), interpolation=cv2.INTER_AREA)
    H, W = im.shape[:2]
    cols = {k: [] for k in ("face", "i", "j", "lon", "lat", "v", "sigma_deg")}
    for fi, (_, fwd, r, u) in enumerate(faces()):
        d = face_dirs(fwd, r, u)
        lon, lat = lonlat(d)
        mx = ((lon + 180.0) / 360.0 * W).astype(np.float32)
        my = ((90.0 - lat) / 180.0 * H).astype(np.float32)
        view = cv2.remap(im, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        a = cv2.cvtColor(view.astype(np.float32) / 255.0, cv2.COLOR_BGR2Lab)[..., 1]
        red = np.clip(a - CONCRETE_A, 0, 40).astype(np.float32)
        v, s, nx, ny = tubeness(red)
        c = crest(v, nx, ny)
        keep = (c > 0.02) & own_region(d, fwd) & (lat >= MIN_LAT_DEG)
        ii, jj = np.nonzero(keep)
        # symmetry: a tube is redder than BOTH sides; the edge of a wide red band
        # (a paint stripe) has red on one side
        off = 2.5 * s[ii, jj] + 1.0
        fx, fy = jj.astype(np.float32), ii.astype(np.float32)
        side_a = cv2.remap(red, (fx + off * nx[ii, jj])[None], (fy + off * ny[ii, jj])[None], cv2.INTER_LINEAR)[0]
        side_b = cv2.remap(red, (fx - off * nx[ii, jj])[None], (fy - off * ny[ii, jj])[None], cv2.INTER_LINEAR)[0]
        centre = red[ii, jj]
        cols.setdefault("side", []).append((np.maximum(side_a, side_b) / (centre + 1e-3)).astype(np.float32))
        cols.setdefault("red", []).append(centre.astype(np.float32))
        # one face pixel spans this many degrees at its place in the view
        px_deg = math.degrees(2 * math.tan(math.radians(FACE_FOV) / 2) / FACE_PX)
        cols["face"].append(np.full(len(ii), fi, np.int8))
        cols["i"].append(ii.astype(np.int16))
        cols["j"].append(jj.astype(np.int16))
        cols["lon"].append(lon[ii, jj].astype(np.float32))
        cols["lat"].append(lat[ii, jj].astype(np.float32))
        cols["v"].append(c[ii, jj].astype(np.float32))
        cols["sigma_deg"].append((s[ii, jj] * px_deg).astype(np.float32))
    out = {k: np.concatenate(v_) for k, v_ in cols.items()}
    np.savez(f, **out)
    return out


# ---------------------------------------------------------------------------
# from crests to centrelines, on the canvas
# ---------------------------------------------------------------------------

SIDE_MAX = 0.5      # a side at more than half the centre's redness means a band's edge, not a tube
STRAIGHT_PX = 80    # face px (about 5 degrees): crest pixels must lie on a straight segment this long


def centrelines(rd: dict, hi: float, lo: float | None = None, side_max: float | None = SIDE_MAX,
                straight_px: int = STRAIGHT_PX) -> list[np.ndarray]:
    """Per view: crest pixels kept by symmetry, hysteresis (hi seeds, lo joins), length and straightness."""
    lo = hi / 2 if lo is None else lo
    kept = []
    for fi in range(len(faces())):
        sel = np.flatnonzero(rd["face"] == fi)
        if side_max is not None:
            sel = sel[rd["side"][sel] <= side_max]
        if not len(sel):
            continue
        img = np.zeros((FACE_PX, FACE_PX), np.uint8)
        weak = rd["v"][sel] >= lo
        img[rd["i"][sel][weak], rd["j"][sel][weak]] = 1
        n, lab = cv2.connectedComponents(img, connectivity=8)
        strong = np.zeros(n, bool)
        lab_at = lab[rd["i"][sel], rd["j"][sel]]
        strong[np.unique(lab_at[(rd["v"][sel] >= hi) & weak])] = True
        size = np.bincount(lab.ravel(), minlength=n)
        good = strong & (size >= MIN_PIECE)
        good[0] = False
        sel = sel[good[lab_at]]
        if straight_px and len(sel):
            # in a flat view an installed pipe is a straight line; a hose coil
            # or a stripe on an arched beam is not
            img[:] = 0
            img[rd["i"][sel], rd["j"][sel]] = 255
            # a crest wobbles by a pixel and has small gaps: thicken it a little
            # and let the line finder bin coarsely, or a long pipe comes back in bits
            img = cv2.dilate(img, np.ones((3, 3), np.uint8))
            segs = cv2.HoughLinesP(img, 2, np.pi / 720, threshold=max(straight_px // 2, 10),
                                   minLineLength=straight_px, maxLineGap=12)
            on = np.zeros_like(img)
            if segs is not None:
                for x0, y0, x1, y1 in segs.reshape(-1, 4):  # OpenCV 5 returns (N, 4)
                    cv2.line(on, (int(x0), int(y0)), (int(x1), int(y1)), 255, 5)
            sel = sel[on[rd["i"][sel], rd["j"][sel]] > 0]
        kept.append(sel)
    return kept


def canvas_of(rd: dict, idx: np.ndarray, w: int, h: int) -> np.ndarray:
    c = np.zeros((h // SCALE, w // SCALE), np.uint8)
    x = ((rd["lon"][idx] + 180.0) / 360.0 * w / SCALE).astype(int) % c.shape[1]
    y = np.clip(((90.0 - rd["lat"][idx]) / 180.0 * h / SCALE).astype(int), 0, c.shape[0] - 1)
    c[y, x] = 255
    return c


def line_score(name: str, lines: np.ndarray) -> dict:
    traces = json.loads((pf.TRACES / f"traces_{name[-7:-4]}.json").read_text(encoding="utf-8"))
    w, h = traces["width"], traces["height"]
    lines = st.ceiling_only(lines, h)
    t_sure = st.ceiling_only(st.trace_canvas(traces, w, h, True), h)
    t_all = st.ceiling_only(st.trace_canvas(traces, w, h, None), h)
    r, p = st.near(t_sure, lines), st.near(lines, t_all)
    return {"recall_sure": r, "precision": p, "f1": 2 * r * p / max(r + p, 1e-9), "px": int((lines > 0).sum())}


def variant_lines(name: str, hi: float, variant: str, side_max: float | None = SIDE_MAX,
                  straight_px: int = STRAIGHT_PX) -> np.ndarray:
    """Centreline canvas for a variant: tube alone, or only inside SAM's (grown) mask."""
    traces = json.loads((pf.TRACES / f"traces_{name[-7:-4]}.json").read_text(encoding="utf-8"))
    w, h = traces["width"], traces["height"]
    rd = ridges(name)
    idx = np.concatenate(centrelines(rd, hi, side_max=side_max, straight_px=straight_px))
    c = canvas_of(rd, idx, w, h)
    c[pf.paint_mask(name)] = 0
    if variant == "tube+sam":
        result = json.loads((ROOT / "data" / "lab" / RUNS[name] / "result.json").read_text(encoding="utf-8"))
        sam = pf.raster(result, w, h, ("outlines", "grown_outlines"))
        c[cv2.dilate(sam, np.ones((2 * TOL_C + 1, 2 * TOL_C + 1), np.uint8)) == 0] = 0
    return c


def overlay(name: str, lines: np.ndarray, path: Path) -> None:
    traces = json.loads((pf.TRACES / f"traces_{name[-7:-4]}.json").read_text(encoding="utf-8"))
    w, h = traces["width"], traces["height"]
    cut = int((90.0 - MIN_LAT_DEG) / 180.0 * h / SCALE)
    lab = pf.lab_small(name).astype(np.float32)
    bg = (cv2.cvtColor(lab, cv2.COLOR_Lab2BGR) * 255).clip(0, 255).astype(np.uint8)
    bg = (bg * 0.45).astype(np.uint8)
    t_all = st.trace_canvas(traces, w, h, None)
    bg[cv2.dilate(t_all, np.ones((5, 5), np.uint8)) > 0] = (0, 170, 0)
    bg[cv2.dilate(lines, np.ones((2, 2), np.uint8)) > 0] = (60, 60, 255)
    cv2.imwrite(str(path), bg[:cut])


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------

HIS = (0.05, 0.08, 0.12, 0.18, 0.25, 0.35)
HI = 0.05   # chosen on held-out photos in `score` (it won three of four folds for "tube")


def segments(name: str) -> list[list[list[float]]]:
    """The photo's fire pipe as straight segments, each an open [lon, lat] polyline.

    Crests that pass symmetry and hysteresis, minus the flat paint, are handed to
    the line finder in each flat view; a straight segment in a flat view is a
    great circle on the sphere, which is what pipe_layout cuts outlines into.
    """
    rd = ridges(name)
    result = json.loads((ROOT / "data" / "lab" / RUNS[name] / "result.json").read_text(encoding="utf-8"))
    w, h = result["width"], result["height"]
    paint = pf.paint_mask(name)
    px = ((rd["lon"] + 180.0) / 360.0 * w / SCALE).astype(int) % paint.shape[1]
    py = np.clip(((90.0 - rd["lat"]) / 180.0 * h / SCALE).astype(int), 0, paint.shape[0] - 1)
    not_paint = ~paint[py, px]
    k = math.tan(math.radians(FACE_FOV) / 2)
    out = []
    views = faces()
    for sel in centrelines(rd, HI, straight_px=0):
        sel = sel[not_paint[sel]]
        if not len(sel):
            continue
        fi = int(rd["face"][sel[0]])
        _, fwd, r, u = views[fi]
        img = np.zeros((FACE_PX, FACE_PX), np.uint8)
        img[rd["i"][sel], rd["j"][sel]] = 255
        img = cv2.dilate(img, np.ones((3, 3), np.uint8))
        segs = cv2.HoughLinesP(img, 2, np.pi / 720, threshold=max(STRAIGHT_PX // 2, 10),
                               minLineLength=STRAIGHT_PX, maxLineGap=12)
        if segs is None:
            continue
        for x0, y0, x1, y1 in segs.reshape(-1, 4):
            t = np.linspace(0, 1, 24)
            jj, ii = x0 + (x1 - x0) * t, y0 + (y1 - y0) * t
            gx = ((jj + 0.5) / FACE_PX * 2 - 1) * k
            gy = -((ii + 0.5) / FACE_PX * 2 - 1) * k
            d = fwd[None] + gx[:, None] * r[None] + gy[:, None] * u[None]
            d /= np.linalg.norm(d, axis=1, keepdims=True)
            lo, la = lonlat(d)
            out.append([[round(float(a), 4), round(float(b), 4)] for a, b in zip(lo, la)])
    return out


def cmd_apply() -> None:
    """Results whose fire pipe is the tube filter's segments, for pipe_layout --from."""
    folder = EXPERIMENT / "layout" / "results"
    folder.mkdir(parents=True, exist_ok=True)
    for name in sorted(RUNS):
        result = json.loads((ROOT / "data" / "lab" / RUNS[name] / "result.json").read_text(encoding="utf-8"))
        segs = segments(name)
        conf = max((float(f["confidence"]) for f in result["detections"] if f["label"] == "fire pipe"), default=0.0)
        res = dict(result)
        res["detections"] = [f for f in result["detections"] if f["label"] != "fire pipe"] + \
            [{"label": "fire pipe", "confidence": conf, "outlines": segs}]
        (folder / f"{name[:-4]}.json").write_text(json.dumps(res), encoding="utf-8")
        print(name[-7:-4], len(segs), "segments", flush=True)


def cmd_ridges() -> None:
    for name in pf.traced():
        rd = ridges(name)
        print(name[-7:-4], len(rd["v"]), "crest pixels")


def cmd_score() -> None:
    EXPERIMENT.mkdir(parents=True, exist_ok=True)
    names = pf.traced()
    table = {}
    for variant in ("tube", "tube+sam"):
        for hi in HIS:
            table[(variant, hi)] = {n: line_score(n, variant_lines(n, hi, variant)) for n in names}
    sam = {}
    for n in names:
        d = pf.photo_features(n)
        sam[n] = {"outline": pf.score_mask(n, d["mask"]), "centre": line_score(n, ((d["lab"] > 0) * 255).astype(np.uint8))}
    print("threshold sweep (mean of four photos):")
    for (variant, hi), rows in table.items():
        r = np.mean([x["recall_sure"] for x in rows.values()])
        p = np.mean([x["precision"] for x in rows.values()])
        print(f"  {variant:9s} hi {hi:.2f}   recall(sure) {r:.2f}   precision {p:.2f}")
    # the threshold for each photo is chosen on the other three (best F1)
    print("\neach photo with the threshold chosen on the other three:")
    out_rows = []
    for variant in ("tube", "tube+sam"):
        for held in names:
            others = [n for n in names if n != held]
            hi = max(HIS, key=lambda t: np.mean([table[(variant, t)][n]["f1"] for n in others]))
            s = table[(variant, hi)][held]
            out_rows.append({"variant": variant, "photo": held[-7:-4], "hi": hi, **s})
            overlay(held, variant_lines(held, hi, variant), EXPERIMENT / f"{variant.replace('+', '_')}_{held[-7:-4]}.png")
    for variant in ("tube", "tube+sam"):
        rs = [r for r in out_rows if r["variant"] == variant]
        print(f"  {variant:9s} " + "  ".join(f"{r['photo']}: R {r['recall_sure']:.2f} P {r['precision']:.2f} (hi {r['hi']})" for r in rs)
              + f"   mean R {np.mean([r['recall_sure'] for r in rs]):.2f} P {np.mean([r['precision'] for r in rs]):.2f}")
    print("  SAM 3     " + "  ".join(f"{n[-7:-4]}: R {sam[n]['outline']['recall_sure']:.2f} P outline {sam[n]['outline']['precision']:.2f} / centre {sam[n]['centre']['precision']:.2f}" for n in names))
    print(f"  SAM 3 mean  recall {np.mean([sam[n]['outline']['recall_sure'] for n in names]):.2f}   precision outline "
          f"{np.mean([sam[n]['outline']['precision'] for n in names]):.2f}, centreline {np.mean([sam[n]['centre']['precision'] for n in names]):.2f}")
    (EXPERIMENT / "scores.json").write_text(json.dumps({
        "sweep": {f"{v}|{h}": rows for (v, h), rows in table.items()},
        "held_out": out_rows, "sam": sam}, indent=1, default=float), encoding="utf-8")


if __name__ == "__main__":
    {"ridges": cmd_ridges, "score": cmd_score, "apply": cmd_apply}[sys.argv[1]]()
