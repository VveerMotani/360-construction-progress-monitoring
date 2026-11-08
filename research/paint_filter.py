"""Experiment 12: telling the fire pipe from the red paint stripes, and scoring it.

    python research/paint_filter.py score      # branch filters (tree, CLIP words, CLIP probe), leave one photo out
    python research/paint_filter.py pieces     # the same on the layout's straight pieces, by plan metres
    python research/paint_filter.py colour     # the colour rule that was kept, and its sensitivity
    python research/paint_filter.py apply      # colour-filtered results for every photo, for pipe_layout
    python research/pipe_layout.py pieces --from research/experiments/2026-10-07-12-paint-filter/filtered
    python research/pipe_layout.py layout --from research/experiments/2026-10-07-12-paint-filter/filtered

RESULT (experiment 12): only the colour rule -- strongly red, wide, flat
brightness -- removes paint without losing pipe, and it barely moves the
layout. The branch and CLIP methods below are the record of what did not work.

WHY BRANCHES

A SAM 3 "red pipe" find is not one pipe: on one photo a single find has
an outline thousands of degrees long and covers pipes, paint and hose coils
together. So the
decision is made on pieces of mask, not on finds. The ceiling part of the
mask is thinned to its centreline, the centreline is cut at its junctions
into branches, and each branch is measured where the photo actually is:

  width_mm    the mask's width across the branch, in millimetres, assuming the
              branch is at the fire pipe's height (1.2 m above the camera)
  shade       how much the brightness changes across the branch (a pipe is a
              lit cylinder; paint is flat)
  rim         how much darker a thin band just outside the branch is than the
              ceiling further out (a pipe throws a shadow line; paint does not)
  along       how rough the brightness is along the branch (paint shows the
              concrete through it; a pipe is smooth)
  red, contrast, chroma, hue   the colour inside, and against the ceiling

A depth-2 decision tree on the branch medians decides pipe or not; it is
small on purpose, so the rule can be read and written down. Labels come from
the hand traces of experiment 11: a branch is pipe when most of it lies
within 20 full-resolution pixels of a traced centreline, not pipe when almost
none of it does, and left out of training otherwise.

The score is experiment 11's, unchanged (score_traces.py), on the mask that is
left: every mask pixel goes with the branch nearest to it.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import score_traces as st  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
EXP10 = ROOT / "research" / "experiments" / "2026-10-01-10-whole-walk-detection"
TRACES = ROOT / "research" / "experiments" / "2026-10-02-11-hand-traces"
EXPERIMENT = ROOT / "research" / "experiments" / "2026-10-07-12-paint-filter"
CACHE = ROOT / ".cache" / "paint_filter"
RUNS = json.loads((EXP10 / "runs.json").read_text(encoding="utf-8"))

SCALE = st.SCALE          # the 1/4 canvas experiment 11 scores on
MIN_LAT_DEG = st.MIN_LAT_DEG
GAP_M = 1.2               # fire pipe above the camera (experiment 10)
TOL_C = st.TOL_PX // SCALE  # 20 full-res px on the canvas
MIN_BRANCH = 4            # canvas px; shorter branches get no vote of their own
T = np.linspace(-2.5, 2.5, 21)    # across the branch, in half-widths
S = np.linspace(-3.0, 3.0, 13)    # along the branch, in half-widths
FEATURES = ("width_mm", "shade", "rim", "along", "red", "contrast", "chroma", "hue")
PIPE_FRAC, PAINT_FRAC = 0.6, 0.1  # label thresholds: share of a branch near a trace


# ---------------------------------------------------------------------------
# mask, centreline, branches
# ---------------------------------------------------------------------------

def raster(result: dict, w: int, h: int, keys: tuple[str, ...]) -> np.ndarray:
    """Fire pipe outlines filled on the 1/4 canvas (as score_traces.mask_canvas, keys chosen)."""
    canvas = np.zeros((h // SCALE, w // SCALE), np.uint8)
    for find in result["detections"]:
        if find["label"] != "fire pipe":
            continue
        for key in keys:
            o = find.get(key)
            if isinstance(o, str):
                o = json.loads(o)
            if not o:
                continue
            if not isinstance(o[0][0], list):
                o = [o]
            for poly in o:
                pts = [st.to_px(p[0], p[1], w, h) for p in poly]
                runs, cur = [], []
                for a, b in zip(pts, pts[1:] + pts[:1]):
                    cur.append(a)
                    if abs(a[0] - b[0]) > w / 2:
                        runs.append(cur)
                        cur = []
                runs.append(cur)
                for r in runs:
                    if len(r) >= 3:
                        cv2.fillPoly(canvas, [np.array([[x / SCALE, y / SCALE] for x, y in r], np.int32)], 255)
    return canvas


def _neighbours(img: np.ndarray) -> list[np.ndarray]:
    P = np.pad(img, 1)
    return [P[:-2, 1:-1], P[:-2, 2:], P[1:-1, 2:], P[2:, 2:], P[2:, 1:-1], P[2:, :-2], P[1:-1, :-2], P[:-2, :-2]]


def thin(mask: np.ndarray) -> np.ndarray:
    """Zhang-Suen thinning to a one-pixel centreline."""
    img = (mask > 0).astype(np.uint8)
    while True:
        changed = False
        for step in (0, 1):
            p2, p3, p4, p5, p6, p7, p8, p9 = nb = _neighbours(img)
            B = sum(n.astype(np.uint8) for n in nb)
            A = sum(((nb[i] == 0) & (nb[(i + 1) % 8] == 1)).astype(np.uint8) for i in range(8))
            if step == 0:
                c = ((p2 & p4 & p6) == 0) & ((p4 & p6 & p8) == 0)
            else:
                c = ((p2 & p4 & p8) == 0) & ((p2 & p6 & p8) == 0)
            rm = (img == 1) & (B >= 2) & (B <= 6) & (A == 1) & c
            if rm.any():
                img[rm] = 0
                changed = True
        if not changed:
            return img.astype(bool)


def branches(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """Centreline cut at its junctions: label image (0 = none) and the label count."""
    sk = thin(mask)
    nb = sum(n.astype(np.uint8) for n in _neighbours(sk.astype(np.uint8)))
    junction = (sk & (nb >= 3)).astype(np.uint8)
    br = sk & ~cv2.dilate(junction, np.ones((3, 3), np.uint8)).astype(bool)
    n, lab = cv2.connectedComponents(br.astype(np.uint8), connectivity=8)
    sizes = np.bincount(lab.ravel(), minlength=n)
    small = sizes < MIN_BRANCH
    small[0] = True
    lab[small[lab]] = 0
    return lab, n


# ---------------------------------------------------------------------------
# features
# ---------------------------------------------------------------------------

def photo_features(name: str) -> dict:
    """Per-branch medians of every feature for one photo, cached."""
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / f"{name[:-4]}.npz"
    if f.exists():
        return dict(np.load(f))
    result = json.loads((ROOT / "data" / "lab" / RUNS[name] / "result.json").read_text(encoding="utf-8"))
    w, h = result["width"], result["height"]
    mask = st.ceiling_only(raster(result, w, h, ("outlines", "grown_outlines")), h)
    lab, n = branches(mask)
    ys, xs = np.nonzero(lab)
    bid = lab[ys, xs]

    # direction of the centreline from its neighbours, and the normal across it
    b = (lab > 0).astype(np.float32)
    k = 4
    X, Y = np.meshgrid(np.arange(-k, k + 1, dtype=np.float32), np.arange(-k, k + 1, dtype=np.float32))
    filt = {name_: cv2.filter2D(b, -1, ker, borderType=cv2.BORDER_CONSTANT)[ys, xs]
            for name_, ker in (("s", np.ones_like(X)), ("x", X), ("y", Y), ("xx", X * X), ("yy", Y * Y), ("xy", X * Y))}
    s = np.maximum(filt["s"], 1)
    cxx = filt["xx"] / s - (filt["x"] / s) ** 2
    cyy = filt["yy"] / s - (filt["y"] / s) ** 2
    cxy = filt["xy"] / s - (filt["x"] / s) * (filt["y"] / s)
    theta = 0.5 * np.arctan2(2 * cxy, cxx - cyy)
    tx, ty = np.cos(theta), np.sin(theta)
    nx, ny = -ty, tx

    dt = cv2.distanceTransform((mask > 0).astype(np.uint8), cv2.DIST_L2, 5)[ys, xs]
    hw = np.maximum(dt * SCALE, 4.0)  # half-width, full-res px
    lat = 90.0 - (ys + 0.5) * SCALE / h * 180.0
    deg_px = 360.0 / (w // SCALE)
    width_deg = 2 * dt * deg_px * np.sqrt((nx * np.cos(np.radians(lat))) ** 2 + ny ** 2)
    width_mm = np.radians(width_deg) * GAP_M / np.sin(np.radians(np.maximum(lat, 1.0))) * 1000

    im = cv2.imread(str(ROOT / "data" / "lab" / RUNS[name] / "image.jpg"))
    cx = xs * SCALE + SCALE / 2
    cy = ys * SCALE + SCALE / 2

    def sample(offs: np.ndarray, ux: np.ndarray, uy: np.ndarray) -> np.ndarray:
        mx = (cx[:, None] + offs[None, :] * hw[:, None] * ux[:, None]).astype(np.float32)
        my = (cy[:, None] + offs[None, :] * hw[:, None] * uy[:, None]).astype(np.float32)
        px = cv2.remap(im, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        return cv2.cvtColor(px.astype(np.float32) / 255.0, cv2.COLOR_BGR2Lab)

    across = sample(T, nx, ny)
    along = sample(S, tx, ty)
    del im
    L, A, Bc = across[..., 0], across[..., 1], across[..., 2]
    inside, mid = np.abs(T) <= 0.6, np.abs(T) <= 0.8
    rim, out = (np.abs(T) >= 0.9) & (np.abs(T) <= 1.5), np.abs(T) >= 1.75
    L_in, L_out = L[:, inside].mean(1), L[:, out].mean(1)
    a_in, b_in = A[:, inside].mean(1), Bc[:, inside].mean(1)
    feats = {
        "width_mm": width_mm,
        "shade": L[:, mid].std(1) / (L_in + 1),
        "rim": (L_out - L[:, rim].min(1)) / (L_out + 1),
        "along": along[..., 0].std(1) / (L_in + 1),
        "red": a_in,
        "contrast": a_in - A[:, out].mean(1),
        "chroma": np.hypot(a_in, b_in),
        "hue": np.degrees(np.arctan2(b_in, a_in)),
    }
    order = np.argsort(bid, kind="stable")
    bid_s = bid[order]
    ids, starts, counts = np.unique(bid_s, return_index=True, return_counts=True)
    med = {k_: np.array([np.median(v[order][s0:s0 + c]) for s0, c in zip(starts, counts)]) for k_, v in feats.items()}
    out_ = {"ids": ids, "length": counts, "mean_lat": np.array([lat[order][s0:s0 + c].mean() for s0, c in zip(starts, counts)])}
    out_.update(med)
    np.savez(f, lab=lab, mask=mask, **out_)
    return dict(np.load(f))


def labels(name: str, data: dict) -> np.ndarray:
    """Share of each branch within TOL of a traced centreline (sure or unsure)."""
    traces = json.loads((TRACES / f"traces_{name[-7:-4]}.json").read_text(encoding="utf-8"))
    t = st.ceiling_only(st.trace_canvas(traces, traces["width"], traces["height"], None), traces["height"])
    near = cv2.dilate(t, np.ones((2 * TOL_C + 1, 2 * TOL_C + 1), np.uint8)) > 0
    lab = data["lab"]
    hit = np.bincount(lab[near & (lab > 0)], minlength=lab.max() + 1)
    tot = np.bincount(lab[lab > 0], minlength=lab.max() + 1)
    return hit[data["ids"]] / np.maximum(tot[data["ids"]], 1)


def matrix(data: dict) -> np.ndarray:
    return np.stack([data[k] for k in FEATURES], axis=1)


# ---------------------------------------------------------------------------
# what a person sees: a flat crop around each branch, through CLIP
# ---------------------------------------------------------------------------

CLIP_PATH = ROOT / ".cache" / "models" / "clip" / "ViT-B-32.pt"
CROP_PX = 224
CROP_POINTS = (0.25, 0.5, 0.75)   # along each branch; their embeddings are averaged
PROMPTS = {
    "pipe": ["a photo of a red steel pipe hanging below a concrete ceiling",
             "a red fire sprinkler pipe with clamps"],
    "paint": ["a photo of a red painted stripe on a concrete beam",
              "red paint on a flat concrete surface"],
    "hose": ["a coiled red flexible hose"],
    "bare": ["a bare concrete ceiling"],
}
_clip = None


def _load_clip():
    global _clip
    if _clip is None:
        import clip
        import torch

        model, preprocess = clip.load(str(CLIP_PATH), device="cuda")
        with torch.no_grad():
            text = {k: model.encode_text(clip.tokenize(v).cuda()).float() for k, v in PROMPTS.items()}
            text = {k: torch.nn.functional.normalize(t, dim=1).mean(0) for k, t in text.items()}
        _clip = (model, preprocess, text)
    return _clip


def flat_crop(im: np.ndarray, lon: float, lat: float, fov_deg: float) -> np.ndarray:
    """A pinhole view of the panorama centred on (lon, lat), CROP_PX square."""
    lo, la = math.radians(lon), math.radians(lat)
    f = np.array([math.cos(la) * math.cos(lo), math.cos(la) * math.sin(lo), math.sin(la)])
    up = np.array([0.0, 0.0, 1.0]) if abs(lat) < 85 else np.array([math.cos(lo), math.sin(lo), 0.0])
    r = np.cross(f, up)
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    k = math.tan(math.radians(fov_deg) / 2)
    g = np.linspace(-k, k, CROP_PX)
    gu, gv = np.meshgrid(g, -g)
    d = f[None, None] + gu[..., None] * r + gv[..., None] * u
    d /= np.linalg.norm(d, axis=2, keepdims=True)
    lons = np.degrees(np.arctan2(d[..., 1], d[..., 0]))
    lats = np.degrees(np.arcsin(np.clip(d[..., 2], -1, 1)))
    h, w = im.shape[:2]
    mx = ((lons + 180.0) / 360.0 * w).astype(np.float32)
    my = ((90.0 - lats) / 180.0 * h).astype(np.float32)
    return cv2.remap(im, mx, my, cv2.INTER_AREA, borderMode=cv2.BORDER_WRAP)


def clip_features(name: str) -> dict:
    """Per branch: the averaged CLIP embedding of flat crops along it, and zero-shot scores. Cached."""
    f = CACHE / f"{name[:-4]}.clip.npz"
    if f.exists():
        return dict(np.load(f))
    import torch
    from PIL import Image

    model, preprocess, text = _load_clip()
    d = photo_features(name)
    lab = d["lab"]
    w_c, h_c = lab.shape[1], lab.shape[0]
    dt = cv2.distanceTransform((d["mask"] > 0).astype(np.uint8), cv2.DIST_L2, 5)
    im = cv2.imread(str(ROOT / "data" / "lab" / RUNS[name] / "image.jpg"))
    ys, xs = np.nonzero(lab)
    bid = lab[ys, xs]
    order = np.argsort(bid, kind="stable")
    ys, xs, bid = ys[order], xs[order], bid[order]
    ids, starts, counts = np.unique(bid, return_index=True, return_counts=True)
    assert np.array_equal(ids, d["ids"])
    deg_px = 360.0 / w_c
    embs = []
    for s0, c in zip(starts, counts):
        py, px = ys[s0:s0 + c].astype(float), xs[s0:s0 + c].astype(float)
        # order the branch's pixels along its main direction
        pts = np.stack([px, py], 1)
        ctr = pts.mean(0)
        _, _, vt = np.linalg.svd(pts - ctr, full_matrices=False)
        along = (pts - ctr) @ vt[0]
        o = np.argsort(along)
        hw_deg = float(np.median(dt[py.astype(int), px.astype(int)])) * deg_px
        fov = float(np.clip(14 * hw_deg, 8.0, 40.0))
        crops = []
        for q in CROP_POINTS:
            i = o[min(int(q * c), c - 1)]
            lon = (px[i] + 0.5) * deg_px - 180.0
            lat = 90.0 - (py[i] + 0.5) * deg_px
            crop = flat_crop(im, lon, lat, fov)
            crops.append(preprocess(Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))))
        with torch.no_grad():
            e = model.encode_image(torch.stack(crops).cuda()).float()
            e = torch.nn.functional.normalize(e, dim=1).mean(0)
            embs.append(torch.nn.functional.normalize(e, dim=0).cpu().numpy())
    E = np.array(embs, dtype=np.float32)
    T_ = np.stack([text[k].cpu().numpy() for k in PROMPTS])
    T_ /= np.linalg.norm(T_, axis=1, keepdims=True)
    logits = 100.0 * E @ T_.T
    p = np.exp(logits - logits.max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)
    np.savez(f, emb=E, zero_shot_pipe=p[:, list(PROMPTS).index("pipe")])
    return dict(np.load(f))


# ---------------------------------------------------------------------------
# the colour rule: paint is a wide, flat, strongly red band
# ---------------------------------------------------------------------------

# Fixed from two profiles across photo 003 before any scoring: a paint stripe
# is a plateau of a* 35-43 about 50-75 px across; the thin pipe beside it
# peaks at a* 20 over about 12 px.
PAINT_A = 28.0        # Lab a*: redder than this is paint-red
PAINT_MIN_MM = 60.0   # and a band at least this wide (at the pipe's height) is paint, not pipe
PAINT_PAD = 2         # canvas px the paint region is grown by before it is cut out
# Added after photo 027 (not traced): a fire main passing close overhead is as
# wide and as red as a stripe, and the first rule cut it. Across the main the
# brightness swings 10-14 L (a lit cylinder); across a stripe it stays within
# 1-2 L. Paint must also be flat.
PAINT_FLAT_L = 3.0    # local brightness spread (std of L over red pixels, 7x7 canvas px) at most this
FLAT_WIN = 7


def lab_small(name: str) -> np.ndarray:
    """The photo at canvas scale in Lab (float16), cached."""
    f = CACHE / f"{name[:-4]}.lab.npy"
    if f.exists():
        return np.load(f)
    im = cv2.imread(str(ROOT / "data" / "lab" / RUNS[name] / "image.jpg"))
    small = cv2.resize(im, (im.shape[1] // SCALE, im.shape[0] // SCALE), interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(small.astype(np.float32) / 255.0, cv2.COLOR_BGR2Lab).astype(np.float16)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.save(f, lab)
    return lab


def paint_mask(name: str, a_min: float = PAINT_A, min_mm: float = PAINT_MIN_MM,
               flat_l: float | None = PAINT_FLAT_L) -> np.ndarray:
    """Ceiling pixels of the 1/4 canvas inside a wide, strongly red band of flat brightness."""
    lab = lab_small(name).astype(np.float32)
    L, a = lab[..., 0], lab[..., 1]
    h_c, w_c = a.shape
    red = (a >= a_min).astype(np.uint8)
    red = cv2.morphologyEx(red, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    if flat_l is not None:
        # brightness spread over the red pixels of a small window only, so the
        # band's own edge against the grey concrete does not count as shading
        r = red.astype(np.float32)
        k = (FLAT_WIN, FLAT_WIN)
        n = cv2.boxFilter(r, -1, k, normalize=False)
        s1 = cv2.boxFilter(L * r, -1, k, normalize=False)
        s2 = cv2.boxFilter(L * L * r, -1, k, normalize=False)
        std = np.sqrt(np.maximum(s2 / np.maximum(n, 1) - (s1 / np.maximum(n, 1)) ** 2, 0))
        red = red & (std <= flat_l).astype(np.uint8)
    dt = cv2.distanceTransform(red, cv2.DIST_L2, 3)  # half-width in canvas px
    lat = 90.0 - (np.arange(h_c) + 0.5) / h_c * 180.0
    deg_px = 360.0 / w_c
    rng = GAP_M / np.sin(np.radians(np.maximum(lat, 1.0)))
    half_min_px = (min_mm / 1000.0 / 2) / rng / np.radians(deg_px)  # per row
    core = (dt >= half_min_px[:, None]).astype(np.uint8)
    # the whole band around each core: grow the core back to the band's edge, inside the red
    paint = core.copy()
    for _ in range(int(np.ceil(half_min_px[lat > MIN_LAT_DEG].max())) + 2):
        grown = cv2.dilate(paint, np.ones((3, 3), np.uint8)) & red
        if np.array_equal(grown, paint):
            break
        paint = grown
    paint = cv2.dilate(paint, np.ones((2 * PAINT_PAD + 1, 2 * PAINT_PAD + 1), np.uint8))
    paint[int((90.0 - MIN_LAT_DEG) / 180.0 * h_c):] = 0
    return paint > 0


def score_colour() -> None:
    EXPERIMENT.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in traced():
        d = photo_features(name)
        traces = json.loads((TRACES / f"traces_{name[-7:-4]}.json").read_text(encoding="utf-8"))
        t_all = st.ceiling_only(st.trace_canvas(traces, traces["width"], traces["height"], None), traces["height"])
        row = {"photo": name[-7:-4], "none": score_mask(name, d["mask"])}
        for a_min in (24.0, 28.0, 32.0):
            for mm in (40.0, 60.0, 80.0):
                p = paint_mask(name, a_min, mm)
                kept = np.where(p, 0, d["mask"]).astype(np.uint8)
                key = f"a{a_min:.0f}_w{mm:.0f}"
                row[key] = score_mask(name, kept)
                row[key]["trace_in_paint"] = float(((t_all > 0) & p).sum() / max((t_all > 0).sum(), 1))
                if a_min == PAINT_A and mm == PAINT_MIN_MM:
                    overlay(name, d["mask"], ~p, EXPERIMENT / f"colour_{name[-7:-4]}.png")
        rows.append(row)
        s = row[f"a{PAINT_A:.0f}_w{PAINT_MIN_MM:.0f}"]
        print(f"{row['photo']}  recall(sure) {row['none']['recall_sure']:.2f} -> {s['recall_sure']:.2f}   "
              f"precision {row['none']['precision']:.2f} -> {s['precision']:.2f}   traced pipe called paint {s['trace_in_paint']:.1%}")
    keys = [k for k in rows[0] if k not in ("photo",)]
    mean = {k: {m: float(np.nanmean([r[k][m] for r in rows])) for m in ("recall_sure", "precision")} for k in keys}
    print("\nmean over the four photos (a* threshold, minimum width mm):")
    for k in keys:
        mark = "  <- fixed in advance" if k == f"a{PAINT_A:.0f}_w{PAINT_MIN_MM:.0f}" else ""
        print(f"  {k:10s} recall(sure) {mean[k]['recall_sure']:.2f}   precision {mean[k]['precision']:.2f}{mark}")
    (EXPERIMENT / "scores_colour.json").write_text(json.dumps({"rows": rows, "mean": mean}, indent=1), encoding="utf-8")


# ---------------------------------------------------------------------------
# the unit the layout uses: straight pieces of outline
# ---------------------------------------------------------------------------

CONTEXT_M = 0.8   # a piece's crop shows about this much ceiling across, at the pipe's height


def _layout():
    import pipe_layout as pl

    return pl


def _densify(v: np.ndarray, step_deg: float = 0.1) -> np.ndarray:
    out = [v[:1]]
    for a, b in zip(v[:-1], v[1:]):
        ang = math.degrees(math.acos(float(np.clip(a @ b, -1, 1))))
        n = max(int(ang / step_deg), 1)
        s = np.linspace(0, 1, n + 1)[1:, None]
        p = a[None] * (1 - s) + b[None] * s
        out.append(p / np.linalg.norm(p, axis=1, keepdims=True))
    return np.concatenate(out)


def _lonlat(v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of pipe_layout.unit with north 0: lon = -heading."""
    return -np.degrees(np.arctan2(v[:, 1], v[:, 0])), np.degrees(np.arcsin(np.clip(v[:, 2], -1, 1)))


def photo_pieces(name: str) -> list[dict]:
    """The fire pipe pieces pipe_layout would use, in its order, with lon/lat and plan metres."""
    pl = _layout()
    result = json.loads((ROOT / "data" / "lab" / RUNS[name] / "result.json").read_text(encoding="utf-8"))
    out = []
    for i, p in enumerate(pl.photo_pieces(result, 0.0)):
        if p["system"] != "fire pipe":
            continue
        placed = pl.piece_on_plan(p["v"], GAP_M)
        metres = 0.0 if placed is None else float(placed[3] - placed[2])
        lon, lat = _lonlat(_densify(p["v"]))
        out.append({"index": i, "lon": lon, "lat": lat, "metres": metres})
    return out


def piece_labels(name: str, pieces: list[dict]) -> np.ndarray:
    traces = json.loads((TRACES / f"traces_{name[-7:-4]}.json").read_text(encoding="utf-8"))
    w, h = traces["width"], traces["height"]
    t = st.ceiling_only(st.trace_canvas(traces, w, h, None), h)
    near = cv2.dilate(t, np.ones((2 * TOL_C + 1, 2 * TOL_C + 1), np.uint8)) > 0
    fr = []
    for p in pieces:
        x = (((p["lon"] + 180.0) / 360.0 * w) // SCALE).astype(int) % near.shape[1]
        y = np.clip(((90.0 - p["lat"]) / 180.0 * h) // SCALE, 0, near.shape[0] - 1).astype(int)
        fr.append(near[y, x].mean())
    return np.array(fr)


def piece_clip(name: str) -> dict:
    """Averaged CLIP embedding of flat crops along each piece, and zero-shot pipe scores. Cached."""
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / f"{name[:-4]}.pieces.npz"
    if f.exists():
        return dict(np.load(f))
    import torch
    from PIL import Image

    model, preprocess, text = _load_clip()
    pieces = photo_pieces(name)
    im = cv2.imread(str(ROOT / "data" / "lab" / RUNS[name] / "image.jpg"))
    embs = []
    for p in pieces:
        crops = []
        for q in CROP_POINTS:
            k = min(int(q * len(p["lon"])), len(p["lon"]) - 1)
            lat = float(p["lat"][k])
            rng = GAP_M / math.sin(math.radians(max(lat, 1.0)))
            fov = float(np.clip(2 * math.degrees(math.atan(CONTEXT_M / 2 / rng)), 6.0, 40.0))
            crop = flat_crop(im, float(p["lon"][k]), lat, fov)
            crops.append(preprocess(Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))))
        with torch.no_grad():
            e = model.encode_image(torch.stack(crops).cuda()).float()
            e = torch.nn.functional.normalize(torch.nn.functional.normalize(e, dim=1).mean(0), dim=0)
        embs.append(e.cpu().numpy())
    E = np.array(embs, dtype=np.float32).reshape(len(pieces), -1)
    T_ = np.stack([text[k].cpu().numpy() for k in PROMPTS])
    T_ /= np.linalg.norm(T_, axis=1, keepdims=True)
    logits = 100.0 * E @ T_.T if len(E) else np.zeros((0, len(PROMPTS)))
    pr = np.exp(logits - logits.max(1, keepdims=True)) if len(E) else logits
    pr = pr / pr.sum(1, keepdims=True) if len(E) else pr
    np.savez(f, emb=E, zero_shot_pipe=pr[:, list(PROMPTS).index("pipe")] if len(E) else np.zeros(0),
             index=np.array([p["index"] for p in pieces]), metres=np.array([p["metres"] for p in pieces]))
    return dict(np.load(f))


def piece_score(name: str, pieces: list[dict], keep: np.ndarray) -> dict:
    """Share of kept plan-metres near a trace, and share of the sure traced pipe near a kept piece."""
    traces = json.loads((TRACES / f"traces_{name[-7:-4]}.json").read_text(encoding="utf-8"))
    w, h = traces["width"], traces["height"]
    fr = piece_labels(name, pieces)
    m = np.array([p["metres"] for p in pieces])
    canvas = np.zeros((h // SCALE, w // SCALE), np.uint8)
    for p, k in zip(pieces, keep):
        if not k:
            continue
        x = ((p["lon"] + 180.0) / 360.0 * w) / SCALE
        y = ((90.0 - p["lat"]) / 180.0 * h) / SCALE
        for a in np.split(np.arange(len(x)), np.flatnonzero(np.abs(np.diff(x)) > canvas.shape[1] / 2) + 1):
            if len(a) >= 2:
                cv2.polylines(canvas, [np.stack([x[a], y[a]], 1).astype(np.int32)], False, 255, 2)
    t_sure = st.ceiling_only(st.trace_canvas(traces, w, h, True), h)
    return {"precision_m": float((m * fr * keep).sum() / max((m * keep).sum(), 1e-9)),
            "recall_sure": st.near(t_sure, canvas), "metres": float((m * keep).sum())}


def piece_training(names: list[str]):
    Es, ys, ws = [], [], []
    for name in names:
        pieces = photo_pieces(name)
        c = piece_clip(name)
        fr = piece_labels(name, pieces)
        sure = ((fr >= PIPE_FRAC) | (fr <= PAINT_FRAC)) & (c["metres"] > 0)
        Es.append(c["emb"][sure])
        ys.append((fr[sure] >= PIPE_FRAC).astype(float))
        ws.append(c["metres"][sure])
    E, y, w = np.concatenate(Es), np.concatenate(ys), np.concatenate(ws)
    w = np.where(y == 1, w / w[y == 1].sum(), w / w[y == 0].sum())
    return E, y, w


def fit_probe(E: np.ndarray, y: np.ndarray, w: np.ndarray, l2: float = 1e-2, steps: int = 400):
    """Weighted logistic regression on embeddings; returns (weights, bias)."""
    import torch

    X = torch.tensor(E).cuda()
    Y = torch.tensor(y, dtype=torch.float32).cuda()
    W = torch.tensor(w / w.sum(), dtype=torch.float32).cuda()
    a = torch.zeros(E.shape[1], device="cuda", requires_grad=True)
    b = torch.zeros(1, device="cuda", requires_grad=True)
    opt = torch.optim.LBFGS([a, b], max_iter=steps, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        z = X @ a * 10 + b
        loss = (W * torch.nn.functional.binary_cross_entropy_with_logits(z, Y, reduction="none")).sum() + l2 * (a * a).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return a.detach().cpu().numpy(), float(b.detach().cpu())


def probe_predict(model, E: np.ndarray) -> np.ndarray:
    a, b = model
    return 1 / (1 + np.exp(-(E @ a * 10 + b)))


# ---------------------------------------------------------------------------
# a depth-2 tree, by hand (numpy only)
# ---------------------------------------------------------------------------

def _gini(y: np.ndarray, w: np.ndarray) -> float:
    W = w.sum()
    if W <= 0:
        return 0.0
    p = (w * y).sum() / W
    return 2 * p * (1 - p)


def _split(X, y, w, min_w):
    best = (_gini(y, w), None, None)
    W = w.sum()
    for j in range(X.shape[1]):
        for q in np.unique(np.quantile(X[:, j], np.linspace(0.02, 0.98, 49))):
            left = X[:, j] <= q
            wl, wr = w[left].sum(), w[~left].sum()
            if wl < min_w or wr < min_w:
                continue
            g = (wl * _gini(y[left], w[left]) + wr * _gini(y[~left], w[~left])) / W
            if g < best[0] - 1e-9:
                best = (g, j, float(q))
    return best[1], best[2]


def fit_tree(X: np.ndarray, y: np.ndarray, w: np.ndarray, depth: int = 2) -> dict:
    min_w = 0.05 * w.sum()

    def grow(idx, d):
        p = float((w[idx] * y[idx]).sum() / max(w[idx].sum(), 1e-12))
        if d == depth:
            return {"p": p}
        j, q = _split(X[idx], y[idx], w[idx], min_w)
        if j is None:
            return {"p": p}
        left = idx[X[idx, j] <= q]
        right = idx[X[idx, j] > q]
        return {"feature": FEATURES[j], "j": j, "q": q, "le": grow(left, d + 1), "gt": grow(right, d + 1), "p": p}

    return grow(np.arange(len(y)), 0)


def predict(tree: dict, X: np.ndarray) -> np.ndarray:
    out = np.zeros(len(X))
    for i, x in enumerate(X):
        node = tree
        while "j" in node:
            node = node["le"] if x[node["j"]] <= node["q"] else node["gt"]
        out[i] = node["p"]
    return out


def describe(tree: dict, indent: str = "") -> str:
    if "j" not in tree:
        return f"{indent}-> {'keep (pipe)' if tree['p'] > 0.5 else 'drop (not pipe)'}  [pipe share {tree['p']:.2f}]\n"
    return (f"{indent}if {tree['feature']} <= {tree['q']:.3g}:\n" + describe(tree["le"], indent + "    ")
            + f"{indent}else:\n" + describe(tree["gt"], indent + "    "))


def training_set(names: list[str], use_clip: bool = False):
    Xs, ys, ws = [], [], []
    for name in names:
        d = photo_features(name)
        frac = labels(name, d)
        sure = (frac >= PIPE_FRAC) | (frac <= PAINT_FRAC)
        Xs.append((clip_features(name)["emb"] if use_clip else matrix(d))[sure])
        ys.append((frac[sure] >= PIPE_FRAC).astype(float))
        ws.append(d["length"][sure].astype(float))
    X, y, w = np.concatenate(Xs), np.concatenate(ys), np.concatenate(ws)
    # balanced: pipe and not-pipe weigh the same in total, each by its length
    w = np.where(y == 1, w / w[y == 1].sum(), w / w[y == 0].sum())
    return X, y, w


# ---------------------------------------------------------------------------
# what is left of the mask, and its score
# ---------------------------------------------------------------------------

def kept_zone(data: dict, keep_branch: np.ndarray) -> np.ndarray:
    """Ceiling pixels that go with a kept branch (each pixel goes with its nearest branch)."""
    lab = data["lab"]
    src = np.where(lab > 0, 0, 255).astype(np.uint8)
    _, near = cv2.distanceTransformWithLabels(src, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
    lut = np.zeros(near.max() + 1, bool)
    lut_b = np.zeros(lab.max() + 1, bool)
    lut_b[data["ids"][keep_branch]] = True
    ys, xs = np.nonzero(lab)
    lut[near[ys, xs]] = lut_b[lab[ys, xs]]
    return lut[near]


def score_mask(name: str, mask: np.ndarray) -> dict:
    traces = json.loads((TRACES / f"traces_{name[-7:-4]}.json").read_text(encoding="utf-8"))
    w, h = traces["width"], traces["height"]
    mask = st.ceiling_only(mask, h)
    t_sure = st.ceiling_only(st.trace_canvas(traces, w, h, True), h)
    t_all = st.ceiling_only(st.trace_canvas(traces, w, h, None), h)
    edge = cv2.Canny(mask, 50, 150)
    return {"recall_sure": st.near(t_sure, mask), "recall_all": st.near(t_all, mask), "precision": st.near(edge, t_all),
            "mask_px": int((mask > 0).sum())}


def overlay(name: str, mask: np.ndarray, kept: np.ndarray, path: Path) -> None:
    traces = json.loads((TRACES / f"traces_{name[-7:-4]}.json").read_text(encoding="utf-8"))
    w, h = traces["width"], traces["height"]
    t_all = st.ceiling_only(st.trace_canvas(traces, w, h, None), h)
    cut = int((90.0 - MIN_LAT_DEG) / 180.0 * h / SCALE)
    pic = np.zeros((*mask.shape, 3), np.uint8)
    pic[(mask > 0) & ~kept] = (200, 90, 0)      # dropped: blue
    pic[(mask > 0) & kept] = (0, 0, 255)         # kept: red
    pic[cv2.dilate(t_all, np.ones((3, 3), np.uint8)) > 0] = (0, 255, 0)  # trace: green
    cv2.imwrite(str(path), pic[:cut])


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------

def traced() -> list[str]:
    return [json.loads(f.read_text(encoding="utf-8"))["photo"] for f in sorted(TRACES.glob("traces_*.json"))]


def features() -> None:
    for name in sorted(RUNS):
        d = photo_features(name)
        print(f"{name[-7:-4]}  {len(d['ids']):4d} branches")


METHODS = ("none", "rule", "words", "probe")
METHOD_NAMES = {
    "none": "no filter (experiment 11)",
    "rule": "hand-made measures, depth-2 tree",
    "words": "CLIP, words only (zero-shot)",
    "probe": "CLIP, small classifier trained on traces",
}


def decide(method: str, held: str, train: list[str], threshold: float = 0.5) -> np.ndarray:
    """Which branches of the held-out photo to keep, by a method fitted on the train photos only."""
    d = photo_features(held)
    if method == "none":
        return np.ones(len(d["ids"]), bool)
    if method == "rule":
        X, y, w = training_set(train)
        return predict(fit_tree(X, y, w), matrix(d)) > threshold
    c = clip_features(held)
    if method == "words":
        return c["zero_shot_pipe"] > threshold
    X, y, w = training_set(train, use_clip=True)
    return probe_predict(fit_probe(X, y, w), c["emb"]) > threshold


def score() -> None:
    EXPERIMENT.mkdir(parents=True, exist_ok=True)
    names = traced()
    rows = []
    for held in names:
        train = [n for n in names if n != held]
        d = photo_features(held)
        row = {"photo": held[-7:-4]}
        for method in METHODS:
            zone = kept_zone(d, decide(method, held, train))
            row[method] = score_mask(held, np.where(zone, d["mask"], 0).astype(np.uint8))
            if method != "none":
                overlay(held, d["mask"], zone, EXPERIMENT / f"{method}_{held[-7:-4]}.png")
        rows.append(row)
        print(row["photo"], "  ".join(f"{m}: R {row[m]['recall_sure']:.2f} P {row[m]['precision']:.2f}" for m in METHODS))
    mean = {m: {k: float(np.nanmean([r[m][k] for r in rows])) for k in ("recall_sure", "recall_all", "precision")} for m in METHODS}
    print("\nmean, each photo scored by a filter fitted on the other three:")
    for m in METHODS:
        print(f"  {METHOD_NAMES[m]:42s} recall(sure) {mean[m]['recall_sure']:.2f}   precision {mean[m]['precision']:.2f}")
    X, y, w = training_set(names)
    rule = fit_tree(X, y, w)
    print("\nrule fitted on all four photos:\n" + describe(rule))
    (EXPERIMENT / "scores.json").write_text(json.dumps({"rows": rows, "mean": mean, "methods": METHOD_NAMES,
                                                         "rule": rule, "rule_text": describe(rule)}, indent=1), encoding="utf-8")


def score_pieces() -> None:
    """The same comparison on the layout's own unit, weighted by plan metres."""
    EXPERIMENT.mkdir(parents=True, exist_ok=True)
    names = traced()
    rows = []
    for held in names:
        train = [n for n in names if n != held]
        pieces = photo_pieces(held)
        c = piece_clip(held)
        fr = piece_labels(held, pieces)
        probe = fit_probe(*piece_training(train))
        p_probe = probe_predict(probe, c["emb"])
        keeps = {"none": np.ones(len(pieces), bool), "words": c["zero_shot_pipe"] > 0.5,
                 "probe": p_probe > 0.5, "oracle": fr >= 0.5}
        row = {"photo": held[-7:-4], "pieces": len(pieces)}
        for k, keep in keeps.items():
            row[k] = piece_score(held, pieces, keep)
        rows.append(row)
        draw_pieces(held, pieces, keeps["probe"], EXPERIMENT / f"pieces_{held[-7:-4]}.png")
        print(row["photo"], f"{len(pieces):3d} pieces  " + "  ".join(
            f"{k}: P {row[k]['precision_m']:.2f} R {row[k]['recall_sure']:.2f} {row[k]['metres']:5.1f} m" for k in keeps))
    mean = {k: {m: float(np.nanmean([r[k][m] for r in rows])) for m in ("precision_m", "recall_sure", "metres")}
            for k in ("none", "words", "probe", "oracle")}
    print("\nmean (each photo by a classifier fitted on the other three):")
    for k, v in mean.items():
        print(f"  {k:7s} precision by metres {v['precision_m']:.2f}   recall(sure) {v['recall_sure']:.2f}   metres per photo {v['metres']:.1f}")
    (EXPERIMENT / "scores_pieces.json").write_text(json.dumps({"rows": rows, "mean": mean}, indent=1), encoding="utf-8")


def draw_pieces(name: str, pieces: list[dict], keep: np.ndarray, path: Path) -> None:
    traces = json.loads((TRACES / f"traces_{name[-7:-4]}.json").read_text(encoding="utf-8"))
    w, h = traces["width"], traces["height"]
    cut = int((90.0 - MIN_LAT_DEG) / 180.0 * h / SCALE)
    pic = np.zeros((h // SCALE, w // SCALE, 3), np.uint8)
    t_all = st.trace_canvas(traces, w, h, None)
    pic[cv2.dilate(t_all, np.ones((3, 3), np.uint8)) > 0] = (0, 160, 0)
    for p, k in zip(pieces, keep):
        x = ((p["lon"] + 180.0) / 360.0 * w) / SCALE
        y = ((90.0 - p["lat"]) / 180.0 * h) / SCALE
        for a in np.split(np.arange(len(x)), np.flatnonzero(np.abs(np.diff(x)) > pic.shape[1] / 2) + 1):
            if len(a) >= 2:
                cv2.polylines(pic, [np.stack([x[a], y[a]], 1).astype(np.int32)], False, (0, 0, 255) if k else (255, 120, 0), 2)
    cv2.imwrite(str(path), pic[:cut])


def _polys(canvas: np.ndarray, w: int, h: int) -> list[list[list[float]]]:
    cs, _ = cv2.findContours(canvas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cs:
        c = cv2.approxPolyDP(c, 0.5, True)[:, 0, :]
        if len(c) < 3:
            continue
        out.append([[round((x + 0.5) * SCALE / w * 360.0 - 180.0, 4), round(90.0 - (y + 0.5) * SCALE / h * 180.0, 4)] for x, y in c])
    return out


def apply() -> None:
    """Results with the paint cut out by the colour rule (and, for a fair baseline,
    unfiltered but re-drawn through the same canvas) for every photo of the walk."""
    for variant in ("filtered", "unfiltered"):
        (EXPERIMENT / variant / "results").mkdir(parents=True, exist_ok=True)
    report = {}
    for name in sorted(RUNS):
        result = json.loads((ROOT / "data" / "lab" / RUNS[name] / "result.json").read_text(encoding="utf-8"))
        w, h = result["width"], result["height"]
        zone = ~paint_mask(name)
        conf = max((float(f["confidence"]) for f in result["detections"] if f["label"] == "fire pipe"), default=0.0)
        whole = raster(result, w, h, ("outlines", "grown_outlines")) > 0
        report[name] = {"mask_px": int(whole.sum()), "cut_px": int((whole & ~zone).sum())}
        for variant, z in (("filtered", zone), ("unfiltered", np.ones_like(zone))):
            find = {"label": "fire pipe", "confidence": conf}
            for key in ("outlines", "grown_outlines"):
                c = raster(result, w, h, (key,))
                find[key] = _polys(np.where(z, c, 0).astype(np.uint8), w, h)
            res = dict(result)
            res["detections"] = [f for f in result["detections"] if f["label"] != "fire pipe"] + [find]
            (EXPERIMENT / variant / "results" / f"{name[:-4]}.json").write_text(json.dumps(res), encoding="utf-8")
        print(f"{name[-7:-4]}  paint cut {report[name]['cut_px'] / max(report[name]['mask_px'], 1):.0%} of the fire pipe mask")
    (EXPERIMENT / "apply.json").write_text(json.dumps(report, indent=1), encoding="utf-8")


if __name__ == "__main__":
    {"features": features, "score": score, "pieces": score_pieces, "colour": score_colour, "apply": apply}[sys.argv[1]]()
