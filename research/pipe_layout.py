"""Experiment 10, steps 2 to 6: from mask outlines on the sphere to pipe lines on the plan.

    python research/pipe_layout.py pins        # check the walk's pins and the plan's scale
    python research/pipe_layout.py pieces      # straight pieces per photo + heading check
    python research/pipe_layout.py gap         # the camera-to-pipe height gap, per system
    python research/pipe_layout.py layout      # the merged layout, lengths, figure

HOW A PIPE IN A 360 PHOTO BECOMES A LINE ON THE PLAN

A straight horizontal pipe seen from one point lies on a great circle of the
sphere: all its directions are perpendicular to one normal n. The pipe's
direction on the plan is n x z (horizontal, in the circle's plane), which needs
no distance at all. Its sideways offset does: the circle's highest point, at
latitude L, is the pipe's closest point, at horizontal distance gap / tan L
where gap is how far the pipe is above the camera. Every offset scales with
the gap, so the gap is fitted afterwards (`gap`) from pipes seen twice.

Outlines are used as they come -- the two long sides of a thin pipe are both
within half a degree of its midline, closer than anything else here -- and
cut into straight pieces by splitting at the point farthest from the fitted
circle until every piece fits to TOL_DEG. Pieces below MIN_LAT_DEG are dropped:
near the horizon the distance is unusable and the floor begins.

Frames: metric heading = north offset - image longitude, counter-clockwise from
+X, y up (docs/coordinates.md). Pieces and lines are in metres, level-local.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "research" / "experiments" / "2026-10-01-10-whole-walk-detection"
# `--from DIR` reads each photo's result from DIR/results/<photo>.json instead of
# the lab's store, and writes pieces, layout and figure to DIR (experiment 12).
OUT = EXPERIMENT
RESULTS: Path | None = None
# The floor plan (PDF) the layout is drawn on. Not included: it is a client drawing.
PLAN_PDF = Path(__import__("os").environ.get("PLAN_PDF", "plan.pdf"))

SYSTEMS = ("fire pipe", "electrical pipe")  # what is on this ceiling now; duct is not installed
TOL_DEG = 1.0        # a piece is straight when every vertex is this close to its circle
MIN_LAT_DEG = 8.0    # the ceiling only: below this a pipe is too far to place, or it is a wall
MIN_SPAN_DEG = 10.0  # shorter pieces are corners, drops and noise
VERTICAL_RATIO = 0.3 # azimuth spread below this share of the span means a vertical run, not a pipe
MAX_RANGE_M = 7.0    # a piece is cut where it would land farther than this
SNAP_DEG = 12.0      # a line this close to the building grid is snapped onto it; others are not pipe runs
JOIN_GAP_M = 2.5     # collinear stretches closer than this along the line become one
MIN_SINGLE_M = 4.0   # a line seen from one photo only must be at least this long to be kept
FOLD_OFFSET_M = 1.0  # second pass: parallel copies of one run, closer than this, become one line
GAP_GRID_M = np.arange(0.4, 2.21, 0.1)
MERGE_ANGLE_DEG = 6.0
MERGE_OFFSET_M = 0.45


# ---------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------

def pins() -> None:
    """Check the pins file. In the project it was exported from the capture
    platform (each photo's position on the plan and its north offset, set by
    hand); that export and the platform are not part of this repository.

    Expected shape of pins.json:
        {"plan": {"px_per_meter", "origin_px_x", "origin_px_y", "rotation_deg",
                  "width_px", "height_px"},
         "photos": {"<file name>": {"seq", "x", "y", "north"}}}
    """
    out = json.loads((EXPERIMENT / "pins.json").read_text(encoding="utf-8"))
    print(len(out["photos"]), "pins;", out["plan"])


def load_pins() -> dict:
    return json.loads((EXPERIMENT / "pins.json").read_text(encoding="utf-8"))


def load_runs() -> dict[str, dict]:
    runs = json.loads((EXPERIMENT / "runs.json").read_text(encoding="utf-8"))
    out = {}
    for name, rid in runs.items():
        f = (RESULTS / f"{name[:-4]}.json") if RESULTS else ROOT / "data" / "lab" / rid / "result.json"
        if f.exists():
            out[name] = json.loads(f.read_text(encoding="utf-8"))
    return out


# ---------------------------------------------------------------------------
# sphere geometry
# ---------------------------------------------------------------------------

def unit(lon_deg: np.ndarray, lat_deg: np.ndarray, north_deg: float) -> np.ndarray:
    """Directions in the metric frame: heading = north - lon, CCW from +X."""
    h = np.radians(north_deg - lon_deg)
    la = np.radians(lat_deg)
    return np.stack([np.cos(h) * np.cos(la), np.sin(h) * np.cos(la), np.sin(la)], axis=1)


def fit_circle(v: np.ndarray) -> tuple[np.ndarray, float]:
    """Normal of the best great circle through unit vectors v, and the worst residual in degrees."""
    _, _, vt = np.linalg.svd(v, full_matrices=False)
    n = vt[-1]
    resid = np.degrees(np.abs(np.arcsin(np.clip(v @ n, -1, 1))))
    return n, float(resid.max())


def split_straight(v: np.ndarray, tol: float = TOL_DEG) -> list[np.ndarray]:
    """Cut a run of directions into pieces that each lie on one great circle."""
    if len(v) < 3:
        return [v]
    n, worst = fit_circle(v)
    if worst <= tol:
        return [v]
    resid = np.abs(np.arcsin(np.clip(v @ n, -1, 1)))
    k = int(resid.argmax())
    if k == 0 or k == len(v) - 1:
        k = len(v) // 2
    return split_straight(v[: k + 1], tol) + split_straight(v[k:], tol)


def piece_on_plan(v: np.ndarray, gap_m: float):
    """A straight piece as a line on the plan: direction, closest-point offset, endpoints.

    Returns (d, p0, t0, t1): unit direction d, closest point p0 relative to the
    camera, and the endpoints as distances along d from p0; or None when the
    piece is not usable at this gap.
    """
    n, _ = fit_circle(v)
    if n[2] < 0:
        n = -n
    horiz = math.hypot(n[0], n[1])
    if horiz < 1e-6:          # the circle is the horizon itself
        return None
    d = np.array([-n[1], n[0]]) / horiz            # n x z, normalised
    top_lat = 90.0 - math.degrees(math.acos(abs(n[2])))  # the circle's highest latitude
    if top_lat < MIN_LAT_DEG:
        return None
    r0 = gap_m / math.tan(math.radians(top_lat))
    az = np.array([-n[0], -n[1]]) / horiz          # horizontal direction of the circle's top point
    # (the top point is z - n_z n, whose horizontal part is -n_z (n_x, n_y))
    p0 = r0 * az
    # endpoints: each direction v lands at horizontal range gap / tan(lat), along its own azimuth
    ts = []
    for e in (v[0], v[-1]):
        lat = math.degrees(math.asin(max(-1.0, min(1.0, e[2]))))
        # where this end of the mask lands; a far end is cut at the range limit
        r = min(MAX_RANGE_M, gap_m / math.tan(math.radians(max(lat, MIN_LAT_DEG))))
        hz = e[:2] / max(1e-9, np.linalg.norm(e[:2]))
        ts.append(float((r * hz - p0) @ d))
    return d, p0, min(ts), max(ts)


# ---------------------------------------------------------------------------
# step 2 and 3: pieces per photo, and the heading check
# ---------------------------------------------------------------------------

def _outlines(find: dict) -> list[list[list[float]]]:
    o = find.get("outlines") or find.get("outline") or []
    if isinstance(o, str):
        o = json.loads(o)
    if o and not isinstance(o[0][0], list):
        o = [o]
    return o


def photo_pieces(result: dict, north_deg: float) -> list[dict]:
    """Straight pieces of every outline, as unit-vector runs with their system."""
    pieces = []
    for find in result["detections"]:
        if find["label"] not in SYSTEMS:
            continue
        for poly in _outlines(find):
            arr = np.array(poly, dtype=float)
            keep = arr[:, 1] >= MIN_LAT_DEG
            # runs of consecutive kept vertices, so a loop that dips below the
            # horizon is cut there rather than joined across the gap
            idx = np.flatnonzero(keep)
            if len(idx) < 3:
                continue
            breaks = np.flatnonzero(np.diff(idx) > 1)
            for run in np.split(idx, breaks + 1):
                if len(run) < 3:
                    continue
                v = unit(arr[run, 0], arr[run, 1], north_deg)
                # the seam: consecutive vertices more than 180 deg apart in lon are
                # still neighbours on the sphere, and unit() has already handled that
                for part in split_straight(v):
                    span = math.degrees(math.acos(float(np.clip(part[0] @ part[-1], -1, 1))))
                    if span < MIN_SPAN_DEG:
                        continue
                    # A vertical run (a drop, a riser, a hanging coil) keeps one
                    # azimuth while its latitude changes; a horizontal pipe sweeps
                    # azimuth as it goes. Only the second is a line on the plan.
                    az = np.degrees(np.arctan2(part[:, 1], part[:, 0]))
                    az_spread = float(np.ptp((az - az[0] + 180) % 360 - 180))
                    if az_spread >= VERTICAL_RATIO * span:
                        pieces.append({"system": find["label"], "v": part, "span": span, "conf": find["confidence"]})
    return pieces


def grid_offset(pieces: list[dict]) -> tuple[float, int]:
    """How far the pieces' directions sit from the building grid, mod 90, by span-weighted circular mean."""
    angs, w = [], []
    for p in pieces:
        n, _ = fit_circle(p["v"])
        if math.hypot(n[0], n[1]) < 1e-6:
            continue
        d = math.degrees(math.atan2(n[0], -n[1]))  # direction of n x z
        angs.append(d)
        w.append(p["span"])
    if not angs:
        return float("nan"), 0
    a = np.radians(np.array(angs) * 4)  # mod 90 -> full circle
    w = np.array(w)
    mean = math.degrees(math.atan2((w * np.sin(a)).sum(), (w * np.cos(a)).sum())) / 4
    return mean, len(angs)


def pieces() -> None:
    data = load_pins()
    runs = load_runs()
    rows = []
    for name, result in sorted(runs.items()):
        pin = data["photos"].get(name)
        if not pin:
            continue
        ps = photo_pieces(result, pin["north"])
        off, n = grid_offset(ps)
        by = {s: sum(1 for p in ps if p["system"] == s) for s in SYSTEMS}
        rows.append({"photo": name, "seq": pin["seq"], "pieces": len(ps), "by_system": by, "grid_offset_deg": off, "pinned_north": pin["north"]})
        print(f"{pin['seq']:3d} {name[-7:-4]}  pieces {len(ps):3d}  {by}  grid offset {off:+6.1f} deg from {n} pieces")
    (OUT / "pieces.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")


# ---------------------------------------------------------------------------
# step 4 and 5: the gap, and the merged layout
# ---------------------------------------------------------------------------

def all_lines(gap_m: float, systems=SYSTEMS, correct_headings: bool = True) -> list[dict]:
    """Every usable piece of every photo as a line on the plan at this gap."""
    data = load_pins()
    runs = load_runs()
    corrections = {}
    if correct_headings and (OUT / "pieces.json").exists():
        for row in json.loads((OUT / "pieces.json").read_text(encoding="utf-8")):
            if row["pieces"] >= 10 and not math.isnan(row["grid_offset_deg"]):
                corrections[row["photo"]] = row["grid_offset_deg"]
    lines = []
    for name, result in runs.items():
        pin = data["photos"].get(name)
        if not pin:
            continue
        north = pin["north"] - corrections.get(name, 0.0)
        cam = np.array([pin["x"], pin["y"]])
        for p in photo_pieces(result, north):
            if p["system"] not in systems:
                continue
            placed = piece_on_plan(p["v"], gap_m)
            if placed is None:
                continue
            d, p0, t0, t1 = placed
            if t1 - t0 < 0.3:
                continue
            lines.append({"system": p["system"], "photo": name, "seq": pin["seq"], "d": d, "p0": cam + p0, "t0": t0, "t1": t1, "conf": p["conf"]})
    return lines


def _angle_of(d: np.ndarray) -> float:
    return math.degrees(math.atan2(d[1], d[0])) % 180.0


def snap(lines: list[dict]) -> list[dict]:
    """Keep the lines that run along the building grid, and make them run exactly along it.

    The sheet's axes are the column grid, and the pipes follow the columns. A
    line more than SNAP_DEG off both axes is a drop, a coil or a mistake, not a
    run. Snapping keeps the line's closest point and turns its direction.
    """
    out = []
    for ln in lines:
        a = _angle_of(ln["d"])
        for axis in (0.0, 90.0):
            if abs((a - axis + 90) % 180 - 90) <= SNAP_DEG:
                d = np.array([1.0, 0.0]) if axis == 0.0 else np.array([0.0, 1.0])
                # endpoints projected onto the snapped line through the same closest point
                e0 = ln["p0"] + ln["t0"] * ln["d"]
                e1 = ln["p0"] + ln["t1"] * ln["d"]
                mid = (e0 + e1) / 2
                perp = np.array([-d[1], d[0]])
                p0 = mid - float((mid - ln["p0"]) @ perp) * perp  # keep the sideways offset at the middle
                t0, t1 = float((e0 - p0) @ d), float((e1 - p0) @ d)
                out.append({**ln, "d": d, "p0": p0, "t0": min(t0, t1), "t1": max(t0, t1)})
                break
    return out


def merge(lines: list[dict]) -> list[dict]:
    """Lines from different photos on the same line become one, with the union of their extents."""
    groups: list[dict] = []
    for ln in sorted(lines, key=lambda l: -(l["t1"] - l["t0"])):
        a = _angle_of(ln["d"])
        for g in groups:
            if g["system"] != ln["system"]:
                continue
            da = abs((a - g["angle"] + 90) % 180 - 90)
            if da > MERGE_ANGLE_DEG:
                continue
            # perpendicular distance from the group's line to this line's closest point
            perp = np.array([-g["d"][1], g["d"][0]])
            off = float((ln["p0"] - g["p0"]) @ perp)
            if abs(off) > MERGE_OFFSET_M:
                continue
            # extents along the group's direction
            s0 = float((ln["p0"] + ln["t0"] * ln["d"] - g["p0"]) @ g["d"])
            s1 = float((ln["p0"] + ln["t1"] * ln["d"] - g["p0"]) @ g["d"])
            s0, s1 = min(s0, s1), max(s0, s1)
            if s0 > g["t1"] + JOIN_GAP_M or s1 < g["t0"] - JOIN_GAP_M:
                continue  # same line, but a different stretch of it
            # the group's offset becomes the length-weighted mean of its members
            w_old = sum(m["t1"] - m["t0"] for m in g["members"])
            w_new = ln["t1"] - ln["t0"]
            g["p0"] = g["p0"] + perp * (off * w_new / (w_old + w_new))
            g["t0"], g["t1"] = min(g["t0"], s0), max(g["t1"], s1)
            g["members"].append(ln)
            g["photos"].add(ln["seq"])
            g["mismatch"].append(abs(off))
            break
        else:
            groups.append({"system": ln["system"], "d": ln["d"], "angle": a, "p0": ln["p0"], "t0": ln["t0"], "t1": ln["t1"],
                           "members": [ln], "photos": {ln["seq"]}, "mismatch": []})
    return groups


def fold(groups: list[dict]) -> list[dict]:
    """Second pass: parallel copies of one run, each photo's own small error apart, become one.

    Each photo places a run with its own error of a few tenths of a metre, so
    one main can come out as two or three parallel lines. Overlapping lines on
    the same axis within FOLD_OFFSET_M are folded into the longer one, at the
    length-weighted offset, keeping every photo that saw either.
    """
    groups = sorted(groups, key=lambda g: -(g["t1"] - g["t0"]))
    done: list[dict] = []
    for g in groups:
        perp = np.array([-g["d"][1], g["d"][0]])
        for h in done:
            if h["system"] != g["system"] or abs(float(h["d"] @ g["d"])) < 0.99:
                continue
            off = float((g["p0"] - h["p0"]) @ perp)
            if abs(off) > FOLD_OFFSET_M:
                continue
            s0 = float((g["p0"] + g["t0"] * g["d"] - h["p0"]) @ h["d"])
            s1 = float((g["p0"] + g["t1"] * g["d"] - h["p0"]) @ h["d"])
            s0, s1 = min(s0, s1), max(s0, s1)
            if s0 > h["t1"] + JOIN_GAP_M or s1 < h["t0"] - JOIN_GAP_M:
                continue
            w_h, w_g = h["t1"] - h["t0"], g["t1"] - g["t0"]
            h["p0"] = h["p0"] + perp * (off * w_g / (w_h + w_g))
            h["t0"], h["t1"] = min(h["t0"], s0), max(h["t1"], s1)
            h["members"] += g["members"]
            h["photos"] |= g["photos"]
            h["mismatch"] += g["mismatch"] + [abs(off)]
            break
        else:
            done.append(g)
    return done


def keep(groups: list[dict]) -> list[dict]:
    """What goes on the plan: agreed by two photos, or long enough on its own."""
    return [g for g in groups if len(g["photos"]) >= 2 or g["t1"] - g["t0"] >= MIN_SINGLE_M]


def gap() -> None:
    """The gap that makes the most pipe seen from two photos land on one line.

    Judged by the length that two or more photos agree on (not its share of
    the total: single-photo lines grow with the gap and would drag the share
    down). Mismatch alone will not do: a small gap keeps every photo's lines
    close to its own pin, so lines rarely meet and the few that do agree by
    luck. Agreement that grows with the gap, then stops growing, is the gap.
    """
    report = {}
    for system in SYSTEMS:
        best = None
        rows = []
        for g in GAP_GRID_M:
            groups = merge(snap(all_lines(float(g), systems=(system,))))
            total = sum(grp["t1"] - grp["t0"] for grp in groups)
            twice = sum(grp["t1"] - grp["t0"] for grp in groups if len(grp["photos"]) >= 2)
            shared = [x for grp in groups for x in grp["mismatch"] if len(grp["photos"]) >= 2]
            share = twice / total if total else 0.0
            mism = float(np.median(shared)) if shared else float("nan")
            rows.append((float(g), share, twice, total, mism, sum(1 for grp in groups if len(grp["photos"]) >= 2)))
            # the gap with the most pipe agreed on, in metres; a tie goes to the smaller gap
            if best is None or twice > best[2] + 1e-9:
                best = rows[-1]
        print()
        print(system)
        for g, share, twice, total, mism, n in rows:
            print(f"  gap {g:4.2f} m  seen twice {twice:6.1f} of {total:6.1f} m ({share * 100:4.1f} %)  {n:3d} lines  median mismatch {mism:4.2f} m")
        if best:
            print(f"  -> gap {best[0]:.2f} m")
            report[system] = {"gap_m": best[0], "share_seen_twice": round(best[1], 3), "length_seen_twice_m": round(best[2], 1), "length_m": round(best[3], 1), "median_mismatch_m": best[4]}
    (OUT / "gap.json").write_text(json.dumps(report, indent=1), encoding="utf-8")


def layout() -> None:
    gaps = json.loads((OUT / "gap.json").read_text(encoding="utf-8")) if (OUT / "gap.json").exists() else {}
    data = load_pins()
    out = {"systems": {}, "lines": []}
    for system in SYSTEMS:
        g = gaps.get(system, {}).get("gap_m", 1.2)
        groups = keep(fold(merge(snap(all_lines(g, systems=(system,))))))
        total = sum(grp["t1"] - grp["t0"] for grp in groups)
        twice = sum(grp["t1"] - grp["t0"] for grp in groups if len(grp["photos"]) >= 2)
        out["systems"][system] = {"gap_m": g, "lines": len(groups), "length_m": round(total, 1), "length_seen_twice_m": round(twice, 1)}
        print(f"{system:16s} gap {g:.2f} m  {len(groups):4d} lines  {total:7.1f} m  of which seen from 2+ photos {twice:6.1f} m")
        for grp in groups:
            a = grp["p0"] + grp["t0"] * grp["d"]
            b = grp["p0"] + grp["t1"] * grp["d"]
            out["lines"].append({"system": system, "a": [round(float(a[0]), 2), round(float(a[1]), 2)], "b": [round(float(b[0]), 2), round(float(b[1]), 2)],
                                 "photos": sorted(grp["photos"]), "length_m": round(grp["t1"] - grp["t0"], 2)})
    (OUT / "layout.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    draw_layout(out, data)


def draw_layout(out: dict, data: dict) -> None:
    import pymupdf
    from PIL import Image, ImageDraw

    plan = data["plan"]
    dpi = 60
    doc = pymupdf.open(PLAN_PDF)
    pix = doc[0].get_pixmap(dpi=dpi)
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples).convert("RGBA")
    k = dpi / 300.0  # the plan was tiled at 300 dpi
    ppm = plan["px_per_meter"] * k
    ox, oy = plan["origin_px_x"] * k, plan["origin_px_y"] * k

    def to_px(x, y):
        return (ox + ppm * x, oy - ppm * y)

    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    pen = ImageDraw.Draw(layer)
    colours = {"fire pipe": (220, 30, 30, 230), "electrical pipe": (230, 170, 0, 230), "hvac duct": (30, 140, 230, 230)}
    for ln in out["lines"]:
        pen.line([to_px(*ln["a"]), to_px(*ln["b"])], fill=colours[ln["system"]], width=3 if len(ln["photos"]) >= 2 else 1)
    for name, pin in data["photos"].items():
        x, y = to_px(pin["x"], pin["y"])
        pen.ellipse([x - 4, y - 4, x + 4, y + 4], outline=(0, 0, 0, 255), fill=(255, 255, 255, 255))
        pen.text((x + 5, y - 6), str(pin["seq"]), fill=(0, 0, 0, 255))
    img = Image.alpha_composite(img, layer).convert("RGB")
    xs = [p["x"] for p in data["photos"].values()]
    ys = [p["y"] for p in data["photos"].values()]
    x0, y0 = to_px(min(xs) - 12, max(ys) + 12)
    x1, y1 = to_px(max(xs) + 12, min(ys) - 12)
    img.crop((int(x0), int(y0), int(x1), int(y1))).save(OUT / "layout.png")
    print("wrote", OUT / "layout.png")


if __name__ == "__main__":
    if "--from" in sys.argv:
        OUT = Path(sys.argv[sys.argv.index("--from") + 1]).resolve()
        RESULTS = OUT / "results"
    {"pins": pins, "pieces": pieces, "gap": gap, "layout": layout}[sys.argv[1]]()
