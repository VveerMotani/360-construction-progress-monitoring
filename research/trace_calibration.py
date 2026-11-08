"""Experiment 15: recalibrating the hand tracing from the tracing itself.

    python research/trace_calibration.py snapshot   # (export of the hand trace + pins; not included)
    python research/trace_calibration.py fit        # headings and height, figures, calibration.json

The Trace step puts a click in a photo on the plan as

    heading  = north offset - image longitude
    distance = h / tan(latitude)
    point    = pin + distance * (cos heading, sin heading)

with each photo's PINNED north offset and one assumed h (1.2 m). The user saw
lines come out slanted and outside the floor plate. Two unknowns are fitted
from the clicks themselves, in this order:

1. A heading correction per photo. A run's direction on the plan does not
   depend on h at all (both ends scale with it), and the fire pipe follows the
   column grid. So each photo is turned by the angle that brings its traced
   pieces onto the grid, mod 90 degrees -- a span-weighted median of how far
   each piece is off, so one diagonal piece cannot drag it.
2. The height h. With the headings fixed, every offset from a camera scales
   with h. The same pipe traced from two photos must land on one line, so h is
   the value at which parallel pieces from different photos agree best.

Clicks within LOW_DEG of the horizon carry most of the error (a half degree at
5 degrees up is a metre), so the fit leans on the higher ones, and the figure
shows them apart.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "research" / "experiments" / "2026-10-07-15-trace-calibration"
SNAPSHOT = EXPERIMENT / "trace_snapshot.json"
FLOOR = (26.3, 74.7, 18.7, 67.0)   # facade lines from the column grid (experiment 14)
LOW_DEG = 10.0                     # clicks below this are "low": used, but trusted less
H_GRID = np.arange(0.4, 3.01, 0.05)
PARALLEL_DEG = 8.0                 # pieces this close in direction can be the same pipe
SAME_PIPE_M = 1.5                  # ... if, at the fitted h, they are this close sideways


def snapshot() -> None:
    """In the project this fetched the hand-traced fire pipe and the pins from
    the capture platform, which is not part of this repository. Supply
    trace_snapshot.json yourself: {"trace": {"marks": [...]}, "pins": {...},
    "names": {...}}."""
    raise SystemExit(f"put the trace export at {SNAPSHOT}")


def load():
    d = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    return d["trace"], d["pins"], d["names"]


def to_plan(pin: dict, lon: float, lat: float, h: float, turn: float = 0.0) -> np.ndarray:
    head = math.radians(pin["north"] + turn - lon)
    d = h / math.tan(math.radians(lat))
    return np.array([pin["x"] + d * math.cos(head), pin["y"] + d * math.sin(head)])


def pieces(trace: dict, pins: dict, names: dict) -> list[dict]:
    """Every pair of neighbouring clicks along a traced pipe, with its photo."""
    out = []
    for m in trace["marks"]:
        if m["kind"] not in ("pipe", "unsure"):
            continue
        for a, b in zip(m["points"], m["points"][1:]):
            span = angle_between(a, b)
            out.append({"frame": m["frame_id"], "photo": names[m["frame_id"]][-7:-4], "a": a, "b": b,
                        "span": span, "low": min(a[1], b[1]) < LOW_DEG})
    return out


def angle_between(a, b) -> float:
    def u(p):
        lo, la = math.radians(p[0]), math.radians(p[1])
        return np.array([math.cos(la) * math.cos(lo), math.cos(la) * math.sin(lo), math.sin(la)])
    return math.degrees(math.acos(float(np.clip(u(a) @ u(b), -1, 1))))


def direction(p: dict, pin: dict, turn: float) -> float:
    a = to_plan(pin, *p["a"], 1.0, turn)
    b = to_plan(pin, *p["b"], 1.0, turn)
    return math.degrees(math.atan2(b[1] - a[1], b[0] - a[0])) % 180.0


def off_grid(angle: float) -> float:
    """Signed degrees from the nearest grid axis, in (-45, 45]."""
    return (angle + 45.0) % 90.0 - 45.0


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    o = np.argsort(values)
    c = np.cumsum(weights[o])
    return float(values[o][np.searchsorted(c, c[-1] / 2)])


def fit() -> None:
    trace, pins, names = load()
    ps = pieces(trace, pins, names)
    h0 = trace["height_m"] or 1.2
    report: dict = {"height_used": h0, "photos": {}}

    # 1. headings
    turns = {}
    for frame in sorted({p["frame"] for p in ps}, key=lambda f: names[f]):
        mine = [p for p in ps if p["frame"] == frame]
        e = np.array([off_grid(direction(p, pins[frame], 0.0)) for p in mine])
        w = np.array([p["span"] * (0.3 if p["low"] else 1.0) for p in mine])
        turn = -weighted_median(e, w)
        after = np.array([off_grid(direction(p, pins[frame], turn)) for p in mine])
        turns[frame] = turn
        report["photos"][names[frame][-7:-4]] = {
            "pinned_north": pins[frame]["north"], "turn_deg": round(turn, 1),
            "pieces": len(mine), "off_grid_before_deg": [round(float(x), 1) for x in e],
            "off_grid_after_deg": [round(float(x), 1) for x in after],
        }
        print(f"photo {names[frame][-7:-4]}: turn {turn:+6.1f} deg   off grid before {np.round(e, 0)}   after {np.round(after, 0)}")

    # 2. height: parallel pieces from different photos should lie on one line
    def sideways(p, q, h):
        a1, b1 = to_plan(pins[p["frame"]], *p["a"], h, turns[p["frame"]]), to_plan(pins[p["frame"]], *p["b"], h, turns[p["frame"]])
        a2, b2 = to_plan(pins[q["frame"]], *q["a"], h, turns[q["frame"]]), to_plan(pins[q["frame"]], *q["b"], h, turns[q["frame"]])
        d = (b1 - a1) / (np.linalg.norm(b1 - a1) + 1e-9)
        n = np.array([-d[1], d[0]])
        # how far q's midpoint is beside p's line, and whether they overlap along it
        mid = (a2 + b2) / 2
        t = [(a1 - a1) @ d, (b1 - a1) @ d]
        s = sorted([(a2 - a1) @ d, (b2 - a1) @ d])
        overlap = min(t[1], s[1]) - max(t[0], s[0])
        return abs((mid - a1) @ n), overlap

    pairs = []
    for i, p in enumerate(ps):
        for q in ps[i + 1:]:
            if p["frame"] == q["frame"]:
                continue
            dp = direction(p, pins[p["frame"]], turns[p["frame"]])
            dq = direction(q, pins[q["frame"]], turns[q["frame"]])
            if abs((dp - dq + 90) % 180 - 90) <= PARALLEL_DEG:
                pairs.append((p, q))
    costs = []
    for h in H_GRID:
        c = []
        for p, q in pairs:
            gap, overlap = sideways(p, q, h)
            if overlap > -1.0:          # they cover a common stretch: candidate same pipe
                c.append(min(gap, SAME_PIPE_M))
        costs.append(np.mean(c) if c else np.nan)
    costs = np.array(costs)
    h_fit = float(H_GRID[np.nanargmin(costs)]) if np.isfinite(costs).any() else h0
    report["height_fit_m"] = h_fit
    report["height_curve"] = {f"{h:.2f}": (None if not np.isfinite(c) else round(float(c), 3)) for h, c in zip(H_GRID, costs)}
    report["pairs_considered"] = len(pairs)
    print(f"height: {len(pairs)} parallel pairs across photos; best agreement at h = {h_fit:.2f} m "
          f"(mean sideways gap {np.nanmin(costs):.2f} m; at 1.20 m: {costs[np.argmin(abs(H_GRID - 1.2))]:.2f} m)")

    # 3. how much is inside the floor plate, before and after
    def inside_share(h, use_turns):
        pts = [to_plan(pins[p["frame"]], *x, h, turns[p["frame"]] if use_turns else 0.0) for p in ps for x in (p["a"], p["b"])]
        x0, x1, y0, y1 = FLOOR
        return float(np.mean([(x0 - 0.5 <= q[0] <= x1 + 0.5) and (y0 - 0.5 <= q[1] <= y1 + 0.5) for q in pts]))
    report["inside_floor_before"] = inside_share(h0, False)
    report["inside_floor_after"] = inside_share(h_fit, True)
    print(f"click points inside the floor plate: {report['inside_floor_before']:.0%} before, {report['inside_floor_after']:.0%} after")
    report["turns_by_frame"] = turns
    (EXPERIMENT / "calibration.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    draw(ps, pins, names, h0, {f: 0.0 for f in turns}, EXPERIMENT / "before.png", f"as traced: pinned headings, h = {h0:.2f} m")
    draw(ps, pins, names, h_fit, turns, EXPERIMENT / "after.png", f"recalibrated: headings turned, h = {h_fit:.2f} m")


def draw(ps, pins, names, h, turns, path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colours = {}
    palette = ["#d62728", "#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b"]
    fig, ax = plt.subplots(figsize=(7, 7))
    x0, x1, y0, y1 = FLOOR
    ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="#888", lw=1))
    for p in ps:
        col = colours.setdefault(p["frame"], palette[len(colours) % len(palette)])
        a = to_plan(pins[p["frame"]], *p["a"], h, turns[p["frame"]])
        b = to_plan(pins[p["frame"]], *p["b"], h, turns[p["frame"]])
        ax.plot([a[0], b[0]], [a[1], b[1]], color=col, lw=2.2, ls=":" if p["low"] else "-")
    for f, col in colours.items():
        pin = pins[f]
        ax.plot(pin["x"], pin["y"], "o", color=col, ms=8, mec="k")
        ax.annotate(names[f][-7:-4], (pin["x"], pin["y"]), xytext=(4, 4), textcoords="offset points", fontsize=9)
    xs = [pins[f]["x"] for f in colours]
    ys = [pins[f]["y"] for f in colours]
    ax.set_xlim(min(xs) - 16, max(xs) + 8)
    ax.set_ylim(min(ys) - 14, max(ys) + 14)
    ax.set_aspect("equal")
    ax.set_title(title + "\n(colour = photo, dotted = a click below 10 deg; grey box = facades)", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print("wrote", path)


RELIABLE_DEG = 8.0  # as lib/trace.ts MIN_RELIABLE_LAT_DEG


def reliable_part(a, b, steps: int = 96):
    """The stretch of the great circle a->b at least RELIABLE_DEG up, as its two ends (lib/trace.ts reliablePart)."""
    def u(p):
        lo, la = math.radians(p[0]), math.radians(p[1])
        return np.array([math.cos(la) * math.cos(lo), math.cos(la) * math.sin(lo), math.sin(la)])
    ua, ub = u(a), u(b)
    om = math.acos(float(np.clip(ua @ ub, -1, 1)))
    if om < 1e-9:
        return None
    pts = []
    for t in np.linspace(0, 1, steps + 1):
        v = (math.sin((1 - t) * om) * ua + math.sin(t * om) * ub) / math.sin(om)
        pts.append((math.degrees(math.atan2(v[1], v[0])), math.degrees(math.asin(float(np.clip(v[2], -1, 1))))))
    keep = [i for i, p in enumerate(pts) if p[1] >= RELIABLE_DEG]
    if len(keep) < 2:
        return None
    return pts[keep[0]], pts[keep[-1]]


def check() -> None:
    """After the pins were corrected: the tracing as the tool now places it (pins as stored, h, low ends cut)."""
    snapshot()
    trace, pins, names = load()
    h = trace["height_m"] or 1.2
    cut = []
    for p in pieces(trace, pins, names):
        part = reliable_part(p["a"], p["b"])
        if part:
            cut.append({**p, "a": list(part[0]), "b": list(part[1]), "low": False})
    e = [abs(off_grid(direction(p, pins[p["frame"]], 0.0))) for p in cut]
    pts = [to_plan(pins[p["frame"]], *x, h) for p in cut for x in (p["a"], p["b"])]
    x0, x1, y0, y1 = FLOOR
    inside = float(np.mean([(x0 <= q[0] <= x1) and (y0 <= q[1] <= y1) for q in pts]))
    over = max(max(q[0] - x1, x0 - q[0], q[1] - y1, y0 - q[1], 0) for q in pts)
    print(f"{len(cut)} pieces kept above {RELIABLE_DEG:.0f} deg; off grid median {np.median(e):.1f} deg, worst {max(e):.1f}; "
          f"inside the floor {inside:.0%}, worst overshoot {over:.1f} m")
    report = json.loads((EXPERIMENT / "calibration.json").read_text(encoding="utf-8"))
    report["after_correction"] = {"pieces_kept": len(cut), "off_grid_median_deg": float(np.median(e)),
                                  "off_grid_worst_deg": float(max(e)), "inside_floor": inside, "worst_overshoot_m": float(over)}
    (EXPERIMENT / "calibration.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    draw(cut, pins, names, h, {p["frame"]: 0.0 for p in cut}, EXPERIMENT / "corrected.png",
         f"pins corrected to the grid, h = {h:.2f} m, low ends cut at {RELIABLE_DEG:.0f} deg")


if __name__ == "__main__":
    {"snapshot": snapshot, "fit": fit, "check": check}[sys.argv[1]]()
