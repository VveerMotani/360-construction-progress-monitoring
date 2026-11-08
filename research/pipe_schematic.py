"""Experiment 14: from lines on the plan to runs, tees and a not-to-scale schematic.

    python research/pipe_schematic.py <layout folder>            # reads <folder>/layout.json, writes there
    python research/pipe_schematic.py <out folder> --sheets     # blank sheets for drawing the answer key

INPUT is a pipe_layout.py layout (lines already snapped to the building's
axes and merged across photos). The steps:

1. The column grid, from the drawing: the columns are the solid black
   rectangles on the sheet. Their centres give 9 grid lines each way between
   the four facades -- numbered 0-10 west to east (0 and 10 are the facades) and
   lettered S, A-I, N south to north (S and N are the facades). The floor plate has no inner
   columns; the grid is carried across it.
2. Runs: collinear lines (offsets within 0.5 m) whose gap is under 3 m are one
   run. A run's offset is put on a grid line when within 0.8 m of it, else
   described as lying in the bay between two lines.
3. Joints, for every east-west run against every north-south run, at the
   point where their lines meet:
     tee       the point is inside one run and at an end of the other (the
               one that ends is the branch)
     elbow     the point is at an end of both
     cross     the point is inside both: branches leaving on both sides (a
               cross fitting) or one pipe passing under another -- photos
               alone do not yet tell which
   "At an end" means within 1 m of it; "inside" means more than 1 m from both
   ends.
4. Figures: on the plan, and as a schematic where every bay is drawn the same
   size -- distances are not kept, the order of runs, bays and joints is.
"""

from __future__ import annotations

import json
import string
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PINS = ROOT / "research" / "experiments" / "2026-10-01-10-whole-walk-detection" / "pins.json"
SHEET = Path(__import__("os").environ.get("PLAN_PDF", "plan.pdf"))  # client drawing, not included
SYSTEM = "fire pipe"
COLLINEAR_M = 0.5
JOIN_GAP_M = 3.0
ON_LINE_M = 0.8
END_M = 1.0


# ---------------------------------------------------------------------------
# the column grid
# ---------------------------------------------------------------------------

def columns() -> list[tuple[float, float, float, float]]:
    """(x, y, width, height) in metres of every solid black rectangle of column size."""
    import cv2
    import pymupdf

    plan = json.loads(PINS.read_text(encoding="utf-8"))["plan"]
    dpi = 100
    pix = pymupdf.open(SHEET)[0].get_pixmap(dpi=dpi)
    img = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)[..., :3]
    dark = (cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) < 60).astype(np.uint8)
    solid = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, _, st, cen = cv2.connectedComponentsWithStats(solid)
    k = 300 / dpi / plan["px_per_meter"]
    out = []
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if 6 <= w <= 60 and 6 <= h <= 60 and a / (w * h) > 0.8:
            cx, cy = cen[i]
            out.append((cx * k, (plan["origin_px_y"] - cy * 300 / dpi) / plan["px_per_meter"], w * k, h * k))
    return out


def _cluster(v: list[float], tol: float = 0.5) -> list[float]:
    v = sorted(v)
    groups = [[v[0]]]
    for x in v[1:]:
        if x - groups[-1][-1] <= tol:
            groups[-1].append(x)
        else:
            groups.append([x])
    return [float(np.mean(g)) for g in groups]


def grid() -> dict:
    """Named grid lines: {'x': [(name, metres)], 'y': [...]} including the facades."""
    cols = [c for c in columns() if max(c[2], c[3]) >= 1.5 and min(c[2], c[3]) >= 0.8]  # the facade columns
    xs = np.array([c[0] for c in cols])
    ys = np.array([c[1] for c in cols])
    west, east, south, north = float(np.min(xs)), float(np.max(xs)), float(np.min(ys)), float(np.max(ys))
    inner_x = _cluster([c[0] for c in cols if (abs(c[1] - south) < 1 or abs(c[1] - north) < 1) and west + 1 < c[0] < east - 1])
    inner_y = _cluster([c[1] for c in cols if (abs(c[0] - west) < 1 or abs(c[0] - east) < 1) and south + 1 < c[1] < north - 1])
    # x lines are numbered with the facades as 0 and the last number, so no
    # name is shared with the lettered y lines (an "E" for east would clash)
    gx = [("0", west)] + [(str(i + 1), x) for i, x in enumerate(inner_x)] + [(str(len(inner_x) + 1), east)]
    gy = [("S", south)] + [(string.ascii_uppercase[i], y) for i, y in enumerate(inner_y)] + [("N", north)]
    return {"x": gx, "y": gy}


def place(v: float, lines: list[tuple[str, float]]) -> str:
    """'on 4' / 'bay 4-5' / 'outside W' for an offset against named grid lines."""
    for name, m in lines:
        if abs(v - m) <= ON_LINE_M:
            return f"on {name}"
    for (n0, m0), (n1, m1) in zip(lines, lines[1:]):
        if m0 < v < m1:
            return f"bay {n0}-{n1}"
    return f"outside {lines[0][0] if v < lines[0][1] else lines[-1][0]}"


# ---------------------------------------------------------------------------
# runs and joints
# ---------------------------------------------------------------------------

def runs_of(lines: list[dict]) -> list[dict]:
    """Collinear lines joined into runs. Each run: axis 'x' (east-west, constant y) or 'y', offset, extent."""
    items = []
    for ln in lines:
        (ax, ay), (bx, by) = ln["a"], ln["b"]
        if abs(ay - by) <= abs(ax - bx):   # east-west
            items.append({"axis": "x", "off": (ay + by) / 2, "t0": min(ax, bx), "t1": max(ax, bx), "photos": set(ln["photos"])})
        else:
            items.append({"axis": "y", "off": (ax + bx) / 2, "t0": min(ay, by), "t1": max(ay, by), "photos": set(ln["photos"])})
    runs = []
    for axis in ("x", "y"):
        its = sorted((i for i in items if i["axis"] == axis), key=lambda i: (i["off"], i["t0"]))
        # group by offset, then join along
        groups: list[list[dict]] = []
        for it in sorted(its, key=lambda i: i["off"]):
            if groups and abs(np.mean([g["off"] for g in groups[-1]]) - it["off"]) <= COLLINEAR_M:
                groups[-1].append(it)
            else:
                groups.append([it])
        for g in groups:
            g.sort(key=lambda i: i["t0"])
            cur = None
            for it in g:
                if cur and it["t0"] - cur["t1"] <= JOIN_GAP_M:
                    if it["t0"] > cur["t1"]:
                        cur["gaps"].append((cur["t1"], it["t0"]))
                    cur["t1"] = max(cur["t1"], it["t1"])
                    cur["photos"] |= it["photos"]
                    cur["parts"].append(it)
                else:
                    cur = {"axis": axis, "t0": it["t0"], "t1": it["t1"], "photos": set(it["photos"]), "gaps": [], "parts": [it]}
                    runs.append(cur)
            for r in runs:
                if "off" not in r:
                    w = np.array([p["t1"] - p["t0"] for p in r["parts"]])
                    r["off"] = float(np.average([p["off"] for p in r["parts"]], weights=np.maximum(w, 1e-3)))
    for i, r in enumerate(runs):
        r["id"] = f"R{i + 1}"
        r["length_m"] = r["t1"] - r["t0"]
        r["seen_twice"] = len(r["photos"]) >= 2
    return runs


def joints(runs: list[dict]) -> list[dict]:
    out = []
    ew = [r for r in runs if r["axis"] == "x"]
    ns = [r for r in runs if r["axis"] == "y"]

    def where(t: float, r: dict) -> str:
        if t < r["t0"] - END_M or t > r["t1"] + END_M:
            return "off"
        if abs(t - r["t0"]) <= END_M or abs(t - r["t1"]) <= END_M:
            return "end"
        return "inside"

    for h in ew:
        for v in ns:
            px, py = v["off"], h["off"]
            wh, wv = where(px, h), where(py, v)
            if "off" in (wh, wv):
                continue
            if wh == "inside" and wv == "inside":
                kind, main, branch = "cross", None, None
            elif wh == "end" and wv == "end":
                kind, main, branch = "elbow", None, None
            else:
                kind = "tee"
                main, branch = (h, v) if wh == "inside" else (v, h)
            out.append({"kind": kind, "x": px, "y": py, "runs": [h["id"], v["id"]],
                        "main": main["id"] if main else None, "branch": branch["id"] if branch else None,
                        "seen_twice": h["seen_twice"] and v["seen_twice"]})
    return out


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

def to_schematic(v: float, lines: list[tuple[str, float]], cell: float) -> float:
    """Grid-line metres -> equal spacing: a bay of any width becomes one cell."""
    ms = [m for _, m in lines]
    return float(np.interp(v, ms, np.arange(len(ms)) * cell, left=-0.5 * cell, right=(len(ms) - 0.5) * cell))


def draw(runs: list[dict], jts: list[dict], g: dict, path_plan: Path, path_schematic: Path) -> None:
    from PIL import Image, ImageDraw, ImageFont

    try:
        font = ImageFont.truetype("arial.ttf", 13)
        small = ImageFont.truetype("arial.ttf", 10)
    except OSError:
        font = small = ImageFont.load_default()
    red, grey, blue = (210, 30, 30), (170, 170, 170), (30, 90, 200)

    def render(xmap, ymap, size, label_runs):
        img = Image.new("RGB", size, "white")
        pen = ImageDraw.Draw(img)
        for name, m in g["x"]:
            X = xmap(m)
            pen.line([(X, ymap(g["y"][0][1])), (X, ymap(g["y"][-1][1]))], fill=grey, width=1)
            pen.text((X - 4, ymap(g["y"][-1][1]) - 20), name, fill=(80, 80, 80), font=font)
        for name, m in g["y"]:
            Y = ymap(m)
            pen.line([(xmap(g["x"][0][1]), Y), (xmap(g["x"][-1][1]), Y)], fill=grey, width=1)
            pen.text((xmap(g["x"][0][1]) - 22, Y - 7), name, fill=(80, 80, 80), font=font)
        for r in runs:
            if r["axis"] == "x":
                a, b = (xmap(r["t0"]), ymap(r["off"])), (xmap(r["t1"]), ymap(r["off"]))
            else:
                a, b = (xmap(r["off"]), ymap(r["t0"])), (xmap(r["off"]), ymap(r["t1"]))
            pen.line([a, b], fill=red if r["seen_twice"] else (240, 150, 150), width=4 if r["seen_twice"] else 2)
            if label_runs:
                pen.text(((a[0] + b[0]) / 2 + 3, (a[1] + b[1]) / 2 + 2), r["id"], fill=(120, 0, 0), font=small)
        for j in jts:
            X, Y = xmap(j["x"]), ymap(j["y"])
            if j["kind"] == "tee":
                pen.ellipse([X - 5, Y - 5, X + 5, Y + 5], fill=blue)
            elif j["kind"] == "elbow":
                pen.rectangle([X - 4, Y - 4, X + 4, Y + 4], outline=blue, width=2)
            else:
                pen.line([(X - 4, Y - 4), (X + 4, Y + 4)], fill=(90, 90, 90), width=2)
                pen.line([(X - 4, Y + 4), (X + 4, Y - 4)], fill=(90, 90, 90), width=2)
        key_y = size[1] - 22
        pen.text((10, key_y), "thick red: seen from 2+ photos   thin: one photo   dot: tee   square: elbow   x: cross (branches both sides, or one pipe under another)",
                 fill=(60, 60, 60), font=small)
        return img

    x0, x1 = g["x"][0][1] - 2, g["x"][-1][1] + 2
    y0, y1 = g["y"][0][1] - 2, g["y"][-1][1] + 2
    s = 13.0  # px per metre
    plan = render(lambda x: 40 + (x - x0) * s, lambda y: 40 + (y1 - y) * s,
                  (int(80 + (x1 - x0) * s), int(80 + (y1 - y0) * s)), False)
    plan.save(path_plan)
    cell = 70.0
    nx, ny = len(g["x"]), len(g["y"])
    sch = render(lambda x: 60 + to_schematic(x, g["x"], cell), lambda y: 60 + (ny - 1) * cell - to_schematic(y, g["y"], cell),
                 (int(120 + (nx - 1) * cell), int(120 + (ny - 1) * cell)), True)
    sch.save(path_schematic)


def main(folder: Path) -> None:
    layout = json.loads((folder / "layout.json").read_text(encoding="utf-8"))
    lines = [ln for ln in layout["lines"] if ln["system"] == SYSTEM]
    g = grid()
    runs = runs_of(lines)
    jts = joints(runs)
    rows = []
    for r in runs:
        lines_ = g["y"] if r["axis"] == "x" else g["x"]
        along = g["x"] if r["axis"] == "x" else g["y"]
        rows.append({
            "id": r["id"], "direction": "east-west" if r["axis"] == "x" else "north-south",
            "where": place(r["off"], lines_), "from": place(r["t0"], along), "to": place(r["t1"], along),
            "length_m": round(r["length_m"], 1), "seen_twice": r["seen_twice"], "photos": len(r["photos"]),
            "unseen_gaps_m": [round(b - a, 1) for a, b in r["gaps"]],
            "tees_on_it": sum(1 for j in jts if j["main"] == r["id"]),
        })
    kinds = {k: sum(1 for j in jts if j["kind"] == k) for k in ("tee", "elbow", "cross")}
    out = {"grid": g, "runs": rows, "joints": jts, "summary": {
        "runs": len(runs), "runs_seen_twice": sum(r["seen_twice"] for r in runs), **kinds,
        "tees_seen_twice": sum(1 for j in jts if j["kind"] == "tee" and j["seen_twice"])}}
    (folder / "schematic.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    draw(runs, jts, g, folder / "runs_on_plan.png", folder / "schematic.png")
    print("grid x:", ", ".join(f"{n} {m:.1f}" for n, m in g["x"]))
    print("grid y:", ", ".join(f"{n} {m:.1f}" for n, m in g["y"]))
    print(out["summary"])
    for row in rows:
        print(f"{row['id']:4s} {row['direction']:11s} {row['where']:11s} {row['from']:>9s} -> {row['to']:9s} "
              f"{row['length_m']:5.1f} m  photos {row['photos']}  tees {row['tees_on_it']}  gaps {row['unseen_gaps_m']}")


if __name__ == "__main__" and len(sys.argv) == 2:
    main(Path(sys.argv[1]).resolve())


def sheets(out: Path) -> None:
    """Blank answer-key sheets: the grid with every photo's pin, equal bays and to scale."""
    from PIL import Image, ImageDraw, ImageFont

    g = grid()
    pins = json.loads(PINS.read_text(encoding="utf-8"))["photos"]
    try:
        font = ImageFont.truetype("arial.ttf", 22)
        small = ImageFont.truetype("arial.ttf", 16)
    except OSError:
        font = small = ImageFont.load_default()

    def blank(xmap, ymap, size, title, background=None):
        img = background if background is not None else Image.new("RGB", size, "white")
        pen = ImageDraw.Draw(img)
        for name, m in g["x"]:
            X = xmap(m)
            pen.line([(X, ymap(g["y"][0][1])), (X, ymap(g["y"][-1][1]))], fill=(150, 150, 150), width=2)
            pen.text((X - 8, ymap(g["y"][-1][1]) - 34), name, fill=(40, 40, 40), font=font)
            pen.text((X - 8, ymap(g["y"][0][1]) + 10), name, fill=(40, 40, 40), font=font)
        for name, m in g["y"]:
            Y = ymap(m)
            pen.line([(xmap(g["x"][0][1]), Y), (xmap(g["x"][-1][1]), Y)], fill=(150, 150, 150), width=2)
            pen.text((xmap(g["x"][0][1]) - 36, Y - 12), name, fill=(40, 40, 40), font=font)
            pen.text((xmap(g["x"][-1][1]) + 14, Y - 12), name, fill=(40, 40, 40), font=font)
        for fname, p in pins.items():
            X, Y = xmap(p["x"]), ymap(p["y"])
            pen.ellipse([X - 9, Y - 9, X + 9, Y + 9], outline=(0, 0, 0), fill=(255, 255, 255), width=2)
            pen.text((X + 11, Y - 9), fname[-7:-4], fill=(0, 0, 0), font=small)  # the photo's file number, as the lab shows it
        pen.text((20, 12), title, fill=(0, 0, 0), font=font)
        pen.text((20, size[1] - 30), "Top of sheet = top of the drawing. Circles: photo pins (file number). Draw fire pipe runs as lines, "
                 "tees as dots, elbows as squares, crosses as X. Unsure: dashed. Not seen: hatch.", fill=(60, 60, 60), font=small)
        return img

    cell = 130.0
    nx, ny = len(g["x"]), len(g["y"])
    size = (int(220 + (nx - 1) * cell), int(240 + (ny - 1) * cell))
    blank(lambda x: 110 + to_schematic(x, g["x"], cell), lambda y: 120 + (ny - 1) * cell - to_schematic(y, g["y"], cell),
          size, "Fire pipe as seen on 21 Sep -- every bay drawn the same size (not to scale)").save(out / "sheet_equal_bays.png")

    # to scale, over a faint copy of the drawing
    import pymupdf

    plan = json.loads(PINS.read_text(encoding="utf-8"))["plan"]
    dpi = 72
    pix = pymupdf.open(SHEET)[0].get_pixmap(dpi=dpi)
    bg = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    k = dpi / 300.0
    ppm = plan["px_per_meter"] * k

    def X(x):
        return plan["origin_px_x"] * k + ppm * x

    def Y(y):
        return plan["origin_px_y"] * k - ppm * y

    x0, x1 = X(g["x"][0][1] - 4), X(g["x"][-1][1] + 4)
    y0, y1 = Y(g["y"][-1][1] + 4), Y(g["y"][0][1] - 4)
    crop = bg.crop((int(x0), int(y0), int(x1), int(y1)))
    scale = 1600 / crop.width
    crop = crop.resize((1600, int(crop.height * scale)))
    crop = Image.blend(Image.new("RGB", crop.size, "white"), crop, 0.35)
    framed = Image.new("RGB", (crop.width + 120, crop.height + 140), "white")
    framed.paste(crop, (60, 70))
    blank(lambda x: 60 + (X(x) - x0) * scale, lambda y: 70 + (Y(y) - y0) * scale, framed.size,
          "Fire pipe as seen on 21 Sep -- to scale, over the finishing layout", framed).save(out / "sheet_to_scale.png")
    print("wrote", out / "sheet_equal_bays.png", "and", out / "sheet_to_scale.png")


if __name__ == "__main__" and len(sys.argv) > 2 and sys.argv[2] == "--sheets":
    sheets(Path(sys.argv[1]).resolve())
