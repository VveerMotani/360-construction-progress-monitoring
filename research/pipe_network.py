"""Experiment 16, step 1 of the drawing build: the fire pipe as a network.

    python research/pipe_network.py trace   # from the hand tracing (Trace step), current pins
    python research/pipe_network.py tube    # from the tube-filter layout of the whole walk (experiment 13)

WHAT A REAL DRAWING HOLDS, AND WHY A NETWORK

A sprinkler layout is not a picture of pipes. IS 15105:2002 clause 4.3.3 has
every pipe junction numbered, and every pipe between two junctions given its
bore and length; NFPA 13 (2019) 27.1.3 asks for pipe size and length per piece
and hydraulic reference points at the junctions. Sizing (IS 15105 Tables 18/21/23: sprinklers downstream decide
the bore) and billing (running metres per diameter) both read the layout as a
network of pieces between junctions. So this step turns lines on the plan into
that network -- layer F-SPRN-PIPE of the drawing:

  1. pieces snapped to the building's axes (more than OFF_GRID_DEG off both
     axes: kept apart, not a ceiling run);
  2. collinear pieces joined into runs (pipe_schematic.runs_of);
  3. where an east-west run meets a north-south one: a tee, an elbow or a
     cross (pipe_schematic.joints, ends within 1 m count as meeting);
  4. every run cut at its junctions: NODES are the junctions and the free ends,
     EDGES the pipe between neighbouring nodes, each with its length, the
     photos that saw it and -- once outlets are marked -- how many outlets it
     carries.

Hand-clicked fittings (tee / elbow / cross in the Trace step) are matched to
the nodes found, as a first check of the topology. Connected components are
counted: one installation is one tree, so many pieces means gaps.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pipe_schematic as ps  # noqa: E402
import trace_calibration as tc  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "research" / "experiments" / "2026-10-08-16-pipe-network"
TUBE_LAYOUT = ROOT / "research" / "experiments" / "2026-10-07-13-tube-filter" / "layout" / "layout.json"
OFF_GRID_DEG = 15.0   # a piece further than this from both axes is not a ceiling run
MATCH_M = 1.0         # a hand-clicked fitting this close to a node confirms it
OUTLET_M = 0.6        # an outlet this close to a pipe sits on it
# Pieces of one pipe traced from different photos -- or twice in one photo --
# landed up to about a metre apart (experiments 15, 16); two real parallel fire pipes are at least a sprinkler
# spacing (2 m, IS 15105 cl. 11.2) apart. So hand-traced pieces closer than
# this are one pipe. The tube layout is already merged within 1 m upstream.
COLLINEAR_M = {"trace": 1.0, "tube": 0.5}


# ---------------------------------------------------------------------------
# inputs: lines on the plan, from either source
# ---------------------------------------------------------------------------

def trace_lines() -> tuple[list[dict], list[dict], list[dict], dict]:
    """The hand tracing as plan lines, recomputed from the clicks with the CURRENT pins
    (the runs stored at save time used the pins as they were then)."""
    tc.snapshot()
    trace, pins, names = tc.load()
    h = trace["height_m"] or 1.2
    lines, joints, outlets = [], [], []
    for m in trace["marks"]:
        pin = pins.get(m["frame_id"])
        if not pin:
            continue
        photo = names[m["frame_id"]][-7:-4]
        if m["kind"] in ("pipe", "unsure"):
            for a, b in zip(m["points"], m["points"][1:]):
                part = tc.reliable_part(a, b)
                if part:
                    pa, pb = tc.to_plan(pin, *part[0], h), tc.to_plan(pin, *part[1], h)
                    lines.append({"a": pa.tolist(), "b": pb.tolist(), "photos": [photo], "unsure": m["kind"] == "unsure"})
        elif m["points"][0][1] >= tc.RELIABLE_DEG:
            p = tc.to_plan(pin, *m["points"][0], h)
            (outlets if m["kind"] == "outlet" else joints).append({"kind": m["kind"], "x": float(p[0]), "y": float(p[1]), "photo": photo})
    return lines, joints, outlets, {"height_m": h, "marks": len(trace["marks"]), "saved": trace["updated_at"]}


def tube_lines() -> tuple[list[dict], list[dict], list[dict], dict]:
    layout = json.loads(TUBE_LAYOUT.read_text(encoding="utf-8"))
    lines = [{"a": ln["a"], "b": ln["b"], "photos": [str(p) for p in ln["photos"]], "unsure": False}
             for ln in layout["lines"] if ln["system"] == "fire pipe"]
    return lines, [], [], {"source": str(TUBE_LAYOUT.relative_to(ROOT))}


# ---------------------------------------------------------------------------
# the network
# ---------------------------------------------------------------------------

def on_grid(lines: list[dict]) -> tuple[list[dict], list[dict]]:
    """Pieces near an axis, turned onto it about their midpoint; the rest apart."""
    kept, off = [], []
    for ln in lines:
        a, b = np.array(ln["a"]), np.array(ln["b"])
        ang = math.degrees(math.atan2(b[1] - a[1], b[0] - a[0])) % 180.0
        dev = min(abs(ang), abs(ang - 90), abs(ang - 180))
        if dev > OFF_GRID_DEG:
            off.append(ln)
            continue
        mid, half = (a + b) / 2, np.linalg.norm(b - a) / 2
        if min(abs(ang), abs(ang - 180)) <= OFF_GRID_DEG:   # east-west
            kept.append({**ln, "a": [mid[0] - half, mid[1]], "b": [mid[0] + half, mid[1]]})
        else:
            kept.append({**ln, "a": [mid[0], mid[1] - half], "b": [mid[0], mid[1] + half]})
    return kept, off


def runs_of(lines: list[dict], collinear_m: float, join_gap_m: float = ps.JOIN_GAP_M) -> list[dict]:
    """Pieces of one pipe joined into a run: same axis, within `collinear_m` sideways AND
    within `join_gap_m` along. pipe_schematic.runs_of grouped by sideways offset
    alone, against the group's running mean, so a short piece far along the same
    line dragged the mean and split two copies of one pipe (seen at line 7 here)."""
    items = []
    for ln in lines:
        (ax, ay), (bx, by) = ln["a"], ln["b"]
        if abs(ay - by) <= abs(ax - bx):
            items.append({"axis": "x", "off": (ay + by) / 2, "t0": min(ax, bx), "t1": max(ax, bx), "photos": set(ln["photos"])})
        else:
            items.append({"axis": "y", "off": (ax + bx) / 2, "t0": min(ay, by), "t1": max(ay, by), "photos": set(ln["photos"])})
    parent = list(range(len(items)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, p in enumerate(items):
        for j in range(i + 1, len(items)):
            q = items[j]
            if p["axis"] == q["axis"] and abs(p["off"] - q["off"]) <= collinear_m \
                    and max(p["t0"], q["t0"]) - min(p["t1"], q["t1"]) <= join_gap_m:
                parent[find(i)] = find(j)
    groups: dict[int, list[dict]] = {}
    for i, it in enumerate(items):
        groups.setdefault(find(i), []).append(it)
    runs = []
    for g in groups.values():
        g.sort(key=lambda it: it["t0"])
        w = np.array([max(it["t1"] - it["t0"], 1e-3) for it in g])
        run = {"axis": g[0]["axis"], "off": float(np.average([it["off"] for it in g], weights=w)),
               "t0": g[0]["t0"], "t1": max(it["t1"] for it in g), "photos": set().union(*(it["photos"] for it in g)),
               "gaps": [], "parts": g}
        reach = g[0]["t1"]
        for it in g[1:]:
            if it["t0"] > reach:
                run["gaps"].append((reach, it["t0"]))
            reach = max(reach, it["t1"])
        runs.append(run)
    for i, r in enumerate(runs):
        r["id"] = f"R{i + 1}"
        r["length_m"] = r["t1"] - r["t0"]
        r["seen_twice"] = len(r["photos"]) >= 2
    return runs


def build(lines: list[dict], collinear_m: float) -> dict:
    straight, off = on_grid(lines)
    runs = runs_of(straight, collinear_m)
    jts = ps.joints(runs)
    nodes: dict[tuple[float, float], dict] = {}

    def node(x: float, y: float, kind: str) -> str:
        key = (round(x / 0.05) * 0.05, round(y / 0.05) * 0.05)
        if key not in nodes:
            nodes[key] = {"id": f"N{len(nodes) + 1}", "x": float(x), "y": float(y), "kind": kind}
        elif kind != "end":
            nodes[key]["kind"] = kind
        return nodes[key]["id"]

    edges = []
    for r in runs:
        # positions along the run where it is cut: its junctions, and its ends --
        # an end within reach of a junction is moved onto it, so pieces meet
        cuts = []
        for j in jts:
            if r["id"] in j["runs"]:
                t = j["x"] if r["axis"] == "x" else j["y"]
                cuts.append((t, j["kind"]))
        ts = [t for t, _ in cuts]
        for end in (r["t0"], r["t1"]):
            if not any(abs(end - t) <= ps.END_M for t in ts):
                cuts.append((end, "end"))
        cuts.sort()
        ids = []
        for t, kind in cuts:
            x, y = (t, r["off"]) if r["axis"] == "x" else (r["off"], t)
            ids.append((t, node(x, y, kind)))
        for (t0, a), (t1, b) in zip(ids, ids[1:]):
            if a == b or t1 - t0 < 0.05:
                continue
            edges.append({"id": f"E{len(edges) + 1}", "a": a, "b": b, "run": r["id"], "length_m": round(t1 - t0, 2),
                          "photos": sorted(r["photos"]), "outlets": 0})
    node_list = list(nodes.values())
    # degree, and connected components (one installation is one tree)
    deg = {n["id"]: 0 for n in node_list}
    parent = {n["id"]: n["id"] for n in node_list}

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for e in edges:
        deg[e["a"]] += 1
        deg[e["b"]] += 1
        parent[find(e["a"])] = find(e["b"])
    for n in node_list:
        n["degree"] = deg[n["id"]]
        n["component"] = find(n["id"])
    comps = sorted({n["component"] for n in node_list})
    cid = {c: i + 1 for i, c in enumerate(comps)}
    for n in node_list:
        n["component"] = cid[n["component"]]
    comp_of = {n["id"]: n["component"] for n in node_list}
    for e in edges:
        e["component"] = comp_of[e["a"]]
    return {"nodes": node_list, "edges": edges, "runs": len(runs), "off_grid_pieces": len(off), "components": len(comps)}


def attach(net: dict, hand_joints: list[dict], outlets: list[dict]) -> dict:
    """Hand-clicked fittings matched to nodes; outlets put on the nearest edge."""
    by_id = {n["id"]: n for n in net["nodes"]}
    matches = []
    for j in hand_joints:
        best = min(net["nodes"], key=lambda n: math.hypot(n["x"] - j["x"], n["y"] - j["y"]), default=None)
        d = math.hypot(best["x"] - j["x"], best["y"] - j["y"]) if best else float("inf")
        if best and d <= MATCH_M:
            best.setdefault("hand", []).append(j["kind"])
            matches.append({"hand": j["kind"], "found": best["kind"], "node": best["id"], "dist_m": round(d, 2), "photo": j["photo"]})
        else:
            matches.append({"hand": j["kind"], "found": None, "node": None, "dist_m": round(d, 2), "photo": j["photo"]})
    placed = 0
    for o in outlets:
        best, bd = None, OUTLET_M
        for e in net["edges"]:
            a, b = by_id[e["a"]], by_id[e["b"]]
            pa, pb, p = np.array([a["x"], a["y"]]), np.array([b["x"], b["y"]]), np.array([o["x"], o["y"]])
            t = np.clip((p - pa) @ (pb - pa) / max((pb - pa) @ (pb - pa), 1e-9), 0, 1)
            d = float(np.linalg.norm(p - (pa + t * (pb - pa))))
            if d <= bd:
                best, bd = e, d
        if best:
            best["outlets"] += 1
            placed += 1
    net["hand_joints"] = matches
    net["outlets_marked"] = len(outlets)
    net["outlets_on_pipe"] = placed
    return net


def summary(net: dict) -> dict:
    kinds: dict[str, int] = {}
    for n in net["nodes"]:
        kinds[n["kind"]] = kinds.get(n["kind"], 0) + 1
    hj = net.get("hand_joints", [])
    return {
        "edges": len(net["edges"]),
        "length_m": round(sum(e["length_m"] for e in net["edges"]), 1),
        "length_seen_twice_m": round(sum(e["length_m"] for e in net["edges"] if len(e["photos"]) >= 2), 1),
        "nodes_by_kind": kinds,
        "components": net["components"],
        "largest_component_share": round(max((sum(e["length_m"] for e in net["edges"] if e["component"] == c)
                                              for c in range(1, net["components"] + 1)), default=0)
                                         / max(sum(e["length_m"] for e in net["edges"]), 1e-9), 2),
        "off_grid_pieces": net["off_grid_pieces"],
        "hand_fittings": len(hj),
        "hand_fittings_on_a_node": sum(1 for m in hj if m["found"]),
        "hand_fittings_same_kind": sum(1 for m in hj if m["found"] == m["hand"]),
        "outlets_marked": net.get("outlets_marked", 0),
        "outlets_on_pipe": net.get("outlets_on_pipe", 0),
    }


# ---------------------------------------------------------------------------
# figure
# ---------------------------------------------------------------------------

def draw(net: dict, path: Path, title: str, pins: dict | None = None, focus: tuple | None = None) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_id = {n["id"]: n for n in net["nodes"]}
    g = ps.grid()
    fig, ax = plt.subplots(figsize=(8, 8))
    for _, m in g["x"]:
        ax.axvline(m, color="#ccc", lw=0.6)
    for _, m in g["y"]:
        ax.axhline(m, color="#ccc", lw=0.6)
    # the column grid's names on the axes: positions read as "between 9 and 10"
    ax.set_xticks([m for _, m in g["x"]], [n for n, _ in g["x"]], fontsize=7)
    ax.set_yticks([m for _, m in g["y"]], [n for n, _ in g["y"]], fontsize=7)
    cmap = plt.get_cmap("tab10")
    for e in net["edges"]:
        a, b = by_id[e["a"]], by_id[e["b"]]
        ax.plot([a["x"], b["x"]], [a["y"], b["y"]], color=cmap((e["component"] - 1) % 10),
                lw=3 if len(e["photos"]) >= 2 else 1.6, solid_capstyle="butt")
        if e["outlets"]:
            ax.text((a["x"] + b["x"]) / 2, (a["y"] + b["y"]) / 2, str(e["outlets"]), fontsize=7, color="k")
    style = {"tee": ("o", "#1d4ed8", 7), "elbow": ("s", "#1d4ed8", 6), "cross": ("X", "#444", 7), "end": ("o", "white", 4)}
    for n in net["nodes"]:
        mk, col, size = style[n["kind"]]
        ax.plot(n["x"], n["y"], mk, color=col, mec="k", ms=size, mew=0.8, zorder=5)
        if n.get("hand"):
            ax.plot(n["x"], n["y"], "o", mfc="none", mec="#16a34a", ms=13, mew=2, zorder=6)
    if pins:
        for p, pin in pins.items():
            ax.plot(pin["x"], pin["y"], "^", color="#f59e0b", mec="k", ms=7)
            ax.annotate(p, (pin["x"], pin["y"]), xytext=(3, 3), textcoords="offset points", fontsize=7)
    if focus:
        ax.set_xlim(focus[0], focus[1])
        ax.set_ylim(focus[2], focus[3])
    ax.set_aspect("equal")
    ax.set_title(title + "\nthick = seen from 2+ photos; dot = tee, square = elbow, X = cross, white = end; "
                 "green ring = a fitting you clicked; colour = connected piece", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print("wrote", path)


def main(source: str) -> None:
    EXPERIMENT.mkdir(parents=True, exist_ok=True)
    lines, hand, outlets, meta = trace_lines() if source == "trace" else tube_lines()
    net = attach(build(lines, COLLINEAR_M[source]), hand, outlets)
    s = summary(net)
    out = {"source": source, "meta": meta, "summary": s, **net}
    (EXPERIMENT / f"network_{source}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(s, indent=1))
    pins = focus = None
    if source == "trace":
        trace, allpins, names = tc.load()
        used = {m["frame_id"] for m in trace["marks"]}
        pins = {names[f][-7:-4]: allpins[f] for f in used if f in allpins}
        xs = [p["x"] for p in pins.values()]
        ys = [p["y"] for p in pins.values()]
        focus = (min(xs) - 12, max(xs) + 4, min(ys) - 10, max(ys) + 10)
    draw(net, EXPERIMENT / f"network_{source}.png", f"fire pipe network from the {'hand tracing' if source == 'trace' else 'tube filter, whole walk'}",
         pins, focus)


if __name__ == "__main__":
    main(sys.argv[1])
