"""Experiment 17: the height of each fire pipe, from the photos that see it.

    python research/pipe_heights.py            # fit, test, write results and figures

WHY THIS WORKS

A straight pipe seen from one photo lands on the plan at a sideways offset
from the camera that is exactly proportional to its height above the camera:
offset = h * k, where k depends only on the image (1 / tan of the angle up to
the pipe, along the direction across the pipe; see pipe_layout.piece_on_plan).
So photo p puts the pipe at

    q = c_p + h * k_p          (c_p: the camera's own position across the pipe)

Two photos that see the same pipe give two such lines; they meet at one h.
With more photos it is a straight-line fit of c_p against k_p. The fit is
only as good as the spread of k: two cameras on OPPOSITE sides of a pipe
(k of opposite sign) pin h down well; two cameras side by side do not.

WHAT IS TESTED

1. Per-pipe heights, where the photos' k spread enough to give one.
2. Leave one photo out: on pipes seen from 3+ photos, does a height fitted on
   the others predict where the left-out photo puts the pipe better than the
   fixed 1.2 m does? This is the honest test -- with two photos the fit is
   exact by construction and says nothing.
3. Bias: copies are grouped at an assumed height before fitting. If the
   fitted heights follow the grouping height, the grouping, not the photos,
   is choosing the answer.

Inputs are experiment 13's tube-filter fire pipe (`layout/results/`) with the
pins and heading corrections it used, so the result applies to the layout in
the report. Output: research/experiments/2026-10-12-17-pipe-heights/.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pipe_layout as pl  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TUBE = ROOT / "research" / "experiments" / "2026-10-07-13-tube-filter" / "layout"
OUT = ROOT / "research" / "experiments" / "2026-10-12-17-pipe-heights"
SYSTEM = "fire pipe"
H_FIXED = 1.2
MIN_SPREAD = 1.0     # a pipe's height is reported only when its photos' k differ by at least this
H_RANGE = (0.3, 3.0) # fitted heights outside this are not a pipe under a slab above a hand-held camera

pl.OUT = TUBE
pl.RESULTS = TUBE / "results"


def lines_with_k(h: float) -> list[dict]:
    """pipe_layout.all_lines at height h, each line also carrying its camera and k.

    k is the line's sideways offset from its camera per metre of height, along
    the grid axis it snaps to. It is found by placing the piece at h = 1 with
    no range limit, where every offset is exactly linear in h.
    """
    data = pl.load_pins()
    runs = pl.load_runs()
    corrections = {}
    for row in json.loads((TUBE / "pieces.json").read_text(encoding="utf-8")):
        if row["pieces"] >= 10 and not math.isnan(row["grid_offset_deg"]):
            corrections[row["photo"]] = row["grid_offset_deg"]
    lines = []
    for name, result in runs.items():
        pin = data["photos"].get(name)
        if not pin:
            continue
        north = pin["north"] - corrections.get(name, 0.0)
        cam = np.array([pin["x"], pin["y"]])
        for p in pl.photo_pieces(result, north):
            if p["system"] != SYSTEM:
                continue
            placed = pl.piece_on_plan(p["v"], h)
            if placed is None:
                continue
            d, p0, t0, t1 = placed
            if t1 - t0 < 0.3:
                continue
            saved, pl.MAX_RANGE_M = pl.MAX_RANGE_M, 1e9
            try:
                d1, q1, a1, b1 = pl.piece_on_plan(p["v"], 1.0)
            finally:
                pl.MAX_RANGE_M = saved
            unit_line = pl.snap([{"d": d1, "p0": q1, "t0": a1, "t1": b1}])
            if not unit_line:
                continue
            ax = unit_line[0]["d"]
            k = float(unit_line[0]["p0"] @ np.array([-ax[1], ax[0]]))
            lines.append({"system": SYSTEM, "photo": name, "seq": pin["seq"], "cam": cam, "k": k,
                          "d": d, "p0": cam + p0, "t0": t0, "t1": t1, "conf": p["conf"]})
    return lines


def per_photo(group: dict) -> list[dict]:
    """One entry per photo in a group: camera offset c and k, length-weighted over its pieces."""
    perp = np.array([-group["d"][1], group["d"][0]])
    by: dict[int, list] = {}
    for m in group["members"]:
        by.setdefault(m["seq"], []).append(m)
    rows = []
    for seq, ms in sorted(by.items()):
        w = np.array([m["t1"] - m["t0"] for m in ms])
        k = float(np.average([m["k"] for m in ms], weights=w))
        rows.append({"seq": seq, "c": float(ms[0]["cam"] @ perp), "k": k, "length": float(w.sum())})
    return rows


def fit(rows: list[dict]) -> tuple[float, float]:
    """Height h and position q with c + h k = q for every photo, least squares."""
    c = np.array([r["c"] for r in rows])
    k = np.array([r["k"] for r in rows])
    a = np.stack([k, -np.ones_like(k)], axis=1)       # h * k - q = -c
    (h, q), *_ = np.linalg.lstsq(a, -c, rcond=None)
    return float(h), float(q)


def spread_of(rows: list[dict]) -> float:
    k = [r["k"] for r in rows]
    return max(k) - min(k)


def groups_at(h_group: float) -> list[dict]:
    groups = pl.fold(pl.merge(pl.snap(lines_with_k(h_group))))
    out = []
    for g in groups:
        rows = per_photo(g)
        if len(rows) < 2:
            continue
        a = g["p0"] + g["t0"] * g["d"]
        b = g["p0"] + g["t1"] * g["d"]
        out.append({"a": a, "b": b, "axis": "x" if abs(g["d"][0]) > 0.5 else "y",
                    "length_m": float(g["t1"] - g["t0"]), "rows": rows, "spread": spread_of(rows)})
    return out


def leave_one_out(groups: list[dict]) -> dict:
    """On pipes seen from 3+ photos: predict each photo's copy from the others."""
    errs_fit, errs_fixed, cases = [], [], []
    for g in groups:
        rows = g["rows"]
        if len(rows) < 3:
            continue
        for j, held in enumerate(rows):
            rest = rows[:j] + rows[j + 1:]
            if spread_of(rest) < MIN_SPREAD:
                continue
            h, q = fit(rest)
            if not H_RANGE[0] <= h <= H_RANGE[1]:
                continue
            q_fixed = float(np.mean([r["c"] + H_FIXED * r["k"] for r in rest]))
            e_fit = abs(held["c"] + h * held["k"] - q)
            e_fixed = abs(held["c"] + H_FIXED * held["k"] - q_fixed)
            errs_fit.append(e_fit)
            errs_fixed.append(e_fixed)
            cases.append({"pipe_mid": [round(float(x), 2) for x in (g["a"] + g["b"]) / 2], "held_out": held["seq"],
                          "h_from_rest": round(h, 2), "miss_fitted_m": round(e_fit, 2), "miss_fixed_m": round(e_fixed, 2)})
    if not cases:
        return {"cases": 0}
    return {"cases": len(cases),
            "median_miss_fitted_m": round(float(np.median(errs_fit)), 2),
            "median_miss_fixed_m": round(float(np.median(errs_fixed)), 2),
            "fitted_better_in": sum(1 for a, b in zip(errs_fit, errs_fixed, strict=True) if a < b),
            "detail": cases}


def summarise(groups: list[dict]) -> dict:
    good = []
    for g in groups:
        if g["spread"] < MIN_SPREAD:
            continue
        h, q = fit(g["rows"])
        g["h"], g["q"] = h, q
        if H_RANGE[0] <= h <= H_RANGE[1]:
            good.append(g)
    hs = np.array([g["h"] for g in good])
    w = np.array([g["length_m"] for g in good])
    out = {
        "pipes_seen_from_2plus": len(groups),
        "pipes_with_enough_spread": sum(1 for g in groups if g["spread"] >= MIN_SPREAD),
        "pipes_with_a_height": len(good),
        "length_with_a_height_m": round(float(w.sum()), 1) if len(good) else 0.0,
    }
    if len(good):
        order = np.argsort(hs)
        cum = np.cumsum(w[order]) / w.sum()
        out.update({
            "median_h_m": round(float(np.median(hs)), 2),
            "length_weighted_median_h_m": round(float(hs[order][np.searchsorted(cum, 0.5)]), 2),
            "quartiles_h_m": [round(float(x), 2) for x in np.percentile(hs, [25, 75])],
        })
    return out


VOTE_MIN_DK = 2.0      # a pair votes only when its photos' k differ by this much (well placed)
VOTE_MIN_OVERLAP = 0.3 # ... and its two pieces overlap along the pipe by this share of the shorter one
                       # (a share, not metres: every length grows with h, so a fixed length would
                       # shut out low heights -- the first run did exactly that)
VOTE_BINS = np.arange(0.2, 3.01, 0.1)


def votes() -> dict:
    """Every pair of pieces that could be one pipe votes for the height that makes them one.

    No grouping first, so no assumed height. Two parallel pieces from photos 1
    and 2 coincide at h = (c2 - c1) / (k1 - k2). Pairs that are really two
    different pipes vote at random; if the pipes share a height, the votes
    pile up there. Each vote is weighted by the pieces' overlap along the pipe
    per metre of height (overlap_m holds that), so that no height is favoured
    just because everything is longer there.
    """
    lines = pl.snap(lines_with_k(H_FIXED))
    for ln in lines:
        ax = ln["d"]
        perp = np.array([-ax[1], ax[0]])
        ln["c"] = float(ln["cam"] @ perp)
        # along-pipe extent per metre of height, relative to the camera (no range limit is
        # applied here, which is why only pairs that overlap generously are used)
        e0 = ln["p0"] + ln["t0"] * ln["d"] - ln["cam"]
        e1 = ln["p0"] + ln["t1"] * ln["d"] - ln["cam"]
        ln["a0"], ln["a1"] = sorted((float(e0 @ ax) / H_FIXED, float(e1 @ ax) / H_FIXED))
        ln["ca"] = float(ln["cam"] @ ax)
    out = {"pairs": [], "opposite": [], "same_side": []}
    for i, p in enumerate(lines):
        for q in lines[i + 1:]:
            if p["seq"] == q["seq"] or abs(float(p["d"] @ q["d"])) < 0.99:
                continue
            dk = p["k"] - q["k"]
            if abs(dk) < VOTE_MIN_DK:
                continue
            h = (q["c"] - p["c"]) / dk
            if not VOTE_BINS[0] <= h <= VOTE_BINS[-1]:
                continue
            lo = max(p["ca"] + h * p["a0"], q["ca"] + h * q["a0"])
            hi = min(p["ca"] + h * p["a1"], q["ca"] + h * q["a1"])
            shorter = h * min(p["a1"] - p["a0"], q["a1"] - q["a0"])
            if shorter <= 0 or (hi - lo) / shorter < VOTE_MIN_OVERLAP:
                continue
            side = "opposite" if p["k"] * q["k"] < 0 else "same_side"
            vote = {"h": round(h, 3), "overlap_m": round((hi - lo) / h, 2), "photos": sorted((p["seq"], q["seq"])), "dk": round(abs(dk), 2)}
            out["pairs"].append(vote)
            out[side].append(vote)
    return out


def vote_peak(vs: list[dict]) -> dict:
    if not vs:
        return {"votes": 0}
    h = np.array([v["h"] for v in vs])
    w = np.array([v["overlap_m"] for v in vs])
    hist, edges = np.histogram(h, bins=VOTE_BINS, weights=w)
    i = int(hist.argmax())
    centre = (edges[i] + edges[i + 1]) / 2
    near = np.abs(h - centre) <= 0.25
    return {"votes": len(vs), "pairs_of_photos": len({tuple(v["photos"]) for v in vs}),
            "peak_bin_m": [round(float(edges[i]), 2), round(float(edges[i + 1]), 2)],
            "weighted_median_within_0.25_of_peak_m": round(float(np.median(h[near])), 2) if near.any() else None,
            "share_of_weight_within_0.25_of_peak": round(float(w[near].sum() / w.sum()), 2),
            "background_share_expected_if_uniform": round(0.5 / (VOTE_BINS[-1] - VOTE_BINS[0]), 2),
            "histogram": [round(float(x), 1) for x in hist]}


def vote_figure(v: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.style.use("ggplot")
    fig, ax = plt.subplots(figsize=(8, 4))
    for key, colour, label in (("opposite", "#c0392b", "photos on opposite sides of the pipe"),
                               ("same_side", "#7f8c8d", "photos on the same side, at different distances")):
        vs = v[key]
        ax.hist([x["h"] for x in vs], bins=VOTE_BINS, weights=[x["overlap_m"] for x in vs], alpha=0.75,
                color=colour, label=f"{label} ({len(vs)} pairs)", edgecolor="white")
    ax.axvline(H_FIXED, color="#333", linestyle="--", linewidth=1, label="1.2 m used so far")
    ax.set_xlabel("height of pipe above camera at which the two pieces coincide (m)")
    ax.set_ylabel("votes, weighted by overlap (m)")
    ax.set_title("Fire pipe: pairs of pieces from two photos, voting for the height", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def figure(groups: list[dict], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.style.use("ggplot")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))
    good = [g for g in groups if "h" in g and H_RANGE[0] <= g["h"] <= H_RANGE[1]]
    ax1.hist([g["h"] for g in good], bins=np.arange(0.3, 3.01, 0.1), weights=[g["length_m"] for g in good],
             color="#c0392b", edgecolor="white")
    ax1.axvline(H_FIXED, color="#333", linestyle="--", linewidth=1)
    ax1.set_xlabel("height of pipe above camera (m)")
    ax1.set_ylabel("metres of pipe")
    ax1.set_title("Fitted height per pipe (spread of k at least %.1f)" % MIN_SPREAD, fontsize=10)
    ax2.scatter([g["spread"] for g in groups if "h" in g], [g["h"] for g in groups if "h" in g],
                s=[8 + 3 * g["length_m"] for g in groups if "h" in g], color="#c0392b", alpha=0.6)
    ax2.axhline(H_FIXED, color="#333", linestyle="--", linewidth=1)
    ax2.set_ylim(0, 3.5)
    ax2.set_xlabel("spread of k among the photos (larger = better placed to measure)")
    ax2.set_ylabel("fitted height (m)")
    ax2.set_title("Height against how well the photos can measure it", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    report = {"system": SYSTEM, "fixed_h_m": H_FIXED, "min_spread": MIN_SPREAD, "by_grouping_height": {}}
    for h_group in (1.0, 1.2, 1.4):
        groups = groups_at(h_group)
        s = summarise(groups)
        s["leave_one_out"] = leave_one_out(groups)
        report["by_grouping_height"][f"{h_group:.1f}"] = s
        loo = s["leave_one_out"]
        print(f"grouped at {h_group:.1f} m: {s['pipes_seen_from_2plus']} pipes seen 2+, "
              f"{s['pipes_with_a_height']} with a height ({s['length_with_a_height_m']} m); "
              f"median h {s.get('median_h_m')}, length-weighted {s.get('length_weighted_median_h_m')}, "
              f"quartiles {s.get('quartiles_h_m')}")
        if loo.get("cases"):
            print(f"   leave one out: {loo['cases']} cases, median miss fitted {loo['median_miss_fitted_m']} m "
                  f"vs fixed {loo['median_miss_fixed_m']} m; fitted better in {loo['fitted_better_in']}")
        if h_group == H_FIXED:
            figure(groups, OUT / "heights.png")
            report["pipes"] = [
                {"a": [round(float(x), 2) for x in g["a"]], "b": [round(float(x), 2) for x in g["b"]],
                 "axis": g["axis"], "length_m": round(g["length_m"], 2), "spread_k": round(g["spread"], 2),
                 "h_m": round(g["h"], 2) if "h" in g else None,
                 "photos": [{"seq": r["seq"], "c": round(r["c"], 2), "k": round(r["k"], 3)} for r in g["rows"]]}
                for g in groups
            ]
    v = votes()
    report["votes"] = {key: vote_peak(v[key]) for key in ("pairs", "opposite", "same_side")}
    for key, s in report["votes"].items():
        print(f"votes, {key}: {s}")
    report["votes"]["all_votes"] = v["pairs"]
    vote_figure(v, OUT / "votes.png")
    (OUT / "heights.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print("wrote", OUT / "heights.json")


if __name__ == "__main__":
    main()
