"""Experiment 23: the pipe plan rebuilt from judged lines only.

    python research/judge_layout.py

Haiku labelled every tube-filter candidate on every photo of the walk (subagents, by
hand-off; the four traced photos reuse experiment 20's blind Haiku run). Here
each straight segment the layout uses (tube_filter.segments) takes the label
most of its crests carry, and the layout (pipe_layout) and network
(pipe_network) are rebuilt, unchanged, from the kept segments:

    pipe only            what the judge called fire sprinkler pipe
    paint removed        everything except paint (conduit kept), for comparison

Heading corrections are experiment 13's (its pieces.json is copied), so the
only difference from today's plan is which lines go in.
"""

from __future__ import annotations

import json
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pipe_layout as pl  # noqa: E402
import pipe_network as pn  # noqa: E402
import tube_filter as tf  # noqa: E402
import vision_judge as vj  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "research" / "experiments" / "2026-10-13-23-judge-whole-walk"
EXP13 = ROOT / "research" / "experiments" / "2026-10-07-13-tube-filter" / "layout"
EXP20_HAIKU = ROOT / "research" / "experiments" / "2026-10-13-20-vision-judge-pilot" / "models" / "haiku" / "labels.json"
MATCH_DEG = 0.4  # a segment point belongs to the nearest labelled crest within this angle

VARIANTS = {"all_kept": {"pipe", "cable", "paint", "other", "unjudged"}, "pipe_only": {"pipe"}, "paint_removed": {"pipe", "cable", "other"}}


def all_labels() -> dict:
    out = json.loads(EXP20_HAIKU.read_text(encoding="utf-8"))
    for part in ("haiku_a", "haiku_b", "haiku_c", "haiku_d"):
        out.update(json.loads((EXP / part / "labels.json").read_text(encoding="utf-8")))
    return out


def unit(lon, lat):
    lo, la = np.radians(lon), np.radians(lat)
    return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], -1)


def segment_labels(name: str, labels: dict) -> tuple[list, list[str]]:
    rd = tf.ridges(name)
    idx, lab = [], []
    for n, c in enumerate(vj.candidates(name), 1):
        idx.append(c["idx"])
        lab += [labels.get(str(n), "unjudged")] * len(c["idx"])
    idx = np.concatenate(idx)
    crest = unit(rd["lon"][idx], rd["lat"][idx]).astype(np.float32)
    cos_min = np.cos(np.radians(MATCH_DEG))
    segs = tf.segments(name)
    out = []
    for seg in segs:
        pts = np.array(seg)
        sim = unit(pts[:, 0], pts[:, 1]).astype(np.float32) @ crest.T  # cosine to every crest
        i = sim.argmax(1)
        votes = Counter(lab[j] for j, c in zip(i, sim[np.arange(len(i)), i]) if c >= cos_min)
        out.append(votes.most_common(1)[0][0] if votes else "unjudged")
    return segs, out


def main() -> None:
    labels = all_labels()
    print("photos labelled:", len(labels), " candidates:", sum(len(v) for v in labels.values()))
    per_photo = {}
    seg_cache = {}
    for name in sorted(tf.RUNS):
        segs, labs = segment_labels(name, labels[name[-7:-4]])
        seg_cache[name] = (segs, labs)
        per_photo[name[-7:-4]] = dict(Counter(labs))
    totals = Counter()
    for c in per_photo.values():
        totals.update(c)
    print("segments by label:", dict(totals))

    summaries = {}
    for variant, keep in VARIANTS.items():
        out_dir = EXP / variant
        (out_dir / "results").mkdir(parents=True, exist_ok=True)
        shutil.copy(EXP13 / "pieces.json", out_dir / "pieces.json")  # same heading corrections
        for name, (segs, labs) in seg_cache.items():
            res = json.loads((EXP13 / "results" / f"{name[:-4]}.json").read_text(encoding="utf-8"))
            fire = [s for s, lab in zip(segs, labs) if lab in keep]
            res["detections"] = [d for d in res["detections"] if d["label"] != "fire pipe"] + \
                [{"label": "fire pipe", "confidence": 0.9, "outlines": fire}]
            (out_dir / "results" / f"{name[:-4]}.json").write_text(json.dumps(res), encoding="utf-8")
        pl.OUT, pl.RESULTS = out_dir, out_dir / "results"
        print(f"\n--- {variant}")
        pl.layout()
        pn.TUBE_LAYOUT, pn.EXPERIMENT = out_dir / "layout.json", out_dir
        pn.main("tube")
        net = json.loads((out_dir / "network_tube.json").read_text(encoding="utf-8"))
        summaries[variant] = net["summary"]
    before = json.loads((ROOT / "research" / "experiments" / "2026-10-08-16-pipe-network" / "network_tube.json")
                        .read_text(encoding="utf-8"))["summary"]
    summaries = {"before (exp 16)": before, **summaries}
    (EXP / "summary.json").write_text(json.dumps({"segments_by_label": dict(totals), "per_photo": per_photo,
                                                  "networks": summaries}, indent=1), encoding="utf-8")
    keys = ("edges", "length_m", "length_seen_twice_m", "components", "largest_component_share")
    print("\n" + " | ".join(["variant"] + list(keys) + ["tee", "cross", "end"]))
    for k, s in summaries.items():
        kinds = s["nodes_by_kind"]
        print(" | ".join([k] + [str(s[x]) for x in keys] + [str(kinds.get(x, 0)) for x in ("tee", "cross", "end")]))


if __name__ == "__main__":
    main()
