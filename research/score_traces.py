"""Experiment 11: SAM 3 fire-pipe masks scored against hand traces, per photo.

    python research/score_traces.py

The traces are centrelines in full-resolution pixels of the equirectangular
photo (research/experiments/2026-10-02-11-hand-traces/traces_*.json). The
masks are the outlines SAM 3 returned in experiment 10 (runs.json there), in
image longitude and latitude. Both are rasterised onto the same 1/4-scale
canvas and scored by distance:

  recall    = share of traced centreline pixels within TOL of a mask pixel
  precision = share of mask pixels (ceiling only) within TOL of a trace pixel

TOL is 20 full-resolution pixels, about 0.6 degrees, which is the width of a
32 mm pipe at 3 m. A loose match on purpose: the trace is a hand-drawn
centreline, the mask is an outline, and the question is whether the model
found the pipe at all, not whether its edge is exact. Precision is measured
on the ceiling half only, the same region the traces cover; everything
below 8 degrees of latitude is ignored on both sides. "Sure" traces and
unsure ones are reported apart.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TRACES = ROOT / "research" / "experiments" / "2026-10-02-11-hand-traces"
RUNS = json.loads((ROOT / "research" / "experiments" / "2026-10-01-10-whole-walk-detection" / "runs.json").read_text(encoding="utf-8"))
SCALE = 4
TOL_PX = 20
MIN_LAT_DEG = 8.0


def to_px(lon: float, lat: float, w: int, h: int) -> tuple[float, float]:
    return ((lon + 180.0) / 360.0 * w, (90.0 - lat) / 180.0 * h)


def mask_canvas(result: dict, w: int, h: int, label: str = "fire pipe") -> np.ndarray:
    canvas = np.zeros((h // SCALE, w // SCALE), np.uint8)
    for find in result["detections"]:
        if find["label"] != label:
            continue
        for key in ("outlines", "grown_outlines"):
            o = find.get(key)
            if isinstance(o, str):
                o = json.loads(o)
            if not o:
                continue
            if not isinstance(o[0][0], list):
                o = [o]
            for poly in o:
                pts = [to_px(p[0], p[1], w, h) for p in poly]
                # split at the seam so a loop across the edge is not drawn as a bar
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


def trace_canvas(traces: dict, w: int, h: int, sure_only: bool | None) -> np.ndarray:
    canvas = np.zeros((h // SCALE, w // SCALE), np.uint8)
    for run in traces["runs"]:
        if sure_only is True and not run["sure"]:
            continue
        if sure_only is False and run["sure"]:
            continue
        pts = np.array([[x / SCALE, y / SCALE] for x, y in run["pts"]], np.int32)
        cv2.polylines(canvas, [pts], False, 255, 1)
    return canvas


def ceiling_only(canvas: np.ndarray, h: int) -> np.ndarray:
    cut = int((90.0 - MIN_LAT_DEG) / 180.0 * h / SCALE)
    out = canvas.copy()
    out[cut:, :] = 0
    return out


def near(a: np.ndarray, b: np.ndarray) -> float:
    """Share of a's pixels within TOL of any b pixel."""
    if a.sum() == 0:
        return float("nan")
    k = 2 * (TOL_PX // SCALE) + 1
    grown = cv2.dilate(b, np.ones((k, k), np.uint8))
    return float(((a > 0) & (grown > 0)).sum() / (a > 0).sum())


def main() -> None:
    rows = []
    for tf in sorted(TRACES.glob("traces_*.json")):
        traces = json.loads(tf.read_text(encoding="utf-8"))
        name = traces["photo"]
        result = json.loads((ROOT / "data" / "lab" / RUNS[name] / "result.json").read_text(encoding="utf-8"))
        w, h = traces["width"], traces["height"]
        mask = ceiling_only(mask_canvas(result, w, h), h)
        t_sure = ceiling_only(trace_canvas(traces, w, h, True), h)
        t_unsure = ceiling_only(trace_canvas(traces, w, h, False), h)
        t_all = ceiling_only(trace_canvas(traces, w, h, None), h)
        mask_edge = cv2.Canny(mask, 50, 150)  # score the outline, not the fill, against a centreline
        row = {
            "photo": name[-7:-4],
            "recall_sure": near(t_sure, mask),
            "recall_unsure": near(t_unsure, mask),
            "recall_all": near(t_all, mask),
            "precision": near(mask_edge, t_all),
            "trace_px_sure": int((t_sure > 0).sum()),
            "trace_px_unsure": int((t_unsure > 0).sum()),
            "mask_px": int((mask > 0).sum()),
            "finds": sum(1 for d in result["detections"] if d["label"] == "fire pipe"),
        }
        rows.append(row)
        # a picture: trace in green, mask fill in red, overlap yellow
        pic = np.zeros((h // SCALE, w // SCALE, 3), np.uint8)
        pic[..., 2] = mask
        pic[..., 1] = cv2.dilate(t_all, np.ones((3, 3), np.uint8))
        cut = int((90.0 - MIN_LAT_DEG) / 180.0 * h / SCALE)
        cv2.imwrite(str(TRACES / f"score_{row['photo']}.png"), pic[:cut])
        print(f"{row['photo']}  recall sure {row['recall_sure']:.2f}  unsure {row['recall_unsure']:.2f}  all {row['recall_all']:.2f}   precision {row['precision']:.2f}   finds {row['finds']}")
    (TRACES / "scores.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    r = np.nanmean([x["recall_sure"] for x in rows]); p = np.nanmean([x["precision"] for x in rows])
    print(f"mean  recall(sure) {r:.2f}  precision {p:.2f}")


if __name__ == "__main__":
    main()
