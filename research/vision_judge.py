"""Experiment 20: can a vision model tell pipe from paint, judging the tube filter's lines?

    python research/vision_judge.py prepare   # numbered candidate sheets for the 4 traced photos
    python research/vision_judge.py score     # score labels.json against the hand traces

The tube filter (experiment 13) proposes thin red lines, exact to the pixel. Here
each of its final lines, per flat view, becomes one numbered CANDIDATE drawn on
the view; a vision model (in this pilot: Claude, looking at the sheets in a chat
session, no API) labels each one pipe / paint / cable / other. The lines labelled
pipe are scored line for line against experiment 11's hand traces with the same
yardstick as experiment 13 (20 px).

Proposals stay pixel-exact; the model only says WHAT each one is.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paint_filter as pf  # noqa: E402
import tube_filter as tf  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "research" / "experiments" / "2026-10-13-20-vision-judge-pilot"
SHEETS = EXPERIMENT / "sheets"
SHEET_PX = 1400          # the view as shown to the model
MIN_CAND_PX = 20         # face px of crest: shorter candidates are not shown (and not kept)
PALETTE = [(0, 255, 255), (255, 0, 255), (0, 255, 0), (255, 128, 0), (0, 128, 255),
           (255, 255, 0), (128, 0, 255), (0, 200, 120)]


def final_crests(name: str, rd: dict) -> list[np.ndarray]:
    """Per face, the crest indices experiment 13's final variant keeps (paint cut included)."""
    result = json.loads((ROOT / "data" / "lab" / tf.RUNS[name] / "result.json").read_text(encoding="utf-8"))
    w, h = result["width"], result["height"]
    paint = pf.paint_mask(name)
    px = ((rd["lon"] + 180.0) / 360.0 * w / tf.SCALE).astype(int) % paint.shape[1]
    py = np.clip(((90.0 - rd["lat"]) / 180.0 * h / tf.SCALE).astype(int), 0, paint.shape[0] - 1)
    not_paint = ~paint[py, px]
    return [sel[not_paint[sel]] for sel in tf.centrelines(rd, tf.HI)]


def candidates(name: str) -> list[dict]:
    """Crests of each face split into connected pieces: one candidate each."""
    rd = tf.ridges(name)
    out = []
    for sel in final_crests(name, rd):
        if not len(sel):
            continue
        fi = int(rd["face"][sel[0]])
        img = np.zeros((tf.FACE_PX, tf.FACE_PX), np.uint8)
        img[rd["i"][sel], rd["j"][sel]] = 1
        n, lab = cv2.connectedComponents(cv2.dilate(img, np.ones((7, 7), np.uint8)), connectivity=8)
        at = lab[rd["i"][sel], rd["j"][sel]]
        for k in range(1, n):
            idx = sel[at == k]
            if len(idx) >= MIN_CAND_PX:
                out.append({"face": fi, "idx": idx})
    return out


def view_image(name: str, fi: int) -> np.ndarray:
    im = cv2.imread(str(ROOT / "data" / "lab" / tf.RUNS[name] / "image.jpg"))
    H, W = im.shape[:2]
    _, fwd, r, u = tf.faces()[fi]
    d = tf.face_dirs(fwd, r, u)
    lon, lat = tf.lonlat(d)
    mx = ((lon + 180.0) / 360.0 * W).astype(np.float32)
    my = ((90.0 - lat) / 180.0 * H).astype(np.float32)
    return cv2.remap(im, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)


def cmd_prepare(every_photo: bool = False, out_dir: Path | None = None) -> None:
    """Numbered sheets for the traced photos (or, with every_photo, the whole walk)."""
    sheets = out_dir / "sheets" if out_dir else SHEETS
    sheets.mkdir(parents=True, exist_ok=True)
    index = {}
    names = sorted(tf.RUNS) if every_photo else pf.traced()
    for name in names:
        rd = tf.ridges(name)
        cands = candidates(name)
        stem = name[-7:-4]
        index[stem] = []
        number = 0
        for fi in sorted({c["face"] for c in cands}):
            view = view_image(name, fi)
            plain = cv2.resize(view, (SHEET_PX, SHEET_PX), interpolation=cv2.INTER_AREA)
            marked = plain.copy()
            k = SHEET_PX / tf.FACE_PX
            for c in [c for c in cands if c["face"] == fi]:
                number += 1
                colour = PALETTE[number % len(PALETTE)]
                ii, jj = rd["i"][c["idx"]], rd["j"][c["idx"]]
                for y, x in zip((ii * k).astype(int), (jj * k).astype(int)):
                    cv2.circle(marked, (int(x), int(y)), 2, colour, -1)
                m = len(ii) // 2
                order = np.argsort(jj if np.ptp(jj) >= np.ptp(ii) else ii)
                y, x = int(ii[order[m]] * k), int(jj[order[m]] * k)
                cv2.putText(marked, str(number), (x + 6, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 5)
                cv2.putText(marked, str(number), (x + 6, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.9, colour, 2)
                index[stem].append({"n": number, "face": fi, "crest_px": int(len(ii))})
            face = tf.faces()[fi][0]
            cv2.imwrite(str(sheets / f"{stem}_{face}_marked.jpg"), marked, [cv2.IMWRITE_JPEG_QUALITY, 88])
            cv2.imwrite(str(sheets / f"{stem}_{face}_plain.jpg"), plain, [cv2.IMWRITE_JPEG_QUALITY, 88])
        print(stem, len(cands), "candidates")
    ((out_dir or EXPERIMENT) / "candidates.json").write_text(json.dumps(index, indent=1), encoding="utf-8")


def cmd_score(path: str | None = None) -> None:
    """Score a labels file (default: this pilot's own) against the hand traces."""
    src = Path(path) if path else EXPERIMENT / "labels.json"
    labels = json.loads(src.read_text(encoding="utf-8"))
    rows = []
    for name in pf.traced():
        stem = name[-7:-4]
        rd = tf.ridges(name)
        cands = candidates(name)
        traces = json.loads((pf.TRACES / f"traces_{stem}.json").read_text(encoding="utf-8"))
        w, h = traces["width"], traces["height"]
        lab = labels.get(stem, {})
        all_idx = np.concatenate([c["idx"] for c in cands])
        pipe_idx = np.concatenate([c["idx"] for n, c in enumerate(cands, 1) if lab.get(str(n)) == "pipe"]
                                  or [np.zeros(0, int)])
        base = tf.line_score(name, tf.canvas_of(rd, all_idx, w, h))
        judged = tf.line_score(name, tf.canvas_of(rd, pipe_idx, w, h))
        counts = {k: sum(1 for v in lab.values() if v == k) for k in ("pipe", "paint", "cable", "other")}
        rows.append({"photo": stem, "candidates": len(cands), **counts,
                     "base_recall": round(base["recall_sure"], 3), "base_precision": round(base["precision"], 3),
                     "judged_recall": round(judged["recall_sure"], 3),
                     "judged_precision": round(judged["precision"], 3)})
        print(rows[-1])
    out = EXPERIMENT / "scores.json" if path is None else src.with_name("scores.json")
    out.write_text(json.dumps(rows, indent=1), encoding="utf-8")


if __name__ == "__main__":
    if sys.argv[1] == "score":
        cmd_score(sys.argv[2] if len(sys.argv) > 2 else None)
    elif sys.argv[1] == "prepare-walk":
        cmd_prepare(every_photo=True, out_dir=ROOT / "research" / "experiments" / "2026-10-13-23-judge-whole-walk")
    else:
        {"prepare": cmd_prepare}[sys.argv[1]]()
