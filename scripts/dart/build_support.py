"""Build-side helpers for the accelerated SAM 3 provider.

Runs ONLY under .tools/dart-venv, driven by scripts/build_dart_engines.ps1.
Nothing here is imported by the main code. The provider
(src/vision/sam3_trt.py) reads the files this writes into
data/dart/ and knows nothing about DART's Python package.

Three jobs, one subcommand each:

    checkpoint   fetch Meta's raw sam3.pt (the HF cache holds only the
                 transformers-format weights; DART's encoder-decoder export
                 and text encoder need Meta's own module layout)
    text-cache   encode a list of phrases with Meta's text tower into
                 data/dart/text_cache.pt, and write the encoder's position
                 encoding for the working resolution
    manifest     record what was built, with what, on which GPU

WHY THE TEXT CACHE IS A DICT OF PHRASES, NOT DART'S CLASS LIST

    DART caches text as one tensor for one fixed class list and refuses a
    cache whose list differs. Our prompt list is different on every run --
    concept phrasings, whatever somebody typed in the lab -- so the cache
    here is keyed by phrase and grows. Each entry is exactly what DART's
    encoder-decoder engine consumes for one class: the 32 token features
    after the resizer, and the padding mask.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path

CACHE_FORMAT = 1
SAM3_REPO = "facebook/sam3"


def _fail(message: str) -> int:
    print(f"ERROR: {message}", file=sys.stderr)
    return 1


# ---------------------------------------------------------------------------
# checkpoint
# ---------------------------------------------------------------------------

def cmd_checkpoint(args: argparse.Namespace) -> int:
    """Print the local path of Meta's sam3.pt, downloading it if needed."""
    from huggingface_hub import hf_hub_download

    try:
        # config.json too, exactly as DART's own download_ckpt_from_hf does.
        hf_hub_download(repo_id=SAM3_REPO, filename="config.json")
        path = hf_hub_download(repo_id=SAM3_REPO, filename="sam3.pt")
    except Exception as exc:  # noqa: BLE001 - one message beats one traceback
        text = str(exc).lower()
        if any(word in text for word in ("gated", "401", "403", "restricted")):
            return _fail(
                "sam3.pt is gated by Meta. Accept the licence at "
                "https://huggingface.co/facebook/sam3, then `hf auth login` in a "
                "project shell (the token lands in .cache/hf, which this venv "
                "shares) and run the build again."
            )
        return _fail(f"could not download sam3.pt: {exc}")
    print(path)
    return 0


# ---------------------------------------------------------------------------
# text-cache
# ---------------------------------------------------------------------------

def _load_cache(path: Path) -> dict:
    import torch

    if not path.exists():
        return {"format": CACHE_FORMAT, "phrases": {}, "source": {}}
    data = torch.load(path, map_location="cpu", weights_only=False)
    if data.get("format") != CACHE_FORMAT:
        raise RuntimeError(
            f"{path} is format {data.get('format')!r}, this tool writes {CACHE_FORMAT}; "
            "delete it and rebuild"
        )
    data.setdefault("source", {})
    return data


def cmd_text_cache(args: argparse.Namespace) -> int:
    import torch

    phrases: list[str] = json.loads(Path(args.phrases).read_text(encoding="utf-8"))
    phrases = [p.strip().lower() for p in phrases if p.strip()]
    out = Path(args.out)
    pos_out = Path(args.pos_enc_out)
    spatial = args.imgsz // 14
    if args.imgsz % 14:
        return _fail(f"imgsz must be divisible by 14, got {args.imgsz}")

    cache = _load_cache(out)
    missing = [p for p in dict.fromkeys(phrases) if p not in cache["phrases"]]
    need_pos = args.force or not pos_out.exists()
    if not missing and not need_pos:
        print(f"text cache complete: {len(cache['phrases'])} phrases, pos-enc present")
        return 0

    # Loading the model is the expensive part (~20 s), so it happens only
    # when there is something to encode.
    from sam3.model_builder import build_sam3_image_model

    print(f"loading {args.checkpoint} (text tower + position encoding only) ...")
    model = build_sam3_image_model(
        checkpoint_path=args.checkpoint,
        device="cuda",
        eval_mode=True,
        load_from_HF=False,
        enable_segmentation=False,
    )

    with torch.inference_mode():
        for start in range(0, len(missing), 32):
            chunk = missing[start : start + 32]
            answer = model.backbone.forward_text(chunk, device="cuda")
            feats = answer["language_features"]  # (32, N, 256), sequence first
            mask = answer["language_mask"]  # (N, 32), True = padding
            for index, phrase in enumerate(chunk):
                cache["phrases"][phrase] = {
                    "text": feats[:, index, :].float().cpu().contiguous(),
                    "mask": mask[index].bool().cpu().contiguous(),
                }
                cache["source"][phrase] = "meta"
            print(f"  encoded {min(start + 32, len(missing))}/{len(missing)}")

        if need_pos:
            encoder = model.backbone.vision_backbone.position_encoding
            pos = encoder(torch.zeros(1, 1, spatial, spatial, device="cuda"))[0]
            torch.save(
                {
                    "format": CACHE_FORMAT,
                    "imgsz": args.imgsz,
                    "spatial": spatial,
                    "pos_enc": pos.float().cpu().contiguous(),  # (256, P, P)
                },
                pos_out,
            )
            print(f"wrote {pos_out} ({tuple(pos.shape)})")

    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, out)
    print(f"wrote {out}: {len(cache['phrases'])} phrases ({len(missing)} new)")
    return 0


# ---------------------------------------------------------------------------
# manifest
# ---------------------------------------------------------------------------

def _git_sha(repo: Path) -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, cwd=repo, check=True,
        ).stdout.strip()
    except Exception:  # noqa: BLE001 - provenance, not a requirement
        return "unknown"


def _backbone_cosines(log: Path) -> dict[str, float]:
    """The FPN cosine similarities export_hf_backbone.py prints.

    The one number that says whether the FP16 engine is the same model. DART
    measured 0.058 for the naive export and >0.999 for the restructured one;
    anything under 0.99 here means the build is not to be trusted.
    """
    found: dict[str, float] = {}
    if not log.exists():
        return found
    level = None
    for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if stripped.startswith("fpn_") and ":" in stripped:
            level = stripped.split(":")[0]
        elif stripped.startswith("Cosine:") and level:
            with contextlib.suppress(IndexError, ValueError):
                found[level] = float(stripped.split()[1])
            level = None
    return found


def cmd_manifest(args: argparse.Namespace) -> int:
    import tensorrt as trt
    import torch

    out_dir = Path(args.dir)
    files = {}
    for name in args.files:
        path = out_dir / name
        if path.exists():
            stat = path.stat()
            files[name] = {"bytes": stat.st_size, "modified": dt.datetime.fromtimestamp(
                stat.st_mtime, dt.UTC).isoformat()}
        else:
            files[name] = None

    manifest = {
        "built_at": dt.datetime.now(dt.UTC).isoformat(),
        "imgsz": args.imgsz,
        "max_classes": args.max_classes,
        "presence": True,
        "tensorrt": trt.__version__,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "dart_commit": _git_sha(Path(args.dart)),
        "backbone_cosine": _backbone_cosines(Path(args.backbone_log)),
        "files": files,
    }
    (out_dir / "engines.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("checkpoint", help="download Meta's sam3.pt, print its path")

    text = sub.add_parser("text-cache", help="encode phrases with Meta's text tower")
    text.add_argument("--checkpoint", required=True)
    text.add_argument("--phrases", required=True, help="JSON list of phrases")
    text.add_argument("--out", required=True)
    text.add_argument("--pos-enc-out", required=True)
    text.add_argument("--imgsz", type=int, default=1008)
    text.add_argument("--force", action="store_true")

    manifest = sub.add_parser("manifest", help="write engines.json")
    manifest.add_argument("--dir", required=True)
    manifest.add_argument("--dart", required=True)
    manifest.add_argument("--imgsz", type=int, required=True)
    manifest.add_argument("--max-classes", type=int, required=True)
    manifest.add_argument("--backbone-log", required=True)
    manifest.add_argument("--files", nargs="+", required=True)

    args = parser.parse_args()
    return {
        "checkpoint": cmd_checkpoint,
        "text-cache": cmd_text_cache,
        "manifest": cmd_manifest,
    }[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
