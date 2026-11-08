"""SAM 3, the same weights, ten times faster: the DART TensorRT provider.

Phase 15.9. `Sam3Provider` (concept.py) runs Meta's SAM 3 through
transformers, one forward pass per phrase per view, at roughly half a second
per phrase per photograph. DART (Detect Anything in Real Time, Turkcan 2026,
arXiv 2603.11441, third_party/DART) showed that the same checkpoint
splits into two TensorRT FP16 engines -- the ViT-H backbone, and the 6+6
encoder-decoder with the scoring head -- with the text encoder cached, and
that the split loses nothing measurable (55.8 vs 56.1 AP on COCO).

This module is the run-time half of that. The engines and the prompt cache
are BUILT by scripts/build_dart_engines.ps1 in DART's own environment; here
they are only loaded. Nothing in this file imports DART's package.

WHAT IS DIFFERENT FROM THE EAGER PROVIDER, AND WHAT IS NOT

    Same weights, same resolution (1008, what SAM 3 was trained at), same
    scoring rule -- sigmoid(logit) x sigmoid(presence), which is what
    transformers' post-processing computes for the eager path. The presence
    head is exported on purpose: without it the two providers would score on
    different scales and no threshold would mean the same thing on both.

    BOXES ONLY. DART's detection-only mode skips the mask decoder, which is
    where the speed comes from; outlines stay on the eager `sam3` entry.

    The provider satisfies `DetectionProvider` (boxes in one view's pixels),
    so everything downstream -- to_sphere, merge_duplicates, the flat-photo
    path, the bench -- is shared and unchanged.

THE PROMPT CACHE, AND WHERE A NEW PHRASE COMES FROM

    DART caches text for one fixed class list. Ours is a dictionary keyed by
    phrase, because the prompt list differs on every run. The build script
    fills it from Meta's own text tower for every phrasing in concepts.py;
    a phrase nobody has asked before is encoded here, in-process, by the
    transformers port of the SAME text tower (only its weights are loaded --
    a third of the model, not the 3.4 GB whole) and appended to the cache so
    it is paid for once. `source` records which encoder produced each entry,
    because "are the two towers numerically the same" is a claim the bench
    has to be able to check, not assume.

PIPELINING

    DART overlaps backbone(frame t+1) with encoder-decoder(frame t) on two
    CUDA streams, so a frame costs max(backbone, enc-dec) rather than the
    sum. Our "video" is the stream of rendered perspective views -- sixteen
    per panorama, across every panorama of a walk -- and `detect_stream`
    keeps that pipeline full across panorama boundaries. Two execution
    contexts on ONE engine, not two engines: contexts share the weights and
    an 8 GB laptop GPU does not want the backbone resident twice.
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from vision.detect import DetectionError, RawDetection

DEFAULT_IMAGE_SIZE = 1008
TEXT_TOKENS = 32
FEATURE_DIM = 256
PATCH = 14
CACHE_FORMAT = 1
# Per-class NMS at the IoU DART's detection-only mode uses. DETR-style
# decoders emit the same object from several of their 200 queries; the
# eager path's mask fusion absorbs those, the box path needs this.
NMS_IOU = 0.7
BUILD_COMMAND = ".\\scripts\\build_dart_engines.ps1"
HF_WEIGHTS = "facebook/sam3"


# ---------------------------------------------------------------------------
# Where the products are, and whether they are usable
# ---------------------------------------------------------------------------

def engine_dir() -> Path:
    """`DART_DIR`, if set. The fallback is the same place."""
    configured = os.environ.get("DART_DIR")
    if configured:
        return Path(configured)
    return Path(__file__).resolve().parents[2] / "data" / "dart"


@dataclass(frozen=True)
class EnginePaths:
    backbone: Path
    enc_dec: Path
    text_cache: Path
    pos_enc: Path
    manifest: Path

    @classmethod
    def locate(cls, directory: Path | None = None) -> EnginePaths:
        root = directory or engine_dir()
        manifest = root / "engines.json"
        imgsz, max_classes = DEFAULT_IMAGE_SIZE, 4
        if manifest.exists():
            try:
                data = json.loads(manifest.read_text(encoding="utf-8"))
                imgsz = int(data.get("imgsz", imgsz))
                max_classes = int(data.get("max_classes", max_classes))
            except (ValueError, TypeError, json.JSONDecodeError):
                pass
        return cls(
            backbone=root / f"hf_backbone_{imgsz}_fp16.engine",
            enc_dec=root / f"enc_dec_{imgsz}_c{max_classes}_fp16.engine",
            text_cache=root / "text_cache.pt",
            pos_enc=root / f"pos_enc_{imgsz}.pt",
            manifest=manifest,
        )

    def missing(self) -> list[Path]:
        return [p for p in (self.backbone, self.enc_dec, self.pos_enc) if not p.exists()]


@dataclass(frozen=True)
class EngineStatus:
    ok: bool
    message: str
    details: dict[str, Any] = field(default_factory=dict)


def _missing_message(paths: EnginePaths, missing: list[Path]) -> str:
    names = ", ".join(p.name for p in missing)
    return (
        f"The accelerated SAM 3 engines are not built ({names} missing from "
        f"{paths.backbone.parent}). Run `{BUILD_COMMAND}` in a project shell -- it "
        "takes about half an hour once per GPU -- or pick the plain `sam3` model, "
        "which needs no build."
    )


def engine_status(directory: Path | None = None) -> EngineStatus:
    """Can this provider run here? A sentence either way, never a traceback.

    Checked in order of how cheaply each failure can be explained: files on
    disk, the tensorrt runtime, then that the engines actually deserialise
    in THIS TensorRT -- the check that catches a version upgrade, since an
    engine only loads in the version that built it.
    """
    paths = EnginePaths.locate(directory)
    missing = paths.missing()
    if missing:
        return EngineStatus(False, _missing_message(paths, missing))

    try:
        import tensorrt as trt
    except ImportError:
        return EngineStatus(
            False,
            "The tensorrt runtime is not installed in the main environment. "
            f"`{BUILD_COMMAND}` installs the pinned version alongside the engines.",
        )

    details: dict[str, Any] = {"tensorrt": trt.__version__}
    if paths.manifest.exists():
        try:
            manifest = json.loads(paths.manifest.read_text(encoding="utf-8"))
            details["built_with"] = manifest.get("tensorrt")
            details["gpu"] = manifest.get("gpu")
            details["backbone_cosine"] = manifest.get("backbone_cosine")
            if manifest.get("tensorrt") and manifest["tensorrt"] != trt.__version__:
                return EngineStatus(
                    False,
                    f"The engines were built with TensorRT {manifest['tensorrt']} and this "
                    f"environment runs {trt.__version__}. Engines do not cross versions: "
                    f"run `{BUILD_COMMAND} -Force` to rebuild them here.",
                    details,
                )
        except (ValueError, json.JSONDecodeError):
            pass

    try:
        logger = trt.Logger(trt.Logger.ERROR)
        runtime = trt.Runtime(logger)
        for path in (paths.backbone, paths.enc_dec):
            with open(path, "rb") as handle:
                engine = runtime.deserialize_cuda_engine(handle.read())
            if engine is None:
                raise RuntimeError(f"{path.name} did not deserialise")
            del engine
    except Exception as exc:  # noqa: BLE001 - the message is the product here
        return EngineStatus(
            False,
            f"The engines exist but do not load in TensorRT {trt.__version__}: {exc}. "
            f"They are specific to the GPU and TensorRT version that built them; "
            f"`{BUILD_COMMAND} -Force` rebuilds them for this machine.",
            details,
        )

    cache_phrases = 0
    if paths.text_cache.exists():
        try:
            import torch

            data = torch.load(paths.text_cache, map_location="cpu", weights_only=False)
            cache_phrases = len(data.get("phrases", {}))
        except Exception:  # noqa: BLE001 - a bad cache is rebuilt on demand, not fatal
            cache_phrases = 0
    details["cached_phrases"] = cache_phrases
    return EngineStatus(
        True,
        f"accelerated SAM 3 ready: TensorRT {trt.__version__}, "
        f"{cache_phrases} phrases cached, {paths.backbone.parent}",
        details,
    )


# ---------------------------------------------------------------------------
# The two engines
# ---------------------------------------------------------------------------

def _trt_dtype_map(trt):
    table = {trt.float32: "float32", trt.float16: "float16", trt.int32: "int32"}
    if hasattr(trt, "bfloat16"):
        table[trt.bfloat16] = "bfloat16"
    return table


class _Backbone:
    """One deserialised backbone engine; `contexts` independent executions of it.

    Input `pixel_values` [1, 3, S, S]; outputs the three FPN levels, of which
    the encoder-decoder consumes only the coarsest ([1, 256, S/14, S/14]).
    Each context owns its buffers and its CUDA stream, so two of them can run
    two images at once, which is what the pipeline does.
    """

    def __init__(self, path: Path, device, contexts: int = 2) -> None:
        import tensorrt as trt
        import torch

        self.device = device
        logger = trt.Logger(trt.Logger.ERROR)
        runtime = trt.Runtime(logger)
        with open(path, "rb") as handle:
            self.engine = runtime.deserialize_cuda_engine(handle.read())
        if self.engine is None:
            raise DetectionError(
                f"{path.name} did not deserialise -- rebuild with {BUILD_COMMAND} -Force"
            )

        dtypes = _trt_dtype_map(trt)
        self.input_name = ""
        self.input_shape: tuple[int, ...] = ()
        self.input_dtype = torch.float32
        outputs: list[tuple[str, tuple[int, ...], Any]] = []
        for index in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(index)
            shape = tuple(self.engine.get_tensor_shape(name))
            dtype = getattr(torch, dtypes.get(self.engine.get_tensor_dtype(name), "float32"))
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.input_name, self.input_shape, self.input_dtype = name, shape, dtype
            else:
                outputs.append((name, shape, dtype))
        # Coarsest level last, whatever the engine called them.
        outputs.sort(key=lambda item: item[1][-1], reverse=True)
        self.output_specs = outputs
        self.image_size = self.input_shape[-1]

        self.slots: list[dict[str, Any]] = []
        for _ in range(contexts):
            context = self.engine.create_execution_context()
            inp = torch.empty(self.input_shape, dtype=self.input_dtype, device=device)
            outs = [torch.empty(shape, dtype=dtype, device=device) for _, shape, dtype in outputs]
            context.set_tensor_address(self.input_name, inp.data_ptr())
            for (name, _, _), buf in zip(outputs, outs, strict=True):
                context.set_tensor_address(name, buf.data_ptr())
            self.slots.append(
                {
                    "context": context,
                    "input": inp,
                    "outputs": outs,
                    "stream": torch.cuda.Stream(device=device),
                }
            )

    def launch(self, slot: int, pixels) -> Any:
        """Start the backbone on `pixels` in `slot`; returns the done event.

        Waits for everything already queued on the current stream first --
        which includes whoever last READ this slot's output buffer -- so a
        slot can be reissued without clobbering a result still in use.
        """
        import torch

        state = self.slots[slot]
        state["input"].copy_(pixels)
        ready = torch.cuda.current_stream(self.device).record_event()
        state["stream"].wait_event(ready)
        state["context"].execute_async_v3(state["stream"].cuda_stream)
        return state["stream"].record_event()

    def coarse_features(self, slot: int):
        """The [1, 256, P, P] level, valid once the slot's event has been waited on."""
        return self.slots[slot]["outputs"][-1]


class _EncoderDecoder:
    """The encoder-decoder-scoring engine: `max_classes` phrases per pass."""

    def __init__(self, path: Path, device) -> None:
        import tensorrt as trt
        import torch

        self.device = device
        logger = trt.Logger(trt.Logger.ERROR)
        runtime = trt.Runtime(logger)
        with open(path, "rb") as handle:
            self.engine = runtime.deserialize_cuda_engine(handle.read())
        if self.engine is None:
            raise DetectionError(
                f"{path.name} did not deserialise -- rebuild with {BUILD_COMMAND} -Force"
            )
        self.context = self.engine.create_execution_context()

        names = {self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)}
        shape = tuple(self.engine.get_tensor_shape("img_feat"))
        self.max_classes, self.spatial = int(shape[0]), int(shape[2])
        self.has_presence = "presence" in names

        mc, sp = self.max_classes, self.spatial
        self.buf_img = torch.zeros(mc, FEATURE_DIM, sp, sp, dtype=torch.float32, device=device)
        self.buf_pos = torch.zeros(mc, FEATURE_DIM, sp, sp, dtype=torch.float32, device=device)
        self.buf_text = torch.zeros(
            TEXT_TOKENS, mc, FEATURE_DIM, dtype=torch.float32, device=device
        )
        self.buf_mask = torch.ones(mc, TEXT_TOKENS, dtype=torch.float32, device=device)
        self.buf_scores = torch.zeros(mc, 200, 1, dtype=torch.float32, device=device)
        self.buf_boxes = torch.zeros(mc, 200, 4, dtype=torch.float32, device=device)
        self.buf_presence = torch.zeros(mc, 1, dtype=torch.float32, device=device)

        self.context.set_tensor_address("img_feat", self.buf_img.data_ptr())
        self.context.set_tensor_address("img_pos", self.buf_pos.data_ptr())
        self.context.set_tensor_address("text_feats", self.buf_text.data_ptr())
        self.context.set_tensor_address("text_mask", self.buf_mask.data_ptr())
        self.context.set_tensor_address("scores", self.buf_scores.data_ptr())
        self.context.set_tensor_address("boxes", self.buf_boxes.data_ptr())
        if self.has_presence:
            self.context.set_tensor_address("presence", self.buf_presence.data_ptr())
        self.stream = torch.cuda.Stream(device=device)
        self._pos_loaded = False

    def set_position_encoding(self, pos) -> None:
        """The encoder's sine position encoding, the same for every image."""
        import torch

        self.buf_pos[:] = pos.to(self.device, torch.float32).expand(self.max_classes, -1, -1, -1)
        self._pos_loaded = True

    def run(self, features, text, mask, count: int):
        """One pass: `count` <= max_classes phrases against one image's features.

        Returns (scores [count, 200, 1], boxes [count, 200, 4] cxcywh in
        [0, 1], presence [count, 1] or None), cloned off the persistent
        buffers so the caller may hold them across the next pass.
        """
        import torch

        assert self._pos_loaded, "position encoding not set"
        self.buf_img[:] = features.to(torch.float32).expand(self.max_classes, -1, -1, -1)
        self.buf_text.zero_()
        self.buf_text[:, :count, :] = text[:, :count, :]
        self.buf_mask.fill_(1.0)
        self.buf_mask[:count, :] = mask[:count, :]

        ready = torch.cuda.current_stream(self.device).record_event()
        self.stream.wait_event(ready)
        self.context.execute_async_v3(self.stream.cuda_stream)
        self.stream.synchronize()

        scores = self.buf_scores[:count].clone()
        boxes = self.buf_boxes[:count].clone()
        presence = self.buf_presence[:count].clone() if self.has_presence else None
        return scores, boxes, presence


# ---------------------------------------------------------------------------
# Text: the cache, and the tower that fills it
# ---------------------------------------------------------------------------

class _TextTower:
    """The transformers port of SAM 3's text encoder, weights only.

    `Sam3Model.get_text_features` is the final-layer-normed CLIP hidden
    states through the 1024->256 projection: the same computation as Meta's
    `VETextEncoder` (ln_final, then `resizer`). Loading it alone reads the
    `text_encoder` and `text_projection` tensors straight out of the
    safetensors file -- about a third of the checkpoint -- rather than
    standing up the whole detector to ask it a word.
    """

    def __init__(self, device) -> None:
        try:
            from huggingface_hub import hf_hub_download
            from safetensors import safe_open
            from transformers import AutoConfig, AutoTokenizer, CLIPTextModel
        except ImportError as exc:  # pragma: no cover - optional install
            raise DetectionError(
                "transformers is not installed, so new phrases cannot be encoded."
            ) from exc
        import torch

        self.device = device
        try:
            config = AutoConfig.from_pretrained(HF_WEIGHTS)
            self.tokenizer = AutoTokenizer.from_pretrained(HF_WEIGHTS)
            weights = hf_hub_download(HF_WEIGHTS, "model.safetensors")
        except Exception as exc:
            message = str(exc).lower()
            if any(word in message for word in ("gated", "401", "403", "restricted")):
                raise DetectionError(
                    "SAM 3's weights are gated by Meta. Request access at "
                    "https://huggingface.co/facebook/sam3 and `hf auth login` in a "
                    "project shell; until then only phrases already in the cache "
                    f"can be asked ({BUILD_COMMAND} -Phrases ... adds more)."
                ) from exc
            raise DetectionError(f"could not load SAM 3's text encoder: {exc}") from exc

        detector = getattr(config, "detector_config", None) or config
        text_config = detector.text_config
        self.model = CLIPTextModel(text_config).to(device).eval()
        self.projection = torch.nn.Linear(
            text_config.hidden_size, FEATURE_DIM
        ).to(device).eval()

        # Checkpoint keys read `detector_model.text_encoder.text_model.<x>`;
        # transformers 5's CLIPTextModel names the same parameter `<x>`.
        prefix = "detector_model.text_encoder.text_model."
        state: dict[str, torch.Tensor] = {}
        projection: dict[str, torch.Tensor] = {}
        with safe_open(weights, framework="pt", device="cpu") as handle:
            for key in handle.keys():  # noqa: SIM118 - safe_open is not a dict
                if key.startswith(prefix):
                    state[key[len(prefix):]] = handle.get_tensor(key)
                elif key.startswith("detector_model.text_projection."):
                    projection[key.split(".")[-1]] = handle.get_tensor(key)
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        missing = [k for k in missing if not k.endswith("position_ids")]
        if missing or not projection:
            raise DetectionError(
                "SAM 3's safetensors did not contain the text tower where expected "
                f"({len(missing)} missing keys); the transformers port may have changed."
            )
        self.projection.load_state_dict(projection)

    def encode(self, phrases: list[str]):
        """-> (text [32, N, 256] float32, mask [N, 32] bool with True = padding)."""
        import torch

        tokens = self.tokenizer(
            phrases,
            padding="max_length",
            max_length=TEXT_TOKENS,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)
        with torch.inference_mode():
            hidden = self.model(
                input_ids=tokens.input_ids, attention_mask=tokens.attention_mask
            ).last_hidden_state
            projected = self.projection(hidden)  # (N, 32, 256)
        text = projected.transpose(0, 1).float().contiguous()
        mask = tokens.attention_mask.eq(0)
        return text, mask


class PhraseCache:
    """phrase -> (text features, padding mask), persisted beside the engines."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.phrases: dict[str, dict[str, Any]] = {}
        self.source: dict[str, str] = {}
        self._lock = threading.Lock()
        if path.exists():
            import torch

            data = torch.load(path, map_location="cpu", weights_only=False)
            if data.get("format") != CACHE_FORMAT:
                raise DetectionError(
                    f"{path} is prompt-cache format {data.get('format')!r}, expected "
                    f"{CACHE_FORMAT}; delete it and run {BUILD_COMMAND}."
                )
            self.phrases = dict(data.get("phrases", {}))
            self.source = dict(data.get("source", {}))

    def missing(self, phrases: Iterable[str]) -> list[str]:
        return [p for p in dict.fromkeys(phrases) if p not in self.phrases]

    def add(self, phrase: str, text, mask, source: str) -> None:
        self.phrases[phrase] = {"text": text.cpu().contiguous(), "mask": mask.cpu().contiguous()}
        self.source[phrase] = source

    def save(self) -> None:
        import torch

        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            partial = self.path.with_suffix(".pt.part")
            torch.save(
                {"format": CACHE_FORMAT, "phrases": self.phrases, "source": self.source},
                partial,
            )
            partial.replace(self.path)

    def batch(self, phrases: list[str], device):
        """Stack cached entries: text [32, N, 256], mask [N, 32] as float (1 = pad)."""
        import torch

        text = torch.stack([self.phrases[p]["text"] for p in phrases], dim=1)
        mask = torch.stack([self.phrases[p]["mask"] for p in phrases], dim=0)
        return text.to(device, torch.float32), mask.to(device, torch.float32)


# ---------------------------------------------------------------------------
# Pixels in, boxes out
# ---------------------------------------------------------------------------

def preprocess(image: np.ndarray, size: int, device):
    """(H, W, 3) uint8 RGB -> [1, 3, size, size] float32, exactly as DART does.

    Squashed to a square, not letterboxed: SAM 3 was trained that way and
    the boxes come back normalised to the original width and height, so
    nothing is lost. Antialiased bilinear on the uint8 tensor, then scale
    to [0, 1] and normalise by 0.5 / 0.5 -- torchvision's v2 pipeline in
    DART's predictor, reproduced call for call.
    """
    import torch
    from torchvision.transforms.v2 import functional as tvf

    tensor = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).to(device)
    tensor = tvf.resize(tensor, [size, size], antialias=True)
    tensor = tvf.to_dtype(tensor, torch.float32, scale=True)
    tensor = tvf.normalize(tensor, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    return tensor.unsqueeze(0)


def boxes_to_raw(
    scores,
    boxes,
    presence,
    labels: list[str],
    width: int,
    height: int,
    threshold: float,
    nms_iou: float = NMS_IOU,
) -> list[RawDetection]:
    """Engine outputs for one image -> `RawDetection`s in that image's pixels.

    Pure tensor arithmetic, kept apart from the engines so it can be tested
    on made-up logits with no GPU. `scores` [N, Q, 1] logits, `boxes`
    [N, Q, 4] cxcywh normalised, `presence` [N, 1] logits or None.
    """
    import torch
    from torchvision.ops import batched_nms

    probs = scores[:, :, 0].float().sigmoid()
    if presence is not None:
        probs = probs * presence.float().sigmoid()
    keep = probs > threshold
    class_index, query_index = keep.nonzero(as_tuple=True)
    if class_index.numel() == 0:
        return []

    confidence = probs[class_index, query_index]
    chosen = boxes[class_index, query_index].float()
    cx, cy, w, h = chosen.unbind(-1)
    xyxy = torch.stack(
        [(cx - w / 2) * width, (cy - h / 2) * height, (cx + w / 2) * width, (cy + h / 2) * height],
        dim=-1,
    )
    if nms_iou < 1.0:
        survivors = batched_nms(xyxy, confidence, class_index, nms_iou)
        confidence = confidence[survivors]
        class_index = class_index[survivors]
        xyxy = xyxy[survivors]

    order = confidence.argsort(descending=True)
    out: list[RawDetection] = []
    for index in order.tolist():
        x0, y0, x1, y1 = xyxy[index].tolist()
        out.append(
            RawDetection(
                label=labels[int(class_index[index])],
                confidence=float(confidence[index]),
                x0=max(0.0, x0),
                y0=max(0.0, y0),
                x1=min(float(width), x1),
                y1=min(float(height), y1),
            )
        )
    return out


def chunked(items: list[str], size: int) -> list[list[str]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


# ---------------------------------------------------------------------------
# The provider
# ---------------------------------------------------------------------------

@dataclass
class StreamStats:
    """Where the milliseconds went over one `detect_stream` call."""

    images: int = 0
    preprocess_s: float = 0.0
    backbone_wait_s: float = 0.0
    enc_dec_s: float = 0.0
    postprocess_s: float = 0.0
    total_s: float = 0.0


class Sam3TrtProvider:
    """SAM 3 through DART's TensorRT engines. Boxes, calibrated scores, fast.

    Same `prompts` / `confidence` contract as `Sam3Provider`. `detect` is the
    `DetectionProvider` seam; `detect_stream` is the same thing over an
    iterator, pipelined, for a walk.
    """

    # Same object, two phrasings: near-identical boxes merge across labels,
    # for the reason `merge_duplicates` documents. Same bar as the eager
    # provider's mask fusion.
    cross_label_iou = 0.70

    def __init__(
        self,
        prompts: list[str] | None = None,
        device: str | None = None,
        confidence: float = 0.4,
        directory: Path | None = None,
    ) -> None:
        from vision.openvocab import default_prompts

        self.prompts = [p.strip().lower() for p in (prompts or default_prompts()) if p.strip()]
        self.confidence = confidence
        self.paths = EnginePaths.locate(directory)
        self.weights = f"{HF_WEIGHTS} (TensorRT FP16, DART)"
        self._device_name = device
        self._backbone: _Backbone | None = None
        self._enc_dec: _EncoderDecoder | None = None
        self._cache: PhraseCache | None = None
        self._tower: _TextTower | None = None
        self._text_batches: list[tuple[list[str], Any, Any]] | None = None
        self._lock = threading.Lock()
        self.last_stats = StreamStats()

    @property
    def name(self) -> str:
        return f"sam3-trt({len(self.prompts)} prompts)"

    @property
    def device(self) -> str:
        self._load()
        return self._device_name or "cpu"

    @property
    def image_size(self) -> int:
        self._load()
        assert self._backbone is not None
        return self._backbone.image_size

    # -- loading ------------------------------------------------------------

    def _load(self):
        if self._backbone is not None:
            return None
        with self._lock:
            if self._backbone is not None:
                return None
            missing = self.paths.missing()
            if missing:
                raise DetectionError(_missing_message(self.paths, missing))
            try:
                import tensorrt  # noqa: F401 - presence is the point
                import torch
            except ImportError as exc:
                raise DetectionError(
                    "The tensorrt runtime is not installed in the main environment; "
                    f"`{BUILD_COMMAND}` installs the pinned version."
                ) from exc
            if not torch.cuda.is_available():
                raise DetectionError(
                    "TensorRT engines need a CUDA GPU and none is visible to torch. "
                    "The plain `sam3` model can run on CPU, slowly."
                )
            self._device_name = self._device_name or "cuda"
            device = torch.device(self._device_name)
            try:
                backbone = _Backbone(self.paths.backbone, device, contexts=2)
                enc_dec = _EncoderDecoder(self.paths.enc_dec, device)
            except DetectionError:
                raise
            except Exception as exc:  # noqa: BLE001 - one message beats one traceback
                raise DetectionError(
                    f"could not load the SAM 3 TensorRT engines: {exc}. They are "
                    f"specific to the GPU and TensorRT version that built them; "
                    f"`{BUILD_COMMAND} -Force` rebuilds them for this machine."
                ) from exc
            if enc_dec.spatial != backbone.image_size // PATCH:
                raise DetectionError(
                    f"engine mismatch: backbone is {backbone.image_size}px but the "
                    f"encoder-decoder expects {enc_dec.spatial * PATCH}px. Both must be "
                    f"built at one -ImageSize; `{BUILD_COMMAND} -Force`."
                )
            pos = torch.load(self.paths.pos_enc, map_location="cpu", weights_only=False)
            if int(pos.get("spatial", -1)) != enc_dec.spatial:
                raise DetectionError(
                    f"{self.paths.pos_enc.name} is for {pos.get('imgsz')}px, the engines "
                    f"for {enc_dec.spatial * PATCH}px; `{BUILD_COMMAND} -Force`."
                )
            enc_dec.set_position_encoding(pos["pos_enc"].unsqueeze(0))
            self._cache = PhraseCache(self.paths.text_cache)
            self._backbone, self._enc_dec = backbone, enc_dec
        return None

    def _ensure_text(self) -> list[tuple[list[str], Any, Any]]:
        """Text features for the prompt list, in engine-sized chunks."""
        if self._text_batches is not None:
            return self._text_batches
        assert self._cache is not None and self._enc_dec is not None
        import torch

        missing = self._cache.missing(self.prompts)
        if missing:
            if self._tower is None:
                self._tower = _TextTower(torch.device(self._device_name or "cuda"))
            text, mask = self._tower.encode(missing)
            for index, phrase in enumerate(missing):
                self._cache.add(phrase, text[:, index, :], mask[index], source="hf")
            self._cache.save()

        device = torch.device(self._device_name or "cuda")
        batches = []
        for chunk in chunked(list(dict.fromkeys(self.prompts)), self._enc_dec.max_classes):
            text, mask = self._cache.batch(chunk, device)
            batches.append((chunk, text, mask))
        self._text_batches = batches
        return batches

    # -- detection ------------------------------------------------------------

    def detect(self, images: list[np.ndarray]) -> list[list[RawDetection]]:
        """One list of detections per image, in the order given."""
        return list(self.detect_stream(iter(images)))

    def detect_stream(self, images: Iterator[np.ndarray]) -> Iterator[list[RawDetection]]:
        """The pipelined form: backbone(n+1) runs while enc-dec(n) is scored.

        Consumes lazily, so a walk's worth of views need never be in memory
        at once, and yields in order. `last_stats` records the breakdown.
        """
        import torch

        self._load()
        assert self._backbone is not None and self._enc_dec is not None
        batches = self._ensure_text()
        backbone, enc_dec = self._backbone, self._enc_dec
        device = torch.device(self._device_name or "cuda")
        size = backbone.image_size
        stats = StreamStats()
        started = time.perf_counter()

        try:
            with torch.inference_mode():
                pending: list[tuple[int, Any, int, int]] = []  # slot, event, width, height
                slot = 0

                def submit(image: np.ndarray) -> None:
                    nonlocal slot
                    mark = time.perf_counter()
                    pixels = preprocess(image, size, device)
                    stats.preprocess_s += time.perf_counter() - mark
                    event = backbone.launch(slot, pixels)
                    pending.append((slot, event, image.shape[1], image.shape[0]))
                    slot = 1 - slot

                def finish() -> list[RawDetection]:
                    used, event, width, height = pending.pop(0)
                    mark = time.perf_counter()
                    torch.cuda.current_stream(device).wait_event(event)
                    event.synchronize()
                    stats.backbone_wait_s += time.perf_counter() - mark
                    features = backbone.coarse_features(used)

                    mark = time.perf_counter()
                    found: list[RawDetection] = []
                    for labels, text, mask in batches:
                        scores, boxes, presence = enc_dec.run(features, text, mask, len(labels))
                        post = time.perf_counter()
                        found.extend(
                            boxes_to_raw(scores, boxes, presence, labels, width, height,
                                         self.confidence)
                        )
                        stats.postprocess_s += time.perf_counter() - post
                    stats.enc_dec_s += time.perf_counter() - mark
                    stats.images += 1
                    return found

                first = next(images, None)
                if first is None:
                    return
                submit(first)
                for image in images:
                    # Two slots: the next image goes in while this one is scored.
                    submit(image)
                    yield finish()
                yield finish()
        except DetectionError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise DetectionError(f"{self.name} failed: {exc}") from exc
        finally:
            stats.total_s = time.perf_counter() - started
            self.last_stats = stats
