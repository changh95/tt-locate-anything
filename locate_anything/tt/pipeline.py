# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Serving pipeline for LocateAnything-3B on a single Blackhole p150a (greedy AR decode).

Everything here is the recipe the port validated in ``tests/bench_locate_anything.py``
(model construction, warm-up, trace-replay decode) and ``tests/test_demo_visualize.py``
(image -> ``<ref>/<box>`` tokens -> pixel boxes), lifted out of the pytest modules so the
server can import it without ``pytest``. The test modules import these names back from
here, so there is one copy.

Nothing in this module opens a device or touches weights at import time. The MTP
(Parallel Box Decoding) path stays in ``tt/mtp.py`` / the tests; it is experimental,
not accuracy-gated, and not served.

Fused path (default since the 2026-09-13 device validation; ``TT_FUSED=0`` = the legacy request
flow above, read ONCE at build through :class:`FusedConfig`: same device ops in the same order
with the same arguments, so the same numerics; the only host-visible differences are the
``"fused"`` key that ``run()`` adds to its result dict and the ``fused`` entry of ``GET /info`` --
the ``/predict`` body is unchanged). The default fused configuration keeps the legacy vision
numerics bit for bit (DEVICE_VALIDATION.md "Results": 360.9 -> 304.9 ms on the demo request).
Per request the device work becomes:

  vision   copy pixels into the persistent input -> ONE metal trace (MoonViT + device patch
           merger + mlp1, ``MoonViT.forward_device``) -> vit_proj stays on device
  prefill  copy the padded prompt ids (2 KB) -> ONE metal trace (embedding gather + 0/1 merge
           matmul + 36 layers, ``LATransformer.fused_prefill_graph``) -> eager slice / norm /
           LM head tail -> ONE logits row read back for the greedy first token
  decode   stock Generator decode trace with on-device greedy sampling: the per-step host work
           drops from a 32 x 152704 bf16 logits readback + host argmax to a token-id readback
           (the library still calls ``seed_manager.get_new_values`` every step -- a seed
           host->device copy on the seeded path / state transitions -- and reads two small
           device tensors back on the ``reset_batch`` step), stale-token guard armed first

Warm-up contract: run 1 compiles everything eagerly (fused ops, no capture) and captures the
library decode(+sampling) trace; the vision and prefill traces are captured right after run 1,
once every long-lived device buffer exists, and run 2 replays them, so READY means every trace
has been replayed once. Prompts outside the captured bucket (S > 512 tokens at the served grid)
or with a different image-token layout fall back to the legacy host merge + eager prefill.
"""
import gc
import os
import re
import time

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

import ttnn
from models.demos.qwen25_vl.tt.common import PagedAttentionConfig, merge_vision_tokens, preprocess_inputs_prefill
from models.tt_transformers.tt.common import Mode, sample_host
from models.tt_transformers.tt.generator import Generator
from models.tt_transformers.tt.model_config import (
    DecodersPrecision,
    MathFidelitySetting,
    ModelArgs,
    ModelOptimizations,
    OpGroup,
)
from locate_anything.reference import la_inputs
from locate_anything.tt.fused import (
    FusedConfig,
    arm_stale_token_guard,
    greedy_sampling_params,
    image_token_rows,
    prefill_bucket_len,
)
from locate_anything.tt.model_la import LATransformer
from locate_anything.tt.vision import MoonViT

# LocateAnything special token ids (from the extracted HF config).
IMAGE_TOKEN_INDEX = la_inputs.IMAGE_TOKEN_INDEX  # 151665
EOS_TOKEN_ID = 151645  # <|im_end|>

# Paged-attention page params (block_size * max_num_blocks must cover max_seq_len).
PAGE_PARAMS = {"page_block_size": 32, "page_max_num_blocks": 1024}

# Validated single-chip device parameters (tests' `device_params`): no cross-chip fabric,
# 50 MB trace region (trace-replay decode), one command queue.
TRACE_REGION_SIZE = 50_000_000
NUM_COMMAND_QUEUES = 1
DEVICE_PARAMS = {"fabric_config": False, "trace_region_size": TRACE_REGION_SIZE, "num_command_queues": NUM_COMMAND_QUEUES}

MAX_SEQ_LEN = 4096
DEFAULT_IN_TOKEN_LIMIT = 1024  # every README number was measured with this (upstream default 25600)
PRECISION_PRESETS = ("accuracy", "bfp8attn")


# --------------------------------------------------------------------------- device
def parse_mesh_shape(text):
    """'1x1' (what tt-model exports), '(1, 1)' or '1,1' -> (rows, cols). Anything else raises."""
    raw = str(text).strip()
    s = raw.lower().replace("(", "").replace(")", "").replace(" ", "")
    for sep in ("x", ","):
        if sep in s:
            parts = s.split(sep)
            if len(parts) == 2 and all(p.isdigit() for p in parts):
                return int(parts[0]), int(parts[1])
    raise RuntimeError(f"cannot parse mesh shape {raw!r}; expected '1x1', '(1, 1)' or '1,1'")


def open_mesh_device(
    mesh_shape=(1, 1),
    trace_region_size=TRACE_REGION_SIZE,
    num_command_queues=NUM_COMMAND_QUEUES,
    physical_device_ids=None,
):
    """Open the single-chip mesh exactly like tt-metal's `mesh_device` fixture does for the
    tests' device_params: `fabric_config: False` (set_fabric is a no-op), the default
    DispatchCoreConfig, the given trace region and command-queue count."""
    kwargs = dict(
        trace_region_size=int(trace_region_size),
        num_command_queues=int(num_command_queues),
        dispatch_core_config=ttnn.DispatchCoreConfig(),
    )
    if physical_device_ids:
        kwargs["physical_device_ids"] = [int(i) for i in physical_device_ids]
    return ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(*mesh_shape), **kwargs)


# --------------------------------------------------------------------------- LLM build
def select_optimizations(model_args, prec=None):
    """Decoder precision preset (all keep BFP8 MLP weights for >=99% PCC):
    accuracy  (default): BF16 attention (WQKV/WO/KV) + HiFi4  -> highest accuracy (validated 0.9928)
    bfp8attn:            all-BFP8 weights + KV  + HiFi4 attn  -> less bandwidth (validated 0.9912)
    """
    prec = prec or os.environ.get("LA_PREC", "accuracy")
    if prec == "bfp8attn":
        hifi4 = MathFidelitySetting.HIFI4
        mo = ModelOptimizations(
            {
                "OpFidelity": {
                    OpGroup.LI_QKV_DECODE: hifi4,
                    OpGroup.LI_QKV_PREFILL: hifi4,
                    OpGroup.SDPA_DECODE: hifi4,
                    OpGroup.SDPA_PREFILL: hifi4,
                    OpGroup.LI_O_DECODE: hifi4,
                    OpGroup.LI_O_PREFILL: hifi4,
                }
            }
        )
        return DecodersPrecision(model_args.n_layers, model_args.model_name, mo)
    if prec != "accuracy":
        raise ValueError(f"unknown LA_PREC {prec!r}; choose one of {PRECISION_PRESETS}")
    return DecodersPrecision.accuracy(model_args.n_layers, model_args.model_name)


_select_optimizations = select_optimizations  # name the tests import


def create_tt_page_table(paged_attention_config, tt_model_args):
    """Random (shuffled) virtual->physical block mapping. Copied from qwen25_vl demo."""
    if paged_attention_config is None:
        return None
    permutation = torch.randperm(paged_attention_config.max_num_blocks)
    reverse_permutation = torch.argsort(permutation)
    return reverse_permutation.reshape(
        tt_model_args.max_batch_size,
        paged_attention_config.max_num_blocks // tt_model_args.max_batch_size,
    )


def create_tt_model(
    mesh_device,
    instruct,
    max_batch_size,
    optimizations,
    max_seq_len,
    page_params,
    dtype=ttnn.bfloat8_b,
    use_paged_kv_cache=True,
):
    """Build LATransformer + paged KV cache. Adapted from qwen25_vl demo create_tt_model.

    Reads ``HF_MODEL`` (tt_transformers' ModelArgs does): it MUST point at the EXTRACTED
    vanilla Qwen2.5-3B dir, never at nvidia/LocateAnything-3B.
    """
    tt_model_args = ModelArgs(
        mesh_device,
        instruct=instruct,
        max_batch_size=max_batch_size,
        optimizations=optimizations,
        max_seq_len=max_seq_len,
        cache_hf=True,
    )
    state_dict = tt_model_args.load_state_dict()

    paged_attention_config = (
        PagedAttentionConfig(
            block_size=page_params["page_block_size"],
            max_num_blocks=page_params["page_max_num_blocks"],
        )
        if use_paged_kv_cache
        else None
    )

    # NOTE: do NOT pass use_paged_kv_cache=True. The stock Attention only calls
    # init_kv_cache() (which allocates `layer_past`) when use_paged_kv_cache is
    # False; the paged vs non-paged *shape* is selected by paged_attention_config.
    # This matches models/tt_transformers/tt/common.py:create_tt_model.
    model = LATransformer(
        args=tt_model_args,
        mesh_device=mesh_device,
        dtype=dtype,
        state_dict=state_dict,
        weight_cache_path=tt_model_args.weight_cache_path(dtype),
        paged_attention_config=paged_attention_config,
    )

    tt_kv_cache = [l.attention.layer_past for l in model.layers] if paged_attention_config else None

    return tt_model_args, model, paged_attention_config, tt_kv_cache


# --------------------------------------------------------------------------- output parsing
_BOX_RE = re.compile(r"<box><(\d+)><(\d+)><(\d+)><(\d+)></box>")
_POINT_RE = re.compile(r"<box><(\d+)><(\d+)></box>")
_REF_RE = re.compile(r"<ref>(.*?)</ref>")
# token-aware iterator: a <ref>label</ref> OR a 4-coord box OR a 2-coord point
_ITEM_RE = re.compile(
    r"<ref>(?P<ref>.*?)</ref>|<box><(?P<x1>\d+)><(?P<y1>\d+)><(?P<x2>\d+)><(?P<y2>\d+)></box>|<box><(?P<px>\d+)><(?P<py>\d+)></box>"
)

_PALETTE = [
    (255, 64, 64),
    (64, 200, 64),
    (64, 128, 255),
    (255, 180, 0),
    (200, 64, 255),
    (0, 200, 200),
    (255, 100, 180),
    (140, 220, 60),
]


def parse_detections(answer: str, W: int, H: int):
    """Walk the answer in order, attaching each box/point to the current <ref> label.

    Coordinates in the answer are normalized 0..1000 over the image the model saw; they are
    mapped onto the ORIGINAL W x H here. Each item also keeps the raw normalized ints.
    """
    dets = []
    cur_label = None
    for m in _ITEM_RE.finditer(answer):
        if m.group("ref") is not None:
            cur_label = m.group("ref").strip()
        elif m.group("x1") is not None:
            x1, y1, x2, y2 = (int(m.group(k)) for k in ("x1", "y1", "x2", "y2"))
            dets.append(
                {
                    "label": cur_label,
                    "box": (x1 / 1000 * W, y1 / 1000 * H, x2 / 1000 * W, y2 / 1000 * H),
                    "box_norm": (x1, y1, x2, y2),
                }
            )
        elif m.group("px") is not None:
            px, py = int(m.group("px")), int(m.group("py"))
            dets.append({"label": cur_label, "point": (px / 1000 * W, py / 1000 * H), "point_norm": (px, py)})
    return dets


def _font(size):
    for p in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def draw_detections(image: Image.Image, dets) -> Image.Image:
    """Return a copy of `image` with the parsed boxes/points drawn on it."""
    img = image.convert("RGB").copy()
    draw = ImageDraw.Draw(img)
    W, H = img.size
    lw = max(2, round(min(W, H) / 300))
    font = _font(max(14, round(min(W, H) / 45)))
    labels = sorted({d.get("label") or "obj" for d in dets})
    color_of = {lab: _PALETTE[i % len(_PALETTE)] for i, lab in enumerate(labels)}
    for d in dets:
        col = color_of.get(d.get("label") or "obj")
        lab = d.get("label") or ""
        if "box" in d:
            x1, y1, x2, y2 = d["box"]
            draw.rectangle([x1, y1, x2, y2], outline=col, width=lw)
            if lab:
                tb = draw.textbbox((0, 0), lab, font=font)
                tw, th = tb[2] - tb[0], tb[3] - tb[1]
                ty = max(0, y1 - th - 4)
                draw.rectangle([x1, ty, x1 + tw + 6, ty + th + 4], fill=col)
                draw.text((x1 + 3, ty + 2), lab, fill=(255, 255, 255), font=font)
        elif "point" in d:
            px, py = d["point"]
            r = lw * 3
            draw.ellipse([px - r, py - r, px + r, py + r], outline=col, width=lw)
            if lab:
                draw.text((px + r + 2, py - r), lab, fill=col, font=font)
    return img


def visualize(image: Image.Image, dets, out_path: str):
    """Draw and save (the demo test's helper)."""
    draw_detections(image, dets).save(out_path)
    return out_path


class _MergeConfig:
    image_token_id = IMAGE_TOKEN_INDEX


# --------------------------------------------------------------------------- pipeline
class LocateAnythingPipeline:
    """One image + one query -> labelled boxes, everything on one Blackhole chip.

    Construction = the demo/benchmark build (LLM via tt_transformers with the validated
    precision preset + paged KV, MoonViT for ONE canonical vision grid). `run()` = the
    demo's greedy AR loop with the benchmark's trace-replay decode. Batch 1, stateful
    (KV cache, page table, decode trace): callers must serialize `run()` calls and issue
    them from one thread.
    """

    def __init__(
        self,
        mesh_device,
        snapshot_dir,
        llm_dir,
        grid_hw,
        *,
        in_token_limit=DEFAULT_IN_TOKEN_LIMIT,
        max_seq_len=MAX_SEQ_LEN,
        prec="accuracy",
        enable_trace=True,
        log=print,
        fused=None,
    ):
        self.log = log
        self.mesh_device = mesh_device
        self.snapshot_dir = snapshot_dir
        self.llm_dir = llm_dir
        self.grid_hw = (int(grid_hw[0]), int(grid_hw[1]))
        self.in_token_limit = int(in_token_limit)
        self.max_seq_len = int(max_seq_len)
        self.prec = prec
        self.enable_trace = bool(enable_trace)
        # TT_FUSED knob family, read once here (or handed in by the server) for the life of the pipeline.
        self.fused = fused if fused is not None else FusedConfig.from_env()
        self._fused_prefill = None  # FusedPrefillState (K/L/N) when the fused LLM boundary is active
        self._device_sampling = False  # M: greedy argmax on device in the decode loop

        # tt_transformers reads the checkpoint dir from HF_MODEL. The launcher exports
        # HF_MODEL=nvidia/LocateAnything-3B (the weights pointer) -- overwrite it with the
        # extracted vanilla Qwen2.5-3B dir BEFORE ModelArgs is built.
        os.environ["HF_MODEL"] = llm_dir
        if any(k in llm_dir.lower() for k in ("instruct", "it")):
            log(
                f"WARNING: extracted-dir path {llm_dir!r} contains 'it'/'instruct'; tt_transformers' ModelArgs "
                "will flag instruct=True (only changes the tensor-cache subdir name for this server)"
            )

        log(f"Loading weights: Qwen2.5-3B LLM from {llm_dir} -> ttnn (bfp8 MLP, {prec} preset, max_seq_len={max_seq_len}) ...")
        t0 = time.perf_counter()
        self.model_args, self.model, self.paged_attention_config, self.tt_kv_cache = create_tt_model(
            mesh_device,
            instruct=False,
            max_batch_size=1,
            optimizations=lambda ma: select_optimizations(ma, prec),
            max_seq_len=self.max_seq_len,
            page_params=PAGE_PARAMS,
            dtype=ttnn.bfloat8_b,
            use_paged_kv_cache=True,
        )
        self.tokenizer = self.model_args.tokenizer
        self.generator = Generator([self.model], [self.model_args], mesh_device)
        self.page_table = create_tt_page_table(self.paged_attention_config, self.model_args)
        self.vocab_size = self.model.vocab_size
        self.device_name = self.model_args.device_name

        # Host embedding table: reuse the bf16 HF model ModelArgs already loaded (cache_hf=True)
        # instead of the demo's second, fp32 Qwen2ForCausalLM load (12 GB). The demo's fp32 load is
        # a pure upcast of this same bf16 table, so `.float()` of these rows is bit-identical.
        hf_model = self.model_args.cached_hf_model
        assert hf_model is not None, "ModelArgs(cache_hf=True) did not keep the HF model"
        self.embed_weight = hf_model.get_input_embeddings().weight.detach().clone()  # [vocab, hidden] bf16
        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
        self.pad_token_id = int(pad_id)
        self.model_args.cached_hf_model = None  # drop ~6 GB of host RAM; ttnn weights are on device / in the cache
        del hf_model
        gc.collect()
        log(f"LLM ready in {time.perf_counter() - t0:.1f}s (device {self.device_name}, vocab {self.vocab_size})")

        log(f"Loading weights: MoonViT vision tower + mlp1 from {snapshot_dir} -> ttnn (grid {self.grid_hw}) ...")
        t0 = time.perf_counter()
        self.vis = MoonViT(mesh_device, snapshot_dir, self.grid_hw, dtype=ttnn.bfloat16, fused=self.fused)
        self.vis.state_dict = None  # host copy of the vision weights is not needed after upload
        gc.collect()
        log(f"MoonViT ready in {time.perf_counter() - t0:.1f}s (L={self.vis.L} patches, {la_inputs.num_image_tokens(self.grid_hw)} image tokens)")

        if self.fused.enabled:
            self._init_fused()

    # ------------------------------------------------------------------ fused (TT_FUSED=1) setup
    def _init_fused(self):
        """Allocate every long-lived device buffer of the fused LLM boundary and decide which fused
        levers this build can use. Runs at construction, i.e. before ANY trace is captured."""
        f = self.fused
        self.log(f"TT_FUSED=1: {f.as_dict()}")
        if f.device_sampling:
            if getattr(self.model, "_supports_on_device_sampling", False) and getattr(self.model, "sampling", None):
                self._device_sampling = True
            else:
                self.log("WARNING: TT_FUSED device sampling requested but the model reports no on-device sampling support; host argmax kept")
        if f.device_merge:
            n_tok = la_inputs.num_image_tokens(self.grid_hw)
            rows = None
            for probe in ("object", "person</c>car", "a"):
                ids = self.tokenizer([la_inputs.build_chat_text(probe, n_tok)], return_tensors="pt")["input_ids"]
                r = image_token_rows(ids, IMAGE_TOKEN_INDEX)
                if rows is None:
                    rows = r
                    probe_len = int(ids.shape[1])
                elif not torch.equal(rows, r):
                    self.log("WARNING: image-token rows depend on the query text; TT_FUSED device merge disabled")
                    rows = None
                    break
            if rows is not None:
                prefill_len = prefill_bucket_len(probe_len, self.max_seq_len)
                self._fused_prefill = self.model.build_fused_prefill_state(
                    prefill_len, self.vis.merged_pad, rows, self.page_table
                )
                self.log(
                    f"TT_FUSED device merge: image tokens at rows [{int(rows[0])}, {int(rows[-1]) + 1}) of the "
                    f"{prefill_len}-token prefill bucket; prompts outside it use the legacy host merge"
                )

    def _fused_prefill_applies(self, input_ids):
        """The device merge / prefill trace serve one bucket and one image-token layout."""
        st = self._fused_prefill
        if st is None:
            return False
        S = int(input_ids.shape[1])
        if S > st.prefill_len or prefill_bucket_len(S, self.max_seq_len) != st.prefill_len:
            return False
        return torch.equal(image_token_rows(input_ids, IMAGE_TOKEN_INDEX), st.image_rows)

    def _prefill_fused(self, input_ids, vit_dev, last_token_idx):
        """K + L + N: ids upload -> (trace | eager fused graph) -> eager tail -> first token.
        Returns (first_token, decode_start_pos, prefill_len)."""
        st = self._fused_prefill
        S = int(input_ids.shape[1])
        self.model.switch_mode(Mode.PREFILL)
        ttnn.copy_host_to_device_tensor(
            self.model.host_prefill_ids(input_ids, st.prefill_len, self.pad_token_id), st.ids_dev
        )
        if st.trace_id is not None:
            if vit_dev.buffer_address() != st.vit_in_address:
                raise RuntimeError(
                    "fused prefill trace was recorded on a different vision output buffer; "
                    "the vision trace must be captured before the prefill trace and both must be persistent"
                )
            ttnn.execute_trace(self.mesh_device, st.trace_id, cq_id=0, blocking=False)
            hidden = st.hidden_out
        else:
            hidden = self.model.fused_prefill_graph(st, vit_dev, self.tt_kv_cache)
        logits = self.model.fused_prefill_logits(hidden, last_token_idx)
        first_tok = self.model.first_token_from_logits(logits, last_token_idx, self.fused.small_readback)
        ttnn.deallocate(logits)
        if st.trace_id is None:
            ttnn.deallocate(hidden)
        return first_tok, S, st.prefill_len

    def _fused_traces_wanted(self):
        return self.fused.enabled and self.enable_trace and (self.fused.vision_trace or self.fused.prefill_trace)

    def capture_traces(self, image, query="object"):
        """Record the vision trace and then the prefill trace on the given (warm-up) inputs.

        Call AFTER a full eager run (program cache warm, library decode trace captured, so its
        persistent inputs already exist): every buffer allocated after a capture may land inside
        that trace's scratch and be clobbered on replay. Each capture executes its graph once
        eagerly first (rf-detr pattern); the prefill trace reads the persistent vision output."""
        if not self._fused_traces_wanted():
            return
        bundle = la_inputs.build_inputs(self.tokenizer, image.convert("RGB"), query, in_token_limit=self.in_token_limit, grid_hw=self.grid_hw)
        if self.fused.vision_trace and not self.vis.trace_captured:
            t0 = time.perf_counter()
            self.vis.capture_trace(bundle["pixel_values"].float())
            self.log(f"TT_FUSED: vision trace captured in {time.perf_counter() - t0:.1f}s")
        st = self._fused_prefill
        if (
            self.fused.prefill_trace
            and st is not None
            and st.trace_id is None
            and self.vis.trace_captured
            and self._fused_prefill_applies(bundle["input_ids"])
        ):
            t0 = time.perf_counter()
            vit_dev = self.vis.trace_output
            self.model.switch_mode(Mode.PREFILL)
            ttnn.copy_host_to_device_tensor(
                self.model.host_prefill_ids(bundle["input_ids"], st.prefill_len, self.pad_token_id), st.ids_dev
            )
            hidden = self.model.fused_prefill_graph(st, vit_dev, self.tt_kv_cache)  # compile pass
            ttnn.synchronize_device(self.mesh_device)
            ttnn.deallocate(hidden)
            trace_id = ttnn.begin_trace_capture(self.mesh_device, cq_id=0)
            try:
                hidden = self.model.fused_prefill_graph(st, vit_dev, self.tt_kv_cache)
            except BaseException:
                # Never leave a capture open: an open capture hangs the device close (p150 lesson).
                try:
                    ttnn.end_trace_capture(self.mesh_device, trace_id, cq_id=0)
                    ttnn.release_trace(self.mesh_device, trace_id)
                except Exception as e:  # noqa: BLE001
                    self.log(f"closing the failed prefill trace capture raised: {e!r}")
                raise
            ttnn.end_trace_capture(self.mesh_device, trace_id, cq_id=0)
            ttnn.synchronize_device(self.mesh_device)
            st.trace_id = trace_id
            st.hidden_out = hidden
            st.vit_in_address = vit_dev.buffer_address()
            self.log(f"TT_FUSED: prefill trace ({st.prefill_len} tokens) captured in {time.perf_counter() - t0:.1f}s")

    @property
    def fused_status(self):
        """What the fused path actually runs with (for /info and the warm-up log)."""
        st = self._fused_prefill
        return {
            "enabled": self.fused.enabled,
            "vision_trace": bool(self.vis.trace_captured) if self.fused.enabled else False,
            "device_merge": st is not None,
            "prefill_trace": bool(st is not None and st.trace_id is not None),
            "prefill_bucket": st.prefill_len if st is not None else None,
            "device_sampling": self._device_sampling,
        }

    # ------------------------------------------------------------------ inference
    @property
    def canonical_size_wh(self):
        return (self.grid_hw[1] * la_inputs.PATCH_SIZE, self.grid_hw[0] * la_inputs.PATCH_SIZE)

    def prompt_len(self, query):
        """Token count of the prompt for `query` at the canonical grid (no device work)."""
        n_tok = la_inputs.num_image_tokens(self.grid_hw)
        enc = self.tokenizer([la_inputs.build_chat_text(query, n_tok)], return_tensors="pt")
        return int(enc["input_ids"].shape[1])

    def run(self, image: Image.Image, query: str, max_new_tokens: int = 128, ignore_eos: bool = False):
        """image (PIL) + query -> dict(raw_text, detections, points, timing_ms, ...).

        `ignore_eos=True` runs exactly `max_new_tokens - 1` decode steps regardless of EOS;
        the warm-up uses it so the decode programs are compiled and the trace captured.
        """
        t_start = time.perf_counter()
        query = str(query).strip()
        if not query:
            raise ValueError("query must be a non-empty string")
        max_new_tokens = int(max_new_tokens)
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be >= 1")
        image = image.convert("RGB")
        W, H = image.size

        # --- inputs (HF-faithful preprocessing, forced onto the canonical grid) ---
        bundle = la_inputs.build_inputs(self.tokenizer, image, query, in_token_limit=self.in_token_limit, grid_hw=self.grid_hw)
        input_ids = bundle["input_ids"]
        attention_mask = bundle["attention_mask"]
        real_seq_len = int(input_ids.shape[1])
        last_token_idx = real_seq_len - 1
        if real_seq_len + max_new_tokens > self.max_seq_len:
            raise ValueError(
                f"prompt ({real_seq_len} tokens) + max_new_tokens ({max_new_tokens}) exceeds max_seq_len {self.max_seq_len}; "
                "shorten the query or lower max_new_tokens"
            )

        # --- vision on device (MoonViT + projector) ---
        # legacy: -> [n_img_tokens, 2048] fp32 on host. Fused: the projection stays on device
        # (trace output when captured) and is only read back for prompts the fused LLM path
        # cannot serve. NOTE: with the vision trace the "vision" time is the non-blocking submit;
        # the wait moves into "prefill" -- compare totals across TT_FUSED settings.
        fused_llm = self._fused_prefill_applies(input_ids)
        t0 = time.perf_counter()
        if self.fused.enabled:
            vit_dev = self.vis.forward_device(bundle["pixel_values"].float())
            vit_proj = None if fused_llm else self.vis.read_projection(vit_dev)
        else:
            vit_dev = None
            vit_proj = self.vis.forward(bundle["pixel_values"].float()).to(torch.float32)
        t_vision = time.perf_counter() - t0

        if fused_llm:
            # --- fused K/L/N: ids upload -> device embed + merge -> (traced) layers -> eager tail ---
            t0 = time.perf_counter()
            first_tok, decode_start_pos, prefill_len = self._prefill_fused(input_ids, vit_dev, last_token_idx)
            prefill_lens = [prefill_len]
            if not self.vis.trace_captured:
                ttnn.deallocate(vit_dev)
            t_prefill = time.perf_counter() - t0
        else:
            if vit_dev is not None and not self.vis.trace_captured:
                ttnn.deallocate(vit_dev)
            # --- host text-embed + merge + prefill padding (qwen25_vl helpers) ---
            text_embeds = F.embedding(input_ids, self.embed_weight).to(torch.float32)  # [1, S, hidden]
            pad_embedding = self.embed_weight[self.pad_token_id].to(torch.float32)  # [hidden]
            input_embeds = merge_vision_tokens(input_ids, text_embeds, vit_proj.to(text_embeds.dtype), _MergeConfig())
            input_prefill_pt, decoding_pos, prefill_lens = preprocess_inputs_prefill(
                [input_embeds[0]], self.model_args, attention_mask, pad_embedding=pad_embedding
            )
            decode_start_pos = int(decoding_pos[0])
            embeds = input_prefill_pt[0].unsqueeze(0).to(torch.float32)  # [1, prefill_len, hidden]

            # --- prefill ---
            t0 = time.perf_counter()
            self.model.switch_mode(Mode.PREFILL)
            tokens_embd, rot_mats_global, tt_page_table, _ = self.model.prepare_inputs_prefill_embeds(
                embeds, start_pos=0, page_table=self.page_table, last_token_idx=last_token_idx
            )
            tt_logits = self.model.ttnn_prefill_forward(
                tokens_embd,
                rot_mats_global=rot_mats_global,
                rot_mats_local=None,
                user_id=0,
                page_table=tt_page_table,
                get_last_token=(last_token_idx // 32) * 32,
                kv_cache=self.tt_kv_cache,
            )
            prefill_last_logits = self.model.process_output_prefill(tt_logits.cpu(), last_token_idx=last_token_idx % 32)
            # Free prefill device tensors so their L1/DRAM regions don't clash with the decode
            # program's circular buffers (mirrors the stock prefill cleanup).
            ttnn.deallocate(tt_logits)
            ttnn.deallocate(tokens_embd)
            if tt_page_table is not None:
                ttnn.deallocate(tt_page_table)
            first_tok = int(torch.argmax(prefill_last_logits[: self.vocab_size]).item())
            t_prefill = time.perf_counter() - t0

        # --- greedy AR decode through the stock Generator (trace replay when enabled) ---
        # M (fused): SamplingParams(temperature=0, top_k=1, top_p=1) selects the library's
        # on-device force-argmax; token ids come back instead of 9.77 MB of logits and, after
        # step 0, the token/position feedback stays on device. The stale-token guard marks slot 0
        # as freshly prefilled so reset_batch takes OUR first token, never the device's leftover.
        t0 = time.perf_counter()
        sampling_params = greedy_sampling_params() if self._device_sampling else None
        generated_ids = [first_tok]
        out_tok = torch.tensor([[first_tok]], dtype=torch.int64)
        current_pos = torch.tensor([decode_start_pos], dtype=torch.int64)
        stopped_on_eos = first_tok == EOS_TOKEN_ID
        num_decode_steps = 0
        if ignore_eos or not stopped_on_eos:
            if sampling_params is not None:
                arm_stale_token_guard(self.generator, slot=0)
            for step in range(max_new_tokens - 1):
                out, _ = self.generator.decode_forward(
                    out_tok,
                    current_pos,
                    page_table=self.page_table,
                    kv_cache=[self.tt_kv_cache],
                    enable_trace=self.enable_trace,
                    reset_batch=(step == 0),
                    sampling_params=sampling_params,
                )
                num_decode_steps += 1
                if sampling_params is not None:
                    next_tok = int(out.reshape(-1)[0].item())  # token ids sampled on device
                else:
                    _, next_tok_t = sample_host(out, temperature=0, top_p=1.0, on_host=True)
                    next_tok = int(next_tok_t.reshape(-1)[0].item())
                current_pos = current_pos + 1
                generated_ids.append(next_tok)
                if next_tok == EOS_TOKEN_ID and not ignore_eos:
                    stopped_on_eos = True
                    break
                out_tok = torch.tensor([[next_tok]], dtype=torch.int64)
        t_decode = time.perf_counter() - t0

        answer = self.tokenizer.decode(generated_ids, skip_special_tokens=False)
        dets = parse_detections(answer, W, H)
        total = time.perf_counter() - t_start
        return {
            "query": query,
            "width": W,
            "height": H,
            "canonical_size": list(self.canonical_size_wh),
            "grid_hw": list(self.grid_hw),
            "num_image_tokens": int(bundle["n_img_tokens"]),
            "prompt_tokens": real_seq_len,
            "prefill_tokens": int(prefill_lens[0]),
            "raw_text": answer,
            "generated_ids": generated_ids,
            "detections": [d for d in dets if "box" in d],
            "points": [d for d in dets if "point" in d],
            "num_generated_tokens": len(generated_ids),
            "stopped_on_eos": bool(stopped_on_eos),
            "decode_mode": ("ar_greedy_trace" if self.enable_trace else "ar_greedy")
            + ("_device_sampling" if self._device_sampling else ""),
            "fused": self.fused_status if self.fused.enabled else None,
            "timing_ms": {
                "vision": round(t_vision * 1000, 1),
                "prefill": round(t_prefill * 1000, 1),
                "decode": round(t_decode * 1000, 1),
                "total": round(total * 1000, 1),
                "decode_tok_s": round(num_decode_steps / t_decode, 2) if t_decode > 0 and num_decode_steps else 0.0,
            },
        }

    def warmup(self, runs=2, decode_steps=4):
        """Compile every program and capture the decode trace before serving: one prefill +
        `decode_steps` unconditional decode steps per run (the benchmark's warm-up), on a
        synthetic image of the canonical size.

        With TT_FUSED=1 the vision and prefill traces are captured after run 1 (see
        ``capture_traces``) and at least two runs are made, so the last warm-up run replays every
        trace exactly the way a request will; a missing capture is a startup error, not a silent
        fallback."""
        img = la_inputs.synthetic_image(self.canonical_size_wh)
        runs = int(runs)
        fused_traces = self._fused_traces_wanted()
        if fused_traces and runs < 2:
            self.log("TT_FUSED=1: forcing 2 warm-up runs (traces are captured after run 1 and replayed in run 2)")
            runs = 2
        out = None
        for i in range(runs):
            t0 = time.perf_counter()
            out = self.run(img, "object", max_new_tokens=int(decode_steps) + 1, ignore_eos=True)
            self.log(
                f"Warmup run {i + 1}/{runs}: {time.perf_counter() - t0:.1f}s "
                f"(vision {out['timing_ms']['vision']} ms, prefill {out['timing_ms']['prefill']} ms, "
                f"decode {out['timing_ms']['decode']} ms for {decode_steps} steps)"
            )
            if i == 0 and fused_traces:
                self.capture_traces(img, "object")
        if fused_traces:
            status = self.fused_status
            self.log(f"TT_FUSED status after warm-up: {status}")
            if self.fused.vision_trace and not status["vision_trace"]:
                raise RuntimeError("TT_FUSED=1: the vision trace was not captured during warm-up")
            if self.fused.prefill_trace and status["device_merge"] and not status["prefill_trace"]:
                raise RuntimeError("TT_FUSED=1: the prefill trace was not captured during warm-up")
        return out

    def release(self):
        """Drop every device-tensor-holding object (the caller then closes the mesh device)."""
        for attr in ("generator", "vis", "tt_kv_cache", "page_table", "model", "model_args", "embed_weight", "_fused_prefill"):
            setattr(self, attr, None)
        gc.collect()
