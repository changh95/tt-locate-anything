# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""``TT_FUSED=1`` knob and the torch-only pieces of the fused (megakernel-style) paths.

Everything the fused paths need that is *not* a device op lives here so it can be unit-tested on
a host without ttnn: the env-knob parser (read ONCE at model build, never per request), the
tile-padding rule, the constant 0/1 tables that replace host gathers with exact device matmuls,
the on-device-sampling stale-token guard, and the validity checks for the SDPA / minimal_matmul
program configs quoted from the op validators in this tt-metal tree
(``sdpa_device_operation.cpp``: chunk sizes must be multiples of 32;
``minimal_matmul_device_operation.cpp``: blocks > 0, ``M_block % subblock_h == 0``,
``N_block % subblock_w == 0``, ``subblock_h * subblock_w <= max_dest_volume`` (4 tiles with the
port's fp32-accumulate compute config), grid >= 2x2).

Default ON since the device validation of 2026-09-13 (DEVICE_VALIDATION.md "Results"): with
``TT_FUSED`` unset or "1" :meth:`FusedConfig.from_env` enables the fused path; ``TT_FUSED=0`` returns
:meth:`FusedConfig.disabled` and every consumer takes the legacy path, bit-for-bit the 2026-09-12
shipped behaviour. The DEFAULT fused configuration keeps the legacy vision numerics bit for bit
(device-verified: `torch.equal` on the projector output, identical PCC to every digit) and adds only
the exact levers: A (vision trace), B (device patch merger), J (row-major upload), K (device
vision->LLM merge), L (prefill trace), N (one-row first token), M (on-device greedy argmax).
Measured on the p150a (host pipeline, demo image + `car`, 10 tokens): 360.9 -> 304.9 ms.

Sub-knobs (read only when the path is enabled) for the A/B and for the two documented, measured
opt-ins that change the vision numerics (NOT default; see DEVICE_VALIDATION.md 3.3/3.4):

  * "fast vision" `LA_FUSED_EXACT_SEQ=1 LA_FUSED_SDPA_CHUNKS=96,352 LA_FUSED_SDPA_EXP_APPROX=0`
    (levers C + F): 304.9 -> 291.7 ms; projector / full-logits PCC vs the torch golden 0.99244 /
    0.99286 (legacy 0.99240 / 0.99189) and the demo box becomes the HF reference's
    `<282><414><607><797>` -- but it no longer equals the shipped legacy string `<606><794>`
    (DEVICE_VALIDATION.md gate G1 as written), so the owner decides.
  * `LA_FUSED_MATMUL=minimal` (levers D/E): another -3.6 ms of vision, full-logits PCC 0.98997
    on the port golden (below the 0.99 gate) -> not default.

  LA_FUSED_VISION_TRACE   1|0  whole MoonViT graph as ONE metal trace (default 1)
  LA_FUSED_MATMUL         linear|minimal  ttnn.linear (default) vs minimal_matmul /
                          dit_minimal_matmul_addcmul_fused kernels for the vision matmuls (+ fused residual add)
  LA_FUSED_SDPA_CHUNKS    "q,k" SDPAProgramConfig chunk sizes, multiples of 32 (default "32,32" = the
                          kernel default the legacy call gets without a program_config; "96,352" is the
                          measured fast setting; "96,1056" clashes in L1 next to the resident LLM)
  LA_FUSED_SDPA_EXP_APPROX 1|0 SDPAProgramConfig.exp_approx_mode (default 1 = the kernel default the
                          legacy call gets; 0 = exact exp, measured PCC-neutral-to-up at no cost)
  LA_FUSED_MM_BLOCKS      "M,K,N,sub_h,sub_w" MinimalMatmulConfig for plain matmuls (default "8,4,4,2,2";
                          sub_h * sub_w <= 4 with the port's HiFi4 + fp32_dest_acc_en config)
  LA_FUSED_DIT_BLOCKS     same for the fused matmul+residual op (default "4,4,4,2,2")
  LA_FUSED_GELU           erf|tanh  vision GELU variant (default erf = the legacy op; tanh is the
                          reference's PytorchGELUTanh but precision-affecting -> gate on device)
  LA_FUSED_L1             1|0  vision working set in L1 instead of DRAM (default 0)
  LA_FUSED_EXACT_SEQ      1|0  exact tile-aligned sequence (no 128-padding, no SDPA mask) when L % 32 == 0;
                          default 0 = keep the legacy 128-padding + mask INSIDE the fused graph, so the
                          default `linear` + `32,32` + `EXP_APPROX=1` + `erf` + `EXACT_SEQ=0` runs the
                          legacy vision numerics op for op (device-verified bit-identical) with only the
                          exact levers A/B/J on top; 1 = lever C (-7 ms, changes the bf16 rounding)
  LA_FUSED_RM_INPUT       1|0  ROW_MAJOR pixel upload + in-graph tilize (default 1)
  LA_FUSED_DEVICE_MERGE   1|0  vision->LLM embedding merge on device (default 1)
  LA_FUSED_PREFILL_TRACE  1|0  36-layer prefill as one metal trace + eager norm/LM-head tail (default 1)
  LA_FUSED_SMALL_READBACK 1|0  first-token logits: read one row instead of 32 x vocab (default 1)
  LA_FUSED_DEVICE_SAMPLING 1|0 greedy argmax on device in the decode loop (default 1)
"""
from __future__ import annotations

import math
import os
from dataclasses import asdict, dataclass

import torch

ENV_KNOB = "TT_FUSED"

# Trace region: the shipped 50 MB holds the decode trace only; with the knob on there are up to
# four traces (vision, prefill, decode, sampling). Unverifiable on host -> generous default,
# LA_TRACE_REGION_SIZE still wins (see FusedConfig.trace_region_size).
LEGACY_TRACE_REGION_SIZE = 50_000_000
FUSED_TRACE_REGION_SIZE = 160_000_000

TILE = 32
LEGACY_PAD_MULTIPLE = 128

# minimal_matmul_device_operation.cpp (validate_on_program_cache_miss):
#   TT_FATAL(cfg.subblock_h * cfg.subblock_w <= get_dest_reg_count(compute_kernel_config))
# compute_kernel_config.cpp get_dest_reg_count: (DEST_REGISTER_FULL_SIZE * DATUMS_PER_ROW) / (32*32)
# = 16 tiles, halved for dst_full_sync_en=False (the WormholeComputeKernelConfig default) and
# halved again for fp32_dest_acc_en=True -> 4 tiles for the port's ck_hifi4 (vision.py), 8 without
# fp32 accumulation (rf-detr's setting, hence its 2x2 subblocks had margin this port does not).
MM_MAX_DEST_TILES_FP32 = 4
MM_MAX_DEST_TILES_BF16 = 8

# sdpa_program_factory.cpp: q/k chunk sizes without a program config are 32/32 and
# get_exp_approx_mode() "defaults to true"; the fused path always passes a program config.
LEGACY_SDPA_CHUNKS = (32, 32)
LEGACY_SDPA_EXP_APPROX = True


def _parse_int_tuple(text: str, n: int, name: str) -> tuple:
    parts = [p.strip() for p in str(text).split(",")]
    if len(parts) != n or not all(p.lstrip("-").isdigit() for p in parts):
        raise ValueError(f"{name}={text!r} must be {n} comma-separated integers")
    return tuple(int(p) for p in parts)


def validate_sdpa_chunks(q_chunk: int, k_chunk: int) -> None:
    """sdpa_device_operation.cpp: q/k chunk sizes must be > 0 and divisible by TILE_SIZE (32)."""
    for name, v in (("q_chunk_size", q_chunk), ("k_chunk_size", k_chunk)):
        if v <= 0 or v % TILE != 0:
            raise ValueError(f"SDPA {name}={v} must be a positive multiple of {TILE}")


def validate_mm_blocks(blocks, fp32_dest_acc: bool = True) -> None:
    """minimal_matmul_device_operation.cpp config checks (grid >= 2x2 is checked on device).

    ``fp32_dest_acc`` selects the dest-register cap of the compute config the blocks will run with
    (the port's vision matmuls use ``fp32_dest_acc_en=True`` -> ``subblock_h * subblock_w <= 4``)."""
    if len(blocks) != 5:
        raise ValueError(f"MinimalMatmulConfig needs (M,K,N,subblock_h,subblock_w), got {blocks}")
    m, k, n, sh, sw = blocks
    if min(m, k, n) <= 0:
        raise ValueError(f"block sizes must be > 0, got M={m} K={k} N={n}")
    if min(sh, sw) <= 0:
        raise ValueError(f"subblock sizes must be > 0, got {sh}x{sw}")
    if m % sh != 0:
        raise ValueError(f"M_block_size ({m}) must be divisible by subblock_h ({sh})")
    if n % sw != 0:
        raise ValueError(f"N_block_size ({n}) must be divisible by subblock_w ({sw})")
    cap = MM_MAX_DEST_TILES_FP32 if fp32_dest_acc else MM_MAX_DEST_TILES_BF16
    if sh * sw > cap:
        raise ValueError(
            f"subblock_h * subblock_w ({sh}x{sw}={sh * sw}) exceeds the {cap}-tile dest register volume "
            f"(fp32_dest_acc_en={fp32_dest_acc}, dst_full_sync_en=False)"
        )


@dataclass(frozen=True)
class FusedConfig:
    """Snapshot of the TT_FUSED knob family, taken once at model build."""

    enabled: bool = False
    vision_trace: bool = False
    matmul: str = "linear"
    sdpa_chunks: tuple = LEGACY_SDPA_CHUNKS  # kernel default; only passed to the op when enabled
    sdpa_exp_approx: bool = LEGACY_SDPA_EXP_APPROX  # kernel default; only passed when enabled
    mm_blocks: tuple = (8, 4, 4, 2, 2)
    dit_blocks: tuple = (4, 4, 4, 2, 2)
    gelu: str = "erf"
    l1: bool = False
    exact_seq: bool = False  # only consulted when enabled (vision_seq_pad(n, enabled and exact_seq)); 1 = lever C
    rowmajor_input: bool = False
    device_merge: bool = False
    prefill_trace: bool = False
    small_readback: bool = False
    device_sampling: bool = False

    @classmethod
    def disabled(cls) -> "FusedConfig":
        return cls()

    @classmethod
    def from_env(cls, env=None) -> "FusedConfig":
        env = os.environ if env is None else env
        # Default ON (2026-09-13 device validation); TT_FUSED=0 (or any value other than "1") = legacy.
        if str(env.get(ENV_KNOB, "1")).strip() != "1":
            return cls.disabled()

        def flag(name, default):
            return str(env.get(name, default)).strip() == "1"

        matmul = str(env.get("LA_FUSED_MATMUL", "linear")).strip()
        if matmul not in ("minimal", "linear"):
            raise ValueError(f"LA_FUSED_MATMUL={matmul!r} must be 'minimal' or 'linear'")
        gelu = str(env.get("LA_FUSED_GELU", "erf")).strip()
        if gelu not in ("erf", "tanh"):
            raise ValueError(f"LA_FUSED_GELU={gelu!r} must be 'erf' or 'tanh'")
        cfg = cls(
            enabled=True,
            vision_trace=flag("LA_FUSED_VISION_TRACE", "1"),
            matmul=matmul,
            sdpa_chunks=_parse_int_tuple(env.get("LA_FUSED_SDPA_CHUNKS", "32,32"), 2, "LA_FUSED_SDPA_CHUNKS"),
            sdpa_exp_approx=flag("LA_FUSED_SDPA_EXP_APPROX", "1"),
            mm_blocks=_parse_int_tuple(env.get("LA_FUSED_MM_BLOCKS", "8,4,4,2,2"), 5, "LA_FUSED_MM_BLOCKS"),
            dit_blocks=_parse_int_tuple(env.get("LA_FUSED_DIT_BLOCKS", "4,4,4,2,2"), 5, "LA_FUSED_DIT_BLOCKS"),
            gelu=gelu,
            l1=flag("LA_FUSED_L1", "0"),
            exact_seq=flag("LA_FUSED_EXACT_SEQ", "0"),
            rowmajor_input=flag("LA_FUSED_RM_INPUT", "1"),
            device_merge=flag("LA_FUSED_DEVICE_MERGE", "1"),
            prefill_trace=flag("LA_FUSED_PREFILL_TRACE", "1"),
            small_readback=flag("LA_FUSED_SMALL_READBACK", "1"),
            device_sampling=flag("LA_FUSED_DEVICE_SAMPLING", "1"),
        )
        cfg.validate()
        return cfg

    def validate(self) -> None:
        validate_sdpa_chunks(*self.sdpa_chunks)
        # both block configs run under vision.py's ck_hifi4 (fp32_dest_acc_en=True) -> 4-tile cap
        validate_mm_blocks(self.mm_blocks, fp32_dest_acc=True)
        validate_mm_blocks(self.dit_blocks, fp32_dest_acc=True)

    @property
    def trace_region_size(self) -> int:
        return FUSED_TRACE_REGION_SIZE if self.enabled else LEGACY_TRACE_REGION_SIZE

    def as_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------- padding rules
def seq_pad_len(n: int, exact_tiles: bool) -> int:
    """Padded row count for ``n`` rows: the port's legacy ``ceil(n / 128) * 128``
    (``exact_tiles=False``) or the smallest tile multiple (``True``)."""
    n = int(n)
    mult = TILE if exact_tiles else LEGACY_PAD_MULTIPLE
    return int(math.ceil(n / mult) * mult)


def vision_seq_pad(n: int, fused_enabled: bool) -> int:
    """MoonViT row padding (sequence and merged rows): exact when the fused path is on AND ``n``
    is already a tile multiple (served grid: L=1056 -> 1056, no mask), otherwise the legacy
    128-rule in both modes (test golden 26x42: 1092 -> 1152 with the legacy mask; merged rows
    264 -> 384). Only tile-aligned lengths change, so the padded+masked path stays the shipped one."""
    n = int(n)
    if fused_enabled and n % TILE == 0:
        return n
    return seq_pad_len(n, exact_tiles=False)


def prefill_bucket_len(seq_len: int, max_prefill_len: int) -> int:
    """qwen25_vl ``preprocess_inputs_prefill``: nearest power of two, at least 128, capped."""
    return int(min(max_prefill_len, max(2 ** math.ceil(math.log(seq_len, 2)), 128)))


# --------------------------------------------------------------------------- patch merger (B)
def patch_merge_host(x: torch.Tensor, grid_hw, merge=(2, 2)) -> torch.Tensor:
    """Reference 2x2 spatial merge (modeling_vit.patch_merger): [>=L, C] -> [L/4, 4*C].

    Rows beyond L (padding) are ignored. This is the legacy host implementation the port
    validated; :func:`build_merge_perms` reproduces it as 0/1 matmuls."""
    h, w = int(grid_hw[0]), int(grid_hw[1])
    kh, kw = merge
    nh, nw = h // kh, w // kw
    C = x.shape[-1]
    seq = x[: h * w].reshape(nh, kh, nw, kw, C)
    return seq.permute(0, 2, 1, 3, 4).contiguous().reshape(nh * nw, kh * kw * C)


def merge_source_rows(grid_hw, merge=(2, 2)) -> torch.Tensor:
    """LongTensor [L/4, 4]: ``rows[n, j]`` is the encoder row that lands in output row ``n``,
    column block ``j = kh * merge_w + kw``."""
    h, w = int(grid_hw[0]), int(grid_hw[1])
    kh, kw = merge
    nh, nw = h // kh, w // kw
    rows = torch.empty(nh * nw, kh * kw, dtype=torch.long)
    for a in range(nh):
        for c in range(nw):
            for b in range(kh):
                for d in range(kw):
                    rows[a * nw + c, b * kw + d] = (kh * a + b) * w + (kw * c + d)
    return rows


def build_merge_perms(grid_hw, seq_pad: int, merged_pad: int, merge=(2, 2)) -> list:
    """Four 0/1 permutation matrices ``P_j`` (float32 ``[merged_pad, seq_pad]``) such that
    ``concat([P_0 @ X, .., P_3 @ X], dim=-1) == pad(patch_merge_host(X))`` for the encoder output
    ``X [seq_pad, C]``: one 1.0 per real output row, all-zero rows for the padding rows (exact
    zero pad, like the legacy ``F.pad``), padding columns never selected."""
    rows = merge_source_rows(grid_hw, merge)
    n_out, n_blocks = rows.shape
    L = int(grid_hw[0]) * int(grid_hw[1])
    if n_out > merged_pad or L > seq_pad:
        raise ValueError(f"merge perms: {n_out} rows / {L} cols do not fit {merged_pad} x {seq_pad}")
    perms = []
    for j in range(n_blocks):
        P = torch.zeros(merged_pad, seq_pad, dtype=torch.float32)
        P[torch.arange(n_out), rows[:, j]] = 1.0
        perms.append(P)
    return perms


# --------------------------------------------------------------------------- vision->LLM merge (K)
def image_token_rows(input_ids: torch.Tensor, image_token_id: int) -> torch.Tensor:
    """Positions of the image placeholder tokens in a [S] or [1, S] id tensor (ascending)."""
    return torch.nonzero(input_ids.reshape(-1) == int(image_token_id), as_tuple=False).reshape(-1)


def build_embed_merge_table(image_rows: torch.Tensor, prefill_len: int, n_img_pad: int) -> torch.Tensor:
    """0/1 matrix ``P [prefill_len, prefill_len + n_img_pad]`` with
    ``P @ concat([E_ids, vit_proj_pad], dim=0) == pad(merge_vision_tokens(ids, E_ids, vit_proj))``:
    text (and pad-token) rows select their own embedding row ``i``; the ``t``-th image row selects
    vision row ``prefill_len + t``. Exact: one 1.0 per row, everything else 0."""
    image_rows = image_rows.reshape(-1).long()
    n_img = int(image_rows.numel())
    if n_img > n_img_pad:
        raise ValueError(f"{n_img} image tokens do not fit the {n_img_pad}-row vision buffer")
    if n_img and (int(image_rows.max()) >= prefill_len or int(image_rows.min()) < 0):
        raise ValueError("image rows outside the prefill bucket")
    P = torch.zeros(prefill_len, prefill_len + n_img_pad, dtype=torch.float32)
    is_img = torch.zeros(prefill_len, dtype=torch.bool)
    is_img[image_rows] = True
    text_rows = torch.nonzero(~is_img, as_tuple=False).reshape(-1)
    P[text_rows, text_rows] = 1.0
    P[image_rows, prefill_len + torch.arange(n_img)] = 1.0
    return P


def pad_prompt_ids(input_ids: torch.Tensor, prefill_len: int, pad_token_id: int) -> torch.Tensor:
    """[1, S] -> [prefill_len] int64: the prompt followed by ``pad_token_id`` (whose embedding row
    is exactly the ``pad_embedding`` qwen25_vl's ``preprocess_inputs_prefill`` fills with)."""
    ids = input_ids.reshape(-1).long()
    if ids.numel() > prefill_len:
        raise ValueError(f"prompt of {ids.numel()} tokens exceeds the {prefill_len}-token bucket")
    return torch.nn.functional.pad(ids, (0, prefill_len - ids.numel()), value=int(pad_token_id))


# --------------------------------------------------------------------------- decode sampling (M)
def arm_stale_token_guard(generator, slot: int = 0) -> set:
    """Mark ``slot`` as freshly prefilled before the first on-device-sampling decode step.

    ``Generator.decode_forward`` (generator.py, reset_batch branch) keeps the *device* token of a
    slot whenever the device position equals the host position (or host+1) unless the slot is in
    ``_slots_prefilled_since_decode``; the library's ``prefill_forward_text`` fills that set but
    this port prefills outside the Generator, so a request whose prompt length equals the previous
    request's final position would otherwise resume from a stale token. Returns the new set."""
    prefilled = getattr(generator, "_slots_prefilled_since_decode", None)
    new = set(prefilled) if prefilled else set()
    new.add(int(slot))
    generator._slots_prefilled_since_decode = new
    return new


def greedy_sampling_params():
    """SamplingParams that TTSampling maps to force-argmax on a single chip:
    ``format_sampling_params`` rewrites temperature 0 to (temp 1.0, top_k 1, top_p 0.0), which is
    exactly ``TTSampling._is_force_argmax_sampling`` (allowed when ``num_devices == 1``)."""
    from models.common.sampling import SamplingParams  # lazy: library import (needs ttnn)

    return SamplingParams(temperature=0.0, top_k=1, top_p=1.0)
