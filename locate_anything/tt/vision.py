# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""TT-NN port of NVIDIA LocateAnything-3B vision tower (MoonViT-SO-400M) + mlp1 projector.

Single Blackhole p150a, batch=1, single image. PRECISION FIRST: bf16 activations /
weights with HiFi4 math fidelity (fp32 dest accumulate) on every matmul + SDPA.

Architecture (mirrors ~/.cache/.../modeling_vit.py, authoritative):
  patch_embed:  Conv2d(3,1152,k=14,s=14) == matmul([L,588] @ [588,1152]) + bias,
                then + bicubic-interpolated Learnable2DInterpPosEmb (host one-time const).
  encoder:      27 x MoonVitEncoderLayer (LayerNorm eps=1e-5, attn_bias=True):
                  x = x + wo(attn(norm0(x))) ;  x = x + mlp(norm1(x))
                attn: fused wqkv(1152->3456) -> 16 heads x head_dim 72 (pad 96),
                      2D-RoPE (interleaved complex convention) on q,k,
                      full bidirectional SDPA (one window, cu_seqlens=[0,L]),
                      wo(1152->1152)+bias.
                mlp:  fc0(1152->4304) -> GELU(tanh) -> fc1(4304->1152).
                final_layernorm after the 27 blocks.
  patch_merger: 2x2 spatial merge -> [L/4, 4608].
  mlp1:         LayerNorm(4608) -> Linear(4608,2048) -> GELU -> Linear(2048,2048).

RoPE gotcha (validated against torch apply_rope at PCC 1.0, and on-device at 0.99999):
  MoonViT uses the *interleaved complex* convention (view_as_complex over adjacent
  pairs), which is EXACTLY what ttnn.experimental.rotary_embedding_llama implements
  given cos/sin built as repeat_interleave of Re/Im(freqs_cis). head_dim 72 is padded
  to 96 with cos=1, sin=0 so the padded lanes are an identity rotation.

Two execution paths, selected ONCE at construction (``fused`` / env ``TT_FUSED``, see
``locate_anything/tt/fused.py``):

* legacy (``TT_FUSED=0``): the 2026-09-12 shipped graph, bit-for-bit -- 386 eager ops on a
  128-padded sequence with an explicit SDPA padding mask, ``ttnn.linear`` everywhere, host
  ``patch_merger`` between the encoder readback and the mlp1 upload, host tilize of the pixels.
* fused (default since the 2026-09-13 device validation, ``TT_FUSED`` unset or ``1``): the rf-detr
  pattern library applied to the same math. The DEFAULT sub-knobs keep the legacy vision numerics
  bit for bit (device-verified ``torch.equal`` on the projector output) and add only the exact levers:
  A. the WHOLE graph (tilize -> patch-embed -> 27 blocks -> final LN -> merge -> mlp1) is one
     metal trace with a persistent device input (``capture_trace`` / ``forward_device``);
  B. the patch merger runs on device as four 0/1 permutation matmuls + one concat (exact);
  J. ROW_MAJOR pixel upload + in-graph ``tilize_with_zero_padding`` (no host tilize).
  Measured opt-ins that change the bf16 rounding (see DEVICE_VALIDATION.md "Results"):
  C. ``LA_FUSED_EXACT_SEQ=1``: a tile-aligned L (served 24x44 -> 1056 = 33 tiles) is not padded
     and the SDPA gets NO attention mask (the kernel masks nothing because nothing is padded);
  F. ``LA_FUSED_SDPA_CHUNKS=96,352 LA_FUSED_SDPA_EXP_APPROX=0``: SDPAProgramConfig chunks + exact exp
     (the default ``32,32`` + approx exp is what the legacy call gets without a program config);
  D/E. ``LA_FUSED_MATMUL=minimal``: ``dit_minimal_matmul_addcmul_fused`` folds the residual add into
     wo / fc1 / patch-embed (+pos_emb), ``minimal_matmul`` replaces ``ttnn.linear`` for wqkv / fc0 /
     mlp1 (full-logits PCC below the 0.99 gate on the port golden -> not default);
  H/I. ``LA_FUSED_GELU=tanh`` (the reference's variant; lower PCC on device), ``LA_FUSED_L1=1``
     (working set in L1; changes ``ttnn.linear`` numerics on the padded graph, below gate).
  Every fused op keeps the port's HiFi4 + fp32-accumulate compute config; the permutation
  matmuls use HiFi4 without fp32 accumulation (exact for 0/1 x bf16, as verified on device by
  the rf-detr port).
"""

import glob
import math
import os

import torch
import torch.nn.functional as F
from safetensors import safe_open

import ttnn
from locate_anything.tt.fused import FusedConfig, build_merge_perms, patch_merge_host, vision_seq_pad

HIDDEN = 1152
N_LAYERS = 27
N_HEADS = 16
HEAD_DIM = 72
PAD_HEAD_DIM = 96  # tile-aligned (multiple of 32)
INTERMEDIATE = 4304
PATCH = 14
MERGE = (2, 2)
LN_EPS = 1e-5
THETA_BASE = 10000.0
POS_EMB_HW = 64
MLP1_IN = HIDDEN * MERGE[0] * MERGE[1]  # 4608
PROJ_OUT = 2048
PATCH_DIM = 3 * PATCH * PATCH  # 588 flattened (c, kh, kw) pixels per patch
PATCH_DIM_PAD = 608  # tile-aligned width the ROW_MAJOR upload is zero-padded to (fused path J)


def _load_vision_state_dict(model_path):
    """Load only vision_model.* and mlp1.* tensors from the HF safetensors snapshot."""
    sd = {}
    for st in sorted(glob.glob(os.path.join(model_path, "*.safetensors"))):
        with safe_open(st, "pt") as f:
            for k in f.keys():
                if k.startswith("vision_model.") or k.startswith("mlp1."):
                    sd[k] = f.get_tensor(k)
    assert sd, f"No vision/mlp1 weights found under {model_path}"
    return sd


def _precompute_freqs_cis(head_dim, max_h, max_w, theta_base=THETA_BASE):
    """Exact port of Rope2DPosEmb._precompute_freqs_cis (returns [max_h, max_w, head_dim/2] complex)."""
    N = max_h * max_w
    flat_pos = torch.arange(0, N).float()
    x_pos = flat_pos % max_w
    y_pos = flat_pos // max_w
    dim_range = torch.arange(0, head_dim, 4)[: (head_dim // 4)].float()  # C/4
    freqs = 1.0 / (theta_base ** (dim_range / head_dim))
    x_freqs = torch.outer(x_pos, freqs).float()
    y_freqs = torch.outer(y_pos, freqs).float()
    x_cis = torch.polar(torch.ones_like(x_freqs), x_freqs)
    y_cis = torch.polar(torch.ones_like(y_freqs), y_freqs)
    freqs_cis = torch.cat([x_cis.unsqueeze(-1), y_cis.unsqueeze(-1)], dim=-1)
    return freqs_cis.reshape(max_h, max_w, -1)  # [max_h, max_w, head_dim/2]


def build_rope_cos_sin(grid_hw, head_dim=HEAD_DIM, pad_head_dim=PAD_HEAD_DIM):
    """Host cos/sin for ttnn.experimental.rotary_embedding_llama (interleaved convention).

    Returns cos,sin torch tensors of shape [1, 1, L, pad_head_dim], padded lanes = identity.
    """
    h, w = int(grid_hw[0]), int(grid_hw[1])
    fc = _precompute_freqs_cis(head_dim, max(h, POS_EMB_HW), max(w, POS_EMB_HW))
    fc = fc[:h, :w].reshape(-1, head_dim // 2)  # [L, head_dim/2] complex
    cos = torch.repeat_interleave(fc.real, 2, dim=-1)  # [L, head_dim]
    sin = torch.repeat_interleave(fc.imag, 2, dim=-1)
    cos_p = F.pad(cos, (0, pad_head_dim - head_dim), value=0.0)
    cos_p[:, head_dim:] = 1.0  # identity rotation on padded lanes (cos=1)
    sin_p = F.pad(sin, (0, pad_head_dim - head_dim), value=0.0)  # sin=0
    return cos_p.unsqueeze(0).unsqueeze(0), sin_p.unsqueeze(0).unsqueeze(0)


def build_patch_embed_const(state_dict, grid_hw):
    """Host: conv weight (flattened to matmul) + per-position interpolated pos_emb.

    Returns (proj_w [588,1152], proj_b [1152], pos_emb [L,1152]).
    """
    h, w = int(grid_hw[0]), int(grid_hw[1])
    conv_w = state_dict["vision_model.patch_embed.proj.weight"].float()  # [1152,3,14,14]
    conv_b = state_dict["vision_model.patch_embed.proj.bias"].float()  # [1152]
    proj_w = conv_w.reshape(conv_w.shape[0], -1).t().contiguous()  # [588,1152]
    pos = state_dict["vision_model.patch_embed.pos_emb.weight"].float()  # [64,64,1152]
    if (h, w) == (POS_EMB_HW, POS_EMB_HW):
        pos_emb = pos.reshape(-1, HIDDEN)
    else:
        pos_emb = (
            F.interpolate(pos.permute(2, 0, 1).unsqueeze(0), size=(h, w), mode="bicubic")
            .squeeze(0)
            .permute(1, 2, 0)
            .reshape(-1, HIDDEN)
        )
    return proj_w, conv_b, pos_emb


def _pad_per_head(t_2d_or_1d, n_heads, head_dim, pad_head_dim):
    """Pad a packed-per-head weight/bias tensor's head_dim from head_dim->pad_head_dim with zeros.

    For a 2D weight the LAST dim is the packed (n_heads*head_dim) output; for 1D it's the only dim.
    """
    if t_2d_or_1d.dim() == 2:
        in_dim = t_2d_or_1d.shape[0]
        t = t_2d_or_1d.reshape(in_dim, n_heads, head_dim)
        t = F.pad(t, (0, pad_head_dim - head_dim))
        return t.reshape(in_dim, n_heads * pad_head_dim)
    else:
        t = t_2d_or_1d.reshape(n_heads, head_dim)
        t = F.pad(t, (0, pad_head_dim - head_dim))
        return t.reshape(-1)


def build_attn_mask(L, seq_pad):
    """Legacy additive SDPA mask over the padded sequence (only when seq_pad > L): real tokens
    attend to all real tokens and never to padding columns; padding rows attend to themselves only
    (keeps their softmax finite). fp32 [1,1,seq_pad,seq_pad]."""
    mask = torch.zeros(1, 1, seq_pad, seq_pad, dtype=torch.float32)
    mask[:, :, :, L:] = float("-inf")  # no token may attend to padding cols
    mask[:, :, L:, :] = float("-inf")  # padding rows attend to nothing (avoid NaN: keep diag)
    for i in range(L, seq_pad):
        mask[0, 0, i, i] = 0.0
    return mask


class MoonViT:
    """TT-NN MoonViT vision tower + mlp1 projector for a single image on one device.

    ``fused`` (a :class:`FusedConfig`; ``None`` = read ``TT_FUSED`` from the environment once)
    selects the legacy or the fused path for the life of the object -- see the module docstring.
    Public surface used by the pipeline/tests: ``forward`` (host result), ``forward_device`` /
    ``read_projection`` (device result, fused pipeline), ``capture_trace``, ``patch_merger``,
    ``L``, ``seq_pad``, ``merged_pad``, ``nmerged``.
    """

    def __init__(self, device, model_path, grid_hw, dtype=ttnn.bfloat16, fused=None):
        self.device = device
        self.dtype = dtype
        self.grid_hw = (int(grid_hw[0]), int(grid_hw[1]))
        self.L = self.grid_hw[0] * self.grid_hw[1]
        self.scale = HEAD_DIM**-0.5  # NOTE: real head_dim (72), not padded
        self.fused = fused if fused is not None else FusedConfig.from_env()
        f = self.fused

        # Row padding (lever C): legacy pads to 128 and masks the padding columns; the fused path
        # keeps a tile-aligned L exact (no padding, no mask) and otherwise uses the legacy rule.
        # LA_FUSED_EXACT_SEQ=0 keeps the legacy padding + mask inside the fused graph (A/B: with
        # linear matmuls and the legacy SDPA config that is the legacy vision numerics, bit for bit).
        exact_seq = f.enabled and f.exact_seq
        self.seq_pad = vision_seq_pad(self.L, exact_seq)
        self.nmerged = self.L // (MERGE[0] * MERGE[1])
        self.merged_pad = vision_seq_pad(self.nmerged, exact_seq)
        # Working-set placement (lever I): DRAM interleaved unless LA_FUSED_L1=1. Legacy calls pass
        # DRAM explicitly where they always did and nothing elsewhere (``self._mc`` is empty).
        self.mem = ttnn.L1_MEMORY_CONFIG if (f.enabled and f.l1) else ttnn.DRAM_MEMORY_CONFIG
        self._mc = {"memory_config": self.mem} if f.enabled else {}

        # Precision-first: HiFi4 + fp32 dest accumulate on every matmul / SDPA.
        self.ck_hifi4 = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        self.ck_sdpa = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

        sd = _load_vision_state_dict(model_path)
        self.state_dict = sd

        # --- patch_embed host consts ---
        proj_w, proj_b, pos_emb = build_patch_embed_const(sd, self.grid_hw)
        self.patch_dim = PATCH_DIM
        if f.enabled and f.rowmajor_input:
            # J: the ROW_MAJOR upload is zero-padded to 608 columns and tilized on device; give
            # the weight matching zero rows (0 * 0 contributes exactly nothing).
            self.patch_dim = PATCH_DIM_PAD
            proj_w = F.pad(proj_w, (0, 0, 0, PATCH_DIM_PAD - PATCH_DIM))
        self.proj_w = self._to_dev(proj_w)  # [588 or 608, 1152]
        self.proj_b = self._to_dev(proj_b.reshape(1, -1))  # [1,1152]
        if f.enabled and self.seq_pad > self.L:
            pos_emb = F.pad(pos_emb, (0, 0, 0, self.seq_pad - self.L))  # exact zeros, saves the in-graph pad
        self.pos_emb = self._to_dev(pos_emb.reshape(1, 1, -1, HIDDEN))  # [1,1,L or seq_pad,HIDDEN]

        # --- rope cos/sin (always bf16: rotary_embedding_llama requires bf16) ---
        cos, sin = build_rope_cos_sin(self.grid_hw)
        self.rope_cos = self._to_dev(cos, dtype=ttnn.bfloat16)  # [1,1,L,pad_head_dim]
        self.rope_sin = self._to_dev(sin, dtype=ttnn.bfloat16)

        # --- attention mask (single full window over the real L tokens) ---
        # Plain non-causal SDPA + additive mask: real tokens attend to all real tokens
        # (full bidirectional), and never to padding rows. Padding-row outputs are sliced off.
        # With the fused exact sequence (seq_pad == L) there is nothing to mask and the SDPA
        # runs mask-free (the provided-mask path was numerically wrong on Blackhole in rf-detr).
        if self.seq_pad > self.L:
            self.attn_mask = self._to_dev(build_attn_mask(self.L, self.seq_pad), dtype=ttnn.bfloat16)
        else:
            self.attn_mask = None
        # transformation matrix for the interleaved rotary op (single tile)
        from models.tt_transformers.tt.common import get_rot_transformation_mat

        self.rope_trans = ttnn.from_torch(
            get_rot_transformation_mat(),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )

        # --- per-block weights ---
        self.blocks = [self._load_block(sd, i) for i in range(N_LAYERS)]

        # --- final layernorm ---
        self.final_ln_w = self._to_dev(sd["vision_model.encoder.final_layernorm.weight"].reshape(1, -1))
        self.final_ln_b = self._to_dev(sd["vision_model.encoder.final_layernorm.bias"].reshape(1, -1))

        # --- mlp1 projector ---
        self.mlp1_ln_w = self._to_dev(sd["mlp1.0.weight"].reshape(1, -1))  # LayerNorm(4608)
        self.mlp1_ln_b = self._to_dev(sd["mlp1.0.bias"].reshape(1, -1))
        self.mlp1_w1 = self._to_dev(sd["mlp1.1.weight"].t().contiguous())  # [4608,2048]
        self.mlp1_b1 = self._to_dev(sd["mlp1.1.bias"].reshape(1, -1))
        self.mlp1_w2 = self._to_dev(sd["mlp1.3.weight"].t().contiguous())  # [2048,2048]
        self.mlp1_b2 = self._to_dev(sd["mlp1.3.bias"].reshape(1, -1))

        # --- fused-path constants and program configs ---
        self._trace_id = None
        self._persistent_in = None
        self._trace_out = None
        if f.enabled:
            self._init_fused()

    # ------------------------------------------------------------------ #
    def _init_fused(self):
        f = self.fused
        grid = self.device.compute_with_storage_grid_size()
        q_chunk, k_chunk = f.sdpa_chunks
        # Lever F is precision-affecting on two counts, both explicit here: the chunk sizes (softmax
        # rescale boundaries; the legacy call passes no program_config = kernel default 32/32) and
        # exp_approx_mode (kernel default WITHOUT a program config is True, sdpa_program_factory.cpp
        # get_exp_approx_mode; the fused default is the exact exp rf-detr ran at no measurable cost).
        # LA_FUSED_SDPA_CHUNKS=32,32 LA_FUSED_SDPA_EXP_APPROX=1 reproduces the legacy kernel config.
        self.sdpa_pc = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=grid,
            q_chunk_size=q_chunk,
            k_chunk_size=k_chunk,
            exp_approx_mode=bool(f.sdpa_exp_approx),
        )
        m, k, n, sh, sw = f.mm_blocks
        self.mm_config = ttnn.MinimalMatmulConfig(
            M_block_size=m, K_block_size=k, N_block_size=n, subblock_h=sh, subblock_w=sw,
            compute_with_storage_grid_size=grid,
        )
        m, k, n, sh, sw = f.dit_blocks
        self.dit_config = ttnn.MinimalMatmulConfig(
            M_block_size=m, K_block_size=k, N_block_size=n, subblock_h=sh, subblock_w=sw,
            compute_with_storage_grid_size=grid,
        )
        # Plain residual add through the addcmul: residual + (h @ W + b) * ones. The fused kernel
        # reads both addcmul inputs through one TensorAccessor type, so the ones vector must live
        # in the same buffer type as the residual (DRAM constants vs L1 activations under I).
        self.ones_hidden = self._to_dev(torch.ones(1, 1, 1, HIDDEN))
        self.ones_hidden_l1 = ttnn.to_memory_config(self.ones_hidden, ttnn.L1_MEMORY_CONFIG) if f.l1 else None
        # 0/1 permutation matmuls are exact at HiFi4 without fp32 accumulation (one nonzero term
        # per output; rf-detr verified bit-exact on this device).
        self.ck_perm = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )
        perms = build_merge_perms(self.grid_hw, self.seq_pad, self.merged_pad, MERGE)
        self.merge_perms = [self._to_dev(P.reshape(1, 1, self.merged_pad, self.seq_pad)) for P in perms]

    def _to_dev(self, t, layout=ttnn.TILE_LAYOUT, dtype=None):
        return ttnn.from_torch(
            t,
            dtype=dtype or self.dtype,
            layout=layout,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _load_block(self, sd, i):
        p = f"vision_model.encoder.blocks.{i}"
        # wqkv: fused [3456,1152] -> need q,k,v per-head padded then re-fused.
        wqkv = sd[f"{p}.wqkv.weight"].float()  # [3456,1152] (out, in)
        wqkv_b = sd[f"{p}.wqkv.bias"].float()  # [3456]
        wq, wk, wv = torch.chunk(wqkv, 3, dim=0)  # each [1152,1152] (out, in)
        bq, bk, bv = torch.chunk(wqkv_b, 3, dim=0)  # each [1152]
        # transpose to (in, out) for matmul, pad out per-head 72->96
        wq_t = _pad_per_head(wq.t().contiguous(), N_HEADS, HEAD_DIM, PAD_HEAD_DIM)  # [1152, 1536]
        wk_t = _pad_per_head(wk.t().contiguous(), N_HEADS, HEAD_DIM, PAD_HEAD_DIM)
        wv_t = _pad_per_head(wv.t().contiguous(), N_HEADS, HEAD_DIM, PAD_HEAD_DIM)
        wqkv_fused = torch.cat([wq_t, wk_t, wv_t], dim=-1)  # [1152, 3*1536]
        bq_p = _pad_per_head(bq, N_HEADS, HEAD_DIM, PAD_HEAD_DIM)
        bk_p = _pad_per_head(bk, N_HEADS, HEAD_DIM, PAD_HEAD_DIM)
        bv_p = _pad_per_head(bv, N_HEADS, HEAD_DIM, PAD_HEAD_DIM)
        wqkv_b_fused = torch.cat([bq_p, bk_p, bv_p], dim=-1)  # [3*1536]

        # wo: [1152,1152] (out,in). nlp_concat_heads emits padded-head layout, so pad wo INPUT
        # (which corresponds to per-head dims) with zeros in the padded lanes.
        wo = sd[f"{p}.wo.weight"].float()  # [1152,1152] (out, in=n_heads*head_dim)
        wo_in = wo.reshape(HIDDEN, N_HEADS, HEAD_DIM)
        wo_in = F.pad(wo_in, (0, PAD_HEAD_DIM - HEAD_DIM))  # pad input head_dim
        wo_t = wo_in.reshape(HIDDEN, N_HEADS * PAD_HEAD_DIM).t().contiguous()  # [1536, 1152] (in, out)
        wo_b = sd[f"{p}.wo.bias"].float()

        return {
            "norm0_w": self._to_dev(sd[f"{p}.norm0.weight"].reshape(1, -1)),
            "norm0_b": self._to_dev(sd[f"{p}.norm0.bias"].reshape(1, -1)),
            "norm1_w": self._to_dev(sd[f"{p}.norm1.weight"].reshape(1, -1)),
            "norm1_b": self._to_dev(sd[f"{p}.norm1.bias"].reshape(1, -1)),
            "wqkv": self._to_dev(wqkv_fused),  # [1152, 4608]
            "wqkv_b": self._to_dev(wqkv_b_fused.reshape(1, -1)),
            "wo": self._to_dev(wo_t),  # [1536, 1152]
            "wo_b": self._to_dev(wo_b.reshape(1, -1)),
            "fc0_w": self._to_dev(sd[f"{p}.mlp.fc0.weight"].t().contiguous()),  # [1152,4304]
            "fc0_b": self._to_dev(sd[f"{p}.mlp.fc0.bias"].reshape(1, -1)),
            "fc1_w": self._to_dev(sd[f"{p}.mlp.fc1.weight"].t().contiguous()),  # [4304,1152]
            "fc1_b": self._to_dev(sd[f"{p}.mlp.fc1.bias"].reshape(1, -1)),
        }

    # ------------------------------------------------------------------ op helpers
    def _layer_norm(self, x, w, b):
        return ttnn.layer_norm(x, epsilon=LN_EPS, weight=w, bias=b, compute_kernel_config=self.ck_hifi4, **self._mc)

    def _ones_for(self, residual):
        """Scale vector of the fused addcmul in the residual's buffer type (see _init_fused)."""
        if residual.memory_config().buffer_type == ttnn.BufferType.L1:
            return self.ones_hidden_l1
        return self.ones_hidden

    def _matmul(self, h, w, b):
        """h @ w + b. Fused path with LA_FUSED_MATMUL=minimal: the minimal_matmul kernel (same
        HiFi4 + fp32-acc compute config; no fused activation -- slower per the rf-detr precedent)."""
        if self.fused.enabled and self.fused.matmul == "minimal":
            return ttnn.experimental.minimal_matmul(
                h, w, bias_tensor=b, config=self.mm_config, memory_config=self.mem, dtype=self.dtype,
                compute_kernel_config=self.ck_hifi4,
            )
        return ttnn.linear(
            h,
            w,
            bias=b,
            compute_kernel_config=self.ck_hifi4,
            dtype=self.dtype,
            memory_config=self.mem,
        )

    def _matmul_residual(self, h, w, b, residual):
        """residual + (h @ w + b): one dit_minimal_matmul_addcmul_fused call (scale = ones) on the
        fused minimal path, else the legacy linear -> add pair (same operand order)."""
        if self.fused.enabled and self.fused.matmul == "minimal":
            return ttnn.experimental.dit_minimal_matmul_addcmul_fused(
                h, w, 1.0, residual, self._ones_for(residual),
                bias_tensor=b, config=self.dit_config, memory_config=self.mem, dtype=self.dtype,
                compute_kernel_config=self.ck_hifi4,
            )
        y = self._matmul(h, w, b)
        out = ttnn.add(residual, y, memory_config=self.mem)
        ttnn.deallocate(y)
        return out

    def _gelu(self, h):
        if self.fused.enabled and self.fused.gelu == "tanh":
            # The reference (PytorchGELUTanh) variant; ttnn.gelu's default is the erf/Accurate one.
            return ttnn.gelu(h, variant=ttnn.GeluVariant.Tanh, **self._mc)
        return ttnn.gelu(h, **self._mc)

    def _attention(self, x_norm, blk):
        """x_norm: [1,1,seq_pad,HIDDEN] -> concatenated heads [1,1,seq_pad,N_HEADS*PAD_HEAD_DIM]
        (the wo projection + residual is applied by the caller)."""
        xqkv = self._matmul(x_norm, blk["wqkv"], blk["wqkv_b"])  # [1,1,seq_pad, 3*N_HEADS*PAD_HEAD_DIM]

        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            xqkv,
            num_heads=N_HEADS,
            num_kv_heads=N_HEADS,
            transpose_k_heads=False,
            memory_config=self.mem,
        )  # each [1, N_HEADS, seq_pad, PAD_HEAD_DIM]
        ttnn.deallocate(xqkv)

        # rotary embeddings (interleaved convention). rotary_embedding_llama requires bf16
        # inputs; cos/sin are bf16. SDPA below runs in the model's activation dtype.
        if q.dtype != ttnn.bfloat16:
            q = ttnn.typecast(q, dtype=ttnn.bfloat16)
        if k.dtype != ttnn.bfloat16:
            k = ttnn.typecast(k, dtype=ttnn.bfloat16)
        q = ttnn.experimental.rotary_embedding_llama(
            q, self.rope_cos, self.rope_sin, self.rope_trans, is_decode_mode=False, **self._mc
        )
        k = ttnn.experimental.rotary_embedding_llama(
            k, self.rope_cos, self.rope_sin, self.rope_trans, is_decode_mode=False, **self._mc
        )

        sdpa_kwargs = dict(self._mc)
        if self.fused.enabled:
            sdpa_kwargs["program_config"] = self.sdpa_pc
        attn = ttnn.transformer.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=self.attn_mask,
            is_causal=False,
            scale=self.scale,
            compute_kernel_config=self.ck_sdpa,
            **sdpa_kwargs,
        )  # [1, N_HEADS, seq_pad, PAD_HEAD_DIM]
        ttnn.deallocate(q)
        ttnn.deallocate(k)
        ttnn.deallocate(v)

        ctx = ttnn.experimental.nlp_concat_heads(attn, memory_config=self.mem)
        ttnn.deallocate(attn)
        return ctx  # [1,1,seq_pad, N_HEADS*PAD_HEAD_DIM]

    def _block(self, x, blk):
        n0 = self._layer_norm(x, blk["norm0_w"], blk["norm0_b"])
        ctx = self._attention(n0, blk)
        ttnn.deallocate(n0)
        x = self._matmul_residual(ctx, blk["wo"], blk["wo_b"], x)
        ttnn.deallocate(ctx)

        n1 = self._layer_norm(x, blk["norm1_w"], blk["norm1_b"])
        h = self._matmul(n1, blk["fc0_w"], blk["fc0_b"])
        ttnn.deallocate(n1)
        h = self._gelu(h)  # GELU variant: legacy erf op; LA_FUSED_GELU=tanh for the reference's
        x = self._matmul_residual(h, blk["fc1_w"], blk["fc1_b"], x)
        ttnn.deallocate(h)
        return x

    # ------------------------------------------------------------------ input
    def host_input(self, pixel_values):
        """Per-image host work: pixel_values torch [L,3,14,14] -> host ttnn tensor to upload.

        Legacy: bf16 TILE [1,1,seq_pad,588] (host tilize). Fused J: bf16 ROW_MAJOR
        [1,1,seq_pad,608] (zero-padded columns; tilized in-graph). Both convert fp32 -> bf16 with
        ttnn.from_torch, so the pixel values are bit-identical between the two."""
        L = pixel_values.shape[0]
        assert L == self.L, f"pixel rows {L} != grid L {self.L}"
        pix_flat = pixel_values.float().reshape(L, -1)  # [L,588] C-order (c,kh,kw)
        if self.seq_pad > L:
            pix_flat = F.pad(pix_flat, (0, 0, 0, self.seq_pad - L))
        if self.fused.enabled and self.fused.rowmajor_input:
            pix_flat = F.pad(pix_flat, (0, PATCH_DIM_PAD - PATCH_DIM))
            return ttnn.from_torch(
                pix_flat.reshape(1, 1, self.seq_pad, PATCH_DIM_PAD), dtype=self.dtype, layout=ttnn.ROW_MAJOR_LAYOUT
            )
        return ttnn.from_torch(pix_flat.reshape(1, 1, self.seq_pad, -1), dtype=self.dtype, layout=ttnn.TILE_LAYOUT)

    def _ingest(self, x):
        """Device input tensor -> TILE [1,1,seq_pad,patch_dim] (in-graph tilize for the ROW_MAJOR upload)."""
        if x.layout == ttnn.ROW_MAJOR_LAYOUT:
            return ttnn.tilize_with_zero_padding(x, memory_config=self.mem, use_multicore=True)
        return x

    def _patch_embed_device(self, x):
        """Fused path: TILE pixels [1,1,seq_pad,patch_dim] -> [1,1,seq_pad,HIDDEN] = x @ W + b + pos_emb
        (pos_emb is host-padded to seq_pad, so no in-graph pad; one fused op on the minimal path)."""
        if self.fused.matmul == "minimal":
            return ttnn.experimental.dit_minimal_matmul_addcmul_fused(
                x, self.proj_w, 1.0, self.pos_emb, self._ones_for(self.pos_emb),
                bias_tensor=self.proj_b, config=self.dit_config, memory_config=self.mem, dtype=self.dtype,
                compute_kernel_config=self.ck_hifi4,
            )
        y = self._matmul(x, self.proj_w, self.proj_b)
        out = ttnn.add(y, self.pos_emb, memory_config=self.mem)
        ttnn.deallocate(y)
        return out

    def patch_embed(self, pixel_values):
        """pixel_values torch [L,3,14,14] -> ttnn [1,1,seq_pad,HIDDEN] (real rows then padding)."""
        if self.fused.enabled:
            x = ttnn.to_device(self.host_input(pixel_values), self.device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            return self._patch_embed_device(self._ingest(x))
        # legacy (shipped) path, unchanged: host tilize + upload, linear, in-graph pad of pos_emb, add
        L = pixel_values.shape[0]
        assert L == self.L, f"pixel rows {L} != grid L {self.L}"
        pix_flat = pixel_values.float().reshape(L, -1)  # [L,588] C-order (c,kh,kw)
        seq_pad = self.seq_pad
        if seq_pad > L:
            pix_flat = F.pad(pix_flat, (0, 0, 0, seq_pad - L))
        x = self._to_dev(pix_flat.reshape(1, 1, seq_pad, -1))  # [1,1,seq_pad,588]
        x = ttnn.linear(
            x,
            self.proj_w,
            bias=self.proj_b,
            compute_kernel_config=self.ck_hifi4,
            dtype=self.dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )  # [1,1,seq_pad,HIDDEN]
        # add pos_emb (only over real L rows)
        if seq_pad > L:
            pe = ttnn.pad(self.pos_emb, [(0, 0), (0, 0), (0, seq_pad - L), (0, 0)], value=0.0)
        else:
            pe = self.pos_emb
        x = ttnn.add(x, pe, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return x

    @staticmethod
    def _seq_pad(L):
        """Legacy padding rule (kept for callers); the instance rule is ``self.seq_pad``."""
        return int(math.ceil(L / 128) * 128)

    def encoder(self, x):
        for blk in self.blocks:
            x = self._block(x, blk)
        x = self._layer_norm(x, self.final_ln_w, self.final_ln_b)
        return x

    def patch_merger(self, x_torch):
        """Host-side 2x2 spatial merge (matches modeling_vit.patch_merger), returns [L/4, 4608].

        The legacy path runs it on host between the encoder readback and the mlp1 upload (a pure
        layout reshuffle); the fused path reproduces it on device with ``_merge_device``.
        """
        return patch_merge_host(x_torch, self.grid_hw, MERGE)  # [L/4, 4608]

    def _merge_device(self, x):
        """Encoder output [1,1,seq_pad,HIDDEN] -> merged [1,1,merged_pad,4608] on device: four
        0/1 permutation matmuls (exact row gathers, rows >= L/4 exactly zero) + one concat."""
        parts = [
            ttnn.matmul(P, x, compute_kernel_config=self.ck_perm, memory_config=self.mem) for P in self.merge_perms
        ]
        merged = ttnn.concat(parts, dim=-1, memory_config=self.mem)
        for p in parts:
            ttnn.deallocate(p)
        return merged

    def mlp1(self, x):
        """x ttnn [1,1,Nmerged,4608] -> [1,1,Nmerged,2048]."""
        x = ttnn.layer_norm(
            x, epsilon=LN_EPS, weight=self.mlp1_ln_w, bias=self.mlp1_ln_b, compute_kernel_config=self.ck_hifi4,
            **self._mc,
        )
        x = self._matmul(x, self.mlp1_w1, self.mlp1_b1)
        x = self._gelu(x)
        x = self._matmul(x, self.mlp1_w2, self.mlp1_b2)
        return x

    # ------------------------------------------------------------------ fused device graph + trace
    def _device_graph(self, x_dev):
        """Whole fused graph, device in / device out: [1,1,seq_pad,patch_dim] (TILE or ROW_MAJOR)
        -> vit_proj [1,1,merged_pad,PROJ_OUT]. No host round trip, so it is trace-capturable."""
        x = self._patch_embed_device(self._ingest(x_dev))
        x = self.encoder(x)
        merged = self._merge_device(x)
        ttnn.deallocate(x)
        proj = self.mlp1(merged)
        ttnn.deallocate(merged)
        if self.fused.l1:
            # The LLM boundary concatenates this with a DRAM embedding gather: keep the handoff in DRAM.
            proj_dram = ttnn.to_memory_config(proj, ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(proj)
            proj = proj_dram
        return proj

    @property
    def trace_captured(self):
        return self._trace_id is not None

    @property
    def trace_output(self):
        """Persistent device output of the captured vision trace (None before capture)."""
        return self._trace_out

    def capture_trace(self, pixel_values):
        """Fused path: allocate the persistent input, run the graph once eagerly (program cache),
        then record it as ONE metal trace (rf-detr ``_capture_trace``). Call this from the
        warm-up AFTER every other long-lived device buffer exists (the LLM's decode-trace inputs
        included): buffers allocated after a capture can land in that trace's scratch.

        The converse hazard is NOT avoided, only ordered around: ``_persistent_in`` and
        ``_trace_out`` are themselves allocated after the library's decode(+sampling) trace was
        captured in warm-up run 1, so they may sit in the decode trace's scratch and be clobbered
        by every decode replay. This is benign only because each request runs
        copy -> vision trace -> prefill trace -> eager tail -> decode, i.e. both buffers are fully
        rewritten before anything reads them again. Do not read ``trace_output`` (or replay the
        prefill trace) after a decode step without re-running the vision trace first."""
        if not (self.fused.enabled and self.fused.vision_trace):
            raise RuntimeError("capture_trace needs TT_FUSED=1 with LA_FUSED_VISION_TRACE=1")
        if self._trace_id is not None:
            return
        host = self.host_input(pixel_values)
        self._persistent_in = ttnn.to_device(host, self.device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        out = self._device_graph(self._persistent_in)  # compile pass
        ttnn.synchronize_device(self.device)
        ttnn.deallocate(out)
        trace_id = ttnn.begin_trace_capture(self.device, cq_id=0)
        try:
            out = self._device_graph(self._persistent_in)
        except BaseException:
            # Never leave a capture open: an open capture hangs the device close (p150 lesson).
            try:
                ttnn.end_trace_capture(self.device, trace_id, cq_id=0)
                ttnn.release_trace(self.device, trace_id)
            except Exception as e:  # noqa: BLE001
                print(f"[vision] closing the failed trace capture raised: {e!r}", flush=True)
            raise
        ttnn.end_trace_capture(self.device, trace_id, cq_id=0)
        ttnn.synchronize_device(self.device)
        self._trace_id = trace_id
        self._trace_out = out

    def release_trace(self):
        """Release the captured vision trace and its persistent buffers (trace-region hygiene for
        sessions that build several MoonViT instances on one device, e.g. DEVICE_VALIDATION 3.3).
        The next ``forward_device`` runs eagerly until ``capture_trace`` is called again."""
        if self._trace_id is not None:
            ttnn.release_trace(self.device, self._trace_id)
            self._trace_id = None
        for attr in ("_trace_out", "_persistent_in"):
            t = getattr(self, attr)
            if t is not None:
                ttnn.deallocate(t)
                setattr(self, attr, None)

    def forward_device(self, pixel_values):
        """pixel_values torch [L,3,14,14] -> vit_proj ttnn DEVICE tensor [1,1,merged_pad,PROJ_OUT]
        (rows >= nmerged are padding). Trace replay when captured (the returned tensor is the
        persistent trace output: do NOT deallocate it), else the eager fused graph (caller owns
        the result). Fused path only; legacy callers use ``forward``."""
        assert self.fused.enabled, "forward_device is the fused path; use forward() on the legacy path"
        host = self.host_input(pixel_values)
        if self._trace_id is not None:
            ttnn.copy_host_to_device_tensor(host, self._persistent_in, cq_id=0)
            ttnn.execute_trace(self.device, self._trace_id, cq_id=0, blocking=False)
            return self._trace_out
        x = ttnn.to_device(host, self.device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return self._device_graph(x)

    def read_projection(self, proj_dev):
        """Device vit_proj [1,1,merged_pad,PROJ_OUT] -> host fp32 [nmerged, PROJ_OUT] (one readback)."""
        return ttnn.to_torch(proj_dev)[0, 0, : self.nmerged].float()

    # ------------------------------------------------------------------ #
    def forward(self, pixel_values, return_intermediates=False):
        """pixel_values torch [L,3,14,14] -> vit_proj torch fp32 [Nmerged, 2048] on host.

        If return_intermediates, also return dict of host tensors for incremental PCC
        (always eager, so test_vision.py can read the encoder output back).
        """
        if self.fused.enabled:
            return self._forward_fused(pixel_values, return_intermediates)
        return self._forward_legacy(pixel_values, return_intermediates)

    def _forward_fused(self, pixel_values, return_intermediates):
        if not return_intermediates:
            proj = self.forward_device(pixel_values)
            proj_torch = self.read_projection(proj)
            if self._trace_id is None:
                ttnn.deallocate(proj)
            return proj_torch
        L = self.L
        x = ttnn.to_device(self.host_input(pixel_values), self.device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        x = self._patch_embed_device(self._ingest(x))
        inter = {"patch_embed": ttnn.to_torch(x)[0, 0, :L].float()}
        x = self.encoder(x)
        inter["encoder_out"] = ttnn.to_torch(x)[0, 0, :L].float()
        merged = self._merge_device(x)
        ttnn.deallocate(x)
        inter["merged"] = ttnn.to_torch(merged)[0, 0, : self.nmerged].float()
        proj = self.mlp1(merged)
        ttnn.deallocate(merged)
        proj_torch = self.read_projection(proj)
        ttnn.deallocate(proj)
        return proj_torch, inter

    def _forward_legacy(self, pixel_values, return_intermediates):
        """The shipped path, op for op: eager graph, host patch_merger between two readbacks."""
        L = self.L
        seq_pad = self.seq_pad

        x = self.patch_embed(pixel_values)  # [1,1,seq_pad,HIDDEN]
        inter = {}
        if return_intermediates:
            inter["patch_embed"] = ttnn.to_torch(x)[0, 0, :L].float()

        x = self.encoder(x)  # [1,1,seq_pad,HIDDEN]
        enc_torch = ttnn.to_torch(x)[0, 0].float()  # [seq_pad,HIDDEN]
        ttnn.deallocate(x)
        if return_intermediates:
            inter["encoder_out"] = enc_torch[:L]

        merged = self.patch_merger(enc_torch)  # [L/4, 4608]
        nmerged = merged.shape[0]
        merged_pad = self.merged_pad
        if merged_pad > nmerged:
            merged = F.pad(merged, (0, 0, 0, merged_pad - nmerged))
        xm = self._to_dev(merged.reshape(1, 1, merged_pad, MLP1_IN))

        proj = self.mlp1(xm)  # [1,1,merged_pad,2048]
        ttnn.deallocate(xm)
        proj_torch = ttnn.to_torch(proj)[0, 0, :nmerged].float()  # [Nmerged,2048]
        ttnn.deallocate(proj)

        if return_intermediates:
            return proj_torch, inter
        return proj_torch
