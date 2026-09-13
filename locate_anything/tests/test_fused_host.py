# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Host-only (torch, NO device) tests for the ``TT_FUSED=1`` paths in ``locate_anything/tt/fused.py``.

Every exact reformulation the fused device graph relies on is checked here against the reference
math it replaces, on the same bf16 values the device sees:

* B. device patch merger = four 0/1 permutation matmuls + concat  ==  ``patch_merge_host`` (torch.equal)
* K. device vision->LLM merge = one 0/1 matmul over concat([E_ids, vit_proj])  ==
     ``merge_vision_tokens`` + ``preprocess_inputs_prefill`` (torch.equal; library helpers when
     importable, inline replica otherwise) and the image-token rows are the same for every query
     (tokenizer from the HF snapshot when present)
* C. padding rule: 1056 -> 1056 (exact, no mask) with the knob, 1152 legacy; 1092 -> 1152 in both
* M. stale-token guard against a stub of ``Generator.decode_forward``'s reset logic, and the greedy
     SamplingParams really format to the force-argmax triple (k=1, p=0, temp=1)
* knob plumbing: ``TT_FUSED`` unset -> disabled config (legacy selection), sub-knob parsing,
  program-config validity (chunk % 32, block / subblock divisibility) as the op validators demand

Run (tree python has pytest 9; ``--noconftest`` keeps tt-metal's device fixtures out of a host run):

  cd models/locate-anything-3b-p150/code
  TREE=/home/deepgadget/experiments/gbp-tt/tt-metal
  TT_METAL_HOME=$TREE PYTHONPATH=$PWD:$TREE:$TREE/ttnn $TREE/python_env/bin/python -m pytest \\
      --noconftest -q locate_anything/tests/test_fused_host.py

or as a plain script (no pytest needed): ``python locate_anything/tests/test_fused_host.py``.
"""
from __future__ import annotations

import inspect
import os
import sys

import torch
import torch.nn.functional as F

try:
    import pytest
except ImportError:  # plain-script mode
    pytest = None

from locate_anything.tt import fused
from locate_anything.tt.fused import (
    FusedConfig,
    arm_stale_token_guard,
    build_embed_merge_table,
    build_merge_perms,
    image_token_rows,
    merge_source_rows,
    pad_prompt_ids,
    patch_merge_host,
    prefill_bucket_len,
    seq_pad_len,
    validate_mm_blocks,
    validate_sdpa_chunks,
    vision_seq_pad,
)

SERVED_GRID = (24, 44)  # LA_IN_TOKEN_LIMIT=1024 on a 1920x1080 image -> L = 1056 = 33 tiles
GOLDEN_GRID = (26, 42)  # test_vision.py golden -> L = 1092, not tile aligned
HIDDEN = 1152
LLM_DIM = 2048
IMAGE_TOKEN_INDEX = 151665
PAD_TOKEN_ID = 151643
PREFILL_LEN = 512


class _Skip(Exception):
    pass


def _skip(msg):
    if pytest is not None:
        pytest.skip(msg)
    raise _Skip(msg)


# --------------------------------------------------------------------------- knob plumbing
def test_knob_zero_selects_legacy():
    # Default ON since the 2026-09-13 device validation: TT_FUSED=0 (or anything but "1") is the legacy path.
    for env in ({"TT_FUSED": "0"}, {"TT_FUSED": ""}, {"TT_FUSED": "yes"}):
        cfg = FusedConfig.from_env(env)
        assert cfg == FusedConfig.disabled()
        assert not cfg.enabled
        # every lever is off, so MoonViT / pipeline / model_la take their legacy branches
        assert not any(
            getattr(cfg, k)
            for k in ("vision_trace", "l1", "rowmajor_input", "device_merge", "prefill_trace", "small_readback", "device_sampling")
        )
        assert cfg.matmul == "linear" and cfg.gelu == "erf"
        assert cfg.trace_region_size == fused.LEGACY_TRACE_REGION_SIZE == 50_000_000
    # the legacy padding rule is what the disabled config produces
    assert vision_seq_pad(1056, False) == 1152 and vision_seq_pad(264, False) == 384
    # unset == "1" (the default fused configuration)
    assert FusedConfig.from_env({}) == FusedConfig.from_env({"TT_FUSED": "1"})
    assert FusedConfig.from_env({}).enabled


def test_knob_on_defaults_and_subknobs():
    cfg = FusedConfig.from_env({"TT_FUSED": "1"})
    assert cfg.enabled and cfg.vision_trace and cfg.device_merge and cfg.prefill_trace
    assert cfg.small_readback and cfg.device_sampling and cfg.rowmajor_input
    # the default fused configuration keeps the legacy vision numerics (device-verified bit-identical):
    # linear matmuls, legacy SDPA kernel config, erf GELU, DRAM, 128-padding + mask
    assert cfg.matmul == "linear" and cfg.gelu == "erf" and not cfg.l1 and not cfg.exact_seq
    assert cfg.sdpa_chunks == fused.LEGACY_SDPA_CHUNKS == (32, 32) and cfg.sdpa_exp_approx is fused.LEGACY_SDPA_EXP_APPROX is True
    assert cfg.mm_blocks == (8, 4, 4, 2, 2) and cfg.dit_blocks == (4, 4, 4, 2, 2)
    assert cfg.trace_region_size == fused.FUSED_TRACE_REGION_SIZE > fused.LEGACY_TRACE_REGION_SIZE
    # the measured "fast vision" opt-in (levers C + F) and the minimal-matmul opt-in (D/E)
    fast = FusedConfig.from_env({"TT_FUSED": "1", "LA_FUSED_EXACT_SEQ": "1", "LA_FUSED_SDPA_CHUNKS": "96,352", "LA_FUSED_SDPA_EXP_APPROX": "0"})
    assert fast.exact_seq and fast.sdpa_chunks == (96, 352) and fast.sdpa_exp_approx is False and fast.matmul == "linear"
    assert FusedConfig.from_env({"TT_FUSED": "1", "LA_FUSED_MATMUL": "minimal"}).matmul == "minimal"
    cfg2 = FusedConfig.from_env(
        {
            "TT_FUSED": "1",
            "LA_FUSED_MATMUL": "linear",
            "LA_FUSED_GELU": "tanh",
            "LA_FUSED_L1": "1",
            "LA_FUSED_SDPA_CHUNKS": "128, 1056",
            "LA_FUSED_MM_BLOCKS": "4,4,4,2,2",
            "LA_FUSED_DEVICE_SAMPLING": "0",
            "LA_FUSED_PREFILL_TRACE": "0",
        }
    )
    assert cfg2.matmul == "linear" and cfg2.gelu == "tanh" and cfg2.l1
    assert cfg2.sdpa_chunks == (128, 1056) and cfg2.mm_blocks == (4, 4, 4, 2, 2)
    assert not cfg2.device_sampling and not cfg2.prefill_trace and cfg2.vision_trace
    assert cfg2.as_dict()["sdpa_chunks"] == (128, 1056)
    # the legacy SDPA kernel configuration is reachable for a chunking-only / exp-only A/B
    cfg3 = FusedConfig.from_env({"TT_FUSED": "1", "LA_FUSED_SDPA_CHUNKS": "32,32", "LA_FUSED_SDPA_EXP_APPROX": "1"})
    assert cfg3.sdpa_chunks == fused.LEGACY_SDPA_CHUNKS == (32, 32)
    assert cfg3.sdpa_exp_approx is fused.LEGACY_SDPA_EXP_APPROX is True
    assert FusedConfig.disabled().sdpa_chunks == (32, 32) and FusedConfig.disabled().sdpa_exp_approx is True
    # LA_FUSED_EXACT_SEQ: default 0 keeps the legacy 128-padding + mask inside the fused graph (the
    # bit-identical default, device-verified in DEVICE_VALIDATION.md); 1 = exact 1056 rows, no mask (lever C)
    assert not cfg.exact_seq and not cfg3.exact_seq
    cfg4 = FusedConfig.from_env({"TT_FUSED": "1", "LA_FUSED_EXACT_SEQ": "1"})
    assert cfg4.exact_seq and cfg4.enabled
    assert vision_seq_pad(1056, cfg4.enabled and cfg4.exact_seq) == 1056
    assert vision_seq_pad(1056, cfg.enabled and cfg.exact_seq) == 1152
    assert vision_seq_pad(264, cfg.enabled and cfg.exact_seq) == 384
    assert cfg.as_dict()["exact_seq"] is False and cfg4.as_dict()["exact_seq"] is True


def test_program_config_validity():
    # sdpa_device_operation.cpp: chunk sizes must be divisible by TILE_SIZE
    validate_sdpa_chunks(96, 352)
    validate_sdpa_chunks(32, 32)
    for bad in ((100, 352), (96, 350), (0, 32), (96, -32)):
        try:
            validate_sdpa_chunks(*bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"chunks {bad} accepted")
    # minimal_matmul_device_operation.cpp: blocks > 0, M % sub_h == 0, N % sub_w == 0,
    # sub_h * sub_w <= max_dest_volume (get_dest_reg_count: 16 / 2 (dst_full_sync_en=False) / 2
    # (fp32_dest_acc_en=True) = 4 tiles under the port's ck_hifi4; 8 without fp32 accumulation)
    assert fused.MM_MAX_DEST_TILES_FP32 == 4 and fused.MM_MAX_DEST_TILES_BF16 == 8
    validate_mm_blocks((8, 4, 4, 2, 2))
    validate_mm_blocks((4, 4, 4, 2, 2))
    validate_mm_blocks((8, 4, 4, 1, 4))  # 4 tiles: at the cap
    validate_mm_blocks((8, 4, 4, 4, 1))
    validate_mm_blocks((8, 4, 8, 2, 4), fp32_dest_acc=False)  # rf-detr-style config fits the 8-tile cap
    for bad in ((3, 4, 4, 2, 2), (8, 4, 3, 2, 2), (0, 4, 4, 2, 2), (8, 4, 4, 0, 2), (8, 4, 4, 2)):
        try:
            validate_mm_blocks(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"blocks {bad} accepted")
    for bad in ((8, 4, 8, 2, 4), (8, 4, 8, 4, 2), (8, 8, 8, 2, 8)):  # the op docstring's "typical" 2x4 / 4x2
        try:
            validate_mm_blocks(bad)  # fp32_dest_acc=True -> TT_FATAL on device, must fail here
        except ValueError:
            pass
        else:
            raise AssertionError(f"blocks {bad} accepted with the fp32-acc cap")
    try:
        validate_mm_blocks((8, 8, 8, 4, 4), fp32_dest_acc=False)  # 16 tiles: over even the bf16 cap
    except ValueError:
        pass
    else:
        raise AssertionError("4x4 subblocks accepted")
    for env in (
        {"TT_FUSED": "1", "LA_FUSED_SDPA_CHUNKS": "96,100"},
        {"TT_FUSED": "1", "LA_FUSED_DIT_BLOCKS": "4,4,3,2,2"},
        {"TT_FUSED": "1", "LA_FUSED_MATMUL": "fast"},
        {"TT_FUSED": "1", "LA_FUSED_GELU": "exact"},
        {"TT_FUSED": "1", "LA_FUSED_MM_BLOCKS": "8,4,4"},
        {"TT_FUSED": "1", "LA_FUSED_MM_BLOCKS": "8,4,8,2,4"},  # 8-tile subblock under fp32 acc
        {"TT_FUSED": "1", "LA_FUSED_DIT_BLOCKS": "4,4,8,4,2"},
    ):
        try:
            FusedConfig.from_env(env)
        except ValueError:
            pass
        else:
            raise AssertionError(f"{env} accepted")
    # Served-grid SDPA arithmetic (sdpa_program_factory.cpp): padded_S = ceil(S / chunk) * chunk;
    # the generated padding mask is used only when padded_Sk != Sk or padded_Sq != Sq.
    L = SERVED_GRID[0] * SERVED_GRID[1]
    q, k = FusedConfig.from_env({"TT_FUSED": "1", "LA_FUSED_SDPA_CHUNKS": "96,352"}).sdpa_chunks  # the measured fast setting
    assert L % q == 0 and L % k == 0, "the fast-vision chunks must not pad the served sequence"
    assert L // q == 11 and L // k == 3
    q, k = FusedConfig.from_env({"TT_FUSED": "1"}).sdpa_chunks  # default = the legacy kernel configuration
    assert (q, k) == (32, 32) and L % q == 0


# --------------------------------------------------------------------------- C. padding rule
def test_seq_pad_rule():
    assert seq_pad_len(1056, exact_tiles=False) == 1152 and seq_pad_len(1056, exact_tiles=True) == 1056
    assert seq_pad_len(1092, exact_tiles=False) == 1152 and seq_pad_len(1092, exact_tiles=True) == 1120
    # the vision rule: exact only for tile-aligned lengths under the knob, legacy otherwise
    assert vision_seq_pad(1056, True) == 1056  # served grid: no padding -> no attention mask
    assert vision_seq_pad(1056, False) == 1152  # legacy: 96 padded rows + mask
    assert vision_seq_pad(1092, True) == 1152 and vision_seq_pad(1092, False) == 1152  # golden grid: legacy both
    assert vision_seq_pad(264, True) == 384 and vision_seq_pad(264, False) == 384  # merged rows, both modes
    assert vision_seq_pad(1024, True) == 1024 and vision_seq_pad(1024, False) == 1024
    assert vision_seq_pad(1, True) == 128


def test_prefill_bucket_len():
    assert prefill_bucket_len(303, 4096) == 512  # "car" at the served grid
    assert prefill_bucket_len(328, 4096) == 512
    assert prefill_bucket_len(512, 4096) == 512
    assert prefill_bucket_len(513, 4096) == 1024
    assert prefill_bucket_len(100, 4096) == 128
    assert prefill_bucket_len(3000, 2048) == 2048


# --------------------------------------------------------------------------- B. patch merger
def _check_merge_perms(grid_hw, fused_enabled):
    L = grid_hw[0] * grid_hw[1]
    nmerged = L // 4
    seq_pad = vision_seq_pad(L, fused_enabled)
    merged_pad = vision_seq_pad(nmerged, fused_enabled)
    perms = build_merge_perms(grid_hw, seq_pad, merged_pad)
    assert len(perms) == 4 and all(P.shape == (merged_pad, seq_pad) for P in perms)
    rows = merge_source_rows(grid_hw)
    for j, P in enumerate(perms):
        assert set(P.unique().tolist()) <= {0.0, 1.0}
        assert torch.equal(P[:nmerged].sum(dim=1), torch.ones(nmerged))  # one source per real row
        assert torch.equal(P[nmerged:], torch.zeros(merged_pad - nmerged, seq_pad))  # exact zero pad rows
        assert torch.equal(P[:, L:].sum(), torch.tensor(0.0))  # padding columns never selected
        assert torch.equal(P[:nmerged].argmax(dim=1), rows[:, j])
    # every encoder row is used exactly once across the four blocks
    used = torch.cat([P.sum(dim=0) for P in perms]).reshape(4, seq_pad).sum(dim=0)
    assert torch.equal(used[:L], torch.ones(L)) and torch.equal(used[L:], torch.zeros(seq_pad - L))

    torch.manual_seed(0)
    X = torch.randn(seq_pad, HIDDEN).to(torch.bfloat16)  # what the encoder hands over (bf16 rows)
    ref = patch_merge_host(X.float(), grid_hw)  # legacy host merge [nmerged, 4608]
    assert ref.shape == (nmerged, 4 * HIDDEN)
    got = torch.cat([P @ X.float() for P in perms], dim=-1)  # exact in fp32 (0/1 x fp32)
    assert torch.equal(got[:nmerged], ref)
    assert torch.equal(got[nmerged:], torch.zeros(merged_pad - nmerged, 4 * HIDDEN))
    # and on the bf16 values themselves: a 0/1 matmul with one nonzero term rounds to the input bit pattern
    got_bf16 = torch.cat([(P.to(torch.bfloat16) @ X) for P in perms], dim=-1)
    assert torch.equal(got_bf16[:nmerged], ref.to(torch.bfloat16))
    assert torch.equal(got_bf16[:nmerged].view(torch.int16), ref.to(torch.bfloat16).view(torch.int16))
    # the legacy padded layout is the same math (padding rows of X are ignored by both)
    ref_pad = F.pad(patch_merge_host(X.float(), grid_hw), (0, 0, 0, merged_pad - nmerged))
    assert torch.equal(got, ref_pad)


def test_merge_perms_served_grid_exact():
    _check_merge_perms(SERVED_GRID, fused_enabled=True)  # 1056 -> 1056, 264 -> 384


def test_merge_perms_golden_grid_padded():
    _check_merge_perms(GOLDEN_GRID, fused_enabled=True)  # 1092 -> 1152 (legacy pad), 273 -> 384


def test_merge_perms_reject_too_small():
    try:
        build_merge_perms(SERVED_GRID, 1024, 384)
    except ValueError:
        return
    raise AssertionError("seq_pad < L accepted")


# --------------------------------------------------------------------------- K. vision->LLM merge
def _library_merge():
    """(merge_vision_tokens, preprocess_inputs_prefill) from tt-metal's qwen25_vl demo, or None."""
    try:
        from models.demos.qwen25_vl.tt.common import merge_vision_tokens, preprocess_inputs_prefill
    except Exception:  # noqa: BLE001 - tt-metal tree not on PYTHONPATH (plain host)
        return None
    return merge_vision_tokens, preprocess_inputs_prefill


def _replica_merge(input_ids, text_embeds, image_embeds, pad_embedding, prefill_len):
    """Inline copy of the two qwen25_vl helpers (masked_scatter into the image rows; pad rows
    filled with pad_embedding up to the power-of-two bucket)."""
    mask = (input_ids == IMAGE_TOKEN_INDEX).unsqueeze(-1).expand_as(text_embeds)
    merged = text_embeds.masked_scatter(mask, image_embeds)[0]
    out = torch.empty(prefill_len, merged.shape[-1], dtype=merged.dtype)
    out[:] = pad_embedding
    out[: merged.shape[0]] = merged
    return out


def _synthetic_prompt(n_img=264, first_img_row=20, n_text_after=19):
    """ids [1, S] shaped like the served prompt: template prefix, the image block, the query."""
    g = torch.Generator().manual_seed(1)
    prefix = torch.randint(0, 150000, (first_img_row,), generator=g)
    suffix = torch.randint(0, 150000, (n_text_after,), generator=g)
    ids = torch.cat([prefix, torch.full((n_img,), IMAGE_TOKEN_INDEX), suffix]).reshape(1, -1)
    return ids


def _check_embed_merge(input_ids, prefill_len, n_img_pad):
    torch.manual_seed(0)
    S = input_ids.shape[1]
    vocab = 152704
    table = torch.randn(vocab, LLM_DIM).to(torch.bfloat16)  # the bf16 embedding table (host == device)
    rows = image_token_rows(input_ids, IMAGE_TOKEN_INDEX)
    n_img = int(rows.numel())
    vit_proj = torch.randn(n_img, LLM_DIM).to(torch.bfloat16)  # mlp1 output rows (bf16 on device)
    garbage = torch.randn(n_img_pad - n_img, LLM_DIM).to(torch.bfloat16) * 1e3  # padded vision rows
    vit_pad = torch.cat([vit_proj, garbage], dim=0)

    # --- reference: the legacy host path (fp32 embeds, merge, pad, then bf16 upload) ---
    text_embeds = F.embedding(input_ids, table).to(torch.float32)
    pad_embedding = table[PAD_TOKEN_ID].to(torch.float32)
    lib = _library_merge()
    if lib is not None:
        merge_vision_tokens, preprocess_inputs_prefill = lib

        class _Cfg:
            image_token_id = IMAGE_TOKEN_INDEX

        class _Args:
            max_seq_len = 4096

        merged = merge_vision_tokens(input_ids, text_embeds, vit_proj.to(text_embeds.dtype), _Cfg())
        embeds, decoding_pos, prefill_lens = preprocess_inputs_prefill(
            [merged[0]], _Args(), torch.ones_like(input_ids), pad_embedding=pad_embedding
        )
        ref = embeds[0]
        assert int(prefill_lens[0]) == prefill_len and int(decoding_pos[0]) == S
    else:
        ref = _replica_merge(input_ids, text_embeds, vit_proj.to(text_embeds.dtype), pad_embedding, prefill_len)
    ref_bf16 = ref.to(torch.bfloat16)  # what ttnn.from_torch(dtype=bfloat16) uploads
    assert ref_bf16.shape == (prefill_len, LLM_DIM)

    # --- fused: ids padded with pad_token_id -> table gather -> concat with the vision rows -> 0/1 matmul ---
    ids_pad = pad_prompt_ids(input_ids, prefill_len, PAD_TOKEN_ID)
    assert ids_pad.shape == (prefill_len,) and torch.equal(ids_pad[:S], input_ids[0]) and (ids_pad[S:] == PAD_TOKEN_ID).all()
    e_ids = table[ids_pad]  # ttnn.embedding on the bf16 table
    P = build_embed_merge_table(rows, prefill_len, n_img_pad)
    assert P.shape == (prefill_len, prefill_len + n_img_pad)
    assert torch.equal(P.sum(dim=1), torch.ones(prefill_len))  # one source row per output row
    assert set(P.unique().tolist()) == {0.0, 1.0}
    assert torch.equal(P[:, prefill_len + n_img :].sum(), torch.tensor(0.0))  # garbage vision rows never selected
    stacked = torch.cat([e_ids, vit_pad], dim=0)
    got = P @ stacked.float()
    assert torch.equal(got.to(torch.bfloat16), ref_bf16)
    assert torch.equal(got, ref)  # fp32 too: the merge is a pure row selection
    got_bf16 = P.to(torch.bfloat16) @ stacked  # on bf16 operands: still the exact rows
    assert torch.equal(got_bf16.view(torch.int16), ref_bf16.view(torch.int16))
    # text rows are embedding rows, image rows are vision rows, pad rows are the pad embedding
    assert torch.equal(got[rows], vit_proj.float())
    assert torch.equal(got[S:], pad_embedding.expand(prefill_len - S, -1))
    return rows


def test_embed_merge_table_matches_reference_synthetic():
    ids = _synthetic_prompt()
    rows = _check_embed_merge(ids, PREFILL_LEN, n_img_pad=384)
    assert torch.equal(rows, torch.arange(20, 284))
    # a longer query in the same bucket and an image block at another offset
    _check_embed_merge(_synthetic_prompt(first_img_row=25, n_text_after=200), PREFILL_LEN, n_img_pad=384)
    # prompt too long for the bucket -> explicit error (the pipeline falls back to the legacy path first)
    try:
        pad_prompt_ids(_synthetic_prompt(n_text_after=300), PREFILL_LEN, PAD_TOKEN_ID)
    except ValueError:
        pass
    else:
        raise AssertionError("over-long prompt accepted")
    try:
        build_embed_merge_table(torch.arange(20, 284), PREFILL_LEN, n_img_pad=200)
    except ValueError:
        pass
    else:
        raise AssertionError("vision buffer smaller than the image block accepted")


def _tokenizer():
    try:
        from transformers import AutoTokenizer
        from locate_anything.reference import la_inputs

        path = la_inputs.find_model_path()
        return AutoTokenizer.from_pretrained(path), la_inputs
    except Exception as e:  # noqa: BLE001
        _skip(f"LocateAnything tokenizer not available on this host: {type(e).__name__}: {e}")


def test_image_rows_constant_across_queries_real_tokenizer():
    tok, la_inputs = _tokenizer()
    n_tok = la_inputs.num_image_tokens(SERVED_GRID)
    assert n_tok == 264
    rows_ref = None
    for query in ("car", "object", "person</c>car", "a red car parked on the left side of the street next to a tree " * 2, "x"):
        ids = tok([la_inputs.build_chat_text(query, n_tok)], return_tensors="pt")["input_ids"]
        rows = image_token_rows(ids, IMAGE_TOKEN_INDEX)
        assert rows.numel() == n_tok
        if rows_ref is None:
            rows_ref = rows
        assert torch.equal(rows, rows_ref), f"image rows moved for query {query!r}"
        assert prefill_bucket_len(int(ids.shape[1]), 4096) == PREFILL_LEN
        # the fused table for this real prompt reproduces the reference merge exactly
        _check_embed_merge(ids, PREFILL_LEN, n_img_pad=384)
    assert torch.equal(rows_ref, torch.arange(20, 20 + n_tok)), rows_ref[:3]
    # the 1000-char query cap can leave the 512 bucket: that is the legacy fallback, not an error
    long_ids = tok([la_inputs.build_chat_text("word " * 200, n_tok)], return_tensors="pt")["input_ids"]
    assert prefill_bucket_len(int(long_ids.shape[1]), 4096) in (512, 1024)


# --------------------------------------------------------------------------- M. on-device sampling
class _StubGenerator:
    pass


def _library_reset_merge(dev_toks, dev_pos, host_toks, host_pos, prefilled, bs=1, i=0):
    """Copy of the slot-merge in Generator.decode_forward (reset_batch branch, generator.py):
    the device token wins when dev_pos in {host_pos, host_pos + 1} unless the slot is in
    ``_slots_prefilled_since_decode``."""
    use_dev = (dev_pos == host_pos) | (dev_pos == host_pos + 1)
    if prefilled:
        for slot in prefilled:
            if i * bs <= slot < (i + 1) * bs:
                use_dev[slot - i * bs] = False
    return torch.where(use_dev, dev_toks.view(-1), host_toks.view(-1))


def test_stale_token_guard():
    gen = _StubGenerator()
    assert not hasattr(gen, "_slots_prefilled_since_decode")
    got = arm_stale_token_guard(gen, slot=0)
    assert got == {0} and gen._slots_prefilled_since_decode == {0}
    gen._slots_prefilled_since_decode = {3}
    assert arm_stale_token_guard(gen) == {0, 3}
    gen._slots_prefilled_since_decode = set()  # what decode_forward leaves behind after step 0
    assert arm_stale_token_guard(gen) == {0}

    # The hazard: previous request ended at device position 303 with device token 777; the new
    # request's prompt is also 303 tokens long and its first token (from OUR prefill) is 42.
    dev_toks, dev_pos = torch.tensor([777]), torch.tensor([303])
    host_toks, host_pos = torch.tensor([42]), torch.tensor([303])
    stale = _library_reset_merge(dev_toks, dev_pos, host_toks, host_pos, prefilled=set())
    assert int(stale[0]) == 777, "without the guard the library resumes from the stale device token"
    fixed = _library_reset_merge(dev_toks, dev_pos, host_toks, host_pos, prefilled=arm_stale_token_guard(_StubGenerator()))
    assert int(fixed[0]) == 42
    # dev_pos == host_pos + 1 (device one step ahead) is the other accepted case -- the guard covers it too;
    # any other device position takes the host token even without the guard
    assert int(_library_reset_merge(dev_toks, torch.tensor([304]), host_toks, host_pos, set())[0]) == 777
    assert int(_library_reset_merge(dev_toks, torch.tensor([304]), host_toks, host_pos, {0})[0]) == 42
    assert int(_library_reset_merge(dev_toks, torch.tensor([302]), host_toks, host_pos, set())[0]) == 42
    assert int(_library_reset_merge(dev_toks, torch.tensor([100]), host_toks, host_pos, set())[0]) == 42


def test_greedy_sampling_params_are_force_argmax():
    try:
        from models.common.sampling import SamplingParams, format_sampling_params
        from models.common.sampling._utils import is_default_value
    except Exception as e:  # noqa: BLE001
        _skip(f"tt-metal sampling library not importable here: {type(e).__name__}")
    sp = fused.greedy_sampling_params()
    assert isinstance(sp, SamplingParams)
    assert sp.temperature == 0.0 and sp.top_k == 1 and sp.top_p == 1.0
    assert sp.presence_penalty == 0.0 and sp.frequency_penalty == 0.0 and sp.repetition_penalty == 1.0
    fp = format_sampling_params(sp, 32)
    # TTSampling._is_force_argmax_sampling: k == 1, p in {0.0, 1.0}, temp == 1.0 (all lanes)
    assert is_default_value(fp.top_k, 1)
    assert is_default_value(fp.top_p, 0.0) or is_default_value(fp.top_p, 1.0)
    assert is_default_value(fp.temperature, 1.0)
    assert len(fp.temperature) == 32 and fp.seed == [None] * 32
    # and the library's decode_forward accepts sampling_params / reset_batch the way the pipeline calls it
    try:
        from models.tt_transformers.tt.generator import Generator
    except Exception as e:  # noqa: BLE001
        _skip(f"Generator not importable here: {type(e).__name__}")
    params = inspect.signature(Generator.decode_forward).parameters
    for name in ("sampling_params", "reset_batch", "enable_trace", "page_table", "kv_cache"):
        assert name in params, name
    src = inspect.getsource(Generator.decode_forward)
    assert "_slots_prefilled_since_decode" in src, "the stale-token guard targets an attribute the library no longer reads"


# --------------------------------------------------------------------------- N / misc
def test_pad_prompt_ids_and_rows():
    ids = torch.tensor([[5, 6, IMAGE_TOKEN_INDEX, IMAGE_TOKEN_INDEX, 9]])
    assert torch.equal(image_token_rows(ids, IMAGE_TOKEN_INDEX), torch.tensor([2, 3]))
    assert torch.equal(image_token_rows(ids[0], IMAGE_TOKEN_INDEX), torch.tensor([2, 3]))
    padded = pad_prompt_ids(ids, 8, PAD_TOKEN_ID)
    assert padded.tolist() == [5, 6, IMAGE_TOKEN_INDEX, IMAGE_TOKEN_INDEX, 9, PAD_TOKEN_ID, PAD_TOKEN_ID, PAD_TOKEN_ID]
    assert padded.dtype == torch.int64
    assert torch.equal(pad_prompt_ids(ids, 5, PAD_TOKEN_ID), ids[0])


def test_merge_source_rows_spot_checks():
    rows = merge_source_rows(SERVED_GRID)
    h, w = SERVED_GRID
    assert rows.shape == (h * w // 4, 4)
    # output row n = i * 22 + k (i < 12, k < 22) gathers rows (2i + kh) * 44 + (2k + kw), kh,kw in {0,1}
    for n, expect in ((0, [0, 1, 44, 45]), (1, [2, 3, 46, 47]), (22, [88, 89, 132, 133]), (263, [1010, 1011, 1054, 1055])):
        assert rows[n].tolist() == expect, (n, rows[n].tolist())
    assert int(rows.max()) == h * w - 1 and rows.unique().numel() == h * w


# --------------------------------------------------------------------------- plain-script runner
if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    skip_types = (_Skip,) + ((pytest.skip.Exception,) if pytest is not None else ())
    passed = failed = skipped = 0
    for name, fn in tests:
        try:
            fn()
        except skip_types as e:
            skipped += 1
            print(f"SKIP  {name}: {e}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL  {name}: {type(e).__name__}: {e}")
        else:
            passed += 1
            print(f"PASS  {name}")
    print(f"{passed} passed, {failed} failed, {skipped} skipped")
    sys.exit(1 if failed else 0)
