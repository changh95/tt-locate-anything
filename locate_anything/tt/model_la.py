# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""TT-NN model wrapper for NVIDIA LocateAnything-3B's LLM backbone.

In autoregressive (AR) mode the LocateAnything LLM is a standard causal
Qwen2.5-3B. The only difference from a vanilla text run is that prefill is fed
*pre-merged* image+text embeddings (host float `[1, S, hidden]`) rather than
token ids. This subclass therefore keeps the entire stock
:class:`models.tt_transformers.tt.model.Transformer` behaviour and only adds an
embeds-driven prefill input preparation method.

RoPE is *standard 1D* (rope_theta=1e6), so we reuse the stock prefill rope
slicing from the parent class -- no mrope (unlike models/demos/qwen25_vl).

Fused path (``TT_FUSED=1``, see ``locate_anything/tt/fused.py``; the legacy
``prepare_inputs_prefill_embeds`` above stays untouched):

* K. vision->LLM boundary on device: the prompt ids go up as a 2 KB uint32 tensor, the stock
  bf16 embedding table gathers them (``self.embd``, the same checkpoint rows the host
  ``F.embedding`` used), and ONE 0/1 permutation matmul over ``concat([E_ids, vit_proj])``
  places the vision rows at the image-token positions and the pad-token rows after the prompt --
  exactly ``merge_vision_tokens`` + ``preprocess_inputs_prefill`` (HiFi4 0/1 matmul on bf16 is
  a bit-exact row gather), without the 1.5 MB vision readback, the host merge/pad and the 2 MB
  embeds tilize+upload.
* L. the 36 decoder layers run as ONE metal trace (``get_last_token=-1`` returns the hidden
  state); the slice -> RMSNorm -> LM head tail runs eagerly with the request's
  ``last_token_idx`` through the library's own ``process_logits_after_prefill_trace`` -- the
  same ops the in-graph tail runs today, so one trace serves every prompt in the bucket. The
  page table and RoPE slices are persistent device tensors (allocated at build, before any
  trace is captured).
* N. the first token comes from ONE logits row (untilize + slice, 305 KB) instead of the
  32 x vocab (9.77 MB) readback; the argmax runs on host on the same bf16 row as today.
"""

from dataclasses import dataclass, field

import torch

import ttnn
from models.tt_transformers.tt.model import Transformer
from locate_anything.tt.fused import build_embed_merge_table, pad_prompt_ids


@dataclass
class FusedPrefillState:
    """Persistent device tensors of the fused prefill (one bucket, one canonical grid)."""

    prefill_len: int
    n_img_pad: int
    image_rows: torch.Tensor  # LongTensor: positions of the image tokens in the prompt
    ids_dev: ttnn.Tensor  # [1,1,1,prefill_len] uint32 ROW_MAJOR, refreshed per request
    merge_table: ttnn.Tensor  # [1,1,prefill_len, prefill_len + n_img_pad] bf16 TILE (0/1)
    page_table: ttnn.Tensor  # [1, max_num_blocks] int32 ROW_MAJOR (constant per server)
    rot_mats: list  # [cos, sin] device slices [1,1,prefill_len,head_dim]
    trace_id: object = None
    hidden_out: ttnn.Tensor = None  # trace output [1,1,prefill_len,dim]
    vit_in_address: int = None  # buffer address of the vision tensor the trace was recorded on
    extra: dict = field(default_factory=dict)


class LATransformer(Transformer):
    """Qwen2.5-3B backbone for LocateAnything with embeds-driven prefill."""

    def prepare_inputs_prefill_embeds(
        self,
        embeds,
        start_pos=0,
        page_table=None,
        chunk_page_table=None,
        last_token_idx=None,
    ):
        """Prepare prefill inputs from pre-merged host embeddings.

        Mirrors the stock :meth:`Transformer.prepare_inputs_prefill` rope-slicing
        logic, but takes already-embedded inputs (image+text merged on host)
        instead of token ids.

        Args:
            embeds: torch float tensor of shape [B=1, S, hidden_dim].
            start_pos: position offset for the rope slice (default 0).
            page_table: optional torch int tensor for paged attention.
            chunk_page_table: optional torch int tensor for chunked prefill.
            last_token_idx: index of the last meaningful token; used to validate
                the requested sequence length fits in the precomputed rope mats.

        Returns:
            (tokens_embd, [cos_slice, sin_slice], tt_page_table, tt_chunk_page_table)
        """
        assert embeds.dim() == 3, "embeds must be a 3D tensor [B=1, S, hidden]"
        assert embeds.shape[0] == 1, "LocateAnything only supports batch_size=1"
        S = embeds.shape[1]

        # [1, S, hidden] -> [1, 1, S, hidden]; shard the hidden dim across the mesh
        tokens_embd = ttnn.from_torch(
            embeds.unsqueeze(1),
            device=self.mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                mesh_device=self.mesh_device,
                dims=(None, 3),
                mesh_shape=self.args.cluster_shape,
            ),
        )

        # --- Stock prefill RoPE slicing (copied from Transformer.prepare_inputs_prefill) ---
        mat_len = self.rope_setup.cos_matrix_prefill.shape[2]
        seq_len = last_token_idx + 1 if last_token_idx is not None else S
        assert mat_len >= seq_len, f"Sequence length {seq_len} exceeds max seq len {mat_len}"

        required_end = start_pos + S
        pad_len = max(0, required_end - mat_len)

        # Slice the precomputed (on-device) cos/sin prefill matrices.
        slice_end = min(mat_len, required_end)
        cos_slice = self.rope_setup.cos_matrix_prefill[:, :, start_pos:slice_end, :]
        sin_slice = self.rope_setup.sin_matrix_prefill[:, :, start_pos:slice_end, :]

        if pad_len > 0:
            # Pad at end of 3rd dim (dim=2) by pad_len.
            padding = [(0, 0)] * 4
            padding[2] = (0, pad_len)
            cos_slice = ttnn.pad(cos_slice, padding=padding, value=0.0)
            sin_slice = ttnn.pad(sin_slice, padding=padding, value=0.0)

        tt_rot_mats_prefill_global = [cos_slice, sin_slice]

        if page_table is not None:
            tt_page_table = ttnn.from_torch(
                page_table,
                device=self.mesh_device,
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )
        else:
            tt_page_table = None

        if chunk_page_table is not None:
            tt_chunk_page_table = ttnn.from_torch(
                chunk_page_table,
                device=self.mesh_device,
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )
        else:
            tt_chunk_page_table = None

        return tokens_embd, tt_rot_mats_prefill_global, tt_page_table, tt_chunk_page_table

    # ------------------------------------------------------------------ fused prefill (K, L, N)
    def build_fused_prefill_state(self, prefill_len, n_img_pad, image_rows, page_table):
        """Allocate the persistent device tensors of the fused prefill for ONE bucket
        (``prefill_len``, a power of two >= the prompt) and ONE canonical grid (``image_rows``,
        the constant positions of the image tokens; ``n_img_pad`` = padded row count of the
        vision output). Must run before any trace is captured (see the module docstring)."""
        prefill_len = int(prefill_len)
        mat_len = self.rope_setup.cos_matrix_prefill.shape[2]
        assert mat_len >= prefill_len, f"prefill bucket {prefill_len} exceeds the RoPE table {mat_len}"
        image_rows = image_rows.reshape(-1).long()

        ids_dev = ttnn.from_torch(
            torch.zeros(1, 1, 1, prefill_len, dtype=torch.int64),
            device=self.mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
        table = build_embed_merge_table(image_rows, prefill_len, int(n_img_pad))
        merge_table = ttnn.from_torch(
            table.reshape(1, 1, prefill_len, prefill_len + int(n_img_pad)),
            device=self.mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        tt_page_table = ttnn.from_torch(
            page_table,
            device=self.mesh_device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
        # Static RoPE slice [0, prefill_len) -- the same slice the legacy path takes per request.
        rot_mats = [
            self.rope_setup.cos_matrix_prefill[:, :, 0:prefill_len, :],
            self.rope_setup.sin_matrix_prefill[:, :, 0:prefill_len, :],
        ]
        # 0/1 gather matmul: HiFi4 keeps the full bf16 mantissa, fp32 accumulation adds nothing.
        self.ck_merge = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )
        return FusedPrefillState(
            prefill_len=prefill_len,
            n_img_pad=int(n_img_pad),
            image_rows=image_rows,
            ids_dev=ids_dev,
            merge_table=merge_table,
            page_table=tt_page_table,
            rot_mats=rot_mats,
        )

    def host_prefill_ids(self, input_ids, prefill_len, pad_token_id):
        """Prompt ids [1, S] -> host ttnn tensor [1,1,1,prefill_len] uint32 (prompt then pad ids),
        the spec of ``FusedPrefillState.ids_dev`` for ``ttnn.copy_host_to_device_tensor``."""
        ids = pad_prompt_ids(input_ids, prefill_len, pad_token_id).reshape(1, 1, 1, -1)
        return ttnn.from_torch(
            ids,
            device=None,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

    def fused_embeds_device(self, st, vit_dev):
        """K: ids_dev + vit_proj [1,1,n_img_pad,dim] -> merged prefill embeddings [1,1,prefill_len,dim]
        = merge_table @ concat([embd(ids), vit_proj], rows). Exact (see module docstring)."""
        e_ids = ttnn.unsqueeze_to_4D(self.embd(st.ids_dev))  # [1,1,prefill_len,dim] bf16 TILE
        stacked = ttnn.concat([e_ids, vit_dev], dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(e_ids)
        embeds = ttnn.matmul(
            st.merge_table, stacked, compute_kernel_config=self.ck_merge, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        ttnn.deallocate(stacked)
        return embeds

    def fused_prefill_graph(self, st, vit_dev, kv_cache):
        """K + the 36 layers (``get_last_token=-1``): device in / device out, trace-capturable.
        Returns the hidden state [1,1,prefill_len,dim]; KV is written into ``kv_cache`` through
        ``st.page_table`` (user 0), exactly like the legacy prefill."""
        embeds = self.fused_embeds_device(st, vit_dev)
        hidden = self.ttnn_prefill_forward(
            embeds,
            rot_mats_global=st.rot_mats,
            rot_mats_local=None,
            user_id=0,
            page_table=st.page_table,
            get_last_token=-1,
            kv_cache=kv_cache,
        )
        ttnn.deallocate(embeds)
        return hidden

    def fused_prefill_logits(self, hidden, last_token_idx):
        """Eager tail: slice(32-row window) -> RMSNorm -> LM head -> logits [1,1,32,vocab_pad]
        (the library's ``process_logits_after_prefill_trace``, the very ops the legacy in-graph
        ``get_last_token`` tail runs)."""
        return self.process_logits_after_prefill_trace(hidden, last_token_idx)

    def first_token_from_logits(self, tt_logits, last_token_idx, small_readback):
        """Greedy first token from the device logits [1,1,32,vocab_pad].

        ``small_readback=False``: the legacy 9.77 MB readback (``process_output_prefill`` on
        ``tt_logits.cpu()``). ``True`` (N): untilize on device, slice the one row
        ``last_token_idx % 32`` and read 305 KB. Both hand the SAME bf16 row (``[:vocab_size]``)
        to ``torch.argmax`` -- identical value and tie-breaking."""
        row_idx = int(last_token_idx) % 32
        if small_readback:
            rm = ttnn.untilize(tt_logits, use_multicore=True, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            row = ttnn.slice(rm, [0, 0, row_idx, 0], [1, 1, row_idx + 1, rm.shape[-1]])
            ttnn.deallocate(rm)
            row_host = ttnn.to_torch(row)[0, 0, 0, : self.vocab_size]
            ttnn.deallocate(row)
        else:
            row_host = self.process_output_prefill(tt_logits.cpu(), last_token_idx=row_idx)[: self.vocab_size]
        return int(torch.argmax(row_host).item())
