# tt-locate-anything

End-to-end port of NVIDIA
[LocateAnything-3B](https://huggingface.co/nvidia/LocateAnything-3B) — an Eagle-family
visual-grounding / open-vocabulary detection VLM — to Tenstorrent **tt-metal**
(tt-nn + tt-metallium), running on a single Blackhole **p150a** chip.

The model is a **MoonViT-SO-400M** vision tower + a 2×2 patch merger + an `mlp1`
projector feeding a **Qwen2.5-3B-Instruct** language model with an extended
detection vocabulary. Given an image and a free-text query ("locate all the
instances that match …"), it emits `<ref>label</ref><box><x1><y1><x2><y2></box>`
token sequences that decode to pixel boxes.

This repository contains:

- a TT-NN MoonViT vision tower + projector (`locate_anything/tt/vision.py`),
- a thin `Transformer` subclass that drives the Qwen2.5-3B backbone from
  pre-merged image+text embeddings (`locate_anything/tt/model_la.py`),
- an experimental on-device **Parallel Box Decoding** (MTP) decoder
  (`locate_anything/tt/mtp.py`),
- a self-contained torch-CPU reference + golden/oracle builders
  (`locate_anything/reference/`),
- pytest suites for per-stage vision PCC, an end-to-end baseline benchmark with a
  PCC accuracy gate, an MTP fidelity test, and two image-in / boxes-out demos.

Unlike a from-scratch ttnn model, **LocateAnything's LLM backbone reuses
tt-metal's own model libraries** — `models.tt_transformers` (the stock Qwen2.5
`Transformer` / `Generator` / paged-KV `Attention` / `MLP`) and
`models.demos.qwen25_vl` (vision-token merge + prefill prep). So this repo is an
*overlay* on a tt-metal checkout, not a standalone reimplementation: point Python
at a built tt-metal and run from here. See **Environment setup**.

---

## Demo

Greedy autoregressive decode and the experimental hybrid-MTP path, both running
the full pipeline (MoonViT vision + Qwen2.5-3B LLM) on one Blackhole p150a.
Reproduce with `pytest locate_anything/tests/test_demo_visualize.py` (AR) or
`test_demo_mtp_visualize.py` (MTP); query and image are set via `LA_QUERY` /
`LA_IMAGE`.

| Input (`media/demo_input.png`) | AR decode (`media/demo_ar.png`) | Hybrid-MTP (`media/demo_mtp.png`) |
|:---:|:---:|:---:|
| ![](media/demo_input.png) | ![](media/demo_ar.png) | ![](media/demo_mtp.png) |

---

## Contents

```
locate_anything/
├── reference/
│   ├── la_inputs.py             # image preprocess + chat-template build (no cv2/lmdb/decord)
│   ├── extract_llm_checkpoint.py# LocateAnything-3B → vanilla Qwen2.5-3B HF dir for tt_transformers
│   ├── run_reference.py         # HF torch-CPU golden dump (vision + prefill logits) → golden.pt
│   ├── mtp_cpu_loop.py          # correct bsz=1 hybrid/fast MTP loop (torch CPU); the device blueprint
│   └── mtp_oracle.py            # picks a box-yielding (image,query); dumps MTP oracle → mtp_oracle.pt
├── tt/
│   ├── vision.py                # TT-NN MoonViT-SO-400M + mlp1 projector (bf16 / HiFi4)
│   ├── model_la.py              # LATransformer: Qwen2.5-3B with embeds-driven prefill
│   └── mtp.py                   # MTPDecoder: on-device Parallel Box Decoding (experimental)
└── tests/
    ├── test_vision.py           # incremental vision PCC vs golden (gate ≥ 0.99)
    ├── bench_locate_anything.py # baseline benchmark: prefill PCC + greedy AR decode + metrics
    ├── test_mtp.py              # device-MTP vs torch-CPU-MTP logit PCC (fidelity)
    ├── test_demo_visualize.py   # image → boxes (greedy AR) visualization
    └── test_demo_mtp_visualize.py # image → boxes (hybrid MTP) visualization
scripts/
└── download_weights.sh         # pull LocateAnything-3B + extract the Qwen2.5-3B LLM dir
conftest.py                     # loads tt-metal's device fixtures (mesh_device, device_params, …)
```

This repo does **not** vendor the tt-metal monorepo, the model weights, or the
torch goldens. The first come from your tt-metal build; the second from the Hugging
Face Hub; the third are generated locally.

---

## Environment setup

1. **Build tt-metal** with its Python bindings (and Tracy if you plan to profile).
   Instructions: https://github.com/tenstorrent/tt-metal. This is the source of
   `ttnn`, `models.tt_transformers`, and `models.demos.qwen25_vl`, all of which
   this repo imports directly.

2. **Install the Python deps** (the tt-metal `python_env` already has most of
   them). A working set on top of `ttnn`:

   ```bash
   pip install torch torchvision transformers safetensors pillow numpy loguru \
               huggingface_hub matplotlib
   ```

   `torch` can be CPU-only — it is used for the reference, host embedding lookup,
   the vision-token merge, and host argmax sampling.

3. **Point Python at tt-metal** and set the runtime env. The single most important
   variable is `TT_VISIBLE_DEVICES`, which constrains UMD to a single chip:

   ```bash
   export TT_METAL_HOME=/path/to/tt-metal
   export ARCH_NAME=blackhole
   export MESH_DEVICE=N150                  # single-chip 1x1 mesh
   export PYTHONPATH=$PWD:$TT_METAL_HOME:$TT_METAL_HOME/ttnn:$TT_METAL_HOME/tools
   # Run on exactly ONE Blackhole chip. Set BOTH so UMD opens only this chip:
   export TT_VISIBLE_DEVICES=0
   export TT_METAL_VISIBLE_DEVICES=0
   ```

   `conftest.py` re-uses tt-metal's own pytest fixtures and hooks (so
   `mesh_device` / `device_params` / `reset_seeds` behave identically to running
   inside the tt-metal tree); it requires `TT_METAL_HOME` to be set.

4. **Download the weights** and extract the LLM directory:

   ```bash
   bash scripts/download_weights.sh
   # → LocateAnything-3B snapshot (LA_MODEL_PATH) + extracted Qwen2.5-3B (HF_MODEL)
   export HF_MODEL=~/.cache/locate_anything/LA-Qwen2.5-3B
   ```

5. **Generate the torch-CPU goldens** the PCC tests compare against:

   ```bash
   python locate_anything/reference/run_reference.py --in-token-limit 1024  # → golden.pt
   python locate_anything/reference/mtp_oracle.py    --in-token-limit 1024  # → mtp_oracle.pt (MTP test only)
   ```

---

## Running the tests

```bash
# 1) Vision tower per-stage PCC vs the torch golden (gate ≥ 0.99 on vit_proj)
pytest -svq locate_anything/tests/test_vision.py

# 2) Baseline benchmark: prefill PCC vs golden + greedy AR decode + metrics
#    Prints greppable: inference_speed=, accuracy=, peak_dram=, decode_tok_s=, vision_ms=, prefill_ms=
pytest -svq locate_anything/tests/bench_locate_anything.py

# 3) Experimental MTP (Parallel Box Decoding): device-MTP vs torch-CPU-MTP logit PCC
pytest -svq locate_anything/tests/test_mtp.py

# 4) Image → boxes demos (set LA_QUERY / LA_IMAGE / LA_OUT)
LA_QUERY="car" pytest -svq locate_anything/tests/test_demo_visualize.py        # greedy AR
LA_QUERY="car" pytest -svq locate_anything/tests/test_demo_mtp_visualize.py    # hybrid MTP

# 5) Tracy-profiled run (requires a Tracy-enabled tt-metal build)
python -m tracy --no-runtime-analysis --collect-noc-traces \
    --profiler-capture-perf-counters=all -v -r -o ./tracy_out \
    -m pytest locate_anything/tests/bench_locate_anything.py
```

`bench_locate_anything.py` env knobs: `LA_PREC` (`accuracy` default — BF16
attention + BFP8 MLP — or `bfp8attn`), `LA_TRACE` (`1` default, trace-replay
decode), `LA_VISION` (`device` default, or `golden` to feed the CPU vision golden
and isolate LLM PCC).

---

## Results

All numbers are for the fixed golden workload (one image + one query) on a single
p150a, **everything on device** (MoonViT vision + Qwen2.5-3B LLM), warm-measured
with trace-replay decode. The accuracy gate is **PCC ≥ 0.99** against the torch-CPU
reference; the decoded box string matches the HF reference.

### Accuracy (PCC vs torch-CPU golden)

| Stage | PCC |
|---|---:|
| Vision `patch_embed`                  | 0.99999 |
| Vision `encoder_out` (27 blocks)      | 0.9809 |
| Vision `vit_proj` (after `mlp1`)      | 0.9911 |
| LLM prefill last-token logits         | 0.9922 |
| **Full on-device logits (vision→LLM)**| **0.9928** |

The vision encoder is computed in bf16 with HiFi4 (fp32 dest accumulate); the
golden is taken in fp32 so the bf16 port has a fair high-precision target (a 27-layer
bf16 tower drifts on its own). The `mlp1` LayerNorm/GELU lifts the projector PCC
above the encoder's. The LLM uses **BF16 attention + BFP8 MLP weights**; BFP8 MLP is
required for the ≥0.99 gate (BFP4 MLP only reaches ~0.935), and the decode `w1`/`w3`
outputs are spilled to DRAM so the BFP8 weight-stream circular buffers fit L1 on one
chip.

### Performance (warm, trace-replay decode, everything on device)

| Metric | Value |
|---|---:|
| Decode throughput        | **~38 tok/s** |
| Vision (MoonViT + proj)  | ~53 ms |
| Prefill                  | ~64 ms |
| End-to-end               | **~2.32 frames/s** |

Decode is **weight-bandwidth bound** (~3 GB/token streamed), not host-bound — which
is why the trace win is large and the second command queue is not.

### Optimization trajectory

Each row was verified to improve throughput while holding PCC ≥ 0.99 (the baseline
row is below the gate and is the starting point, not a kept result). Discarded
experiments are listed below the table.

| # | Commit | Change | FPS | decode tok/s | PCC |
|---:|---|---|---:|---:|---:|
| 0 | `e99e3bb` | Initial e2e port: Qwen2.5-3B LLM on 1×p150a (perf / BFP4 MLP), CPU vision golden, greedy AR decode | 0.19 | — | 0.935¹ |
| 1 | `e9d4899` | BFP8 MLP (accuracy preset); DRAM-spill decode `w1`/`w3` to fit L1 → meets the gate | 0.57 | 7.5 | 0.992 |
| 2 | `eb03f7b` | Move MoonViT vision + `mlp1` projector on device — full pipeline on one chip | 1.29 | 7.5 | 0.993 |
| 3 | `08af3aa` | Trace-replay decode (5× faster decode) | **2.32** | **38** | 0.993 |

¹ Below the 0.99 gate — kept only as the bring-up baseline; MLP weight precision
(BFP4→BFP8) dominates LLM accuracy, not attention precision.

Rejected after measurement:

- **Second command queue (2CQ).** No gain — the stock `tt_transformers` generator
  only issues on `cq_id=0`, and decode is device-compute / weight-bandwidth bound.
- **All-BFP8 attention + KV.** Slower *and* lower PCC (0.9928 → 0.9912); BF16
  attention is already efficient on Blackhole.
- **MTP / Parallel Box Decoding as the default.** ~60 tok/s over 16 forwards
  (~1.7× decode), but greedy MTP is intrinsically approximate — it degenerates
  under temperature 0 and does **not** reproduce greedy-AR boxes (NVIDIA's hybrid
  mode itself uses sampling). It fails the strict ≥0.99 / match-AR gate, so AR +
  trace stays the default. MTP is kept as an optional fast mode whose **device port
  is faithful to the torch-CPU MTP** (end-to-end device-vs-torch logit PCC ~0.986).

**Within the strict ≥0.99-PCC / deterministic-match gate, trace-AR decode
(38 tok/s, 0.9928 PCC, everything on device) is the speed ceiling.**

---

## Architecture notes

- **MoonViT-SO-400M** (`tt/vision.py`): hidden 1152, 27 encoder layers, 16 heads
  (head_dim 72, padded to 96 for tile alignment), intermediate 4304, patch 14,
  GELU-tanh. `Conv2d` patch embed expressed as a single `(588, 1152)` matmul +
  host-precomputed bicubic-interpolated 2D position embedding; **interleaved-complex
  2D RoPE** mapped onto `ttnn.experimental.rotary_embedding_llama`; full bidirectional
  packed attention via masked SDPA; final LayerNorm; **2×2 patch merge** on host
  (a pure reshape) → `mlp1` projector `LayerNorm(4608) → Linear → GELU → Linear` to
  2048-d. PCC validated at 0.99999 (patch embed) / 0.9911 (projector).
- **Qwen2.5-3B LLM** (`tt/model_la.py`): the stock `tt_transformers` `Transformer`,
  with one added method that feeds **pre-merged image+text embeddings** into prefill
  instead of token ids (standard 1D RoPE, `rope_theta=1e6`; *no* mrope). hidden 2048,
  36 layers, 16 heads / 2 KV heads (GQA), head_dim 128, intermediate 11008, tied
  embeddings, extended vocab 152681. Image embeds are scattered into the text
  embeds at `image_token_index` (151665) on host, then prefill runs on device with
  a paged KV cache; greedy decode runs through the stock `Generator` with trace.
- **Parallel Box Decoding / MTP** (`tt/mtp.py`, experimental): predicts a whole
  6-token box per forward using the block-bidirectional "generation window"
  attention from NVIDIA's reference, with lazy KV commit. Closely reproduces the
  torch-CPU MTP loop in `reference/mtp_cpu_loop.py`.

**Host vs device.** The forward keeps the heavy compute on chip; host work is the
image layout reshape and 2×2 patch merge, the text-token embedding lookup and the
vision↔text embedding merge, and argmax sampling (the 152681-wide vocab exceeds the
on-device-sampling 64K/split limit, but host argmax is negligible since decode is
device-bound).

---

## Known caveats

- **Overlay on tt-metal, not standalone.** The LLM path imports `tt_transformers`
  and `qwen25_vl` directly; `conftest.py` reuses tt-metal's device fixtures. You
  need a built tt-metal checkout on `PYTHONPATH` (`TT_METAL_HOME` set). This is by
  design — the port deliberately reuses the tuned stock Qwen2.5 implementation
  rather than re-deriving it.
- **One image per forward, single chip.** Batch is 1 and the model runs on exactly
  one p150a; nothing is sharded across chips. Set `TT_VISIBLE_DEVICES` to isolate
  the chip you want.
- **MTP is approximate and not accuracy-gated.** It is a faithful device port of the
  torch-CPU MTP (logit PCC ~0.986) and runs the model's intended fast path, but
  greedy MTP ≠ greedy AR by construction; for deterministic, gate-passing detection
  use the default AR path. The MTP demo accepts `LA_TEMP` / `LA_TOP_P` /
  `LA_REP_PEN` for the model's intended sampling.
- **Goldens are generated, not shipped.** `golden.pt` / `mtp_oracle.pt` depend on
  your torch/transformers build; regenerate them with the `reference/` scripts if
  you change the image, query, or token-limit.
- **BFP8 MLP is mandatory** for the ≥0.99 gate; the decode `w1`/`w3` DRAM spill is
  what makes it fit L1 on a single chip (the alternative BFP4 MLP drops PCC to
  ~0.935).

---

## License

Apache 2.0 (matches upstream LocateAnything, Qwen2.5, MoonViT/Kimi-VL, and tt-metal).

---

## Acknowledgements

- Original model: NVIDIA **LocateAnything-3B** — https://huggingface.co/nvidia/LocateAnything-3B
- Language backbone: **Qwen2.5-3B-Instruct** — https://huggingface.co/Qwen/Qwen2.5-3B-Instruct
- Vision tower: **MoonViT** (Kimi-VL / Moonshot AI), SigLIP-SO400M shape
- Runtime: Tenstorrent **tt-metal / tt-nn** — https://github.com/tenstorrent/tt-metal
