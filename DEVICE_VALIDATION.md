# Device validation plan for the `TT_FUSED=1` paths (branch `opt/locate-anything-3b-p150-megakernel`)

Everything on this branch was written and tested WITHOUT a device (host torch tests, host-ttnn
import checks, reading the op validators in the gbp-tt tree). Nothing below has run on a p150a
yet; every ms figure is an estimate from `reports/megakernel/locate-anything-3b-p150.md`. The
knob is OFF by default: with `TT_FUSED` unset the package issues the same device ops as
`tt-model-package`, in the same order with the same arguments (same numerics; verified by code
review only, 3.1 measures it). The only host-visible differences with the knob off are a new
`"fused": None` key in the `LocateAnythingPipeline.run()` result dict and a `fused` entry in
`GET /info`; the `/predict` body is unchanged. The shipped card numbers are the baseline to beat.

## 0. What is on the branch (all behind `TT_FUSED=1`, read once at build)

| lever | where | exactness (host claim) | sub-knob (default) |
|---|---|---|---|
| A. whole MoonViT graph as one metal trace, persistent pixel input | `tt/vision.py` `capture_trace` / `forward_device` | bit-identical (same ops) | `LA_FUSED_VISION_TRACE=1` |
| B. device patch merger: 4 x 0/1 permutation matmuls + concat | `tt/vision.py` `_merge_device`, tables `tt/fused.py:build_merge_perms` | bit-identical (host-proven vs `patch_merge_host`) | intrinsic |
| C. exact L=1056 (33 tiles): no 128-padding, NO SDPA mask | `tt/fused.py:vision_seq_pad`, `MoonViT.attn_mask=None` | bit-identical on kept rows / bf16-rounding (chunk boundaries) | intrinsic when L % 32 == 0 |
| D. `dit_minimal_matmul_addcmul_fused` for wo+residual, fc1+residual, patch-embed+pos_emb | `MoonViT._matmul_residual`, `_patch_embed_device` | bf16-rounding (one rounding instead of two) | `LA_FUSED_MATMUL=minimal` |
| E. `minimal_matmul` for wqkv / fc0 / mlp1 | `MoonViT._matmul` | bf16-rounding (blocking) | `LA_FUSED_MATMUL=minimal`, `LA_FUSED_MM_BLOCKS=8,4,4,2,2`, `LA_FUSED_DIT_BLOCKS=4,4,4,2,2` |
| F. `SDPAProgramConfig` chunks + exact exp | `MoonViT.sdpa_pc` | **precision-affecting**: chunk sizes change the softmax rescale boundaries AND `exp_approx_mode=False` replaces the approximate exp the legacy call gets by default (no `program_config` -> `sdpa_program_factory.cpp` `get_exp_approx_mode` "defaulting to true", chunks 32/32). rf-detr ran exact exp at no measurable cost with accuracy up; unmeasured for this model | `LA_FUSED_SDPA_CHUNKS=96,352`, `LA_FUSED_SDPA_EXP_APPROX=0` (legacy kernel config = `32,32` + `EXP_APPROX=1`) |
| H. tanh GELU (reference variant) | `MoonViT._gelu` | precision-affecting | `LA_FUSED_GELU=erf` (A/B: `tanh`) |
| I. vision working set in L1 | `MoonViT.mem` | bit-identical | `LA_FUSED_L1=0` (A/B: `1`) |
| J. ROW_MAJOR pixel upload + in-graph `tilize_with_zero_padding` | `MoonViT.host_input` / `_ingest` | bit-identical (same `ttnn.from_torch` bf16 conversion) | `LA_FUSED_RM_INPUT=1` |
| K. vision->LLM merge on device (ids upload, `embd` gather, 0/1 merge matmul) | `tt/model_la.py` `fused_embeds_device`, table `tt/fused.py:build_embed_merge_table` | bit-identical (host-proven vs `merge_vision_tokens` + `preprocess_inputs_prefill`) | `LA_FUSED_DEVICE_MERGE=1` |
| L. 36-layer prefill as one metal trace + eager slice/norm/LM-head tail | `pipeline.capture_traces`, `LATransformer.fused_prefill_graph` / `fused_prefill_logits` | bit-identical (library's own `process_logits_after_prefill_trace`) | `LA_FUSED_PREFILL_TRACE=1` |
| M. on-device greedy sampling in the decode loop + stale-token guard | `pipeline.run` (`greedy_sampling_params`, `arm_stale_token_guard`) | exact except argmax tie-breaking | `LA_FUSED_DEVICE_SAMPLING=1` |
| N. first token from ONE logits row (untilize + slice, 305 KB) | `LATransformer.first_token_from_logits` | bit-identical (same bf16 row to the same `torch.argmax`) | `LA_FUSED_SMALL_READBACK=1` |
| trace region default 50 MB -> 160 MB when the knob is on | `tt/fused.py`, `server/app.py` | n/a | `LA_TRACE_REGION_SIZE` |

Host evidence already collected (re-run first, section 2): `code/locate_anything/tests/test_fused_host.py`
14 passed (tree python 3.10, torch 2.11, pytest 9), and every edited module imports with the host ttnn.

## 1. Setup (from `SERVING.md` section 1)

```bash
ROOT=/home/deepgadget/experiments/tt-models
REPO=$ROOT/models/locate-anything-3b-p150          # branch opt/locate-anything-3b-p150-megakernel
TREE=/home/deepgadget/experiments/gbp-tt/tt-metal   # tt-model.yaml source.tt_metal (main 8b98410e730)
PY=$TREE/python_env/bin/python
export TT_METAL_HOME=$TREE ARCH_NAME=blackhole
export PYTHONPATH=$REPO/code:$TREE:$TREE/ttnn        # repo code FIRST (mlp.py overlay must shadow stock)
export HF_MODEL=~/.cache/tt-model/locate-anything-3b-p150/weights/la-qwen2_5-3b   # extracted Qwen2.5-3B
export LA_WEIGHTS_DIR=$(ls -d ~/.cache/huggingface/hub/models--nvidia--LocateAnything-3B/snapshots/c32291ca*)
export TT_CACHE_PATH=~/.cache/tt-model/locate-anything-3b-p150/tensors
cd $REPO/code
```

Legacy baseline for every comparison (`TT_FUSED` unset): card numbers of 2026-09-12, warm
`/predict` with `media/demo_input.png` + `car`, 10 tokens: vision 47.4 ms, prefill 55.8 ms,
decode 224.0 ms (22.4 ms/token), total 343.4 ms; `raw_text` =
`<ref>car</ref><box><282><414><606><794></box><|im_end|>`; PCC vs torch reference: vision
projector 0.9911, prefill 0.9922, full 0.9928 (gate >= 0.99).

## 2. Host checks (no device; re-run on the validation host before touching the chip)

```bash
$PY -m pytest --noconftest -q -p no:cacheprovider locate_anything/tests/test_fused_host.py   # expect 14 passed
$PY -c "import locate_anything.tt.vision, locate_anything.tt.model_la, locate_anything.tt.pipeline, locate_anything.tt.fused"
```

## 3. Step-by-step device pass

Gates used throughout: **G1** box string on the demo image equals the legacy string above;
**G2** generated token ids equal the legacy run token-for-token on >= 3 queries (`car`,
`person</c>car`, one ~140-char query) for the bit-identical levers, and PCC >= 0.99 (projector,
prefill, full) plus G1 for the bf16-rounding / precision-affecting ones; **G3** 20 back-to-back
requests alternating two images and three queries return the same ids as the same requests run
individually (no cross-request state leak); **G4** no hang, no `TT_THROW`, cold boot < 30 min.

### 3.1 Legacy regression (knob off) -- must be a no-op

```bash
unset TT_FUSED
$PY -m pytest -svq locate_anything/tests/test_vision.py                       # PCC gates as before
$PY -m pytest -svq locate_anything/tests/bench_locate_anything.py             # accuracy=..., inference_speed=...
```
Expect the card numbers (section 1). Any difference here is a bug on this branch.

### 3.2 Fused vision ops, eager, on the PCC golden (exercises D/E/F/J and the legacy pad+mask branch, L=1092)

```bash
TT_FUSED=1 $PY -m pytest -svq locate_anything/tests/test_vision.py
TT_FUSED=1 LA_FUSED_MATMUL=linear $PY -m pytest -svq locate_anything/tests/test_vision.py
```
`forward(return_intermediates=True)` runs the fused graph eagerly (no trace). Gate: `vit_proj`
PCC >= 0.99 (legacy 0.9911); `patch_embed` / `encoder_out` / `vit_raw(merged)` >= 0.99. The golden
grid (26x42) is NOT tile aligned, so this run keeps the 1152 padding + mask -- lever C is
exercised only at the served grid (3.3).

### 3.3 Served-grid vision exactness / PCC, legacy vs fused (same pixels, one device, no server)

```python
# TT_METAL_HOME/PYTHONPATH as in section 1; run with $PY
import torch, ttnn
from PIL import Image
from locate_anything.reference import la_inputs
from locate_anything.tt.fused import FusedConfig
from locate_anything.tt.vision import MoonViT
from locate_anything.tt import pipeline as P
from models.common.utility_functions import comp_pcc
dev = P.open_mesh_device((1, 1), trace_region_size=160_000_000)
snap = la_inputs.find_model_path()
pix, grid = la_inputs.preprocess_image_fixed(Image.open("../media/demo_input.png"), (24, 44), 1024)
legacy = MoonViT(dev, snap, grid, fused=FusedConfig.disabled())
ref = legacy.forward(pix.float()); del legacy
import gc
LIN = {"TT_FUSED": "1", "LA_FUSED_MATMUL": "linear"}
for env in ({**LIN, "LA_FUSED_SDPA_CHUNKS": "32,32", "LA_FUSED_SDPA_EXP_APPROX": "1"},   # legacy SDPA kernel config
            {**LIN, "LA_FUSED_SDPA_EXP_APPROX": "1"},                                   # chunking-only delta
            LIN,                                                                        # + exact exp (F default)
            {"TT_FUSED": "1"}, {"TT_FUSED": "1", "LA_FUSED_L1": "1"},
            {"TT_FUSED": "1", "LA_FUSED_GELU": "tanh"}, {"TT_FUSED": "1", "LA_FUSED_SDPA_CHUNKS": "96,1056"}):
    vis = MoonViT(dev, snap, grid, fused=FusedConfig.from_env(env))
    eager = vis.forward(pix.float())                       # fused ops, no trace
    vis.capture_trace(pix.float()); traced = vis.forward(pix.float())   # trace replay
    print(env, "eager", comp_pcc(ref, eager, 0.99)[1], "max|d|", (ref - eager).abs().max().item(),
          "trace==eager", torch.equal(eager, traced))
    vis.release_trace()   # frees the trace-region slot and the persistent in/out buffers ...
    del vis; gc.collect()  # ... and the ~1 GB of bf16 weights before the next instance is built
ttnn.close_mesh_device(dev)
```
Seven instances share one device session, so each trace MUST be released before the next capture
(160 MB trace region; `MoonViT.release_trace`) and the weights collected explicitly.
Expectations: `trace==eager` True for every config (A is bit-identical to the eager fused graph).
The first config (`linear`, chunks `32,32`, `EXP_APPROX=1`) is the legacy SDPA kernel
configuration with only the mask removed and the padding rows dropped (B, C, J are exact), so it
is the one expected to sit within a few bf16 ULP of `ref` on the kept rows -- if it does not, a
lever other than F moved the numbers. The next two rungs isolate F: `EXP_APPROX=1` with the
default `96,352` chunks measures the rescale-boundary effect alone, `LIN` adds the exact exp.
Neither has a ULP expectation: gate them with PCC >= 0.99 vs the torch golden (`test_vision.py`,
legacy 0.9911, expected flat-to-up as in rf-detr) and G1/G2 in 3.4. `minimal` (D/E): PCC >= 0.99
vs the golden and G1; `tanh` GELU: PCC vs the torch golden expected UP from 0.9911 -- gate it on
G1, not PCC alone.

### 3.4 End-to-end A/B ladder (server or `LocateAnythingPipeline` on the host)

Run each rung as `tt-model serve` with the env below (or the SERVING.md section 1 host uvicorn), then,
from the section-1 working directory `$REPO/code`,
`$PY locate_anything/server/smoke_test.py --url http://127.0.0.1:20000` (G1; the script needs only
`urllib` + `PIL`, so any python with Pillow works) and the
3-query id comparison (G2) against rung 0; record `timing_ms` from the response (`/info` shows
`fused.config` and `fused.status`). NOTE: with the vision trace, `timing_ms.vision` is the
non-blocking submit and the wait moves into `prefill`; compare `total`.

| rung | env | levers added | estimate (NOT measured) |
|---|---|---|---|
| 0 | `TT_FUSED` unset | legacy | 343.4 ms |
| 1 | `TT_FUSED=1 LA_FUSED_MATMUL=linear LA_FUSED_VISION_TRACE=0 LA_FUSED_DEVICE_MERGE=0 LA_FUSED_PREFILL_TRACE=0 LA_FUSED_SMALL_READBACK=0 LA_FUSED_DEVICE_SAMPLING=0` | B, C, F, J (eager) | vision -3..-4 ms |
| 2 | rung 1 with `LA_FUSED_MATMUL=minimal` | + D/E | vision -2..-7 ms |
| 3 | rung 2 with `LA_FUSED_VISION_TRACE=1` | + A | vision -6..-8 ms |
| 4 | rung 3 with `LA_FUSED_DEVICE_MERGE=1 LA_FUSED_SMALL_READBACK=1` | + K, N | glue/prefill -3..-4 ms |
| 5 | rung 4 with `LA_FUSED_PREFILL_TRACE=1` | + L | prefill -3..-6 ms |
| 6 | `TT_FUSED=1` (all defaults) | + M | decode -2..-3 ms/token |
| A/B | `TT_FUSED=1 LA_FUSED_L1=1`; `LA_FUSED_GELU=tanh`; F: `LA_FUSED_SDPA_CHUNKS=32,32 LA_FUSED_SDPA_EXP_APPROX=1` (= legacy kernel config) vs `LA_FUSED_SDPA_EXP_APPROX=1` alone (chunking only) vs default (chunks + exact exp) vs `LA_FUSED_SDPA_CHUNKS` in `96,1056` / `128,352`; `LA_FUSED_MM_BLOCKS=4,4,4,2,2` (any `sub_h*sub_w > 4` is rejected at build, see 4.4) | I, H, F, E variants | I -2..-4 ms; others unknown |

Total estimate for rung 6: 343 -> ~285-305 ms (-11..-17 %). If a rung fails to boot, the log
line `TT_FUSED status after warm-up` / the `RuntimeError` names the missing capture; turn that
sub-knob off and continue the ladder.

### 3.5 Lever-specific checks

* **K exactness on device** (bit-identical claim): in a host session with the pipeline built
  (`TT_FUSED=1`), after `pipe.warmup()`, compute the legacy host embeds for a prompt (the
  `F.embedding` / `merge_vision_tokens` / `preprocess_inputs_prefill` block of `pipeline.run`) with
  `vit_proj = pipe.vis.read_projection(pipe.vis.trace_output)` and compare
  `ttnn.to_torch(pipe.model.fused_embeds_device(pipe._fused_prefill, pipe.vis.trace_output))[0, 0]`
  to `embeds[0].to(torch.bfloat16)` with `torch.equal`. Same for B: `torch.equal` of the device
  merged tensor (`inter["merged"]` from `forward(return_intermediates=True)`) against
  `patch_merge_host(inter["encoder_out"])`.
* **M tie-breaking**: G2 token-for-token on >= 3 queries x >= 10 tokens; also force a tie check by
  comparing `ttnn.argmax` vs `torch.argmax` on the legacy logits of the first step (read them back
  once with `LA_FUSED_DEVICE_SAMPLING=0`). A mismatch is a tie (both arguments have equal bf16
  logits) or a bug -- inspect the logit values before deciding.
* **M stale-token hazard**: two consecutive requests with prompts of the SAME token count (e.g.
  `car` then `dog`, both 303 tokens) on different images; the second must not start from the first
  request's last device token (`arm_stale_token_guard` makes `decode_forward` take the host token;
  the log-free way to see it is G2 on the second request).
* **L/A trace hygiene**: the prefill trace refuses to replay when the vision output buffer moved
  (`RuntimeError: fused prefill trace was recorded on a different vision output buffer`) -- if
  that fires, the vision trace was recaptured or `LA_FUSED_VISION_TRACE=0`; both traces must be
  captured in `pipeline.capture_traces` after warm-up run 1 (which also creates the decode
  trace's persistent inputs). Run G3 for allocation-after-capture corruption (section 4.2 says
  which buffers are exposed and why the per-request order makes it benign).
* **Cold boot**: three captures (decode+sampling in run 1, vision + prefill after it) -- record
  the warm-up time; the shipped boot is 101 s (39 s warm-up run 1).

## 4. Constraints that could NOT be verified on the host (check these first when something fails)

1. Trace-region budget: vision (~340 ops) + prefill (~800 ops) + decode (~950) + sampling traces in
   `LA_TRACE_REGION_SIZE` (default 160 MB with the knob; 50 MB legacy). Symptom: capture failure /
   `trace_region_size` TT_FATAL -> raise the env var.
2. Allocation-after-capture: buffers allocated after a trace is recorded may sit in that trace's
   scratch and be overwritten by its replays (`Generator._prepare_decode_trace_text` docstring).
   Two directions matter:
   * fused buffers created at BUILD (`ids_dev`, merge table, page table, RoPE slices, KV cache,
     merge perms, ones vector) exist before any capture -> safe by construction;
   * the vision trace's `_persistent_in` / `_trace_out` and the prefill trace's `hidden_out` are
     allocated in `capture_traces`, i.e. AFTER the library decode(+sampling) trace was captured in
     warm-up run 1, so they MAY live in the decode trace's scratch and be clobbered on every
     decode replay. This is benign only through ordering, not isolation: each request runs
     copy -> vision trace -> prefill trace -> eager tail -> decode, so every one of those buffers
     is fully rewritten before it is read again. Any change that reads `vis.trace_output` or
     replays the prefill trace after a decode step (multi-turn, batching, re-using the vision
     output across queries) must re-run the vision trace first or move the captures before run 1.
   The sampling module's per-step host copies / `seed_manager.get_new_values` were not audited ->
   G3 (20 alternating requests) is the observable check for both.
3. `ttnn.tilize_with_zero_padding` on a ROW_MAJOR bf16 `[1,1,1056,608]` upload (row = 1216 B) and
   `ttnn.to_device` / `copy_host_to_device_tensor` of that host tensor on a 1x1 MeshDevice.
4. `dit_minimal_matmul_addcmul_fused` / `minimal_matmul` at these shapes: M = 1056 (33 tiles, not a
   multiple of the 4/8-tile M blocks), fc1 K = 4304 logical / 4320 padded (135 K tiles), wqkv N =
   4608 (144 tiles), patch-embed K = 608 (19 tiles), mlp1 K = 4608 / N = 2048; the ones scale vector
   `[1,1,1,1152]` must share the residual's buffer type (DRAM by default, L1 under I); `dtype=bf16`
   output. Kernel speed vs `ttnn.linear` unknown (rf-detr's 3-5x was on 3-10x smaller N).
   Dest-register cap (host-checked, device TT_FATAL otherwise): `subblock_h * subblock_w <=
   get_dest_reg_count(compute_kernel_config)` = 16 / 2 (`dst_full_sync_en=False`, the
   `WormholeComputeKernelConfig` default) / 2 (`fp32_dest_acc_en=True` in the port's `ck_hifi4`) =
   **4 tiles**, so the default 2x2 subblocks sit exactly at the cap and the op docstring's
   "typical" 2x4 / 4x2 are illegal here (`validate_mm_blocks` rejects them at build). rf-detr ran
   2x2 with `fp32_dest_acc_en=False` (cap 8) -- its margin does not transfer. If a larger subblock
   is wanted for speed, the matmul compute config has to drop fp32 accumulation (precision-affecting,
   gate on G1/PCC) or enable `dst_full_sync_en`.
5. SDPA with `q_chunk=96, k_chunk=352` on `[1,16,1056,96]` (K/V chunk 352x96 bf16 = 66 KB per head
   per core + scores) -- L1 fit unverified; no mask, `Sq = Sk = 1056` unpadded; alternative
   `96,1056` single pass (405 KB K+V). `exp_approx_mode=False` by default (`LA_FUSED_SDPA_EXP_APPROX=0`):
   the legacy call has no program config and therefore runs the kernel's default approximate exp
   with 32/32 chunks, so lever F is precision-affecting even at `LA_FUSED_MATMUL=linear`; the
   3.3 ladder isolates the two components.
6. `rotary_embedding_llama` with cos/sin `[1,1,1056,96]` matching the input Ht = 33 (odd tile
   count; today the legacy path runs Ht 36 vs 33 with a zero-fill warning).
7. `nlp_create_qkv_heads` / `nlp_concat_heads` on seq 1056 (33 tiles) and `ttnn.concat` of four
   TILE `[1,1,384,1152]` tensors along the last dim, and of `[1,1,512,2048]` + `[1,1,384,2048]` along
   dim 2 (K).
8. `ttnn.embedding` of a `[1,1,1,512]` uint32 ROW_MAJOR id tensor with `layout=TILE` -> `[1,1,512,2048]`
   (the library's own text-prefill shape; `unsqueeze_to_4D` is applied as there).
9. `ttnn.untilize` of the BFP8 LM-head logits `[1,1,32,152704]` (the decode path does this on the
   same tensor spec) followed by a ROW_MAJOR `ttnn.slice` of one row (N).
10. `Tensor.buffer_address()` on a MeshDevice tensor is used only as a guard (prefill trace input
    identity); if it raises on a mesh tensor, replace the check with a Python identity check.
11. `TTSampling` force-argmax on Blackhole for vocab 152704 (4 splits of 38176) and the
    `ttnn.argmax` tie rule (torch: first index).
12. GELU tanh variant numerics on Blackhole (H) and the L1 working set next to the LLM's resident
    buffers (I; this model already needed the `mlp.py` DRAM spill for decode CBs).
13. Every ms estimate in section 3.4.

## Results (device, 2026-09-13)

Hardware pass on the p150a (one validation agent, one device). Evidence: `logs/megakernel-validate/locate-anything-3b/`
(every log, probe script and JSON named below), one row per experiment in `reports/megakernel/VALIDATION.md`.
Environments: host `gbp-tt/tt-metal/python_env` (py3.10) for the model's own pytest suites and the probes; the shipped
image `tt-model/locate-anything-3b-p150:04871c7cfa03` with the branch code bind-mounted (`run_img.sh`, flags from
`tt-model serve --print`) for the served A/B and the final gate. No dev image was needed (pytest is in the image), but
the pytest DEVICE tests cannot run there (`/opt/tt-metal/conftest.py` is this repo's conftest, which execs itself, and
tt-metal's root conftest needs `tests/scripts` that are not packaged), so the in-image gate ran the fixture-free
equivalents `probe_pipeline.py` / `probe_vision_grid.py` (same code paths as the server and `test_vision.py`).

### 0. Goldens (not shipped) and the HF reference

`run_reference.py --in-token-limit 1024` re-run with the port's own versions (torch 2.7.1 / transformers 4.53.0,
`base/tt-metal/python_env`; the gbp-tt venv's transformers 5.12.1 rejects the snapshot remote code): `golden.pt`
(HF `assets/teaser.jpg`, `person</c>car`, grid 26x42) and `golden_demo.pt` (`media/demo_input.png`, `car`, the served
24x44 grid). The HF model's own `generate(generation_mode="slow", temperature=0)` was also run on CPU for 15 prompts
(3 demo queries + 12 asset-image prompts squashed to 24x44 like the server; `gen_ref_outputs.py`, `data/ref_outputs.json`).
**Finding: the HF reference for the demo image + `car` is `<ref>car</ref><box><282><414><607><797></box>` (native bf16
and fp32-vision alike), not the shipped device string `<606><794>` (IoU 0.989).** The card's "matches the HF reference"
was the legacy device output; gate G1 below was therefore anchored on the legacy string, not on the reference.

### 1. Legacy regression (`TT_FUSED` unset on the branch = today's `TT_FUSED=0`)

`test_vision.py`: 26x42 golden patch_embed 0.9999958 / encoder_out 0.98553 / vit_raw 0.98553 / **vit_proj 0.99240**
(card 0.9911 with the original golden); 24x44 golden 0.9999965 / 0.97683 / 0.97683 / **0.99575**. `bench_locate_anything.py`
(port golden): full-pipeline last-token logits PCC **0.99189** (card 0.9928), vision warm 49.4 ms, prefill 54.9 ms,
40.0 tok/s, peak DRAM 6.56 GB; `LA_VISION=golden` (LLM only) **0.99146** (card 0.9922). Host pipeline 30 x demo+`car`:
total 360.9 ms median (349.0-373.8; vision 48.4, prefill 63.2, decode 228.2 = 39.5 tok/s). Served in the shipped image
(pure, no mounts): smoke PASS, 340 ms. The box string `<box>None</box>` seen for teaser+`person</c>car` is token 4064 =
the literal text `None`, the model's "no instance" answer (not an out-of-vocab id). All within noise of the card; the
branch's legacy path is the shipped graph (`s1_legacy.log`, `s4_rung0.json`, `s5_serve_legacy.log`).

### 2. Host tests

`test_fused_host.py` 14 passed (before and after the knob/default changes; `s2_host_tests*.log`).

### 3. Fused path on device -- what ran, numbers, verdicts

Every lever ran on the device at the first attempt: no validate error, no L1/CB clash, no layout or dtype rejection,
no trace-capture failure at the served grid (the only device errors of the pass are the two `96,1056` SDPA variants
below). Hygiene fix committed first: a failed capture is closed (`end_trace_capture` + `release_trace`) in a
`finally` in `MoonViT.capture_trace` and `LocateAnythingPipeline.capture_traces`.

Vision, served grid 24x44 (`probe_vision_grid.py`, 20 timed forwards, one device session; `s3_vision_grid.json`,
`s3c_vision_grid_exact.json`): legacy 47.5 ms, vit_proj PCC 0.99575. `trace == eager` (`torch.equal`) and device merge
== `patch_merge_host` (`torch.equal`) for every fused config. Decomposition of the vision gain:

| config | eager / traced ms | PCC vs golden | vs legacy output | lever |
|---|---:|---:|---|---|
| strict: `linear` + `32,32` approx exp + `LA_FUSED_EXACT_SEQ=0` (legacy padding + mask) | 43.8 / **43.4** | 0.9957524134556435 (= legacy to every digit) | **`torch.equal`, max\|d\| 0.0** | A + B + J: -4.4 ms, exact |
| + C (`LA_FUSED_EXACT_SEQ=1`: 1056 rows, no mask) with the legacy SDPA config | 36.8 / 36.8 | 0.99560 | max\|d\| 4.8, equal 8.8 % (matmul blocking changes at M=1056) | C: -7.0 ms, bf16-rounding |
| + F (`96,352` chunks, exact exp) = `linear` | 30.5 / 30.5 | 0.99583 | max\|d\| 4.0 | F: -6.3 ms (exact exp free, approx exp lower PCC) |
| + D/E (`LA_FUSED_MATMUL=minimal`) | 26.9 / 27.1 | 0.99555 | max\|d\| 5.5 | D/E: -3.6 ms |
| + I (`LA_FUSED_L1=1`) | 24.9 / 24.9 | 0.99555 (= minimal) | -- | I: -2.0 ms (bit-identical to minimal at this grid) |
| H (`LA_FUSED_GELU=tanh`, with minimal) | 26.1 / 26.7 | 0.99464 | -- | H: precision down |
| `96,1056` single K pass (with minimal) | 26.6 / 26.7 | 0.99579 | -- | vision-only OK; fails next to the LLM (below) |

Model metric (`bench_locate_anything.py accuracy=` = full-pipeline logits PCC, gate >= 0.99; `s3b_exact.log`,
`s3d_batch.log`): port golden 26x42 (padded + mask branch): legacy 0.99189; strict 0.991889747176919 (= legacy to every
digit); **`linear` (F) 0.99286**; minimal **0.98997 (below gate)**; linear+tanh 0.99199 (< linear); linear+L1
**0.98952 (below gate; L1 placement changes `ttnn.linear` numerics on the padded graph, so I is NOT bit-identical
there)**; linear+approx exp 0.99070 (< exact exp); minimal+`96,1056` `TT_THROW: Statically allocated circular buffers
on core range [0-0 - 10-9] grow to 1872768 B which is beyond max L1 size of 1572864 B`. Served-grid golden 24x44:
legacy 0.99353, strict 0.99353 (= legacy), linear 0.99227 (box `<607><797>` = the HF reference), minimal 0.99026,
linear+tanh 0.99302.

End-to-end ladder (`probe_pipeline.py`: server-identical build + warm-up, then G1 / G2 / robustness set / G3 / K
exactness / 30 timed demo+`car` requests, fresh process per rung; `s4_ladder.log`, `s4b_ab.log`, `s4_*.json`):

| rung | total ms median (min-max) | gates |
|---|---:|---|
| 0 legacy | 360.9 (349.0-373.8) | G1 ok, G3 0/20, stale ok |
| 1 linear eager (B, C, F, J) | 333.9 (327.9-336.5) | G1 fails (`<607><797>` = HF reference), G3 0/20 |
| 2 + D/E | 336.6 (328.2-340.7) | idem |
| 3 + A vision trace | 326.1 (323.5-334.7) | idem; tokens identical to rung 2 (22/22) |
| 4 + K, N | 314.6 (311.6-341.3) | K `torch.equal` True; tokens identical to rung 3 (22/22) |
| 5 + L prefill trace | 313.9 (310.7-322.3) | tokens identical to rung 4 (22/22); L = -0.6 ms (noise) |
| 6 + M device argmax (all levers, minimal) | **287.1** (286.7-288.0) | decode 201.8 = 44.6 tok/s; 20/22 prompts identical to rung 5 |
| **strict** (A, B, J, K, L, N, M; legacy vision numerics) | **304.9 (304.3-306.0)** | **G1 ok (= legacy string), G2 3/3 token-identical to legacy, G3 0/20, stale ok, K exact; 19/22 prompts identical to legacy** |
| `linear` (strict + C + F) | 291.7 (290.9-292.6) | G1 = HF reference string, not the legacy one; PCC gates pass on both goldens |
| strict + C only | 298.2 (297.6-299.2) | C -6.7 ms, F -6.4 ms end to end |
| linear + `96,1056` | error | `TT_THROW program.cpp:1925` (L1/CB clash next to the resident LLM) |
| linear, `LA_FUSED_PREFILL_TRACE=0` | 291.9 (291.1-292.9) | L worth 0 ms (kept: exact, harmless) |

Peak DRAM after warm-up 6.55-6.56 GB in every rung (legacy 6.55); warm-up 0.6-2.6 s on a warm kernel cache.

**M (on-device greedy argmax), `probe_tie.py` (`s4b_tie_*.json`, `s4c_tie_device_pos.json`):** the persistent decode
trace's `current_pos` and `rot_idx` equal the host position after every step of every prompt (no off-by-one). The 3-query
G2 set is token-identical; the 2-3 prompts that differ between host argmax and device argmax are the degenerate
"no such object" prompts (teaser.jpg|object, referring.png|person, pointing.png|cup) where every variant including the
HF reference emits chaotic whole-image / zero-area boxes; they diverge at a step whose host top-2 margin is 0.125-0.25
(1-4 bf16 ULP at logit ~9) and the device token's host logit is 0.125-0.25 below the host maximum -- i.e. the two decode
traces yield ULP-different logits at near-tie steps (root cause not isolated); exact ties (margin 0.0) resolve identically.
Verdict: keep (-2.8 ms/token), documented as "not bit-identical to host argmax at <= 4-ULP margins".

**HF reference agreement over the 15 prompts (`compare_ref.py`, native reference):** legacy exact 6/15, same box
structure 10/15, mean IoU 0.711; `linear` 7/15 / 13/15 / 0.747 (demo `car` exact); minimal + host argmax 7/15 / 13/15 /
0.936; minimal + device argmax 6/15 / 13/15 / 0.898. The bf16 tower is chaotic at the 1-3 coordinate-unit level; the
three degenerate prompts dominate the IoU spread.

### 4. Per-lever verdicts

| lever | verdict | numbers |
|---|---|---|
| A vision trace | **keep** (default) | exact (`trace == eager`, tokens identical); with B/J -4.4 ms vision |
| B device patch merger | **keep** (default) | `torch.equal` to `patch_merge_host` on device |
| J row-major upload + in-graph tilize | **keep** (default) | exact; ~0 ms alone |
| K device vision->LLM merge | **keep** (default) | `torch.equal` to the host embeds; with N -11.4 ms |
| L prefill trace | **keep** (default) | exact (tokens identical); 0 ms measured gain |
| N one-row first token | **keep** (default) | exact (same bf16 row) |
| M on-device greedy argmax | **keep** (default) | -25 ms per 10-token request; ULP-level flips on near-tie degenerate prompts only, positions verified |
| C exact sequence, no mask | **not default** (knob `LA_FUSED_EXACT_SEQ=1`) | -6.7 ms end to end, PCC gates pass, but the demo string moves off the legacy string (to the HF reference's) -> fails G1 as written |
| F SDPA chunks 96,352 + exact exp | **not default** (knobs `LA_FUSED_SDPA_CHUNKS=96,352 LA_FUSED_SDPA_EXP_APPROX=0`) | -6.4 ms end to end, full PCC 0.99286 (> legacy) on the port golden; same G1 caveat; `96,1056` clashes in L1 |
| D/E minimal matmuls | **dropped** (knob `LA_FUSED_MATMUL=minimal`) | -3.6 ms vision; full PCC 0.98997 < 0.99 on the port golden |
| H tanh GELU | **dropped** (knob) | lower PCC on both goldens |
| I L1 working set | **dropped** (knob) | full PCC 0.98952 < 0.99 with linear on the port golden; not bit-identical there |

### 5. Default and served A/B (shipped image, branch code bind-mounted, 100 warm requests each; `s5_chain.log`,
`s5_served_{legacy,fused}.json`, `s5_serve_{legacy,fused}.log`)

Default flipped: `TT_FUSED` unset = the **strict** fused configuration (`LA_FUSED_MATMUL=linear`, `LA_FUSED_SDPA_CHUNKS=32,32`,
`LA_FUSED_SDPA_EXP_APPROX=1`, `LA_FUSED_EXACT_SEQ=0`, erf GELU, DRAM), `TT_FUSED=0` = legacy; `tt-model.yaml serve.env`
carries `TT_FUSED: "1"` explicitly. Boot on the warm kernel cache: container start -> `Application startup complete`
~17 s (LLM 5.1 s, MoonViT 1.2 s, warm-up 5.5 s incl. both captures; legacy ~14 s).

| | `TT_FUSED=0` (knob) | default (fused, strict) |
|---|---:|---:|
| `/info` fused.status | all False | vision_trace / device_merge / prefill_trace / device_sampling True, bucket 512 |
| smoke_test.py | PASS 356.1 ms | PASS 303.9 ms (`ar_greedy_trace_device_sampling`, 44.5 tok/s) |
| G2 + robustness raw_text | 9/9 == host legacy run | 9/9 == host strict run (3 demo queries == legacy) |
| 100 warm requests, `timing_ms.total` median / min / max | **357.45 / 350.7 / 364.6** | **304.7 / 304.1 / 305.7** |
| stages (median) | vision 47.6, prefill 62.5, decode 231.9 (39.1 tok/s) | vision submit 1.6, prefill (+vision wait) 86.3, decode 202.3 (44.5 tok/s) |
| client wall (median) | 0.40 s | 0.34 s |
| malformed requests | bad base64 -> 400, missing query -> 422, health ok after | same |
| `docker stop -t 60` | "closing mesh device" -> "shutdown complete" -> exit; `tt-smi -s` ok | same |

Both containers (and the PURE shipped image without any mount, `s7_shipped_exit.log`) exit with code 139 after
"Finished server process" -- a pre-existing interpreter-teardown segfault after the device is closed, not introduced
here; `docker ps` empty and `tt-smi -s` OK after every stop.

Final gate in the shipped image (`s6_image_gate2.log`, `s6_image_probe_{default,legacy}.json`, `s6_image_vision_exact.json`):
default 303.95 ms (303.2-304.9), G1 ok, G2 3/3, G3 0/20, K exact; `TT_FUSED=0` 356.8 ms (352.6-360.5), G1 ok, G2 3/3;
vision default vs legacy `torch.equal` (max|d| 0.0), the fast-vision opt-in 30.3 ms / PCC 0.99583. True cold boot of
the fused default (this model's JIT kernel cache moved aside, weights / BFP8 cache warm; `s8_cold_boot.log`): container
start -> smoke PASS **50 s** (LLM 8.3 s, MoonViT 1.2 s, warm-up 32.4 s = run 1 32.1 s of JIT incl. both captures 0.1 s
each, run 2 0.2 s); the 2026-09-12 legacy cold boot was 101 s incl. the one-time 44 s BFP8 conversion and a 39.1 s
warm-up run 1. Cold-boot cache 419 MB; the fuller original cache (1.3 GB, every A/B variant) was restored.

### 6. What remains unverified / open for the owner

* G1's anchor: the gate was written as "equals the legacy string"; the HF reference disagrees with the legacy string on
  the demo image. Keeping C + F (`LA_FUSED_EXACT_SEQ=1 LA_FUSED_SDPA_CHUNKS=96,352 LA_FUSED_SDPA_EXP_APPROX=0`) gives
  291.7 ms (-4.3 % more), PCC gates pass on both goldens and the demo box becomes the reference's -- the owner may flip
  these three defaults; every number is in `VALIDATION.md`.
* M: the two decode traces (host logits vs on-device argmax) are not bit-identical at the ULP level (near-tie flips on
  degenerate prompts); root cause not isolated.
* `la_inputs.find_model_path()` (used by the tests/probes, not the server) fails inside the image without
  `LA_WEIGHTS_DIR` because `snapshot_download(local_files_only=True)` sees the manifest-excluded `assets/*`.
* The card's accuracy numbers moved because the golden had to be regenerated (0.9928 -> 0.9919 full, 0.9911 -> 0.9924
  projector, 0.9922 -> 0.9915 prefill); the device numerics of the legacy path did not change (strict == legacy bit-for-bit).
* Not re-run: the MTP tests (not served), the 1024-token prefill bucket (falls back to the legacy host merge by design),
  multi-turn / batch > 1 (not supported).
