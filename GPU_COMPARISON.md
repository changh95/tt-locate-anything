# locate-anything-3b-p150 — Blackhole p150a vs RTX 5090 (same host, same weights, same input)

Date 2026-09-14. Facts only; every GPU number below was measured in this pass, every p150a number is copied
(with its source line) from the validation / publish reports. The p150a was NOT touched.

## What was run

| | |
|---|---|
| Model | NVIDIA LocateAnything-3B (MoonViT-SO-400M vision tower, 27 blocks, + 2-layer `mlp1` projector + Qwen2.5-3B-Instruct LLM with the detection vocabulary, 36 layers, 152 681-token head; 3.518 B parameters). The port has no standalone torch reference, so — as the brief says — the **HF remote code shipped in the weights snapshot** was run (`modeling_locateanything.LocateAnythingForConditionalGeneration` = `MoonVitPretrainedModel` + `mlp1` + the snapshot's own `Qwen2ForCausalLM`), built exactly as the port's `code/locate_anything/reference/run_reference.py` builds the golden model: `AutoModel.from_pretrained(..., trust_remote_code=True)` with `_attn_implementation="sdpa"` on the top / text / vision configs (magi-attention and flash_attn are not installed; the code falls back to SDPA). Greedy loop = the model's own `generate(generation_mode="slow", temperature=0, use_cache=True)` semantics (argmax, KV cache, stop on `<|im_end|>`), re-implemented with a `DynamicCache` so the three stages can be timed; the model's own `generate()` was also run and produces the same tokens |
| Weights | `nvidia/LocateAnything-3B` @ `c32291ca5e996f5a7a485845b4f57a233936bba0` (tt-model.yaml `weights.revision` = `serve.env.TT_WEIGHTS_REVISION`), the cached snapshot `~/.cache/huggingface/hub/models--nvidia--LocateAnything-3B/snapshots/c32291ca…/` (2 safetensors shards, bf16, 7.0 GB). Loaded through a flat copy `logs/gpu-vs-p150/locate-anything-3b/snapshot_flat/` (.py/.json copied, the shards **symlinked** to the same blobs — nothing re-downloaded) because transformers resolves the snapshot's symlinked modeling file to `blobs/<sha>` and then cannot find its sibling modules (`generate_utils.py`, …) |
| Input | `media/demo_input.png` (1920x1080 RGBA -> RGB) + query `car` — the card's quickstart / smoke-test request. Preprocessing = the served path's own `code/locate_anything/reference/la_inputs.py::build_inputs(tok, img, "car", in_token_limit=1024, grid_hw=(24, 44))` (what `pipeline.run` calls): HF-faithful bicubic rescale then squash to the canonical **616x336** (24x44 patches of 14 px = 1056 patches, `pixel_values [1056,3,14,14]` fp32 2.5 MB, 264 merged image tokens), chat template -> **303 prompt tokens**; batch 1; `max_new_tokens=128` (server default), greedy, stop on EOS |
| Output | `<ref>car</ref><box><282><414><607><797></box><|im_end|>` — 10 tokens (1 from prefill + 9 decode steps, EOS at step 9) in **every** GPU precision and on the CPU fp32 reference = the HF reference string recorded in `DEVICE_VALIDATION.md` §0. The p150a serves `<282><414><606><794>` (IoU 0.989 against this, `DEVICE_VALIDATION.md` §0 / `VALIDATION_SUMMARY.md:41`) |
| GPU | NVIDIA GeForce RTX 5090 (sm_120), driver 580.126.18, power limit 600 W, 32607 MiB; idle 29.0 W (run 1; 28.7 W after the last process exited) |
| venv | **`/home/deepgadget/experiments/tt-models/.venv-gpu/la`** (created in this stage, see "Environment fixes") — Python 3.12.13, torch 2.11.0+cu128, CUDA 12.8, cuDNN 9.19, **transformers 4.53.0**, tokenizers 0.21.4, peft 0.17.1, accelerate 1.15.0, numpy 1.26.4, safetensors 0.8.0, huggingface_hub 0.36.2, pillow 12.3.0, triton 3.6.0 |
| Host CPU (CPU reference / host stages) | AMD Ryzen 9 7950X 16-core (32 threads), torch 32 intra-op threads for the CPU reference. Shared with other agents' CPU jobs during this pass (load average 1–9, see "Repeatability") |
| Scripts | `logs/gpu-vs-p150/locate-anything-3b/bench_la.py` (uses `logs/gpu-vs-p150/bench_common.py`), `bench_la_compile.py` (torch.compile stage), `probe_decode_sync.py` (decode-loop probe); logs `run2.log` (reported run), `run_full.log` (run 1), `run_compile.log`, `probe_decode_sync.log`, `smoke.log`; raw JSON `reports/gpu-vs-p150/locate-anything-3b.json` (= `logs/gpu-vs-p150/locate-anything-3b/result.json`; run 1 kept as `run1_with_compile.json`); CPU reference tensors `logs/gpu-vs-p150/locate-anything-3b/cpu_ref_fp32.pt` (vit_proj, full prefill logits, 9 decode-step logits, ids) |
| Commands | `.venv-gpu/la/bin/python bench_la.py --iters 50 --warmup 10 --hf-generate-iters 20 --out logs/gpu-vs-p150/locate-anything-3b/run2.json` (run 1: same, default `--out`), then `.venv-gpu/la/bin/python bench_la_compile.py --iters 50 --warmup 10`, `.venv-gpu/la/bin/python probe_decode_sync.py` |
| Loop | per precision and variant: 10 warm-ups + 50 timed iterations, `torch.cuda.synchronize()` before/after each iteration and between the three stages; wall-clock (perf_counter) is the primary number, CUDA-event time recorded alongside (within 0.02 ms of wall); power = `nvidia-smi` sampled every 200 ms during the timed loop (mean over the loop; each loop lasts 8–11 s) |
| p150a source | `reports/gpu-vs-p150/p150_numbers.json` -> `models/locate-anything-3b-p150/DEVICE_VALIDATION.md:378` (served A/B, 100 warm requests, fused default: stage medians **vision submit 1.6 · prefill incl. vision wait 86.3 · decode 202.3 ms = 44.5 tok/s**, total 304.7), `reports/megakernel/PUBLISH_SUMMARY.md:17` (Hub `tt serve`: **total 304.8 ms**, 44.5 tok/s), `reports/megakernel/VALIDATION_SUMMARY.md:41` (legacy 357 ms, PCC 0.9919) |

Timing definitions (matched to the p150a `timing_ms` keys):

- **incl_h2d** = `input_ids` / `attention_mask` / `pixel_values` / `grid` `.to("cuda")` (pageable, 2.5 MB) + **vision** (`extract_feature` + `mlp1`) + **prefill** (303 tokens, `DynamicCache`, argmax of the last logits) + **decode** (greedy steps with a `.item()` token readback per step for the EOS check, as the p150a reads one token id back per step). Compare with the p150a stage-median sum **vision 1.6 + prefill 86.3 + decode 202.3 = 290.2 ms** (there is no single device key; the vision trace is submitted non-blocking and its wait sits inside prefill, so the like-for-like sub-rows are **vision+prefill 87.9 ms** and **decode 202.3 ms**; uploads and token readbacks are inside the p150a numbers).
- **excl_h2d** = inputs resident on the GPU, exactly 9 decode steps, the argmax fed back on the device, ids read back after the timed region. The reference's own forward keeps two host syncs (`input_ids[0][-1].item()` in the prefill mask builder, `nonzero()` in `find_prefix_seq_length_by_pe`); they are part of the reference and were not removed.
- **served-like** = incl_h2d + the same host work `pipeline.run` does inside the p150a `timing_ms.total`: `build_inputs` on the already-decoded PIL image (rescale/squash/normalise/patchify + tokenisation, **13.1 ms** median here) + `tokenizer.decode` + box regex (0.03 ms). The base64/PNG decode (23.6 ms for this file) is outside `timing_ms.total` on the p150a and is excluded here too. Compare with **304.8 ms**. Consistency check: the p150a's own total − stage sum = 304.8 − 290.2 = 14.6 ms of host work, vs 13.1 ms measured here on the same CPU.

## Correctness check (GPU vs CPU fp32 reference)

CPU fp32 reference (same code path, 32 threads): vision 7.03 s + prefill 10.6 s + decode 10.0 s = 27.7 s; output
`<ref>car</ref><box><282><414><607><797></box><|im_end|>` -> `car` box `[541.4, 447.1, 1165.4, 860.8]` px. GPU vs this
reference (PCC = Pearson correlation, flattened; decode PCC = min over the 9 steps, all on identical token prefixes):

| GPU precision | vit_proj `[264,2048]` | prefill logits `[303,152681]` (all / last token) | max abs diff last logits | decode logits (min of 9) | tokens |
|---|---:|---:|---:|---:|---|
| fp32 strict (no TF32) | **1.000000** | **1.000000 / 1.000000** | 2.6e-4 | **1.000000** | identical (10) |
| TF32 | 0.999965 | 0.999788 / 0.999917 | 0.19 | 0.999869 | identical |
| bf16 autocast | 0.997806 | 0.990418 / 0.993623 | 1.58 | 0.997063 | identical |
| fp16 autocast | 0.999903 | 0.999467 / 0.999912 | 0.18 | 0.999855 | identical |
| bf16 native weights (`model.to(bfloat16)`) | 0.997474 | 0.988848 / 0.997927 | 0.97 | 0.997012 | identical |
| bf16 native + torch.compile | 0.997652 | 0.988730 / 0.999573 | — | 0.999504 | identical |

fp32 strict PCC > 0.999 on every tensor (1.000000), so the timings are of the right model. For reference, the p150a's
full-pipeline logits PCC against the port's goldens is 0.9919 (`VALIDATION_SUMMARY.md:41`); the GPU's bf16 rows land in the
same region (0.989–0.998), fp16 autocast and TF32 are 10x closer to fp32.

## GPU latency (RTX 5090, batch 1, run 2 — reported; 50 timed iterations after 10 warm-ups)

Stage medians are from the incl_h2d loop (H2D is 0.2–0.3 ms and is listed separately). All rows generated 10 tokens (9 decode steps) in every iteration.

| precision | first call ms | **incl_h2d** median / min / p90 ms | excl_h2d median / min / p90 ms | H2D | vision | prefill | **vision+prefill** | **decode (9 steps)** | **tok/s** | power mean W (max) | GPU util | peak alloc / reserved MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fp32 strict (`allow_tf32=False`, 'highest') | 380 | **215.9** / 215.6 / 219.2 | 215.5 / 215.3 / 219.1 | 0.2 | 41.3 | 50.7 | 92.0 | 123.5 | 72.9 | 402 (411) | 93 % | 14 900 / 15 604 |
| fp32 + TF32 (matmul + cudnn, 'high') | 192 | **185.7** / 185.5 / 189.3 | 185.3 / 185.1 / 188.5 | 0.2 | 29.4 | 32.5 | 61.9 | 123.5 | 72.9 | 331 (337) | 92 % | 14 900 / 15 604 |
| bf16 autocast (fp32 weights) | 250 | **218.4** / 218.1 / 222.0 | 218.0 / 217.8 / 221.9 | 0.2 | 29.1 | 31.5 | 60.7 | 157.3 | 57.2 | 351 (353) | 91 % | 15 482 / 16 202 |
| fp16 autocast (fp32 weights) | 254 | **214.7** / 214.4 / 218.5 | 215.1 / 214.2 / 223.3 | 0.2 | 28.9 | 28.4 | 57.2 | 157.1 | 57.3 | 361 (363) | 91 % | 15 482 / 16 202 |
| bf16 native weights (checkpoint dtype; beyond the brief's list) | 211 | **158.4** / 155.2 / 163.5 | 154.6 / 153.8 / 157.5 | 0.3 | 27.5 | 21.5 | 49.1 | 108.7 | 82.8 (85.0 excl) | 312 (324) | 74 % | 7 675 / 8 388 |
| bf16 native + **torch.compile** (vision tower + each of the 36 decoder layers, `dynamic=True`, mode default; compile 84.5 s; run between run 1 and run 2) | 146 | **136.5** / 135.2 / 144.3 | 136.4 / 135.1 / 143.7 | 0.2 | 21.8 | 18.1 | 39.9 | 96.3 | 93.4 | 332 | 67 % | 7 778 |
| the model's own `generate(generation_mode="slow", temperature=0)`, bf16 native, inputs resident, 20 iters | — | 152.0 / 151.6 / 154.0 | — | — | — | — | — | — | — | 326 | — | — |

Other measurements: model load safetensors -> CPU fp32 2.2–8.6 s (page-cache dependent), CPU fp32 -> cuda 1.7 s
(14.6 GiB allocated after load; 7.4 GiB after the bf16 cast); GPU idle 29.0 W; PCIe H2D of the 2.5 MB input 0.2 ms.

Observations that matter for reading the table (measured, not inferred):

- **Decode is host-bound on the GPU.** fp32 strict and TF32 give the same 123.5 ms for 9 steps although TF32 halves vision+prefill; bf16 weights (half the bytes to stream) only reach 108.7 ms; excl_h2d ≈ incl_h2d in every row; `probe_decode_sync.py` (bf16, 9 steps, 2x20 passes): per-step `.item()` readback 105.6 / 105.8 ms, no readback 105.2 / 105.5 ms, pure device loop 104.8 / 104.8 ms — the readback costs < 1 ms. ~12 ms per step is Python + kernel-launch overhead of the snapshot's Qwen2 code (36 layers, the 4-D mask rebuilt per step, `DynamicCache` concatenation). torch.compile of the layers takes it to 10.7 ms/step (93 tok/s); CUDA graphs were not attempted (the remote code's data-dependent Python — `.item()`, `nonzero()`, per-batch mask loops — would break them). The GPU is far from its bandwidth limit here (7 GB of bf16 weights per step would be ~4 ms at 1.8 TB/s).
- **bf16 / fp16 autocast over fp32 weights is slower than plain fp32 for decode** (157 vs 123.5 ms): autocast re-casts the fp32 weights to the low-precision dtype inside every step (extra cast kernels and launches on an already launch-bound loop) and only pays off in the larger vision/prefill matmuls (60 ms vs 92 ms). The checkpoint's native bf16 (`model.to(bfloat16)`, what `run_reference.py` loads as well) is the realistic GPU deployment precision: **158.4 ms** incl_h2d, 82.8 tok/s, 7.7 GiB.
- fp16 autocast is numerically fine (PCC 0.9999 prefill, 0.99986 decode, same tokens) but not faster than bf16 autocast.

## Comparison with the p150a (same weights, same 303-token prompt, same 10-token output)

p150a precision: bf16 activations, bf8/bf16 weights (tt-transformers Qwen2.5-VL path, `LA_PREC=accuracy`); its power was **not** measured in any pass, so no power/efficiency claim is made. Ratio = p150a ms / GPU ms (> 1 = GPU faster).

| GPU precision | GPU incl_h2d (H2D + vision + prefill + decode) | p150a vision+prefill+decode 290.2 ms -> ratio | GPU vision+prefill | p150a 87.9 ms -> ratio | GPU decode 9 steps (tok/s) | p150a 202.3 ms (44.5 tok/s) -> ratio | GPU served-like (incl + 13.1 ms host pre/post) | p150a served total 304.8 ms -> ratio | GPU power |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fp32 strict | 215.9 | **1.34x** | 92.0 | **0.96x** (p150a faster) | 123.5 (72.9) | 1.64x | 229.0 | 1.33x | 402 W |
| fp32 + TF32 | 185.7 | **1.56x** | 61.9 | 1.42x | 123.5 (72.9) | 1.64x | 198.9 | 1.53x | 331 W |
| bf16 autocast | 218.4 | 1.33x | 60.7 | 1.45x | 157.3 (57.2) | 1.29x | 231.5 | 1.32x | 351 W |
| fp16 autocast | 214.7 | 1.35x | 57.2 | 1.54x | 157.1 (57.3) | 1.29x | 227.8 | 1.34x | 361 W |
| bf16 native weights | 158.4 | **1.83x** | 49.1 | 1.79x | 108.7 (82.8) | **1.86x** | 171.6 | **1.78x** | 312 W |
| bf16 native + torch.compile | 136.5 | **2.13x** | 39.9 | 2.20x | 96.3 (93.4) | 2.10x | 149.7 | 2.04x | 332 W |

Reading: against the reference run as-is in the checkpoint's own dtype (bf16) the RTX 5090 is 1.8x faster end to end
(1.8x on vision+prefill, 1.9x on decode: 83 vs 44.5 tok/s); with the decoder layers compiled 2.1x. In strict fp32 the GPU
is 1.3x faster overall and the p150a's fused vision+prefill (87.9 ms) is slightly faster than the GPU's fp32 vision+prefill
(92.0 ms). Both sides are limited by per-step overhead in decode rather than by memory bandwidth (p150a 22.5 ms/step
with on-device argmax; GPU 12.1 ms/step eager bf16, 10.7 ms/step compiled).

## Repeatability

Run 1 (`run_full.log`, 20:56–20:59, host load average 3.6–9.3 from other agents' CPU jobs) vs run 2 (reported, `run2.log`,
21:02–21:05, load average 1.1–2.4), medians incl_h2d ms: fp32 strict 215.9 / 215.9, TF32 186.3 / 185.7, bf16 autocast 218.9 /
218.4, fp16 autocast 217.0 / 214.7, bf16 native 161.2 / 158.4; `generate()` 150.9 / 152.0. Every median agrees within 2 %.
Because decode is host-bound, heavy host load does move it: a 3-iteration smoke run under load average ~9 measured
fp32 decode 206 ms (vs 123.5 ms in both full runs) — the full runs were taken with the host quieter and agree with each other.

## Environment fixes (inside this benchmark's venvs only; the p150a and the repo were not touched)

1. `.venv-gpu/main` (transformers 5.12.1) cannot load this remote code: `transformers.dynamic_module_utils` resolves the snapshot's symlinked modeling file to `blobs/<sha>` and then searches for the relative imports next to the blob (`FileNotFoundError: …/blobs/generate_utils.py`); with a flat copy it then fails on the remote `LocateAnythingPreTrainedModel._check_and_adjust_attn_implementation()` (4.x signature, `TypeError: unexpected keyword argument 'allow_all_kernels'`), and the snapshot's Qwen2 code relies on 4.x legacy-cache tuple semantics. `DEVICE_VALIDATION.md` §0 records the same ("the gbp-tt venv's transformers 5.12.1 rejects the snapshot remote code") and produced the port's goldens with transformers 4.53.0. **Fix (1 attempt): a dedicated venv `.venv-gpu/la`** = `uv venv --python 3.12` + `torch==2.11.0` (cu128 index), `transformers==4.53.0`, `tokenizers<0.22`, `peft<0.18` (+ `accelerate`; the remote code imports `peft`), `pillow`, `numpy<2`, `safetensors`, `huggingface_hub<1`, `einops`, `scipy`, `psutil`. During the first diagnosis `peft`+`accelerate` were also added to `main` (additive, unused afterwards); nothing else was installed into `main` / `pi05` / `xvla`.
2. Flat snapshot copy (see Weights) — no weight bytes duplicated or downloaded.
3. MoonViT's `Rope2DPosEmb` caches its complex `freqs_cis` as a plain attribute on the device of the first call; after the CPU reference it is reset so the GPU run recomputes it on cuda (otherwise `cuda:0 vs cpu` in `apply_rope`).
4. Under autocast `mlp1` returns bf16/fp16 while the LLM embedding `index_put` needs fp32: the projector output is cast to the embedding dtype before the LLM, exactly what `run_reference.py` does (`vit_proj_for_llm`); a no-op in fp32 / bf16 native.

## Not measured / caveats

- p150a power (no efficiency claim); p150a numbers were not re-measured — they are the 2026-09-13/14 served medians cited above.
- The p150a squashes the demo image onto the same 24x44 grid, so both sides see 264 image tokens and 303 prompt tokens; a larger `LA_IN_TOKEN_LIMIT` (upstream default 25600) would change both.
- torch.compile with CUDA graphs (`mode="reduce-overhead"`) was not attempted for the reasons above; the compiled row therefore shows the Inductor-fusion gain only.
- GPU tensors are pageable-host uploads (what a server's preprocess produces); pinned memory was not used (H2D is 0.2 ms either way).
- The served-like GPU number adds only the host work that sits inside the p150a `timing_ms.total`; neither side includes HTTP, base64 or PNG decode.
