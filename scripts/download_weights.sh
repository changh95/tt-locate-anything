#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Fetch everything tt-locate-anything needs to run:
#   1. The NVIDIA LocateAnything-3B snapshot (MoonViT vision tower + mlp1 projector
#      + Qwen2.5-3B language model + tokenizer + assets) from the Hugging Face Hub.
#   2. A vanilla Qwen2.5-3B HF directory extracted from that snapshot, which
#      tt-metal's `tt_transformers` ModelArgs loads as the LLM backbone.
#
# The torch-CPU goldens that the PCC tests compare against are NOT downloaded —
# generate them locally after this script (see "Next steps" below); they depend on
# your torch/transformers build.
#
# Overridable via env:
#   PYTHON         python to use            (default: python)
#   LA_MODEL_PATH  LocateAnything-3B dir    (default: HF cache, auto-discovered)
#   LA_LLM_DIR     extracted Qwen2.5-3B dir (default: ~/.cache/locate_anything/LA-Qwen2.5-3B)

set -euo pipefail

PYTHON="${PYTHON:-python}"
HF_REPO="nvidia/LocateAnything-3B"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "Python:        $PYTHON"
echo "HF repo:       $HF_REPO"

# --- 1. LocateAnything-3B snapshot (not gated; downloads unauthenticated) ---
echo "... downloading $HF_REPO snapshot (vision + LLM + tokenizer + assets)"
SNAP_DIR="$(
  "$PYTHON" - "$HF_REPO" <<'PY'
import os, sys
from huggingface_hub import snapshot_download
repo = sys.argv[1]
path = os.environ.get("LA_MODEL_PATH") or snapshot_download(repo_id=repo)
print(path)
PY
)"
echo "ok  snapshot at: $SNAP_DIR"
export LA_MODEL_PATH="$SNAP_DIR"

# --- 2. Extract a vanilla Qwen2.5-3B HF dir for tt_transformers ---
# extract_llm_checkpoint.py strips the `language_model.` prefix, synthesizes a
# Qwen2 config from LocateAnything's nested text_config, and copies the tokenizer.
echo "... extracting Qwen2.5-3B LLM checkpoint"
PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}" "$PYTHON" "$REPO_ROOT/locate_anything/reference/extract_llm_checkpoint.py"
LLM_DIR="${LA_LLM_DIR:-$HOME/.cache/locate_anything/LA-Qwen2.5-3B}"

echo
echo "Done."
echo "  LocateAnything-3B snapshot : $SNAP_DIR   (export LA_MODEL_PATH=...)"
echo "  Extracted Qwen2.5-3B LLM   : $LLM_DIR    (export HF_MODEL=...)"
echo
echo "Next steps (generate the torch-CPU goldens used by the PCC tests):"
echo "  # baseline + vision goldens (reference/golden.pt):"
echo "  $PYTHON locate_anything/reference/run_reference.py --in-token-limit 1024"
echo "  # MTP oracle for the experimental MTP test (reference/mtp_oracle.pt):"
echo "  $PYTHON locate_anything/reference/mtp_oracle.py --in-token-limit 1024"
