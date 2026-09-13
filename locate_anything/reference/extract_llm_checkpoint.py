# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Extract a standard Qwen2.5-3B HF checkpoint from the LocateAnything-3B weights.

tt_transformers expects a vanilla HF model dir (config.json model_type=qwen2 +
model.* / lm_head.* weights + tokenizer). We strip the `language_model.` prefix
and synthesize a Qwen2 config from LocateAnything's nested text_config.

Output dir is reusable across runs; skips re-extraction if already present.

Runs both as a module (``from locate_anything.reference import extract_llm_checkpoint``;
the server calls :func:`extract` with explicit paths) and as a plain script
(``python locate_anything/reference/extract_llm_checkpoint.py``, env ``LA_MODEL_PATH`` /
``LA_LLM_DIR`` / ``LA_FORCE_EXTRACT``).
"""
import json
import os
import shutil
import sys

from safetensors import safe_open
from safetensors.torch import save_file

try:
    from locate_anything.reference import la_inputs
except ImportError:  # run as a plain script from anywhere
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import la_inputs  # noqa: E402

DEFAULT_OUT_DIR = os.path.expanduser("~/.cache/locate_anything/LA-Qwen2.5-3B")
TOKENIZER_FILES = [
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "added_tokens.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "generation_config.json",
    "chat_template.json",
]
# What a finished extraction must contain for tt_transformers' ModelArgs to load it.
REQUIRED_OUTPUTS = ("config.json", "model.safetensors", "tokenizer_config.json", "vocab.json", "merges.txt")


def default_out_dir():
    return os.environ.get("LA_LLM_DIR", DEFAULT_OUT_DIR)


def is_extracted(out_dir):
    return all(os.path.isfile(os.path.join(out_dir, f)) for f in REQUIRED_OUTPUTS)


def build_qwen_config(cfg_full):
    """Synthesize a vanilla Qwen2 config.json dict from LocateAnything's nested text_config."""
    tc = cfg_full["text_config"]
    return {
        "architectures": ["Qwen2ForCausalLM"],
        "model_type": "qwen2",
        "hidden_size": tc["hidden_size"],
        "intermediate_size": tc["intermediate_size"],
        "num_hidden_layers": tc["num_hidden_layers"],
        "num_attention_heads": tc["num_attention_heads"],
        "num_key_value_heads": tc["num_key_value_heads"],
        "head_dim": tc["hidden_size"] // tc["num_attention_heads"],
        "max_position_embeddings": tc["max_position_embeddings"],
        "rms_norm_eps": tc["rms_norm_eps"],
        "rope_theta": tc["rope_theta"],
        "vocab_size": tc["vocab_size"],
        "tie_word_embeddings": tc.get("tie_word_embeddings", True),
        "hidden_act": tc.get("hidden_act", "silu"),
        "bos_token_id": tc.get("bos_token_id", 151643),
        "eos_token_id": tc.get("eos_token_id", 151645),
        "torch_dtype": "bfloat16",
        "use_sliding_window": tc.get("use_sliding_window", False),
        "sliding_window": tc.get("sliding_window", None),
        "attention_dropout": 0.0,
        "initializer_range": tc.get("initializer_range", 0.02),
        "transformers_version": "4.53.0",
    }


def _copy_tokenizer_files(mp, out_dir):
    for f in TOKENIZER_FILES:
        src = os.path.join(mp, f)
        if not os.path.exists(src):
            continue
        dst = os.path.join(out_dir, f)
        if f == "tokenizer_config.json":
            # The upstream file carries an `auto_map` pointing at LocateAnything's remote-code
            # processor. AutoTokenizer never needs it (tokenizer_class is the stock
            # Qwen2Tokenizer) but its presence makes transformers ask for trust_remote_code.
            # The extracted dir must load as a plain Qwen2 checkpoint, so drop it.
            cfg = json.load(open(src))
            cfg.pop("auto_map", None)
            with open(dst, "w") as fh:
                json.dump(cfg, fh, indent=2)
        else:
            shutil.copy(src, dst)


def extract(model_path=None, out_dir=None, force=None, log=print):
    """Extract the LLM checkpoint from the snapshot at `model_path` into `out_dir`.

    Idempotent: config.json and the tokenizer files are (re)written every time (cheap);
    model.safetensors is written once, atomically (tmp + rename), and reused afterwards
    unless `force` (or env LA_FORCE_EXTRACT=1). Returns `out_dir`.
    """
    mp = model_path or la_inputs.find_model_path()
    out_dir = out_dir or default_out_dir()
    if force is None:
        force = os.environ.get("LA_FORCE_EXTRACT") == "1"
    log(f"[extract] source: {mp}")
    log(f"[extract] dest:   {out_dir}")
    os.makedirs(out_dir, exist_ok=True)

    cfg_full = json.load(open(os.path.join(mp, "config.json")))
    qwen_cfg = build_qwen_config(cfg_full)
    with open(os.path.join(out_dir, "config.json"), "w") as fh:
        json.dump(qwen_cfg, fh, indent=2)
    log(f"[extract] wrote config.json (vocab={qwen_cfg['vocab_size']}, layers={qwen_cfg['num_hidden_layers']})")

    _copy_tokenizer_files(mp, out_dir)

    out_weights = os.path.join(out_dir, "model.safetensors")
    if os.path.exists(out_weights) and not force:
        log(f"[extract] weights already present, skipping: {out_weights}")
        return out_dir

    # gather language_model.* tensors from all shards
    idx = json.load(open(os.path.join(mp, "model.safetensors.index.json")))
    shards = sorted(set(idx["weight_map"].values()))
    tensors = {}
    for shard in shards:
        path = os.path.join(mp, shard)
        with safe_open(path, framework="pt") as f:
            for k in f.keys():
                if k.startswith("language_model."):
                    new_k = k[len("language_model.") :]  # -> model.* / lm_head.*
                    tensors[new_k] = f.get_tensor(k).contiguous()
    has_lm_head = "lm_head.weight" in tensors
    if not has_lm_head and qwen_cfg["tie_word_embeddings"]:
        tensors["lm_head.weight"] = tensors["model.embed_tokens.weight"].contiguous()
    log(f"[extract] {len(tensors)} LLM tensors (lm_head={'tied/explicit' if 'lm_head.weight' in tensors else 'MISSING'})")
    tmp = out_weights + ".tmp"
    save_file(tensors, tmp, metadata={"format": "pt"})
    os.replace(tmp, out_weights)
    log(f"[extract] saved {out_weights} ({os.path.getsize(out_weights)/1e9:.2f} GB)")
    return out_dir


def main():
    return extract()


if __name__ == "__main__":
    main()
