# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Self-contained input construction for NVIDIA LocateAnything-3B.

Replicates the HF `LocateAnythingImageProcessor` + `LocateAnythingProcessor`
chat-template exactly, but WITHOUT importing the repo's processor module
(which hard-imports cv2/lmdb/decord that are not installed here).

Used by both the torch CPU reference and the tt-nn device port so inputs are
byte-identical. Pure torch + PIL: no torchvision (the serving image pins torch to
tt-metal's version and must not drag a second torch in through torchvision).
"""
import glob
import math
import os

import numpy as np
import torch
from PIL import Image

# The upstream weights repo (also what tt-model exports as HF_MODEL at serve time).
HF_REPO_ID = "nvidia/LocateAnything-3B"

# --- special tokens / ids (from config.json) ---
IMAGE_TOKEN = "<IMG_CONTEXT>"
IMAGE_START_TOKEN = "<img>"
IMAGE_END_TOKEN = "</img>"
IMAGE_TOKEN_INDEX = 151665

_PROMPT = "Locate all the instances that matches the following description: "

# image normalization (preprocessor_config.json)
MEAN = (0.5, 0.5, 0.5)
STD = (0.5, 0.5, 0.5)
PATCH_SIZE = 14
MERGE = (2, 2)
IN_TOKEN_LIMIT = 25600  # preprocessor_config.json


def find_model_path():
    """Locate the LocateAnything-3B snapshot dir.

    Resolution order:
      1. ``LA_MODEL_PATH`` / ``LA_WEIGHTS_DIR`` -- an explicit directory holding the snapshot;
      2. the snapshot already present in the Hugging Face cache (``HF_HOME``-aware; the
         revision comes from ``TT_WEIGHTS_REVISION`` when set, else the cached ``main``),
         resolved offline through ``huggingface_hub``;
      3. the legacy ``~/.cache/huggingface`` glob for dev boxes without ``HF_HOME``.
    """
    for var in ("LA_MODEL_PATH", "LA_WEIGHTS_DIR"):
        env = os.environ.get(var)
        if env and os.path.isdir(env):
            return env
    repo = os.environ.get("LA_HF_REPO", HF_REPO_ID)
    revision = os.environ.get("TT_WEIGHTS_REVISION") or None
    try:
        from huggingface_hub import snapshot_download

        return snapshot_download(repo_id=repo, revision=revision, local_files_only=True)
    except Exception:  # noqa: BLE001 - fall through to the legacy glob
        pass
    pat = os.path.expanduser("~/.cache/huggingface/hub/models--nvidia--LocateAnything-3B/snapshots/*/")
    cands = sorted(glob.glob(pat))
    if not cands:
        raise FileNotFoundError(
            f"No LocateAnything-3B snapshot found (set LA_MODEL_PATH, or download {repo} into the HF cache; "
            f"also tried {pat})"
        )
    return cands[-1].rstrip("/")


def rescaled_size(size_wh, in_token_limit=IN_TOKEN_LIMIT):
    """(W, H) the HF processor's `rescale` produces for an image of size (W, H). Arithmetic only."""
    w, h = int(size_wh[0]), int(size_wh[1])
    p = PATCH_SIZE
    if (w // p) * (h // p) > in_token_limit:
        scale = math.sqrt(in_token_limit / ((w // p) * (h // p)))
        w, h = int(w * scale), int(h * scale)
    pad_h = MERGE[0] * p
    pad_w = MERGE[1] * p
    target_w = math.ceil(w / pad_w) * pad_w
    target_h = math.ceil(h / pad_h) * pad_h
    if target_w // p >= 512 or target_h // p >= 512:
        raise ValueError("Exceed pos emb")
    return target_w, target_h


def _rescale(image: Image.Image, in_token_limit=IN_TOKEN_LIMIT) -> Image.Image:
    """Exact port of LocateAnythingImageProcessor.rescale."""
    w, h = image.size
    p = PATCH_SIZE
    if (w // p) * (h // p) > in_token_limit:
        scale = math.sqrt(in_token_limit / ((w // p) * (h // p)))
        image = image.resize((int(w * scale), int(h * scale)), Image.Resampling.BICUBIC)
    new_w, new_h = image.size
    pad_h = MERGE[0] * p
    pad_w = MERGE[1] * p
    target_w = math.ceil(new_w / pad_w) * pad_w
    target_h = math.ceil(new_h / pad_h) * pad_h
    if target_w != new_w or target_h != new_h:
        image = image.resize((target_w, target_h), Image.Resampling.BICUBIC)
    w, h = image.size
    if w // p >= 512 or h // p >= 512:
        raise ValueError("Exceed pos emb")
    return image


def to_tensor(image: Image.Image) -> torch.Tensor:
    """PIL RGB -> float32 [3,H,W] in [0,1]. Same values as torchvision's `to_tensor`."""
    arr = np.asarray(image.convert("RGB"), dtype=np.uint8)
    t = torch.from_numpy(arr.copy()).permute(2, 0, 1).contiguous()
    return t.to(torch.float32).div_(255.0)


def normalize(t: torch.Tensor, mean=MEAN, std=STD) -> torch.Tensor:
    """Per-channel (x - mean) / std on a [3,H,W] float tensor. Same values as torchvision's `normalize`."""
    mean_t = torch.tensor(mean, dtype=t.dtype).view(-1, 1, 1)
    std_t = torch.tensor(std, dtype=t.dtype).view(-1, 1, 1)
    return (t - mean_t) / std_t


def patchify(t: torch.Tensor):
    """[3,H,W] normalized tensor -> (patches [L,3,14,14] in (h_patch, w_patch) raster order, grid_hw)."""
    C, H, W = t.shape
    p = PATCH_SIZE
    patches = t.reshape(C, H // p, p, W // p, p)
    patches = patches.permute(1, 3, 0, 2, 4).contiguous().view(-1, C, p, p)
    return patches, (H // p, W // p)


def preprocess_image(image: Image.Image, in_token_limit=IN_TOKEN_LIMIT):
    """Returns (pixel_values [L,3,14,14] float, grid_hw (H_patches, W_patches))."""
    image = _rescale(image.convert("RGB"), in_token_limit)
    return patchify(normalize(to_tensor(image)))


def preprocess_image_fixed(image: Image.Image, grid_hw, in_token_limit=IN_TOKEN_LIMIT):
    """Like `preprocess_image`, but the output grid is forced to `grid_hw` (H_patches, W_patches).

    The HF-faithful `_rescale` runs first (so an image whose natural grid already equals
    `grid_hw` -- e.g. the demo image at the validated token limit -- is preprocessed
    byte-identically to `preprocess_image`); anything else is then squashed (bicubic, no
    padding) to exactly grid_hw*14 pixels. The model's boxes are normalized 0..1000 over the
    image it sees, so the squash maps back to the original image without extra bookkeeping.
    """
    h, w = int(grid_hw[0]), int(grid_hw[1])
    target = (w * PATCH_SIZE, h * PATCH_SIZE)
    img = _rescale(image.convert("RGB"), in_token_limit)
    if img.size != target:
        img = img.resize(target, Image.Resampling.BICUBIC)
    patches, got = patchify(normalize(to_tensor(img)))
    assert got == (h, w), f"fixed-grid preprocessing produced {got}, expected {(h, w)}"
    return patches, got


def num_image_tokens(grid_hw):
    """Merged vision-token count for one image."""
    return (grid_hw[0] * grid_hw[1]) // (MERGE[0] * MERGE[1])


def build_chat_text(query: str, n_img_tokens: int) -> str:
    """Replicates LocateAnythingProcessor.py_apply_chat_template + media replacement."""
    text_body = _PROMPT + query + "."
    img_block = f"<image 1>{IMAGE_START_TOKEN}{IMAGE_TOKEN * n_img_tokens}{IMAGE_END_TOKEN}"
    return (
        "<|im_start|>system\n"
        "You are a helpful assistant.\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        f"{img_block}{text_body}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def build_inputs(tokenizer, image: Image.Image, query: str, in_token_limit=IN_TOKEN_LIMIT, grid_hw=None):
    """Full input bundle for both reference and device runs.

    `grid_hw`, when given, forces the vision grid (see `preprocess_image_fixed`); the server
    uses it so MoonViT is built once for a single canonical grid.

    Returns dict: input_ids [1,S], attention_mask [1,S], pixel_values [L,3,14,14],
    image_grid_hws np.int32 [1,2], grid_hw tuple, n_img_tokens int.
    """
    if grid_hw is None:
        pixel_values, grid_hw = preprocess_image(image, in_token_limit)
    else:
        pixel_values, grid_hw = preprocess_image_fixed(image, grid_hw, in_token_limit)
    n_tok = num_image_tokens(grid_hw)
    text = build_chat_text(query, n_tok)
    enc = tokenizer([text], return_tensors="pt")
    input_ids = enc["input_ids"]
    attn = enc.get("attention_mask", torch.ones_like(input_ids))
    n_in_ids = int((input_ids[0] == IMAGE_TOKEN_INDEX).sum().item())
    assert n_in_ids == n_tok, f"image-token mismatch: ids={n_in_ids} expected={n_tok}"
    return {
        "input_ids": input_ids,
        "attention_mask": attn,
        "pixel_values": pixel_values,
        "image_grid_hws": np.array([grid_hw], dtype=np.int32),
        "grid_hw": grid_hw,
        "n_img_tokens": n_tok,
    }


def load_test_image(path=None):
    """Load a deterministic test image. Falls back to a synthetic image."""
    if path is None:
        try:
            mp = find_model_path()
        except FileNotFoundError:
            mp = None
        if mp:
            for name in ("teaser.jpg", "coco_lvis.png", "dense_object_detection.png", "referring.png"):
                cand = os.path.join(mp, "assets", name)
                if os.path.exists(cand) and os.path.getsize(cand) > 0:
                    path = cand
                    break
    if path and os.path.exists(path):
        return Image.open(path).convert("RGB"), path
    return synthetic_image(), "synthetic-448x448"


def synthetic_image(size_wh=(448, 448)):
    """Deterministic synthetic RGB image (two coloured blocks on black), any size."""
    w, h = int(size_wh[0]), int(size_wh[1])
    g = np.zeros((h, w, 3), dtype=np.uint8)
    g[h // 4 : (3 * h) // 4, w // 4 : (3 * w) // 4] = (200, 80, 40)
    g[h // 9 : h // 4 + h // 20, (2 * w) // 3 : (8 * w) // 9] = (40, 160, 220)
    return Image.fromarray(g)
