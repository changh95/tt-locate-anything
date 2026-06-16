# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""TT-NN port of NVIDIA LocateAnything-3B for a single Blackhole p150a.

Subpackages:
  reference/  self-contained torch-CPU reference + input/golden builders
  tt/         the TT-NN device implementation (MoonViT vision, Qwen2.5-3B LLM, MTP)
  tests/      pytest suites (vision PCC, baseline benchmark, MTP, demos)
"""
