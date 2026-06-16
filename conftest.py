# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Pytest bootstrap for tt-locate-anything.

Unlike a pure-ttnn model, LocateAnything's LLM backbone *reuses* tt-metal's
``models.tt_transformers`` (the stock Qwen2.5 Transformer / Generator / MLP) and
``models.demos.qwen25_vl`` (vision-token merge + prefill prep). Its tests also use
tt-metal's device fixtures (``mesh_device``, ``device_params``, ``reset_seeds`` …)
together with the matching pytest hooks (``pytest_addoption``,
``pytest_generate_tests``) and the per-test device cleanup.

Rather than reimplement that machinery, this conftest:

  1. puts the repo root on ``sys.path`` so ``import locate_anything`` works, and
  2. loads tt-metal's *own* root ``conftest.py`` into this module's namespace, so
     every fixture and hook it defines becomes active here exactly as if the tests
     were run from inside the tt-metal tree.

``exec`` is used rather than ``import`` because pytest reserves the module name
``conftest`` and disallows listing it in ``pytest_plugins``.

Requires ``TT_METAL_HOME`` to point at a built tt-metal checkout whose Python
bindings (``$TT_METAL_HOME``, ``$TT_METAL_HOME/ttnn``) are importable. See README.
"""

from __future__ import annotations

import os
import sys

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)  # make `import locate_anything` resolve

_TT_METAL_HOME = os.environ.get("TT_METAL_HOME")
assert _TT_METAL_HOME and os.path.isdir(_TT_METAL_HOME), (
    "Set TT_METAL_HOME to a built tt-metal checkout. tt-locate-anything reuses "
    "tt-metal's tt_transformers / qwen25_vl model libraries and its pytest device "
    "fixtures; it does not vendor the tt-metal monorepo. See the README."
)

# Ensure tt-metal's Python packages are importable (ttnn bindings + model libs).
for _p in (_TT_METAL_HOME, os.path.join(_TT_METAL_HOME, "ttnn"), os.path.join(_TT_METAL_HOME, "tools")):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

# Pull tt-metal's root conftest (device fixtures + hooks + cleanup) into our
# namespace so pytest discovers them as if defined here.
_TT_CONFTEST = os.path.join(_TT_METAL_HOME, "conftest.py")
assert os.path.isfile(_TT_CONFTEST), f"tt-metal conftest.py not found at {_TT_CONFTEST}"
with open(_TT_CONFTEST) as _f:
    exec(compile(_f.read(), _TT_CONFTEST, "exec"), globals())
