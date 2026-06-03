#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Patch FlagEmbedding 1.3.x / 1.4.x for compatibility with transformers >= 4.52.

transformers 4.52+ removed two docstring constants from gemma2:
  - GEMMA2_START_DOCSTRING
  - GEMMA2_INPUTS_DOCSTRING

FlagEmbedding's Gemma reranker imports them unconditionally, causing an
ImportError at startup. This script adds try/except fallbacks in-place.

Run once after `pip install -r requirements.txt`:
    python scripts/patch_flagembedding.py
"""

import sys
import importlib
from pathlib import Path

TARGET_IMPORT = "from transformers.models.gemma2.modeling_gemma2 import Gemma2MLP, repeat_kv, Gemma2Attention, Gemma2DecoderLayer, GEMMA2_START_DOCSTRING"
TARGET_INPUTS = "from transformers.models.gemma2.modeling_gemma2 import GEMMA2_INPUTS_DOCSTRING"

REPLACEMENT = """\
from transformers.models.gemma2.modeling_gemma2 import Gemma2MLP, repeat_kv, Gemma2Attention, Gemma2DecoderLayer
# GEMMA2_START_DOCSTRING and GEMMA2_INPUTS_DOCSTRING were removed in transformers 4.52+; provide fallbacks
try:
    from transformers.models.gemma2.modeling_gemma2 import GEMMA2_START_DOCSTRING
except ImportError:
    GEMMA2_START_DOCSTRING = ""
try:
    from transformers.models.gemma2.modeling_gemma2 import GEMMA2_INPUTS_DOCSTRING
except ImportError:
    GEMMA2_INPUTS_DOCSTRING = ""\
"""


def find_gemma_model_py() -> Path:
    try:
        import FlagEmbedding
        fe_root = Path(FlagEmbedding.__file__).parent
    except ImportError:
        sys.exit("ERROR: FlagEmbedding is not installed in the current environment.")

    target = fe_root / "inference" / "reranker" / "decoder_only" / "models" / "gemma_model.py"
    if not target.exists():
        sys.exit(f"ERROR: Expected file not found:\n  {target}")
    return target


def patch(path: Path) -> None:
    text = path.read_text(encoding="utf-8")

    if "GEMMA2_START_DOCSTRING was removed" in text:
        print(f"Already patched: {path}")
        return

    if TARGET_IMPORT not in text:
        print(f"WARNING: Expected import line not found — file may have changed.\n  {path}")
        print("Skipping automatic patch. Check manually.")
        return

    # Replace the two import lines with the safe fallback block
    patched = text.replace(
        TARGET_IMPORT + "\n" + TARGET_INPUTS,
        REPLACEMENT,
    )

    if patched == text:
        # Try replacing just the first line (in case INPUTS line is absent)
        patched = text.replace(TARGET_IMPORT, REPLACEMENT)

    path.write_text(patched, encoding="utf-8")
    print(f"Patched: {path}")


def verify() -> None:
    # Force reimport to check
    import importlib
    mods_to_remove = [k for k in sys.modules if "FlagEmbedding" in k]
    for m in mods_to_remove:
        del sys.modules[m]
    try:
        from FlagEmbedding import BGEM3FlagModel  # noqa: F401
        print("Verification OK: FlagEmbedding imports cleanly.")
    except ImportError as e:
        print(f"Verification FAILED: {e}")
        sys.exit(1)


if __name__ == "__main__":
    gemma_py = find_gemma_model_py()
    patch(gemma_py)
    verify()
