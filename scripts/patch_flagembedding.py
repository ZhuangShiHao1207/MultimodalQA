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

import os
import sys
import subprocess
from pathlib import Path

# Avoid OpenMP DLL crash on Windows (pyarrow + torch both load libiomp5md.dll)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

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
    """Locate gemma_model.py WITHOUT importing FlagEmbedding.

    Importing FlagEmbedding triggers the broken import in gemma_model.py
    (which is exactly what we need to patch), causing a crash. Instead we
    use pip to find the package location.
    """
    # Strategy 1: use pip show to get the install location
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "show", "FlagEmbedding"],
            capture_output=True, text=True, timeout=15,
        )
        for line in result.stdout.splitlines():
            if line.startswith("Location:"):
                site_dir = Path(line.split(":", 1)[1].strip())
                target = site_dir / "FlagEmbedding" / "inference" / "reranker" / "decoder_only" / "models" / "gemma_model.py"
                if target.exists():
                    return target
    except Exception:
        pass

    # Strategy 2: search sys.path for the package
    for site_dir in sys.path:
        candidate = Path(site_dir) / "FlagEmbedding" / "inference" / "reranker" / "decoder_only" / "models" / "gemma_model.py"
        if candidate.exists():
            return candidate

    sys.exit("ERROR: Cannot locate FlagEmbedding's gemma_model.py. Is FlagEmbedding installed?")


def patch(path: Path) -> None:
    text = path.read_text(encoding="utf-8")

    if "GEMMA2_START_DOCSTRING" in text and "were removed" in text:
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


def verify(path: Path) -> None:
    """Verify the patch was applied correctly by checking file content.

    We cannot safely import FlagEmbedding here because the very import
    that breaks (gemma_model.py) may still crash if the patch failed,
    and even after patching, other DLL/loading issues on Windows can
    cause segfaults during import. Instead, verify the file content.
    """
    text = path.read_text(encoding="utf-8")

    # Check that the original unguarded import lines are gone.
    # The patched file still contains the symbol names inside indented
    # try/except blocks, so we check: if a "from ... import GEMMA2_..."
    # line is at column 0 (no indent), it's an unguarded top-level import.
    for line in text.splitlines():
        if not line.startswith("from "):
            continue
        if "GEMMA2_START_DOCSTRING" in line and "Gemma2DecoderLayer" in line:
            print("Verification FAILED: unguarded GEMMA2_START_DOCSTRING import still present.")
            sys.exit(1)
        if "GEMMA2_INPUTS_DOCSTRING" in line and line.startswith("from "):
            print("Verification FAILED: unguarded GEMMA2_INPUTS_DOCSTRING import still present.")
            sys.exit(1)

    # Check that our fallback block exists
    if "GEMMA2_START_DOCSTRING" in text and "were removed" in text and "try:" in text:
        print("Verification OK: patch applied correctly.")
    else:
        print("Verification WARNING: could not confirm patch markers in file.")
        print("Please verify manually that try/except fallbacks are in place.")


if __name__ == "__main__":
    gemma_py = find_gemma_model_py()
    patch(gemma_py)
    verify(gemma_py)
