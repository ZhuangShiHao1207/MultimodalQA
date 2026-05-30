"""Verify all project dependencies are correctly installed."""
import sys


def check():
    results = []

    # ── Core: PyTorch ────────────────────────────────────────────
    try:
        import torch
        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else (
            "MPS" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "CPU"
        )
        results.append(f"[OK] PyTorch {torch.__version__} | device: {gpu}")
    except Exception as e:
        results.append(f"[FAIL] PyTorch: {e}")

    # ── Document Parsing ────────────────────────────────────────
    try:
        import docling
        # Docling doesn't always expose __version__; fall back to importlib.metadata
        try:
            v = docling.__version__
        except AttributeError:
            import importlib.metadata
            v = importlib.metadata.version("docling")
        results.append(f"[OK] Docling {v}")
    except Exception as e:
        results.append(f"[FAIL] Docling: {e}")

    # ── Embedding Model ─────────────────────────────────────────
    try:
        from FlagEmbedding import BGEM3FlagModel  # noqa: F401
        results.append("[OK] FlagEmbedding (BGE-M3) importable")
    except Exception as e:
        results.append(f"[FAIL] FlagEmbedding: {e}")

    # ── transformers (used directly for CVE-2025-32434 monkey-patch) ─
    try:
        import transformers
        results.append(f"[OK] transformers {transformers.__version__}")
    except Exception as e:
        results.append(f"[FAIL] transformers: {e}")

    # ── Vector Store ────────────────────────────────────────────
    try:
        import chromadb
        results.append(f"[OK] chromadb {chromadb.__version__}")
    except Exception as e:
        results.append(f"[FAIL] chromadb: {e}")

    # ── LLM API ─────────────────────────────────────────────────
    try:
        import zhipuai
        results.append(f"[OK] ZhipuAI SDK {zhipuai.__version__}")
    except Exception as e:
        results.append(f"[FAIL] ZhipuAI: {e}")

    # ── Backend (FastAPI + Uvicorn + Pydantic + multipart) ──────
    try:
        import fastapi
        results.append(f"[OK] FastAPI {fastapi.__version__}")
    except Exception as e:
        results.append(f"[FAIL] FastAPI: {e}")

    try:
        import uvicorn
        results.append(f"[OK] uvicorn {uvicorn.__version__}")
    except Exception as e:
        results.append(f"[FAIL] uvicorn: {e}")

    try:
        import pydantic
        results.append(f"[OK] pydantic {pydantic.VERSION}")
    except Exception as e:
        results.append(f"[FAIL] pydantic: {e}")

    try:
        import multipart  # python-multipart imports as `multipart`
        results.append("[OK] python-multipart importable")
    except Exception as e:
        results.append(f"[FAIL] python-multipart: {e}")

    # ── Utilities ───────────────────────────────────────────────
    try:
        import numpy as np
        results.append(f"[OK] numpy {np.__version__}")
    except Exception as e:
        results.append(f"[FAIL] numpy: {e}")

    try:
        from dotenv import load_dotenv  # noqa: F401
        results.append("[OK] python-dotenv")
    except Exception as e:
        results.append(f"[FAIL] python-dotenv: {e}")

    # ── Evaluation ──────────────────────────────────────────────
    try:
        import Levenshtein  # noqa: F401
        results.append("[OK] python-Levenshtein")
    except Exception as e:
        results.append(f"[FAIL] python-Levenshtein: {e}")

    # ── Print results ───────────────────────────────────────────
    print("\n" + "=" * 55)
    print("  MultimodalQA Dependency Verification")
    print("=" * 55)

    failed = 0
    for r in results:
        print(f"  {r}")
        if "[FAIL]" in r:
            failed += 1

    print("=" * 55)
    if failed == 0:
        print(f"  All {len(results)} dependencies verified successfully!")
    else:
        print(f"  {failed}/{len(results)} dependencies FAILED")
    print("=" * 55 + "\n")

    return failed == 0


if __name__ == "__main__":
    success = check()
    sys.exit(0 if success else 1)
