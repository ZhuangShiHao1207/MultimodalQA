"""Verify all project dependencies are correctly installed."""
import sys

def check():
    results = []

    try:
        import torch
        results.append(f"[OK] PyTorch {torch.__version__} | CUDA: {torch.cuda.is_available()} | GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
    except Exception as e:
        results.append(f"[FAIL] PyTorch: {e}")

    try:
        import docling
        results.append(f"[OK] Docling {docling.__version__}")
    except Exception as e:
        results.append(f"[FAIL] Docling: {e}")

    try:
        from FlagEmbedding import BGEM3FlagModel
        results.append("[OK] FlagEmbedding (BGE-M3) importable")
    except Exception as e:
        results.append(f"[FAIL] FlagEmbedding: {e}")

    try:
        import zhipuai
        results.append(f"[OK] ZhipuAI SDK {zhipuai.__version__}")
    except Exception as e:
        results.append(f"[FAIL] ZhipuAI: {e}")

    try:
        import langchain
        results.append(f"[OK] LangChain {langchain.__version__}")
    except Exception as e:
        results.append(f"[FAIL] LangChain: {e}")

    try:
        import faiss
        results.append(f"[OK] FAISS (faiss-cpu)")
    except Exception as e:
        results.append(f"[FAIL] FAISS: {e}")

    try:
        import gradio
        results.append(f"[OK] Gradio {gradio.__version__}")
    except Exception as e:
        results.append(f"[FAIL] Gradio: {e}")

    try:
        from dotenv import load_dotenv
        results.append("[OK] python-dotenv")
    except Exception as e:
        results.append(f"[FAIL] python-dotenv: {e}")

    try:
        import sentence_transformers
        results.append(f"[OK] sentence-transformers {sentence_transformers.__version__}")
    except Exception as e:
        results.append(f"[FAIL] sentence-transformers: {e}")

    try:
        import datasets
        results.append(f"[OK] datasets {datasets.__version__}")
    except Exception as e:
        results.append(f"[FAIL] datasets: {e}")

    try:
        import Levenshtein
        results.append("[OK] python-Levenshtein")
    except Exception as e:
        results.append(f"[FAIL] python-Levenshtein: {e}")

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
