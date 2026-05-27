#!/bin/bash
# ============================================
# MultimodalQA - 启动脚本
# Usage:
#   ./run.sh --platform win    (Windows, CUDA)
#   ./run.sh --platform mac    (macOS, MPS)
#   ./run.sh --platform cpu    (纯 CPU 模式)
#
# Examples:
#   ./run.sh --platform win test_parsing
#   ./run.sh --platform mac test_indexing
#   ./run.sh --platform win app
# ============================================

set -e

PLATFORM="win"
SCRIPT=""
CONDA_ENV="multimodalQA"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --platform|-p)
            PLATFORM="$2"
            shift 2
            ;;
        --env|-e)
            CONDA_ENV="$2"
            shift 2
            ;;
        *)
            SCRIPT="$1"
            shift
            ;;
    esac
done

# Platform-specific environment
case "$PLATFORM" in
    win|windows)
        export PYTHONIOENCODING=utf-8
        export HF_HUB_DISABLE_SYMLINKS=1
        echo "[Platform: Windows (CUDA)]"
        ;;
    mac|macos)
        export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
        echo "[Platform: macOS (MPS)]"
        ;;
    cpu|linux)
        echo "[Platform: CPU only]"
        ;;
    *)
        echo "Unknown platform: $PLATFORM"
        echo "Usage: ./run.sh --platform [win|mac|cpu] <script>"
        exit 1
        ;;
esac

# Determine what to run
case "$SCRIPT" in
    test_parsing|parsing)
        echo "[Running: test_parsing.py]"
        conda run -n "$CONDA_ENV" python scripts/test_parsing.py
        ;;
    test_indexing|indexing)
        echo "[Running: test_indexing.py]"
        conda run -n "$CONDA_ENV" python scripts/test_indexing.py
        ;;
    eval|evaluate)
        echo "[Running: Evaluation (Multimodal vs Text-only)]"
        conda run -n "$CONDA_ENV" python evaluation/run_eval.py
        ;;
    app|serve)
        echo "[Running: Application server]"
        conda run -n "$CONDA_ENV" python -m uvicorn backend.main:app --reload --port 8000 --reload-exclude "scripts/*" --reload-exclude "docling_output/*" --reload-exclude "evaluation/*" --reload-exclude "frontend/*"
        ;;
    frontend|web)
        echo "[Running: Vue frontend dev server]"
        cd frontend && npm run dev
        ;;
    dev|all)
        echo "[Starting: Backend (port 8000) + Frontend (port 5173)]"
        echo "  Backend: http://localhost:8000/docs"
        echo "  Frontend: http://localhost:5173"
        conda run -n "$CONDA_ENV" python -m uvicorn backend.main:app --reload --port 8000 &
        cd frontend && npm run dev
        ;;
    verify|env)
        echo "[Running: Environment verification]"
        conda run -n "$CONDA_ENV" python scripts/verify_env.py
        ;;
    *)
        if [ -n "$SCRIPT" ]; then
            echo "[Running: $SCRIPT]"
            conda run -n "$CONDA_ENV" python "$SCRIPT"
        else
            echo "Available commands:"
            echo "  ./run.sh --platform win test_parsing"
            echo "  ./run.sh --platform win test_indexing"
            echo "  ./run.sh --platform mac app"
            echo "  ./run.sh --platform win verify"
        fi
        ;;
esac
