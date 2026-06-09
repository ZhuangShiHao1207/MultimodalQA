"""
evaluation/ablation_images.py

Ablation study: effect of max_images on QA accuracy.
Sweeps max_images in [0, 1, 2, 3, 5] and records accuracy / ANLS
for all questions, visual-only, and non-visual subsets.

Run (from project root, after building indexes):
    python -m evaluation.ablation_images

Output:
    evaluation/ablation_results.json   -- full per-question breakdown
    evaluation/figures/fig6_ablation_images.png  -- line plot
"""

import sys
import os
import json
import logging
import hashlib
import time
from pathlib import Path

if sys.platform == "win32":
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")

import transformers.utils.import_utils
if hasattr(transformers.utils.import_utils, "check_torch_load_is_safe"):
    transformers.utils.import_utils.check_torch_load_is_safe = lambda: None
import transformers.modeling_utils
if hasattr(transformers.modeling_utils, "check_torch_load_is_safe"):
    transformers.modeling_utils.check_torch_load_is_safe = lambda: None

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.indexing import BGEEmbedder, VectorStore
from src.retrieval import MultiVectorRetriever
from src.generation import GroundedGenerator
from evaluation.metrics import anls_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR   = project_root / "data"
CHROMA_DIR = project_root / "backend" / "chroma_data"
QA_PATH    = project_root / "evaluation" / "datasets" / "self_built_qa.json"
OUT_PATH   = project_root / "evaluation" / "ablation_results.json"
FIG_DIR    = project_root / "evaluation" / "figures"

MAX_IMAGES_SWEEP = [0, 1, 2, 3, 5]
TOP_K = 5


def get_doc_id(filename: str) -> str:
    return hashlib.md5(filename.encode()).hexdigest()[:8]


def run_ablation():
    with open(QA_PATH, encoding="utf-8") as f:
        qa_dataset = json.load(f)
    logger.info(f"Loaded {len(qa_dataset)} questions")

    # Load collections
    documents = {}
    for qa in qa_dataset:
        fname = qa["document"]
        if fname not in documents:
            doc_id = get_doc_id(fname)
            try:
                store = VectorStore(
                    collection_name=f"doc_{doc_id}",
                    persist_dir=str(CHROMA_DIR),
                )
                documents[fname] = {"doc_id": doc_id, "store": store}
                logger.info(f"Loaded '{fname}' (vectors={store.size})")
            except Exception as e:
                logger.error(f"Failed to load {fname}: {e}")

    if not documents:
        logger.error("No indexes found. Run backend first.")
        return

    embedder  = BGEEmbedder(model_name="BAAI/bge-m3", device="auto", use_fp16=True)
    generator = GroundedGenerator(model="glm-4.6v", max_tokens=2048)

    # For each question, collect answers at EACH max_images value
    # We retrieve context once per (question, max_images) value
    # To save API cost: reuse the text-only answer (max_images=0) cached result
    # Structure: results[qid][max_images] = {answer, anls, correct}

    all_results = {}

    for i, qa in enumerate(qa_dataset):
        qid   = qa["id"]
        fname = qa["document"]
        if fname not in documents:
            logger.warning(f"Skip {qid}: doc not loaded")
            continue

        store     = documents[fname]["store"]
        retriever = MultiVectorRetriever(
            embedder=embedder, vector_store=store,
            top_k=TOP_K, score_threshold=0.3
        )

        all_results[qid] = {
            "question":       qa["question"],
            "type":           qa["type"],
            "requires_visual": qa["requires_visual"],
            "difficulty":     qa.get("difficulty", "unknown"),
            "gold_answers":   qa["gold_answers"],
            "by_max_images":  {},
        }

        logger.info(f"[{i+1}/{len(qa_dataset)}] {qid} ({qa['type']}): {qa['question'][:60]}")

        for max_images in MAX_IMAGES_SWEEP:
            try:
                t0  = time.time()
                ctx = retriever.retrieve_with_context(qa["question"], max_images=max_images)
                res = generator.generate(qa["question"], ctx)
                lat = round(time.time() - t0, 2)

                answer = res.get("answer", "")
                score  = anls_score(answer, qa["gold_answers"])

                all_results[qid]["by_max_images"][str(max_images)] = {
                    "answer":     answer,
                    "anls":       score,
                    "correct":    score >= 0.5,
                    "latency_sec": lat,
                    "num_images_sent": len(ctx.get("image_contexts", [])),
                }
                logger.info(f"  max_images={max_images}: ANLS={score:.2f}  {lat:.1f}s")

            except Exception as e:
                logger.error(f"  max_images={max_images} error: {e}")
                all_results[qid]["by_max_images"][str(max_images)] = {
                    "answer": "", "anls": 0.0, "correct": False, "latency_sec": -1,
                    "num_images_sent": 0,
                }

    # ── Aggregate ──────────────────────────────────────────────────────────────
    questions = list(all_results.values())
    visual     = [q for q in questions if q["requires_visual"]]
    non_visual = [q for q in questions if not q["requires_visual"]]

    summary_by_k = {}
    for k in MAX_IMAGES_SWEEP:
        ks = str(k)
        def _acc(subset):
            vals = [q["by_max_images"].get(ks, {}).get("correct", False) for q in subset]
            return sum(vals) / len(vals) if vals else 0.0
        def _anls(subset):
            vals = [q["by_max_images"].get(ks, {}).get("anls", 0.0) for q in subset]
            return sum(vals) / len(vals) if vals else 0.0
        def _lat(subset):
            lats = [q["by_max_images"].get(ks, {}).get("latency_sec", -1) for q in subset if q["by_max_images"].get(ks, {}).get("latency_sec", -1) >= 0]
            return round(sum(lats)/len(lats), 2) if lats else -1

        summary_by_k[ks] = {
            "all":       {"accuracy": _acc(questions),  "anls": _anls(questions),  "avg_latency": _lat(questions)},
            "visual":    {"accuracy": _acc(visual),     "anls": _anls(visual),     "avg_latency": _lat(visual)},
            "non_visual":{"accuracy": _acc(non_visual), "anls": _anls(non_visual), "avg_latency": _lat(non_visual)},
        }
        logger.info(
            f"max_images={k}: "
            f"all={summary_by_k[ks]['all']['accuracy']:.0%}  "
            f"visual={summary_by_k[ks]['visual']['accuracy']:.0%}  "
            f"non_vis={summary_by_k[ks]['non_visual']['accuracy']:.0%}"
        )

    output = {
        "sweep": MAX_IMAGES_SWEEP,
        "top_k": TOP_K,
        "total_questions": len(questions),
        "summary_by_max_images": summary_by_k,
        "details": all_results,
    }

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved to {OUT_PATH}")

    _plot_ablation(summary_by_k)
    return output


def _plot_ablation(summary_by_k: dict):
    """Generate fig6_ablation_images.png from ablation results."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker
        import numpy as np

        # Font setup (same as visualize.py)
        candidates = ["PingFang SC","Heiti SC","STHeiti","Microsoft YaHei","SimHei","DejaVu Sans"]
        from matplotlib import font_manager as fm
        available = {f.name for f in fm.fontManager.ttflist}
        for c in candidates:
            if c in available:
                matplotlib.rcParams["font.family"] = c
                break
        matplotlib.rcParams["axes.unicode_minus"] = False

        ks      = sorted(summary_by_k.keys(), key=int)
        x       = [int(k) for k in ks]
        acc_all = [summary_by_k[k]["all"]["accuracy"]        for k in ks]
        acc_vis = [summary_by_k[k]["visual"]["accuracy"]     for k in ks]
        acc_nv  = [summary_by_k[k]["non_visual"]["accuracy"] for k in ks]
        lat_all = [summary_by_k[k]["all"]["avg_latency"]     for k in ks]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

        # Left: accuracy curves
        ax1.plot(x, acc_all, "o-",  color="#4C72B0", linewidth=2, markersize=7, label="全部题目 (All)")
        ax1.plot(x, acc_vis, "s--", color="#C44E52", linewidth=2, markersize=7, label="视觉题 (Visual)")
        ax1.plot(x, acc_nv,  "^:",  color="#55A868", linewidth=2, markersize=7, label="非视觉题 (Non-visual)")
        ax1.set_xlabel("max_images（传入 VLM 的最大图片数）", fontsize=11)
        ax1.set_ylabel("Accuracy", fontsize=11)
        ax1.set_title("图片数量消融实验 — 准确率", fontsize=12)
        ax1.set_xticks(x)
        ax1.set_ylim(0, 1.1)
        ax1.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1))
        ax1.legend(fontsize=10)
        ax1.grid(axis="y", linestyle="--", alpha=0.4)
        ax1.spines["top"].set_visible(False)
        ax1.spines["right"].set_visible(False)
        for xi, yi in zip(x, acc_all):
            ax1.annotate(f"{yi:.0%}", (xi, yi), textcoords="offset points",
                         xytext=(0, 8), ha="center", fontsize=9, color="#4C72B0")

        # Right: latency curve
        ax2.plot(x, lat_all, "D-", color="#8172B2", linewidth=2, markersize=7)
        ax2.set_xlabel("max_images", fontsize=11)
        ax2.set_ylabel("平均响应时间（秒/题）", fontsize=11)
        ax2.set_title("图片数量消融实验 — 延迟", fontsize=12)
        ax2.set_xticks(x)
        ax2.grid(axis="y", linestyle="--", alpha=0.4)
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)
        for xi, yi in zip(x, lat_all):
            if yi >= 0:
                ax2.annotate(f"{yi:.1f}s", (xi, yi), textcoords="offset points",
                             xytext=(0, 8), ha="center", fontsize=9)

        plt.tight_layout()
        FIG_DIR.mkdir(exist_ok=True)
        out = FIG_DIR / "fig6_ablation_images.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved {out}")
    except Exception as e:
        logger.warning(f"Plot failed: {e}")


if __name__ == "__main__":
    run_ablation()
