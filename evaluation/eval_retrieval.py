"""
evaluation/eval_retrieval.py
Retrieval-layer evaluation: Image Recall@K and Chunk Recall@K.

This script measures whether the retriever surfaces the correct document
elements (images / text chunks) for each question, independently of whether
the downstream VLM gives the right answer.

─────────────────────────────────────────────────────────────
USAGE
─────────────────────────────────────────────────────────────
  # Run full retrieval eval (needs backend indexes built first):
  python -m evaluation.eval_retrieval

  # Dry-run: just validate the expected_elements annotations
  python -m evaluation.eval_retrieval --validate-only

─────────────────────────────────────────────────────────────
STEP 1 — Annotate expected_elements in self_built_qa.json
─────────────────────────────────────────────────────────────
For each QA item, add an "expected_elements" field:

  {
    "id": "q04",
    "requires_visual": true,
    "expected_elements": {
      "images": ["figure_1_page5.png"],   ← filename only, no full path
      "text_keywords": [],                ← optional: words that must appear in retrieved chunk
      "tables": []                        ← optional: table label/caption keywords
    }
  }

For text / table questions, leave "images" as [] and fill in keywords.
For visual questions, "images" is the primary signal.

─────────────────────────────────────────────────────────────
METRICS
─────────────────────────────────────────────────────────────
  Image Recall@K   = fraction of expected images appearing in top-K retrieved images
  Chunk Recall@K   = fraction of questions where at least one retrieved chunk
                     contains all specified text_keywords (case-insensitive)
  MRR (images)     = mean reciprocal rank of first expected image in results
"""

import sys
import os
import json
import logging
import hashlib
import argparse
from pathlib import Path
from typing import Optional

# Platform fixes (same as run_eval.py)
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR   = project_root / "data"
CHROMA_DIR = project_root / "backend" / "chroma_data"
QA_PATH    = project_root / "evaluation" / "datasets" / "self_built_qa.json"


def get_doc_id(filename: str) -> str:
    return hashlib.md5(filename.encode()).hexdigest()[:8]


# ═══════════════════════════════════════════════════════════════════════════════
# Core metric helpers
# ═══════════════════════════════════════════════════════════════════════════════

def image_recall_at_k(retrieved_image_paths: list[str],
                      expected_filenames: list[str]) -> float:
    """
    Recall@K for images.
    retrieved_image_paths: full paths returned by retriever (ordered by score)
    expected_filenames:    basenames specified in expected_elements["images"]
    Returns fraction of expected images found anywhere in the retrieved list.
    """
    if not expected_filenames:
        return None  # not applicable
    retrieved_names = [Path(p).name for p in retrieved_image_paths]
    hits = sum(1 for e in expected_filenames if e in retrieved_names)
    return hits / len(expected_filenames)


def image_mrr(retrieved_image_paths: list[str],
              expected_filenames: list[str]) -> float:
    """
    Mean Reciprocal Rank for images.
    Returns 1/rank of the first hit, 0 if none found.
    """
    if not expected_filenames:
        return None
    retrieved_names = [Path(p).name for p in retrieved_image_paths]
    for rank, name in enumerate(retrieved_names, start=1):
        if name in expected_filenames:
            return 1.0 / rank
    return 0.0


def chunk_recall(retrieved_text_chunks: list[str],
                 keywords: list[str]) -> Optional[float]:
    """
    Chunk Recall: 1.0 if any retrieved chunk contains ALL keywords (case-insensitive).
    Returns None if no keywords specified (not applicable).
    """
    if not keywords:
        return None
    combined = " ".join(retrieved_text_chunks).lower()
    all_found = all(kw.lower() in combined for kw in keywords)
    return 1.0 if all_found else 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# Validation helper
# ═══════════════════════════════════════════════════════════════════════════════

def validate_annotations(qa_dataset: list) -> None:
    """Check that expected_elements annotations are present and well-formed."""
    missing = []
    for qa in qa_dataset:
        ee = qa.get("expected_elements")
        if ee is None:
            missing.append(qa["id"])
        else:
            # basic schema check
            assert isinstance(ee.get("images", []), list), f"{qa['id']}: images must be list"
            assert isinstance(ee.get("text_keywords", []), list), f"{qa['id']}: text_keywords must be list"
            assert isinstance(ee.get("tables", []), list), f"{qa['id']}: tables must be list"

    if missing:
        logger.warning(
            f"{len(missing)} questions missing 'expected_elements': {missing}\n"
            "  → Add the field to self_built_qa.json before running retrieval eval.\n"
            "  → See the docstring at the top of this file for the schema."
        )
    else:
        logger.info("All questions have 'expected_elements' annotations. ✓")


# ═══════════════════════════════════════════════════════════════════════════════
# Main evaluation loop
# ═══════════════════════════════════════════════════════════════════════════════

def run_retrieval_eval(top_k: int = 5, score_threshold: float = 0.3):
    with open(QA_PATH, encoding="utf-8") as f:
        qa_dataset = json.load(f)

    logger.info(f"Loaded {len(qa_dataset)} QA pairs from {QA_PATH}")
    validate_annotations(qa_dataset)

    # Load ChromaDB collections
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
                logger.info(f"Loaded index: '{fname}'  (id={doc_id}, vectors={store.size})")
            except Exception as e:
                logger.error(f"Failed to load index for {fname}: {e}")

    if not documents:
        logger.error("No indexes found — run the backend first to build ChromaDB indexes.")
        return

    embedder = BGEEmbedder(model_name="BAAI/bge-m3", device="auto", use_fp16=True)

    results = []
    for qa in qa_dataset:
        qid    = qa["id"]
        fname  = qa["document"]
        ee     = qa.get("expected_elements", {})

        if fname not in documents:
            logger.warning(f"Skipping {qid}: document not loaded")
            continue

        store     = documents[fname]["store"]
        retriever = MultiVectorRetriever(
            embedder=embedder, vector_store=store,
            top_k=top_k, score_threshold=score_threshold,
        )

        ctx = retriever.retrieve_with_context(qa["question"], max_images=top_k)

        retrieved_img_paths  = [img["image_path"] for img in ctx["image_contexts"]]
        retrieved_img_scores = [img["score"]      for img in ctx["image_contexts"]]
        retrieved_txt_chunks = [t["content"]      for t in ctx["text_contexts"]]
        retrieved_tbl_chunks = [t["content"]      for t in ctx["table_contexts"]]

        expected_imgs     = ee.get("images", [])
        expected_kws      = ee.get("text_keywords", [])

        img_recall = image_recall_at_k(retrieved_img_paths, expected_imgs)
        img_mrr    = image_mrr(retrieved_img_paths, expected_imgs)
        txt_recall = chunk_recall(retrieved_txt_chunks + retrieved_tbl_chunks, expected_kws)

        r = {
            "id":               qid,
            "type":             qa["type"],
            "requires_visual":  qa["requires_visual"],
            # image retrieval
            "expected_images":       expected_imgs,
            "retrieved_images":      [Path(p).name for p in retrieved_img_paths],
            "retrieved_img_scores":  [round(s, 3) for s in retrieved_img_scores],
            "image_recall":          round(img_recall, 3) if img_recall is not None else None,
            "image_mrr":             round(img_mrr,    3) if img_mrr    is not None else None,
            # text/chunk retrieval
            "expected_keywords":     expected_kws,
            "chunk_recall":          round(txt_recall, 3) if txt_recall is not None else None,
        }
        results.append(r)

        status = ""
        if img_recall is not None:
            status += f"ImgRecall={img_recall:.0%} MRR={img_mrr:.2f} "
        if txt_recall is not None:
            status += f"ChunkRecall={txt_recall:.0%}"
        logger.info(f"  [{qid}] {status or '(no annotation)'}")

    # ── Aggregate ──────────────────────────────────────────────────────────────
    img_qs      = [r for r in results if r["image_recall"]  is not None]
    txt_qs      = [r for r in results if r["chunk_recall"]  is not None]
    vis_img_qs  = [r for r in img_qs  if r["requires_visual"]]

    def avg(lst, key):
        vals = [r[key] for r in lst if r[key] is not None]
        return round(sum(vals) / len(vals), 3) if vals else None

    summary = {
        "top_k":               top_k,
        "score_threshold":     score_threshold,
        "total_questions":     len(results),
        "image_recall": {
            "all_visual_questions":  avg(vis_img_qs, "image_recall"),
            "all_annotated":         avg(img_qs,     "image_recall"),
            "mrr":                   avg(img_qs,     "image_mrr"),
            "n":                     len(img_qs),
        },
        "chunk_recall": {
            "all_annotated": avg(txt_qs, "chunk_recall"),
            "n":             len(txt_qs),
        },
        "details": results,
    }

    # ── Print ──────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"RETRIEVAL EVALUATION  (top_k={top_k}, threshold={score_threshold})")
    print("=" * 60)
    ir = summary["image_recall"]
    cr = summary["chunk_recall"]
    if ir["n"]:
        print(f"  Image Recall@{top_k} (visual Qs, n={ir['n']}): {ir['all_visual_questions']:.0%}")
        print(f"  Image MRR              (annotated, n={ir['n']}): {ir['mrr']:.3f}")
    else:
        print("  Image Recall: no annotations yet — add expected_elements[\"images\"] to QA dataset")
    if cr["n"]:
        print(f"  Chunk Recall           (annotated, n={cr['n']}): {cr['all_annotated']:.0%}")
    else:
        print("  Chunk Recall: no annotations yet — add expected_elements[\"text_keywords\"]")

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path = project_root / "evaluation" / "retrieval_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\nSaved to: {out_path}")
    return summary


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retrieval-layer evaluation")
    parser.add_argument("--validate-only", action="store_true",
                        help="Only check that expected_elements annotations exist, don't run retrieval")
    parser.add_argument("--top-k",   type=int,   default=5,   help="Retriever top_k (default 5)")
    parser.add_argument("--threshold", type=float, default=0.3, help="Score threshold (default 0.3)")
    args = parser.parse_args()

    sys.path.insert(0, str(project_root))

    if args.validate_only:
        with open(QA_PATH, encoding="utf-8") as f:
            qa = json.load(f)
        validate_annotations(qa)
    else:
        run_retrieval_eval(top_k=args.top_k, score_threshold=args.threshold)
