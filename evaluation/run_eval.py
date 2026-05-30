"""
Evaluation: Multimodal RAG vs Text-only RAG on multi-document QA dataset.
Routes each question to the correct ChromaDB collection based on the `document` field.
Uses the persistent ChromaDB indexes built by the backend.
"""
import sys
import os
import json
import logging
import hashlib
from pathlib import Path
from datetime import datetime

# Platform fixes (must come before importing transformers)
if sys.platform == "win32":
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")

import transformers.utils.import_utils
if hasattr(transformers.utils.import_utils, 'check_torch_load_is_safe'):
    transformers.utils.import_utils.check_torch_load_is_safe = lambda: None
import transformers.modeling_utils
if hasattr(transformers.modeling_utils, 'check_torch_load_is_safe'):
    transformers.modeling_utils.check_torch_load_is_safe = lambda: None

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.indexing import BGEEmbedder, VectorStore
from src.retrieval import MultiVectorRetriever
from src.generation import GroundedGenerator
from evaluation.metrics import anls_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ============================================================
# Document → doc_id mapping (matches backend/services.py logic)
# ============================================================
DATA_DIR = project_root / "data"
CHROMA_DIR = project_root / "backend" / "chroma_data"


def get_doc_id_for_filename(filename: str) -> str:
    """Replicate backend's hash-based doc_id mapping."""
    return hashlib.md5(filename.encode()).hexdigest()[:8]


def run_evaluation():
    """Run full evaluation on multi-document QA dataset."""
    qa_path = project_root / "evaluation" / "datasets" / "self_built_qa.json"
    with open(qa_path, "r", encoding="utf-8") as f:
        qa_dataset = json.load(f)
    logger.info(f"Loaded {len(qa_dataset)} QA pairs")

    # Build mapping from PDF filename → doc_id and load ChromaDB collections
    documents = {}
    for qa in qa_dataset:
        fname = qa["document"]
        if fname not in documents:
            doc_id = get_doc_id_for_filename(fname)
            collection_name = f"doc_{doc_id}"
            try:
                store = VectorStore(
                    collection_name=collection_name,
                    persist_dir=str(CHROMA_DIR),
                )
                if store.size == 0:
                    logger.error(f"Empty collection for {fname} (id={doc_id})")
                    continue
                documents[fname] = {"doc_id": doc_id, "store": store}
                logger.info(f"Loaded index for '{fname}' (id={doc_id}, vectors={store.size})")
            except Exception as e:
                logger.error(f"Failed to load index for {fname}: {e}")
                continue

    if not documents:
        logger.error("No documents could be loaded! Run the backend first to build indexes.")
        return

    # Initialize models (singletons)
    embedder = BGEEmbedder(model_name="BAAI/bge-m3", device="auto", use_fp16=True)
    generator = GroundedGenerator(model="glm-4.6v", max_tokens=4096)

    # Run evaluation
    results = []

    for i, qa in enumerate(qa_dataset):
        question = qa["question"]
        gold_answers = qa["gold_answers"]
        doc_filename = qa["document"]

        if doc_filename not in documents:
            logger.warning(f"Skipping {qa['id']}: document not loaded")
            continue

        store = documents[doc_filename]["store"]
        retriever = MultiVectorRetriever(embedder=embedder, vector_store=store, top_k=5)

        logger.info(f"\n[{i+1}/{len(qa_dataset)}] {qa['id']} ({qa['type']}) Q: {question[:80]}")

        # Multimodal RAG (with images)
        try:
            context_mm = retriever.retrieve_with_context(question, max_images=3)
            result_mm = generator.generate(question, context_mm)
            answer_mm = result_mm.get("answer", "")
        except Exception as e:
            logger.error(f"  MM error: {e}")
            answer_mm = ""

        # Text-only RAG (strip images)
        try:
            context_text = retriever.retrieve_with_context(question, max_images=0)
            context_text["image_contexts"] = []
            result_text = generator.generate(question, context_text)
            answer_text = result_text.get("answer", "")
        except Exception as e:
            logger.error(f"  TO error: {e}")
            answer_text = ""

        # Compute ANLS scores
        score_mm = anls_score(answer_mm, gold_answers)
        score_text = anls_score(answer_text, gold_answers)

        results.append({
            "id": qa["id"],
            "document": doc_filename,
            "question": question,
            "type": qa["type"],
            "requires_visual": qa["requires_visual"],
            "gold_answers": gold_answers,
            "multimodal_answer": answer_mm,
            "text_only_answer": answer_text,
            "multimodal_anls": score_mm,
            "text_only_anls": score_text,
            "mm_correct": score_mm >= 0.5,
            "to_correct": score_text >= 0.5,
        })

        logger.info(f"  MM[ANLS={score_mm:.3f}]: {answer_mm[:120]}")
        logger.info(f"  TO[ANLS={score_text:.3f}]: {answer_text[:120]}")

    # ─── Aggregate metrics ─────────────────────────────────
    n = len(results)
    mm_avg = sum(r["multimodal_anls"] for r in results) / max(n, 1)
    to_avg = sum(r["text_only_anls"] for r in results) / max(n, 1)
    mm_acc = sum(1 for r in results if r["mm_correct"]) / max(n, 1)
    to_acc = sum(1 for r in results if r["to_correct"]) / max(n, 1)

    # By question type
    types = sorted(set(r["type"] for r in results))
    type_metrics = {}
    for t in types:
        tr = [r for r in results if r["type"] == t]
        type_metrics[t] = {
            "count": len(tr),
            "multimodal_anls": sum(r["multimodal_anls"] for r in tr) / len(tr),
            "text_only_anls": sum(r["text_only_anls"] for r in tr) / len(tr),
            "mm_accuracy": sum(1 for r in tr if r["mm_correct"]) / len(tr),
            "to_accuracy": sum(1 for r in tr if r["to_correct"]) / len(tr),
        }

    # Visual vs non-visual
    visual = [r for r in results if r["requires_visual"]]
    non_visual = [r for r in results if not r["requires_visual"]]

    summary = {
        "timestamp": datetime.now().isoformat(),
        "dataset": "self_built_qa (corrected)",
        "model": "glm-4.6v",
        "total_questions": n,
        "documents_evaluated": list(documents.keys()),
        "overall": {
            "multimodal_anls": mm_avg,
            "text_only_anls": to_avg,
            "multimodal_accuracy": mm_acc,
            "text_only_accuracy": to_acc,
            "anls_improvement": mm_avg - to_avg,
            "accuracy_improvement": mm_acc - to_acc,
        },
        "by_type": type_metrics,
        "visual_questions": {
            "count": len(visual),
            "multimodal_anls": sum(r["multimodal_anls"] for r in visual) / max(len(visual), 1),
            "text_only_anls": sum(r["text_only_anls"] for r in visual) / max(len(visual), 1),
            "mm_accuracy": sum(1 for r in visual if r["mm_correct"]) / max(len(visual), 1),
            "to_accuracy": sum(1 for r in visual if r["to_correct"]) / max(len(visual), 1),
        },
        "non_visual_questions": {
            "count": len(non_visual),
            "multimodal_anls": sum(r["multimodal_anls"] for r in non_visual) / max(len(non_visual), 1),
            "text_only_anls": sum(r["text_only_anls"] for r in non_visual) / max(len(non_visual), 1),
            "mm_accuracy": sum(1 for r in non_visual if r["mm_correct"]) / max(len(non_visual), 1),
            "to_accuracy": sum(1 for r in non_visual if r["to_correct"]) / max(len(non_visual), 1),
        },
        "details": results,
    }

    # ─── Print summary ─────────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("EVALUATION RESULTS (Multi-document, Corrected QA)")
    logger.info("=" * 70)
    logger.info(f"\nOverall:")
    logger.info(f"  Multimodal RAG: ANLS={mm_avg:.4f}  Acc={mm_acc:.2%}")
    logger.info(f"  Text-only  RAG: ANLS={to_avg:.4f}  Acc={to_acc:.2%}")
    logger.info(f"  Improvement:    +{mm_avg - to_avg:.4f} ANLS, +{(mm_acc - to_acc)*100:.1f}pp Acc")

    logger.info(f"\nBy Question Type:")
    for t, m in type_metrics.items():
        logger.info(f"  {t:10s} ({m['count']}): MM={m['multimodal_anls']:.3f}/{m['mm_accuracy']:.0%}  TO={m['text_only_anls']:.3f}/{m['to_accuracy']:.0%}")

    logger.info(f"\nVisual vs Non-Visual:")
    v = summary['visual_questions']
    nv = summary['non_visual_questions']
    logger.info(f"  Visual    ({v['count']}): MM={v['multimodal_anls']:.3f}/{v['mm_accuracy']:.0%}  TO={v['text_only_anls']:.3f}/{v['to_accuracy']:.0%}")
    logger.info(f"  NonVisual ({nv['count']}): MM={nv['multimodal_anls']:.3f}/{nv['mm_accuracy']:.0%}  TO={nv['text_only_anls']:.3f}/{nv['to_accuracy']:.0%}")

    # Print failures for the report
    mm_failures = [r for r in results if not r["mm_correct"]]
    if mm_failures:
        logger.info(f"\nMultimodal RAG FAILURES ({len(mm_failures)}/{n}):")
        for f in mm_failures:
            logger.info(f"  {f['id']} [{f['type']}] expected: {f['gold_answers'][0][:60]}")
            logger.info(f"            got: {f['multimodal_answer'][:120]}")

    # Save results
    output_path = project_root / "evaluation" / "results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logger.info(f"\nFull results saved to: {output_path}")

    return summary


if __name__ == "__main__":
    run_evaluation()
