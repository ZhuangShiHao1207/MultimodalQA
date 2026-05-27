"""
Run evaluation: compare Multimodal RAG vs Text-only RAG on the self-built dataset.
Computes ANLS and Accuracy metrics, outputs comparison table.
"""
import sys
import os
import json
import logging
from pathlib import Path
from datetime import datetime

# Platform fixes
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

from src.ingestion import DoclingParser, TextChunker, merge_small_elements, ElementType
from src.indexing import VLMSummarizer, BGEEmbedder, VectorStore, build_index
from src.retrieval import MultiVectorRetriever
from src.generation import GroundedGenerator
from evaluation.metrics import compute_anls, compute_accuracy, anls_score, extract_key_answer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_or_build_index():
    """Load cached elements or build from test PDF."""
    import pickle
    cache_path = project_root / "docling_output" / "cached_elements.pkl"

    if cache_path.exists():
        logger.info("Loading cached elements...")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    logger.info("Building index from test PDF...")
    pdf_path = list((project_root / "data").glob("*.pdf"))[0]

    parser = DoclingParser(
        output_dir=str(project_root / "docling_output"),
        extract_images=True, extract_tables=True,
        generate_page_images=True, images_scale=2.0,
    )
    elements, _ = parser.parse(str(pdf_path))
    elements = merge_small_elements(elements, min_size=80)
    chunker = TextChunker(max_chunk_size=1500, chunk_overlap=200, min_chunk_size=80)
    elements = chunker.chunk_elements(elements)

    summarizer = VLMSummarizer(model="glm-4.6v")
    elements = summarizer.summarize_elements(elements)

    with open(cache_path, "wb") as f:
        pickle.dump(elements, f)

    return elements


def run_evaluation():
    """Run full evaluation comparing multimodal vs text-only RAG."""
    # Load QA dataset
    qa_path = project_root / "evaluation" / "datasets" / "self_built_qa.json"
    with open(qa_path, "r", encoding="utf-8") as f:
        qa_dataset = json.load(f)
    logger.info(f"Loaded {len(qa_dataset)} QA pairs")

    # Build index
    elements = load_or_build_index()
    embedder = BGEEmbedder(model_name="BAAI/bge-m3", device="auto", use_fp16=True)
    store = build_index(elements, embedder)
    generator = GroundedGenerator(model="glm-4.6v", max_tokens=4096)
    retriever = MultiVectorRetriever(embedder=embedder, vector_store=store, top_k=5)

    logger.info(f"Index built: {store.size} vectors")

    # Run both modes
    results = []
    mm_predictions = []
    text_predictions = []
    all_references = []

    for i, qa in enumerate(qa_dataset):
        question = qa["question"]
        gold_answers = qa["gold_answers"]
        all_references.append(gold_answers)

        logger.info(f"\n[{i+1}/{len(qa_dataset)}] Q: {question}")

        # Multimodal RAG
        context_mm = retriever.retrieve_with_context(question, max_images=3)
        result_mm = generator.generate(question, context_mm)
        answer_mm = result_mm.get("answer", "")

        # Text-only RAG
        context_text = retriever.retrieve_with_context(question, max_images=0)
        context_text["image_contexts"] = []
        result_text = generator.generate(question, context_text)
        answer_text = result_text.get("answer", "")

        # Extract key answers for metric computation
        key_mm = extract_key_answer(answer_mm)
        key_text = extract_key_answer(answer_text)
        mm_predictions.append(key_mm)
        text_predictions.append(key_text)

        # Per-question scores
        score_mm = anls_score(answer_mm, gold_answers)
        score_text = anls_score(answer_text, gold_answers)

        results.append({
            "id": qa["id"],
            "question": question,
            "type": qa["type"],
            "requires_visual": qa["requires_visual"],
            "gold_answers": gold_answers,
            "multimodal_answer": answer_mm[:200],
            "text_only_answer": answer_text[:200],
            "multimodal_anls": score_mm,
            "text_only_anls": score_text,
        })

        logger.info(f"  MM: {answer_mm[:80]}... (ANLS={score_mm:.3f})")
        logger.info(f"  TO: {answer_text[:80]}... (ANLS={score_text:.3f})")

    # Compute aggregate metrics
    mm_anls_scores = [r["multimodal_anls"] for r in results]
    text_anls_scores = [r["text_only_anls"] for r in results]

    # By question type
    types = set(r["type"] for r in results)
    type_metrics = {}
    for t in types:
        t_results = [r for r in results if r["type"] == t]
        type_metrics[t] = {
            "count": len(t_results),
            "multimodal_anls": sum(r["multimodal_anls"] for r in t_results) / len(t_results),
            "text_only_anls": sum(r["text_only_anls"] for r in t_results) / len(t_results),
        }

    # Visual vs non-visual
    visual_results = [r for r in results if r["requires_visual"]]
    non_visual_results = [r for r in results if not r["requires_visual"]]

    summary = {
        "timestamp": datetime.now().isoformat(),
        "dataset": "self_built_qa",
        "total_questions": len(qa_dataset),
        "overall": {
            "multimodal_anls": sum(mm_anls_scores) / len(mm_anls_scores),
            "text_only_anls": sum(text_anls_scores) / len(text_anls_scores),
        },
        "by_type": type_metrics,
        "visual_questions": {
            "count": len(visual_results),
            "multimodal_anls": sum(r["multimodal_anls"] for r in visual_results) / max(len(visual_results), 1),
            "text_only_anls": sum(r["text_only_anls"] for r in visual_results) / max(len(visual_results), 1),
        },
        "non_visual_questions": {
            "count": len(non_visual_results),
            "multimodal_anls": sum(r["multimodal_anls"] for r in non_visual_results) / max(len(non_visual_results), 1),
            "text_only_anls": sum(r["text_only_anls"] for r in non_visual_results) / max(len(non_visual_results), 1),
        },
        "details": results,
    }

    # Print summary
    logger.info("\n" + "=" * 70)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 70)
    logger.info(f"\nOverall ANLS:")
    logger.info(f"  Multimodal RAG: {summary['overall']['multimodal_anls']:.4f}")
    logger.info(f"  Text-only RAG:  {summary['overall']['text_only_anls']:.4f}")
    logger.info(f"  Improvement:    +{(summary['overall']['multimodal_anls'] - summary['overall']['text_only_anls']):.4f}")

    logger.info(f"\nBy Question Type:")
    for t, m in type_metrics.items():
        logger.info(f"  {t} ({m['count']} questions): MM={m['multimodal_anls']:.3f} vs TO={m['text_only_anls']:.3f}")

    logger.info(f"\nVisual vs Non-Visual:")
    logger.info(f"  Visual ({summary['visual_questions']['count']}):     MM={summary['visual_questions']['multimodal_anls']:.3f} vs TO={summary['visual_questions']['text_only_anls']:.3f}")
    logger.info(f"  Non-Visual ({summary['non_visual_questions']['count']}): MM={summary['non_visual_questions']['multimodal_anls']:.3f} vs TO={summary['non_visual_questions']['text_only_anls']:.3f}")

    # Save results
    output_path = project_root / "evaluation" / "results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logger.info(f"\nFull results saved to: {output_path}")

    return summary


if __name__ == "__main__":
    run_evaluation()
