"""
Evaluation: Multimodal RAG vs Text-only RAG on multi-document QA dataset.
Routes each question to the correct ChromaDB collection based on the `document` field.
Uses the persistent ChromaDB indexes built by the backend.

If a document's index is missing (vectors=0), the script automatically runs the
full ingestion pipeline (Docling parse → VLM summarize → BGE-M3 embed → ChromaDB)
without needing the backend/frontend running. Pass --skip-build to disable this
and fail fast instead.
"""
import sys
import os
import json
import shutil
import logging
import hashlib
import time
import argparse
from pathlib import Path
from datetime import datetime

# Platform fixes (must come before importing transformers)
if sys.platform == "win32":
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
try:
    import pyarrow  # noqa: must come before torch
except Exception:
    pass

import transformers.utils.import_utils
if hasattr(transformers.utils.import_utils, 'check_torch_load_is_safe'):
    transformers.utils.import_utils.check_torch_load_is_safe = lambda: None
import transformers.modeling_utils
if hasattr(transformers.modeling_utils, 'check_torch_load_is_safe'):
    transformers.modeling_utils.check_torch_load_is_safe = lambda: None

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.indexing import BGEEmbedder, VectorStore, build_index
from src.retrieval import MultiVectorRetriever
from src.generation import GroundedGenerator
from evaluation.metrics import anls_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ============================================================
# Document → doc_id mapping (matches backend/services.py logic)
# ============================================================
DATA_DIR    = project_root / "data"
CHROMA_DIR  = project_root / "backend" / "chroma_data"
DOCS_DIR    = project_root / "backend" / "documents"


def get_doc_id_for_filename(filename: str) -> str:
    """Replicate backend's hash-based doc_id mapping."""
    return hashlib.md5(filename.encode()).hexdigest()[:8]


# ============================================================
# Auto index builder — runs the full pipeline when index missing
# ============================================================
def ensure_index_built(filename: str, embedder: BGEEmbedder, skip_build: bool = False) -> VectorStore | None:
    """
    Return the VectorStore for *filename*, building the index from scratch if needed.

    Steps (mirrors backend/services.py::process_document_sync):
      1. Locate PDF in data/ or backend/documents/<id>/source.pdf
      2. Docling parse → elements
      3. Merge + chunk
      4. VLM summarize figures/tables
      5. Copy images to persistent backend/documents/<id>/images|pages/
      6. BGE-M3 embed → ChromaDB persist

    Args:
        filename:   PDF basename as stored in self_built_qa.json
        embedder:   shared BGEEmbedder instance (avoid reloading the model)
        skip_build: if True, return None instead of building

    Returns:
        VectorStore with vectors > 0, or None on failure.
    """
    doc_id = get_doc_id_for_filename(filename)
    collection_name = f"doc_{doc_id}"

    # --- Check if already built -------------------------------------------------
    store = VectorStore(collection_name=collection_name, persist_dir=str(CHROMA_DIR))
    if store.size > 0:
        logger.info(f"Index already exists for '{filename}' ({store.size} vectors) — skipping build")
        return store

    if skip_build:
        logger.error(f"No index for '{filename}' and --skip-build is set. Skipping document.")
        return None

    # --- Locate source PDF ------------------------------------------------------
    pdf_path = DATA_DIR / filename
    if not pdf_path.exists():
        # Try backend/documents/<id>/source.pdf
        pdf_path = DOCS_DIR / doc_id / "source.pdf"
    if not pdf_path.exists():
        logger.error(f"PDF not found for '{filename}' (tried data/ and backend/documents/). Skipping.")
        return None

    logger.info(f"Building index for '{filename}' (id={doc_id}) from {pdf_path} ...")
    t_start = time.time()

    doc_dir = DOCS_DIR / doc_id
    doc_dir.mkdir(parents=True, exist_ok=True)
    (doc_dir / "images").mkdir(exist_ok=True)
    (doc_dir / "pages").mkdir(exist_ok=True)

    # Copy PDF to documents dir if not already there
    dest_pdf = doc_dir / "source.pdf"
    if not dest_pdf.exists():
        shutil.copy2(pdf_path, dest_pdf)

    try:
        # Stage 1: Parse with Docling
        from src.ingestion import DoclingParser, TextChunker, merge_small_elements
        logger.info(f"  [1/4] Parsing with Docling...")
        parser = DoclingParser(
            output_dir=str(doc_dir / "docling_output"),
            extract_images=True,
            extract_tables=True,
            generate_page_images=True,
            images_scale=2.0,
        )
        elements, _ = parser.parse(str(pdf_path))
        logger.info(f"  [1/4] Parsed {len(elements)} elements")

        # Stage 2: Merge + chunk
        logger.info(f"  [2/4] Chunking...")
        elements = merge_small_elements(elements, min_size=80)
        chunker = TextChunker(max_chunk_size=1500, chunk_overlap=200, min_chunk_size=80)
        elements = chunker.chunk_elements(elements)
        logger.info(f"  [2/4] {len(elements)} chunks")

        # Stage 3: VLM summaries for figures/tables
        from src.ingestion.models import ElementType
        from src.indexing import VLMSummarizer
        visual_count = sum(1 for e in elements if e.type in (ElementType.FIGURE, ElementType.TABLE))
        logger.info(f"  [3/4] Summarizing {visual_count} visual elements with GLM-4.6V...")
        summarizer = VLMSummarizer(model="glm-4.6v")
        elements = summarizer.summarize_elements(elements)

        # Stage 4: Copy images to persistent paths
        for elem in elements:
            if elem.image_path and Path(elem.image_path).exists():
                if elem.type == ElementType.PAGE_IMAGE:
                    dest = doc_dir / "pages" / Path(elem.image_path).name
                else:
                    dest = doc_dir / "images" / Path(elem.image_path).name
                if not dest.exists():
                    shutil.copy2(elem.image_path, dest)
                elem.image_path = str(dest)

        # Stage 5: Embed + persist to ChromaDB
        logger.info(f"  [4/4] Embedding with BGE-M3 → ChromaDB...")
        store = build_index(
            elements, embedder,
            persist_dir=str(CHROMA_DIR),
            collection_name=collection_name,
        )

        # Cleanup Docling temp output
        docling_out = doc_dir / "docling_output"
        if docling_out.exists():
            shutil.rmtree(docling_out)

        elapsed = round(time.time() - t_start, 1)
        logger.info(f"  Done: {store.size} vectors indexed in {elapsed}s")
        return store

    except Exception as e:
        logger.error(f"Failed to build index for '{filename}': {e}", exc_info=True)
        return None


def run_evaluation(skip_build: bool = False):
    """Run full evaluation on multi-document QA dataset."""
    qa_path = project_root / "evaluation" / "datasets" / "self_built_qa.json"
    with open(qa_path, "r", encoding="utf-8") as f:
        qa_dataset = json.load(f)
    logger.info(f"Loaded {len(qa_dataset)} QA pairs")

    # Initialize embedder once (shared across all documents)
    embedder = BGEEmbedder(model_name="BAAI/bge-m3", device="auto", use_fp16=True)
    generator = GroundedGenerator(model="glm-4.6v", max_tokens=4096)

    # Build/load ChromaDB collections for every unique document
    documents = {}
    for qa in qa_dataset:
        fname = qa["document"]
        if fname not in documents:
            store = ensure_index_built(fname, embedder, skip_build=skip_build)
            if store is not None:
                documents[fname] = {"doc_id": get_doc_id_for_filename(fname), "store": store}
                logger.info(f"Ready: '{fname}' ({store.size} vectors)")
            else:
                logger.warning(f"Skipping all questions for '{fname}' (no index)")

    if not documents:
        logger.error("No documents could be loaded or built! Check data/ directory.")
        return

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

        # ── Multimodal RAG (with images) ──────────────────────────────────────
        try:
            t0 = time.time()
            context_mm = retriever.retrieve_with_context(question, max_images=3)
            result_mm = generator.generate(question, context_mm, mode="auto")
            latency_mm = round(time.time() - t0, 2)
            answer_mm = result_mm.get("answer", "")
        except Exception as e:
            logger.error(f"  MM error: {e}")
            answer_mm = ""
            latency_mm = -1.0
            context_mm = {"image_contexts": []}

        # ── Text-only Grounded RAG (refuses when no evidence) ─────────────────
        try:
            t0 = time.time()
            context_text = retriever.retrieve_with_context(question, max_images=0)
            context_text["image_contexts"] = []
            result_text = generator.generate(question, context_text, mode="grounded")
            latency_to = round(time.time() - t0, 2)
            answer_text = result_text.get("answer", "")
        except Exception as e:
            logger.error(f"  TO error: {e}")
            answer_text = ""
            latency_to = -1.0

        # ── Text-only Open RAG (allowed to infer) ─────────────────────────────
        try:
            t0 = time.time()
            context_open = retriever.retrieve_with_context(question, max_images=0)
            context_open["image_contexts"] = []
            result_open = generator.generate(question, context_open, mode="open")
            latency_open = round(time.time() - t0, 2)
            answer_open = result_open.get("answer", "")
        except Exception as e:
            logger.error(f"  Open error: {e}")
            answer_open = ""
            latency_open = -1.0

        # Compute ANLS scores
        score_mm   = anls_score(answer_mm,   gold_answers)
        score_text = anls_score(answer_text, gold_answers)
        score_open = anls_score(answer_open, gold_answers)

        results.append({
            "id": qa["id"],
            "document": doc_filename,
            "question": question,
            "type": qa["type"],
            "requires_visual": qa["requires_visual"],
            "difficulty": qa.get("difficulty", "unknown"),
            "gold_answers": gold_answers,
            "multimodal_answer":   answer_mm,
            "text_only_answer":    answer_text,
            "text_open_answer":    answer_open,
            "multimodal_anls":     score_mm,
            "text_only_anls":      score_text,
            "text_open_anls":      score_open,
            "mm_correct":          score_mm   >= 0.5,
            "to_correct":          score_text >= 0.5,
            "open_correct":        score_open >= 0.5,
            "latency_mm_sec":      latency_mm,
            "latency_to_sec":      latency_to,
            "latency_open_sec":    latency_open,
            "num_images_sent":     len(context_mm.get("image_contexts", [])),
        })

        logger.info(f"  MM  [ANLS={score_mm:.3f}, {latency_mm:.1f}s, {len(context_mm.get('image_contexts',[]))}img]: {answer_mm[:80]}")
        logger.info(f"  TO  [ANLS={score_text:.3f}, {latency_to:.1f}s]: {answer_text[:80]}")
        logger.info(f"  Open[ANLS={score_open:.3f}, {latency_open:.1f}s]: {answer_open[:80]}")

    # ─── Aggregate metrics ─────────────────────────────────
    n = len(results)
    mm_avg   = sum(r["multimodal_anls"]  for r in results) / max(n, 1)
    to_avg   = sum(r["text_only_anls"]   for r in results) / max(n, 1)
    open_avg = sum(r["text_open_anls"]   for r in results) / max(n, 1)
    mm_acc   = sum(1 for r in results if r["mm_correct"])   / max(n, 1)
    to_acc   = sum(1 for r in results if r["to_correct"])   / max(n, 1)
    open_acc = sum(1 for r in results if r["open_correct"]) / max(n, 1)

    # Latency stats (exclude failed calls marked as -1)
    valid_mm_lat   = [r["latency_mm_sec"]   for r in results if r["latency_mm_sec"]   >= 0]
    valid_to_lat   = [r["latency_to_sec"]   for r in results if r["latency_to_sec"]   >= 0]
    valid_open_lat = [r["latency_open_sec"] for r in results if r["latency_open_sec"] >= 0]
    avg_lat_mm   = round(sum(valid_mm_lat)   / len(valid_mm_lat),   2) if valid_mm_lat   else -1
    avg_lat_to   = round(sum(valid_to_lat)   / len(valid_to_lat),   2) if valid_to_lat   else -1
    avg_lat_open = round(sum(valid_open_lat) / len(valid_open_lat), 2) if valid_open_lat else -1
    avg_img_cnt  = round(sum(r["num_images_sent"] for r in results) / max(n, 1), 2)

    # By question type
    types = sorted(set(r["type"] for r in results))
    type_metrics = {}
    for t in types:
        tr = [r for r in results if r["type"] == t]
        type_metrics[t] = {
            "count":           len(tr),
            "multimodal_anls": sum(r["multimodal_anls"]  for r in tr) / len(tr),
            "text_only_anls":  sum(r["text_only_anls"]   for r in tr) / len(tr),
            "text_open_anls":  sum(r["text_open_anls"]   for r in tr) / len(tr),
            "mm_accuracy":     sum(1 for r in tr if r["mm_correct"])   / len(tr),
            "to_accuracy":     sum(1 for r in tr if r["to_correct"])   / len(tr),
            "open_accuracy":   sum(1 for r in tr if r["open_correct"]) / len(tr),
        }

    # By difficulty
    difficulties = sorted(set(r.get("difficulty","unknown") for r in results))
    diff_metrics = {}
    for d in difficulties:
        dr = [r for r in results if r.get("difficulty") == d]
        diff_metrics[d] = {
            "count":       len(dr),
            "mm_accuracy": sum(1 for r in dr if r["mm_correct"])   / len(dr),
            "to_accuracy": sum(1 for r in dr if r["to_correct"])   / len(dr),
            "open_accuracy":sum(1 for r in dr if r["open_correct"])/ len(dr),
        }

    # Visual vs non-visual
    visual     = [r for r in results if r["requires_visual"]]
    non_visual = [r for r in results if not r["requires_visual"]]

    summary = {
        "timestamp": datetime.now().isoformat(),
        "dataset": "self_built_qa (124 questions, 5 domains)",
        "model": "glm-4.6v",
        "total_questions": n,
        "documents_evaluated": list(documents.keys()),
        "overall": {
            "multimodal_anls":    mm_avg,
            "text_only_anls":     to_avg,
            "text_open_anls":     open_avg,
            "multimodal_accuracy": mm_acc,
            "text_only_accuracy":  to_acc,
            "text_open_accuracy":  open_acc,
            "mm_vs_to_improvement":   round(mm_acc - to_acc,   4),
            "mm_vs_open_improvement": round(mm_acc - open_acc, 4),
        },
        "latency": {
            "mm_avg_sec":          avg_lat_mm,
            "to_avg_sec":          avg_lat_to,
            "open_avg_sec":        avg_lat_open,
            "avg_images_per_query": avg_img_cnt,
            "mm_overhead_vs_to_sec": round(avg_lat_mm - avg_lat_to, 2) if avg_lat_mm >= 0 and avg_lat_to >= 0 else -1,
        },
        "by_type":       type_metrics,
        "by_difficulty": diff_metrics,
        "visual_questions": {
            "count":           len(visual),
            "multimodal_anls": sum(r["multimodal_anls"]  for r in visual) / max(len(visual), 1),
            "text_only_anls":  sum(r["text_only_anls"]   for r in visual) / max(len(visual), 1),
            "text_open_anls":  sum(r["text_open_anls"]   for r in visual) / max(len(visual), 1),
            "mm_accuracy":     sum(1 for r in visual if r["mm_correct"])   / max(len(visual), 1),
            "to_accuracy":     sum(1 for r in visual if r["to_correct"])   / max(len(visual), 1),
            "open_accuracy":   sum(1 for r in visual if r["open_correct"]) / max(len(visual), 1),
        },
        "non_visual_questions": {
            "count":           len(non_visual),
            "multimodal_anls": sum(r["multimodal_anls"]  for r in non_visual) / max(len(non_visual), 1),
            "text_only_anls":  sum(r["text_only_anls"]   for r in non_visual) / max(len(non_visual), 1),
            "text_open_anls":  sum(r["text_open_anls"]   for r in non_visual) / max(len(non_visual), 1),
            "mm_accuracy":     sum(1 for r in non_visual if r["mm_correct"])   / max(len(non_visual), 1),
            "to_accuracy":     sum(1 for r in non_visual if r["to_correct"])   / max(len(non_visual), 1),
            "open_accuracy":   sum(1 for r in non_visual if r["open_correct"]) / max(len(non_visual), 1),
        },
        "details": results,
    }

    # ─── Print summary ─────────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("EVALUATION RESULTS (5 domains, 124 questions)")
    logger.info("=" * 70)
    logger.info(f"\nOverall (3-way comparison):")
    logger.info(f"  Multimodal RAG  : ANLS={mm_avg:.4f}  Acc={mm_acc:.2%}")
    logger.info(f"  Text-only Grounded: ANLS={to_avg:.4f}  Acc={to_acc:.2%}")
    logger.info(f"  Text-only Open  : ANLS={open_avg:.4f}  Acc={open_acc:.2%}")
    logger.info(f"  MM vs TO:   +{(mm_acc-to_acc)*100:.1f}pp  |  MM vs Open: +{(mm_acc-open_acc)*100:.1f}pp")

    logger.info(f"\nLatency (avg per question):")
    logger.info(f"  Multimodal RAG  : {avg_lat_mm:.1f}s  (avg {avg_img_cnt:.1f} images/query)")
    logger.info(f"  Text-only Grounded: {avg_lat_to:.1f}s")
    logger.info(f"  Text-only Open  : {avg_lat_open:.1f}s")
    logger.info(f"  Image overhead (MM vs TO): +{summary['latency']['mm_overhead_vs_to_sec']:.1f}s/query")

    logger.info(f"\nBy Question Type:")
    for t, m in type_metrics.items():
        logger.info(f"  {t:8s}({m['count']:3d}): MM={m['mm_accuracy']:.0%}  TO={m['to_accuracy']:.0%}  Open={m['open_accuracy']:.0%}")

    logger.info(f"\nBy Difficulty:")
    for d, m in diff_metrics.items():
        logger.info(f"  {d:8s}({m['count']:3d}): MM={m['mm_accuracy']:.0%}  TO={m['to_accuracy']:.0%}  Open={m['open_accuracy']:.0%}")

    logger.info(f"\nVisual vs Non-Visual:")
    v  = summary["visual_questions"]
    nv = summary["non_visual_questions"]
    logger.info(f"  Visual   ({v['count']:3d}): MM={v['mm_accuracy']:.0%}  TO={v['to_accuracy']:.0%}  Open={v['open_accuracy']:.0%}")
    logger.info(f"  NonVisual({nv['count']:3d}): MM={nv['mm_accuracy']:.0%}  TO={nv['to_accuracy']:.0%}  Open={nv['open_accuracy']:.0%}")

    # Print MM failures for error analysis
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
    parser = argparse.ArgumentParser(description="MultimodalQA evaluation script")
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Do not auto-build missing indexes; skip documents without an existing index instead.",
    )
    args = parser.parse_args()
    run_evaluation(skip_build=args.skip_build)
