"""
End-to-end QA test: Compare Multimodal RAG vs Text-only RAG.
Tests the complete pipeline: query -> retrieval -> generation with citations.
"""
import sys
import os

# Platform-specific fixes
if sys.platform == "win32":
    os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
    os.environ["PYTHONIOENCODING"] = "utf-8"

import transformers.utils.import_utils
if hasattr(transformers.utils.import_utils, 'check_torch_load_is_safe'):
    transformers.utils.import_utils.check_torch_load_is_safe = lambda: None
import transformers.modeling_utils
if hasattr(transformers.modeling_utils, 'check_torch_load_is_safe'):
    transformers.modeling_utils.check_torch_load_is_safe = lambda: None

import logging
import pickle
import json
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.ingestion import DoclingParser, TextChunker, merge_small_elements, ElementType
from src.indexing import VLMSummarizer, BGEEmbedder, VectorStore, build_index
from src.retrieval import MultiVectorRetriever
from src.generation import GroundedGenerator
from src.baseline import TextOnlyRAG

# Logging
log_file = project_root / "docling_output" / "test_qa.log"
log_file.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(str(log_file), encoding="utf-8", mode="w"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


def load_or_build_elements():
    """Load cached elements or build from scratch."""
    cache_path = project_root / "docling_output" / "cached_elements.pkl"

    if cache_path.exists():
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    # Parse
    parser = DoclingParser(
        output_dir=str(project_root / "docling_output"),
        extract_images=True, extract_tables=True,
        generate_page_images=True, images_scale=2.0,
    )
    pdf_path = list((project_root / "data").glob("*.pdf"))[0]
    elements, _ = parser.parse(str(pdf_path))
    elements = merge_small_elements(elements, min_size=80)
    chunker = TextChunker(max_chunk_size=1500, chunk_overlap=200, min_chunk_size=80)
    elements = chunker.chunk_elements(elements)

    # Summarize
    summarizer = VLMSummarizer(model="glm-4.6v")
    elements = summarizer.summarize_elements(elements)

    with open(cache_path, "wb") as f:
        pickle.dump(elements, f)

    return elements


def main():
    logger.info("=" * 65)
    logger.info("  MULTIMODAL RAG vs TEXT-ONLY RAG - Comparison Test")
    logger.info("=" * 65)

    # Load elements
    logger.info("\n[1] Loading document elements...")
    elements = load_or_build_elements()
    type_counts = {}
    for e in elements:
        type_counts[e.type.value] = type_counts.get(e.type.value, 0) + 1
    logger.info(f"    Elements: {type_counts}")

    # Initialize shared components
    logger.info("\n[2] Initializing models...")
    embedder = BGEEmbedder(model_name="BAAI/bge-m3", device="auto", use_fp16=True)
    generator = GroundedGenerator(model="glm-4.6v", temperature=0.1)

    # Build Multimodal RAG index
    logger.info("\n[3] Building Multimodal RAG index...")
    persist_dir = str(Path.home() / "multimodalqa_vectorstore")
    mm_store = build_index(elements, embedder, persist_dir=persist_dir)
    mm_retriever = MultiVectorRetriever(embedder=embedder, vector_store=mm_store, top_k=5)
    logger.info(f"    Multimodal index: {mm_store.size} vectors")

    # Build Text-only RAG index
    logger.info("\n[4] Building Text-only RAG baseline index...")
    text_rag = TextOnlyRAG(embedder=embedder, generator=generator)
    text_store = text_rag.build_index(elements)
    logger.info(f"    Text-only index: {text_store.size} vectors")

    # Test queries - designed to test different modalities
    test_queries = [
        {
            "question": "K-Means和GMM哪种方法的聚类准确率更高？具体数据是多少？",
            "type": "table_query",
            "description": "需要表格数据支撑",
        },
        {
            "question": "训练过程中，Random初始化和K-Means++初始化的损失收敛速度有什么区别？",
            "type": "figure_query",
            "description": "需要图表视觉信息",
        },
        {
            "question": "不同协方差结构对GMM性能有什么影响？",
            "type": "text_query",
            "description": "主要依赖文本描述",
        },
    ]

    # Run comparison
    logger.info("\n" + "=" * 65)
    logger.info("  COMPARISON RESULTS")
    logger.info("=" * 65)

    results = []

    for i, tq in enumerate(test_queries, 1):
        question = tq["question"]
        logger.info(f"\n{'─' * 65}")
        logger.info(f"  Q{i} [{tq['type']}]: {question}")
        logger.info(f"  (设计意图: {tq['description']})")
        logger.info(f"{'─' * 65}")

        # Multimodal RAG answer
        logger.info(f"\n  >>> 多模态 RAG 回答:")
        mm_context = mm_retriever.retrieve_with_context(question, max_images=2)
        mm_result = generator.generate(question, mm_context)
        logger.info(f"      模式: {mm_result['mode']}")
        logger.info(f"      引用页: {mm_result['referenced_pages']}")
        logger.info(f"      答案: {mm_result['answer'][:300]}...")
        if mm_result['citations']:
            logger.info(f"      溯源标签: {mm_result['citations']}")

        # Text-only RAG answer
        logger.info(f"\n  >>> 纯文本 RAG 回答:")
        text_result = text_rag.query(question)
        logger.info(f"      模式: {text_result['mode']}")
        logger.info(f"      引用页: {text_result['referenced_pages']}")
        logger.info(f"      答案: {text_result['answer'][:300]}...")
        if text_result['citations']:
            logger.info(f"      溯源标签: {text_result['citations']}")

        results.append({
            "question": question,
            "type": tq["type"],
            "multimodal": {
                "answer": mm_result["answer"],
                "mode": mm_result["mode"],
                "citations": mm_result["citations"],
                "pages": mm_result["referenced_pages"],
            },
            "text_only": {
                "answer": text_result["answer"],
                "mode": text_result["mode"],
                "citations": text_result["citations"],
                "pages": text_result["referenced_pages"],
            },
        })

    # Save results
    output_path = project_root / "docling_output" / "qa_comparison_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    logger.info(f"\n{'=' * 65}")
    logger.info(f"  TEST COMPLETE - Results saved to {output_path.name}")
    logger.info(f"{'=' * 65}")


if __name__ == "__main__":
    main()
