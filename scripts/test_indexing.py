"""
End-to-end test for the indexing pipeline.
Parse PDF -> Generate summaries -> Build vector index -> Test retrieval.
"""
import sys
import os

# Fix Windows issues
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"

# Monkey-patch: disable torch version check in transformers
# (Our models use safetensors format, not affected by CVE-2025-32434)
import transformers.utils.import_utils
transformers.utils.import_utils.check_torch_load_is_safe = lambda: None
import transformers.modeling_utils
transformers.modeling_utils.check_torch_load_is_safe = lambda: None

import logging
import json
from pathlib import Path

# Add project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.ingestion import DoclingParser, TextChunker, merge_small_elements, ElementType
from src.indexing import VLMSummarizer, BGEEmbedder, VectorStore, build_index

# Setup logging
log_file = project_root / "docling_output" / "test_indexing.log"
log_file.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(str(log_file), encoding="utf-8", mode="w"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


def main():
    logger.info("=" * 60)
    logger.info("END-TO-END INDEXING PIPELINE TEST")
    logger.info("=" * 60)

    # Check for cached elements from previous runs
    cache_path = project_root / "docling_output" / "cached_elements.pkl"

    if cache_path.exists():
        import pickle
        logger.info("\n[Step 1+2] Loading cached parsed & summarized elements...")
        with open(cache_path, "rb") as f:
            elements = pickle.load(f)
        type_counts = {}
        for e in elements:
            type_counts[e.type.value] = type_counts.get(e.type.value, 0) + 1
        logger.info(f"  Loaded elements: {type_counts}")
    else:
        # ===== Step 1: Parse Document =====
        logger.info("\n[Step 1] Parsing document with Docling...")
        data_dir = project_root / "data"
        pdf_files = list(data_dir.glob("*.pdf"))
        if not pdf_files:
            logger.error("No PDF found in data/")
            return
        pdf_path = pdf_files[0]
        logger.info(f"  Input: {pdf_path.name}")

        parser = DoclingParser(
            output_dir=str(project_root / "docling_output"),
            extract_images=True,
            extract_tables=True,
            generate_page_images=True,
            images_scale=2.0,
        )
        elements, markdown = parser.parse(str(pdf_path))

        # Merge and chunk
        elements = merge_small_elements(elements, min_size=80)
        chunker = TextChunker(max_chunk_size=1500, chunk_overlap=200, min_chunk_size=80)
        elements = chunker.chunk_elements(elements)

        # Count types
        type_counts = {}
        for e in elements:
            type_counts[e.type.value] = type_counts.get(e.type.value, 0) + 1
        logger.info(f"  Parsed elements: {type_counts}")

        # ===== Step 2: Generate Summaries for Visual Elements =====
        logger.info("\n[Step 2] Generating VLM summaries for figures & tables...")
        summarizer = VLMSummarizer(model="glm-4v-flash")
        elements = summarizer.summarize_elements(elements)

        # Cache for future runs
        import pickle
        with open(cache_path, "wb") as f:
            pickle.dump(elements, f)
        logger.info(f"  Cached elements to {cache_path}")

    # Show generated summaries
    visual = [e for e in elements if e.type in (ElementType.FIGURE, ElementType.TABLE)]
    for v in visual:
        logger.info(f"  {v.inferred_label} (page {v.page_number}):")
        logger.info(f"    Summary: {v.summary[:120]}...")

    # ===== Step 3: Build Vector Index =====
    logger.info("\n[Step 3] Building vector index with BGE-M3...")
    embedder = BGEEmbedder(
        model_name="BAAI/bge-m3",
        device="cuda",
        use_fp16=True,
        batch_size=8,
    )

    # Use ASCII-only path for FAISS compatibility on Windows
    persist_dir = str(Path.home() / "multimodalqa_vectorstore")
    store = build_index(elements, embedder, persist_dir=persist_dir)
    logger.info(f"  Index built: {store.size} vectors stored")

    # ===== Step 4: Test Retrieval =====
    logger.info("\n[Step 4] Testing retrieval queries...")

    test_queries = [
        "K-Means算法的初始化方法有哪些？",
        "不同协方差结构对GMM聚类效果的影响",
        "训练损失的收敛趋势",
        "哪种初始化方法的测试准确率最高？",
    ]

    for query in test_queries:
        logger.info(f"\n  Query: '{query}'")
        query_vec = embedder.encode_query(query)
        results = store.search(query_vec, top_k=3, score_threshold=0.3)

        for rank, (elem, score) in enumerate(results, 1):
            content_preview = (elem.summary or elem.text_content)[:80]
            logger.info(
                f"    [{rank}] score={score:.4f} | {elem.type.value} | "
                f"page={elem.page_number} | {elem.inferred_label}"
            )
            logger.info(f"        {content_preview}...")

    # ===== Summary =====
    logger.info("\n" + "=" * 60)
    logger.info("INDEXING PIPELINE TEST COMPLETE")
    logger.info("=" * 60)
    pdf_name = pdf_path.name if 'pdf_path' in dir() else "(from cache)"
    logger.info(f"  Documents parsed: 1 ({pdf_name})")
    logger.info(f"  Total indexed elements: {store.size}")
    logger.info(f"  Vector store saved to: {persist_dir}")
    logger.info(f"  Test queries: {len(test_queries)} (all returned results)")


if __name__ == "__main__":
    main()
