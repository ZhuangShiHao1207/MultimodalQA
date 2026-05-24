"""
Test script for document parsing pipeline.
Parses the test PDF and reports extraction results.
"""
import sys
import os

# Platform-specific fixes
if sys.platform == "win32":
    os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"   # Windows symlink privilege issue
    os.environ["PYTHONIOENCODING"] = "utf-8"        # Windows console encoding

import json
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.ingestion import DoclingParser, TextChunker, merge_small_elements, ElementType

# Configure logging - output to file to avoid Windows encoding issues
log_file = project_root / "docling_output" / "test_parsing.log"
log_file.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(str(log_file), encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger(__name__)


def main():
    # Find test PDF
    data_dir = project_root / "data"
    pdf_files = list(data_dir.glob("*.pdf"))

    if not pdf_files:
        logger.error("No PDF files found in data/ directory!")
        sys.exit(1)

    pdf_path = pdf_files[0]
    logger.info(f"=" * 60)
    logger.info(f"Testing document parsing pipeline")
    logger.info(f"Input: {pdf_path.name}")
    logger.info(f"=" * 60)

    # Step 1: Parse document
    logger.info("\n[Step 1] Parsing with Docling...")
    parser = DoclingParser(
        output_dir=str(project_root / "docling_output"),
        extract_images=True,
        extract_tables=True,
        generate_page_images=True,
        images_scale=2.0,
        table_mode="accurate",
    )

    elements, markdown = parser.parse(str(pdf_path))

    # Step 2: Merge small elements
    logger.info("\n[Step 2] Merging small text elements...")
    elements = merge_small_elements(elements, min_size=80, max_merged_size=1500)

    # Step 3: Chunk text elements
    logger.info("\n[Step 3] Chunking text elements...")
    chunker = TextChunker(max_chunk_size=1500, chunk_overlap=200, min_chunk_size=80)
    chunked_elements = chunker.chunk_elements(elements)

    # Report results
    logger.info("\n" + "=" * 60)
    logger.info("PARSING RESULTS SUMMARY")
    logger.info("=" * 60)

    # Count by type
    type_counts = {}
    for elem in chunked_elements:
        t = elem.type.value
        type_counts[t] = type_counts.get(t, 0) + 1

    logger.info(f"\nTotal elements after chunking: {len(chunked_elements)}")
    for t, count in sorted(type_counts.items()):
        logger.info(f"  {t}: {count}")

    # Show text chunks
    text_chunks = [e for e in chunked_elements if e.type == ElementType.TEXT]
    logger.info(f"\n--- Text Chunks (showing first 5) ---")
    for i, chunk in enumerate(text_chunks[:5]):
        logger.info(f"\n  [{i+1}] Page {chunk.page_number} | Heading: '{chunk.heading_context}'")
        preview = chunk.text_content[:150].replace('\n', ' ')
        logger.info(f"      Content: {preview}...")
        logger.info(f"      Length: {len(chunk.text_content)} chars")

    # Show tables
    tables = [e for e in chunked_elements if e.type == ElementType.TABLE]
    logger.info(f"\n--- Tables ({len(tables)} total) ---")
    for i, tbl in enumerate(tables[:3]):
        logger.info(f"\n  [{i+1}] {tbl.inferred_label} | Page {tbl.page_number}")
        if tbl.caption:
            logger.info(f"      Caption: {tbl.caption}")
        preview = tbl.text_content[:200].replace('\n', ' | ')
        logger.info(f"      Content: {preview}")

    # Show figures
    figures = [e for e in chunked_elements if e.type == ElementType.FIGURE]
    logger.info(f"\n--- Figures ({len(figures)} total) ---")
    for i, fig in enumerate(figures[:5]):
        logger.info(f"\n  [{i+1}] {fig.inferred_label} | Page {fig.page_number}")
        if fig.caption:
            logger.info(f"      Caption: {fig.caption}")
        logger.info(f"      Image: {fig.image_path}")

    # Show page images
    pages = [e for e in chunked_elements if e.type == ElementType.PAGE_IMAGE]
    logger.info(f"\n--- Page Images ({len(pages)} total) ---")
    for p in pages[:3]:
        logger.info(f"  Page {p.page_number}: {p.image_path}")

    # Save full markdown preview
    logger.info(f"\n--- Markdown Export (first 500 chars) ---")
    logger.info(markdown[:500])

    # Save structured output for inspection
    output_path = project_root / "docling_output" / "parsed_elements.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            [e.to_dict() for e in chunked_elements],
            f, ensure_ascii=False, indent=2
        )
    logger.info(f"\nFull element list saved to: {output_path}")

    logger.info("\n" + "=" * 60)
    logger.info("PARSING TEST COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
