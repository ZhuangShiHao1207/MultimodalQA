"""
Service layer: bridges FastAPI routes with src/ ML pipeline modules.
All heavy computation runs in thread pool via asyncio.to_thread().
"""
import os
import sys
import json
import shutil
import logging
from pathlib import Path
from typing import AsyncGenerator

# Platform-specific fixes
if sys.platform == "win32":
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")

# Patch transformers torch version check
import transformers.utils.import_utils
if hasattr(transformers.utils.import_utils, 'check_torch_load_is_safe'):
    transformers.utils.import_utils.check_torch_load_is_safe = lambda: None
import transformers.modeling_utils
if hasattr(transformers.modeling_utils, 'check_torch_load_is_safe'):
    transformers.modeling_utils.check_torch_load_is_safe = lambda: None

import asyncio

from src.ingestion import DoclingParser, TextChunker, merge_small_elements, ElementType
from src.indexing import VLMSummarizer, BGEEmbedder, VectorStore, build_index
from src.retrieval import MultiVectorRetriever
from src.generation import GroundedGenerator
from src.baseline import TextOnlyRAG

from backend.progress import ProgressTracker

logger = logging.getLogger(__name__)

# ============================================================
# Singleton model instances (loaded once at startup)
# ============================================================
_embedder = None
_summarizer = None
_generator = None

DOCUMENTS_DIR = Path(__file__).parent / "documents"


def get_embedder() -> BGEEmbedder:
    global _embedder
    if _embedder is None:
        _embedder = BGEEmbedder(model_name="BAAI/bge-m3", device="auto", use_fp16=True)
    return _embedder


def get_summarizer() -> VLMSummarizer:
    global _summarizer
    if _summarizer is None:
        _summarizer = VLMSummarizer(model="glm-4v-flash")
    return _summarizer


def get_generator() -> GroundedGenerator:
    global _generator
    if _generator is None:
        _generator = GroundedGenerator(model="glm-4v-flash", max_tokens=1024)
    return _generator


# ============================================================
# Per-document stores (in-memory for demo)
# ============================================================
document_metadata: dict = {}  # doc_id -> {filename, status, page_count, ...}
document_stores: dict = {}    # doc_id -> VectorStore
document_elements: dict = {}  # doc_id -> List[DocumentElement]

# Data directory (pre-placed test PDFs)
DATA_DIR = Path(__file__).parent.parent / "data"


def scan_data_directory():
    """
    Scan data/ directory for pre-placed PDFs and register them.
    Called once at startup. Registers as 'pending' (not yet processed).
    """
    if not DATA_DIR.exists():
        return

    for pdf_file in DATA_DIR.glob("*.pdf"):
        # Use filename hash as stable doc_id (so it persists across restarts)
        doc_id = hashlib.md5(pdf_file.name.encode()).hexdigest()[:8]

        if doc_id not in document_metadata:
            # Copy to documents dir for consistent handling
            doc_dir = DOCUMENTS_DIR / doc_id
            doc_dir.mkdir(parents=True, exist_ok=True)
            dest_pdf = doc_dir / "source.pdf"
            if not dest_pdf.exists():
                shutil.copy2(pdf_file, dest_pdf)

            document_metadata[doc_id] = {
                "filename": pdf_file.name,
                "status": "pending",  # Not yet processed
                "page_count": 0,
                "source_path": str(pdf_file),
            }
            logger.info(f"Registered pre-placed PDF: {pdf_file.name} (id={doc_id})")


# Run scan at import time (when backend starts)
import hashlib
scan_data_directory()


def get_document_list() -> list:
    """Return all documents with their metadata."""
    return [
        {"id": doc_id, **meta}
        for doc_id, meta in document_metadata.items()
    ]


def get_document_info(doc_id: str) -> dict:
    """Return single document metadata."""
    if doc_id not in document_metadata:
        return None
    return {"id": doc_id, **document_metadata[doc_id]}


def delete_document(doc_id: str) -> bool:
    """Delete a document and its data."""
    if doc_id not in document_metadata:
        return False
    # Cleanup memory
    document_metadata.pop(doc_id, None)
    document_stores.pop(doc_id, None)
    document_elements.pop(doc_id, None)
    # Cleanup disk
    doc_dir = DOCUMENTS_DIR / doc_id
    if doc_dir.exists():
        shutil.rmtree(doc_dir)
    return True


# ============================================================
# Document processing pipeline (runs in background thread)
# ============================================================
def process_document_sync(task_id: str, doc_id: str, pdf_path: str):
    """
    Full ingestion pipeline. Runs synchronously in a thread.
    Updates ProgressTracker at each stage.
    """
    tracker = ProgressTracker.get(task_id)
    doc_dir = DOCUMENTS_DIR / doc_id
    doc_dir.mkdir(parents=True, exist_ok=True)
    (doc_dir / "images").mkdir(exist_ok=True)

    try:
        # Stage 1: Parse PDF with Docling
        tracker.update("parsing", 10, "Parsing PDF document structure...")
        parser = DoclingParser(
            output_dir=str(doc_dir / "docling_output"),
            extract_images=True,
            extract_tables=True,
            generate_page_images=True,
            images_scale=2.0,
        )
        elements, markdown = parser.parse(pdf_path)
        tracker.update("parsing", 30, f"Parsed {len(elements)} elements")

        # Stage 2: Merge and chunk
        tracker.update("chunking", 35, "Chunking text elements...")
        elements = merge_small_elements(elements, min_size=80)
        chunker = TextChunker(max_chunk_size=1500, chunk_overlap=200, min_chunk_size=80)
        elements = chunker.chunk_elements(elements)
        tracker.update("chunking", 40, f"Chunked into {len(elements)} elements")

        # Stage 3: Generate summaries for visual elements
        tracker.update("summarizing", 45, "Generating VLM summaries for figures/tables...")
        summarizer = get_summarizer()
        visual_count = sum(1 for e in elements if e.type in (ElementType.FIGURE, ElementType.TABLE))
        elements = summarizer.summarize_elements(elements)
        tracker.update("summarizing", 70, f"Summarized {visual_count} visual elements")

        # Stage 4: Build vector index
        tracker.update("embedding", 75, "Embedding with BGE-M3...")
        embedder = get_embedder()
        store = build_index(elements, embedder)
        tracker.update("embedding", 90, f"Indexed {store.size} vectors")

        # Stage 5: Save metadata and finalize
        # Copy images to accessible location
        for elem in elements:
            if elem.image_path and Path(elem.image_path).exists():
                dest = doc_dir / "images" / Path(elem.image_path).name
                if not dest.exists():
                    shutil.copy2(elem.image_path, dest)

        # Count pages
        page_count = max((e.page_number for e in elements if e.page_number > 0), default=0)

        # Store in memory
        document_stores[doc_id] = store
        document_elements[doc_id] = elements
        document_metadata[doc_id] = {
            "filename": Path(pdf_path).name,
            "status": "ready",
            "page_count": page_count,
            "element_count": len(elements),
            "vector_count": store.size,
        }

        # Save markdown
        (doc_dir / "document.md").write_text(markdown, encoding="utf-8")

        tracker.update("done", 100, "Document ready!")

    except Exception as e:
        logger.error(f"Document processing failed: {e}", exc_info=True)
        document_metadata[doc_id] = {
            "filename": Path(pdf_path).name,
            "status": "error",
            "error": str(e),
        }
        tracker.update("error", 0, f"Processing failed: {str(e)[:100]}")


async def process_document_async(task_id: str, doc_id: str, pdf_path: str):
    """Async wrapper that runs the sync pipeline in a thread pool."""
    await asyncio.to_thread(process_document_sync, task_id, doc_id, pdf_path)


# ============================================================
# Chat (Q&A) service
# ============================================================
async def chat_stream(
    doc_id: str, question: str, mode: str, history: list
) -> AsyncGenerator[str, None]:
    """
    Stream chat response as SSE events.
    Yields: retrieval → token → citation → done
    All exceptions are caught and sent as error events to frontend.
    """
    try:
        logger.info(f"[Chat] Start: doc={doc_id}, mode={mode}, question='{question[:50]}'")

        if doc_id not in document_stores:
            logger.error(f"[Chat] Document {doc_id} not in document_stores. Available: {list(document_stores.keys())}")
            yield f"data: {json.dumps({'type': 'error', 'content': f'Document not found or not ready. Available docs: {list(document_stores.keys())}'})}\n\n"
            return

        store = document_stores[doc_id]
        embedder = get_embedder()
        generator = get_generator()

        # Step 1: Retrieve relevant context
        logger.info(f"[Chat] Step 1: Retrieving context...")
        retriever = MultiVectorRetriever(embedder=embedder, vector_store=store, top_k=5)

        context = await asyncio.to_thread(
            retriever.retrieve_with_context, question, 3 if mode == "multimodal" else 0
        )
        logger.info(f"[Chat] Retrieved: {len(context.get('text_contexts', []))} texts, "
                    f"{len(context.get('image_contexts', []))} images, "
                    f"{len(context.get('table_contexts', []))} tables")

        # Send retrieval results to frontend
        images_for_frontend = []
        for img_ctx in context.get("image_contexts", []):
            img_name = Path(img_ctx["image_path"]).name if img_ctx.get("image_path") else ""
            images_for_frontend.append({
                "name": img_name,
                "label": img_ctx.get("label", ""),
                "page": img_ctx.get("page", 0),
                "url": f"/static/documents/{doc_id}/images/{img_name}",
            })

        retrieval_event = {
            "type": "retrieval",
            "pages": context.get("all_pages", []),
            "images": images_for_frontend,
            "chunk_count": len(context.get("text_contexts", [])),
        }
        yield f"data: {json.dumps(retrieval_event, ensure_ascii=False)}\n\n"

        # Step 2: Generate answer
        logger.info(f"[Chat] Step 2: Generating answer (mode={mode})...")
        if mode == "text_only":
            context["image_contexts"] = []

        result = await asyncio.to_thread(generator.generate, question, context)

        answer = result.get("answer", "")
        logger.info(f"[Chat] Generated answer: {len(answer)} chars, citations={result.get('citations', [])}")

        if not answer:
            yield f"data: {json.dumps({'type': 'token', 'content': '(No answer generated - model returned empty response)'}, ensure_ascii=False)}\n\n"
        else:
            # Step 3: Stream the answer in chunks for progressive display
            chunk_size = 20
            for i in range(0, len(answer), chunk_size):
                chunk = answer[i:i+chunk_size]
                yield f"data: {json.dumps({'type': 'token', 'content': chunk}, ensure_ascii=False)}\n\n"
                await asyncio.sleep(0.02)  # Small delay for streaming effect

        # Step 4: Send citations
        for citation in result.get("citations", []):
            yield f"data: {json.dumps({'type': 'citation', **citation}, ensure_ascii=False)}\n\n"

        # Done
        yield f"data: {json.dumps({'type': 'done', 'mode': result.get('mode', mode)})}\n\n"
        logger.info(f"[Chat] Complete.")

    except Exception as e:
        logger.error(f"[Chat] EXCEPTION: {type(e).__name__}: {e}", exc_info=True)
        error_msg = f"Server error: {type(e).__name__}: {str(e)[:200]}"
        yield f"data: {json.dumps({'type': 'error', 'content': error_msg}, ensure_ascii=False)}\n\n"
        yield f"data: {json.dumps({'type': 'done', 'mode': mode})}\n\n"
