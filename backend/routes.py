"""
API route definitions.
Thin layer: validates input, calls services, formats response.
"""
import uuid
import shutil
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel

from backend.services import (
    process_document_async,
    chat_stream,
    get_document_list,
    get_document_info,
    delete_document,
    DOCUMENTS_DIR,
)
from backend.progress import ProgressTracker

router = APIRouter(prefix="/api")


# ─── Request/Response Models ─────────────────────────────────

class UploadResponse(BaseModel):
    task_id: str
    document_id: str


class ChatRequest(BaseModel):
    document_id: str
    question: str
    mode: str = "multimodal"  # "multimodal" or "text_only"
    history: list = []


# ─── Document Upload & Processing ────────────────────────────

@router.post("/upload", response_model=UploadResponse)
async def upload_pdf(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    """Upload a PDF file and start background processing."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported")

    task_id = str(uuid.uuid4())[:8]
    doc_id = str(uuid.uuid4())[:8]

    # Save uploaded file (keep original filename for display)
    doc_dir = DOCUMENTS_DIR / doc_id
    doc_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = doc_dir / "source.pdf"

    with open(pdf_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Store original filename in metadata immediately
    from backend.services import document_metadata
    document_metadata[doc_id] = {
        "filename": file.filename,  # Use original upload filename
        "status": "processing",
        "page_count": 0,
    }

    # Initialize progress tracker
    ProgressTracker(task_id)

    # Start background processing
    background_tasks.add_task(process_document_async, task_id, doc_id, str(pdf_path))

    return UploadResponse(task_id=task_id, document_id=doc_id)


@router.get("/upload/{task_id}/progress")
async def upload_progress(task_id: str):
    """SSE stream: sends parsing progress events until done."""
    if not ProgressTracker.exists(task_id):
        raise HTTPException(404, "Task not found")

    tracker = ProgressTracker.get(task_id)
    return StreamingResponse(
        tracker.stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ─── Document Management ─────────────────────────────────────

@router.get("/documents")
async def list_documents():
    """Return all uploaded documents."""
    return get_document_list()


@router.post("/documents/{doc_id}/process")
async def process_existing_document(doc_id: str, background_tasks: BackgroundTasks):
    """
    Trigger processing for a pre-registered (pending) document.
    Used for PDFs in data/ folder that were auto-detected on startup.
    """
    from backend.services import document_metadata, DOCUMENTS_DIR

    if doc_id not in document_metadata:
        raise HTTPException(404, "Document not found")

    meta = document_metadata[doc_id]
    if meta.get("status") == "ready":
        return {"message": "Already processed", "task_id": None}

    # Find the PDF file
    pdf_path = DOCUMENTS_DIR / doc_id / "source.pdf"
    if not pdf_path.exists():
        raise HTTPException(404, "PDF file not found on disk")

    # Start processing
    task_id = str(uuid.uuid4())[:8]
    ProgressTracker(task_id)
    document_metadata[doc_id]["status"] = "processing"

    background_tasks.add_task(process_document_async, task_id, doc_id, str(pdf_path))

    return {"task_id": task_id, "document_id": doc_id}


@router.get("/documents/{doc_id}")
async def get_document(doc_id: str):
    """Return single document metadata."""
    info = get_document_info(doc_id)
    if not info:
        raise HTTPException(404, "Document not found")
    return info


@router.delete("/documents/{doc_id}")
async def remove_document(doc_id: str):
    """Delete a document and all its data."""
    if not delete_document(doc_id):
        raise HTTPException(404, "Document not found")
    return {"status": "deleted"}


@router.get("/documents/{doc_id}/images/{img_name}")
async def get_image(doc_id: str, img_name: str):
    """Serve an extracted figure image."""
    img_path = DOCUMENTS_DIR / doc_id / "images" / img_name
    if not img_path.exists():
        raise HTTPException(404, "Image not found")
    return FileResponse(str(img_path), media_type="image/png")


@router.get("/documents/{doc_id}/pages/{page_num}")
async def get_page_image(doc_id: str, page_num: int):
    """Serve a rendered page image (for citation preview)."""
    # Try common naming patterns
    for pattern in [f"page_{page_num}.png", f"page_{page_num:02d}.png"]:
        page_path = DOCUMENTS_DIR / doc_id / "pages" / pattern
        if page_path.exists():
            return FileResponse(str(page_path), media_type="image/png")

    raise HTTPException(404, f"Page {page_num} image not found")

    return FileResponse(str(img_path), media_type="image/png")


# ─── Chat (Q&A) ──────────────────────────────────────────────

@router.post("/chat")
async def chat(request: ChatRequest):
    """
    Direct JSON response (not SSE).
    Returns complete answer with retrieval info and citations.
    """
    from backend.services import chat_direct
    result = await chat_direct(
        doc_id=request.document_id,
        question=request.question,
        mode=request.mode,
        history=request.history,
    )
    return result
