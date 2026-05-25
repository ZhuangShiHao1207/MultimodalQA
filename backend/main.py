"""
FastAPI application entry point.
Sets up CORS, routes, and static file serving.
"""
import sys
import logging
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Configure logging - show all backend activity
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
# Reduce noise from httpx/httpcore
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

# Ensure project root is in path for src/ imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.routes import router

app = FastAPI(
    title="MultimodalQA - Document Intelligence Assistant",
    version="0.1.0",
    description="Multimodal RAG system for document understanding and QA",
)

# CORS: allow Vue dev server (port 5173) to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API routes
app.include_router(router)

# Serve extracted document images as static files
documents_dir = Path(__file__).parent / "documents"
documents_dir.mkdir(exist_ok=True)
app.mount("/static/documents", StaticFiles(directory=str(documents_dir)), name="documents")


@app.get("/")
async def root():
    return {"message": "MultimodalQA API is running", "docs": "/docs"}
