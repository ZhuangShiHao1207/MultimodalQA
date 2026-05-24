# Indexing Module
"""
Handles summary generation (via VLM API) and vector embedding (BGE-M3).
Builds the dual-layer storage: FAISS vector index + DocStore.
"""
from .summarizer import VLMSummarizer
from .embedder import BGEEmbedder, VectorStore, build_index

__all__ = [
    "VLMSummarizer",
    "BGEEmbedder",
    "VectorStore",
    "build_index",
]
