# Baseline Module
"""
Pure text RAG baseline for comparison.
Strips all visual elements, uses only extracted text for retrieval and generation.
"""
from .text_only_rag import TextOnlyRAG

__all__ = ["TextOnlyRAG"]
