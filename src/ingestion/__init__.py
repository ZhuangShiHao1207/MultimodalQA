# Document Ingestion Module
"""
Handles PDF parsing, layout analysis, and content extraction.
Uses IBM Docling for structured document decomposition.
"""
from .models import DocumentElement, ElementType, BoundingBox
from .docling_parser import DoclingParser
from .chunker import TextChunker, merge_small_elements

__all__ = [
    "DocumentElement",
    "ElementType",
    "BoundingBox",
    "DoclingParser",
    "TextChunker",
    "merge_small_elements",
]
