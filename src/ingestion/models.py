"""
Data models for document elements.
Defines the unified DocumentElement structure used across the pipeline.
"""
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
from pathlib import Path


class ElementType(Enum):
    """Type of document element."""
    TEXT = "text"
    TABLE = "table"
    FIGURE = "figure"
    PAGE_IMAGE = "page_image"


@dataclass
class BoundingBox:
    """Bounding box coordinates (normalized 0-1 or pixel-based)."""
    x0: float
    y0: float
    x1: float
    y1: float


@dataclass
class DocumentElement:
    """
    Unified representation of a document element.
    This is the core data structure passed through the entire pipeline.
    """
    # Unique identifier
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    # Element type
    type: ElementType = ElementType.TEXT

    # Source document info
    document_id: str = ""
    document_name: str = ""

    # Location info
    page_number: int = 0
    bbox: Optional[BoundingBox] = None

    # Content
    text_content: str = ""            # For TEXT: the chunk text; For TABLE: markdown representation
    image_path: Optional[str] = None  # Path to saved image file (for FIGURE/TABLE/PAGE_IMAGE)

    # Metadata
    heading_context: str = ""         # Parent heading hierarchy (e.g., "Section 1 > Subsection 1.1")
    caption: str = ""                 # Caption for figures/tables
    inferred_label: str = ""          # Human-readable label (e.g., "Figure 3", "Table 2")

    # For retrieval
    summary: str = ""                 # VLM-generated summary (filled later by indexing module)
    embedding_vector: Optional[list] = None  # Filled by embedding module

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "id": self.id,
            "type": self.type.value,
            "document_id": self.document_id,
            "document_name": self.document_name,
            "page_number": self.page_number,
            "bbox": vars(self.bbox) if self.bbox else None,
            "text_content": self.text_content[:200] + "..." if len(self.text_content) > 200 else self.text_content,
            "image_path": self.image_path,
            "heading_context": self.heading_context,
            "caption": self.caption,
            "inferred_label": self.inferred_label,
            "summary": self.summary,
        }

    def __repr__(self):
        content_preview = self.text_content[:50] + "..." if len(self.text_content) > 50 else self.text_content
        return (
            f"DocumentElement(type={self.type.value}, page={self.page_number}, "
            f"label='{self.inferred_label}', content='{content_preview}')"
        )
