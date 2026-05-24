"""
Text chunking module.
Splits extracted text elements into retrieval-friendly chunks.
"""
import logging
from typing import List

from .models import DocumentElement, ElementType

logger = logging.getLogger(__name__)


class TextChunker:
    """
    Split text elements into chunks suitable for embedding and retrieval.
    Supports heading-based and fixed-size chunking strategies.
    """

    def __init__(
        self,
        max_chunk_size: int = 1500,
        chunk_overlap: int = 200,
        min_chunk_size: int = 100,
    ):
        """
        Args:
            max_chunk_size: Maximum characters per chunk
            chunk_overlap: Overlap between consecutive chunks
            min_chunk_size: Minimum characters for a valid chunk
        """
        self.max_chunk_size = max_chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size

    def chunk_elements(self, elements: List[DocumentElement]) -> List[DocumentElement]:
        """
        Process all elements: chunk text elements, pass through non-text elements.

        Args:
            elements: List of DocumentElements from the parser

        Returns:
            List of DocumentElements with text elements chunked
        """
        chunked = []

        for elem in elements:
            if elem.type == ElementType.TEXT:
                # Split long text elements
                chunks = self._split_text_element(elem)
                chunked.extend(chunks)
            else:
                # Tables, figures, page images pass through unchanged
                chunked.append(elem)

        logger.info(
            f"Chunking complete: {len(elements)} elements -> {len(chunked)} chunks"
        )
        return chunked

    def _split_text_element(self, element: DocumentElement) -> List[DocumentElement]:
        """
        Split a single text element into smaller chunks if it exceeds max_chunk_size.
        Preserves metadata (page, heading context, etc.) across chunks.
        """
        text = element.text_content

        # If text is short enough, return as-is
        if len(text) <= self.max_chunk_size:
            if len(text) >= self.min_chunk_size:
                return [element]
            else:
                return []  # Too short, skip

        # Split into chunks with overlap
        chunks = []
        start = 0
        chunk_index = 0

        while start < len(text):
            end = start + self.max_chunk_size

            # Try to break at a natural boundary (sentence end, paragraph)
            if end < len(text):
                # Look for sentence boundaries near the end
                breakpoints = [
                    text.rfind("。", start + self.max_chunk_size // 2, end),  # Chinese period
                    text.rfind(". ", start + self.max_chunk_size // 2, end),  # English period
                    text.rfind("\n", start + self.max_chunk_size // 2, end),  # Newline
                    text.rfind("；", start + self.max_chunk_size // 2, end),  # Chinese semicolon
                    text.rfind("; ", start + self.max_chunk_size // 2, end),  # English semicolon
                ]
                # Use the latest valid breakpoint
                best_break = max(bp for bp in breakpoints if bp > 0) if any(bp > 0 for bp in breakpoints) else -1

                if best_break > 0:
                    end = best_break + 1

            chunk_text = text[start:end].strip()

            if len(chunk_text) >= self.min_chunk_size:
                # Create a new DocumentElement for this chunk
                chunk_elem = DocumentElement(
                    type=ElementType.TEXT,
                    document_id=element.document_id,
                    document_name=element.document_name,
                    page_number=element.page_number,
                    bbox=element.bbox,
                    text_content=chunk_text,
                    heading_context=element.heading_context,
                    inferred_label=f"Chunk {chunk_index + 1}" if chunk_index > 0 else "",
                )
                chunks.append(chunk_elem)
                chunk_index += 1

            # Move start with overlap
            start = end - self.chunk_overlap if end < len(text) else len(text)

        return chunks if chunks else [element]  # Return original if splitting produced nothing


def merge_small_elements(
    elements: List[DocumentElement],
    min_size: int = 100,
    max_merged_size: int = 1500,
) -> List[DocumentElement]:
    """
    Merge consecutive small text elements from the same page into larger chunks.
    This helps when Docling splits text too aggressively.

    Args:
        elements: List of DocumentElements
        min_size: Elements smaller than this may be merged
        max_merged_size: Don't merge beyond this size

    Returns:
        List with small elements merged where appropriate
    """
    if not elements:
        return elements

    merged = []
    buffer: List[DocumentElement] = []

    for elem in elements:
        # Only merge TEXT elements
        if elem.type != ElementType.TEXT:
            # Flush buffer first
            if buffer:
                merged.append(_merge_buffer(buffer))
                buffer = []
            merged.append(elem)
            continue

        # If element is large enough on its own, flush buffer and add it
        if len(elem.text_content) >= min_size and not buffer:
            merged.append(elem)
            continue

        # Try to merge small elements
        if buffer:
            # Check if this element is on the same page and under same heading
            last = buffer[-1]
            total_len = sum(len(e.text_content) for e in buffer) + len(elem.text_content)

            if (
                last.page_number == elem.page_number
                and last.heading_context == elem.heading_context
                and total_len <= max_merged_size
            ):
                buffer.append(elem)
            else:
                # Flush buffer and start new
                merged.append(_merge_buffer(buffer))
                buffer = [elem]
        else:
            buffer = [elem]

    # Flush remaining buffer
    if buffer:
        merged.append(_merge_buffer(buffer))

    return merged


def _merge_buffer(buffer: List[DocumentElement]) -> DocumentElement:
    """Merge a list of text elements into a single element."""
    if len(buffer) == 1:
        return buffer[0]

    merged_text = "\n".join(e.text_content for e in buffer)
    return DocumentElement(
        type=ElementType.TEXT,
        document_id=buffer[0].document_id,
        document_name=buffer[0].document_name,
        page_number=buffer[0].page_number,
        bbox=buffer[0].bbox,
        text_content=merged_text,
        heading_context=buffer[0].heading_context,
    )
