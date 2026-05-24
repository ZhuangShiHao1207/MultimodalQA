"""
Pure text RAG baseline.
Strips all visual elements (images), uses only extracted text for retrieval and generation.
Used for ablation study: comparing multimodal RAG vs text-only RAG.
"""
import logging
from typing import List

from src.ingestion.models import DocumentElement, ElementType
from src.indexing.embedder import BGEEmbedder, VectorStore, build_index
from src.retrieval.retriever import MultiVectorRetriever
from src.generation.generator import GroundedGenerator

logger = logging.getLogger(__name__)


class TextOnlyRAG:
    """
    Text-only RAG baseline that deliberately removes all visual information.
    This creates the ablation condition for comparing against multimodal RAG.
    """

    def __init__(self, embedder: BGEEmbedder, generator: GroundedGenerator):
        self.embedder = embedder
        self.generator = generator
        self.vector_store = None
        self.retriever = None

    def build_index(self, elements: List[DocumentElement]) -> VectorStore:
        """
        Build a text-only index by filtering out visual elements.
        Tables are included as text (markdown), but figures are completely removed.
        """
        # Filter: keep only TEXT and TABLE elements (no figures, no page images)
        # For tables: use text_content (markdown) but clear image_path and summary
        text_elements = []
        for elem in elements:
            if elem.type == ElementType.TEXT:
                text_elements.append(elem)
            elif elem.type == ElementType.TABLE:
                # Keep table as markdown text, but strip VLM summary
                # This simulates "OCR-only" extraction
                stripped = DocumentElement(
                    id=elem.id,
                    type=elem.type,
                    document_id=elem.document_id,
                    document_name=elem.document_name,
                    page_number=elem.page_number,
                    bbox=elem.bbox,
                    text_content=elem.text_content,  # Keep markdown
                    image_path=None,                  # Strip image
                    caption=elem.caption,
                    inferred_label=elem.inferred_label,
                    summary="",                       # Strip VLM summary
                    heading_context=elem.heading_context,
                )
                text_elements.append(stripped)
            # FIGURE and PAGE_IMAGE elements are completely dropped

        logger.info(
            f"TextOnlyRAG: {len(text_elements)} text elements "
            f"(dropped {len(elements) - len(text_elements)} visual elements)"
        )

        # Build index using only text content (no summaries)
        self.vector_store = build_index(text_elements, self.embedder)
        self.retriever = MultiVectorRetriever(
            embedder=self.embedder,
            vector_store=self.vector_store,
            top_k=5,
            score_threshold=0.3,
        )
        return self.vector_store

    def query(self, question: str) -> dict:
        """
        Answer a question using text-only RAG (no images sent to VLM).

        Args:
            question: Natural language question

        Returns:
            Same format as GroundedGenerator.generate() output
        """
        if not self.retriever:
            raise RuntimeError("Must call build_index() first")

        # Retrieve (will only find text and table-text elements)
        context = self.retriever.retrieve_with_context(question, max_images=0)

        # Force text-only generation (no images in context)
        context["image_contexts"] = []

        # Generate answer
        result = self.generator.generate(question, context)
        result["mode"] = "text_only_baseline"
        return result
