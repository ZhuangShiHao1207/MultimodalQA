"""
Multi-Vector Retriever with proxy recall.
When a summary vector is hit, the original image/table is recalled for generation.
"""
import logging
from typing import List, Tuple, Optional
from pathlib import Path

from src.ingestion.models import DocumentElement, ElementType
from src.indexing.embedder import BGEEmbedder, VectorStore

logger = logging.getLogger(__name__)


class MultiVectorRetriever:
    """
    Retriever that searches the vector store and performs proxy recall:
    - Text hits: return the text chunk directly
    - Table/Figure hits: return the element WITH its original image path
      (the image is what gets sent to the VLM for generation)
    """

    def __init__(
        self,
        embedder: BGEEmbedder,
        vector_store: VectorStore,
        top_k: int = 5,
        score_threshold: float = 0.3,
    ):
        self.embedder = embedder
        self.vector_store = vector_store
        self.top_k = top_k
        self.score_threshold = score_threshold

    def retrieve(self, query: str) -> List[Tuple[DocumentElement, float]]:
        """
        Retrieve relevant document elements for a query.

        Args:
            query: Natural language question

        Returns:
            List of (DocumentElement, score) tuples, sorted by relevance.
            For visual elements, the element contains the image_path for proxy recall.
        """
        query_vec = self.embedder.encode_query(query)
        results = self.vector_store.search(
            query_vec, top_k=self.top_k, score_threshold=self.score_threshold
        )

        if not results:
            logger.warning(f"No results found for query: '{query[:50]}...'")

        return results

    def retrieve_with_context(
        self, query: str, max_images: int = 3
    ) -> dict:
        """
        Retrieve and organize results into a structured context for generation.

        Args:
            query: Natural language question
            max_images: Maximum number of images to include

        Returns:
            dict with keys:
                - text_contexts: list of (text, page, score) for text chunks
                - image_contexts: list of (image_path, caption/summary, page, score)
                - table_contexts: list of (markdown, page, score)
                - all_pages: set of referenced page numbers
        """
        results = self.retrieve(query)

        text_contexts = []
        image_contexts = []
        table_contexts = []
        all_pages = set()

        image_count = 0

        for elem, score in results:
            all_pages.add(elem.page_number)

            if elem.type == ElementType.TEXT:
                text_contexts.append({
                    "content": elem.text_content,
                    "page": elem.page_number,
                    "heading": elem.heading_context,
                    "score": score,
                })

            elif elem.type == ElementType.TABLE:
                table_contexts.append({
                    "content": elem.text_content,  # Markdown table
                    "summary": elem.summary,
                    "page": elem.page_number,
                    "label": elem.inferred_label,
                    "score": score,
                })

            elif elem.type == ElementType.FIGURE:
                if image_count < max_images and elem.image_path:
                    # Resolve image path (may need fallback for old metadata)
                    img_path = Path(elem.image_path)
                    if not img_path.exists():
                        # Fallback: derive backend doc_id from collection name (format: "doc_{id}")
                        collection_name = self.vector_store.collection_name
                        backend_doc_id = collection_name.replace("doc_", "", 1) if collection_name.startswith("doc_") else elem.document_id
                        fallback = Path(__file__).parent.parent.parent / "backend" / "documents" / backend_doc_id / "images" / img_path.name
                        if fallback.exists():
                            img_path = fallback
                            elem.image_path = str(fallback)
                            logger.debug(f"Resolved image path via fallback: {fallback}")
                    if img_path.exists():
                        image_contexts.append({
                            "image_path": str(img_path),
                            "summary": elem.summary,
                            "caption": elem.caption,
                            "page": elem.page_number,
                            "label": elem.inferred_label,
                            "score": score,
                        })
                        image_count += 1
                    else:
                        logger.warning(f"Figure image not found: {elem.image_path}")

        return {
            "text_contexts": text_contexts,
            "image_contexts": image_contexts,
            "table_contexts": table_contexts,
            "all_pages": sorted(all_pages),
        }
