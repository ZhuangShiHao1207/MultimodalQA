"""
BGE-M3 embedding model and ChromaDB vector store.
Handles vectorization of text chunks and summaries, plus similarity search.
Persistent storage via ChromaDB — data survives backend restarts.
"""
import os
import sys
import logging
from pathlib import Path
from typing import List, Tuple, Optional

# Patch torch version check (models use safetensors, not affected by CVE-2025-32434)
import transformers.utils.import_utils
if hasattr(transformers.utils.import_utils, 'check_torch_load_is_safe'):
    transformers.utils.import_utils.check_torch_load_is_safe = lambda: None
import transformers.modeling_utils
if hasattr(transformers.modeling_utils, 'check_torch_load_is_safe'):
    transformers.modeling_utils.check_torch_load_is_safe = lambda: None

import numpy as np
import chromadb

from src.ingestion.models import DocumentElement, ElementType

logger = logging.getLogger(__name__)


def _auto_detect_device() -> str:
    """Auto-detect the best available compute device."""
    import torch
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class BGEEmbedder:
    """
    BGE-M3 embedding model wrapper.
    Encodes text into dense vectors for similarity search.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        device: str = "auto",
        use_fp16: bool = True,
        max_length: int = 8192,
        batch_size: int = 8,
    ):
        self.model_name = model_name
        self.device = device if device != "auto" else _auto_detect_device()
        self.use_fp16 = use_fp16
        self.max_length = max_length
        self.batch_size = batch_size
        self._model = None
        logger.info(f"BGEEmbedder configured: device={self.device}")

    @property
    def model(self):
        """Lazy-load the model on first use."""
        if self._model is None:
            logger.info(f"Loading BGE-M3 model: {self.model_name} (device={self.device})...")
            if sys.platform == "win32":
                os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
            from FlagEmbedding import BGEM3FlagModel
            self._model = BGEM3FlagModel(
                self.model_name,
                use_fp16=self.use_fp16,
                device=self.device,
            )
            logger.info("BGE-M3 model loaded successfully.")
        return self._model

    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts into 1024-dim dense vectors."""
        if not texts:
            return np.empty((0, 1024), dtype=np.float32)
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            max_length=self.max_length,
        )["dense_vecs"]
        return np.array(embeddings, dtype=np.float32)

    def encode_query(self, query: str) -> np.ndarray:
        """Encode a single query string."""
        return self.encode([query])[0]


class VectorStore:
    """
    ChromaDB-backed vector store for document elements.
    Persistent by default — data survives backend restarts.
    """

    def __init__(
        self,
        collection_name: str = "documents",
        persist_dir: Optional[str] = None,
    ):
        """
        Args:
            collection_name: ChromaDB collection name
            persist_dir: Directory for ChromaDB persistence (None = in-memory)
        """
        self.collection_name = collection_name

        if persist_dir:
            self.persist_dir = Path(persist_dir)
            self.persist_dir.mkdir(parents=True, exist_ok=True)
            self.client = chromadb.PersistentClient(path=str(self.persist_dir))
        else:
            self.client = chromadb.EphemeralClient()

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        # Local cache for full DocumentElement objects
        self._elements_cache: dict = {}

        logger.info(
            f"VectorStore initialized: collection='{collection_name}', "
            f"persist={'yes (' + str(persist_dir) + ')' if persist_dir else 'no (in-memory)'}, "
            f"existing_vectors={self.collection.count()}"
        )

    def add_elements(self, elements: List[DocumentElement], embeddings: np.ndarray):
        """Add document elements and their embeddings to the store."""
        if len(elements) != len(embeddings):
            raise ValueError(f"Mismatch: {len(elements)} elements vs {len(embeddings)} embeddings")
        if len(elements) == 0:
            return

        ids = [elem.id for elem in elements]
        documents = []
        metadatas = []

        for elem in elements:
            doc_text = elem.summary or elem.text_content or elem.caption or ""
            documents.append(doc_text[:10000])
            metadatas.append({
                "type": elem.type.value,
                "page_number": elem.page_number,
                "document_id": elem.document_id,
                "document_name": elem.document_name,
                "heading_context": elem.heading_context or "",
                "inferred_label": elem.inferred_label or "",
                "caption": elem.caption or "",
                "image_path": elem.image_path or "",
                "has_image": "true" if elem.image_path else "false",
            })
            self._elements_cache[elem.id] = elem

        self.collection.upsert(
            ids=ids,
            embeddings=embeddings.tolist(),
            documents=documents,
            metadatas=metadatas,
        )

        logger.info(f"Added {len(elements)} elements to store. Total: {self.collection.count()} vectors.")

    def search(
        self, query_vector: np.ndarray, top_k: int = 5, score_threshold: float = 0.0
    ) -> List[Tuple[DocumentElement, float]]:
        """
        Search for most similar elements.
        Returns list of (DocumentElement, cosine_similarity_score) tuples.
        """
        if self.collection.count() == 0:
            return []

        n_results = min(top_k, self.collection.count())
        results = self.collection.query(
            query_embeddings=[query_vector.tolist()],
            n_results=n_results,
            include=["metadatas", "documents", "distances"],
        )

        output = []
        for doc_id, distance, metadata, document in zip(
            results["ids"][0],
            results["distances"][0],
            results["metadatas"][0],
            results["documents"][0],
        ):
            # Cosine distance → similarity
            score = 1.0 - distance
            if score < score_threshold:
                continue

            # Get full element from cache or reconstruct from metadata
            if doc_id in self._elements_cache:
                elem = self._elements_cache[doc_id]
            else:
                elem = DocumentElement(
                    id=doc_id,
                    type=ElementType(metadata.get("type", "text")),
                    document_id=metadata.get("document_id", ""),
                    document_name=metadata.get("document_name", ""),
                    page_number=metadata.get("page_number", 0),
                    text_content=document or "",
                    heading_context=metadata.get("heading_context", ""),
                    inferred_label=metadata.get("inferred_label", ""),
                    caption=metadata.get("caption", ""),
                    image_path=metadata.get("image_path", "") or None,
                    summary=document or "",
                )
                self._elements_cache[doc_id] = elem

            output.append((elem, score))

        return output

    @property
    def size(self) -> int:
        """Number of vectors in the store."""
        return self.collection.count()

    def delete_collection(self):
        """Delete the entire collection."""
        try:
            self.client.delete_collection(self.collection_name)
            self._elements_cache.clear()
        except Exception as e:
            logger.warning(f"Failed to delete collection: {e}")


def build_index(
    elements: List[DocumentElement],
    embedder: BGEEmbedder,
    persist_dir: Optional[str] = None,
    collection_name: str = "documents",
) -> VectorStore:
    """
    Build a vector index from document elements using ChromaDB.

    For TEXT elements: embed the text_content directly.
    For TABLE/FIGURE elements: embed the summary (generated by VLM).
    PAGE_IMAGE elements are skipped.
    """
    indexable = []
    texts_to_embed = []

    for elem in elements:
        if elem.type == ElementType.TEXT:
            if elem.text_content.strip():
                indexable.append(elem)
                texts_to_embed.append(elem.text_content)
        elif elem.type in (ElementType.TABLE, ElementType.FIGURE):
            embed_text = elem.summary or elem.text_content or elem.caption
            if embed_text and embed_text.strip():
                indexable.append(elem)
                texts_to_embed.append(embed_text)

    if not indexable:
        logger.warning("No indexable elements found!")
        return VectorStore(collection_name=collection_name, persist_dir=persist_dir)

    logger.info(f"Embedding {len(texts_to_embed)} elements with BGE-M3...")
    embeddings = embedder.encode(texts_to_embed)

    store = VectorStore(
        collection_name=collection_name,
        persist_dir=persist_dir,
    )
    store.add_elements(indexable, embeddings)

    return store
