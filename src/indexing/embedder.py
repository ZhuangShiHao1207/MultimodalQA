"""
BGE-M3 embedding model and FAISS vector store.
Handles vectorization of text chunks and summaries, plus similarity search.
"""
import os
import logging
import pickle
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
import faiss

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
        """
        Args:
            model_name: HuggingFace model identifier
            device: 'cuda', 'mps', 'cpu', or 'auto' (auto-detect)
            use_fp16: Use FP16 for faster inference
            max_length: Maximum token length
            batch_size: Batch size for encoding
        """
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
            # Windows symlink workaround (not needed on macOS/Linux)
            import sys
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
        """
        Encode a list of texts into dense vectors.

        Args:
            texts: List of text strings to encode

        Returns:
            numpy array of shape (len(texts), 1024)
        """
        if not texts:
            return np.empty((0, self.dimension if hasattr(self, 'dimension') else 1024), dtype=np.float32)

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
    FAISS-based vector store for document elements.
    Supports building index, similarity search, and persistence.
    """

    def __init__(
        self,
        dimension: int = 1024,
        index_type: str = "FlatIP",
        persist_dir: Optional[str] = None,
    ):
        """
        Args:
            dimension: Vector dimension (1024 for BGE-M3)
            index_type: FAISS index type ('FlatIP' for inner product / cosine)
            persist_dir: Directory for saving/loading index
        """
        self.dimension = dimension
        self.persist_dir = Path(persist_dir) if persist_dir else None

        # Create FAISS index
        if index_type == "FlatIP":
            self.index = faiss.IndexFlatIP(dimension)
        elif index_type == "FlatL2":
            self.index = faiss.IndexFlatL2(dimension)
        else:
            raise ValueError(f"Unsupported index type: {index_type}")

        # Document store: maps index position -> DocumentElement
        self.doc_store: List[DocumentElement] = []

        logger.info(f"VectorStore initialized: dim={dimension}, type={index_type}")

    def add_elements(
        self, elements: List[DocumentElement], embeddings: np.ndarray
    ):
        """
        Add document elements and their embeddings to the store.

        Args:
            elements: List of DocumentElements
            embeddings: Corresponding embedding vectors (n, 1024)
        """
        if len(elements) != len(embeddings):
            raise ValueError(
                f"Mismatch: {len(elements)} elements vs {len(embeddings)} embeddings"
            )

        # Normalize vectors for cosine similarity (FlatIP)
        faiss.normalize_L2(embeddings)

        # Add to FAISS index
        self.index.add(embeddings)

        # Store elements
        self.doc_store.extend(elements)

        logger.info(
            f"Added {len(elements)} elements to store. "
            f"Total: {self.index.ntotal} vectors."
        )

    def search(
        self, query_vector: np.ndarray, top_k: int = 5, score_threshold: float = 0.0
    ) -> List[Tuple[DocumentElement, float]]:
        """
        Search for most similar elements to a query vector.

        Args:
            query_vector: Query embedding (1024,)
            top_k: Number of results to return
            score_threshold: Minimum similarity score

        Returns:
            List of (DocumentElement, score) tuples, sorted by score descending
        """
        if self.index.ntotal == 0:
            return []

        # Normalize query vector
        query_vector = query_vector.reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(query_vector)

        # Search
        scores, indices = self.index.search(query_vector, min(top_k, self.index.ntotal))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or score < score_threshold:
                continue
            results.append((self.doc_store[idx], float(score)))

        return results

    def save(self, path: Optional[str] = None):
        """Save index and doc store to disk."""
        save_dir = Path(path) if path else self.persist_dir
        if not save_dir:
            raise ValueError("No persist directory specified")

        save_dir.mkdir(parents=True, exist_ok=True)

        # FAISS on Windows doesn't handle non-ASCII paths well
        # Use short relative path or ASCII-only path
        index_path = str(save_dir / "index.faiss")
        store_path = str(save_dir / "doc_store.pkl")

        try:
            faiss.write_index(self.index, index_path)
        except RuntimeError:
            # Fallback: save to temp and move
            import tempfile, shutil
            with tempfile.NamedTemporaryFile(suffix=".faiss", delete=False) as tmp:
                faiss.write_index(self.index, tmp.name)
                shutil.move(tmp.name, index_path)

        # Save doc store
        with open(store_path, "wb") as f:
            pickle.dump(self.doc_store, f)

        logger.info(f"VectorStore saved to {save_dir} ({self.index.ntotal} vectors)")

    def load(self, path: Optional[str] = None):
        """Load index and doc store from disk."""
        load_dir = Path(path) if path else self.persist_dir
        if not load_dir:
            raise ValueError("No persist directory specified")

        index_path = load_dir / "index.faiss"
        store_path = load_dir / "doc_store.pkl"

        if not index_path.exists() or not store_path.exists():
            raise FileNotFoundError(f"No saved index found in {load_dir}")

        self.index = faiss.read_index(str(index_path))

        with open(store_path, "rb") as f:
            self.doc_store = pickle.load(f)

        logger.info(f"VectorStore loaded from {load_dir} ({self.index.ntotal} vectors)")

    @property
    def size(self) -> int:
        """Number of vectors in the store."""
        return self.index.ntotal


def build_index(
    elements: List[DocumentElement],
    embedder: BGEEmbedder,
    persist_dir: Optional[str] = None,
) -> VectorStore:
    """
    Build a complete vector index from document elements.

    For TEXT elements: embed the text_content directly.
    For TABLE/FIGURE elements: embed the summary (generated by VLM).
    PAGE_IMAGE elements are skipped (they're only for reference).

    Args:
        elements: List of all DocumentElements (text + visual)
        embedder: BGE-M3 embedding model
        persist_dir: Where to save the index

    Returns:
        Populated VectorStore
    """
    # Filter elements that should be indexed
    indexable = []
    texts_to_embed = []

    for elem in elements:
        if elem.type == ElementType.TEXT:
            if elem.text_content.strip():
                indexable.append(elem)
                texts_to_embed.append(elem.text_content)

        elif elem.type in (ElementType.TABLE, ElementType.FIGURE):
            # Use summary for embedding (falls back to text_content/caption)
            embed_text = elem.summary or elem.text_content or elem.caption
            if embed_text and embed_text.strip():
                indexable.append(elem)
                texts_to_embed.append(embed_text)

        # PAGE_IMAGE elements are not indexed (used only for generation stage)

    if not indexable:
        logger.warning("No indexable elements found!")
        return VectorStore(persist_dir=persist_dir)

    logger.info(f"Embedding {len(texts_to_embed)} elements with BGE-M3...")

    # Encode all texts
    embeddings = embedder.encode(texts_to_embed)

    # Build vector store
    store = VectorStore(
        dimension=embeddings.shape[1],
        index_type="FlatIP",
        persist_dir=persist_dir,
    )
    store.add_elements(indexable, embeddings)

    # Save if persist_dir specified
    if persist_dir:
        store.save()

    return store
