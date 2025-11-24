"""Embedding service for vector representations."""
import numpy as np
from typing import List
from sentence_transformers import SentenceTransformer
from core.config import EMBEDDER_NAME
from core.logging import setup_logger

logger = setup_logger(__name__)

# Initialize embedding model
embedder = SentenceTransformer(EMBEDDER_NAME)
logger.info(f"Embedding model loaded: {EMBEDDER_NAME}")


def _normalize(v: np.ndarray) -> np.ndarray:
    """L2 normalization."""
    n = np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
    return v / n


def embed_passages(texts: List[str]) -> np.ndarray:
    """
    Embed document chunks using E5 instruct format.

    Args:
        texts: List of text chunks to embed

    Returns:
        Normalized embeddings array
    """
    prepped = [f"passage: {t}" for t in texts]
    vecs = embedder.encode(prepped, convert_to_numpy=True, normalize_embeddings=False)
    return _normalize(vecs).astype(np.float32)


def embed_query(q: str) -> np.ndarray:
    """
    Embed query using E5 instruct format.

    Args:
        q: Query string

    Returns:
        Normalized embedding array
    """
    prepped = [f"query: {q}"]
    vec = embedder.encode(prepped, convert_to_numpy=True, normalize_embeddings=False)
    return _normalize(vec).astype(np.float32)
