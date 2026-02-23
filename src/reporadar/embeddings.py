"""Embedding computation with graceful fallback when sentence-transformers is not installed."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

try:
    from sentence_transformers import SentenceTransformer

    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False

_MODEL_NAME = "all-MiniLM-L6-v2"


def _check_available() -> None:
    """Raise RuntimeError if sentence-transformers is not installed."""
    if not EMBEDDINGS_AVAILABLE:
        raise RuntimeError(
            "sentence-transformers is not installed. "
            "Install it with: pip install reporadar[embeddings]"
        )


@lru_cache(maxsize=1)
def _get_model() -> Any:
    """Return a cached SentenceTransformer instance (lazy-loaded)."""
    _check_available()
    return SentenceTransformer(_MODEL_NAME)


def compute_embedding(text: str) -> np.ndarray:
    """Encode a text string and return a 1D numpy array."""
    _check_available()
    model = _get_model()
    return model.encode(text, convert_to_numpy=True)


def compute_repo_embedding(repo_path: Path) -> np.ndarray | None:
    """Read README + docs from repo, combine text, and return embedding.

    Returns None if no text is found.
    """
    _check_available()
    texts: list[str] = []

    # Read README files
    for name in ("README.md", "README.rst", "README.txt", "README"):
        readme = repo_path / name
        if readme.is_file():
            content = readme.read_text(encoding="utf-8", errors="ignore")
            if content.strip():
                texts.append(content[:5000])  # Limit to avoid excessive text
            break

    # Read docs directory
    docs_dir = repo_path / "docs"
    if docs_dir.is_dir():
        for doc_file in sorted(docs_dir.glob("*.md"))[:5]:
            content = doc_file.read_text(encoding="utf-8", errors="ignore")
            if content.strip():
                texts.append(content[:2000])

    if not texts:
        return None

    combined = "\n\n".join(texts)
    return compute_embedding(combined)


def compute_paper_embedding(paper: dict[str, Any]) -> np.ndarray:
    """Encode title + abstract of a paper and return embedding."""
    _check_available()
    text = paper["title"] + ". " + paper["abstract"]
    return compute_embedding(text)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))
