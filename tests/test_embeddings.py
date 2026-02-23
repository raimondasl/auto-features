"""Tests for reporadar.embeddings."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


class TestEmbeddingsAvailable:
    @patch("reporadar.embeddings.EMBEDDINGS_AVAILABLE", True)
    @patch("reporadar.embeddings._get_model")
    def test_compute_embedding_returns_array(self, mock_get_model: MagicMock) -> None:
        from reporadar.embeddings import compute_embedding

        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([0.1, 0.2, 0.3])
        mock_get_model.return_value = mock_model

        result = compute_embedding("test text")

        assert isinstance(result, np.ndarray)
        assert result.shape == (3,)
        mock_model.encode.assert_called_once_with("test text", convert_to_numpy=True)

    @patch("reporadar.embeddings.EMBEDDINGS_AVAILABLE", True)
    @patch("reporadar.embeddings._get_model")
    def test_compute_embedding_shape(self, mock_get_model: MagicMock) -> None:
        from reporadar.embeddings import compute_embedding

        mock_model = MagicMock()
        mock_model.encode.return_value = np.zeros(384)
        mock_get_model.return_value = mock_model

        result = compute_embedding("hello world")

        assert result.shape == (384,)


class TestEmbeddingsUnavailable:
    @patch("reporadar.embeddings.EMBEDDINGS_AVAILABLE", False)
    def test_compute_embedding_raises(self) -> None:
        from reporadar.embeddings import compute_embedding

        with pytest.raises(RuntimeError, match="sentence-transformers is not installed"):
            compute_embedding("test")

    @patch("reporadar.embeddings.EMBEDDINGS_AVAILABLE", False)
    def test_compute_repo_embedding_raises(self) -> None:
        from reporadar.embeddings import compute_repo_embedding

        with pytest.raises(RuntimeError, match="sentence-transformers is not installed"):
            compute_repo_embedding(Path("."))

    @patch("reporadar.embeddings.EMBEDDINGS_AVAILABLE", False)
    def test_compute_paper_embedding_raises(self) -> None:
        from reporadar.embeddings import compute_paper_embedding

        with pytest.raises(RuntimeError, match="sentence-transformers is not installed"):
            compute_paper_embedding({"title": "Test", "abstract": "Test abstract"})


class TestCosineSimilarity:
    def test_identical_vectors(self) -> None:
        from reporadar.embeddings import cosine_similarity

        a = np.array([1.0, 2.0, 3.0])
        assert cosine_similarity(a, a) == pytest.approx(1.0)

    def test_orthogonal_vectors(self) -> None:
        from reporadar.embeddings import cosine_similarity

        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert cosine_similarity(a, b) == pytest.approx(0.0)

    def test_opposite_vectors(self) -> None:
        from reporadar.embeddings import cosine_similarity

        a = np.array([1.0, 2.0, 3.0])
        b = np.array([-1.0, -2.0, -3.0])
        assert cosine_similarity(a, b) == pytest.approx(-1.0)

    def test_zero_vector(self) -> None:
        from reporadar.embeddings import cosine_similarity

        a = np.array([1.0, 2.0])
        b = np.zeros(2)
        assert cosine_similarity(a, b) == 0.0


class TestComputeRepoEmbedding:
    @patch("reporadar.embeddings.EMBEDDINGS_AVAILABLE", True)
    @patch("reporadar.embeddings._get_model")
    def test_reads_readme(self, mock_get_model: MagicMock, tmp_path: Path) -> None:
        from reporadar.embeddings import compute_repo_embedding

        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([0.1, 0.2, 0.3])
        mock_get_model.return_value = mock_model

        readme = tmp_path / "README.md"
        readme.write_text("# My Project\nThis is a test repo.", encoding="utf-8")

        result = compute_repo_embedding(tmp_path)

        assert result is not None
        assert isinstance(result, np.ndarray)
        mock_model.encode.assert_called_once()
        call_text = mock_model.encode.call_args[0][0]
        assert "My Project" in call_text

    @patch("reporadar.embeddings.EMBEDDINGS_AVAILABLE", True)
    @patch("reporadar.embeddings._get_model")
    def test_no_text_returns_none(self, mock_get_model: MagicMock, tmp_path: Path) -> None:
        from reporadar.embeddings import compute_repo_embedding

        result = compute_repo_embedding(tmp_path)

        assert result is None

    @patch("reporadar.embeddings.EMBEDDINGS_AVAILABLE", True)
    @patch("reporadar.embeddings._get_model")
    def test_includes_docs(self, mock_get_model: MagicMock, tmp_path: Path) -> None:
        from reporadar.embeddings import compute_repo_embedding

        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([0.1, 0.2, 0.3])
        mock_get_model.return_value = mock_model

        readme = tmp_path / "README.md"
        readme.write_text("# Main readme", encoding="utf-8")
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        (docs_dir / "guide.md").write_text("# Guide content", encoding="utf-8")

        result = compute_repo_embedding(tmp_path)

        assert result is not None
        call_text = mock_model.encode.call_args[0][0]
        assert "Main readme" in call_text
        assert "Guide content" in call_text


class TestComputePaperEmbedding:
    @patch("reporadar.embeddings.EMBEDDINGS_AVAILABLE", True)
    @patch("reporadar.embeddings._get_model")
    def test_concatenates_title_and_abstract(self, mock_get_model: MagicMock) -> None:
        from reporadar.embeddings import compute_paper_embedding

        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([0.5, 0.5])
        mock_get_model.return_value = mock_model

        paper = {"title": "My Paper", "abstract": "Some abstract text."}
        result = compute_paper_embedding(paper)

        assert isinstance(result, np.ndarray)
        call_text = mock_model.encode.call_args[0][0]
        assert call_text == "My Paper. Some abstract text."
