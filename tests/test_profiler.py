"""Tests for reporadar.profiler."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from reporadar.profiler import (
    RepoProfile,
    _extract_anchors,
    _extract_keywords,
    _infer_domains,
    _parse_package_json,
    _parse_pyproject_toml,
    _parse_requirements_txt,
    profile_repo,
)


class TestParseRequirementsTxt:
    def test_parses_packages(self, fixtures_dir: Path) -> None:
        packages = _parse_requirements_txt(fixtures_dir / "sample_requirements.txt")
        assert "torch" in packages
        assert "transformers" in packages
        assert "langchain" in packages
        assert "numpy" in packages
        assert "pytest" in packages

    def test_skips_comments_and_flags(self, fixtures_dir: Path) -> None:
        packages = _parse_requirements_txt(fixtures_dir / "sample_requirements.txt")
        for pkg in packages:
            assert not pkg.startswith("#")
            assert not pkg.startswith("-")

    def test_strips_version_specifiers(self, fixtures_dir: Path) -> None:
        packages = _parse_requirements_txt(fixtures_dir / "sample_requirements.txt")
        for pkg in packages:
            assert ">=" not in pkg
            assert "=" not in pkg

    def test_nonexistent_file(self, tmp_path: Path) -> None:
        result = _parse_requirements_txt(tmp_path / "missing.txt")
        assert result == []


class TestParsePyprojectToml:
    def test_parses_dependencies(self, fixtures_dir: Path) -> None:
        packages = _parse_pyproject_toml(fixtures_dir / "sample_pyproject.toml")
        assert "torch" in packages
        assert "transformers" in packages
        assert "fastapi" in packages
        assert "scikit-learn" in packages

    def test_parses_optional_dependencies(self, fixtures_dir: Path) -> None:
        packages = _parse_pyproject_toml(fixtures_dir / "sample_pyproject.toml")
        assert "pytest" in packages
        assert "mypy" in packages

    def test_nonexistent_file(self, tmp_path: Path) -> None:
        result = _parse_pyproject_toml(tmp_path / "missing.toml")
        assert result == []


class TestParsePackageJson:
    def test_parses_dependencies(self, fixtures_dir: Path) -> None:
        packages = _parse_package_json(fixtures_dir / "sample_package.json")
        assert "react" in packages
        assert "next" in packages

    def test_parses_dev_dependencies(self, fixtures_dir: Path) -> None:
        packages = _parse_package_json(fixtures_dir / "sample_package.json")
        assert "typescript" in packages
        assert "eslint" in packages

    def test_handles_scoped_packages(self, fixtures_dir: Path) -> None:
        packages = _parse_package_json(fixtures_dir / "sample_package.json")
        # @tanstack/react-query should be cleaned
        assert any("tanstack" in p for p in packages)

    def test_nonexistent_file(self, tmp_path: Path) -> None:
        result = _parse_package_json(tmp_path / "missing.json")
        assert result == []


class TestExtractAnchors:
    def test_collects_from_all_manifests(self, tmp_repo: Path) -> None:
        anchors = _extract_anchors(tmp_repo)
        # From requirements.txt
        assert "torch" in anchors
        # From pyproject.toml
        assert "fastapi" in anchors
        # From package.json
        assert "react" in anchors

    def test_deduplicates(self, tmp_repo: Path) -> None:
        anchors = _extract_anchors(tmp_repo)
        # torch appears in both requirements.txt and pyproject.toml
        assert anchors.count("torch") == 1

    def test_empty_repo(self, tmp_repo_empty: Path) -> None:
        anchors = _extract_anchors(tmp_repo_empty)
        assert anchors == []


class TestInferDomains:
    def test_maps_known_packages(self) -> None:
        domains = _infer_domains(["torch", "transformers", "fastapi"])
        assert "deep learning" in domains
        assert "NLP" in domains
        assert "web APIs" in domains

    def test_empty_anchors(self) -> None:
        domains = _infer_domains([])
        assert domains == []

    def test_unknown_packages_ignored(self) -> None:
        domains = _infer_domains(["some-obscure-lib", "another-one"])
        assert domains == []


class TestExtractKeywords:
    def test_returns_sorted_keywords(self) -> None:
        docs = ["retrieval augmented generation with long context transformers"]
        keywords = _extract_keywords(docs, [])
        assert len(keywords) > 0
        # Should be sorted by weight descending
        weights = [w for _, w in keywords]
        assert weights == sorted(weights, reverse=True)

    def test_fallback_to_anchors_when_no_docs(self) -> None:
        keywords = _extract_keywords([], ["torch", "transformers"])
        assert len(keywords) == 2
        assert keywords[0] == ("torch", 1.0)

    def test_respects_max_keywords(self) -> None:
        docs = ["word " * 100]  # Lots of text
        keywords = _extract_keywords(docs, [], max_keywords=5)
        assert len(keywords) <= 5


class TestProfileRepo:
    def test_full_profile(self, tmp_repo: Path) -> None:
        profile = profile_repo(tmp_repo)

        assert isinstance(profile, RepoProfile)
        assert len(profile.keywords) > 0
        assert len(profile.anchors) > 0
        assert len(profile.domains) > 0

    def test_minimal_repo(self, tmp_repo_minimal: Path) -> None:
        profile = profile_repo(tmp_repo_minimal)

        assert isinstance(profile, RepoProfile)
        assert len(profile.keywords) > 0
        # No manifest files → no anchors
        assert profile.anchors == []
        assert profile.domains == []

    def test_empty_repo(self, tmp_repo_empty: Path) -> None:
        profile = profile_repo(tmp_repo_empty)

        assert isinstance(profile, RepoProfile)
        # No docs, no anchors → empty keywords
        assert profile.keywords == []
        assert profile.anchors == []
        assert profile.domains == []

    def test_nonexistent_dir(self) -> None:
        with pytest.raises(NotADirectoryError):
            profile_repo("/nonexistent/path/12345")

    def test_keywords_have_positive_weights(self, tmp_repo: Path) -> None:
        profile = profile_repo(tmp_repo)
        for _term, weight in profile.keywords:
            assert weight > 0
