"""Shared test fixtures for RepoRadar."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture()
def fixtures_dir() -> Path:
    return FIXTURES_DIR


@pytest.fixture()
def tmp_repo(tmp_path: Path) -> Path:
    """Create a temporary repo directory with sample manifest files."""
    # Copy fixture files into the temp dir with their canonical names
    shutil.copy(FIXTURES_DIR / "sample_readme.md", tmp_path / "README.md")
    shutil.copy(FIXTURES_DIR / "sample_requirements.txt", tmp_path / "requirements.txt")
    shutil.copy(FIXTURES_DIR / "sample_pyproject.toml", tmp_path / "pyproject.toml")
    shutil.copy(FIXTURES_DIR / "sample_package.json", tmp_path / "package.json")
    return tmp_path


@pytest.fixture()
def tmp_repo_minimal(tmp_path: Path) -> Path:
    """Create a temporary repo with only a README."""
    shutil.copy(FIXTURES_DIR / "sample_readme.md", tmp_path / "README.md")
    return tmp_path


@pytest.fixture()
def tmp_repo_empty(tmp_path: Path) -> Path:
    """An empty directory (no README, no manifests)."""
    return tmp_path
