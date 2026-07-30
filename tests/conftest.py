"""Shared test fixtures for RepoRadar."""

from __future__ import annotations

import shutil
import urllib.request
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture(autouse=True)
def _no_network(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Fail any test that reaches the network instead of letting it silently do so.

    A test that quietly hits arXiv/HF/S2 is slow, flaky and offline-hostile — and it
    hides mocking mistakes: ``enrichment: provider: off`` was landing as the YAML
    boolean ``False``, so two tests ran live enrichment while claiming to have it
    disabled.

    The verdict is raised at **teardown**, not at the call site, because every source
    adapter degrades gracefully behind ``except Exception`` — an exception thrown from
    ``urlopen`` would be swallowed and the test would still pass, which is exactly how
    the original mistake stayed invisible. Tests that genuinely exercise the HTTP layer
    patch ``urlopen`` themselves, which replaces this and records nothing.

    Both HTTP stacks have to be blocked: RepoRadar's own adapters use stdlib
    ``urllib``, but the ``arxiv`` package (and therefore ``collector``) uses
    ``requests``, so guarding only ``urlopen`` left the biggest caller uncovered.
    """
    attempted: list[str] = []

    def _blocked(*args: Any, **kwargs: Any) -> Any:
        target = args[0] if args else kwargs.get("url", "?")
        attempted.append(str(getattr(target, "full_url", target)))
        raise OSError("network access is blocked in tests")

    def _blocked_session(self: Any, method: str, url: str, *args: Any, **kwargs: Any) -> Any:
        attempted.append(f"{method} {url}")
        raise OSError("network access is blocked in tests")

    monkeypatch.setattr(urllib.request, "urlopen", _blocked)
    try:
        import requests

        monkeypatch.setattr(requests.Session, "request", _blocked_session)
    except ImportError:  # pragma: no cover - requests ships with arxiv
        pass
    yield
    if attempted:
        raise AssertionError(
            f"test attempted {len(attempted)} live network request(s): {attempted[:3]}. "
            "Patch the source adapter (or urlopen) so the test runs offline."
        )


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
