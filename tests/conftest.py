"""Shared test fixtures for RepoRadar."""

from __future__ import annotations

import shutil
import urllib.request
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from reporadar import arxiv_cache, arxiv_rate, s2_rate

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


@pytest.fixture(autouse=True)
def _no_arxiv_throttle_sleep() -> Iterator[None]:
    """Drop the arXiv politeness interval to 0 for the suite.

    Every arXiv path now takes a turn at one process-wide 3-second gate. That is correct
    against the live API and absurd against mocks: a few hundred patched requests would add
    an hour of real sleeping to a 90-second suite. The shipped default is asserted in
    tests/test_arxiv_rate.py so this fixture cannot quietly become the product's behaviour.
    """
    previous = arxiv_rate.set_min_interval(0.0)
    try:
        yield
    finally:
        arxiv_rate.set_min_interval(previous)


@pytest.fixture(autouse=True)
def _no_arxiv_response_cache() -> Iterator[None]:
    """No test reads or writes the real on-disk arXiv response cache.

    `arxiv_cache` is a module-level global, and **`evals/harness.py` switches it on at
    import time** — `arxiv_cache.configure(ARXIV_CACHE_DIR)`, pointed at
    `evals/.work/arxiv-cache` with a seven-day TTL. pytest imports every test module during
    collection, `pythonpath = ["evals"]`, and several `tests/test_eval_*.py` import harness.
    So the cache was live for the whole session before a single test ran.

    Two things follow, and both are worse than they sound. `collect_papers` consults the
    cache *before* its client, so a test that has carefully mocked `arxiv.Client` can have
    its mock never called at all — the assertion then fails somewhere far away, as
    `IndexError` on an empty list. And `put(..., empty_is_real=True)` means a test whose
    mock returns nothing WRITES an empty entry, which every later test asking the same
    query inherits. Reproduced directly: with the cache on, one `collect_papers` returning
    `[]` makes the next one return `[]` too, with `results.called == False`.

    That also makes the suite's result depend on the developer's disk. A machine that has
    run the evals has ~1,100 real entries and sails through; a fresh CI runner starts empty
    and can poison itself mid-session. A test suite must not be able to tell.

    The cache is a good thing — for the eval harness, which wants it, and which configures
    it for itself. It just has no business being on during unit tests.
    """
    previous_dir = arxiv_cache._directory
    previous_ttl = arxiv_cache._ttl_s
    arxiv_cache.configure(None)
    try:
        yield
    finally:
        arxiv_cache.configure(previous_dir, previous_ttl)


@pytest.fixture(autouse=True)
def _no_s2_throttle_sleep() -> Iterator[None]:
    """Same, for the Semantic Scholar 1 RPS gate.

    Needed for a second reason beyond speed: a mocked 429 calls
    ``s2_rate.note_throttled()``, which parks the shared clock 30 seconds into the future.
    Without this the retry tests would really sleep for half a minute each, and the *next*
    test touching S2 would inherit the hold. The shipped interval is asserted in
    tests/test_s2_rate.py so this fixture cannot quietly become the product's behaviour.
    """
    previous = s2_rate.set_min_interval(0.0)
    try:
        yield
    finally:
        s2_rate.set_min_interval(previous)


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
