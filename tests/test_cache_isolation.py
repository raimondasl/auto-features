"""Unit tests must not share a real on-disk cache with the eval harness.

`arxiv_cache` is a module-level global and `evals/harness.py` switches it on at import
time. pytest imports every test module during collection, `pythonpath = ["evals"]`, and
several `tests/test_eval_*.py` import harness — so a seven-day disk cache at
`evals/.work/arxiv-cache` was live for the whole session before any test ran.

`collect_papers` consults that cache *before* its client, so a test that mocked
`arxiv.Client` could have its mock never called; and `put(..., empty_is_real=True)` means a
test whose mock returns nothing writes an empty entry that every later test asking the same
query inherits. The result depended on whether the developer's disk happened to hold real
entries — a suite that passes on a machine which has run the evals and fails on a fresh
runner is not testing the code.

`tests/conftest.py::_no_arxiv_response_cache` disables it per test. These are the guards
that say so, because a fixture nobody asserts about is a fixture that can be deleted.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import arxiv

from reporadar import arxiv_cache
from reporadar.collector import collect_papers
from reporadar.config import ArxivConfig

QUERY = "all:shared-cache-probe"
CFG = ArxivConfig(max_results_per_query=50, lookback_days=30)
# Mirrors the fields `collector.collect_papers` hashes; used only to report whether an
# entry exists when an assertion fails.
CACHE_FIELDS = {
    "kind": "search",
    "query": QUERY,
    "max_results": CFG.max_results_per_query,
    "sort_by": CFG.sort_by,
}


def _result(entry_id: str = "http://arxiv.org/abs/2401.00001v1") -> arxiv.Result:
    now = datetime.now(UTC)
    return arxiv.Result(
        entry_id=entry_id,
        title="Test Paper",
        summary="A test abstract.",
        published=now,
        updated=now,
        categories=["cs.CL"],
        authors=[arxiv.Result.Author("Alice Smith")],
    )


def _collect_with(results: list[arxiv.Result]) -> tuple[list[dict], MagicMock]:
    with patch("reporadar.collector.arxiv.Client") as MockClient:
        MockClient.return_value.results.return_value = iter(results)
        papers = collect_papers([QUERY], CFG)
    return papers, MockClient.return_value


def _diagnose(client: MagicMock) -> str:
    """Why did `collect_papers` not consult the mock?

    There are exactly two ways: the response cache answered first, or `_shared_client`
    handed back a client cached under a recycled `id(arxiv.Client)` from an earlier patch.
    A bare `assert client.results.called` distinguishes neither, and on 2026-08-20 that cost
    two CI rounds and three wrong hypotheses on a failure that reproduced only on the
    runner. The assertion now carries the state that separates them.
    """
    from reporadar import arxiv_cache, collector

    cached = arxiv_cache.get(dict(CACHE_FIELDS))
    return (
        f"\n    arxiv_cache.enabled()={arxiv_cache.enabled()}"
        f" _directory={arxiv_cache._directory!r} _ttl_s={arxiv_cache._ttl_s}"
        f"\n    entry for this query present={cached is not None} value={cached!r}"
        f"\n    arxiv_cache.stats()={arxiv_cache.stats()}"
        f"\n    _CLIENTS entries={len(collector._CLIENTS)}"
        f" this client cached={any(v is client for v in collector._CLIENTS.values())}"
        f"\n    client.results.called={client.results.called}"
        f" call_count={client.results.call_count}"
    )


class TestTheSuiteHasNoRealCache:
    def test_the_cache_is_off_inside_a_test(self) -> None:
        """Importing the eval harness anywhere in the session must not reach a test."""
        assert not arxiv_cache.enabled()
        assert arxiv_cache._directory is None

    def test_a_mocked_client_is_always_consulted(self, tmp_path) -> None:
        """The failure mode, stated directly: a mock that never gets called.

        Even if something re-enables the cache mid-test, `collect_papers` must not serve a
        stored answer to a test that supplied a client. This is what turned an assertion
        about `matched_query` into `IndexError: list index out of range` — the paper the
        mock offered was never fetched.
        """
        _collect_with([_result()])  # populate whatever there is to populate
        papers, client = _collect_with([_result()])
        assert client.results.called, (
            "collect_papers answered without asking the client" + _diagnose(client)
        )
        assert len(papers) == 1, _diagnose(client)

    def test_an_empty_result_does_not_poison_the_next_test(self) -> None:
        """`put(..., empty_is_real=True)` is right for the product and lethal between tests.

        With the cache on, this sequence returns 0 papers the second time and the second
        mock is never called. That is the cross-test leak, reproduced as a test.
        """
        empty, _ = _collect_with([])
        assert empty == []

        papers, client = _collect_with([_result()])
        assert client.results.called, "the second mock was never consulted" + _diagnose(client)
        assert len(papers) == 1, "an earlier result was served to this one" + _diagnose(client)

    def test_nothing_was_written_to_disk(self, tmp_path) -> None:
        """The other half: a test must not leave entries behind for the next run either."""
        arxiv_cache.configure(tmp_path)
        try:
            arxiv_cache.configure(None)
            _collect_with([_result()])
        finally:
            arxiv_cache.configure(None)
        assert list(tmp_path.iterdir()) == []


class TestTheGuardCanFail:
    """A fixture nobody can see failing is a fixture that silently stops working."""

    def test_re_enabling_the_cache_is_what_breaks_it(self, tmp_path) -> None:
        """Turning the cache back on inside one test reproduces the defect exactly, which
        is the evidence that the fixture — not luck — is what keeps the suite honest."""
        arxiv_cache.configure(tmp_path)
        try:
            empty, _ = _collect_with([])
            assert empty == []
            papers, client = _collect_with([_result()])
            assert not client.results.called, "expected the cache to answer instead"
            assert papers == []
        finally:
            arxiv_cache.configure(None)
