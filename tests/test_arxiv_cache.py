"""Tests for the arXiv response cache.

`arxiv_rate` governs how fast we ask arXiv; this governs how often we ask the same thing.
The second turned out to be the binding constraint — ~760 requests in a day, four sweeps
of one 25-case benchmark issuing byte-identical queries, and the last two cases refused.

A cache is also a good way to introduce quiet wrongness, so the tests here are mostly
about the ways it must NOT help:

* it must be off unless configured, or `rr update` silently starts serving stale digests;
* it must never store an empty result, because "arXiv found nothing" and "arXiv refused"
  look identical once written to disk, and this project has already scored seven pools of
  429-storm zeros as honest measurements;
* a damaged entry must be a miss, not an empty list, for the same reason;
* the key must cover everything that changes the response, and nothing that does not.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pytest

from reporadar import arxiv_cache


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path: Path) -> Any:
    arxiv_cache.configure(None)
    arxiv_cache.reset_stats()
    yield
    arxiv_cache.configure(None)
    arxiv_cache.reset_stats()


PAPERS = [{"arxiv_id": "2401.00001", "title": "A paper", "published": "2026-01-01T00:00:00+00:00"}]
FIELDS = {"kind": "search", "query": "all:retrieval", "max_results": 50, "sort_by": "relevance"}


class TestDisabledByDefault:
    def test_get_returns_none_when_unconfigured(self) -> None:
        assert arxiv_cache.get(FIELDS) is None

    def test_put_is_a_no_op_when_unconfigured(self, tmp_path: Path) -> None:
        arxiv_cache.put(FIELDS, PAPERS)
        assert not list(tmp_path.iterdir())

    def test_enabled_reflects_configuration(self, tmp_path: Path) -> None:
        assert not arxiv_cache.enabled()
        arxiv_cache.configure(tmp_path)
        assert arxiv_cache.enabled()
        arxiv_cache.configure(None)
        assert not arxiv_cache.enabled()

    def test_zero_ttl_disables_it(self, tmp_path: Path) -> None:
        """A TTL of zero means 'do not reuse', not 'reuse forever'."""
        arxiv_cache.configure(tmp_path, ttl_s=0)
        assert not arxiv_cache.enabled()
        arxiv_cache.put(FIELDS, PAPERS)
        assert arxiv_cache.get(FIELDS) is None


class TestRoundTrip:
    def test_stores_and_returns_papers(self, tmp_path: Path) -> None:
        arxiv_cache.configure(tmp_path)
        arxiv_cache.put(FIELDS, PAPERS)
        assert arxiv_cache.get(FIELDS) == PAPERS
        assert arxiv_cache.stats()["hits"] == 1

    def test_expired_entries_are_a_miss(self, tmp_path: Path) -> None:
        arxiv_cache.configure(tmp_path, ttl_s=60)
        arxiv_cache.put(FIELDS, PAPERS)
        path = next(tmp_path.glob("*.json"))
        entry = json.loads(path.read_text(encoding="utf-8"))
        entry["cached_at"] = time.time() - 3600
        path.write_text(json.dumps(entry), encoding="utf-8")
        assert arxiv_cache.get(FIELDS) is None
        assert arxiv_cache.stats()["expired"] == 1

    def test_entry_records_what_it_was_keyed_on(self, tmp_path: Path) -> None:
        """So a cache directory can be audited by reading it, not by trusting this module."""
        arxiv_cache.configure(tmp_path)
        arxiv_cache.put(FIELDS, PAPERS)
        entry = json.loads(next(tmp_path.glob("*.json")).read_text(encoding="utf-8"))
        assert entry["keyed_on"] == FIELDS


class TestAnEmptyResultIsOnlyCachedWhenObserved:
    """The single most dangerous thing this module could do.

    An empty list and a failed fetch are the same bytes on disk; seven pools were once
    cached empty after an arXiv 429 storm and scored as legitimate zeros. So an empty is
    stored only when the caller states it observed one — which `collector` can, because
    `_query_with_retry` raises instead of returning [] when it gives up.
    """

    def test_empty_papers_are_dropped_by_default(self, tmp_path: Path) -> None:
        arxiv_cache.configure(tmp_path)
        arxiv_cache.put(FIELDS, [])
        assert not list(tmp_path.glob("*.json"))
        assert arxiv_cache.get(FIELDS) is None

    def test_an_observed_empty_is_stored(self, tmp_path: Path) -> None:
        """Otherwise a query that genuinely matches nothing costs a request forever.

        Measured on `rag`: 2 of its 5 queries match nothing, so the blanket rule was
        re-fetching 40% of that case's queries on every single run.
        """
        arxiv_cache.configure(tmp_path)
        arxiv_cache.put(FIELDS, [], empty_is_real=True)
        assert arxiv_cache.get(FIELDS) == []

    def test_a_cached_empty_is_a_hit_not_a_miss(self, tmp_path: Path) -> None:
        """`[]` and `None` must stay distinguishable to the caller: one means 'arXiv said
        nothing matches', the other means 'ask arXiv'."""
        arxiv_cache.configure(tmp_path)
        arxiv_cache.put(FIELDS, [], empty_is_real=True)
        assert arxiv_cache.get(FIELDS) is not None
        assert arxiv_cache.stats()["hits"] == 1

    def test_corrupt_entry_is_a_miss_not_an_empty_list(self, tmp_path: Path) -> None:
        arxiv_cache.configure(tmp_path)
        arxiv_cache.put(FIELDS, PAPERS)
        next(tmp_path.glob("*.json")).write_text("{not json", encoding="utf-8")
        assert arxiv_cache.get(FIELDS) is None

    def test_entry_with_non_list_papers_is_a_miss(self, tmp_path: Path) -> None:
        arxiv_cache.configure(tmp_path)
        path = tmp_path / "x.json"
        path.write_text(json.dumps({"cached_at": time.time(), "papers": "nope"}), encoding="utf-8")
        assert arxiv_cache.get({"kind": "search", "query": "whatever"}) is None


class TestTheKey:
    @pytest.mark.parametrize(
        "changed",
        [
            {"query": "all:something-else"},
            {"max_results": 100},
            {"sort_by": "submitted"},
            {"kind": "ids"},
        ],
    )
    def test_anything_that_changes_the_response_changes_the_key(
        self, tmp_path: Path, changed: dict[str, Any]
    ) -> None:
        arxiv_cache.configure(tmp_path)
        arxiv_cache.put(FIELDS, PAPERS)
        assert arxiv_cache.get({**FIELDS, **changed}) is None

    def test_field_order_does_not_change_the_key(self, tmp_path: Path) -> None:
        """A dict built in a different order is the same query, not a new fetch.

        Keys are canonicalised with sort_keys rather than str()-formatted — the drift that
        made a verdict cache keyed without its prompt return answers to another question.
        """
        arxiv_cache.configure(tmp_path)
        arxiv_cache.put(FIELDS, PAPERS)
        reordered = dict(reversed(list(FIELDS.items())))
        assert arxiv_cache.get(reordered) == PAPERS
