"""Tests for reporadar.mcp_server — the pure data-gathering helpers.

These exercise the MCP tool bodies without the optional `mcp` SDK (the helpers
don't import it; only build_server/run_stdio do).
"""

from __future__ import annotations

from pathlib import Path

from reporadar.config import RankingConfig
from reporadar.mcp_server import (
    explain_relevance_payload,
    profile_payload,
    ranked_papers_payload,
    rate_paper_action,
    search_corpus_payload,
)
from reporadar.store import PaperStore


def _paper(arxiv_id: str, title: str = "A Paper") -> dict:
    return {
        "arxiv_id": arxiv_id,
        "title": title,
        "authors": ["Alice"],
        "abstract": "We propose a concrete method.",
        "categories": ["cs.LG"],
        "published": "2024-01-01T00:00:00+00:00",
        "updated": "2024-01-01T00:00:00+00:00",
        "url": f"http://arxiv.org/abs/{arxiv_id}",
        "pdf_url": f"http://arxiv.org/pdf/{arxiv_id}",
    }


def _seed(store: PaperStore) -> int:
    for aid in ("2401.00001v1", "2401.00002v1"):
        store.upsert_paper(_paper(aid))
    run_id = store.record_run(["q1"], 2, 0)
    store.save_scores(
        run_id,
        [
            {"arxiv_id": "2401.00001v1", "score_total": 0.9, "keyword_score": 0.6},
            {"arxiv_id": "2401.00002v1", "score_total": 0.4, "keyword_score": 0.2},
        ],
    )
    return run_id


class TestRankedPapers:
    def test_returns_best_first(self, tmp_path: Path) -> None:
        with PaperStore(tmp_path / "papers.db") as store:
            _seed(store)
            out = ranked_papers_payload(store, limit=10)
            assert out["run_id"] is not None
            ids = [p["arxiv_id"] for p in out["papers"]]
            assert ids == ["2401.00001v1", "2401.00002v1"]  # by score_total desc
            assert out["papers"][0]["title"] == "A Paper"

    def test_respects_limit(self, tmp_path: Path) -> None:
        with PaperStore(tmp_path / "papers.db") as store:
            _seed(store)
            assert len(ranked_papers_payload(store, limit=1)["papers"]) == 1

    def test_no_runs(self, tmp_path: Path) -> None:
        with PaperStore(tmp_path / "papers.db") as store:
            out = ranked_papers_payload(store)
            assert out["run_id"] is None and out["papers"] == []

    def test_withdrawn_paper_carries_a_warning(self, tmp_path: Path) -> None:
        """An agent never sees the digest's warning section.

        get_ranked_papers hands papers straight to a coding agent, so the withdrawal
        flag has to travel with the paper itself — an agent acting on a retracted
        result is the exact harm this signal exists to prevent.
        """
        with PaperStore(tmp_path / "papers.db") as store:
            _seed(store)
            store.save_signals([("2401.00001v1", "withdrawn", "comment", None)])
            out = ranked_papers_payload(store, limit=10)
        by_id = {p["arxiv_id"]: p for p in out["papers"]}
        assert by_id["2401.00001v1"]["withdrawn"] is True
        assert "retracted" in by_id["2401.00001v1"]["warning"]
        # A clean paper must not gain the key at all — absent, not False-y noise.
        assert "withdrawn" not in by_id["2401.00002v1"]

    def test_checked_clean_paper_carries_no_warning(self, tmp_path: Path) -> None:
        with PaperStore(tmp_path / "papers.db") as store:
            _seed(store)
            store.save_signals([("2401.00001v1", "withdrawn", None, None)])
            out = ranked_papers_payload(store, limit=10)
        assert all("withdrawn" not in p for p in out["papers"])


class TestExplainRelevance:
    def test_found(self, tmp_path: Path) -> None:
        with PaperStore(tmp_path / "papers.db") as store:
            _seed(store)
            # version-insensitive match (2401.00001 vs the stored 2401.00001v1)
            out = explain_relevance_payload(store, "2401.00001", RankingConfig())
            assert out["arxiv_id"] == "2401.00001v1"
            assert "keyword" in out["explanation"]

    def test_not_in_run(self, tmp_path: Path) -> None:
        with PaperStore(tmp_path / "papers.db") as store:
            _seed(store)
            out = explain_relevance_payload(store, "9999.99999", RankingConfig())
            assert "error" in out

    def test_no_runs(self, tmp_path: Path) -> None:
        with PaperStore(tmp_path / "papers.db") as store:
            assert "error" in explain_relevance_payload(store, "2401.00001", RankingConfig())

    def test_explains_that_a_paper_was_withdrawn(self, tmp_path: Path) -> None:
        # "Why was this ranked?" must answer "it was withdrawn and penalized".
        with PaperStore(tmp_path / "papers.db") as store:
            _seed(store)
            store.save_signals([("2401.00001v1", "withdrawn", "comment", None)])
            out = explain_relevance_payload(store, "2401.00001", RankingConfig())
        assert out["withdrawn"] is True
        assert "retracted" in out["warning"]


class TestRatePaper:
    def test_valid_rating_persists(self, tmp_path: Path) -> None:
        with PaperStore(tmp_path / "papers.db") as store:
            store.upsert_paper(_paper("2401.00001v1"))
            out = rate_paper_action(store, "2401.00001v1", 5)
            assert out == {"ok": True, "arxiv_id": "2401.00001v1", "rating": 5}
            assert store.get_all_ratings()  # persisted

    def test_out_of_range_rejected(self, tmp_path: Path) -> None:
        with PaperStore(tmp_path / "papers.db") as store:
            store.upsert_paper(_paper("2401.00001v1"))
            assert "error" in rate_paper_action(store, "2401.00001v1", 9)
            assert not store.get_all_ratings()  # nothing persisted


class TestSearchPapers:
    def test_searches_the_whole_corpus(self, tmp_path: Path) -> None:
        with PaperStore(tmp_path / "papers.db") as store:
            _seed(store)  # two papers, both abstracts contain "concrete method"
            out = search_corpus_payload(store, "concrete method", limit=5)

        assert out["query"] == "concrete method"
        assert out["count"] >= 1
        for p in out["papers"]:
            assert {"arxiv_id", "title", "search_score"} <= set(p)
            assert p["search_score"] is not None

    def test_no_match_is_empty(self, tmp_path: Path) -> None:
        with PaperStore(tmp_path / "papers.db") as store:
            _seed(store)
            out = search_corpus_payload(store, "zzzznomatch", limit=5)
        assert out["count"] == 0 and out["papers"] == []


class TestProfilePayload:
    def test_shapes_the_profile(self, tmp_path: Path) -> None:
        (tmp_path / "README.md").write_text(
            "# retrieval augmented generation library\n\nDense passage retrieval and reranking.",
            encoding="utf-8",
        )
        out = profile_payload(tmp_path)
        assert set(out) == {"keywords", "anchors", "domains"}
        assert isinstance(out["keywords"], list)
        # keyword entries are [term, weight] pairs
        assert all(len(kw) == 2 for kw in out["keywords"])
