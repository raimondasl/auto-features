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
        """`papers` is the Top Picks tier, not everything scored. The 0.4 paper is below
        the heuristic threshold and reaches `maybe_relevant` — where the digest puts it —
        rather than being handed to an agent as a recommendation."""
        with PaperStore(tmp_path / "papers.db") as store:
            _seed(store)
            out = ranked_papers_payload(store, limit=10)
            assert out["run_id"] is not None
            assert [p["arxiv_id"] for p in out["papers"]] == ["2401.00001v1"]
            assert [p["arxiv_id"] for p in out["maybe_relevant"]] == ["2401.00002v1"]
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
        # Beside the recommendations, not among them. `digest_window` takes retracted
        # papers out before the window's cut -- so the slot they would have wasted goes
        # to the next paper -- and this payload carries them in their own list for the
        # same reason the digest keeps a muted section: the agent still has to HEAR about
        # a retraction it might otherwise have found on its own.
        assert all(p["arxiv_id"] != "2401.00001v1" for p in out["papers"])
        flagged = {p["arxiv_id"]: p for p in out["muted"]}
        assert flagged["2401.00001v1"]["withdrawn"] is True
        assert "retracted" in flagged["2401.00001v1"]["warning"]
        # A clean paper must not gain the key at all — absent, not False-y noise.
        assert all("withdrawn" not in p for p in out.get("maybe_relevant", []))

    def test_checked_clean_paper_carries_no_warning(self, tmp_path: Path) -> None:
        with PaperStore(tmp_path / "papers.db") as store:
            _seed(store)
            store.save_signals([("2401.00001v1", "withdrawn", None, None)])
            out = ranked_papers_payload(store, limit=10)
        assert all("withdrawn" not in p for p in out["papers"])
        # A *checked and clean* paper is not a retraction, so nothing is muted at all
        # and the key must be absent entirely rather than present and empty.
        assert "muted" not in out


class TestRankedPapersIsTheDigestsAnswer:
    """`get_ranked_papers` used to be `get_scores_for_run(run_id)[:limit]` — the raw
    heuristic/RRF order, ungated. So an agent and a human reading the same repository at
    the same run got materially different recommendations, and the agent got the weaker
    set: on the benchmark the gate is where the precision comes from (0.892 with it).

    Routing it through `digest_window` makes three consumers share one rule — the digest,
    `rr explain`, and this."""

    def test_the_gate_filters_when_triage_is_enabled(self, tmp_path: Path) -> None:
        with PaperStore(tmp_path / "papers.db") as store:
            run_id = _seed(store)
            store.save_llm_scores(
                run_id,
                {
                    "2401.00001v1": {"llm_score": 1, "llm_reason": "background only"},
                    "2401.00002v1": {"llm_score": 3, "llm_reason": "directly applicable"},
                },
            )
            out = ranked_papers_payload(store, limit=10, triage_threshold=2)
        assert [p["arxiv_id"] for p in out["papers"]] == ["2401.00002v1"]

    def test_the_rerank_floats_a_buried_actionable_paper(self, tmp_path: Path) -> None:
        """The lower-ranked paper is the actionable one. Without the rerank the agent
        sees it second, or — at a limit of 1 — not at all."""
        with PaperStore(tmp_path / "papers.db") as store:
            run_id = _seed(store)  # 00001 scores 0.9, 00002 scores 0.4
            store.save_llm_scores(
                run_id,
                {
                    "2401.00001v1": {"llm_score": 1, "llm_reason": ""},
                    "2401.00002v1": {"llm_score": 3, "llm_reason": ""},
                },
            )
            out = ranked_papers_payload(store, limit=1, triage_threshold=2, rerank=True)
        assert [p["arxiv_id"] for p in out["papers"]] == ["2401.00002v1"]

    def test_an_ungated_repo_falls_back_to_the_heuristic_tiers(self, tmp_path: Path) -> None:
        """`triage_threshold=None` is what a repo that never ran the gate passes, and it
        must mean "the heuristic thresholds are the only rule there is" rather than "gate
        on a column that is null everywhere" — which would hand back an empty list
        instead of the ranking the repo does have."""
        with PaperStore(tmp_path / "papers.db") as store:
            _seed(store)
            out = ranked_papers_payload(store, limit=10, triage_threshold=None)
        assert [p["arxiv_id"] for p in out["papers"]] == ["2401.00001v1"]
        assert [p["arxiv_id"] for p in out["maybe_relevant"]] == ["2401.00002v1"]

    def test_limit_cannot_reach_past_the_window(self, tmp_path: Path) -> None:
        """`top_n` is what RepoRadar was willing to display; `limit` only trims it. A
        paper outside the window is one the product declined to show, and a caller
        asking for more must not be able to promote it."""
        with PaperStore(tmp_path / "papers.db") as store:
            _seed(store)
            out = ranked_papers_payload(store, limit=50, top_n=1)
        assert [p["arxiv_id"] for p in out["papers"]] == ["2401.00001v1"]


class TestSearchSaysHowMuchThereWasToSearch:
    def test_the_corpus_size_travels_with_the_results(self, tmp_path: Path) -> None:
        """A caller that gets three hits cannot otherwise tell a narrow CORPUS from a
        narrow QUERY, and those call for opposite next moves — search again, or stop
        expecting this store to know. It is also the whole variable in P27's wide arm,
        where 48 of one agent's 87 tool calls searched a corpus of about a dozen papers
        while the product's holds everything ever fetched."""
        with PaperStore(tmp_path / "papers.db") as store:
            _seed(store)
            out = search_corpus_payload(store, "concrete method", limit=10)
        assert out["corpus_size"] == 2
        assert out["count"] <= out["corpus_size"]

    def test_it_counts_the_whole_corpus_not_the_latest_run(self, tmp_path: Path) -> None:
        """`search_papers` reads `get_all_papers`, which spans every run ever made — that
        is what makes it a different tool from `get_ranked_papers`, and what the wide arm
        widens."""
        with PaperStore(tmp_path / "papers.db") as store:
            _seed(store)  # 2 papers, both scored in the run
            store.upsert_paper(_paper("2401.00003v1", "Never ranked"))
            out = search_corpus_payload(store, "paper", limit=10)
        assert out["corpus_size"] == 3


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


class TestTheServerCanInitialiseARepository:
    """`setup_repo`'s body, which is what removes the terminal from the plugin's setup.

    Exercised through the pure helper for the same reason as everything else in this
    file: CI installs `--extra dev --extra evals` and never has the `mcp` SDK, so the
    tool wrapper is unreachable here. `scripts/mcp_smoke.py` covers the wrapper against
    a real fresh resolve.
    """

    def test_it_asks_for_categories_rather_than_guessing(self, tmp_path: Path) -> None:
        """The one field no benchmark number justifies. A default here is the expensive
        kind of convenience: cs.LG/cs.CL fits an ML repository and no other, and a wrong
        list starves every stage downstream."""
        from reporadar.mcp_server import setup_repo_action

        (tmp_path / "README.md").write_text("# demo", encoding="utf-8")
        result = setup_repo_action(tmp_path, tmp_path / ".reporadar.yml")

        assert result["status"] == "needs_input"
        assert result["missing"] == ["categories"]
        assert not (tmp_path / ".reporadar.yml").exists(), "must not write a guessed config"
        # The caller is handed evidence to choose from, not just an error.
        assert "keywords" in result["repo_profile"]
        assert result["retry"]["tool"] == "setup_repo"

    def test_it_writes_the_categories_it_was_given(self, tmp_path: Path) -> None:
        from reporadar.config import load_config
        from reporadar.mcp_server import setup_repo_action

        (tmp_path / "README.md").write_text("# demo", encoding="utf-8")
        config_path = tmp_path / ".reporadar.yml"
        result = setup_repo_action(tmp_path, config_path, categories=["cs.CR", "cs.DC"])

        assert result["status"] == "ok"
        assert config_path.exists()
        assert (tmp_path / ".reporadar").is_dir()
        assert load_config(config_path).arxiv.categories == ["cs.CR", "cs.DC"]

    def test_the_written_config_keeps_the_comments_that_justify_it(self, tmp_path: Path) -> None:
        """The categories line is rewritten in place rather than round-tripping the YAML.
        Every other value in the measured config carries the measurement behind it in a
        comment, and a dump-and-reload would silently drop all of them."""
        from reporadar.mcp_server import setup_repo_action

        (tmp_path / "README.md").write_text("# demo", encoding="utf-8")
        config_path = tmp_path / ".reporadar.yml"
        setup_repo_action(tmp_path, config_path, categories=["math.OC"])

        body = config_path.read_text(encoding="utf-8")
        assert "categories: [math.OC]" in body
        assert "CHANGE THIS" in body, "the measured template's own guidance is gone"
        assert body.count("#") > 50, "the config lost the comments that carry its evidence"

    def test_it_does_not_overwrite_an_existing_config(self, tmp_path: Path) -> None:
        from reporadar.mcp_server import setup_repo_action

        config_path = tmp_path / ".reporadar.yml"
        config_path.write_text("repo_path: .\n", encoding="utf-8")
        result = setup_repo_action(tmp_path, config_path, categories=["cs.LG"])

        assert result["status"] == "already_configured"
        assert config_path.read_text(encoding="utf-8") == "repo_path: .\n"


class TestUnconfiguredIsAResultNotACrash:
    def test_the_payload_names_the_tool_that_fixes_it(self, tmp_path: Path) -> None:
        """`rr mcp` used to exit 1 here with the fix on stderr, where no MCP client shows
        it — the user saw "server failed to start" and never learned the cause."""
        from reporadar.mcp_server import not_configured_payload

        payload = not_configured_payload(tmp_path / ".reporadar.yml")
        assert payload["status"] == "not_configured"
        assert payload["retry"]["tool"] == "setup_repo"
        assert str(tmp_path) in payload["config_path"]


class TestProgressReachesTheClient:
    """The heartbeat is load-bearing: Copilot CLI's 180 s per-request timeout is reset by
    every progress notification, so a silent multi-minute collection is cancelled."""

    def test_every_pipeline_message_becomes_one_numbered_event(self) -> None:
        from reporadar.mcp_server import McpReporter

        sent: list[tuple[int, str]] = []
        reporter = McpReporter(emit=lambda n, m: sent.append((n, m)))

        reporter.info("Profiling repo: /x")
        reporter.warn("  no abstract for 2 papers")
        reporter.info("Triaging top 50 papers")

        assert [n for n, _ in sent] == [1, 2, 3]
        assert sent[0][1] == "Profiling repo: /x"
        assert sent[1][1] == "no abstract for 2 papers", "leading whitespace should be trimmed"
        assert reporter.messages == [m for _, m in sent]

    def test_blank_messages_do_not_burn_a_progress_step(self) -> None:
        from reporadar.mcp_server import McpReporter

        sent: list[tuple[int, str]] = []
        reporter = McpReporter(emit=lambda n, m: sent.append((n, m)))
        reporter.info("")
        reporter.info("   ")
        reporter.info("real")
        assert sent == [(1, "real")]

    def test_a_failing_progress_send_does_not_kill_the_collection(self) -> None:
        """Found by running the tool rather than reasoning about it: `report_progress`
        raises when there is no request context, and the exception escaped the Reporter
        and aborted the whole pipeline. Minutes of network and LLM work discarded because
        a status line could not be delivered."""
        from reporadar.mcp_server import McpReporter

        def hostile(n: int, message: str) -> None:
            raise ValueError("Context is not available outside of a request")

        reporter = McpReporter(emit=hostile)
        reporter.info("Profiling repo: /x")  # must not raise
        reporter.warn("something")
        assert reporter.messages == ["Profiling repo: /x", "something"], (
            "the run should still be recorded even when nobody could be told about it"
        )
