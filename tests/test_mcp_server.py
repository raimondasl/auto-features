"""Tests for reporadar.mcp_server — the pure data-gathering helpers.

These exercise the MCP tool bodies without the optional `mcp` SDK (the helpers
don't import it; only build_server/run_stdio do).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

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

    def test_warnings_are_kept_apart_from_progress(self) -> None:
        """A configured stage that could not run changes what the digest means, and it is
        the difference between a thin result and a thin literature. Flattened into sixty
        progress lines it is gone the moment the next one lands."""
        from reporadar.mcp_server import McpReporter

        reporter = McpReporter(emit=lambda n, m: None)
        reporter.info("Profiling repo: /x")
        reporter.warn("  HyDE discovery unavailable: index not found")
        reporter.info("Collecting from arxiv")

        assert reporter.warnings == ["HyDE discovery unavailable: index not found"]
        # Still in the narration too -- the user watching progress should see it happen.
        assert reporter.messages == [
            "Profiling repo: /x",
            "HyDE discovery unavailable: index not found",
            "Collecting from arxiv",
        ]

    def test_a_blank_warning_is_not_recorded_as_one(self) -> None:
        from reporadar.mcp_server import McpReporter

        reporter = McpReporter(emit=lambda n, m: None)
        reporter.warn("   ")
        assert reporter.warnings == []
        assert reporter.messages == []


class TestSetupRepoConfiguresTheGateForTheKeyYouHave:
    """Reported from a real install. `setup_repo` wrote `provider: claude` while the plugin's
    own README says one OpenAI key is enough — so following the documentation produced a
    config demanding a credential nobody had asked for, and the failure arrived minutes into
    collection as "no Claude API key configured"."""

    @pytest.fixture(autouse=True)
    def _isolated(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Detection reads the real credentials FILE as well as the environment — which is
        right for the product and means a test that only clears env vars would be answered
        by whatever the developer happens to have stored."""
        monkeypatch.setenv("REPORADAR_CONFIG_DIR", str(tmp_path / "creds"))
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    def test_an_openai_key_gets_an_openai_gate(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from reporadar.config import load_config
        from reporadar.mcp_server import setup_repo_action

        monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-aaaaaaaaaa")
        (tmp_path / "README.md").write_text("# demo", encoding="utf-8")
        config_path = tmp_path / ".reporadar.yml"

        result = setup_repo_action(tmp_path, config_path, categories=["cs.LG"])

        assert result["gate_provider"] == "openai"
        assert result["gate_key_present"] is True
        cfg = load_config(config_path)
        assert cfg.suggestions.provider == "openai"
        # The MODEL travels with the provider. `provider: openai` alone falls through to the
        # gpt-4o-mini default, which is not a configuration any published number describes.
        assert cfg.suggestions.openai_model == "gpt-5.6-luna"

    def test_an_anthropic_key_gets_the_measured_claude_gate(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from reporadar.config import load_config
        from reporadar.mcp_server import setup_repo_action

        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-bbbbbbbbbb")
        (tmp_path / "README.md").write_text("# demo", encoding="utf-8")
        config_path = tmp_path / ".reporadar.yml"

        result = setup_repo_action(tmp_path, config_path, categories=["cs.LG"])
        assert result["gate_provider"] == "claude"
        assert load_config(config_path).suggestions.provider == "claude"

    def test_with_no_key_at_all_it_says_so_and_names_the_command(self, tmp_path: Path) -> None:
        """Collection still runs, but ungated, which measured net@2 -11 — so a caller that
        reports the digest without mentioning it is describing the wrong thing."""
        from reporadar.mcp_server import setup_repo_action

        (tmp_path / "README.md").write_text("# demo", encoding="utf-8")
        result = setup_repo_action(tmp_path, tmp_path / ".reporadar.yml", categories=["cs.LG"])
        assert result["gate_key_present"] is False
        assert "rr auth" in result["next"] and "-11" in result["next"]

    def test_an_explicit_provider_overrides_the_detection(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from reporadar.config import load_config
        from reporadar.mcp_server import setup_repo_action

        monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-aaaaaaaaaa")
        (tmp_path / "README.md").write_text("# demo", encoding="utf-8")
        config_path = tmp_path / ".reporadar.yml"

        setup_repo_action(tmp_path, config_path, categories=["cs.LG"], provider="claude")
        assert load_config(config_path).suggestions.provider == "claude"

    def test_the_directory_being_configured_is_reported(self, tmp_path: Path) -> None:
        """The server's working directory is chosen by the editor, and has in practice been
        the plugin's own install directory rather than the user's project. Until the server
        asks the client for its roots, reporting the path is what lets a caller catch it."""
        from reporadar.mcp_server import setup_repo_action

        (tmp_path / "README.md").write_text("# demo", encoding="utf-8")
        result = setup_repo_action(tmp_path, tmp_path / ".reporadar.yml", categories=["cs.LG"])
        assert result["repo_path"] == str(tmp_path)


class TestFindingTheRepositoryTheClientMeans:
    """The server's working directory is chosen by the editor, and has in practice been the
    plugin's own install directory — so a digest was built for the plugin rather than the
    user's code. MCP roots is the protocol's answer; these cover the choosing."""

    def test_a_file_uri_becomes_a_path(self) -> None:
        from reporadar.mcp_server import root_uri_to_path

        assert root_uri_to_path("file:///home/me/proj") == Path("/home/me/proj")

    @pytest.mark.skipif(sys.platform != "win32", reason="drive-letter handling is Windows-only")
    def test_a_windows_uri_survives_the_leading_slash(self) -> None:
        """`file:///C:/x` carries a slash before the drive letter that `Path` alone mangles.
        Asserted only on Windows: `url2pathname` has no reason to strip it elsewhere, and a
        POSIX server receiving a Windows URI is not a case that arises -- a remote VS Code
        sends the remote paths, not the ones on the machine you are sitting at."""
        from reporadar.mcp_server import root_uri_to_path

        assert root_uri_to_path("file:///C:/Users/me/proj") == Path("C:/Users/me/proj")

    def test_percent_escapes_are_decoded(self) -> None:
        from reporadar.mcp_server import root_uri_to_path

        got = root_uri_to_path("file:///home/me/my%20proj")
        assert got is not None and got.name == "my proj"

    def test_a_non_file_root_is_ignored_rather_than_guessed_at(self) -> None:
        from reporadar.mcp_server import root_uri_to_path

        assert root_uri_to_path("https://example.com/repo") is None

    def test_the_root_containing_the_working_directory_wins(self, tmp_path: Path) -> None:
        """A multi-root workspace offers several and the protocol does not say which is
        current. The one we were started inside is the best available evidence."""
        from reporadar.mcp_server import choose_repo_root

        other, here = tmp_path / "other", tmp_path / "here"
        for d in (other, here):
            d.mkdir()
        assert choose_repo_root([other, here], cwd=here / "src") == here

    def test_the_deepest_containing_root_wins_when_they_nest(self, tmp_path: Path) -> None:
        from reporadar.mcp_server import choose_repo_root

        outer = tmp_path / "outer"
        inner = outer / "packages" / "app"
        inner.mkdir(parents=True)
        assert choose_repo_root([outer, inner], cwd=inner / "src") == inner

    def test_a_single_root_is_taken_even_from_elsewhere(self, tmp_path: Path) -> None:
        from reporadar.mcp_server import choose_repo_root

        only = tmp_path / "only"
        only.mkdir()
        assert choose_repo_root([only], cwd=tmp_path / "unrelated") == only

    def test_roots_that_are_not_directories_are_skipped(self, tmp_path: Path) -> None:
        from reporadar.mcp_server import choose_repo_root

        missing = tmp_path / "gone"
        real = tmp_path / "real"
        real.mkdir()
        assert choose_repo_root([missing, real], cwd=tmp_path) == real

    def test_nothing_usable_means_nothing_rather_than_a_guess(self, tmp_path: Path) -> None:
        """Falling back to the working directory is the caller's job, and it says so in the
        payload. Inventing a root here would hide which one it was."""
        from reporadar.mcp_server import choose_repo_root

        assert choose_repo_root([tmp_path / "gone"], cwd=tmp_path) is None
        assert choose_repo_root([], cwd=tmp_path) is None

    def test_a_told_path_must_actually_exist(self, tmp_path: Path) -> None:
        """A typo'd path should be refused, not recorded and then used for the rest of the
        session — which would replace a wrong directory with a nonexistent one."""
        from reporadar.mcp_server import setup_repo_action

        (tmp_path / "README.md").write_text("# demo", encoding="utf-8")
        # setup_repo_action itself takes a resolved path; the validation lives in the tool
        # wrapper, so this pins the pure helper's contract: it writes where it is told.
        target = tmp_path / "sub"
        target.mkdir()
        result = setup_repo_action(target, target / ".reporadar.yml", categories=["cs.LG"])
        assert result["repo_path"] == str(target)
        assert (target / ".reporadar.yml").exists()


class TestItRefusesToGuessWhichRepositoryYouMean:
    """The bug that survived two releases: the server inferred the project from its working
    directory, which an editor sets to the plugin's own folder, and then answered as if that
    were the user's code. Guessing was the defect — so it stops instead."""

    def test_a_checkout_is_recognised(self, tmp_path: Path) -> None:
        from reporadar.mcp_server import looks_like_a_project

        assert not looks_like_a_project(tmp_path)
        (tmp_path / "pyproject.toml").write_text("[project]", encoding="utf-8")
        assert looks_like_a_project(tmp_path)

    def test_a_git_directory_alone_is_enough(self, tmp_path: Path) -> None:
        """A freshly cloned repo with nothing else in it is still a repo."""
        from reporadar.mcp_server import looks_like_a_project

        (tmp_path / ".git").mkdir()
        assert looks_like_a_project(tmp_path)

    def test_a_plugin_installation_is_recognised_as_one(self, tmp_path: Path) -> None:
        """Named separately from "not a project" because it is the failure that happened,
        and "this is a plugin installation" is far more use than "this is not a project"."""
        from reporadar.mcp_server import looks_like_a_plugin_install, looks_like_a_project

        (tmp_path / "plugin.json").write_text("{}", encoding="utf-8")
        (tmp_path / ".mcp.json").write_text("{}", encoding="utf-8")
        assert looks_like_a_plugin_install(tmp_path)
        assert not looks_like_a_project(tmp_path)

    def test_a_project_is_not_mistaken_for_a_plugin(self, tmp_path: Path) -> None:
        from reporadar.mcp_server import looks_like_a_plugin_install

        (tmp_path / "pyproject.toml").write_text("[project]", encoding="utf-8")
        assert not looks_like_a_plugin_install(tmp_path)


class TestCollectionRunsSomewhereAndSaysWhere:
    """`collect_payload` — what `update_corpus` does, out where it can be tested.

    The branch that matters is the one that only happens when something has gone wrong. The
    server cannot run dense discovery (it installs the light `[mcp]` extra), so with HyDE
    configured it delegates collection to a `uvx` environment that can — and when that fails,
    it must still collect. A plugin whose digest disappears because a subprocess would not
    start has traded a missing retrieval channel for a missing answer.
    """

    class _Result:
        run_id, stopped, queries, papers, scores = 3, None, ["q"], ["p"], ["s"]

    class _Report:
        def __init__(self) -> None:
            self.infos: list[str] = []
            self.warns: list[str] = []

        def info(self, message: str) -> None:
            self.infos.append(message)

        def warn(self, message: str) -> None:
            self.warns.append(message)

    def _ran_here(self, monkeypatch: pytest.MonkeyPatch) -> list[dict]:
        """Record in-process pipeline calls instead of making one."""
        import reporadar.pipeline

        calls: list[dict] = []

        def fake(cfg, **kwargs):  # noqa: ANN001, ANN003
            calls.append(kwargs)
            return TestCollectionRunsSomewhereAndSaysWhere._Result()

        monkeypatch.setattr(reporadar.pipeline, "run_pipeline", fake)
        return calls

    def _places(self, tmp_path: Path) -> dict:
        return {
            "repo": tmp_path,
            "config_path": tmp_path / ".reporadar.yml",
            "db": tmp_path / ".reporadar" / "papers.db",
        }

    def test_without_delegation_it_collects_here_and_names_the_environment(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from reporadar import delegate
        from reporadar.mcp_server import collect_payload

        calls = self._ran_here(monkeypatch)
        monkeypatch.setattr(
            delegate, "plan", lambda *a, **k: delegate.Plan(command=None, reason="no HyDE")
        )
        report = self._Report()

        payload = collect_payload(object(), report=report, **self._places(tmp_path))

        assert len(calls) == 1
        assert payload["papers"] == 1 and payload["run_id"] == 3
        assert "no HyDE" in payload["collected_in"]
        assert not report.warns

    def test_a_delegated_run_reports_the_environment_it_used(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from reporadar import delegate
        from reporadar.mcp_server import collect_payload

        calls = self._ran_here(monkeypatch)
        monkeypatch.setattr(
            delegate,
            "plan",
            lambda *a, **k: delegate.Plan(
                command=["uvx"], cwd=tmp_path, spec="reporadar-papers[hyde]==9.9.9", reason="r"
            ),
        )
        monkeypatch.setattr(
            delegate,
            "run",
            lambda plan, report: {
                "run_id": 8,
                "stopped": None,
                "queries": 7,
                "papers": 114,
                "scored": 114,
            },
        )

        payload = collect_payload(object(), report=self._Report(), **self._places(tmp_path))

        assert not calls, "delegating means NOT also collecting in this process"
        assert payload["papers"] == 114
        assert "reporadar-papers[hyde]==9.9.9" in payload["collected_in"]

    def test_a_failed_delegation_still_collects_and_says_what_was_lost(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The safety net. Whatever goes wrong in the subprocess — no network for PyPI, a
        broken cache, a version that will not resolve — the user still gets a digest, and
        is told it is the keyword-only one."""
        from reporadar import delegate
        from reporadar.mcp_server import collect_payload

        calls = self._ran_here(monkeypatch)
        monkeypatch.setattr(
            delegate,
            "plan",
            lambda *a, **k: delegate.Plan(command=["uvx"], cwd=tmp_path, spec="s", reason="r"),
        )

        def boom(plan, report):  # noqa: ANN001, ARG001
            raise delegate.DelegationError("could not start uvx: no such file")

        monkeypatch.setattr(delegate, "run", boom)
        report = self._Report()

        payload = collect_payload(object(), report=report, **self._places(tmp_path))

        assert len(calls) == 1, "the fallback must actually collect"
        assert payload["papers"] == 1
        assert "after the delegated run failed" in payload["collected_in"]
        assert any("keyword retrieval only" in w for w in report.warns)
        assert any("no such file" in w for w in report.warns), "say WHY it failed"

    def test_a_plan_that_cannot_delegate_forwards_its_warning(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`plan` explains the reasons the pipeline itself never sees — no uvx, no version
        to pin. Those reach the user only if this forwards them."""
        from reporadar import delegate
        from reporadar.mcp_server import collect_payload

        self._ran_here(monkeypatch)
        monkeypatch.setattr(
            delegate,
            "plan",
            lambda *a, **k: delegate.Plan(
                command=None, reason="uvx is not on PATH", warning="install uv, or run: ..."
            ),
        )
        report = self._Report()

        collect_payload(object(), report=report, **self._places(tmp_path))

        assert any("install uv" in w for w in report.warns)

    def test_the_pipeline_gets_the_repository_the_client_named(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Not `cfg.repo_path`. The server resolves the repository from the client's roots,
        and the in-process pipeline must be pointed at that one — the same rule the
        delegation check enforces on the child."""
        from reporadar import delegate
        from reporadar.mcp_server import collect_payload

        calls = self._ran_here(monkeypatch)
        monkeypatch.setattr(delegate, "plan", lambda *a, **k: delegate.Plan(command=None))
        places = self._places(tmp_path)

        collect_payload(object(), report=self._Report(), **places)

        assert calls[0]["repo_path"] == places["repo"]
        assert calls[0]["db_path"] == places["db"]
