"""Tests for reporadar.privacy — the `rr audit` registry, redaction, and drift guard."""

from __future__ import annotations

import re
from pathlib import Path

from reporadar.config import (
    ArxivConfig,
    EmailHookConfig,
    HooksConfig,
    PrivacyConfig,
    QueriesConfig,
    RankingConfig,
    RecommendationsConfig,
    RepoRadarConfig,
    SignalsConfig,
    SuggestionsConfig,
)
from reporadar.privacy import (
    DESTINATIONS,
    NONE,
    REPO_AND_CONTENT,
    REPO_DERIVED,
    active_destinations,
    audit_plan,
    compile_patterns,
    on_demand_destinations,
    redact,
    redact_all,
)
from reporadar.profiler import RepoProfile

SRC = Path(__file__).resolve().parent.parent / "src" / "reporadar"

# How an outbound call looks in this codebase. `arxiv.Client` covers the arxiv package
# (which uses requests internally); `SentenceTransformer(` covers the one destination
# that downloads weights rather than calling an API; the rest are the stdlib/urllib and
# requests paths every adapter and the LLM transport use.
_OUTBOUND_MARKERS = (
    "urllib.request.urlopen",
    "urlopen(",
    "arxiv.Client",
    "requests.get",
    "requests.post",
    "requests.Session",
    "httpx.",
    "http.client",
    "SentenceTransformer(",
    "subprocess.run",  # gh_issues shells out to the `gh` CLI
)

# Modules that match a marker but do not themselves reach a third party.
_NOT_A_DESTINATION = {
    "scheduler",  # subprocess: installs a crontab/schtasks entry locally
    "watcher",  # subprocess: desktop notification; its fetching is delegated
    "cli",  # subprocess/webbrowser: opens a local browser, delegates all fetching
    "semantic",  # imports embeddings' public API; the download is embeddings' to declare
    "vec_index",  # same: local sqlite-vec index over vectors someone else fetched
    "embedding_cache",  # same: stores vectors, fetches nothing
    # Borrows specter's private constants and its `load_or_fetch_vectors`, but calls it
    # with offline=True (evaluation.py:175) — `rr eval` re-scores historical papers and
    # must not make network calls to do it. Verified at specter.py:116-129.
    "evaluation",
}

# Text markers alone miss a module that reaches the network through a helper imported
# from another module — `specter` does exactly this, borrowing `_s2_batch_post` from
# `citations`, and it is one of the registry's own declared destinations. So a module
# that pulls a *private* name out of an outbound module is treated as outbound too.
# Private specifically: `from reporadar.store import PaperStore` is ordinary layering,
# but a leading-underscore helper crossing a module boundary is the shared-transport
# pattern. Propagation runs to a fixpoint, so a two-hop chain is caught as well.
_PRIVATE_IMPORT_RE = re.compile(r"from\s+reporadar\.([\w.]+)\s+import\s+([^\n()]+|\([^)]*\))")


def _module_name(path: Path) -> str:
    return str(path.relative_to(SRC).with_suffix("")).replace("\\", "/").replace("/", ".")


def _modules_making_outbound_calls() -> set[str]:
    """Walk the package for modules that can talk to something outside the process."""
    sources: dict[str, str] = {}
    for path in SRC.rglob("*.py"):
        # privacy.py names the markers in its own prose; __init__ files are re-exports.
        if path.name in ("__init__.py", "privacy.py"):
            continue
        sources[_module_name(path)] = path.read_text(encoding="utf-8", errors="replace")

    found = {
        name
        for name, text in sources.items()
        if any(marker in text for marker in _OUTBOUND_MARKERS)
    }

    changed = True
    while changed:
        changed = False
        for name, text in sources.items():
            if name in found:
                continue
            for target, imported in _PRIVATE_IMPORT_RE.findall(text):
                if target not in found:
                    continue
                if any(part.strip().startswith("_") for part in imported.split(",")):
                    found.add(name)
                    changed = True
                    break

    return found - _NOT_A_DESTINATION


class TestRegistryDoesNotDrift:
    def test_every_outbound_module_is_declared(self) -> None:
        """A privacy feature that rots is worse than none, because people act on it.

        `rr audit` tells a user on a proprietary codebase what leaves their machine.
        A hand-maintained list would be wrong the first time someone adds a source and
        forgets — so adding an integration without declaring it here fails CI instead.

        The detector is static, so state its limit honestly: it recognises the request
        shapes in `_OUTBOUND_MARKERS` and propagates through private helpers imported
        from an already-outbound module. A module reaching the network by some third
        route — a new HTTP library, a public wrapper — would still slip past, so this
        raises the cost of an undeclared destination rather than making one impossible.
        """
        declared = {d.module for d in DESTINATIONS}
        actual = _modules_making_outbound_calls()
        undeclared = actual - declared
        assert not undeclared, (
            f"these modules make outbound calls but are missing from "
            f"privacy.DESTINATIONS: {sorted(undeclared)}. Add a Destination "
            f"describing what leaves, or add the module to _NOT_A_DESTINATION with "
            f"a reason."
        )

    def test_the_guard_would_actually_catch_a_new_module(self) -> None:
        # Proves the detector works, so a green run means something.
        found = _modules_making_outbound_calls()
        assert "sources.dblp" in found
        assert "signals.hn" in found
        assert "llm_client" in found

    def test_the_guard_sees_a_module_that_borrows_another_s_transport(self) -> None:
        """`specter` makes no request of its own — it imports `_s2_batch_post`.

        A text-marker detector misses it entirely, which meant the guard could not see
        one of the registry's *own* declared destinations, and would equally miss a new
        module added the same way. Propagation through private cross-module imports is
        what closes that, so assert it directly rather than trusting the total.
        """
        found = _modules_making_outbound_calls()
        assert "specter" in found, "propagation through a borrowed helper regressed"
        specter_src = (SRC / "specter.py").read_text(encoding="utf-8")
        assert not any(m in specter_src for m in _OUTBOUND_MARKERS), (
            "specter now makes a direct call, so this test no longer proves propagation"
        )

    def test_the_guard_sees_a_weights_download_not_just_an_api_call(self) -> None:
        # `embeddings` downloads all-MiniLM-L6-v2 from huggingface.co on first use. It
        # is a destination even though it never calls an API, and it was missed until
        # the marker list stopped assuming "outbound" means "HTTP request".
        assert "embeddings" in _modules_making_outbound_calls()
        assert "embeddings" in {d.module for d in DESTINATIONS}

    def test_no_destination_is_declared_for_a_module_that_vanished(self) -> None:
        for dest in DESTINATIONS:
            module_path = SRC / (dest.module.replace(".", "/") + ".py")
            assert module_path.exists(), f"{dest.module} is declared but does not exist"

    def test_every_destination_says_what_it_sends(self) -> None:
        for dest in DESTINATIONS:
            assert dest.sends.strip(), f"{dest.service} declares no payload"
            assert dest.enabled_by.strip(), f"{dest.service} declares no trigger"


class TestActiveDestinations:
    def test_a_default_config_reaches_few_places(self) -> None:
        active = active_destinations(RepoRadarConfig())
        services = {d.service for d in active}
        assert "arXiv" in services  # the core source, always
        # Off by default and must not appear.
        assert "OpenAlex" not in services
        assert "DBLP" not in services
        assert "Anthropic" not in services
        assert "Hacker News (Algolia)" not in services

    def test_enabling_a_source_shows_up(self) -> None:
        cfg = RepoRadarConfig(sources=["arxiv", "dblp", "openalex"])
        services = {d.service for d in active_destinations(cfg)}
        assert "DBLP" in services and "OpenAlex" in services

    def test_the_llm_provider_decides_whether_anything_leaves(self) -> None:
        # The sharpest distinction in the whole feature: the same prompt either goes
        # to a third party or to a process on your own machine.
        claude = RepoRadarConfig(suggestions=SuggestionsConfig(provider="claude"))
        ollama = RepoRadarConfig(suggestions=SuggestionsConfig(provider="ollama"))
        claude_dests = {d.service: d for d in active_destinations(claude)}
        ollama_dests = {d.service: d for d in active_destinations(ollama)}
        assert claude_dests["Anthropic"].sensitivity == REPO_AND_CONTENT
        assert "Anthropic" not in ollama_dests
        assert ollama_dests["Ollama (local)"].sensitivity == NONE

    def test_biorxiv_is_reported_as_sending_nothing_repo_derived(self) -> None:
        # Verified against sources/biorxiv.py: it requests a date interval and filters
        # locally, so unlike every other source it transmits no inferred vocabulary.
        cfg = RepoRadarConfig(sources=["arxiv", "biorxiv"])
        biorxiv = next(d for d in active_destinations(cfg) if d.service == "bioRxiv")
        assert biorxiv.sensitivity == NONE

    def test_most_sensitive_first(self) -> None:
        cfg = RepoRadarConfig(
            sources=["arxiv", "biorxiv"], suggestions=SuggestionsConfig(provider="claude")
        )
        order = [d.sensitivity for d in active_destinations(cfg)]
        rank = {REPO_AND_CONTENT: 0, REPO_DERIVED: 1}
        assert order[0] == REPO_AND_CONTENT
        assert order == sorted(order, key=lambda s: rank.get(s, 9))

    def test_notify_only_appears_once_a_hook_is_configured(self) -> None:
        bare = {d.service for d in on_demand_destinations(RepoRadarConfig())}
        assert "Slack / Discord / SMTP" not in bare
        wired = RepoRadarConfig(hooks=HooksConfig(slack_webhook_url="https://hooks.example/x"))
        assert "Slack / Discord / SMTP" in {d.service for d in on_demand_destinations(wired)}

    def test_an_email_only_hook_counts_too(self) -> None:
        cfg = RepoRadarConfig(hooks=HooksConfig(email=EmailHookConfig(smtp_host="smtp.example")))
        assert "Slack / Discord / SMTP" in {d.service for d in on_demand_destinations(cfg)}

    def test_a_broken_gate_reports_rather_than_hides(self) -> None:
        # Under-reporting is the one failure this feature must never have.
        class Broken:
            def __getattr__(self, name: str) -> None:
                raise RuntimeError("boom")

        every = active_destinations(Broken(), include_on_demand=True)
        assert len(every) == len(DESTINATIONS)

    def test_on_demand_destinations_are_reported_separately_not_hidden(self) -> None:
        """`rr gh-issues` posts paper text into your repo's issues.

        It is not part of `rr update`, but omitting it from a privacy audit because of
        that would hide a real outbound flow from the person asking what leaves.
        """
        cfg = RepoRadarConfig(hooks=HooksConfig(slack_webhook_url="https://hooks.example/x"))
        update_run = {d.service for d in active_destinations(cfg)}
        on_demand = {d.service for d in on_demand_destinations(cfg)}
        assert "GitHub" not in update_run
        assert "GitHub" in on_demand
        assert "Slack / Discord / SMTP" in on_demand


class TestRedaction:
    def test_removes_a_literal_term(self) -> None:
        patterns = compile_patterns(["projectatlas"])
        assert redact("projectatlas retrieval pipeline", patterns) == "retrieval pipeline"

    def test_is_case_insensitive(self) -> None:
        patterns = compile_patterns(["Atlas"])
        assert "atlas" not in redact("ATLAS and atlas and Atlas", patterns).lower()

    def test_regexes_need_an_explicit_prefix(self) -> None:
        patterns = compile_patterns([r"re:proj-\d+"])
        assert redact("proj-1234 embedding search", patterns) == "embedding search"

    def test_entries_are_literal_by_default(self) -> None:
        """`C++` is a *valid* Python 3.11 regex — `++` is a possessive quantifier.

        Compiled as a pattern it redacts only the letter C, leaving `++ parser` in the
        query while the user believes their codename was scrubbed. Silent partial
        redaction is the worst thing a privacy filter can do: it is indistinguishable
        from working. So entries mean exactly what they say.
        """
        patterns = compile_patterns(["C++", "thing (v2)"])
        assert redact("C++ parser", patterns) == "parser"
        assert redact("thing (v2) results", patterns) == "results"

    def test_a_literal_entry_is_not_interpreted_as_a_pattern(self) -> None:
        # `a.c` must not match `abc`.
        assert redact("abc and a.c", compile_patterns(["a.c"])) == "abc and"

    def test_an_invalid_regex_after_the_prefix_falls_back_to_literal(self) -> None:
        # Dropping it would leave the user believing a term is being scrubbed.
        assert redact("a[b unbalanced", compile_patterns(["re:a[b"])) == "unbalanced"

    def test_collapses_the_whitespace_it_leaves(self) -> None:
        patterns = compile_patterns(["middle"])
        assert redact("start middle end", patterns) == "start end"

    def test_no_patterns_is_a_no_op(self) -> None:
        assert redact("anything at all", []) == "anything at all"
        assert redact_all(["a", "b"], []) == ["a", "b"]

    def test_entries_emptied_by_redaction_are_dropped(self) -> None:
        # A query that redacts to nothing must not be sent as an empty search.
        patterns = compile_patterns(["secret"])
        assert redact_all(["secret", "secret models"], patterns) == ["models"]

    def test_empty_pattern_strings_are_ignored(self) -> None:
        # An empty regex matches everywhere and would blank every query.
        assert redact("keep this text", compile_patterns(["", "  "])) == "keep this text"


class TestAuditPlan:
    def _profile(self) -> RepoProfile:
        return RepoProfile(
            keywords=[("projectatlas", 0.9), ("retrieval", 0.8)],
            anchors=["torch", "projectatlas-sdk"],
            domains=["NLP"],
        )

    def test_reports_the_queries_that_would_be_sent(self) -> None:
        plan = audit_plan(RepoRadarConfig(), self._profile(), ["all:retrieval", "all:atlas"])
        assert plan["queries"] == ["all:retrieval", "all:atlas"]

    def test_reports_the_diff_between_redacted_and_unredacted_queries(self) -> None:
        # The audit reports the *real* post-redaction queries plus the same build with
        # redaction off, so "what did my filter actually remove" is answerable.
        cfg = RepoRadarConfig(privacy=PrivacyConfig(redact=["projectatlas"]))
        plan = audit_plan(
            cfg,
            self._profile(),
            ["all:retrieval"],
            queries_unredacted=["all:projectatlas", "all:retrieval"],
        )
        assert plan["queries"] == ["all:retrieval"]
        assert plan["redaction_changed_anything"] is True
        assert plan["n_patterns"] == 1

    def test_says_when_redaction_changed_nothing(self) -> None:
        # Patterns that match nothing protect nothing, and a user who believes
        # otherwise is worse off than one who knows.
        cfg = RepoRadarConfig(privacy=PrivacyConfig(redact=["neverappears"]))
        plan = audit_plan(
            cfg, self._profile(), ["all:retrieval"], queries_unredacted=["all:retrieval"]
        )
        assert plan["redaction_changed_anything"] is False
        assert plan["n_patterns"] == 1

    def test_reports_the_real_queries_even_without_an_unredacted_baseline(self) -> None:
        # Omitting the baseline must never cause the audit to *re-redact* and print
        # strings that differ from what build_queries actually hands the sources.
        cfg = RepoRadarConfig(privacy=PrivacyConfig(redact=["retrieval"]))
        plan = audit_plan(cfg, self._profile(), ["all:retrieval"])
        assert plan["queries"] == ["all:retrieval"]
        assert plan["redaction_changed_anything"] is False

    def test_reports_whether_source_scanning_is_on(self) -> None:
        cfg = RepoRadarConfig()
        assert audit_plan(cfg, self._profile(), [])["scans_source"] is False
        cfg.profiler.scan_source = True
        assert audit_plan(cfg, self._profile(), [])["scans_source"] is True

    def test_redacts_the_reported_keywords_and_anchors(self) -> None:
        cfg = RepoRadarConfig(privacy=PrivacyConfig(redact=["projectatlas"]))
        plan = audit_plan(cfg, self._profile(), [])
        assert not any("projectatlas" in k for k in plan["keywords"])
        assert "torch" in plan["anchors"]


class TestPrivacyConfig:
    def test_defaults_to_no_redaction(self) -> None:
        assert RepoRadarConfig().privacy.redact == []

    def test_parsed_from_yaml(self, tmp_path: Path) -> None:
        from reporadar.config import load_config

        cfg_file = tmp_path / ".reporadar.yml"
        cfg_file.write_text(
            "repo_path: .\nprivacy:\n  redact:\n    - projectatlas\n    - 're:proj-\\d+'\n",
            encoding="utf-8",
        )
        cfg = load_config(cfg_file)
        assert cfg.privacy.redact == ["projectatlas", r"re:proj-\d+"]

    def test_patterns_compile_from_config(self) -> None:
        cfg = RepoRadarConfig(privacy=PrivacyConfig(redact=["atlas", r"re:proj-\d+"]))
        patterns = compile_patterns(cfg.privacy.redact)
        assert len(patterns) == 2
        assert all(isinstance(p, re.Pattern) for p in patterns)


class TestSensitivityCoverage:
    def test_every_declared_sensitivity_is_a_known_class(self) -> None:
        known = {REPO_DERIVED, REPO_AND_CONTENT, "interests", NONE}
        for dest in DESTINATIONS:
            assert dest.sensitivity in known, f"{dest.service}: {dest.sensitivity}"

    def test_the_signals_and_ranking_gates_work(self) -> None:
        cfg = RepoRadarConfig(
            ranking=RankingConfig(w_specter=1.0, w_citation_proximity=1.0),
            signals=SignalsConfig(hackernews=True),
            recommendations=RecommendationsConfig(enabled=True),
        )
        services = {d.service for d in active_destinations(cfg)}
        assert "Semantic Scholar (SPECTER2 vectors)" in services
        assert "Semantic Scholar (references)" in services
        assert "Hacker News (Algolia)" in services
        assert "Semantic Scholar (recommendations)" in services


class TestQueryRedactionWiring:
    """Redaction has to reach the actual outbound query strings, not just a helper."""

    def _profile(self, keywords: list[str]) -> RepoProfile:
        return RepoProfile(
            keywords=[(k, 1.0) for k in keywords], anchors=[], domains=[], source_signals={}
        )

    def test_term_is_absent_from_every_built_query(self) -> None:
        from reporadar.collector import build_queries

        queries = build_queries(
            self._profile(["projectatlas", "retrieval"]),
            QueriesConfig(seed=["projectatlas ranking"], redact=["projectatlas"]),
            ArxivConfig(categories=["cs.IR"]),
        )
        assert queries, "redaction must not wipe out the query set entirely"
        assert not any("projectatlas" in q for q in queries)
        # And the surviving terms are still there: redaction, not destruction.
        assert any("ranking" in q for q in queries)
        assert any("retrieval" in q for q in queries)

    def test_redaction_leaves_no_degenerate_query(self) -> None:
        # Redacting the assembled string instead of the terms would emit
        # `(all: ) AND (cat:cs.IR)` - a broken query, not a private one.
        from reporadar.collector import build_queries

        queries = build_queries(
            self._profile(["projectatlas"]),
            QueriesConfig(seed=["projectatlas"], redact=["projectatlas"]),
            ArxivConfig(categories=["cs.IR"]),
        )
        for q in queries:
            assert "all:" not in q or re.search(r"all:\s*\"?\s*\)", q) is None, q

    def test_config_load_mirrors_privacy_terms_onto_queries(self, tmp_path: Path) -> None:
        # The wiring that makes redaction unmissable: build_queries reads it off
        # QueriesConfig, so the mirror is what connects `privacy.redact` to reality.
        from reporadar.config import load_config

        cfg_file = tmp_path / ".reporadar.yml"
        cfg_file.write_text(
            "repo_path: .\nprivacy:\n  redact:\n    - projectatlas\n", encoding="utf-8"
        )
        cfg = load_config(cfg_file)
        assert cfg.queries.redact == ["projectatlas"]
        assert cfg.suggestions.redact == ["projectatlas"]


class TestLLMPromptRedactionWiring:
    """The LLM path sends far more than query strings - it sends the profile."""

    def test_prompt_is_redacted_before_dispatch(self, monkeypatch) -> None:  # type: ignore[no-untyped-def]
        import reporadar.llm_client as llm_client

        captured: list[str] = []

        def fake_dispatch(prompt: str, cfg: object, max_tokens: int) -> str:
            captured.append(prompt)
            return "ok"

        monkeypatch.setattr(llm_client, "_dispatch", fake_dispatch)
        llm_client.complete(
            "This repo, ProjectAtlas, ranks documents.",
            SuggestionsConfig(provider="ollama", redact=["projectatlas"]),
        )
        assert captured, "dispatch was never reached"
        assert "ProjectAtlas" not in captured[0]
        assert "projectatlas" not in captured[0].lower()
        assert "ranks documents" in captured[0]

    def test_redaction_does_not_reformat_the_prompt_it_passes_through(
        self,
        monkeypatch,  # type: ignore[no-untyped-def]
    ) -> None:
        """Turning redaction on must not change how the prompt is *structured*.

        The real triage prompt is a multi-line rubric ending in a JSON instruction.
        A redactor that collapsed all whitespace would flatten it to a single line —
        the term would be gone, but the model would also be answering a differently
        formatted question, and nothing would say so.
        """
        import reporadar.llm_client as llm_client
        from reporadar.triage import build_triage_prompt

        captured: list[str] = []
        monkeypatch.setattr(
            llm_client,
            "_dispatch",
            lambda prompt, cfg, max_tokens: (captured.append(prompt), "ok")[1],
        )
        paper = {
            "title": "Learning to Rank for ProjectAtlas",
            "abstract": "We rank documents.",
            "categories": ["cs.IR"],
        }
        profile = RepoProfile(
            keywords=[("retrieval", 1.0)], anchors=["torch"], domains=["IR"], source_signals={}
        )
        prompt = build_triage_prompt(paper, profile)
        assert prompt.count("\n") > 5, "fixture no longer resembles the real prompt"

        llm_client.complete(prompt, SuggestionsConfig(provider="ollama", redact=["projectatlas"]))
        sent = captured[0]
        assert "ProjectAtlas" not in sent
        assert sent.count("\n") == prompt.count("\n"), "redaction reformatted the prompt"


class TestAuditCommand:
    """`rr audit` is the user-facing half; it must run offline and stay readable."""

    def _repo(self, tmp_path: Path, extra: str = "") -> Path:
        (tmp_path / "README.md").write_text(
            "# ProjectAtlas\n\nA retrieval and ranking service.\n", encoding="utf-8"
        )
        (tmp_path / ".reporadar.yml").write_text(
            f"repo_path: {tmp_path}\n"
            "arxiv:\n  categories: [cs.IR]\n"
            "queries:\n  seed: ['projectatlas ranking']\n" + extra,
            encoding="utf-8",
        )
        return tmp_path / ".reporadar.yml"

    def test_lists_destinations_and_the_real_queries(self, tmp_path: Path) -> None:
        from click.testing import CliRunner

        from reporadar.cli import cli

        cfg = self._repo(tmp_path)
        result = CliRunner().invoke(cli, ["audit", "--config", str(cfg)])
        assert result.exit_code == 0, result.output
        assert "arXiv" in result.output
        assert "export.arxiv.org" in result.output
        assert "all:" in result.output  # the actual query syntax that goes out
        assert "not configured" in result.output  # no privacy.redact in this config

    def test_reports_what_redaction_removed(self, tmp_path: Path) -> None:
        from click.testing import CliRunner

        from reporadar.cli import cli

        cfg = self._repo(tmp_path, "privacy:\n  redact:\n    - projectatlas\n")
        result = CliRunner().invoke(cli, ["audit", "--config", str(cfg)])
        assert result.exit_code == 0, result.output
        assert "projectatlas" not in result.output.lower().replace("- projectatlas", "")
        assert "pattern(s) active" in result.output

    def test_warns_when_the_patterns_match_nothing(self, tmp_path: Path) -> None:
        # A user who configures redaction and gets silence will assume it worked.
        from click.testing import CliRunner

        from reporadar.cli import cli

        cfg = self._repo(tmp_path, "privacy:\n  redact:\n    - neverappearsanywhere\n")
        result = CliRunner().invoke(cli, ["audit", "--config", str(cfg)])
        assert result.exit_code == 0, result.output
        assert "nothing matched" in result.output

    def test_json_output_is_serialisable_and_has_no_lambda_reprs(self, tmp_path: Path) -> None:
        import json as json_mod

        from click.testing import CliRunner

        from reporadar.cli import cli

        cfg = self._repo(tmp_path, "sources: [arxiv, dblp]\n")
        result = CliRunner().invoke(cli, ["audit", "--config", str(cfg), "--json"])
        assert result.exit_code == 0, result.output
        payload = json_mod.loads(result.output)
        assert {d["service"] for d in payload["destinations"]} >= {"arXiv", "DBLP"}
        assert payload["queries"], "the audit must report the real query strings"
        assert "<function" not in result.output, "a predicate leaked into the report"

    def test_output_survives_a_legacy_windows_console(self, tmp_path: Path) -> None:
        """Every printed string must be ASCII, as the rest of this CLI's output is.

        A destination description written with a typographic dash raises
        UnicodeEncodeError on a `cmd.exe` still at codepage 437, turning the privacy
        report into a traceback for exactly the cautious user it exists for. ASCII is
        the codebase's existing convention for printed strings, so asserting it here
        keeps the freshly-added `DESTINATIONS` prose from being the first exception.
        """
        from click.testing import CliRunner

        from reporadar.cli import cli

        cfg = self._repo(tmp_path, "sources: [arxiv, dblp, openalex]\nprivacy:\n  redact: [x]\n")
        result = CliRunner().invoke(cli, ["audit", "--config", str(cfg)])
        assert result.exit_code == 0, result.output
        result.output.encode("ascii")  # raises if any character is unrepresentable

    def test_the_audit_makes_no_network_call(self, tmp_path: Path) -> None:
        # It reports what *would* be sent. If it sent anything to find out, the
        # command would defeat its own purpose. The autouse _no_network fixture
        # asserts this at teardown; the run below is what gives it something to check.
        from click.testing import CliRunner

        from reporadar.cli import cli

        cfg = self._repo(tmp_path, "sources: [arxiv, dblp, openalex]\n")
        result = CliRunner().invoke(cli, ["audit", "--config", str(cfg)])
        assert result.exit_code == 0, result.output


class TestAuditMatchesWhatUpdateTransmits:
    """The audit's core claim: the strings it prints are the strings that go out.

    Asserting substrings ("all:" appears somewhere) cannot detect the ways that claim
    actually breaks — a profile built differently from the one `update` uses, or a
    truncated list. These build the transmitted set independently and demand equality.
    """

    def _repo(self, tmp_path: Path, cfg_extra: str) -> tuple[Path, Path]:
        repo = tmp_path / "repo"
        repo.mkdir()
        (repo / "README.md").write_text(
            "# Internal Search\n\nAn internal search service for document ranking.\n",
            encoding="utf-8",
        )
        (repo / "engine.py").write_text(
            "def blackbriar_rerank(docs):\n    return docs\n\n\n"
            "class BlackbriarPipeline:\n    pass\n",
            encoding="utf-8",
        )
        cfg_file = tmp_path / ".reporadar.yml"
        cfg_file.write_text(
            f"repo_path: {repo.as_posix()}\narxiv:\n  categories: [cs.IR]\n" + cfg_extra,
            encoding="utf-8",
        )
        return repo, cfg_file

    def _transmitted(self, repo: Path, cfg_file: Path) -> list[str]:
        from reporadar.collector import build_queries
        from reporadar.config import load_config
        from reporadar.profiler import profile_repo

        cfg = load_config(cfg_file)
        # Exactly how cli.update builds them (profile_repo(..., profiler_cfg=cfg.profiler)).
        return build_queries(profile_repo(repo, profiler_cfg=cfg.profiler), cfg.queries, cfg.arxiv)

    def test_prints_every_query_with_source_scanning_off(self, tmp_path: Path) -> None:
        from click.testing import CliRunner

        from reporadar.cli import cli

        repo, cfg_file = self._repo(tmp_path, "")
        result = CliRunner().invoke(cli, ["audit", "--config", str(cfg_file)])
        assert result.exit_code == 0, result.output
        for q in self._transmitted(repo, cfg_file):
            assert q in result.output, f"transmitted but never printed: {q}"

    def test_prints_every_query_with_source_scanning_on(self, tmp_path: Path) -> None:
        """The case that was broken: `scan_source: true` changes what `update` sends.

        The audit omitted `profiler_cfg`, so it reported a docs-only query set while
        `update` transmitted identifiers lifted from the source — the internal codenames
        a proprietary codebase runs this command to find. Worse, the report printed
        "profiler.scan_source is on" directly beneath a query list proving it was not.
        """
        from click.testing import CliRunner

        from reporadar.cli import cli

        repo, cfg_file = self._repo(tmp_path, "profiler:\n  scan_source: true\n")
        transmitted = self._transmitted(repo, cfg_file)
        assert any("blackbriar" in q.lower() for q in transmitted), (
            "fixture no longer exercises source scanning"
        )

        result = CliRunner().invoke(cli, ["audit", "--config", str(cfg_file)])
        assert result.exit_code == 0, result.output
        for q in transmitted:
            assert q in result.output, f"transmitted but never printed: {q}"

    def test_the_scan_source_notice_agrees_with_the_query_list(self, tmp_path: Path) -> None:
        # A report that contradicts itself is worse than one that says less.
        from click.testing import CliRunner

        from reporadar.cli import cli

        _repo, cfg_file = self._repo(tmp_path, "profiler:\n  scan_source: true\n")
        result = CliRunner().invoke(cli, ["audit", "--config", str(cfg_file)])
        assert "scan_source is on" in result.output
        assert "blackbriar" in result.output.lower(), (
            "the report claims source scanning is on but shows no source-derived term"
        )


class TestReviewFixes:
    """Regressions for defects the adversarial review confirmed."""

    def test_a_shell_hook_is_reported_as_a_destination(self) -> None:
        # `hooks.on_digest` runs an arbitrary command with RR_* env vars. It was the one
        # configured hook the notify predicate did not check, so a config using only it
        # was reported as reaching nothing at all.
        cfg = RepoRadarConfig(hooks=HooksConfig(on_digest="curl -d $RR_SUMMARY https://x/"))
        services = {d.service for d in on_demand_destinations(cfg)}
        assert "Your `hooks.on_digest` shell command" in services

    def test_the_shell_hook_is_rated_at_the_ceiling(self) -> None:
        # An arbitrary command can exfiltrate anything; a privacy report must not imply
        # a bound it cannot enforce.
        cfg = RepoRadarConfig(hooks=HooksConfig(on_digest="./send.sh"))
        hook = next(
            d for d in on_demand_destinations(cfg) if d.service.startswith("Your `hooks.on_digest")
        )
        assert hook.sensitivity == REPO_AND_CONTENT

    def test_no_shell_hook_means_no_such_destination(self) -> None:
        services = {d.service for d in on_demand_destinations(RepoRadarConfig())}
        assert not any(s.startswith("Your `hooks.on_digest") for s in services)

    def test_a_regex_matching_the_empty_string_is_rejected(self) -> None:
        """`re:x*` matches at every position, so substitution letter-spaces the query.

        `learning to rank` became `l e a r n i n g t o r a n k` and was transmitted like
        that: the term stays legible to any reader, the query is destroyed, and the user
        is told redaction is active. Worse than no redaction on both counts.
        """
        assert compile_patterns([r"re:x*"]) == []
        assert redact("learning to rank", compile_patterns([r"re:x*"])) == "learning to rank"

    def test_a_normal_regex_still_compiles(self) -> None:
        # The empty-match guard must not swallow legitimate patterns.
        patterns = compile_patterns([r"re:proj-\d+"])
        assert len(patterns) == 1
        assert redact("all:proj-42 ranking", patterns) == "all: ranking"

    def test_untouched_lines_keep_their_indentation(self) -> None:
        # redact() re-spaced every line, so enabling redaction stripped the indentation
        # from the triage rubric even on lines holding no redacted term — the exact
        # reformatting the docstring promised it avoided.
        text = "Score this paper:\n  0 = unrelated\n  1 = same topic\nReturn JSON."
        assert redact(text, compile_patterns(["projectatlas"])) == text

    def test_a_redacted_line_does_close_the_gap_it_left(self) -> None:
        # The flip side: a line that WAS cut gets re-spaced, so removal leaves no hole.
        out = redact("  1 = projectatlas topic", compile_patterns(["projectatlas"]))
        assert out == "1 = topic"

    def test_json_stays_parseable_when_the_config_warns(self, tmp_path: Path) -> None:
        # Validation warnings went to stdout, so `rr audit --json | jq` broke for any
        # config that triggered one.
        import json as json_mod

        from click.testing import CliRunner

        from reporadar.cli import cli
        from reporadar.config import load_config, validate_config

        repo = tmp_path / "repo"
        repo.mkdir()
        (repo / "README.md").write_text("# X\n\nranking service\n", encoding="utf-8")
        cfg_file = tmp_path / ".reporadar.yml"
        # Enabling OpenAlex without an api_key is a documented validate_config warning,
        # and is exactly the config a privacy-conscious user would audit.
        cfg_file.write_text(
            f"repo_path: {repo.as_posix()}\n"
            "sources: [arxiv, openalex]\n"
            "arxiv:\n  categories: [cs.IR]\n",
            encoding="utf-8",
        )
        assert validate_config(load_config(cfg_file)), "fixture no longer triggers a warning"

        result = CliRunner().invoke(cli, ["audit", "--config", str(cfg_file), "--json"])
        assert result.exit_code == 0, result.output
        json_mod.loads(result.stdout)  # raises if prose was printed above the document
