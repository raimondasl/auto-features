"""The v2 baseline prompt, and the two things it could have quietly destroyed.

`BASELINE_PROMPT_V2` lets the baseline recommend journal, conference and bioRxiv papers
instead of arXiv alone. That is a *witness-generator* change and owes no A/B validation. What
it does owe is proof that it left the **published comparator** exactly where it was, because
two mechanisms made that easy to get wrong:

* **`_cache_path` had no discriminator in it.** `_disc` is a staleness check, not an address:
  a mismatched entry is re-run and written back to the same file. So editing the prompt in
  place would not have invalidated the 34 stored answers, it would have OVERWRITTEN them.
  That is not hypothetical — `compiler`, `graph` and `storage` still hold a 128-character
  restoration note where their transcript used to be, after a 30-turn re-run displaced the
  12-turn entry on 2026-08-09. Their reasoning is gone for good.
* **`run_baseline` replays `_parse_recommendations` over cached `raw` on every hit.** So
  widening the parser to accept a DOI is not an additive change to future runs; it re-derives
  the ids of every stored run, and those ids are the gold set every published recall figure
  divides by.

The tests below pin both: v1's discriminator and cache path are byte-for-byte what they were,
and the widened parser yields exactly what the narrow one did on all 34 stored answers.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evals"))

import baseline as baseline_mod  # noqa: E402
from fill_cli_baseline import PINNED_DISCRIMINATOR  # noqa: E402

from reporadar.paper_id import canonical_ref, dedup_id, doi_key  # noqa: E402

CACHE = ROOT / "evals" / "cache" / "baseline"


def _block(items: list[dict]) -> str:
    return "prose before\n```json\n" + json.dumps(items) + "\n```"


class TestTheComparatorDidNotMove:
    """v1 is what `+1.84 / paired +3.88` was measured against. Nothing here may touch it."""

    def test_the_v1_discriminator_is_still_the_pinned_one(self):
        """The single most load-bearing assertion in this file.

        `fill_cli_baseline` refuses to run when this moves, because a moved discriminator
        means all 25 published caches re-run and the gold set is silently redefined. Adding
        a second prompt must not move it — which is exactly why `_discriminator` hashes
        `prompt_for(version)` rather than, say, the whole `PROMPTS` mapping.
        """
        assert baseline_mod._discriminator("cli", "", None) == PINNED_DISCRIMINATOR

    def test_v1_writes_to_the_path_it_always_wrote_to(self):
        assert baseline_mod._cache_path("cli", "ann") == CACHE / "cli" / "ann.json"
        assert baseline_mod._cache_path("api", "ann") == CACHE / "api" / "ann.json"

    def test_v2_cannot_land_on_a_v1_file(self):
        v1 = baseline_mod._cache_path("cli", "ann", "v1")
        v2 = baseline_mod._cache_path("cli", "ann", "v2")
        assert v1 != v2
        assert v2.parent != v1.parent, "a v2 run must not share a directory with v1"
        assert v2.name == v1.name

    def test_the_two_versions_hash_differently(self):
        assert baseline_mod._discriminator("cli", "", None, "v2") != PINNED_DISCRIMINATOR

    def test_editing_v2_invalidates_v2_and_nothing_else(self, monkeypatch):
        """The text is hashed, not the label.

        A discriminator that hashed `"v2"` would keep matching while the prompt underneath it
        changed — a cache key that has stopped tracking its own input, which is the failure
        `_disc` exists to prevent.
        """
        before_v1 = baseline_mod._discriminator("cli", "", None, "v1")
        before_v2 = baseline_mod._discriminator("cli", "", None, "v2")
        monkeypatch.setitem(baseline_mod.PROMPTS, "v2", "an edited prompt")
        assert baseline_mod._discriminator("cli", "", None, "v2") != before_v2
        assert baseline_mod._discriminator("cli", "", None, "v1") == before_v1


class TestTheWidenedParserOnStoredAnswers:
    """`run_baseline` re-parses cached `raw` on every hit, so this is a live rule, not a
    migration. If it moves, published denominators move with it."""

    def _old_rule(self, text: str) -> tuple[list[str], list[str]]:
        """`_parse_recommendations` as it was before `_ID_KEYS` — arxiv_id or title."""
        ids: list[str] = []
        titles: list[str] = []
        saw = False
        for block in re.findall(r"```(?:json)?\s*(\[.*?\])\s*```", text, re.DOTALL):
            try:
                items = json.loads(block)
            except json.JSONDecodeError:
                continue
            saw = True
            for it in items:
                if not isinstance(it, dict):
                    continue
                if it.get("arxiv_id"):
                    ids.append(str(it["arxiv_id"]))
                elif it.get("title"):
                    titles.append(str(it["title"]))
        if not saw:
            from verify import extract_arxiv_ids

            for rid in extract_arxiv_ids(text):
                if rid not in ids:
                    ids.append(rid)
        return ids, titles

    def test_every_stored_answer_parses_exactly_as_before(self):
        """Accepting `id` and `doi` is only safe because no stored item carries them.

        Checked here rather than asserted in a comment: a survey found 130 recommendation
        items across the caches carrying exactly `arxiv_id` and `title`. If a future cache
        arrives with an `id` field written under v1, this fails and the widening has to be
        re-argued instead of silently re-deriving the gold set.

        **Skipped, not failed, where the caches are absent.** `evals/cache/baseline/` is
        gitignored, so CI has no cache tree at all and never can. The first version asserted
        `seen >= 30` unconditionally to stop the loop passing vacuously — the right worry,
        the wrong mechanism: it turned "this machine legitimately has no data" into a red
        build, and it broke main for two merges because the gates were run here, where the
        tree exists, and not read on CI, where it does not. A skip keeps the guard honest
        (a skipped test is visibly not-run) without claiming a defect that is not there.
        """
        caches = sorted(CACHE.rglob("*.json"))
        if not caches:
            pytest.skip("no local baseline cache (gitignored); nothing to re-parse")
        assert len(caches) >= 30, "the cache tree is present but far smaller than expected"
        for path in caches:
            raw = json.loads(path.read_text(encoding="utf-8")).get("raw") or ""
            assert baseline_mod._parse_recommendations(raw) == self._old_rule(raw), path.name


class TestParsingAV2Answer:
    def test_a_doi_under_the_v2_key_survives(self):
        ids, titles = baseline_mod._parse_recommendations(
            _block([{"id": "10.1038/s41586-021-03819-2", "title": "AlphaFold"}])
        )
        assert ids == ["10.1038/s41586-021-03819-2"] and titles == []

    def test_the_v1_key_still_works_under_v2(self):
        """The model is not obliged to obey the new schema, and habit is strong."""
        ids, _ = baseline_mod._parse_recommendations(_block([{"arxiv_id": "2401.12345"}]))
        assert ids == ["2401.12345"]

    def test_a_doi_named_doi_is_accepted(self):
        ids, _ = baseline_mod._parse_recommendations(_block([{"doi": "10.1101/2024.01.01.123456"}]))
        assert ids == ["10.1101/2024.01.01.123456"]

    def test_an_empty_array_is_still_an_abstention(self):
        """The property the whole authoritative-block rule exists to protect."""
        ids, titles = baseline_mod._parse_recommendations(_block([]))
        assert (ids, titles) == ([], [])
        assert baseline_mod._has_answer_block(_block([]))

    def test_a_title_only_entry_still_goes_to_titles(self):
        _, titles = baseline_mod._parse_recommendations(_block([{"title": "Attention"}]))
        assert titles == ["Attention"]

    def test_an_empty_id_falls_through_to_the_title(self):
        """`{"id": ""}` is the shape a model produces when it cannot find an identifier.
        Recording it as an id would send the empty string to `resolve_reference`, which
        classifies it `hallucinated` — charging the model for a paper it never named."""
        ids, titles = baseline_mod._parse_recommendations(_block([{"id": "", "title": "T"}]))
        assert (ids, titles) == ([], ["T"])

    def test_the_recommended_doi_form_is_one_the_resolver_accepts(self):
        """A round trip, so the prompt and the verifier cannot drift apart.

        The prompt names "bare form" and gives an example; `verify.resolve_reference` routes
        a reference on `doi_key` returning non-empty. If those two ever disagree, every
        non-arXiv pick becomes `unjudgeable` and vanishes — the exact silent-loss failure v2
        was written to remove — so the example in the prompt is extracted and tested.
        """
        examples = re.findall(r'"(10\.[^"]+)"', baseline_mod.BASELINE_PROMPT_V2)
        assert examples, "the v2 prompt no longer shows a DOI example"
        for doi in examples:
            assert doi_key(doi), doi


class TestTheOneVariableClaim:
    def test_v2_keeps_v1_task_and_abstention_wording_verbatim(self):
        """v2 claims to change scope and schema only. This pins the half that must not move.

        If v1's task framing is ever reworded, this fails and forces the same edit into v2 —
        rather than letting the two drift until a comparator re-measurement is attributing a
        score difference to "non-arXiv papers" when half of it is better prompting.
        """
        shared = baseline_mod.BASELINE_PROMPT.split("End your response")[0]
        assert baseline_mod.BASELINE_PROMPT_V2.startswith(shared)
        assert "Use an empty array [] if you recommend nothing." in baseline_mod.BASELINE_PROMPT_V2

    def test_v2_asks_for_the_key_the_parser_reads(self):
        assert '"id"' in baseline_mod.BASELINE_PROMPT_V2
        assert "id" in baseline_mod._ID_KEYS


class TestUnknownVersionsFailLoudly:
    def test_prompt_for_rejects_an_unknown_version(self):
        with pytest.raises(ValueError, match="unknown prompt version"):
            baseline_mod.prompt_for("v3")

    def test_run_baseline_refuses_before_spending_anything(self, tmp_path, monkeypatch):
        """A typo'd version that fell back to v1 would produce a v2-labelled artifact full of
        v1 draws, and no row in it would look wrong. Void, not null — so it raises."""
        monkeypatch.setattr(
            baseline_mod,
            "_run_cli",
            lambda *_a, **_kw: pytest.fail("must not run under an unknown version"),
        )
        with pytest.raises(ValueError):
            baseline_mod.run_baseline(tmp_path, repo_name="x", prompt_version="v3")


class TestCanonicalRef:
    """One id per reference, now that two id schemes can arrive from the same run."""

    def test_it_is_dedup_id_for_every_id_the_project_already_holds(self):
        """The change-nothing guarantee, checked against the real gold set rather than
        asserted. `gold_spread` now canonicalises picks with this instead of `dedup_id`."""
        gold = json.loads((ROOT / "evals" / "gold_targets.json").read_text(encoding="utf-8"))
        ids = [i for case in gold["targets"].values() for i in case]
        assert len(ids) >= 50, "the gold set looks empty; this test would prove nothing"
        for pid in ids:
            assert canonical_ref(pid) == dedup_id(pid), pid

    @pytest.mark.parametrize(
        "written",
        [
            "10.1038/S41586-021-03819-2",
            "doi:10.1038/s41586-021-03819-2",
            "https://doi.org/10.1038/s41586-021-03819-2",
        ],
    )
    def test_every_spelling_of_one_doi_collapses_to_one_id(self, written):
        """The concrete break this prevents: a pick stored as the model wrote it, against a
        target the resolver hands back prefixed, failing `targets <= picks` for every
        non-arXiv paper in the artifact."""
        assert canonical_ref(written) == "doi:10.1038/s41586-021-03819-2"

    def test_it_is_idempotent(self):
        for ref in ("2401.12345v2", "cond-mat/0403023v2", "10.1038/x", "doi:10.1038/x", "ss:abc"):
            once = canonical_ref(ref)
            assert canonical_ref(once) == once, ref


class TestTheArtifactCannotBeOpenedUnderTheWrongFlag:
    """`gold_spread` keys rows `{draw}/{case}`, with no room for a version.

    So the version lives in the filename and in a stamp inside the file, and the two are
    checked against each other on load. Without that, a copied or renamed artifact would
    merge v1 and v2 draws under one set of draw numbers, and every figure `report` prints
    would silently average two different searchers.
    """

    def test_a_v1_artifact_opened_as_v2_is_refused(self, tmp_path, monkeypatch):
        import gold_spread

        stray = tmp_path / "gold_spread_v2.json"
        stray.write_text(json.dumps({"prompt_version": "v1", "results": {}}), encoding="utf-8")
        monkeypatch.setattr(gold_spread, "out_path", lambda *_a: stray)
        with pytest.raises(SystemExit, match="written under prompt"):
            gold_spread.load_artifact("v2")

    def test_an_unstamped_artifact_is_v1(self, tmp_path, monkeypatch):
        """The 79 rows written before versions existed carry no stamp, and they are v1 by
        construction — it was the only prompt there was."""
        import gold_spread

        old = tmp_path / "gold_spread.json"
        old.write_text(json.dumps({"results": {"1/ann": {}}}), encoding="utf-8")
        monkeypatch.setattr(gold_spread, "out_path", lambda *_a: old)
        assert gold_spread.load_artifact("v1")["results"] == {"1/ann": {}}

    def test_each_version_addresses_its_own_file(self):
        import gold_spread

        assert gold_spread.out_path("v1") == gold_spread.OUT
        assert gold_spread.out_path("v2") != gold_spread.OUT
        assert gold_spread.out_path("v2").name == "gold_spread_v2.json"


class TestTheModelIsAConfigurationAxisToo:
    """Opus 5 needed the same treatment the v2 prompt got, for the same reason.

    `_cache_path` is an ADDRESS. Two models writing one file is a silent overwrite, not a
    comparison — the 2026-08-09 displacement that cost `compiler`, `graph` and `storage`
    their transcripts, arriving through a second door.
    """

    def test_the_default_model_changes_no_path_and_no_hash(self):
        """The whole safety argument: today's configuration is byte-identical to yesterday's."""
        assert baseline_mod.DEFAULT_MODEL == baseline_mod.BASELINE_MODEL
        assert baseline_mod._cache_path("cli", "ann") == CACHE / "cli" / "ann.json"
        assert baseline_mod._discriminator("cli", "", None) == PINNED_DISCRIMINATOR

    def test_a_second_model_gets_its_own_directory_and_hash(self):
        v2_48 = baseline_mod._cache_path("cli", "ann", "v2", baseline_mod.DEFAULT_MODEL)
        v2_o5 = baseline_mod._cache_path("cli", "ann", "v2", "claude-opus-5")
        assert v2_48.parent != v2_o5.parent
        assert baseline_mod._discriminator(
            "cli", "", None, "v2", "claude-opus-5"
        ) != baseline_mod._discriminator("cli", "", None, "v2")

    def test_the_two_axes_compose(self):
        """v1+default, v2+default, v1+opus5 and v2+opus5 are four distinct destinations."""
        paths = {
            baseline_mod._cache_path("cli", "ann", pv, m)
            for pv in ("v1", "v2")
            for m in (baseline_mod.DEFAULT_MODEL, "claude-opus-5")
        }
        assert len(paths) == 4

    def test_the_tag_is_stable_and_safe(self):
        assert baseline_mod.model_tag("claude-opus-5") == "opus5"
        assert baseline_mod.model_tag("claude-opus-4-8") == "opus48"
        assert "/" not in baseline_mod.model_tag("some/vendor:model-1.5")


class TestTheFlagListCannotDisagreeWithTheRecordedModel:
    """A run filed under one model that actually ran another is the void-not-null shape:
    every field well-formed, the answer to a different question."""

    def test_flags_for_substitutes_rather_than_appends(self):
        flags = baseline_mod.flags_for(model="claude-opus-5", max_turns=30)
        assert flags.count("--model") == 1 and flags.count("--max-turns") == 1
        assert flags[flags.index("--model") + 1] == "claude-opus-5"
        assert flags[flags.index("--max-turns") + 1] == "30"

    def test_it_leaves_the_shipped_list_alone_when_asked_for_nothing(self):
        assert baseline_mod.flags_for() == baseline_mod.CLAUDE_FLAGS
        assert baseline_mod.flags_for() is not baseline_mod.CLAUDE_FLAGS, "must not alias"

    def test_a_missing_flag_is_added_not_ignored(self):
        assert baseline_mod.flags_for(["--output-format", "json"], model="m")[-2:] == [
            "--model",
            "m",
        ]

    def test_run_cli_runs_the_model_it_was_given_not_the_one_in_the_flags(self, monkeypatch):
        """The trap this closes: a caller passing `model=` and a stale flag list would have
        hashed and pathed on the new model while running the old one."""
        seen: dict[str, list[str]] = {}

        class _Proc:
            returncode = 0
            stdout = '{"result": "```json\n[]\n```", "total_cost_usd": 0.0}'
            stderr = ""

        monkeypatch.setattr(baseline_mod, "cli_auth_mode", lambda *_a, **_kw: "api")
        monkeypatch.setattr(
            baseline_mod.subprocess,
            "run",
            lambda cmd, **_kw: (seen.update(cmd=cmd), _Proc())[1],
        )
        stale = baseline_mod.flags_for(model=baseline_mod.DEFAULT_MODEL)
        baseline_mod._run_cli(Path("."), flags=stale, timeout=5, model="claude-opus-5")
        assert seen["cmd"][seen["cmd"].index("--model") + 1] == "claude-opus-5"


class TestAThrottleIsNotAResult:
    """The 2026-08-27 lesson, at the run level.

    An Opus 5 sweep exhausted the subscription 21 runs in. The other 54 rows were each
    recorded as a terminal `error` in ~400 ms, having done no work at all — and `report` then
    showed two draws at a **100% failure rate**, which reads as "this model cannot do the
    task" when it means "we ran out of credit". Worse, nothing would ever have re-run them:
    resume keyed on a row being present, and they were.

    Same shape as `lookup_failed` vs `unjudgeable` one level down, and as C-4 before that:
    our infrastructure failing must never harden into a fact about the thing being measured.
    """

    class _Proc:
        returncode = 1
        stderr = ""
        stdout = (
            '{"type":"result","subtype":"success","is_error":true,'
            '"api_error_status":429,"duration_ms":423}'
        )

    def _run(self, monkeypatch, proc):
        calls = {"n": 0}

        def fake_run(_cmd, **_kw):
            calls["n"] += 1
            return proc

        monkeypatch.setattr(baseline_mod, "cli_auth_mode", lambda *_a, **_kw: "api")
        monkeypatch.setattr(baseline_mod.subprocess, "run", fake_run)
        monkeypatch.setattr(baseline_mod.time, "sleep", lambda _s: None)
        return baseline_mod._run_cli(Path("."), flags=None, timeout=5), calls

    def test_a_quota_exhaustion_is_its_own_status(self, monkeypatch):
        out, _ = self._run(monkeypatch, self._Proc())
        assert out["status"] == "throttled"

    def test_it_does_not_burn_the_retries(self, monkeypatch):
        """In-process backoff cannot restore a quota, and two more attempts at three seconds
        just delay the resume that actually can."""
        _, calls = self._run(monkeypatch, self._Proc())
        assert calls["n"] == 1

    def test_a_genuine_failure_is_still_an_error_and_still_retries(self, monkeypatch):
        class _Real:
            returncode = 1
            stderr = "the agent hit its turn limit"
            stdout = ""

        out, calls = self._run(monkeypatch, _Real())
        assert out["status"] == "error"
        assert calls["n"] == baseline_mod._CLI_MAX_RETRIES + 1

    def test_the_markers_match_what_the_cli_actually_emitted(self):
        """Pinned against the real payload rather than a paraphrase of it."""
        real = '{"type":"result","is_error":true,"api_error_status":429,"duration_ms":423}'
        assert any(m in real.lower() for m in baseline_mod._CLI_THROTTLE_MARKERS)


class TestResumeSkipsResultsAndNotAbsences:
    def test_an_unasked_row_is_re_attempted(self):
        """`throttled` and `no_cli_login` mean the model was never asked, so the row is not a
        measurement. Resume must treat it as work outstanding, not work done."""
        import gold_spread

        for status in gold_spread.UNASKED:
            assert status in gold_spread.UNASKED, status
        assert "throttled" in gold_spread.UNASKED
        assert "no_cli_login" in gold_spread.UNASKED

    def test_a_real_failure_stays_terminal(self):
        """`error` and `timeout` are runs the agent WAS asked and could not finish. That is a
        fact about the configuration and belongs in the failure rate, not in the retry queue —
        re-running them until they pass would launder a real result into a better one."""
        import gold_spread

        assert "error" not in gold_spread.UNASKED
        assert "timeout" not in gold_spread.UNASKED
