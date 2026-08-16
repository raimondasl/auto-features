"""Tests for the product/benchmark divergence audit.

The audit exists because two of this project's published corrections — C-9 (the query
bridge) and C-12 (the version-strip before a source merge) — are the same defect: **one
invariant, two implementations, one of them fixed**. Both were found by accident while
looking for something else, months apart.

So the tests here are not about the script's formatting. They pin the two things that
decide whether it can catch the next one:

* its **checkers actually see the defect** — each is run against a file that has it,
  because a checker that cannot fail is worse than no checker (it reads as a clean bill);
* its **declared exemptions stay honest** — a difference between the shipped default and
  the measured configuration is fine, an *undeclared* one is the bug, and a declaration
  left behind for a field nobody compares any more is a lie in the audit's own output.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evals"))

from audit_product_divergence import (  # noqa: E402
    BENCHMARK_HEADLINE,
    DECLARED,
    HYDE_MERGE_FIX,
    NOT_UNDER_TEST,
    PIPELINE_MODULES,
    SAME_VINTAGE_CONTAINERS,
    _frozen_pool_papers,
    _hand_rolled_windows,
    _norm_calls,
    _raw_merges,
    _run_stamp,
    _window_callers,
    config_divergences,
    config_leaves,
    coverage_gaps,
    effective_shipped,
    preset_divergences,
    surface_divergences,
    template_values,
)


class TestTheCheckersSeeTheDefect:
    """Each pass, run against a file that has the bug it is supposed to name."""

    def test_a_hand_rolled_version_strip_is_found(self, tmp_path: Path) -> None:
        f = tmp_path / "offender.py"
        f.write_text('known = {p["arxiv_id"].split("v")[0] for p in papers}\n', encoding="utf-8")
        assert [(rule) for _, rule, _ in _norm_calls(f)] == ["split-v"]

    def test_the_shared_helper_is_recognised_under_both_names(self, tmp_path: Path) -> None:
        """`cli.py` re-exports it as `_dedup_id`; missing the alias would report it dirty."""
        f = tmp_path / "ok.py"
        f.write_text(
            'a = dedup_id(p["arxiv_id"])\nb = _dedup_id(p["arxiv_id"])\n', encoding="utf-8"
        )
        assert [rule for _, rule, _ in _norm_calls(f)] == ["dedup_id", "dedup_id"]

    def test_a_raw_id_merge_is_found(self, tmp_path: Path) -> None:
        f = tmp_path / "merge.py"
        f.write_text('if p["arxiv_id"] not in seen:\n    pass\n', encoding="utf-8")
        assert [kind for _, kind, _ in _raw_merges(f)] == ["RAW MERGE"]

    def test_a_repaired_merge_is_not_reported(self, tmp_path: Path) -> None:
        f = tmp_path / "fixed.py"
        f.write_text('if dedup_id(p["arxiv_id"]) not in seen:\n    pass\n', encoding="utf-8")
        assert _raw_merges(f) == []

    def test_a_same_vintage_lookup_is_declared_not_flagged(self, tmp_path: Path) -> None:
        """A dict built from the very list being iterated needs no normalising."""
        f = tmp_path / "lookup.py"
        f.write_text('x = s["arxiv_id"] in papers_by_id\n', encoding="utf-8")
        assert [kind for _, kind, _ in _raw_merges(f)] == ["declared"]

    def test_a_re_derived_digest_window_is_found(self, tmp_path: Path) -> None:
        """The drop-withdrawn-then-cut rule, copied instead of called."""
        f = tmp_path / "copy.py"
        f.write_text(
            'window = [p for p in scored if not p.get("withdrawn_in")][:top_n]\n',
            encoding="utf-8",
        )
        assert [line for line, _ in _hand_rolled_windows(f)] == [1]

    def test_the_module_that_owns_the_rule_is_exempt(self, tmp_path: Path) -> None:
        """`digest.py` IS the implementation; flagging it would make the check unusable."""
        f = tmp_path / "digest.py"
        f.write_text(
            'window = [p for p in scored if not p.get("withdrawn_in")][:top_n]\n',
            encoding="utf-8",
        )
        assert _hand_rolled_windows(f) == []

    def test_a_commented_out_copy_is_not_a_defect(self, tmp_path: Path) -> None:
        """Comments quote the rule to explain it; only live code is the failure."""
        f = tmp_path / "commented.py"
        f.write_text(
            '# window = [p for p in scored if not p.get("withdrawn_in")][:top_n]\n',
            encoding="utf-8",
        )
        assert _hand_rolled_windows(f) == []

    def test_calling_the_shared_helper_is_recognised(self, tmp_path: Path) -> None:
        f = tmp_path / "caller.py"
        f.write_text("w, _ = digest_window(scored, top_n)\n", encoding="utf-8")
        assert _window_callers(f) == [1]


class TestTheCurrentTreeIsClean:
    """The audit's own verdict, asserted — so a regression fails CI, not a later reader."""

    @pytest.mark.parametrize("parts", PIPELINE_MODULES, ids=lambda p: p[-1])
    def test_no_module_hand_rolls_a_version_strip(self, parts: tuple[str, ...]) -> None:
        path = Path(__file__).resolve().parents[1].joinpath(*parts)
        offenders = [(line, text) for line, rule, text in _norm_calls(path) if rule == "split-v"]
        assert offenders == [], f"{path.name}: {offenders}"

    @pytest.mark.parametrize("parts", PIPELINE_MODULES, ids=lambda p: p[-1])
    def test_no_module_merges_on_a_raw_id(self, parts: tuple[str, ...]) -> None:
        path = Path(__file__).resolve().parents[1].joinpath(*parts)
        offenders = [(line, text) for line, kind, text in _raw_merges(path) if kind == "RAW MERGE"]
        assert offenders == [], f"{path.name}: {offenders}"

    @pytest.mark.parametrize("parts", PIPELINE_MODULES, ids=lambda p: p[-1])
    def test_no_module_re_derives_the_digest_window(self, parts: tuple[str, ...]) -> None:
        path = Path(__file__).resolve().parents[1].joinpath(*parts)
        assert _hand_rolled_windows(path) == [], f"{path.name} grew a second window rule"

    def test_both_callers_actually_share_the_window(self) -> None:
        """The other half: a guard that only forbids copies passes on a file that deleted
        the call entirely, which is how `to_plain_keywords` sat correct and unused."""
        root = Path(__file__).resolve().parents[1]
        # `pipeline.py`, not `cli.py`, since the orchestrator moved there on 2026-08-16 --
        # and this guard is what asked the question ("has the caller moved?") rather than
        # passing quietly on a file that had stopped calling it.
        for parts in (("src", "reporadar", "pipeline.py"), ("src", "reporadar", "digest.py")):
            assert _window_callers(root.joinpath(*parts)), f"{parts[-1]} no longer calls it"

    def test_every_config_difference_is_declared(self) -> None:
        """The durable half of the audit.

        `arxiv.lookback_days` shipped at 14 days for a month while every headline was
        measured all-time — nobody was lying, the two numbers simply lived in different
        files. This fails the moment a shipped default and the measured configuration
        part company without a written reason.
        """
        undeclared = [d for d in config_divergences() if d.status == "DIVERGENT"]
        assert undeclared == [], (
            "the benchmark is no longer measuring the shipped default for: "
            + ", ".join(d.name for d in undeclared)
        )


class TestTheDeclarationsStayHonest:
    def test_every_declared_field_is_actually_compared(self) -> None:
        """A declaration for a field the audit no longer looks at reads as coverage."""
        assert set(DECLARED) <= set(BENCHMARK_HEADLINE)

    def test_every_declared_field_actually_differs(self) -> None:
        """If a difference is closed, the exemption must go with it — otherwise the next
        person to change that default gets no warning."""
        flagged = {d.name for d in config_divergences() + surface_divergences()}
        for key in DECLARED:
            assert key in flagged, (
                f"{key} no longer differs; remove it from DECLARED so the guard can bite"
            )

    def test_every_compared_field_resolves_against_the_product(self) -> None:
        """A renamed config field must fail loudly, not drop out of the comparison."""
        assert set(BENCHMARK_HEADLINE) <= set(config_leaves())

    def test_each_exempt_container_carries_its_reason(self) -> None:
        assert all(reason.strip() for reason in SAME_VINTAGE_CONTAINERS.values())

    def test_each_unmeasured_field_carries_its_reason(self) -> None:
        """ "Not measured" without a reason is an assertion, not a decision."""
        assert all(reason.strip() for reason in NOT_UNDER_TEST.values())

    def test_a_field_is_measured_or_excused_but_never_both(self) -> None:
        assert set(BENCHMARK_HEADLINE) & set(NOT_UNDER_TEST) == set()

    def test_the_declared_values_themselves_are_pinned(self) -> None:
        """A declaration keyed on the field NAME excuses any value that field ever takes.

        `ranking.w_embedding`'s reason argues specifically about 1.5 against 0.0; edit the
        template to 2.5 and the name-keyed exemption still covers it, so the audit would
        report "declared" about a decision nobody made. Pinning the values means changing
        one fails here and forces the reason to be rewritten with it.
        """
        shipped = effective_shipped()
        assert {k: shipped[k] for k in DECLARED} == {
            "triage.enabled": False,
            "suggestions.provider": "template",
            "triage.finescale.enabled": False,
            "hyde.enabled": False,
            "ranking.hybrid": False,
            "ranking.w_embedding": 1.5,
            "triage.finescale.timeout": 30,
        }


class TestTheConfigSurfacesAreBothRead:
    """The failure this pass was rewritten for.

    Its previous version compared twelve hand-listed fields against the dataclass
    defaults. `rr init` writes a *template*, and where the template sets a value that is
    what a user runs — so the audit was reporting clean about `ranking.w_embedding`, which
    is 0.0 in the dataclass, 0.0 in every published number, and 1.5 in the file every user
    gets. Wrong surface AND hand-scoped list: both are pinned here.
    """

    def test_leaves_are_found_recursively(self) -> None:
        leaves = config_leaves()
        assert "triage.finescale.threshold" in leaves  # two levels down
        assert "hooks.email.smtp_port" in leaves
        assert "triage" not in leaves  # the container itself is not a leaf

    def test_the_template_is_parsed_not_duplicated(self) -> None:
        """If this audit kept its own copy of the template it would drift from it, which
        is the exact failure the audit exists to catch."""
        written = template_values()
        assert written["output.top_n"] == 15
        assert written["ranking.w_embedding"] == 1.5

    def test_what_a_user_runs_prefers_the_template(self) -> None:
        assert config_leaves()["ranking.w_embedding"] == 0.0
        assert effective_shipped()["ranking.w_embedding"] == 1.5

    def test_a_template_field_absent_from_the_template_falls_back(self) -> None:
        assert "triage.enabled" not in template_values()
        assert effective_shipped()["triage.enabled"] == config_leaves()["triage.enabled"]

    def test_the_surface_disagreement_is_reported(self) -> None:
        names = {d.name for d in surface_divergences()}
        assert "ranking.w_embedding" in names

    def test_every_config_leaf_is_classified(self) -> None:
        """The check that makes this pass exhaustive instead of hand-scoped.

        A new config field fails the audit until somebody decides whether the benchmark
        measures it. C-14b is the argument: a guard that reads only where a bug was last
        found reports clean about everywhere it never looked.
        """
        gaps = coverage_gaps()
        assert gaps == [], "unclassified config fields: " + ", ".join(d.name for d in gaps)

    def test_an_unclassified_field_is_caught(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Mutation: a checker that cannot fail is a clean bill of health, not a check."""
        import audit_product_divergence as audit

        with_new = {**audit.config_leaves(), "new.knob": 7}
        monkeypatch.setattr(audit, "config_leaves", lambda: with_new)
        assert [d.name for d in audit.coverage_gaps()] == ["new.knob"]

    def test_a_stale_classification_is_caught(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The other direction: a field classified but since deleted from the config."""
        import audit_product_divergence as audit

        monkeypatch.setitem(audit.NOT_UNDER_TEST, "gone.field", "deleted last week")
        assert [d.name for d in audit.coverage_gaps()] == ["gone.field"]


class TestTheRecommendedConfigIsTheMeasuredOne:
    """`rr init --measured` tells a user this configuration reaches +5.42. That is a claim
    about a specific run, and claims decay the way code does — `arxiv.lookback_days`
    shipped at 14 days for a month while every headline ran all-time. So it is checked
    against `BENCHMARK_HEADLINE` field by field, with **no exemption mechanism**: unlike
    the default template, which may differ for declared reasons, this file's whole purpose
    is to reproduce the measured run, so any difference is a documentation defect.
    """

    def test_the_preset_reproduces_every_measured_field(self) -> None:
        drift = preset_divergences()
        assert drift == [], "rr init --measured no longer matches the benchmark: " + ", ".join(
            f"{d.name} (preset={d.product}, measured={d.benchmark})" for d in drift
        )

    def test_the_preset_check_has_no_exemption_mechanism(self) -> None:
        """DECLARED must not silence preset drift. If it could, the recommendation could
        be excused away from the measurement it cites, which is the one thing this check
        exists to prevent."""
        assert all(d.status == "PRESET DRIFT" for d in preset_divergences())
        source = (
            Path(__file__).resolve().parents[1] / "evals" / "audit_product_divergence.py"
        ).read_text(encoding="utf-8")
        body = source.split("def preset_divergences(")[1].split("\ndef ")[0]
        assert "DECLARED" not in body

    def test_preset_drift_is_caught(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Mutation: edit the measured config away from the benchmark and this must fail."""
        import audit_product_divergence as audit

        drifted = {**audit.measured_preset(), "triage.top_k": 15}
        monkeypatch.setattr(audit, "measured_preset", lambda: drifted)
        assert [d.name for d in audit.preset_divergences()] == ["triage.top_k"]

    def test_a_field_missing_from_the_preset_is_drift_not_a_pass(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The preset may omit a field only when the dataclass default already matches.
        A field the dataclass gets wrong and the preset omits must not read as agreement."""
        import audit_product_divergence as audit

        monkeypatch.setattr(audit, "config_leaves", lambda: {})
        monkeypatch.setattr(audit, "measured_preset", lambda: {})
        assert len(audit.preset_divergences()) == len(BENCHMARK_HEADLINE)


class TestTheStagesTheProductShipsWithout:
    """Every stage the paper credits is off in the shipped default, and each is declared.

    This is not a bug to fix in the audit — the gate needs a key, the rescore needs a
    second vendor's key, HyDE needs 1.1 GB — but an undeclared version of it is how a
    product ends up measured in a configuration nobody runs.
    """

    @pytest.mark.parametrize(
        "field",
        ["triage.enabled", "triage.finescale.enabled", "hyde.enabled", "ranking.hybrid"],
    )
    def test_the_stage_is_off_by_default_and_says_why(self, field: str) -> None:
        assert effective_shipped()[field] is False
        assert BENCHMARK_HEADLINE[field] is True
        assert DECLARED[field].strip()

    def test_enabling_the_gate_takes_two_fields(self) -> None:
        """The pipeline gates on `triage.enabled AND suggestions.provider in (ollama,
        claude)`, so `triage.enabled: true` alone is a no-op. Pinned because a reader of
        the config alone would not guess it. (In `pipeline.py` since 2026-08-16, shared by
        `rr update` and `rr watch` -- previously inline in `cli.update`.)"""
        source = (
            Path(__file__).resolve().parents[1] / "src" / "reporadar" / "pipeline.py"
        ).read_text(encoding="utf-8")
        assert 'cfg.triage.enabled and cfg.suggestions.provider in ("ollama", "claude")' in source
        assert effective_shipped()["suggestions.provider"] == "template"

    def test_the_hyde_encoder_default_matches_the_one_the_benchmark_measures(self) -> None:
        """`hyde.discover(model_name=MODEL_NAME)` is what the eval uses; `HydeConfig.model`
        is what the product uses. Two names for one encoder, and the index only answers to
        the one it was built with."""
        from reporadar.config import HydeConfig
        from reporadar.hyde import MODEL_NAME

        assert HydeConfig().model == MODEL_NAME == BENCHMARK_HEADLINE["hyde.model"]


class TestFrozenPoolsAreActuallyRead:
    """`_frozen_pool_papers` reported 1,250 papers as 0 and printed "0 dup" about a pool
    it had not parsed — void, not null, in the audit's own blast-radius pass. v1 froze
    `[paper, score]` PAIRS after ranking; v2 freezes paper dicts before it."""

    def test_v2_candidates_are_read(self) -> None:
        papers, fmt = _frozen_pool_papers({"candidates": [{"arxiv_id": "2401.00001"}]})
        assert fmt == "v2/candidates" and len(papers) == 1

    def test_v1_ranked_pairs_are_unwrapped(self) -> None:
        blob = {"ranked": [[{"arxiv_id": "2401.00001"}, 0.9], [{"arxiv_id": "2401.00002"}, 0.8]]}
        papers, fmt = _frozen_pool_papers(blob)
        assert fmt == "v1/ranked"
        assert [p["arxiv_id"] for p in papers] == ["2401.00001", "2401.00002"]

    def test_a_partly_parsed_pool_says_so_rather_than_reporting_a_subset(self) -> None:
        papers, fmt = _frozen_pool_papers({"candidates": [{"arxiv_id": "2401.1"}, {"oops": 1}]})
        assert fmt == "v2/PARTIAL" and len(papers) == 1

    def test_an_unknown_shape_is_named_not_treated_as_empty(self) -> None:
        assert _frozen_pool_papers({"something_else": []}) == ([], "unrecognised")
        assert _frozen_pool_papers(["not a dict"]) == ([], "unrecognised")


class TestDuplicatesAreDatedNotJustCounted:
    """A duplicate in a run recorded before the merge fix is history; one after it is a
    live bug. A bare count conflates them, and the reader has to re-derive the dates."""

    def test_a_run_stamp_is_parsed_from_the_filename(self) -> None:
        assert _run_stamp("judge-gpt-5.5-bigrams_verified-20260813T171319Z.json") == (
            "2026-08-13T17:13Z"
        )

    def test_an_unstamped_filename_does_not_masquerade_as_recent(self) -> None:
        assert _run_stamp("some-hand-named-run.json") == ""

    def test_the_only_contaminated_run_predates_the_fix(self) -> None:
        assert _run_stamp("judge-gpt-5.5-bigrams_verified-20260813T171319Z.json") < HYDE_MERGE_FIX


class TestBothRunnersFailTheSameWay:
    """A collection failure must never be scoreable, in either eval runner.

    `evals/harness.py` raises on `CollectionError` because scoring a throttled fetch as an
    honest result once supplied −17 of a −21 delta (C-4). `evals/run_eval.py` printed a
    warning and carried on with whatever it had — which, with a second source enabled,
    means a domain-purity number computed on a pool missing its arXiv half, indexed and
    printed like any other. Two runners, one invariant, one of them fixed: the shape this
    whole audit is looking for.
    """

    def test_collect_live_raises_instead_of_returning_a_partial_pool(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import run_eval

        from reporadar.collector import CollectionError

        def boom(*_args: object, **_kwargs: object) -> list[dict[str, object]]:
            raise CollectionError("429 from arXiv")

        monkeypatch.setattr("reporadar.collector.collect_papers", boom)
        profile = run_eval.profile_case_repo(Path(__file__).resolve().parents[1])
        case = {"expected_categories": ["cs.LG"], "name": "x"}
        with pytest.raises(CollectionError):
            run_eval.collect_live(profile, case, ["arxiv", "semantic_scholar"], {}, 90)
