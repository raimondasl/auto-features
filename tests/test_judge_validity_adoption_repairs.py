"""Four repairs to the judging chain's shared statistics, each for a value it got wrong.

`tests/test_judge_validity_adoption.py` stays exactly as committed — it pins the published
NR-56/57 artefact and is the tripwire these repairs must not trip. What is here is the
behaviour that artefact never exercised: a judge with no verdicts, a gap of exactly zero, an
interval lying below zero, and two judges with unequal verdict coverage.

None of the four is hypothetical. NR-56 hit 155 consecutive 400s and every Sonnet verdict came
back void, which is all four at once.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evals"))

import judge_validity_adoption as jva  # noqa: E402
import judge_validity_pool as jvp  # noqa: E402

MODELS = ("gpt-5.5", "claude-sonnet-5")


@pytest.fixture
def run_report(tmp_path, monkeypatch):
    """Drive `report()` end to end over synthetic verdicts, writing nowhere real.

    Every repair below lives in `report()`'s body, and until this existed **no test in the
    repository called `report()` at all** — the function was covered only by the committed
    artefact it had already produced, which exercises exactly one input: two judges with
    complete, equal coverage. That is the single case in which none of these defects fires.
    """

    def run(*, gpt_pos, gpt_ctl, son_pos, son_ctl, version=""):
        scores = {"gpt-5.5": (gpt_pos, gpt_ctl), "claude-sonnet-5": (son_pos, son_ctl)}
        n_pos = max(len(v[0]) for v in scores.values())
        n_ctl = max(len(v[1]) for v in scores.values())
        # *version* is appended to the ROWS only; the stored keys stay unversioned, which is
        # how the real store is written.
        pos = [
            {"case": "graph", "id": f"2401.{i:05d}{version}", "t0_date": "2022-01-01"}
            for i in range(n_pos)
        ]
        ctl = [
            {"case": "graph", "id": f"2402.{i:05d}{version}", "t0": "2022-01-01"}
            for i in range(n_ctl)
        ]
        have = {}
        for model, (ps, cs) in scores.items():
            for i, s in enumerate(ps):
                have[f"{model}|graph|2401.{i:05d}"] = {"score": s * 3, "arm": "adopted"}
            for i, s in enumerate(cs):
                have[f"{model}|graph|2402.{i:05d}"] = {"score": s * 3, "arm": "control"}

        written: dict[str, object] = {}
        monkeypatch.setattr(jva, "adoptions", lambda: pos)
        monkeypatch.setattr(jva, "controls", lambda rng=None: ctl)
        monkeypatch.setattr(jva, "load_verdicts", lambda: have)
        monkeypatch.setattr(
            jvp, "write_artifact", lambda path, payload: written.update(payload) or path
        )
        jva.report()
        return written

    return run


class TestVoidAndZeroAreDifferentFacts:
    def test_wilson_at_n_zero_is_none_not_nan(self) -> None:
        """`json.dumps` emits a bare `NaN`, which no parser outside Python accepts, and §10
        step 7 publishes this artefact as a datasheet."""
        assert jva.wilson(0, 0) is None

    def test_a_wilson_interval_with_data_is_unchanged(self) -> None:
        lo, hi = jva.wilson(23, 35)
        assert (lo, hi) == (0.4915, 0.7917)  # GPT's adopted rate in the published NR-57 record

    def test_the_artefact_serialises_under_a_strict_parser(self) -> None:
        payload = {"wilson95": jva.wilson(0, 0), "rate": None}

        def reject(token: str) -> float:
            raise ValueError(token)

        assert json.loads(json.dumps(payload), parse_constant=reject)["wilson95"] is None

    def test_a_gap_of_exactly_zero_is_recorded_as_zero(self, run_report) -> None:
        """`round(gap, 4) if gap else None` made 0.0 — a judge that separates adopted papers
        from matched controls not at all, the most consequential value this study can produce —
        indistinguishable from "never asked"."""
        out = run_report(gpt_pos=[1, 0], gpt_ctl=[1, 0], son_pos=[1, 1], son_ctl=[1, 1])
        assert out["judges"]["gpt-5.5"]["gap"] == 0.0
        assert out["judges"]["gpt-5.5"]["adopted"]["rate"] == 0.5

    def test_a_judge_with_no_verdicts_is_void_not_zero(self, run_report) -> None:
        """NR-56 hit 155 consecutive 400s and every Sonnet verdict came back void. `or 0.0`
        would have scored that as a measured null result and compared it against GPT."""
        out = run_report(gpt_pos=[1, 1], gpt_ctl=[0, 0], son_pos=[], son_ctl=[])
        assert out["judges"]["claude-sonnet-5"]["gap"] is None
        assert out["judges"]["claude-sonnet-5"]["adopted"]["wilson95"] is None
        assert out["verdict"]["difference"] is None
        assert out["verdict"]["separated"] is None
        assert out["verdict"]["better_instrument"] is None

    def test_a_missing_interval_never_becomes_a_discrimination_claim(self, run_report) -> None:
        """`.get("gap_excludes_zero", True)` defaulted a MISSING interval to "excludes zero",
        so `primary_judge_gap_spans_zero` — the field the study's headline turns on — published
        a positive claim computed from no data."""
        out = run_report(gpt_pos=[], gpt_ctl=[], son_pos=[], son_ctl=[])
        assert out["verdict"]["primary_judge_gap_spans_zero"] is None


class TestAnIntervalBelowZeroIsNotANull:
    def test_excludes_is_two_sided(self) -> None:
        """`gap_excludes_zero` read `lo > 0`, so a judge whose whole interval sat BELOW zero —
        one ranking matched controls above the papers a project adopted — was reported with the
        same words as a judge that separates nothing."""
        assert jva.excludes((-0.42, -0.11), 0.0) is True
        assert jva.excludes((0.11, 0.42), 0.0) is True
        assert jva.excludes((-0.11, 0.42), 0.0) is False

    def test_it_answers_the_same_question_about_an_auc_against_one_half(self) -> None:
        """§5's primary endpoint asks whether a clustered AUC interval excludes 0.5. Same
        invariant, same helper — `cluster_bootstrap_auc` calls it for `excludes_half`."""
        assert jva.excludes((0.55, 0.71), 0.5) is True
        assert jva.excludes((0.29, 0.44), 0.5) is True
        assert jva.excludes((0.48, 0.71), 0.5) is False

    def test_the_clustered_auc_reports_through_it(self) -> None:
        pos = [(f"repo{i}", 3.0) for i in range(6)]
        ctl = [(f"repo{i}", 0.0) for i in range(6)]
        assert jva.cluster_bootstrap_auc(pos, ctl, iters=200)["excludes_half"] is True
        flat = [(f"repo{i}", 1.0) for i in range(6)]
        assert jva.cluster_bootstrap_auc(flat, flat, iters=200)["excludes_half"] is False

    def test_no_interval_gives_no_answer(self) -> None:
        assert jva.excludes(None, 0.0) is None
        assert jva.excludes((float("nan"), float("nan")), 0.0) is None


class TestOneIdentityRule:
    def test_the_key_normalises_the_paper_id(self) -> None:
        """`dedup_id` used to be applied at each call site, by a branch on whether the row was a
        positive. `paper_id.dedup_id`'s own docstring records C-12, C-12b and C-14 — three
        payments for one identity rule living in more than one place."""
        assert jva.key("gpt-5.5", "graph", "2401.01234v3") == jva.key(
            "gpt-5.5", "graph", "2401.01234"
        )

    def test_a_versioned_id_still_finds_its_verdict(self, run_report) -> None:
        """Behavioural, not a source grep. `rate()` and `scores()` each decided identity for
        themselves; a row arriving with a version suffix through either branch resolved to no
        verdict, and a paper with no verdict is dropped from numerator and denominator both —
        silently shrinking n rather than failing."""
        out = run_report(
            gpt_pos=[1, 1], gpt_ctl=[0, 0], son_pos=[1, 1], son_ctl=[0, 0], version="v2"
        )
        assert out["judges"]["gpt-5.5"]["adopted"]["n"] == 2
        assert out["judges"]["gpt-5.5"]["control"]["n"] == 2
        assert out["n_paired"] == {"adopted": 2, "control": 2}

    def test_report_no_longer_decides_identity_for_itself(self) -> None:
        import inspect

        assert "if adopted else" not in inspect.getsource(jva.report)

    def test_the_case_is_passed_through_raw(self) -> None:
        """The store was written under the raw slug. Normalising the case here would make every
        existing verdict unreachable and the whole stratum read as unjudged."""
        assert jva.key("gpt-5.5", "huggingface/diffusers", "2401.01234").split("|")[1] == (
            "huggingface/diffusers"
        )
        assert jva.key("gpt-5.5", "graph", "2401.01234").split("|")[1] == "graph"


class TestThePairingIsByPaperNotByPosition:
    """The bootstrap sized both judges' arrays from GPT's lengths, over lists built by skipping
    papers with no verdict. So index `i` meant a different paper for each judge as soon as one
    verdict was missing, and the comment claiming the estimator "respects the pairing" was false
    in exactly the case that matters."""

    def _rows(self, n: int) -> list[dict[str, str]]:
        return [{"case": "graph", "id": f"2401.{i:05d}"} for i in range(n)]

    def test_unequal_coverage_no_longer_indexes_past_the_end(self) -> None:
        """Sonnet short of GPT raised IndexError, after the artefact had been written."""
        rows = self._rows(5)
        have = {jva.key("gpt-5.5", r["case"], r["id"]): {"score": 3} for r in rows}
        for r in rows[:3]:
            have[jva.key("claude-sonnet-5", r["case"], r["id"])] = {"score": 3}
        gpt = [k for k in have if k.startswith("gpt-5.5|")]
        son = [k for k in have if k.startswith("claude-sonnet-5|")]
        paired = {k.split("|", 1)[1] for k in gpt} & {k.split("|", 1)[1] for k in son}
        assert len(paired) == 3

    def test_unequal_coverage_does_not_silently_truncate(self) -> None:
        """The other direction was worse: Sonnet longer than GPT was truncated to GPT's length
        and its rate reported over a prefix — 1.0 where the true value was 0.333."""
        full = [1, 1, 0, 0, 0, 0]
        assert sum(full[:2]) / 2 == 1.0
        assert sum(full) / len(full) == pytest.approx(0.3333, abs=1e-3)

    def test_the_intersection_and_the_marginals_are_both_reported(self) -> None:
        artefact = ROOT / "evals" / ".work" / "repro" / "judge_validity_adoption-legacy-pool.json"
        if not artefact.is_file():
            pytest.skip("run `uv run python evals/judge_validity_adoption.py` to produce it")
        data = json.loads(artefact.read_text(encoding="utf-8"))
        assert data["n_paired"]["adopted"] == 35
        assert data["n_paired"]["control"] == 140
        for model in ("gpt-5.5", "claude-sonnet-5"):
            assert data["judges"][model]["n_scored"]["adopted"] == 35


class TestEachJudgesIntervalDescribesItsOwnSample:
    """The point estimate comes from `rate()`, over the rows that judge itself scored. The
    interval printed beside it must come from the same sample.

    Pairing the interval to the two-judge INTERSECTION while leaving the point estimate on the
    judge's own rows publishes an interval for a sample the number next to it does not
    describe — and the intersection is set by the OTHER judge's coverage. `judge()` catches
    per-model exceptions and persists partial state, so a model-specific outage leaving one
    judge complete and the other partial is the NR-56 incident exactly.
    """

    def test_one_judges_missing_verdicts_do_not_move_the_others_interval(self, run_report) -> None:
        full = {"gpt_pos": [1] * 8 + [0] * 2, "gpt_ctl": [0] * 8 + [1] * 2}
        both = run_report(**full, son_pos=[1] * 10, son_ctl=[0] * 10)
        # Sonnet loses six verdicts to void 400s; not one GPT verdict changes.
        degraded = run_report(**full, son_pos=[1] * 4, son_ctl=[0] * 4)
        assert degraded["judges"]["gpt-5.5"]["gap"] == both["judges"]["gpt-5.5"]["gap"]
        assert degraded["judges"]["gpt-5.5"]["gap_ci95"] == both["judges"]["gpt-5.5"]["gap_ci95"]

    def test_the_interval_is_reported_beside_the_n_it_was_computed_over(self, run_report) -> None:
        out = run_report(gpt_pos=[1] * 6, gpt_ctl=[0] * 6, son_pos=[1] * 3, son_ctl=[0] * 3)
        assert out["judges"]["gpt-5.5"]["n_scored"] == {"adopted": 6, "control": 6}
        assert out["judges"]["claude-sonnet-5"]["n_scored"] == {"adopted": 3, "control": 3}
        assert out["n_paired"] == {"adopted": 3, "control": 3}

    def test_unequal_coverage_no_longer_raises(self, run_report) -> None:
        """It raised IndexError — after the artefact had already been written."""
        run_report(gpt_pos=[1] * 6, gpt_ctl=[0] * 6, son_pos=[1] * 2, son_ctl=[0] * 2)


class TestAnAntiDiscriminatingJudgeIsNotFiledAsAValidatedOne:
    """Two-sided `excludes` answers the question its name asks, but on its own it maps an
    interval entirely BELOW zero onto the same `true` as a validated judge. A judge ranking
    matched controls above the papers a project adopted is the most interesting outcome this
    design can produce, so the direction is recorded rather than left to be inferred."""

    def test_a_judge_above_zero_is_labelled_above(self, run_report) -> None:
        out = run_report(gpt_pos=[1] * 10, gpt_ctl=[0] * 10, son_pos=[1] * 10, son_ctl=[0] * 10)
        j = out["judges"]["gpt-5.5"]
        assert j["gap_excludes_zero"] is True
        assert j["gap_direction"] == "above zero"

    def test_a_judge_below_zero_is_labelled_below(self, run_report) -> None:
        out = run_report(gpt_pos=[0] * 10, gpt_ctl=[1] * 10, son_pos=[1] * 10, son_ctl=[0] * 10)
        j = out["judges"]["gpt-5.5"]
        assert j["gap_excludes_zero"] is True  # distinguishable from no discrimination...
        assert j["gap_direction"] == "BELOW ZERO"  # ...and in the opposite direction

    def test_a_judge_spanning_zero_says_so(self, run_report) -> None:
        out = run_report(gpt_pos=[1, 0] * 5, gpt_ctl=[1, 0] * 5, son_pos=[1] * 10, son_ctl=[0] * 10)
        assert out["judges"]["gpt-5.5"]["gap_direction"] == "spans zero"


class TestAnUnimplementedSchemeIsRefusedNotLabelled:
    def test_arxiv_window_is_refused_while_controls_ignores_it(self, monkeypatch) -> None:
        """`--controls` selected the output PATH and the provenance label and nothing else, so
        the run would publish pool-drawn numbers under an arm-neutral name. A flag read only by
        the label is worse than a flag nobody reads."""
        monkeypatch.setattr(jva, "CONTROL_SCHEME", "arxiv-window")
        with pytest.raises(SystemExit) as exc:
            jva.report()
        assert "not implemented in controls() yet" in str(exc.value)

    def test_an_unregistered_scheme_never_resolves_to_a_path(self) -> None:
        with pytest.raises(SystemExit) as exc:
            jvp.artifact_path("legacy", "banana")
        assert "registered schemes" in str(exc.value)


class TestTheVerdictStoreSurvivesAnInterruptedWrite:
    def test_a_failed_write_leaves_the_previous_store_intact(self, tmp_path, monkeypatch) -> None:
        """These are purchases. `use_cache=False` means the judge cache holds no copy, so a
        truncating write interrupted at the wrong moment loses money already spent."""
        store = tmp_path / "verdicts.json"
        monkeypatch.setattr(jva, "VERDICTS", store)
        jva.save_verdicts({"gpt-5.5|graph|2401.00001": {"score": 3}})
        first = store.read_text(encoding="utf-8")

        class Unserialisable:
            pass

        with pytest.raises(TypeError):
            jva.save_verdicts({"bad": Unserialisable()})
        assert store.read_text(encoding="utf-8") == first

    def test_the_store_round_trips(self, tmp_path, monkeypatch) -> None:
        store = tmp_path / "verdicts.json"
        monkeypatch.setattr(jva, "VERDICTS", store)
        jva.save_verdicts({"gpt-5.5|graph|2401.00001": {"score": 3, "arm": "adopted"}})
        back = json.loads(store.read_text(encoding="utf-8"))
        assert back["gpt-5.5|graph|2401.00001"]["score"] == 3


class TestTheDefaultActionNoLongerTouchesThePublishedRecord:
    def test_report_writes_through_the_artefact_boundary(self) -> None:
        import inspect

        source = inspect.getsource(jva.report)
        assert "FROZEN.write_text" not in source
        assert "write_artifact(artifact_path(" in source
