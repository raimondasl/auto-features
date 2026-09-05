"""Transportability, the void contamination split, P1-P9, and the datasheet. [PREREG §5, §8, §10]

The four pieces that turn computed endpoints into a published result. Three properties do most
of the work here.

**A subgroup carries no consequence.** §5 registers the strata contrast, the star bands and the
`ids_v2(T0) >= 10` sensitivity as descriptive heterogeneity; every one of them is underpowered
by construction and none re-weights the pooled primary.

**A void is not a zero.** The contamination split has no published training cutoff to split on
and never will, so it reports `status: not_computed` with its reason — values null, never a
date, never 0, and the key never absent.

**A prediction is scored from the quantity it names**, and never below the n it registered at.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT / "evals", ROOT / "evals" / "frame", ROOT / "src"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import judge_validity_pool as jvp  # noqa: E402

ITERS = 300
MODELS = ("gpt-5.5", "claude-sonnet-5")


def _analysis(n_legacy: int = 4, n_pool: int = 4, per: int = 4):  # noqa: ANN202
    """Two strata that share no repository, as §2.2 guarantees."""
    pos, ctl, verdicts = [], [], {}
    for stratum, n, prefix in (("legacy", n_legacy, "graph"), ("pool", n_pool, "o/r")):
        for c in range(n):
            case = f"{prefix}{c}"
            for i in range(per):
                pid = f"2401.{c:02d}{i:03d}"
                pos.append(
                    {
                        "case": case,
                        "id": pid,
                        "stratum": stratum,
                        "seeds_at_t0": 12 if c % 2 == 0 else 3,
                        "adoption_date": None if stratum == "legacy" else "2025-06-01",
                    }
                )
                for m in MODELS:
                    verdicts[jvp.verdict_key(m, case, pid)] = {
                        "score": 3,
                        "model": m,
                        "case": case,
                        "id": pid,
                        "arm": "adopted",
                    }
                for j in range(2):
                    cid = f"2402.{c:02d}{i}{j}"
                    ctl.append({"case": case, "id": cid, "for_positive": pid})
                    for m in MODELS:
                        verdicts[jvp.verdict_key(m, case, cid)] = {
                            "score": 0,
                            "model": m,
                            "case": case,
                            "id": cid,
                            "arm": "control",
                        }
    analysis = {
        "positives": pos,
        "by_stratum": {
            "legacy": [p for p in pos if p["stratum"] == "legacy"],
            "pool": [p for p in pos if p["stratum"] == "pool"],
        },
        "analysis_set_positives": len(pos),
        "n_clusters": n_legacy + n_pool,
    }
    return analysis, ctl, verdicts


def _candidate_csv(tmp_path: Path, analysis: dict) -> Path:
    """A stand-in candidate list carrying the synthetic pool cases, so `transportability`'s
    real guard — every pool positive must appear in the list the walk used — is exercised
    rather than tripped by the fixture."""
    src = tmp_path / "candidates.csv"
    lines = ["full_name,created_at,pushed_at,stars,language,topics,slice"]
    for i, case in enumerate(sorted({str(r["case"]) for r in analysis["by_stratum"]["pool"]})):
        stars = (150, 700, 3000, 20000)[i % 4]
        lo = {150: 150, 700: 500, 3000: 2500, 20000: 10000}[stars]
        lines.append(f"{case},2020-01-01,2020-01-01,{stars},Python,,machine-learning|{lo}|2016")
    src.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return src


class TestTheStarBandsComeFromTheFrozenSnapshot:
    def test_the_bands_are_the_registered_slices(self) -> None:
        import enumerate_pool as ep

        for lo, hi in ep.STAR_SLICES:
            assert jvp.star_band(lo) == (f"{lo}-{hi}" if hi is not None else f"{lo}+")

    def test_a_missing_count_is_a_band_and_not_a_hole(self) -> None:
        """Three contributing legacy repositories do not resolve to a candidate row. Imputing a
        count would invent the covariate the contrast is cut on; dropping them would quietly
        shrink the legacy stratum."""
        assert jvp.star_band(None) == "unknown"

    def test_the_candidate_list_is_checked_against_its_own_slice_column(self, tmp_path) -> None:
        """A band edge chosen at analysis time is exactly the discretion an unchoosable pulse
        exists to remove, so the enumeration's own grid is the authority."""
        src = tmp_path / "c.csv"
        src.write_text(
            "full_name,created_at,pushed_at,stars,language,topics,slice\n"
            "a/b,2020-01-01,2020-01-01,50,Python,,machine-learning|100|2016\n",
            encoding="utf-8",
        )
        with pytest.raises(SystemExit) as exc:
            jvp.star_bands(src)
        assert "disagrees with itself" in str(exc.value)

    def test_the_real_candidate_list_is_internally_consistent(self) -> None:
        assert len(jvp.star_bands()) == 17888

    def test_the_count_is_labelled_as_a_snapshot_property(self) -> None:
        """There is no historical star index, so this is a value at Dp and not at T0 — and
        recomputing from the live API would be a fresh measurement taken after the positives
        were visible."""
        analysis, ctl, v = _analysis(n_pool=0)
        out = jvp.transportability(MODELS, analysis, ctl, v, iters=ITERS)
        assert out["stars_measured_on"] == jvp.DP
        assert "no historical index" in out["stars_note"]


class TestTransportabilityDecidesNothing:
    def test_each_stratum_gets_the_full_estimator(self, tmp_path: Path) -> None:
        analysis, ctl, v = _analysis()
        out = jvp.transportability(
            MODELS, analysis, ctl, v, iters=ITERS, candidates=_candidate_csv(tmp_path, analysis)
        )
        for name in ("legacy", "pool"):
            entry = out["strata"][name]["gpt-5.5"]
            assert set(entry) >= {"auc", "n_clusters", "n_positives", "n_controls"}

    def test_the_strata_contrast_is_independent_not_paired(self) -> None:
        """§2.2 excludes the 37 legacy repositories from the pool's population, so the two
        strata share no repository and there is nothing to pair on."""
        import inspect

        assert "share no repository" in inspect.getsource(jvp.transportability)

    def test_nothing_here_carries_a_consequence(self, tmp_path: Path) -> None:
        analysis, ctl, v = _analysis()
        out = jvp.transportability(
            MODELS, analysis, ctl, v, iters=ITERS, candidates=_candidate_csv(tmp_path, analysis)
        )
        assert out["pre_committed_consequence"] is None
        assert out["underpowered"] is True

    def test_a_pool_case_absent_from_the_candidate_list_raises(self) -> None:
        """Every walked candidate came from that file, so a miss means the analysis is reading
        a different list than the walk used."""
        analysis, ctl, v = _analysis(n_legacy=0)
        with pytest.raises(SystemExit) as exc:
            jvp.transportability(MODELS, analysis, ctl, v, iters=ITERS)
        assert "absent from" in str(exc.value)

    def test_the_pre_declared_sensitivity_costs_nothing(self) -> None:
        """§2.3 retains `ids_v2(T0) >= 10` as a strict subset of the judged set, so nothing is
        re-mined and nothing is re-judged."""
        analysis, ctl, v = _analysis()
        out = jvp.sensitivity_seeds_ge_10(MODELS, analysis, ctl, v, iters=ITERS)
        assert out["threshold"] == 10
        assert out["n_positives"] == 16  # the even-indexed clusters carry seeds_at_t0 = 12
        assert out["pre_committed_consequence"] is None


class TestTheContaminationSplitIsAnExplicitVoid:
    def test_the_cutoffs_are_still_blank_in_the_registered_file(self) -> None:
        """Parsed from §5 rather than hard-coded, because §7 requires them recorded there and
        not filled in from code after the positives are visible."""
        assert jvp.training_cutoffs() == {"gpt-5.5": None, "claude-sonnet-5": None}

    def test_it_reports_not_computed_with_both_reasons(self) -> None:
        analysis, ctl, v = _analysis()
        out = jvp.contamination_split(MODELS, analysis, ctl, v, iters=ITERS)
        assert out["status"] == "not_computed"
        assert out["cutoffs"] == {m: None for m in MODELS}
        assert "deliberately does not guess it" in out["why"]
        assert "44 of 94" in out["why"]

    def test_it_states_the_consequence_rather_than_burying_it(self) -> None:
        """§6 item 6 names this split as the only instrument against recognition bias."""
        analysis, ctl, v = _analysis()
        out = jvp.contamination_split(MODELS, analysis, ctl, v, iters=ITERS)
        assert "unmitigated confound" in out["consequence"]
        assert "PREREG-judge-validity-pool.md" in out["how"]

    def test_the_values_are_null_and_never_zero(self) -> None:
        """`mcp_arm_report` set the shape: an unrun arm reported as 0 reads as a measurement,
        and this is its absence."""
        analysis, ctl, v = _analysis()
        out = jvp.contamination_split(MODELS, analysis, ctl, v, iters=ITERS)
        assert all(out["cutoffs"][m] is None for m in MODELS)
        assert 0 not in out["cutoffs"].values()

    def test_undated_legacy_rows_are_kept_apart_from_failed_searches(self) -> None:
        """~30 structurally undated legacy rows must not read as 30 failed searches on the
        pool."""
        analysis, ctl, v = _analysis()
        cov = jvp.contamination_split(MODELS, analysis, ctl, v, iters=ITERS)[
            "adoption_date_coverage"
        ]
        assert cov["n_undated_legacy"] == 16 and cov["n_undated_pool_miss"] == 0

    def test_with_cutoffs_supplied_it_splits_on_the_date(self) -> None:
        analysis, ctl, v = _analysis(n_legacy=0, n_pool=4)
        out = jvp.contamination_split(
            MODELS,
            analysis,
            ctl,
            v,
            iters=ITERS,
            cutoffs={m: "2025-01-01" for m in MODELS},
        )
        assert out["status"] == "computed"
        assert out["by_judge"]["gpt-5.5"]["n_post_cutoff"] == 16
        assert "never be differenced" in out["_note"]

    def test_a_control_without_its_anchor_raises(self) -> None:
        """The pool-scheme drawer writes no `for_positive`, so the subset could not be joined."""
        analysis, ctl, v = _analysis(n_legacy=0, n_pool=2)
        for row in ctl:
            del row["for_positive"]
        with pytest.raises(SystemExit) as exc:
            jvp.contamination_split(
                MODELS, analysis, ctl, v, iters=ITERS, cutoffs={m: "2025-01-01" for m in MODELS}
            )
        assert "for_positive" in str(exc.value)


class TestEachPredictionIsScoredFromWhatItNames:
    SUMMARY = {
        "q_over_b0": 0.15,
        "y_over_b0": 1.4,
        "capped_positives": 100,
        "walked": 800,
        "budget": 1200,
        "n_prefix_decided": 290,
    }

    def test_the_yield_rates_come_from_the_unconditional_prefix(self) -> None:
        """A prefix whose length was chosen by the yield it produced is the inverse sampling
        §3.2 exists to avoid."""
        out = jvp.score_predictions(self.SUMMARY, {"analysis_set_positives": 130}, {}, {}, {})
        assert out["P3"]["observed"] == 0.15 and out["P3"]["in_bracket"] is True
        assert out["P4"]["observed"] == 1.4 and out["P4"]["in_bracket"] is True

    def test_p5_reads_the_stop_count_not_the_analysis_set(self) -> None:
        out = jvp.score_predictions(self.SUMMARY, {"analysis_set_positives": 71}, {}, {}, {})
        assert out["P5"]["stop_rule_capped_positives"] == 100 and out["P5"]["met"] is True

    def test_p6_is_not_scored_below_the_n_it_registered_at(self) -> None:
        """Scoring it at 71 would convert a power shortfall into a failed prediction, and §3.4
        requires the AUC to run at whatever n exists regardless."""
        out = jvp.score_predictions(self.SUMMARY, {"analysis_set_positives": 71}, {}, {}, {})
        assert out["P6"]["status"] == "not_evaluable_at_this_n"
        assert "130" in out["P6"]["why"]

    def test_p6_is_scored_once_the_n_is_there(self) -> None:
        primaries = {m: {"auc": 0.65, "excludes_half": True} for m in MODELS}
        out = jvp.score_predictions(
            self.SUMMARY,
            {"analysis_set_positives": 130},
            primaries,
            {},
            {"excludes_zero": False},
        )
        assert out["P6"]["both_exclude_half"] is True

    def test_p8_needs_both_judges(self) -> None:
        """One judge clearing 1.5 does not establish that the paper-level interval would have
        been materially too narrow."""
        primaries = {"gpt-5.5": {"design_effect": 2.0}, "claude-sonnet-5": {"design_effect": 1.1}}
        out = jvp.score_predictions(
            self.SUMMARY, {"analysis_set_positives": 130}, primaries, {}, {}
        )
        assert out["P8"]["met"] is False
        assert out["P8"]["observed"]["gpt-5.5"] == 2.0

    def test_p9_counts_gross_adoptions_and_says_genesis_is_structurally_zero(
        self, tmp_path
    ) -> None:
        src = tmp_path / "a.json"
        src.write_text(
            json.dumps(
                [{"reverse_cited": True, "genesis": False}] * 1
                + [{"reverse_cited": False, "genesis": False}] * 9
            ),
            encoding="utf-8",
        )
        out = jvp.score_predictions(
            self.SUMMARY, {"analysis_set_positives": 130}, {}, {}, {}, pool_adoptions=src
        )
        assert out["P9"]["n_gross_adoptions"] == 10 and out["P9"]["share"] == 0.1
        assert out["P9"]["met"] is True
        assert "structurally zero" in out["P9"]["_note"]

    def test_p1_and_p2_are_not_rescored_here(self) -> None:
        """Both were scored against the legacy re-mine before the walk and are recorded in §8."""
        out = jvp.score_predictions(self.SUMMARY, {"analysis_set_positives": 130}, {}, {}, {})
        assert out["P1"]["status"] == "scored_before_the_walk"


class TestTheDatasheetNamesEveryComponent:
    def test_all_seven_are_present_by_name(self) -> None:
        """A missing component must be visible as an empty one, not as an absent key."""
        analysis, ctl, v = _analysis(n_legacy=1, n_pool=0)
        sheet = jvp.datasheet("SEED", analysis, ctl, v, [], models=MODELS)
        for name in jvp.DATASHEET_COMPONENTS:
            assert name in sheet

    def test_the_raw_ordinal_scores_are_published_unthresholded(self) -> None:
        """The primary is computed over the four rubric levels, and the legacy artefact stores
        only counts above the bar — so this distribution could not be recovered from it."""
        analysis, ctl, v = _analysis(n_legacy=1, n_pool=0)
        sheet = jvp.datasheet("SEED", analysis, ctl, v, [], models=MODELS)
        scores = sheet["raw_ordinal_scores"]["gpt-5.5"]
        assert {r["score"] for r in scores} == {0, 3}

    def test_void_and_timeout_are_separate_lists(self) -> None:
        analysis, ctl, v = _analysis(n_legacy=1, n_pool=0)
        sheet = jvp.datasheet(
            "SEED", analysis, ctl, v, [{"outcome": "timeout", "id": "x"}], models=MODELS
        )
        lists = sheet["void_and_timeout_lists"]
        assert set(lists) == {"judging", "walk_timeouts", "walk_failures"}
        assert lists["judging"] == [{"outcome": "timeout", "id": "x"}]

    def test_the_seed_is_tied_in_by_digest_not_duplicated(self) -> None:
        analysis, ctl, v = _analysis(n_legacy=1, n_pool=0)
        sheet = jvp.datasheet("a-secret-pulse-value", analysis, ctl, v, [], models=MODELS)
        assert "a-secret-pulse-value" not in json.dumps(sheet["seed_and_pulse"])
        assert len(sheet["seed_and_pulse"]["seed_sha256"]) == 64

    def test_the_limitations_this_chain_accumulated_are_carried(self) -> None:
        analysis, ctl, v = _analysis(n_legacy=1, n_pool=0)
        sheet = jvp.datasheet("SEED", analysis, ctl, v, [], models=MODELS)
        joined = " ".join(sheet["limitations"])
        assert "documentation channel only" in joined
        assert "44 of 94" in joined and "X5" in joined

    def test_the_ledger_string_false_is_never_read_as_truthy(self) -> None:
        """`csv.DictReader` returns the string "False", which is truthy — an analysis that
        forgot would report every walked candidate as a qualifier."""
        import inspect

        assert 'qualifies") == "True"' in inspect.getsource(jvp.datasheet)
