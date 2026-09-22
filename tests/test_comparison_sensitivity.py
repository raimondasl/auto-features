"""Pin how far NR-52's comparison with Opus 5 depends on the penalty, the controls and the cases.

`evals/comparison_sensitivity.json` holds, per label and case, each arm's actionable and
unactionable counts, recovered from the net@2 and shown counts NR-52 stored. These tests recompute
the quoted figures from those rows with the estimators NR-52 used, bigram_report.paired_bootstrap
and band_testbeds.sign_test, rather than reading back the summary the script wrote. They recover
the rows from evals/rung1_second_judge.json a second time, independently, and check the two agree.
They then check the stored summary against the same recomputation. Everything here is descriptive
and post hoc, and nothing in it is a test of a hypothesis.

What is pinned, because prose depends on it:

* The penalty. Under GPT-5.5 the margin is -30/37 + 21/37 * lambda. It changes sign at
  lambda 10/7, so it is negative at lambda 1 and positive from 1.5. Under consensus the root is
  9/8. Under Sonnet the margin is negative for every lambda >= 0. No GPT-5.5 or consensus interval
  excludes zero at any lambda. Sonnet's excludes zero at lambda 1 only.
* Lambda 2 gives NR-52 back: +0.32, +0.57 and -3.41, with its published intervals.
* The negative controls. RepoRadar shows nothing on webdev, cli and http, and Opus 5 shows 31
  papers there. Under GPT-5.5 they carry +11 of the +12 total, and without them the margin is
  +0.03 [-2.15, +2.29]. Under Sonnet the margin without them is -5.35 [-8.41, -2.32].
* The cases. On the 22 development cases the GPT-5.5 margin is +2.27. On the 15 later cases it is
  -2.53 [-5.53, +0.20], and under Sonnet -5.73 [-10.33, -1.47].
* The kappa ceiling. On the aug20 band GPT-5.5 and Sonnet agree at kappa 0.199, binary at >= 2.
  Their marginals allow at most 0.248.
"""

from __future__ import annotations

import json
import statistics
from fractions import Fraction
from pathlib import Path
from typing import Any

# The estimators NR-52 used, not second implementations. evals/ is on pytest's pythonpath.
import band_testbeds as tb
import bigram_report
import pytest
import yaml
from bigram_report import paired_bootstrap
from second_judge import cohens_kappa

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "evals" / "comparison_sensitivity.json"
RUNG1 = ROOT / "evals" / "rung1_second_judge.json"
DEPENDENCE = ROOT / "evals" / "judge_dependence.json"
BENCHMARK = ROOT / "evals" / "benchmark.yaml"

LABELS = ("gpt", "consensus", "sonnet_only")
LAMBDAS = (1.0, 1.5, 2.0, 3.0, 4.0)
CONTROLS = ("webdev", "cli", "http")
FIELDS = {"rr_a", "rr_u", "op_a", "op_u", "delta_net2"}
DEVELOPMENT = {
    "ann",
    "cli",
    "columnar",
    "compiler",
    "crypto",
    "cv",
    "db",
    "diffusion",
    "encryption",
    "graph",
    "http",
    "linter",
    "llminfer",
    "numerics",
    "peft",
    "rag",
    "rl",
    "speech",
    "storage",
    "systems",
    "vectordb",
    "webdev",
}
THIN = {"thin-gnn", "thin-kv", "thin-lang"}
SCIENTIFIC = {
    "bio-align",
    "bio-kmer",
    "bio-mdsim",
    "bio-mdtraj",
    "bio-scvi",
    "bio-singlecell",
    "mat-chgpot",
    "mat-descriptors",
    "mat-featurize",
    "mat-mlip",
    "mat-phonon",
    "mat-toolkit",
}

# Each tolerance is half the last digit the figure is quoted to, so a misprint of one fails.
DP2 = 0.005
DP3 = 0.0005

# NR-52 as published, lambda = 2.
PUBLISHED = {
    "gpt": (0.32, -1.78, 2.51),
    "consensus": (0.57, -1.73, 2.92),
    "sonnet_only": (-3.41, -7.00, 0.54),
}
# (label, lambda): margin, interval, wins, losses, ties.
SWEEP = {
    ("gpt", 1.0): (-0.24, -1.97, 1.51, 18, 16, 3),
    ("gpt", 1.5): (0.04, -1.86, 1.97, 19, 18, 0),
    ("gpt", 2.0): (0.32, -1.78, 2.51, 17, 17, 3),
    ("gpt", 3.0): (0.89, -1.78, 3.81, 18, 18, 1),
    ("gpt", 4.0): (1.46, -1.89, 5.22, 20, 16, 1),
    ("consensus", 1.0): (-0.08, -1.86, 1.70, 19, 15, 3),
    ("consensus", 1.5): (0.24, -1.77, 2.24, 20, 17, 0),
    ("consensus", 2.0): (0.57, -1.73, 2.92, 18, 16, 3),
    ("consensus", 3.0): (1.22, -1.76, 4.41, 19, 17, 1),
    ("consensus", 4.0): (1.86, -1.84, 6.05, 21, 15, 1),
    ("sonnet_only", 1.0): (-2.73, -5.03, -0.32, 11, 24, 2),
    ("sonnet_only", 1.5): (-3.07, -5.97, 0.07, 13, 23, 1),
    ("sonnet_only", 2.0): (-3.41, -7.00, 0.54, 13, 23, 1),
    ("sonnet_only", 3.0): (-4.08, -9.14, 1.62, 12, 24, 1),
    ("sonnet_only", 4.0): (-4.76, -11.27, 2.76, 11, 26, 0),
}
# label: sum of delta a, sum of delta u over the 37 cases, and the crossover lambda.
LINEAR = {
    "gpt": (-30, -21, Fraction(10, 7)),
    "consensus": (-27, -24, Fraction(9, 8)),
    "sonnet_only": (-76, 25, None),
}
PRECISION = {"gpt": (0.889, 0.846), "consensus": (0.882, 0.832), "sonnet_only": (0.585, 0.714)}
# label: control deltas, total, and the margin without the controls with its interval and w/l/t.
CONTROL_FIGURES = {
    "gpt": ({"webdev": 4, "cli": -3, "http": 10}, 12, (0.03, -2.15, 2.29, 15, 16, 3)),
    "consensus": ({"webdev": 7, "cli": 3, "http": 16}, 21, (-0.15, -2.38, 2.12, 15, 16, 3)),
    "sonnet_only": ({"webdev": 7, "cli": 12, "http": 37}, -126, (-5.35, -8.41, -2.32, 10, 23, 1)),
}
OPUS5_SHOWN_ON_CONTROLS = {"webdev": 5, "cli": 6, "http": 20}
# (label, group): margin, interval, wins, losses, ties.
GROUPS = {
    ("gpt", "development"): (2.27, -0.50, 5.05, 13, 9, 0),
    ("gpt", "later"): (-2.53, -5.53, 0.20, 4, 8, 3),
    ("gpt", "scientific"): (-2.58, -6.08, 0.75, 4, 6, 2),
    ("gpt", "core"): (1.72, -0.72, 4.28, 13, 11, 1),
    ("consensus", "development"): (2.82, -0.05, 5.73, 14, 8, 0),
    ("consensus", "later"): (-2.73, -5.93, 0.13, 4, 8, 3),
    ("consensus", "scientific"): (-2.83, -6.67, 0.75, 4, 6, 2),
    ("consensus", "core"): (2.20, -0.32, 4.88, 14, 10, 1),
    ("sonnet_only", "development"): (-1.82, -6.91, 3.86, 8, 13, 1),
    ("sonnet_only", "later"): (-5.73, -10.33, -1.47, 5, 10, 0),
    ("sonnet_only", "scientific"): (-4.58, -10.00, 0.25, 5, 7, 0),
    ("sonnet_only", "core"): (-2.84, -7.44, 2.40, 8, 16, 1),
}
GROUP_SIZES = {"development": 22, "later": 15, "scientific": 12, "core": 25}


@pytest.fixture(scope="module")
def artifact() -> dict:
    # Asserted rather than skipped. The artifact is tracked, so its absence is a broken
    # repository, and a skip here would be this project's own void-not-null failure: the
    # suite would stay green while the figures it guards went unchecked.
    assert ARTIFACT.is_file(), f"{ARTIFACT} is tracked and must be present"
    return json.loads(ARTIFACT.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def rung1() -> dict:
    assert RUNG1.is_file(), f"{RUNG1} is tracked and must be present"
    return json.loads(RUNG1.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def dependence() -> dict:
    assert DEPENDENCE.is_file(), f"{DEPENDENCE} is tracked and must be present"
    return json.loads(DEPENDENCE.read_text(encoding="utf-8"))


def _rows(artifact: dict, label: str) -> dict[str, dict[str, int]]:
    return artifact["per_case"][label]


def _deltas(rows: dict[str, dict[str, int]], cases: list[str], lam: float) -> list[float]:
    return [
        float((rows[c]["rr_a"] - lam * rows[c]["rr_u"]) - (rows[c]["op_a"] - lam * rows[c]["op_u"]))
        for c in cases
    ]


def _margin(d: list[float]) -> tuple[float, float, float, int, int, int]:
    lo, hi = paired_bootstrap(d)
    sg = tb.sign_test(d)
    return statistics.mean(d), lo, hi, sg["pos"], sg["neg"], sg["ties"]


def _check(got: tuple, want: tuple, where: Any) -> None:
    """Margin and interval to the quoted 2 dp, and wins, losses and ties exactly."""
    assert got[:3] == pytest.approx(want[:3], abs=DP2), where
    assert got[3:] == want[3:], where


@pytest.fixture(scope="module")
def sweep(artifact: dict) -> dict[tuple[str, float], tuple]:
    """Every lambda cell, recomputed from the artifact's rows. About 15 bootstraps, so once."""
    out = {}
    for label in LABELS:
        rows = _rows(artifact, label)
        for lam in LAMBDAS:
            out[(label, lam)] = _margin(_deltas(rows, sorted(rows), lam))
    return out


@pytest.fixture(scope="module")
def groups(artifact: dict) -> dict[tuple[str, str], tuple]:
    members = _group_members(artifact)
    return {
        (label, name): _margin(_deltas(_rows(artifact, label), cases, 2.0))
        for label in LABELS
        for name, cases in members.items()
    }


def _group_members(artifact: dict) -> dict[str, list[str]]:
    everything = set(_rows(artifact, "gpt"))
    return {
        "development": sorted(DEVELOPMENT),
        "later": sorted(everything - DEVELOPMENT),
        "scientific": sorted(SCIENTIFIC),
        "core": sorted(everything - SCIENTIFIC),
    }


def _includes_zero(lo: float, hi: float) -> bool:
    return lo <= 0 <= hi


class TestTheArtifactIsComplete:
    def test_three_labels_of_37_cases(self, artifact: dict) -> None:
        assert set(artifact["per_case"]) == set(LABELS)
        for label in LABELS:
            rows = _rows(artifact, label)
            assert len(rows) == 37, label
            assert sorted(rows) == sorted(_rows(artifact, "gpt")), label
            for case, r in rows.items():
                assert set(r) == FIELDS, (label, case)

    def test_counts_are_non_negative_integers(self, artifact: dict) -> None:
        for label in LABELS:
            for case, r in _rows(artifact, label).items():
                for field in ("rr_a", "rr_u", "op_a", "op_u"):
                    assert isinstance(r[field], int) and r[field] >= 0, (label, case, field)
                net = (r["rr_a"] - 2 * r["rr_u"]) - (r["op_a"] - 2 * r["op_u"])
                assert r["delta_net2"] == net, (label, case)

    def test_the_rows_are_nr52s(self, artifact: dict, rung1: dict) -> None:
        """Recovered again from rung1_second_judge.json: a = (net + 2n) / 3, u = (n - net) / 3."""
        for label in LABELS:
            stored = rung1["labels"][label]["per_case"]
            assert set(stored) == set(_rows(artifact, label))
            for case, s in stored.items():
                want = {}
                for arm, net, n in (("rr", s["rr"], s["rr_n"]), ("op", s["opus5"], s["opus5_n"])):
                    assert (net + 2 * n) % 3 == 0 and (n - net) % 3 == 0, (label, case, arm)
                    want[f"{arm}_a"], want[f"{arm}_u"] = (net + 2 * n) // 3, (n - net) // 3
                    assert min(want[f"{arm}_a"], want[f"{arm}_u"]) >= 0, (label, case, arm)
                want["delta_net2"] = s["delta"]
                assert _rows(artifact, label)[case] == want, (label, case)

    def test_the_shown_counts_are_nr52s(self, artifact: dict, rung1: dict) -> None:
        for label in LABELS:
            rows = _rows(artifact, label).values()
            assert sum(r["rr_a"] + r["rr_u"] for r in rows) == 306
            assert sum(r["op_a"] + r["op_u"] for r in rows) == 357
            assert rung1["labels"][label]["rr_papers_scored"] == 306
            assert rung1["labels"][label]["opus5_papers_scored"] == 357

    def test_the_bootstrap_is_nr52s(self, artifact: dict) -> None:
        boot = artifact["summary"]["bootstrap"]
        assert (boot["draws"], boot["seed"]) == (
            bigram_report.BOOTSTRAP_N,
            bigram_report.BOOTSTRAP_SEED,
        )
        assert (boot["draws"], boot["seed"]) == (10000, 20260812)

    def test_it_says_it_is_post_hoc(self, artifact: dict) -> None:
        assert "post hoc" in artifact["summary"]["what"]


class TestLambdaTwoIsNR52:
    @pytest.mark.parametrize("label", LABELS)
    def test_from_the_artifacts_rows(self, sweep: dict, label: str) -> None:
        margin, lo, hi = PUBLISHED[label]
        assert sweep[(label, 2.0)][:3] == pytest.approx((margin, lo, hi), abs=DP2)

    @pytest.mark.parametrize("label", LABELS)
    def test_from_rung1_second_judge_directly(self, rung1: dict, label: str) -> None:
        """The stored per-case deltas, bootstrapped the way rung1_second_judge.report does."""
        per_case = rung1["labels"][label]["per_case"]
        d = [float(per_case[c]["delta"]) for c in sorted(per_case)]
        lo, hi = paired_bootstrap(d)
        assert [round(statistics.mean(d), 2), round(lo, 2), round(hi, 2)] == list(PUBLISHED[label])
        stored = rung1["labels"][label]
        assert [stored["margin"], *stored["ci95"]] == list(PUBLISHED[label])


class TestThePenalty:
    @pytest.mark.parametrize(("label", "lam"), list(SWEEP))
    def test_margin_at_each_lambda(self, sweep: dict, label: str, lam: float) -> None:
        _check(sweep[(label, lam)], SWEEP[(label, lam)], (label, lam))

    @pytest.mark.parametrize("label", LABELS)
    def test_the_linear_form_is_exact(self, artifact: dict, sweep: dict, label: str) -> None:
        """margin(lambda) = mean(delta a) - lambda * mean(delta u), for every lambda."""
        rows = _rows(artifact, label).values()
        sum_da = sum(r["rr_a"] - r["op_a"] for r in rows)
        sum_du = sum(r["rr_u"] - r["op_u"] for r in rows)
        assert (sum_da, sum_du) == LINEAR[label][:2]
        for lam in LAMBDAS:
            exact = Fraction(sum_da, 37) - Fraction(lam) * Fraction(sum_du, 37)
            assert sweep[(label, lam)][0] == pytest.approx(float(exact), abs=1e-12)

    @pytest.mark.parametrize("label", LABELS)
    def test_the_crossover(self, artifact: dict, label: str) -> None:
        """10/7 under GPT-5.5, 9/8 under consensus, and none under Sonnet."""
        rows = _rows(artifact, label).values()
        root = Fraction(sum(r["rr_a"] - r["op_a"] for r in rows)) / sum(
            r["rr_u"] - r["op_u"] for r in rows
        )
        want = LINEAR[label][2]
        if want is None:
            assert root < 0
        else:
            assert root == want
        stored = artifact["summary"]["penalty"][label]["crossover"]
        assert stored["exact"] == (None if want is None else str(want))
        assert stored["root"] == pytest.approx(float(root), abs=1e-12)

    def test_the_crossovers_to_quoted_precision(self) -> None:
        assert float(LINEAR["gpt"][2]) == pytest.approx(1.43, abs=DP2)
        assert float(LINEAR["consensus"][2]) == pytest.approx(1.125, abs=DP3)

    def test_under_gpt_and_consensus_the_sign_turns_between_1_and_1_5(self, sweep: dict) -> None:
        for label in ("gpt", "consensus"):
            assert sweep[(label, 1.0)][0] < 0 < sweep[(label, 1.5)][0], label

    def test_under_sonnet_the_margin_is_negative_at_every_lambda(self, sweep: dict) -> None:
        assert all(sweep[("sonnet_only", lam)][0] < 0 for lam in LAMBDAS)

    @pytest.mark.parametrize("label", LABELS)
    def test_pooled_precision(self, artifact: dict, label: str) -> None:
        rows = _rows(artifact, label).values()
        rr = sum(r["rr_a"] for r in rows) / sum(r["rr_a"] + r["rr_u"] for r in rows)
        op = sum(r["op_a"] for r in rows) / sum(r["op_a"] + r["op_u"] for r in rows)
        assert (rr, op) == pytest.approx(PRECISION[label], abs=DP3)

    def test_where_each_arm_crosses_the_break_even(self, artifact: dict) -> None:
        """Under GPT-5.5 and consensus both arms clear lambda / (1 + lambda) at every lambda.
        Under Sonnet RepoRadar falls below it from lambda 1.5, and Opus 5 from lambda 3."""
        below: dict[str, set[float]] = {}
        for label in LABELS:
            rows = _rows(artifact, label).values()
            for arm in ("rr", "op"):
                a = sum(r[f"{arm}_a"] for r in rows)
                p = a / (a + sum(r[f"{arm}_u"] for r in rows))
                below[f"{label}/{arm}"] = {lam for lam in LAMBDAS if p <= lam / (1 + lam)}
        assert below == {
            "gpt/rr": set(),
            "gpt/op": set(),
            "consensus/rr": set(),
            "consensus/op": set(),
            "sonnet_only/rr": {1.5, 2.0, 3.0, 4.0},
            "sonnet_only/op": {3.0, 4.0},
        }


class TestThePenaltyIntervalSigns:
    def test_no_gpt_or_consensus_interval_excludes_zero(self, sweep: dict) -> None:
        for label in ("gpt", "consensus"):
            for lam in LAMBDAS:
                assert _includes_zero(*sweep[(label, lam)][1:3]), (label, lam)

    def test_sonnets_excludes_zero_at_lambda_1_only(self, sweep: dict) -> None:
        assert sweep[("sonnet_only", 1.0)][2] < 0
        for lam in LAMBDAS[1:]:
            assert _includes_zero(*sweep[("sonnet_only", lam)][1:3]), lam


class TestTheControls:
    def test_they_are_the_benchmarks_negative_controls(self) -> None:
        bench = yaml.safe_load(BENCHMARK.read_text(encoding="utf-8"))
        flagged = {c["name"] for c in bench["cases"] if c.get("negative_control")}
        assert flagged == set(CONTROLS)

    @pytest.mark.parametrize("label", LABELS)
    def test_repo_radar_shows_nothing_on_them(self, artifact: dict, label: str) -> None:
        rows = _rows(artifact, label)
        for case in CONTROLS:
            assert rows[case]["rr_a"] + rows[case]["rr_u"] == 0, (label, case)
            shown = rows[case]["op_a"] + rows[case]["op_u"]
            assert shown == OPUS5_SHOWN_ON_CONTROLS[case], (label, case)

    @pytest.mark.parametrize("label", LABELS)
    def test_their_contribution(self, artifact: dict, label: str) -> None:
        rows = _rows(artifact, label)
        want, total, _ = CONTROL_FIGURES[label]
        assert {c: rows[c]["delta_net2"] for c in CONTROLS} == want
        assert sum(r["delta_net2"] for r in rows.values()) == total
        stored = artifact["summary"]["controls"][label]
        assert stored["deltas"] == want
        assert (stored["controls_sum"], stored["total"]) == (sum(want.values()), total)
        assert stored["per_case_contribution"] == pytest.approx(sum(want.values()) / 37, abs=1e-12)

    def test_under_gpt_they_carry_11_of_12(self, artifact: dict) -> None:
        stored = artifact["summary"]["controls"]["gpt"]
        assert stored["share_of_total"] == pytest.approx(11 / 12, abs=1e-12)
        assert stored["per_case_contribution"] == pytest.approx(0.30, abs=DP2)

    @pytest.mark.parametrize("label", LABELS)
    def test_the_margin_without_them(self, artifact: dict, label: str) -> None:
        rows = _rows(artifact, label)
        kept = [c for c in sorted(rows) if c not in CONTROLS]
        assert len(kept) == 34
        _check(_margin(_deltas(rows, kept, 2.0)), CONTROL_FIGURES[label][2], label)

    def test_interval_signs_without_them(self, artifact: dict) -> None:
        """Without the controls, GPT-5.5's margin is about zero and Sonnet's excludes it."""
        rows = {label: _rows(artifact, label) for label in ("gpt", "sonnet_only")}
        kept = [c for c in sorted(rows["gpt"]) if c not in CONTROLS]
        _, lo, hi, *_ = _margin(_deltas(rows["gpt"], kept, 2.0))
        assert _includes_zero(lo, hi)
        _, lo, hi, *_ = _margin(_deltas(rows["sonnet_only"], kept, 2.0))
        assert hi < 0


class TestTheGroups:
    def test_membership(self, artifact: dict) -> None:
        import comparison_sensitivity as cs

        members = _group_members(artifact)
        assert set(cs.DEVELOPMENT) == DEVELOPMENT and set(cs.THIN) == THIN
        for name, size in GROUP_SIZES.items():
            assert len(members[name]) == size, name
            assert artifact["groups"][name] == members[name], name
        assert set(members["later"]) == SCIENTIFIC | THIN
        assert set(members["core"]) == DEVELOPMENT | THIN

    def test_the_development_list_is_testbed_as(self, artifact: dict) -> None:
        """Testbed A's run file is gitignored, so this compares against it only when present.
        The artifact records that the script made the same comparison when it was written."""
        assert artifact["summary"]["development_list_checked_against_run_file"] is True
        if tb.POOL50.is_file():
            run = json.loads(tb.POOL50.read_text(encoding="utf-8"))
            assert {e["case"] for e in run} == DEVELOPMENT

    def test_the_thin_cases_are_dated(self, artifact: dict) -> None:
        note = artifact["groups"]["_note"]
        assert "2026-08-09" in note and "only the 12 scientific cases" in note
        assert artifact["summary"]["groups_note"] == note

    @pytest.mark.parametrize(("label", "group"), list(GROUPS))
    def test_group_margin(self, groups: dict, label: str, group: str) -> None:
        _check(groups[(label, group)], GROUPS[(label, group)], (label, group))

    def test_only_sonnets_later_interval_excludes_zero(self, groups: dict) -> None:
        excluded = {k for k, v in groups.items() if not _includes_zero(v[1], v[2])}
        assert excluded == {("sonnet_only", "later")}
        assert groups[("sonnet_only", "later")][2] < 0


class TestTheKappaCeiling:
    @staticmethod
    def _labels(dependence: dict) -> tuple[list[int], list[int]]:
        rows = [r for r in dependence["rows"] if r["band"] == "aug20"]
        assert len(rows) == 324
        return (
            [1 if r["gpt"] >= tb.ACTIONABLE else 0 for r in rows],
            [1 if r["sonnet"] >= tb.ACTIONABLE else 0 for r in rows],
        )

    def test_the_table(self, dependence: dict) -> None:
        g, s = self._labels(dependence)
        pairs = list(zip(g, s, strict=True))
        table = tuple(pairs.count(p) for p in ((1, 1), (1, 0), (0, 1), (0, 0)))
        assert table == (156, 127, 4, 37)

    def test_observed_kappa(self, dependence: dict) -> None:
        g, s = self._labels(dependence)
        n = len(g)
        p_g, p_s = sum(g) / n, sum(s) / n
        p_o = sum(1 for x, y in zip(g, s, strict=True) if x == y) / n
        p_e = p_g * p_s + (1 - p_g) * (1 - p_s)
        kappa = cohens_kappa(g, s)
        assert kappa == pytest.approx((p_o - p_e) / (1 - p_e), abs=1e-12)
        assert kappa == pytest.approx(0.199, abs=DP3)

    def test_the_ceiling(self, dependence: dict, artifact: dict) -> None:
        """kappa_max = (p_o_max - p_e) / (1 - p_e), p_o_max = min(p_g, p_s) + min(1-p_g, 1-p_s)."""
        g, s = self._labels(dependence)
        n = len(g)
        p_g, p_s = sum(g) / n, sum(s) / n
        p_e = p_g * p_s + (1 - p_g) * (1 - p_s)
        p_o_max = min(p_g, p_s) + min(1 - p_g, 1 - p_s)
        ceiling = (p_o_max - p_e) / (1 - p_e)
        assert ceiling == pytest.approx(0.248, abs=DP3)
        assert cohens_kappa(g, s) / ceiling == pytest.approx(0.802, abs=DP3)
        stored = artifact["summary"]["kappa"]
        assert stored["kappa"] == pytest.approx(cohens_kappa(g, s), abs=1e-12)
        assert stored["kappa_max"] == pytest.approx(ceiling, abs=1e-12)
        assert stored["table"] == {"both": 156, "gpt_only": 127, "sonnet_only": 4, "neither": 37}


class TestTheSummaryIsTheRows:
    """Every stored figure equals the recomputation from the rows, intervals included."""

    def test_the_sweep(self, artifact: dict, sweep: dict) -> None:
        for (label, lam), (m, lo, hi, w, loss, t) in sweep.items():
            e = artifact["summary"]["penalty"][label]["by_lambda"][f"{lam:g}"]
            assert e["margin"] == pytest.approx(m, abs=1e-12), (label, lam)
            assert e["ci95"] == pytest.approx([lo, hi], abs=1e-12), (label, lam)
            assert (e["wins"], e["losses"], e["ties"], e["n_cases"]) == (w, loss, t, 37)
            assert e["break_even_precision"] == pytest.approx(lam / (1 + lam), abs=1e-12)

    def test_the_arm_means(self, artifact: dict) -> None:
        for label in LABELS:
            rows = _rows(artifact, label).values()
            for lam in LAMBDAS:
                e = artifact["summary"]["penalty"][label]["by_lambda"][f"{lam:g}"]
                rr = statistics.mean(r["rr_a"] - lam * r["rr_u"] for r in rows)
                op = statistics.mean(r["op_a"] - lam * r["op_u"] for r in rows)
                assert e["rr_mean_net"] == pytest.approx(rr, abs=1e-12), (label, lam)
                assert e["opus5_mean_net"] == pytest.approx(op, abs=1e-12), (label, lam)

    def test_the_controls(self, artifact: dict) -> None:
        for label in LABELS:
            rows = _rows(artifact, label)
            stored = artifact["summary"]["controls"][label]
            for part, cases in (
                ("with", sorted(rows)),
                ("without", [c for c in sorted(rows) if c not in CONTROLS]),
            ):
                m, lo, hi, w, loss, t = _margin(_deltas(rows, cases, 2.0))
                e = stored[part]
                assert e["margin"] == pytest.approx(m, abs=1e-12), (label, part)
                assert e["ci95"] == pytest.approx([lo, hi], abs=1e-12), (label, part)
                assert (e["wins"], e["losses"], e["ties"]) == (w, loss, t), (label, part)

    def test_the_groups(self, artifact: dict, groups: dict) -> None:
        for (label, name), (m, lo, hi, w, loss, t) in groups.items():
            e = artifact["summary"]["groups"][label][name]
            assert e["margin"] == pytest.approx(m, abs=1e-12), (label, name)
            assert e["ci95"] == pytest.approx([lo, hi], abs=1e-12), (label, name)
            assert (e["wins"], e["losses"], e["ties"]) == (w, loss, t), (label, name)
            assert e["n_cases"] == GROUP_SIZES[name]
