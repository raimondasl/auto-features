"""Pin the descriptive follow-ups to the third judge (NR-69).

`evals/third_judge_followups.json` is written by `evals/third_judge_followups.py` from two tracked
artifacts, `evals/third_judge.json` and `evals/judge_dependence.json`. These tests rebuild it from
those inputs and compare, and they recompute the figures the record and the paper quote from the
raw rows, with the estimators the script borrows rather than with the script. Everything here is
descriptive and post hoc, and nothing in it is a test of a hypothesis.

What is pinned, because prose depends on it:

* The stage under Gemini on the aug20 band. Against withholding the band it loses 6.86 net@2 per
  repository, [-9.16, -4.81] on NR-66's draws. Against showing the whole band it gains 3.35,
  [+2.41, +4.41]. It admits papers Gemini calls actionable at 0.320. The GPT-5.5 and Sonnet
  figures on the same rows and draws are NR-66's, exactly.
* Nesting on the band. Gemini accepts 90 of 324 papers. All 90 are GPT-5.5 acceptances and 81
  are Sonnet acceptances. Gemini and GPT-5.5 agree at kappa 0.106, which is the most their
  marginals allow. Gemini and Sonnet agree at 0.454, against a ceiling of 0.566.
* The comparison under Gemini. Our arm averages -7.30 net@2 per repository and the baseline
  -5.03, so both fall below abstaining. The controls contribute +62 of the -84 total. Without
  them the margin is -4.29 [-8.21, -0.35].
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

import pytest
import third_judge_followups as tf
from bigram_report import paired_bootstrap
from second_judge import cohens_kappa

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "evals" / "third_judge_followups.json"
THIRD = ROOT / "evals" / "third_judge.json"
DEPENDENCE = ROOT / "evals" / "judge_dependence.json"

pytestmark = pytest.mark.skipif(
    not (ARTIFACT.is_file() and THIRD.is_file() and DEPENDENCE.is_file()),
    reason="the follow-ups need the third judge's tracked artifact",
)


@pytest.fixture(scope="module")
def art() -> dict[str, Any]:
    return dict(json.loads(ARTIFACT.read_text(encoding="utf-8")))


@pytest.fixture(scope="module")
def band() -> list[dict[str, Any]]:
    """The aug20 band rebuilt here, joined on the ids third_judge.json carries."""
    third = json.loads(THIRD.read_text(encoding="utf-8"))
    dep = json.loads(DEPENDENCE.read_text(encoding="utf-8"))
    from anonymous.paper_id import dedup_id

    gem = {
        (m["case"], dedup_id(m["id"])): r["score"]
        for r in third["rows"]
        for m in r["members"]
        if m["population"] == "band"
    }
    return [{**r, "gemini": gem[(r["case"], r["id"])]} for r in dep["rows"] if r["band"] == "aug20"]


def test_the_artifact_rebuilds_from_its_inputs(art: dict[str, Any]) -> None:
    assert json.loads(json.dumps(tf.build())) == art


class TestStageValue:
    def test_the_other_judges_are_nr66_exactly(self, art: dict[str, Any]) -> None:
        dep = json.loads(DEPENDENCE.read_text(encoding="utf-8"))
        nr66 = dep["summary"]["bands"]["aug20"]["value"]["4o"]
        mine = art["stage_value"]["by_judge"]
        for judge in ("gpt", "sonnet"):
            for k in ("stage_minus_none", "stage_minus_all"):
                assert mine[judge][k]["point"] == nr66[judge]["overall"][k]["point"]
                assert mine[judge][k]["ci"] == nr66[judge]["overall"][k]["ci"]

    def test_the_gemini_points_recompute_from_the_rows(self, band: list[dict[str, Any]]) -> None:
        adm = [r for r in band if r["adm4o"]]
        stage = sum(1 if r["gemini"] >= 2 else -2 for r in adm)
        every = sum(1 if r["gemini"] >= 2 else -2 for r in band)
        assert round(stage / 37, 2) == -6.86
        assert round((stage - every) / 37, 2) == 3.35
        assert round(sum(r["gemini"] >= 2 for r in adm) / len(adm), 3) == 0.320

    def test_the_quoted_gemini_figures(self, art: dict[str, Any]) -> None:
        g = art["stage_value"]["by_judge"]["gemini"]
        none, every = g["stage_minus_none"], g["stage_minus_all"]
        assert [round(x, 2) for x in [none["point"], *none["ci"]]] == [-6.86, -9.16, -4.81]
        assert [round(x, 2) for x in [every["point"], *every["ci"]]] == [3.35, 2.41, 4.41]
        assert round(g["band_actionable"]["point"], 3) == 0.278

    def test_the_sign_pattern_holds_for_every_judge(self, art: dict[str, Any]) -> None:
        """The stage beats withholding exactly when its admitted precision clears 2/3."""
        for v in art["stage_value"]["by_judge"].values():
            clears = v["admitted_precision"]["point"] > 2 / 3
            assert (v["stage_minus_none"]["point"] > 0) == clears


class TestNesting:
    def test_every_gemini_acceptance_is_a_gpt_acceptance(self, band: list[dict[str, Any]]) -> None:
        gem = [r for r in band if r["gemini"] >= 2]
        assert len(gem) == 90
        assert sum(r["gpt"] >= 2 for r in gem) == 90
        assert sum(r["sonnet"] >= 2 for r in gem) == 81

    @pytest.mark.parametrize(
        ("a", "b", "kappa", "ceiling"),
        [
            ("gpt", "sonnet", 0.199, 0.248),
            ("gpt", "gemini", 0.106, 0.106),
            ("sonnet", "gemini", 0.454, 0.566),
        ],
    )
    def test_kappa_and_its_ceiling(
        self, band: list[dict[str, Any]], a: str, b: str, kappa: float, ceiling: float
    ) -> None:
        x = [1 if r[a] >= 2 else 0 for r in band]
        y = [1 if r[b] >= 2 else 0 for r in band]
        p_a, p_b = sum(x) / len(x), sum(y) / len(y)
        p_e = p_a * p_b + (1 - p_a) * (1 - p_b)
        k_max = (min(p_a, p_b) + min(1 - p_a, 1 - p_b) - p_e) / (1 - p_e)
        assert round(cohens_kappa(x, y), 3) == kappa
        assert round(k_max, 3) == ceiling


class TestComparison:
    @pytest.fixture(scope="class")
    def per_case(self) -> dict[str, dict[str, Any]]:
        third = json.loads(THIRD.read_text(encoding="utf-8"))
        return dict(third["summary"]["E4"]["per_case"])

    def test_both_arms_fall_below_abstaining(self, per_case: dict[str, dict[str, Any]]) -> None:
        ours = statistics.mean(v["ours"] for v in per_case.values())
        base = statistics.mean(v["baseline"] for v in per_case.values())
        assert (round(ours, 2), round(base, 2), round(ours - base, 2)) == (-7.30, -5.03, -2.27)

    def test_the_controls_and_the_margin_without_them(
        self, per_case: dict[str, dict[str, Any]], art: dict[str, Any]
    ) -> None:
        controls = ("webdev", "cli", "http")
        assert sum(per_case[c]["delta"] for c in controls) == 62
        assert sum(v["delta"] for v in per_case.values()) == -84
        kept = [float(v["delta"]) for c, v in sorted(per_case.items()) if c not in controls]
        lo, hi = paired_bootstrap(kept)
        got = [round(statistics.mean(kept), 2), round(lo, 2), round(hi, 2)]
        assert got == [-4.29, -8.21, -0.35]
        w = art["comparison"]["without_controls"]
        assert [round(w["margin"], 2), *(round(x, 2) for x in w["ci95"])] == got

    @pytest.mark.parametrize(
        ("pair", "want"),
        [
            ("gpt_minus_sonnet", [3.73, 0.41, 6.73]),
            ("gpt_minus_gemini", [2.59, -1.78, 6.89]),
            ("sonnet_minus_gemini", [-1.14, -3.97, 1.62]),
        ],
    )
    def test_the_judges_margins_on_the_same_runs(
        self, per_case: dict[str, dict[str, Any]], pair: str, want: list[float]
    ) -> None:
        """Recomputed from NR-68's per-case deltas and E4's, with NR-52's paired_bootstrap."""
        sens = json.loads((ROOT / "evals" / "comparison_sensitivity.json").read_text("utf-8"))
        deltas = {
            "gpt": {c: v["delta_net2"] for c, v in sens["per_case"]["gpt"].items()},
            "sonnet": {c: v["delta_net2"] for c, v in sens["per_case"]["sonnet_only"].items()},
            "gemini": {c: v["delta"] for c, v in per_case.items()},
        }
        a, b = pair.split("_minus_")
        d = [float(deltas[a][c] - deltas[b][c]) for c in sorted(per_case)]
        lo, hi = paired_bootstrap(d)
        assert [round(statistics.mean(d), 2), round(lo, 2), round(hi, 2)] == want
