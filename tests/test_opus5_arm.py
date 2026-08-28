"""P26: the Opus 5 comparator arm, completed to all 37 cases. [C-34]

The arm was quoted at **+1.90** over 25 core + 6 bio. Six materials-science runs finish it,
and they do not extend the result -- they bend it:

* **all 37: +1.08, CI [-0.97, +3.16]**, 20W/15L. The margin now crosses zero.
* **matsci 6: -3.17.** The one cohort RepoRadar loses, and it loses on precision too (0.841
  against Opus 5's 0.895) -- everywhere else the ordering is the other way round.
* **On the 32 cases where Opus 5 does not over-answer, the two systems are level: -0.06.**
  Every point of the margin comes from five cases where it does, four of which RepoRadar
  answers by abstaining entirely. The advantage is shyness, not retrieval.

net@2 charges 2 per false positive precisely to price shyness, so this is a real advantage
and not an artefact. It is simply not the claim "we find better papers", and the paper should
not be allowed to imply that it is.

`evals/results/` is gitignored, so the RepoRadar side is frozen here for the same reason
`gold_targets.json`, `bio_matched_arm.json` and `multisource_arm.json` exist.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "evals" / "opus5_arm.json"


@pytest.fixture(scope="module")
def artifact() -> dict:
    return json.loads(FROZEN.read_text(encoding="utf-8"))


class TestTheArmIsWhatItClaims:
    def test_the_configuration_is_the_one_the_other_opus5_rows_used(self, artifact):
        """Six runs added to an arm collected a day earlier are only comparable if every
        axis matches. `--max-turns` defaults to 12 and every Opus 5 row is at 30, so taking
        the default would have silently produced a different configuration under one name --
        the failure `out_path` and `_discriminator` were built to prevent one level up."""
        cfg = artifact["opus5_config"]
        assert cfg["prompt_version"] == "v2"
        assert cfg["max_turns"] == 30
        assert cfg["model"] == "claude-opus-5"
        assert cfg["effort"] is None, "unpinned; the CLI default, as on all 37"
        assert cfg["cli_auth"] == "subscription"
        assert cfg["draw"] == 1

    def test_it_covers_the_whole_cohort(self, artifact):
        assert artifact["opus5_config"]["n_cases"] == 37
        assert len(artifact["per_case"]) == 37
        assert artifact["cohorts"]["all37"]["n_cases"] == 37

    def test_the_reporadar_arms_differ_only_in_sources(self, artifact):
        """Three same-day arms. The comparison would mean nothing if the window or the
        embedding weight moved between them."""
        arms = artifact["reporadar_arms"]
        assert {a["digest_window"] for a in arms.values()} == {15}
        assert {a["w_embedding"] for a in arms.values()} == {1.5}
        assert arms["arxiv"]["sources"] == ["arxiv"]
        assert arms["arxiv_epmc"]["sources"] == ["arxiv", "europepmc"]
        assert arms["arxiv_openalex"]["sources"] == ["arxiv", "openalex"]


class TestTheCompletedHeadline:
    def test_the_margin_now_crosses_zero(self, artifact):
        """+1.90 over 31 cases became +1.08 over 37, and the interval opened. Nothing was
        re-measured to make that happen -- twelve cases the arm had always intended to cover
        were simply finished. A margin that survives only while a third of its cohort is
        missing is a margin worth reporting with its interval attached."""
        allc = artifact["cohorts"]["all37"]
        assert allc["paired_delta_primary_minus_opus5"] == 1.08
        lo, hi = allc["ci95"]
        assert lo < 0 < hi, "crosses zero"
        assert (allc["wins"], allc["losses"]) == (20, 15)

    def test_the_materials_cohort_reverses_it(self, artifact):
        mat = artifact["cohorts"]["matsci6"]
        assert mat["paired_delta_primary_minus_opus5"] == -3.17
        assert mat["opus5"]["mean_net2"] > mat["arxiv_epmc"]["mean_net2"]

    def test_matsci_is_the_only_cohort_where_opus5_is_more_precise(self, artifact):
        """Which is what stops the reversal being read as a volume artefact. Opus 5 does
        return more per case there (12.7 against 10.5) and net@2 sums over what is returned,
        so volume alone could have produced the sign -- but it wins on the rate as well."""
        for name in ("core25", "bio6"):
            co = artifact["cohorts"][name]
            assert co["arxiv_epmc"]["precision"] > co["opus5"]["precision"], name
        mat = artifact["cohorts"]["matsci6"]
        assert mat["opus5"]["precision"] > mat["arxiv_epmc"]["precision"]
        assert mat["arxiv_epmc"]["precision"] == 0.841, "RepoRadar's worst cohort"
        assert mat["opus5"]["precision"] == 0.895, "Opus 5's best"

    def test_the_cohorts_are_not_averaged(self, artifact):
        """The all37 figure exists and is a blend of a cohort won and a cohort lost. The
        project's convention is to report the parts."""
        core = artifact["cohorts"]["core25"]["paired_delta_primary_minus_opus5"]
        mat = artifact["cohorts"]["matsci6"]["paired_delta_primary_minus_opus5"]
        allc = artifact["cohorts"]["all37"]["paired_delta_primary_minus_opus5"]
        assert mat < allc < core


class TestWhereTheMarginActuallyLives:
    def test_the_two_systems_are_level_when_opus5_does_not_overanswer(self, artifact):
        """**The finding.** 32 of 37 cases, and the gap is -0.06 -- indistinguishable from
        identical. First seen on the core 25 alone (21 cases, -0.29) and now replicated at
        n=32 with twelve cases that were not in the original split."""
        d = artifact["margin_decomposition"]["opus5_not_overanswered"]
        assert d["n_cases"] == 32
        assert abs(d["paired_delta"]) < 0.2

    def test_all_of_the_margin_comes_from_five_cases(self, artifact):
        """105% of it, which is not a rounding artefact: the other 32 cases contribute a
        small NEGATIVE amount, so the five carry more than the whole."""
        over = artifact["margin_decomposition"]["opus5_overanswered"]
        assert over["n_cases"] == 5
        assert over["paired_delta"] == 8.4
        assert over["share_of_total_margin"] > 1.0

    def test_four_of_them_are_cases_reporadar_declines_to_answer(self, artifact):
        """`cli`, `http`, `linter`, `webdev` -- RepoRadar returns nothing and scores 0;
        Opus 5 returns 5 to 20 papers and scores -7 on average. This is the gate refusing a
        repository it has no good candidates for, and it is worth 70% of the total margin.

        It is also the reason a matched-volume comparison cannot be used to check this
        result: capping both systems at min(n) sets k=0 on exactly these four, deleting the
        behaviour under examination and answering a different question.
        """
        ab = artifact["margin_decomposition"]["reporadar_abstained"]
        assert ab["cases"] == ["cli", "http", "linter", "webdev"]
        assert ab["reporadar_mean"] == 0, "abstaining scores 0, never negative"
        assert ab["opus5_mean"] == -7
        assert ab["share_of_total_margin"] == 0.7

    def test_on_the_cases_both_systems_answer_the_margin_is_small(self, artifact):
        ans = artifact["margin_decomposition"]["reporadar_answered"]
        assert ans["n_cases"] == 33
        assert ans["paired_delta"] == 0.36


class TestOpus5IsNotWinningThroughNonArxivMaterial:
    """The question P26 was asked, and the answer runs opposite to the obvious guess.

    The v2 prompt lets the baseline cite anything, and Opus 5 uses that freely -- 34% of its
    core-25 picks and 70% of its bio picks are not arXiv ids. Materials science is where it
    uses it LEAST (6.6%), and materials science is the one cohort it wins.
    """

    def test_the_matsci_win_is_almost_entirely_arxiv(self, artifact):
        mat = artifact["cohorts"]["matsci6"]["opus5"]
        assert mat["n_non_arxiv"] == 5, "of 76 papers shown"
        assert mat["non_arxiv_share"] == 0.066
        assert mat["non_arxiv_net2_per_case"] == 0.33
        assert mat["non_arxiv_net2_per_case"] / mat["mean_net2"] < 0.05, "under 4% of +8.67"

    def test_it_is_opus5s_lowest_non_arxiv_share_of_any_cohort(self, artifact):
        """So the causation cannot run the suggested way: if non-arXiv reach were what beat
        RepoRadar, the cohort with a fifth of the reach would not be the cohort it wins."""
        shares = {
            name: artifact["cohorts"][name]["opus5"]["non_arxiv_share"]
            for name in ("core25", "bio6", "matsci6")
        }
        assert min(shares, key=lambda k: shares[k]) == "matsci6"
        assert shares["core25"] > 0.30 and shares["bio6"] > 0.60

    def test_it_wins_on_arxiv_papers_specifically(self, artifact):
        """Restricted to arXiv papers alone -- the material both systems can reach -- Opus 5
        is more precise on materials science and less precise everywhere else."""
        mat = artifact["cohorts"]["matsci6"]
        assert mat["opus5"]["arxiv_precision"] > mat["arxiv_epmc"]["arxiv_precision"]
        core = artifact["cohorts"]["core25"]
        assert core["opus5"]["arxiv_precision"] < core["arxiv_epmc"]["arxiv_precision"]

    def test_no_source_arm_rescues_reporadar_on_matsci(self, artifact):
        """All three RepoRadar configurations lose to Opus 5 there, and the OpenAlex arm --
        the one that reaches furthest outside arXiv -- is the worst of them. Adding sources
        is not the lever for this cohort."""
        mat = artifact["cohorts"]["matsci6"]
        for label in ("arxiv", "arxiv_epmc", "arxiv_openalex"):
            assert mat[label]["mean_net2"] < mat["opus5"]["mean_net2"], label
        assert mat["arxiv_openalex"]["mean_net2"] < mat["arxiv"]["mean_net2"]


class TestTheAccountingIsInternallyConsistent:
    def test_every_case_appears_once_in_each_split(self, artifact):
        n = len(artifact["per_case"])
        d = artifact["margin_decomposition"]
        assert d["opus5_overanswered"]["n_cases"] + d["opus5_not_overanswered"]["n_cases"] == n
        assert d["reporadar_abstained"]["n_cases"] + d["reporadar_answered"]["n_cases"] == n

    def test_the_arxiv_only_arm_shows_no_non_arxiv_papers(self, artifact):
        for name, co in artifact["cohorts"].items():
            assert co["arxiv"]["n_non_arxiv"] == 0, name

    def test_what_the_arm_cost(self, artifact):
        """Recorded because the comparator is the expensive half of this project and the
        figure decides whether draws 2 and 3 are ever worth finishing."""
        assert artifact["opus5_config"]["cost_usd"] == 351.4
