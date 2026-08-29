"""Item 10 stage 1: a wider HyDE union does reach more witnesses. [NR-46]

NR-45 found 164 of Opus 5's actionable picks sitting in our own index at median rank 1,087,
against a cut of 100. That says the union is too narrow; it does not say widening helps, and
**NR-11** recorded a wider pool meeting a near-binary gate and making the headline *worse*. So
stage 1 asks the cheap question — does reach move — over artifacts on disk, with **no pool
re-collection and no judge calls**: a witness counts as reached at cut *K* if it is already
pooled or its HyDE rank is below K.

| | reach | vs baseline |
|---|---|---|
| pool as collected | 0.1654 | — |
| **baseline: simulated at the shipped cut of 100** | **0.2231** | 0.0000 |
| cut 1,000 | **0.4481** | **+0.2250** |
| cut 5,000 | 0.6077 | +0.3846 |
| cut 10,000 | 0.6712 | +0.4481 |

Pre-registered bar was **≥ 0.25 at K = 1000**, kill below 0.20. Measured **0.4481** — passes
at nearly double, +101% relative.

**The baseline is the simulation at the shipped cut, not the collected pool**, and that
distinction is worth 0.058. The pool was collected with a *different hypothesis draw* — they
are LLM output regenerated per collection, which is why `rr_hyde_hypotheses` is a POOL_FLAG.
The excess over the collected pool is uniform across cases whose hypotheses were cached (6.0%)
and freshly generated (5.4%), which is a draw effect and not an old-versus-new one; index drift
is excluded because the shards were last written 2026-08-06 and the pool collected 2026-08-20.
Measuring the cut against the collected pool would have billed a hypothesis redraw as widening.

**A ceiling bounds all of this at 0.7654.** 121 of the 122 witnesses that no cut reaches are
non-arXiv ids — a dense index of arXiv abstracts cannot return a Europe PMC or OpenAlex paper
however wide the cut. Cut 10,000 reaches 87.7% *of that ceiling*, not of 1.0.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "evals" / "hyde_cut_reach.json"
BAR = 0.25
KILL = 0.20


@pytest.fixture(scope="module")
def artifact() -> dict:
    return json.loads(FROZEN.read_text(encoding="utf-8"))


class TestThePreRegisteredBarWasCleared:
    def test_the_bar_and_kill_were_fixed_before_the_run(self, artifact):
        """The bar has to predate the number or it is a description, not a decision."""
        p = artifact["pre_registered"]
        assert p["bar_at_1000"] == BAR
        assert p["kill_at_1000"] == KILL
        assert p["written_before_running"] is True

    def test_reach_at_1000_clears_it_by_a_wide_margin(self, artifact):
        v = artifact["verdict"]
        assert v["reach_at_1000"] == 0.4481
        assert v["passes_bar"] is True
        assert v["killed"] is False
        assert v["marginal"] is False
        assert v["reach_at_1000"] > 1.5 * BAR

    def test_the_curve_is_monotone_in_the_cut(self, artifact):
        """A wider cut can only add papers. If this ever fails, the rank bookkeeping is
        scrambled and no number in the file means anything."""
        cuts = sorted(int(c) for c in artifact["curve"])
        reached = [artifact["curve"][str(c)]["reached"] for c in cuts]
        assert reached == sorted(reached)


class TestTheBaselineIsLikeForLike:
    def test_the_gain_is_measured_against_the_same_hypotheses(self, artifact):
        """The correction that matters. Comparing cut 1,000 against the *collected pool*
        gives +0.283 and bills a hypothesis redraw as if it were the widening. Against the
        simulation at the shipped cut — same hypotheses, only the cut moving — it is +0.225."""
        v = artifact["verdict"]
        assert v["baseline"] == artifact["curve"]["100"]["p"]
        assert v["absolute_gain"] == 0.225
        assert "not the collected pool" in v["baseline_is"]
        assert v["baseline"] != artifact["pool_as_collected"]["p"]

    def test_the_hypothesis_draw_effect_is_recorded_separately(self, artifact):
        """+0.058 at the same cut, from a different draw of the same generator. Measured by
        accident and kept because it is C-7's rule reaching HyDE: a single draw's level is not
        a property of the method. It is also why the two arms of the paid run must share one
        pinned hypothesis set."""
        h = artifact["hypothesis_draw_effect"]
        assert h["actual"] == 0.1654
        assert h["simulated_same_cut"] == 0.2231
        assert h["delta"] == 0.0577
        assert h["delta"] > 0.05, "large enough to confound an arm that ignored it"


class TestTheCeilingIsStated:
    def test_most_of_what_no_cut_reaches_is_not_arxiv(self, artifact):
        """A dense index of arXiv abstracts cannot return a DOI, at any cut. 121 of 122."""
        w = artifact["witnesses"]
        assert w["total_non_self"] == 520
        assert w["unreachable_by_hyde"] == 122
        assert w["unreachable_non_arxiv"] == 121
        assert w["unreachable_arxiv_not_indexed"] == 1

    def test_the_curve_is_read_against_that_ceiling(self, artifact):
        """0.6712 at cut 10,000 is 87.7% of what this channel can ever reach, not 67% of the
        witness set's total. Recording the ceiling stops the obvious misreading."""
        w = artifact["witnesses"]
        assert w["hyde_ceiling"] == 0.7654
        assert artifact["curve"]["10000"]["p"] < w["hyde_ceiling"]
        assert artifact["verdict"]["share_of_ceiling_at_1000"] == 0.5855


class TestItCoversTheCohortItNeedsTo:
    def test_every_case_has_hypotheses(self, artifact):
        """17 of the 37 had none, including all six materials cases — the cohort we lose
        outright and where 61.8% of NR-45's losses are this stage. Measuring without them
        would have answered for two thirds of the evidence."""
        assert artifact["cases_with_hypotheses"] == 37
        assert artifact["cases_measured"] == 37

    def test_the_comparator_source_moves_most(self, artifact):
        """`cli-v2-opus5@30` is the comparator whose papers NR-45 traced. Its reach goes
        0.202 -> 0.424 at cut 1,000 — the widening is reaching the population the item exists
        to reach, not just any papers."""
        base = artifact["curve"]["100"]["by_source"]["cli-v2-opus5@30"]
        wide = artifact["curve"]["1000"]["by_source"]["cli-v2-opus5@30"]
        assert base["p"] == 0.202
        assert wide["p"] == 0.4238
        assert wide["p"] > 2 * base["p"]


class TestItSaysWhatItCannotSee:
    def test_the_artifact_disclaims_the_second_order_effects(self, artifact):
        """Reach is a lower bound on retrieval and says nothing about net@2. The ranker
        renormalises over a 10x pool and `gate_depth` still shows the gate 50 of it — which is
        exactly where NR-11's damage lived. The caveat travels in the file so a reader who
        greps `reach_at_1000` finds it."""
        c = artifact["_comment"]
        assert "bounds reach from BELOW" in c
        assert "gate_depth" in c
        assert "NR-11" in c
