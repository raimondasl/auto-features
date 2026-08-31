"""Round 2's papers do reach the gate — a fifth of it. [NR-50]

NR-49 closed PRF-HyDE on reach and left one thing open: reach is counted over 520 witnesses
while the pools differ across their whole contents, so net@2 was not settled. NR-47 and NR-48
had between them shown what actually binds — the gate reads a fixed `gate_depth` of a **ranked**
pool, so a candidate outside the top 50 cannot change the digest whatever its quality.

**Round 2 takes 340 of 1650 window slots — 20.61%, 10.3 per case**, displacing 349 shipped
papers, spread from 2/50 (`bio-kmer`) to 21/50 (`thin-kv`) across every cohort.

| pre-registered ladder | at Δp = 0.2 | at Δp = 0.092 | |
|---|---|---|---|
| < 5% | < 0.25 | < 0.11 | kill |
| 5–16% | 0.25–0.78 | 0.11–0.36 | grey |
| **≥ 16% ← 20.61% lands here** | ≥ 0.78 | ≥ 0.36 | **paid arm licensed** |

**The share is a magnitude, not a direction.** If round 2's papers are worse than the ones they
displace, the same arithmetic gives a loss of the same size. What the probe establishes is that
the effect would be *resolvable*, which is precisely what NR-49's reach null could not say.

**And the free quality prior points the other way from the licence.** Where any judgement
exists, papers entering score **0.714** and papers displaced **0.660** — Δp = **+0.054**,
favouring round 2 but implying only **+0.28 net@2/case**, inside the ±0.78 the bootstrap
resolves at n = 37. Both are recorded. The pre-registration is honoured because moving a bar
after seeing the data is the exact failure NR-49 documented; the prior is recorded because it
is what a reader deciding whether to spend needs. The Δp is itself weak: **61% of entering and
73% of displaced papers are void**, and the judged subset is selected by having been shown.

**The depth grid says round 2 is marginal, and the probe committed to that reading first.**
`gate_depth` 100 had never been run at the shipped `top_k` — NR-48 moved depth only at
`top_k` 1000 — so the ranking is taken once to 300 and every share derived from it:

| depth | 25 | **50** | 100 | 150 | 300 |
|---|---|---|---|---|---|
| round-2 share | 17.58% | **20.61%** | 22.33% | 24.14% | 26.62% |

The share **rises**, which the docstring committed in advance to reading as *"round 2's papers
are marginal and 20.61% flattered them"*. Density by band makes it plain: **15.2%** of ranks
1–10, 23.6% of 26–50, 29.1% of 151–300. Round 2 is weakest exactly where the digest is drawn.

Since the digest is the top-15 window, round 2's **16.77%** share there is the better predictor
of its digest share than the 20.61% the decision was made on — cutting the implied effect from
+1.03 to **+0.84** at a generous Δp and to **+0.23** at the Δp actually measured.

**The licence is not revoked.** It was registered at depth 50 and depth 50 is what it reads.
Picking the depth that flatters a result after seeing five of them is the same error as moving
a bar, wearing a grid instead.

**Why this is the interesting result.** Round 2's candidates are **72% new** to the pool, the
ranker rates them highly enough for a fifth of the window, and they are *not* witness material —
NR-49 measured reach as flat. Those three only fit together one way: round 2 retrieves papers
that look strongly repo-relevant and that no other system surfaced. Discovery or drift is what
net@2 would decide.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "evals" / "prf_rank_probe.json"


@pytest.fixture(scope="module")
def artifact() -> dict:
    return json.loads(FROZEN.read_text(encoding="utf-8"))


class TestRoundTwoReachesTheGate:
    def test_it_holds_a_fifth_of_the_window(self, artifact):
        assert artifact["cases"] == 33
        assert artifact["gate_depth"] == 50
        assert artifact["round2_window_slots"] == 340
        assert artifact["window_slots_total"] == 1650
        assert artifact["share_of_window"] == pytest.approx(0.2061, abs=0.001)

    def test_the_pre_registered_licence_fires(self, artifact):
        v = artifact["verdict"]
        assert v["killed"] is False
        assert v["in_grey_band"] is False
        assert v["licenses_paid_arm"] is True

    def test_it_is_not_carried_by_a_few_cases(self, artifact):
        """C-7's shape. Every case contributes and the spread is wide but not degenerate —
        a result driven by two cohorts would be a different claim entirely."""
        counts = [c["round2_in_window"] for c in artifact["per_case"].values()]
        assert len(counts) == 33
        assert min(counts) >= 2
        assert max(counts) <= 21
        assert sum(1 for c in counts if c >= 5) >= 25

    def test_displacement_is_reported_and_slightly_exceeds_insertion(self, artifact):
        """349 displaced against 340 admitted. The extra nine are shipped papers reordering
        among themselves: the merged pool changes BM25 and the RRF fusion, so the baseline
        ranking is perturbed too. Recorded rather than rounded away."""
        assert artifact["displaced_total"] == 349
        assert artifact["displaced_total"] >= artifact["round2_window_slots"]


class TestTheBarWasSetBeforeTheDataAndWithAnEffectSize:
    def test_all_three_rungs_are_registered(self, artifact):
        pre = artifact["pre_registered"]
        assert pre["kill_below"] == 0.05
        assert pre["license_at_or_above"] == 0.16
        assert pre["share_needed_at_observed_dp"] == 0.34

    def test_the_arithmetic_is_written_down_not_asserted(self, artifact):
        """The correction NR-49 earned. A bar with no effect size can be cleared by noise, so
        this one carries the chain that produced it — window to digest to 3Δp to bootstrap."""
        a = artifact["pre_registered"]["arithmetic"]
        assert "0.166" in a
        assert "0.78" in a
        assert "necessary, not sufficient" in a
        assert "NR49" in artifact["pre_registered"]["fixes_what_NR49_got_wrong"].replace("-", "")

    def test_the_configuration_is_the_generous_one(self, artifact):
        """Round 2 at top_k 100 merged into the WHOLE shipped pool, not NR-49's budget match.
        A kill here would have been decisive for the matched arm too."""
        assert "deliberately generous" in artifact["_comment"]


class TestTheDepthGridFillsAnEmptyCell:
    """`gate_depth` 100 had never been run at the shipped `top_k`. NR-48 varied depth only at
    `top_k` 1000 (arms B and C), and the only other depth comparison on record is a 22-case
    pool-300-against-50 wash from 2026-08-07 that predates HyDE, verified bigrams and
    `w_embedding`. One deep ranking pass fills the row for free."""

    def test_every_depth_is_derived_from_one_pass(self, artifact):
        d = artifact["by_depth"]
        assert set(d) == {"depth_25", "depth_50", "depth_100", "depth_150", "depth_300"}
        assert d["depth_100"]["share"] == pytest.approx(0.2233, abs=0.001)

    def test_only_depth_50_is_marked_as_the_decision(self, artifact):
        """The guard against reading the grid as five chances to pass. Exactly one depth is
        the registered one, and it is the one the bar was set at."""
        marked = [
            k for k, v in artifact["by_depth"].items() if v["is_the_registered_decision_depth"]
        ]
        assert marked == ["depth_50"]

    def test_the_registered_depth_reproduces_the_headline(self, artifact):
        """A self-check: the deep pass must give back exactly what the depth-50 pass gave."""
        assert artifact["by_depth"]["depth_50"]["round2_slots"] == artifact["round2_window_slots"]
        assert artifact["by_depth"]["depth_50"]["share"] == artifact["share_of_window"]


class TestRoundTwoIsMarginalWithinTheWindow:
    def test_the_share_rises_with_depth(self, artifact):
        """Monotone across all five depths. The docstring pre-committed to reading a rising
        share as 'marginal, and 20.61% flattered them' — this is that branch firing, not a
        reading chosen after the fact."""
        shares = [artifact["by_depth"][f"depth_{d}"]["share"] for d in (25, 50, 100, 150, 300)]
        assert shares == sorted(shares)
        assert artifact["top_of_window"]["density_rises_with_depth"] is True

    def test_round_two_is_weakest_at_the_very_top(self, artifact):
        """15.15% of ranks 1-10 — below the 16% licence threshold — against 20.61% of ranks
        1-50. The digest is drawn from the top, so this is the part that matters most."""
        t = artifact["top_of_window"]
        assert t["share_in_ranks_1_10"] < artifact["pre_registered"]["license_at_or_above"]
        assert t["share_in_ranks_1_10"] < t["share_in_ranks_1_50"]

    def test_the_digest_estimate_falls_once_corrected(self, artifact):
        """16.77% of the top-15 rather than 20.61% of the top-50: 1.39 digest papers per case
        instead of 1.71, and +0.84 instead of +1.03 at a generous Δp — now close to the ±0.78
        line rather than comfortably past it."""
        t = artifact["top_of_window"]
        assert t["implied_digest_papers_per_case"] == pytest.approx(1.39, abs=0.02)
        gen = t["implied_net2_from_top15"]["at_generous_dp_0.20"]
        assert gen < artifact["verdict"]["implied_net2_per_case"]["at_generous_dp_0.20"]
        assert t["implied_net2_from_top15"]["at_dp_measured_here_0.054"] < 0.3

    def test_the_licence_still_reads_depth_50(self, artifact):
        """The discipline, again. Every refinement since the bar was set has pointed the same
        way — toward an effect too small to resolve — and none of them revokes a bar that was
        registered in advance. The reader gets both and decides."""
        assert artifact["verdict"]["licenses_paid_arm"] is True
        assert artifact["by_depth"]["depth_50"]["is_the_registered_decision_depth"] is True


class TestTheDirectionIsNotEstablished:
    def test_the_share_is_a_magnitude_only(self, artifact):
        """Symmetric by construction: implied net@2 is |share x 50 x 0.166 x 3dp|, and the sign
        comes from dp, which this probe does not measure with any strength."""
        i = artifact["verdict"]["implied_net2_per_case"]
        assert i["at_generous_dp_0.20"] > 0.78
        assert i["at_observed_dp_0.092"] < 0.78

    def test_the_quality_prior_favours_round_two_but_weakly(self, artifact):
        q = artifact["quality_of_the_swap"]
        assert q["round2_entering"]["precision"] == 0.714
        assert q["shipped_displaced"]["precision"] == 0.66
        assert q["observed_dp"] == pytest.approx(0.054, abs=0.001)

    def test_and_at_that_dp_the_effect_would_not_resolve(self, artifact):
        """The tension the artifact exists to hold open. 20.6% clears a licence set at a
        generous dp = 0.2; the dp actually observed implies +0.28, inside the noise."""
        at = artifact["verdict"]["at_the_dp_measured_here"]
        assert at["implied_net2_per_case"] == pytest.approx(0.277, abs=0.01)
        assert at["would_resolve"] is False

    def test_the_licence_was_not_revoked_after_seeing_the_prior(self, artifact):
        """The discipline. Moving a bar after the data arrives is the failure NR-49 recorded;
        the fix is to report both, not to re-decide."""
        assert artifact["verdict"]["licenses_paid_arm"] is True
        assert "honoured" in artifact["verdict"]["at_the_dp_measured_here"]["_comment"]

    def test_most_of_the_swap_is_void_not_null(self, artifact):
        """207 of 340 entering and 255 of 349 displaced were never judged. C-4 and C-30: an
        absent measurement is not a bad one, and a Δp over the judged remainder is a hint."""
        q = artifact["quality_of_the_swap"]
        assert q["round2_entering"]["void_never_judged"] == 207
        assert q["shipped_displaced"]["void_never_judged"] == 255
        assert "VOID" in artifact["verdict"]["at_the_dp_measured_here"]["_comment"]
