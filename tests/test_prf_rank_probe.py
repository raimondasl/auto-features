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
