"""The gate is a third of the paired variance. Worth fixing, not transformative. [NR-54]

NR-52 recorded that the shipped arm's mean net@2 moves between two of our own runs (+5.73,
+5.51). The follow-up quoted a per-case sd of **2.23** and suggested the benchmark's resolution
might be gate-limited. **That framing was wrong**: those two runs used *different pool
directories* — `pool-core25-arxiv` and `pool-cut100` — whose candidate sets share a median
Jaccard of only **0.365**. Identical `pool_size` on all 37 cases (same flags produce the same
`#queries × max_results`) is what made them look comparable; the contents were not.

So this holds the pool **byte-identical** and re-runs the shipped config. Retrieval contributes
nothing; what remains is the gate (Haiku, uncached, no temperature sent → default 1.0), the
fine-scale rescore, and any newly-shown papers.

| | |
|---|---|
| per-case net@2 delta | mean −0.14, **sd 1.44** |
| cases byte-identical | 10 / 37 |
| median digest Jaccard | 0.857 |

**Pre-registered reading: < 1.0 minor, ≥ 1.5 dominates → GREY**, and the grey resolves toward
*worth doing, not transformative*.

| component | sd |
|---|---|
| total, across different pools | 2.23 |
| **gate / downstream** | **1.44** |
| implied pool collection | 1.71 |

**The dividend is smaller than the sd suggests, because resolution scales with the square root
of variance.** The gate is **35% of the paired variance** in a frozen-pool arm, and removing it
tightens the interval by **20%** — half-width 0.78 → **0.63** — not by 35%. The earlier
conversational estimate of ±0.30 assumed the whole 2.23 was gate noise; it is 1.44.

**It does not rescue the ladder.** Rungs in `RESEARCH-net2-directions.md` run +0.20 to +0.45 and
stay below 0.63, so the bundle-only rule survives untouched.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "evals" / "gate_draw_variance.json"


@pytest.fixture(scope="module")
def artifact() -> dict:
    return json.loads(FROZEN.read_text(encoding="utf-8"))


class TestThePoolWasActuallyHeldFixed:
    def test_the_design_isolates_what_it_claims_to(self, artifact):
        """The correction that makes this measurement mean anything. The prior number could not
        isolate the gate because the pools differed; this one can because they do not."""
        assert artifact["pool_held_identical"] is True
        assert artifact["n_cases"] == 37
        assert artifact["context"]["pool_median_jaccard"] == 0.365

    def test_the_earlier_framing_is_named_as_corrected(self, artifact):
        """C-17: the superseded reading is recorded, not deleted. A reader who meets the 2.23
        elsewhere should find out here why it cannot answer this question."""
        assert "DIFFERENT pools" in artifact["_comment"]
        assert "could never have isolated the gate" in artifact["_comment"]


class TestTheGateIsStochasticButNotDominant:
    def test_the_measured_sd(self, artifact):
        assert artifact["gate_sd_per_case"] == pytest.approx(1.44, abs=0.01)
        assert artifact["mean_delta"] == pytest.approx(-0.14, abs=0.01)

    def test_most_cases_change_even_with_the_pool_fixed(self, artifact):
        """10 of 37 byte-identical, median digest Jaccard 0.857. With retrieval held constant,
        ~14% of the digest still moves between runs — that is the gate resampling."""
        assert artifact["cases_byte_identical"] == 10
        assert artifact["median_digest_jaccard"] == pytest.approx(0.857, abs=0.005)

    def test_it_lands_in_the_pre_registered_grey_band(self, artifact):
        v = artifact["verdict"]
        assert v["grey"] is True
        assert v["gate_dominates"] is False
        assert v["gate_minor"] is False

    def test_the_pool_is_still_the_larger_component(self, artifact):
        """1.71 against 1.44. Retrieval drift remains the bigger source, which is the opposite
        of what the 2.23 was being read to imply."""
        assert artifact["context"]["implied_pool_component_sd"] == pytest.approx(1.71, abs=0.01)
        assert artifact["verdict"]["pool_is_the_larger_component"] is True


class TestTheDividendIsComputedInTheOpen:
    def test_resolution_improves_by_a_fifth_not_a_third(self, artifact):
        """The arithmetic that decides it: 35% of the variance removed buys 20% off the
        interval, because resolution goes as the square root."""
        m = artifact["measurement_dividend"]
        assert m["gate_share_of_variance"] == pytest.approx(0.35, abs=0.01)
        assert m["half_width_if_gate_deterministic"] == pytest.approx(0.63, abs=0.01)
        assert m["tighter_by"] == pytest.approx(0.20, abs=0.01)

    def test_the_earlier_overestimate_is_corrected_on_the_record(self, artifact):
        """±0.30 was claimed in conversation on the assumption that all 2.23 was gate noise.
        Recorded here with its cause rather than quietly replaced."""
        assert "0.30" in m_comment(artifact)
        assert "assumed the whole 2.23 was gate noise" in m_comment(artifact)

    def test_the_ladder_still_cannot_be_resolved_rung_by_rung(self, artifact):
        """+0.20 to +0.45 against a best case of 0.63. The bundle-only rule is unaffected, so
        this changes the project's hygiene and not its plan."""
        assert artifact["measurement_dividend"]["ladder_rungs_still_unresolvable"] is True
        assert "not enough" in artifact["verdict"]["reading"].lower()


def m_comment(artifact: dict) -> str:
    return artifact["measurement_dividend"]["_comment"]
