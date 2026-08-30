"""Item 10 stage 2: the wider HyDE cut costs net@2. The item closes. [NR-47]

The paid arm stage 1 licensed. Two same-day arms over 37 cases, everything fixed but
`hyde.top_k` — 100 against 1000 — sharing **one pinned hypothesis set**, because NR-46
measured a hypothesis redraw at +0.058 witness reach and two arms drawing their own could not
have told the cut from the draw.

**The pre-registered kill condition — "net@2 must not fall" — fires.**

| cohort | control | treatment | paired | |
|---|---|---|---|---|
| **all 37** | +5.51 | +4.73 | **−0.78** CI [−1.59, −0.03] | 13w/17l/7t, p = 0.58 |
| core 25 | +5.92 | +5.40 | −0.52 | |
| bio 6 | +4.67 | +4.17 | −0.50 | |
| matsci 6 | +4.67 | +2.50 | −2.17 | |

Stage 1 was right that reach would double. It bought nothing — the fourth pool expansion this
project has measured as a wash or worse (NR-11, P4, now this), and the first where the
mechanism is visible in the same run.

**The diagnostic is worth more than the headline, and it is not "the papers are bad".**

```
kept     164   precision 0.878
added    110   precision 0.882    +1.92/case
dropped  142   precision 0.901    -2.70/case
digest 8.3 -> 7.4 per case, from a pool 5.9x larger
```

Papers the wider cut adds are **indistinguishable in quality** from the ones already there.
Splitting the −0.78 exactly: **−0.609 from showing 32 fewer papers (78%)**, −0.175 from the
ones it did show being slightly worse. A candidate set six times larger meets a gate that still
reads `gate_depth` 50 of it, so extra reach arrives as dilution and *fewer* admissions.

That is the "the gate never saw them" branch stage 1 pre-registered, and it makes the follow-up
specific: widen the gate's input, not the retrieval cut. **This does not license shipping a
wider cut** — the arm says the opposite.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "evals" / "hyde_cut_arm.json"


@pytest.fixture(scope="module")
def artifact() -> dict:
    return json.loads(FROZEN.read_text(encoding="utf-8"))


class TestTheKillConditionFired:
    def test_net2_fell(self, artifact):
        a = artifact["cohorts"]["all37"]
        assert a["n_cases"] == 37
        assert a["control_mean"] == 5.51
        assert a["treatment_mean"] == 4.73
        assert a["paired_delta"] == -0.78
        assert artifact["verdict"]["killed"] is True

    def test_every_cohort_moves_the_same_way(self, artifact):
        """No cohort is rescued by the wider cut — including matsci, the one NR-45 identified
        as losing 61.8% of its papers at exactly this stage. It falls hardest, at −2.17."""
        for label in ("core25", "bio6", "matsci6", "all37"):
            assert artifact["cohorts"][label]["paired_delta"] < 0, label
        assert artifact["cohorts"]["matsci6"]["paired_delta"] < -2

    def test_the_evidence_is_reported_at_its_real_strength(self, artifact):
        """The bootstrap interval sits below zero; the sign test does not resolve. Both are
        recorded, because quoting only the interval would overstate a 13w/17l record and
        quoting only the sign test would hide a consistent negative mean."""
        a = artifact["cohorts"]["all37"]
        assert a["ci95"][1] < 0
        assert a["sign_p"] > 0.05
        assert artifact["verdict"]["ci_excludes_zero"] is True
        assert artifact["verdict"]["sign_test_resolves"] is False


class TestTheReasonIsNotTheNewPapers:
    def test_what_the_cut_adds_is_as_good_as_what_was_there(self, artifact):
        """0.882 against 0.878. If the added papers were junk this would be a simple story
        about a noisier pool; they are not, so it is not."""
        d = artifact["diagnostic"]
        assert d["added"]["precision"] >= d["kept"]["precision"]
        assert artifact["verdict"]["added_papers_are_good"] is True

    def test_the_digest_shrank_from_a_pool_six_times_larger(self, artifact):
        """The finding. 5.9× the candidates, 0.9 fewer papers shown per case — a gate reading
        a fixed depth of a diluted ranking admits less, not more."""
        d = artifact["diagnostic"]
        assert d["digest_size"]["treatment"] < d["digest_size"]["control"]
        assert artifact["pool_growth"]["factor"] > 5
        assert artifact["verdict"]["digest_shrank"] is True

    def test_most_of_the_loss_is_the_shrinkage(self, artifact):
        """−0.609 of −0.784 is showing fewer papers; −0.175 is showing worse ones. The split
        is exact by construction, which is why it can carry the follow-up's weight."""
        ls = artifact["diagnostic"]["loss_split"]
        assert ls["share_from_showing_fewer"] > 0.7
        assert ls["from_showing_fewer"] < ls["from_showing_worse"] < 0
        assert ls["sums_to_delta"] == pytest.approx(
            artifact["cohorts"]["all37"]["paired_delta"], abs=0.01
        )

    def test_the_follow_up_names_the_gate_not_the_retrieval(self, artifact):
        assert "gate_depth" in artifact["verdict"]["follow_up"]


class TestTheArmWasCleanEnoughToBelieve:
    def test_both_arms_pinned_the_same_hypotheses(self, artifact):
        """Without this the run answers nothing: NR-46 measured a redraw at +0.058 reach, a
        quarter of the effect size, so unpinned arms would confound the draw with the cut."""
        for name, arm in artifact["arms"].items():
            assert arm["pinned_hypotheses"] is True, name
        assert artifact["arms"]["control_top_k_100"]["hyde_top_k"] == 100
        assert artifact["arms"]["treat_top_k_1000"]["hyde_top_k"] == 1000

    def test_the_throttled_case_was_repaired_rather_than_dropped(self, artifact):
        """`bio-mdtraj`'s control pool fell back to keyword-only on an arXiv 429 — zero HyDE
        candidates. Paired against a treatment arm that had HyDE, its delta would have
        measured HyDE existing at all rather than the cut: the single largest confound
        available in a 37-case paired test. Re-collected at identical flags and spliced in,
        because 36 of 37 with an unexplained gap is worse than 37 with a stated repair."""
        r = artifact["repaired_case"]
        assert r["case"] == "bio-mdtraj"
        assert "429" in r["why"]
        assert "ZERO HyDE candidates" in r["why"]
        assert r["case"] in artifact["per_case"]

    def test_the_artifact_does_not_read_as_a_licence_to_ship(self, artifact):
        """A reader who greps this file for a number should meet the verdict, not just the
        reach story stage 1 told."""
        assert "FIRES" in artifact["_comment"]
        assert artifact["verdict"]["net2_fell"] is True
