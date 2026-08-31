"""PRF-HyDE does not move net@2. Item 12 closes, and the last lead with it. [NR-51]

The arm NR-50 licensed, ~$15. Control is the shipped arm; treatment is identical in every
flag except a pinned hypothesis file carrying **round 1 ∪ round 2** — 8 abstracts per case,
searched through the shipped `hyde.discover` at `top_k` 100, so the pool comes from the
product's own code and the fingerprint is honest about being a different pool.

| arm | net@2 | digest/case | precision |
|---|---|---|---|
| **control (ships)** | **+5.51** | 8.3 | 0.889 |
| treatment (PRF) | +5.32 | 8.5 | 0.876 |

```
primary,  all 37   -0.19   CI [-0.84, +0.43]   9w/8l/20t   p = 1.0000
secondary, 33      -0.21   CI [-0.97, +0.52]   9w/8l/16t   p = 1.0000
```

**The pre-registered kill fires: |−0.19| < 0.78.** A wash against what ships, for ~250 extra
candidates per case.

**The free prior had the sign wrong, and that is the useful half.** NR-50 read the judge cache
over *window* papers, got Δp = **+0.054** favouring round 2, and flagged it as weak — 61% of
entering and 73% of displaced papers were void, and the judged subset is selected by having
been shown. Measured in the digest: **85 added at 0.882 displacing 77 at 0.935, Δp = −0.053.**
Same magnitude, opposite sign. Reporting it as a hint rather than a measurement was right; the
number itself was not evidence about the direction at all.

**The four no-round-2 cases are exact ties with byte-identical picks** — the arm's own proof
that the two sides differ in one thing only, and that nothing was fabricated where PRF has
nothing to feed on.

**Two cases were repaired**, for NR-47's reason: `compiler` and `numerics` lost HyDE entirely
to an arXiv 429 and 503, collecting keyword-only pools (519 → 211, 589 → 202). Paired against
a control that *has* HyDE they would have measured "HyDE existing at all" rather than "round 2
added". Re-collected at identical flags (519 → 843, 589 → 957).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "evals" / "prf_arm.json"


@pytest.fixture(scope="module")
def artifact() -> dict:
    return json.loads(FROZEN.read_text(encoding="utf-8"))


class TestTheKillFired:
    def test_the_arm_is_a_wash(self, artifact):
        p = artifact["primary_all_37"]
        assert p["n_cases"] == 37
        assert p["paired_delta"] == -0.19
        assert p["ci95"][0] < 0 < p["ci95"][1]
        assert p["sign_p"] > 0.05

    def test_below_the_bar_registered_before_the_run(self, artifact):
        pre = artifact["pre_registered"]
        assert pre["bar"] == 0.78
        assert pre["written_before_the_run"] is True
        assert abs(artifact["primary_all_37"]["paired_delta"]) < pre["bar"]
        assert artifact["verdict"]["killed"] is True
        assert artifact["verdict"]["resolves"] is False

    def test_the_secondary_agrees_and_was_declared_in_advance(self, artifact):
        """The 33 cases where PRF actually acts. Declared beforehand precisely so it could not
        be produced afterwards as a rescue — and it does not rescue anything: −0.21."""
        s = artifact["secondary_33_with_round2"]
        assert s["n_cases"] == 33
        assert s["paired_delta"] < 0
        assert s["ci95"][0] < 0 < s["ci95"][1]

    def test_the_expected_outcome_was_recorded_first(self, artifact):
        """Every refinement pointed at a null before the arm ran, and that was written down so
        the result could not be re-read as a surprise or as a vindication."""
        why = artifact["pre_registered"]["expected_outcome_was_null"]
        assert "+0.28" in why and "+0.23" in why


class TestThePriorHadTheSignWrong:
    def test_what_prf_adds_is_worse_than_what_it_displaces(self, artifact):
        d = artifact["digest_churn"]
        assert d["added"]["n"] == 85
        assert d["dropped"]["n"] == 77
        assert d["added"]["precision"] < d["dropped"]["precision"]
        assert d["observed_dp"] == -0.053

    def test_it_inverts_nr50s_free_estimate(self, artifact):
        """+0.054 predicted, −0.053 measured. Near-identical magnitude, opposite sign. The
        judge-cache prior was reported as a hint with 61%/73% void and a selected sample; this
        is what that caveat was worth, and the artifact says so rather than quietly dropping
        the earlier number."""
        d = artifact["digest_churn"]
        assert d["nr50_prior_dp"] == 0.054
        assert d["prior_had_the_sign_wrong"] is True
        assert d["observed_dp"] * d["nr50_prior_dp"] < 0

    def test_the_digest_grew_while_the_score_fell(self, artifact):
        """8.3 → 8.5 papers per case at 0.889 → 0.876 precision. PRF shows slightly more and
        slightly worse, which is the whole −0.19 in one line."""
        a = artifact["arms"]
        assert a["treatment"]["digest_per_case"] > a["control"]["digest_per_case"]
        assert a["treatment"]["precision"] < a["control"]["precision"]
        assert a["treatment"]["mean_net2"] < a["control"]["mean_net2"]


class TestTheArmWasCleanEnoughToBelieve:
    def test_the_no_round2_cases_are_exact_ties(self, artifact):
        """The arm's own validation. Where the shipped run showed nothing, there is nothing to
        feed on, so the treatment IS the control — byte for byte. Had these differed, the two
        arms would have been differing in something other than the second round."""
        n = artifact["no_round2_cases"]
        assert sorted(n["cases"]) == ["cli", "http", "linter", "webdev"]
        assert n["all_exact_ties"] is True

    def test_they_were_never_padded_with_a_fresh_draw(self, artifact):
        """Filling them would have fabricated a treatment where PRF does nothing, and NR-46
        measured a fresh draw at +0.0577 reach — larger than anything under test."""
        assert "fabricated a treatment" in artifact["no_round2_cases"]["_comment"]

    def test_the_throttled_cases_were_repaired_not_dropped(self, artifact):
        """NR-47's confound, twice. A keyword-only pool paired against a control with HyDE
        measures HyDE's existence, not the treatment."""
        r = artifact["repaired_cases"]
        for case in ("compiler", "numerics"):
            assert r[case]["degraded_pool"] < r[case]["shipped_pool"]
            assert r[case]["repaired_pool"] > r[case]["shipped_pool"]
        assert "429" in r["_comment"] and "503" in r["_comment"]

    def test_no_cohort_rescues_it(self, artifact):
        """core25 −0.48 is the one that carries weight; bio and matsci move positively on six
        cases each, which C-7 says is not a finding."""
        c = artifact["cohorts"]
        assert c["core25"] < 0
        assert set(c) == {"core25", "bio6", "matsci6"}


class TestTheDirectionIsClosed:
    def test_the_artifact_reads_as_a_close(self, artifact):
        assert artifact["verdict"]["direction_closed"] is True
        assert "KILL" in artifact["_comment"] or "NULL" in artifact["_comment"]

    def test_it_names_the_three_measured_levers(self, artifact):
        """Retrieval width (NR-47), gate depth (NR-48), iterative retrieval (NR-51). The
        generalisation worth keeping is that all three were measured and none pays."""
        note = artifact["verdict"]["closing_note"]
        assert "NR-47" in note and "NR-48" in note
        assert "iterative retrieval" in note
