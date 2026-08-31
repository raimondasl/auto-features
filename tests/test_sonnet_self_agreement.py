"""Sonnet agrees with itself at 0.80 and with GPT at 0.199. NR-52 stands. [NR-53]

After NR-52 shipped, a code read found that `_call_claude` sends **no temperature**, so the
Anthropic default (1.0) applies and every Sonnet verdict is a *sample*, while the GPT judge
runs greedy at `temperature=0`. The worry was that NR-52's Sonnet-only sign flip might be one
judge disagreeing with itself rather than with GPT. This measures that, and the worry is wrong.

| statistic | value | |
|---|---|---|
| exact score agreement | 0.806 | |
| **kappa, binary at ≥2** | **0.7979** | GPT-vs-Sonnet on the band: **0.199** |
| kappa, quadratic 0–3 | 0.8197 | |
| sonnet-only label flips (≥2) | **8.4%** | CI [4.5%, 12.3%], bar was ≤10% |
| consensus label flips (≥1) | 1.6% | CI [0.0%, 3.3%] |

**Pre-registered PASS: kappa ≥ 0.6 and flip ≤ 10%.** Both clear.

**Sonnet's self-agreement is 4× its agreement with GPT.** That gap is the finding: whatever
separates the two judges is a property of the judges, not of the sampler. NR-52's conclusion —
that the comparator margin is judge-dependent in *direction* — survives, and the temperature
omission is hygiene worth fixing rather than a correction owed.

**Magnitude still carries label noise, and it is small.** An 8.4% flip rate over ~9 shown papers
per case injects ~2.49 net@2 per case, an SE of 0.41 on a 37-case margin. Added in quadrature to
the paired bootstrap's ~1.92, that widens the Sonnet-only interval by about **2%** — between-case
variation dominates, as it did before. The sign is robust to resampling the judge; the width was
never limited by it.

**Where the instability lives is not where it looks.** Movement by original score runs 19.6% at
score 1, 14.3% at score 2 and **32.4% at score 3** — the most confident band is the least stable
in raw score. But a 3→2 move crosses no threshold, so it costs nothing; the 8.4% is driven by
1↔2 crossings. Raw agreement and decision agreement are different quantities, which is why the
probe pre-registered the flip rate rather than kappa.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "evals" / "sonnet_self_agreement.json"


@pytest.fixture(scope="module")
def artifact() -> dict:
    return json.loads(FROZEN.read_text(encoding="utf-8"))


class TestTheProbeRanAtTheSettingsUnderTest:
    def test_it_measured_the_sampled_configuration(self, artifact):
        """The whole point: this had to run BEFORE the temperature fix. At temperature 0
        self-agreement is trivially ~1.0 and the quantity is unrecoverable."""
        assert "default 1.0" in artifact["temperature"]
        assert artifact["model"] == "claude-sonnet-5"

    def test_the_bars_were_written_before_any_replicate(self, artifact):
        pre = artifact["pre_registered"]
        assert pre["written_before_any_replicate"] is True
        assert pre["pass"] == {"kappa_at_least": 0.6, "flip_rate_at_most": 0.1}
        assert pre["kill"] == {"kappa_below": 0.4, "flip_rate_at_least": 0.2}

    def test_the_statistic_was_chosen_for_the_decision_not_the_literature(self, artifact):
        """Kappa is what gets quoted; a threshold flip is what moves a margin. Registering the
        flip rate in advance is what stops the friendlier of the two being picked after."""
        why = artifact["pre_registered"]["statistic_is_flip_rate_not_kappa"]
        assert "propagates into a margin" in why


class TestSelfAgreementIsHigh:
    def test_kappa_clears_the_bar(self, artifact):
        assert artifact["kappa_binary_at_2"] == pytest.approx(0.7979, abs=0.001)
        assert artifact["kappa_binary_at_2"] >= artifact["pre_registered"]["pass"]["kappa_at_least"]

    def test_the_flip_rate_clears_the_bar(self, artifact):
        f = artifact["label_flips"]["sonnet_only_ge2"]
        assert f["flips"] == 16
        assert f["n"] == 191
        assert f["rate"] == pytest.approx(0.0838, abs=0.001)
        assert f["rate"] <= artifact["pre_registered"]["pass"]["flip_rate_at_most"]

    def test_the_verdict_is_pass(self, artifact):
        v = artifact["verdict"]
        assert v["passes"] is True
        assert v["kills"] is False
        assert v["grey"] is False


class TestTheCrossJudgeGapIsRealNotSampling:
    def test_self_agreement_far_exceeds_cross_judge_agreement(self, artifact):
        """0.798 against 0.199 — a 4x gap. Whatever separates the judges is a property of the
        judges, so NR-52's judge-dependence conclusion is not a temperature artifact."""
        ref = artifact["reference_points"]["gpt_sonnet_kappa_on_band"]
        assert ref == 0.199
        assert artifact["kappa_binary_at_2"] > 3.5 * ref

    def test_the_consensus_label_is_the_stabler_one(self, artifact):
        """1.6% against 8.4%. `Sonnet >= 1` sits away from where the judge wavers — which is
        also exactly why NR-52 measured it as barely binding."""
        assert (
            artifact["label_flips"]["consensus_ge1"]["rate"]
            < artifact["label_flips"]["sonnet_only_ge2"]["rate"]
        )

    def test_the_implied_perturbation_is_recorded(self, artifact):
        """~2.49 net@2 per case, an SE of 0.41 at n=37 — about 2% added in quadrature to the
        bootstrap's ~1.92. Reported because the arithmetic was fixed in advance, and because a
        reader should see that the margin's width was never limited by judge sampling."""
        f = artifact["label_flips"]["sonnet_only_ge2"]
        assert f["implied_margin_sd_per_case"] == pytest.approx(2.49, abs=0.05)
        assert artifact["reference_points"]["sonnet_only_margin_nr52"] == -3.41


class TestRawStabilityIsNotDecisionStability:
    def test_the_most_confident_band_moves_most(self, artifact):
        """32.4% of score-3 papers changed score, against 14.3% of score-2 and 19.6% of
        score-1. Counterintuitive, and harmless: a 3->2 move crosses no threshold."""
        m = artifact["movement_by_original_score"]
        assert m["3"]["rate"] > m["2"]["rate"]
        assert m["3"]["rate"] > 0.3

    def test_and_yet_the_label_flip_rate_stays_low(self, artifact):
        """The reason the two diverge: 19.4% of raw scores moved, but only 8.4% of them crossed
        the >=2 line. Quoting raw agreement alone would have overstated the damage."""
        assert artifact["exact_score_agreement"] < 0.85
        assert artifact["label_flips"]["sonnet_only_ge2"]["rate"] < 0.10
