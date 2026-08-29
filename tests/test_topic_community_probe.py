"""Item 6: OpenAlex Topics do not order the gate-admitted band. [NR-44]

From "Topic Is Not Agenda" (arXiv:2605.07158), reduced to its cheapest testable form and run
for **$0** — no LLM, no judge. Pre-registered bar: **AUC ≥ 0.65** at the `judge==3` target on
testbed B, where the fine-scale incumbent scores **0.760** and the NR-21 metadata family 0.585.

| arm | AUC | 95% CI | |
|---|---|---|---|
| `repo_text` — OpenAlex reads the repo's description | **0.453** | [0.343, 0.571] | fails |
| `own_papers` — community from the case's non-band papers | **0.409** | [0.297, 0.529] | fails |

Both sit *below chance* at the point estimate and both intervals exclude the bar. **Two
independent failure modes, either one fatal:**

**The taxonomy is too coarse.** `own_papers_modal` is `subfield:Artificial Intelligence` for
six of nine scored cases. A band that is already topically homogeneous gets one constant topic,
and a constant cannot order anything. This is the arm that needed no classifier and it is the
weaker of the two.

**And the classifier cannot read software prose.** `diffusion` (the diffusers library)
classifies as *NMR spectroscopy and applications* in **Physics and Astronomy**; `cv`
(detectron2) as *Brain Tumor Detection and Classification* in **Neuroscience**; `crypto` and
`graph` both as *Computational Physics and Python Applications*. That is this project's own
register-mismatch finding (§5) reaching a new instrument: repository text describes what a
project *has*, in software vocabulary, and an academic classifier reads "diffusion" as
diffusion MRI. Three of eight correct.

**A note on what the testbed could and could not have shown.** PLANS describes a "602-paper
labelled set"; the band inside it is **108 papers, 41 positives**, three of twelve cases
contribute none, and `peft` (32) and `diffusion` (22) supply half. At that size the interval is
±0.11, so this testbed could *never* have separated a 0.65 pass from the 0.585 null — but it
resolves a miss this large without difficulty, and both intervals exclude the bar. The item is
closed on a clear negative, not on an ambiguous one.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "evals" / "topic_community_probe.json"
BAR = 0.65


@pytest.fixture(scope="module")
def artifact() -> dict:
    return json.loads(FROZEN.read_text(encoding="utf-8"))


class TestThePreRegisteredBarWasMissed:
    def test_neither_arm_reaches_it(self, artifact):
        for name, arm in artifact["arms"].items():
            assert arm["auc_judge3"] < BAR, name
            assert arm["passes_bar"] is False, name
        assert artifact["verdict"]["passes_bar"] is False

    def test_both_intervals_exclude_the_bar(self, artifact):
        """The claim that makes this a decision rather than a shrug. The bar is not merely
        unmet — it is outside both 95% intervals, so the miss is not a sampling accident."""
        for name, arm in artifact["arms"].items():
            assert arm["auc_judge3_ci95"][1] < BAR, name
        assert artifact["verdict"]["ci_contains_bar"] is False

    def test_it_does_not_even_beat_the_metadata_family(self, artifact):
        """NR-21's metadata features — age, citations, ranks — sit at 0.585, and were
        themselves recorded as too weak to use. This is below that, and below chance."""
        assert artifact["verdict"]["beats_metadata_family"] is False
        for arm in artifact["arms"].values():
            assert arm["auc_judge3"] < 0.5

    def test_chance_is_inside_the_interval_so_the_reading_is_null_not_inverted(self, artifact):
        """Both point estimates land under 0.5, which would be a *negative* signal if taken
        literally. The intervals contain 0.5, so the honest reading is no signal — not an
        anti-signal to be exploited by flipping the sign."""
        assert artifact["verdict"]["ci_contains_chance"] is True
        for arm in artifact["arms"].values():
            lo, hi = arm["auc_judge3_ci95"]
            assert lo <= 0.5 <= hi


class TestWhyItFails:
    def test_the_taxonomy_collapses_the_repositories_together(self, artifact):
        """The failure of the arm that needs no classifier, and the more fundamental of the
        two. OpenAlex's subfield is `Artificial Intelligence` for most scored cases; within a
        band already filtered to one topic, the signal is a constant."""
        modals = [
            v.get("own_papers_modal")
            for v in artifact["per_case"].values()
            if v.get("own_papers_modal")
        ]
        assert modals.count("subfield:Artificial Intelligence") >= 6
        assert len(set(modals)) <= 4, "nine cases, at most a handful of distinct communities"

    def test_the_classifier_misreads_software_prose(self, artifact):
        """The register mismatch reaching a new instrument. These are not near misses — they
        are different fields of science."""
        pc = artifact["per_case"]
        assert pc["diffusion"]["repo_text_field"] == "Physics and Astronomy"
        assert pc["cv"]["repo_text_field"] == "Neuroscience"
        assert "Computational Physics" in pc["crypto"]["repo_text_topic"]
        assert "Computational Physics" in pc["graph"]["repo_text_topic"]

    def test_a_case_with_no_community_is_excluded_not_scored_zero(self, artifact):
        """`systems` returns HTTP 500 from the classifier at every text length tried. It is
        dropped from the `repo_text` arm and counted, because a 0 would assert "wrong
        community" where all we have is "no answer" — and it still contributes to the arm that
        does not need the classifier."""
        assert "systems" in artifact["coverage"]["cases_without_repo_topic"]
        assert artifact["per_case"]["systems"]["scored_repo_text"] == 0
        assert artifact["per_case"]["systems"]["scored_own_papers"] > 0


class TestTheTestbedIsSmallerThanTheItemAssumed:
    def test_the_band_is_108_papers_not_602(self, artifact):
        """PLANS says "the 602-paper labelled set". 602 is the whole labelled set; the score-2
        band the probe can actually score is 108, and three cases contribute none at all."""
        t = artifact["testbed"]
        assert t["labelled_papers"] == 602
        assert t["band2_papers"] == 108
        assert sorted(t["cases_with_no_band"]) == ["cli", "http", "webdev"]

    def test_it_could_resolve_this_miss_even_though_it_could_not_resolve_a_pass(self, artifact):
        """Worth separating, because the two are different claims. An interval of about ±0.11
        cannot tell 0.65 from the 0.585 null — so a *pass* was never available here. It has no
        trouble at all telling 0.65 from 0.45, which is what actually happened."""
        assert artifact["verdict"]["ci_contains_bar"] is False
        widths = [
            a["auc_judge3_ci95"][1] - a["auc_judge3_ci95"][0] for a in artifact["arms"].values()
        ]
        assert max(widths) > 2 * (BAR - 0.585), "too wide to have confirmed a marginal pass"

    def test_coverage_is_reported_so_the_denominator_is_visible(self, artifact):
        """Papers OpenAlex does not hold are excluded rather than scored 0. The count travels
        with the result so nobody reads the AUC as covering the whole band."""
        c = artifact["coverage"]
        assert c["resolved"] > 90
        assert c["unresolved"] >= 0
        assert c["resolved"] + c["unresolved"] == artifact["testbed"]["band2_papers"]
