"""P24: multi-source retrieval, measured against a matched control at last. [NR-41]

Every earlier multi-source figure in this project compared a run with a source enabled against
a run collected on a *different day*. P21's bio +4.00 was one of those. This arm holds
everything fixed but `sources`: two fresh frozen pools collected the same day, same 37 cases,
same window, same `w_embedding`, same HyDE, same gate.

The result is much smaller than the uncontrolled figure suggested, and the mechanism is not
the one the collision probes (P22/P23) implied:

* **core 25: +0.32, CI [-0.24, +0.88]** — and Europe PMC supplies **2 of 205 shown papers**.
  The channel is not poisoning software digests with biology; it is being rejected almost
  entirely, which is what the +0.00 for the one previously-measured channel also said.
* **scientific 12: +1.00, CI [-0.67, +2.50]**, with non-arXiv at 23% of the digest and 0.96
  precision. Real work, n=12, interval crossing zero.
* **P21's +4.00 does not survive the control.** Most of it was the collection, not the source.

`evals/results/` is gitignored, so these numbers are frozen here for the same reason
`gold_targets.json` and `bio_matched_arm.json` exist: a document citing a run file nothing in
the repository holds is citing nothing.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "evals" / "multisource_arm.json"


@pytest.fixture(scope="module")
def artifact() -> dict:
    return json.loads(FROZEN.read_text(encoding="utf-8"))


class TestTheArmsDifferInExactlyOneThing:
    def test_only_sources_changed(self, artifact):
        """The whole value of the arm. If any other axis moved, the delta is not attributable
        to the source and the number below means nothing."""
        c, t = artifact["arms"]["control_arxiv"], artifact["arms"]["treat_arxiv_epmc"]
        assert c["sources"] == ["arxiv"]
        assert t["sources"] == ["arxiv", "europepmc"]
        assert c["digest_window"] == t["digest_window"] == 15
        assert c["w_embedding"] == t["w_embedding"] == 1.5

    def test_the_pool_configs_differ_only_in_sources(self, artifact):
        """Now checkable at all because of the flag recording added for exactly this — before
        it, `rr_hyde` and `rr_all_time` appeared in no artifact and a reader had to trust that
        the two commands matched."""
        c = artifact["arms"]["control_arxiv"]["pool_config"]
        t = artifact["arms"]["treat_arxiv_epmc"]["pool_config"]
        assert c and t, "pool_config is what makes this assertion possible"
        differing = {k for k in c.keys() | t.keys() if c.get(k) != t.get(k)}
        assert differing == {"sources"}, differing

    def test_both_arms_cover_the_same_cases(self, artifact):
        c = set(artifact["arms"]["control_arxiv"]["per_case"])
        t = set(artifact["arms"]["treat_arxiv_epmc"]["per_case"])
        assert c == t and len(c) == 37


class TestTheControlIsTrustworthy:
    def test_it_reproduces_the_published_headline(self, artifact):
        """A fresh collection, fresh HyDE pass and fresh draw land +0.12 from a pool frozen
        twelve days earlier — far inside this project's own noise floor (Jaccard 0.49 on the
        ranked top-10, its largest variance term). That is what licenses reading the treatment
        delta as an effect rather than as a redraw."""
        r = artifact["control_reproduces_headline"]
        assert r["headline_mean"] == 5.72
        assert abs(r["delta"]) < 0.5, r


class TestWhatMultiSourceActuallyBuys:
    def test_core25_is_indistinguishable_from_zero(self, artifact):
        co = artifact["cohorts"]["core25"]
        assert co["paired_delta"] == 0.32
        assert (co["wins"], co["losses"]) == (8, 4)

    def test_and_the_reason_is_that_it_barely_appears(self, artifact):
        """**2 of 205 shown papers.** The collision probes measured 68% of Europe PMC's
        results as off-domain for these repositories; this is what that looks like downstream
        — not bad papers in the digest, but a channel the gate discards almost entirely."""
        co = artifact["cohorts"]["core25"]
        assert co["non_arxiv_shown"] == 2
        assert co["non_arxiv_share"] < 0.02

    def test_the_scientific_cohort_is_where_it_does_work(self, artifact):
        sci = artifact["cohorts"]["scientific12"]
        assert sci["paired_delta"] == 1.0
        assert sci["non_arxiv_share"] > 0.20

    def test_the_cohorts_are_reported_apart(self, artifact):
        """They answer different questions and the project's convention is not to average
        across them — the `all37` figure exists, but it is a blend of a null and an effect."""
        co, sci, allc = (artifact["cohorts"][k] for k in ("core25", "scientific12", "all37"))
        assert co["n_cases"] + sci["n_cases"] == allc["n_cases"]
        assert co["paired_delta"] < allc["paired_delta"] < sci["paired_delta"]


class TestTheGateHandlesTheCollision:
    def test_almost_every_non_arxiv_paper_that_is_shown_is_actionable(self, artifact):
        """29 of 30 across all 37 cases. This is the evidence that retired the "relevance
        filter" item: the collision never reaches the digest, so there is nothing for a filter
        to remove. It was proposed before checking, and the check refuted it."""
        allc = artifact["cohorts"]["all37"]
        assert allc["non_arxiv_actionable"] == 29
        assert allc["non_arxiv_shown"] == 30

    def test_no_case_shows_more_non_arxiv_papers_than_it_shows_papers(self, artifact):
        for label, arm in artifact["arms"].items():
            for case, row in arm["per_case"].items():
                assert 0 <= row["n_non_arxiv"] <= row["n"], f"{label}/{case}"
                assert row["n_non_arxiv_actionable"] <= row["n_non_arxiv"], f"{label}/{case}"

    def test_the_control_arm_shows_no_non_arxiv_papers_at_all(self, artifact):
        """`--sources arxiv` must contribute none. A non-zero count here would mean the arms
        were not what they claim and the delta measures something else."""
        ctrl = artifact["arms"]["control_arxiv"]["per_case"]
        assert sum(r["n_non_arxiv"] for r in ctrl.values()) == 0
