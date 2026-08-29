"""The dense index's recall gauge, frozen. [PLANS item 4]

The binary-quantized arXiv index sits under HyDE — the project's biggest converted change
(+1.36 net@2, its first p < 0.05) — and until now was verified for **identity** and unmeasured
for **usefulness**. `verify_encoder` proves our vectors reproduce the published ones bit for
bit; nothing proved the index still *finds* anything. Binarisation, column pruning, a bad
yearly shard or an encoder swap could have cost fifteen points of recall in silence.

LitSearch (arXiv:2407.18940) over the shipped index, shipped encoder, shipped search path:

| arm | R@5 | R@20 | R@100 | median rank when found |
|---|---|---|---|---|
| bare | 0.247 | 0.376 | **0.560** | 8 |
| prefixed | 0.259 | 0.396 | 0.530 | 5 |

**This is not a net@2 claim and must never be quoted as one.** A researcher asking "where can
I find work on X" is a different register from a repository that needs a paper to act on —
§5's register-mismatch finding is that the two do not transfer. It answers exactly one
question: *does the index still retrieve what it retrieved before?*

Two things worth reading twice. First, **456 of 456 gold arXiv papers are already in the
index**, so a shortfall here can only be retrieval, never coverage — that is what makes this a
fidelity gauge rather than a corpus measure. Second, the aggregate table above says the prefix
wins at the top, and **the paired test says that is noise**: McNemar puts @5 at p = 0.34 and
@20 at p = 0.12, while the one resolved difference (@100, p = 0.020) favours *bare*.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "evals" / "litsearch_recall.json"


@pytest.fixture(scope="module")
def artifact() -> dict:
    return json.loads(FROZEN.read_text(encoding="utf-8"))


class TestItMeasuresTheIndexAndNotTheCorpus:
    def test_every_gold_arxiv_paper_is_already_in_the_index(self, artifact):
        """The load-bearing fact. 458 of LitSearch's 574 gold papers carry an arXiv id and
        **456 of 456 distinct ones are in our shards** — so a query that fails, fails at
        retrieval. If this ever drops below 100%, the number stops being a fidelity gauge and
        starts being a coverage measure, and the two move for different reasons."""
        c = artifact["coverage"]
        assert c["gold_corpusids"] == 574
        assert c["gold_with_arxiv_id"] == 458
        assert c["gold_in_index"] == 456

    def test_unanswerable_queries_are_excluded_rather_than_scored(self, artifact):
        """99 of 597 have no gold paper the index could return. Scoring them as misses would
        measure LitSearch's overlap with arXiv, drift every time arXiv grows, and bury the
        signal underneath — void read as null, the failure this project keeps paying for."""
        c = artifact["coverage"]
        assert c["queries_total"] == 597
        assert c["queries_answerable"] == 498
        assert artifact["n_scored"] == c["queries_answerable"]
        assert len(artifact["per_query"]) == 498

    def test_the_encoder_was_verified_before_anything_was_embedded(self, artifact):
        """If our vectors stopped matching the index, every query would return confident
        nonsense and recall would collapse — correctly, but for a reason a recall number
        cannot name. The measurement refuses to run without Hamming 0, so an identity failure
        is reported as one instead of arriving disguised as a retrieval result."""
        assert artifact["encoder"] == "mixedbread-ai/mxbai-embed-large-v1"
        assert artifact["encoder_verified"] is True
        assert artifact["encoder_hamming"] == [0, 0, 0, 0, 0]


class TestTheFrozenLevel:
    def test_the_shipped_arm(self, artifact):
        bare = artifact["arms"]["bare"]
        assert bare["query_prefix"] is None
        assert bare["recall_at_5"] == 0.247
        assert bare["recall_at_20"] == 0.3755
        assert bare["recall_at_100"] == 0.5602
        assert bare["found_in_top_100"] == 279

    def test_recall_sits_where_a_gauge_can_see_movement(self, artifact):
        """The property that makes this worth running at all. A gauge pinned near 0 or near 1
        cannot report a regression — it has nowhere to fall from, or the fall is inside its
        own noise. Every k lands between 0.2 and 0.6."""
        for arm in artifact["arms"].values():
            for k in (5, 20, 100):
                assert 0.15 < arm[f"recall_at_{k}"] < 0.70, arm

    def test_recall_is_monotone_in_k(self, artifact):
        """Trivially true of a correct implementation and worth asserting anyway: it is the
        cheapest possible check that the rank bookkeeping is not scrambled."""
        for arm in artifact["arms"].values():
            assert arm["recall_at_5"] < arm["recall_at_20"] < arm["recall_at_100"]

    def test_a_miss_is_recorded_as_unknown_rather_than_as_rank_101(self, artifact):
        """The search is capped at 100, so a paper not found has an *unknown* rank, not a
        large one. It is stored as -1, every consumer treats it as a miss, and none of them
        averages it — averaging would silently convert "we did not look further" into "it
        ranked just outside", which is a different claim."""
        ranks = [r["rank_bare"] for r in artifact["per_query"]]
        assert -1 in ranks
        assert all(r == -1 or 0 <= r < 100 for r in ranks)
        found = [r for r in ranks if r >= 0]
        assert len(found) == artifact["arms"]["bare"]["found_in_top_100"]


class TestTheQueryPrefixWasDecidedByTheProperTest:
    """`mxbai-embed-large-v1` is asymmetric — documents bare, queries behind "Represent this
    sentence for searching relevant passages: ". The index holds bare abstracts, so the prefix
    belongs on the query side or nowhere, and which one is a measurement.

    The aggregate table says the prefix wins at the top: +0.012 at k=5, +0.020 at k=20. Both
    vanish under a paired test. The only difference that resolves is at k=100 and it favours
    bare. **Two aggregate rates over the same 498 questions are not a comparison** — that is
    the whole reason the arms are run together and their per-query ranks stored.
    """

    def test_the_apparent_wins_at_the_top_are_not_significant(self, artifact):
        for k in (5, 20):
            cell = artifact["prefix_comparison"][f"at_{k}"]
            assert cell["only_prefixed"] > cell["only_bare"], f"@{k}: the prefix does lead"
            assert not cell["significant_at_05"], f"@{k}: but not resolvably"

    def test_the_one_resolved_difference_favours_the_shipped_form(self, artifact):
        cell = artifact["prefix_comparison"]["at_100"]
        assert cell["only_bare"] == 26
        assert cell["only_prefixed"] == 11
        assert cell["p_value"] == 0.0201
        assert cell["significant_at_05"]

    def test_the_gauge_is_frozen_on_the_form_the_product_actually_uses(self, artifact):
        """HyDE embeds hypothetical abstracts with no prefix, so `bare` is the arm that
        describes the shipped path. It is also the arm the only resolved comparison favours,
        which is a convenient agreement rather than a reason."""
        assert artifact["arms"]["bare"]["query_prefix"] is None
        assert artifact["arms"]["prefixed"]["query_prefix"].startswith("Represent this sentence")


class TestItSaysWhatItIsNot:
    def test_the_artifact_disclaims_the_net2_reading_in_its_own_text(self, artifact):
        """A number this easy to quote out of context should carry its own scope. The comment
        travels with the file, so a reader who finds `recall_at_20` in a grep has the caveat
        one line away rather than in a document they have not opened."""
        c = artifact["_comment"]
        assert "REGRESSION GAUGE" in c
        assert "Not a net@2 claim" in c
        assert "register" in c
