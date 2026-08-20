"""`rr why` must describe the decision the digest actually made, and no other.

The command exists because a zero in `net@2` is ambiguous: on the 2026-08-20 re-baseline four
of 25 benchmark repositories returned nothing, and every one had actionable papers in its pool
— ruff's had nine. A correct abstention and a silent discard of nine good papers score
identically, so the metric could not distinguish them.

The risk in a command like this is that it becomes a second, drifting implementation of the
tier rules — the C-9/C-12/C-14 shape. `explain_paper` therefore calls `digest_window` and
`categorize_papers` themselves. `TestTheExplanationMatchesTheDigest` is the guard that says so,
and it is not decorative: the first version of `_verdict` reasoned from the gate score directly
and reported "so it is muted" under a trace whose own DIGEST line said `not shown`. With
`triage.rerank` on, a low gate score does not mute a paper — it demotes it out of the window
entirely, which is a different answer to a user asking where their paper went.
"""

from __future__ import annotations

import pytest

from reporadar.config import FinescaleConfig, OutputConfig, RepoRadarConfig, TriageConfig
from reporadar.explain import explain_paper


def _cfg(
    *,
    triage: bool = True,
    finescale: bool = False,
    top_n: int = 15,
    top_k: int = 50,
    rerank: bool = True,
    min_actionable: int = 2,
) -> RepoRadarConfig:
    cfg = RepoRadarConfig()
    cfg.output = OutputConfig(top_n=top_n)
    cfg.triage = TriageConfig(
        enabled=triage,
        top_k=top_k,
        min_actionable=min_actionable,
        rerank=rerank,
        finescale=FinescaleConfig(enabled=finescale),
    )
    return cfg


def _paper(arxiv_id: str, score: float, **extra) -> dict:
    base = {
        "arxiv_id": arxiv_id,
        "title": f"Paper {arxiv_id}",
        "abstract": "abs",
        "categories": ["cs.LG"],
        "published": "2024-01-01T00:00:00+00:00",
        "score_total": score,
        "keyword_score": score,
        "matched_query": "all:transformers",
    }
    base.update(extra)
    return base


def _pool(*papers: dict) -> list[dict]:
    return sorted(papers, key=lambda p: -p["score_total"])


def _step(ex, name: str):
    return next(s for s in ex.steps if s.name == name)


class TestWhereItStopped:
    def test_a_paper_that_was_never_collected(self) -> None:
        ex = explain_paper(
            "2401.99999", _pool(_paper("2401.00001", 0.9)), _cfg(), known_to_store=False
        )
        assert not ex.found
        assert ex.stopped_at == "collected"
        assert "never saw this paper" in ex.verdict
        # The remedy has to name the thing that would actually change it.
        assert "arxiv.categories" in ex.remedy
        assert "europepmc" in ex.remedy

    def test_a_top_pick_says_so(self) -> None:
        ex = explain_paper(
            "2401.00001",
            _pool(_paper("2401.00001", 0.9, llm_score=3), _paper("2401.00002", 0.5, llm_score=3)),
            _cfg(),
        )
        assert ex.stopped_at == ""
        assert "Top Pick" in ex.verdict
        assert _step(ex, "digest").detail == "Top Picks"

    def test_the_gate_never_reached_it(self) -> None:
        """The gate reads the top `triage.top_k`; a paper below that is unproven, not rejected."""
        pool = _pool(
            *[_paper(f"2401.{i:05d}", 1.0 - i / 100, llm_score=3) for i in range(1, 4)],
            _paper("2401.00099", 0.1),
        )
        ex = explain_paper("2401.00099", pool, _cfg(top_k=3, top_n=15))
        assert ex.stopped_at in {"gate", "window"}
        assert "triage.top_k" in ex.remedy or "output.top_n" in ex.remedy

    def test_the_band_paper_that_failed_its_probability(self) -> None:
        """A score-2 paper must also clear P >= 2/3, and the message must say the number."""
        ex = explain_paper(
            "2401.00001",
            _pool(_paper("2401.00001", 0.9, llm_score=2, finescale_p=0.41)),
            _cfg(finescale=True),
        )
        assert ex.stopped_at == "finescale"
        assert "0.410" in ex.verdict
        assert "3p" in ex.remedy  # the threshold is derived, not tuned
        assert _step(ex, "digest").detail == "Maybe relevant"

    def test_the_band_paper_that_cleared_it(self) -> None:
        ex = explain_paper(
            "2401.00001",
            _pool(_paper("2401.00001", 0.9, llm_score=2, finescale_p=0.88)),
            _cfg(finescale=True),
        )
        assert ex.stopped_at == ""
        assert _step(ex, "finescale").outcome == "pass"

    def test_a_band_paper_with_no_probability_is_unproven(self) -> None:
        ex = explain_paper(
            "2401.00001", _pool(_paper("2401.00001", 0.9, llm_score=2)), _cfg(finescale=True)
        )
        assert _step(ex, "finescale").outcome == "stop"
        assert "no probability" in ex.verdict

    def test_the_repositorys_own_paper_is_muted(self) -> None:
        ex = explain_paper(
            "2401.00001",
            _pool(_paper("2401.00001", 0.9, llm_score=3)),
            _cfg(),
            cited_ids=frozenset({"2401.00001"}),
        )
        assert ex.stopped_at == "muted"
        assert "already cites it" in ex.verdict

    def test_a_withdrawn_paper_is_muted(self) -> None:
        ex = explain_paper(
            "2401.00001",
            _pool(_paper("2401.00001", 0.9, llm_score=3, withdrawn_in="v2")),
            _cfg(),
        )
        assert ex.stopped_at == "muted"
        assert "withdrawn" in ex.verdict

    def test_with_the_gate_off_the_heuristic_threshold_is_the_rule(self) -> None:
        ex = explain_paper("2401.00001", _pool(_paper("2401.00001", 0.2)), _cfg(triage=False))
        assert _step(ex, "gate").outcome == "skip"
        assert "gate off" in ex.verdict or "gate is off" in ex.verdict or "-8.12" in ex.remedy


class TestTheExplanationMatchesTheDigest:
    """The guard. An explanation that contradicts the digest is worse than none."""

    @pytest.mark.parametrize("llm", [0, 1, 2, 3])
    @pytest.mark.parametrize("rerank", [True, False])
    def test_the_verdict_never_contradicts_the_tier(self, llm: int, rerank: bool) -> None:
        """The bug this test was written for.

        `_verdict` used to read the gate score and announce a tier from it. With
        `triage.rerank` on, a gate score below the threshold demotes a paper *out of the
        window*, so the digest shows it nowhere — while the old code said "so it is muted".
        The tier from `categorize_papers` is ground truth; the scores only explain it.
        """
        pool = _pool(
            _paper("2401.00001", 0.9, llm_score=llm),
            *[_paper(f"2401.{i:05d}", 0.8 - i / 100, llm_score=3) for i in range(2, 20)],
        )
        ex = explain_paper("2401.00001", pool, _cfg(top_n=15, rerank=rerank))
        tier = _step(ex, "digest").detail
        v = ex.verdict.lower()
        if tier == "not shown":
            assert "not shown" in v or "window" in v, (tier, ex.verdict)
            assert "it is muted" not in v, "claimed a tier the digest did not give it"
        elif tier == "Top Picks":
            assert "top pick" in v
        elif tier == "Muted":
            assert "muted" in v
        elif tier == "Maybe relevant":
            assert "maybe" in v

    def test_every_stage_is_reported_once_and_in_order(self) -> None:
        ex = explain_paper(
            "2401.00001", _pool(_paper("2401.00001", 0.9, llm_score=3)), _cfg(finescale=True)
        )
        names = [s.name for s in ex.steps]
        assert names == ["collected", "ranked", "gate", "finescale", "window", "digest"]

    def test_a_versioned_id_finds_the_paper(self) -> None:
        """Ids arrive with and without versions; `dedup_id` is the shared rule."""
        ex = explain_paper("2401.00001v3", _pool(_paper("2401.00001", 0.9, llm_score=3)), _cfg())
        assert ex.found


class TestItStaysReadOnly:
    def test_the_module_imports_nothing_that_can_spend_money(self) -> None:
        """`rr why` is the one command a user runs when confused; it must never cost.

        Asserted on the import list rather than by mocking, because the failure mode is a
        future edit adding a live lookup for a paper that is not in the store — which would
        be the natural thing to write and exactly the wrong thing.
        """
        from pathlib import Path

        source = Path("src/reporadar/explain.py").read_text(encoding="utf-8")
        for banned in ("llm_client", "requests", "urllib", "arxiv", "openai", "anthropic"):
            assert f"import {banned}" not in source, banned
        assert "def save" not in source and ".save_" not in source
