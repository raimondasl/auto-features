"""The fine-scale stage must only pay for papers the digest can still show.

`cli.update` used to build the rescore band over **every** triaged paper, while
`digest.categorize_papers` drops withdrawn papers and then cuts to `output.top_n`. Anything
outside that window reaches no tier at all, so the product was buying gpt-4o-mini calls for
papers it would never display — cheap when `triage.top_k` was 15, and tripled on 2026-08-14
when the measured depth experiment moved the default to 50.

The fix is a *shared* window, not a second one. Re-deriving the ordering inside `cli.update`
would mean two implementations of a rule whose subtlety is easy to miss — withdrawn papers
leave **before** the cut, so each one pulls the next paper up into the window — and the two
copies would disagree only on runs where a retraction landed in the top slots, the rarest
and least testable case. C-9, C-12 and C-14 are all that shape. These tests pin both halves:
the window's own behaviour, and that the two callers genuinely share it.
"""

from __future__ import annotations

import shutil
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from reporadar.cli import cli
from reporadar.digest import categorize_papers, digest_window
from reporadar.store import PaperStore

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def _rec(arxiv_id: str, score: float, llm: int | None = None, withdrawn: str | None = None):
    return {
        "arxiv_id": arxiv_id,
        "score_total": score,
        "llm_score": llm,
        "withdrawn_in": withdrawn,
    }


class TestDigestWindow:
    def test_it_cuts_at_top_n(self) -> None:
        recs = [_rec(f"p{i}", 1.0 - i / 10) for i in range(5)]
        window, withdrawn = digest_window(recs, top_n=2)
        assert [r["arxiv_id"] for r in window] == ["p0", "p1"]
        assert withdrawn == []

    def test_a_withdrawn_paper_pulls_the_next_one_into_the_window(self) -> None:
        """The rule a re-implementation would get wrong.

        Withdrawn papers are removed *before* the cut, so the slot one would have taken
        goes to the next paper rather than being wasted. Filter after the cut instead and
        the window silently shrinks by one per withdrawal.
        """
        recs = [
            _rec("keep1", 0.9),
            _rec("gone", 0.8, withdrawn="cs.CL"),
            _rec("promoted", 0.7),
            _rec("out", 0.6),
        ]
        window, withdrawn = digest_window(recs, top_n=2)
        assert [r["arxiv_id"] for r in window] == ["keep1", "promoted"]
        assert [r["arxiv_id"] for r in withdrawn] == ["gone"]

    def test_rerank_reorders_before_the_cut(self) -> None:
        recs = [_rec("low_llm", 0.9, llm=0), _rec("buried", 0.1, llm=3)]
        plain, _ = digest_window(recs, top_n=1, triage_threshold=2, rerank=False)
        assert [r["arxiv_id"] for r in plain] == ["low_llm"]
        reranked, _ = digest_window(recs, top_n=1, triage_threshold=2, rerank=True)
        assert [r["arxiv_id"] for r in reranked] == ["buried"]

    def test_rerank_without_a_triage_threshold_is_a_noop(self) -> None:
        recs = [_rec("a", 0.9, llm=0), _rec("b", 0.1, llm=3)]
        window, _ = digest_window(recs, top_n=1, triage_threshold=None, rerank=True)
        assert [r["arxiv_id"] for r in window] == ["a"]

    def test_it_does_not_mutate_its_input(self) -> None:
        recs = [_rec("a", 0.1, llm=0), _rec("b", 0.9, llm=3)]
        before = [r["arxiv_id"] for r in recs]
        digest_window(recs, top_n=2, triage_threshold=2, rerank=True)
        assert [r["arxiv_id"] for r in recs] == before

    def test_the_window_is_exactly_what_reaches_a_tier(self) -> None:
        """The claim `cli.update` relies on: outside the window, nothing is displayable.

        If this ever stops holding, scoping the rescore band by the window starts hiding
        papers instead of just saving money.
        """
        recs = [
            _rec("a", 0.9, llm=3),
            _rec("w", 0.85, llm=3, withdrawn="cs.CL"),
            _rec("b", 0.8, llm=2),
            _rec("c", 0.7, llm=0),
            _rec("d", 0.6, llm=3),
        ]
        window, _ = digest_window(recs, top_n=3, triage_threshold=2, rerank=True)
        top, maybe, muted = categorize_papers(recs, top_n=3, triage_threshold=2, rerank=True)
        tiered = {p["arxiv_id"] for p in top + maybe + muted}
        assert {r["arxiv_id"] for r in window} | {"w"} == tiered


def _repo_with_gate(tmp_path: Path, *, top_n: int, top_k: int) -> Path:
    """A repo whose digest window is deliberately narrower than its gate."""
    shutil.copy(FIXTURES_DIR / "sample_readme.md", tmp_path / "README.md")
    cfg = tmp_path / ".reporadar.yml"
    cfg.write_text(
        f"repo_path: {tmp_path}\n"
        "arxiv:\n  categories: [cs.CL]\n  max_results_per_query: 10\n  lookback_days: 14\n"
        "queries:\n  seed: []\n  exclude: []\n"
        "ranking:\n  w_keyword: 1.0\n  w_category: 0.5\n  w_recency: 0.0\n"
        "enrichment:\n  provider: 'off'\n"
        "signals:\n  integrity: false\n"
        "suggestions:\n  provider: claude\n  claude_api_key: test-key\n"
        f"triage:\n  enabled: true\n  top_k: {top_k}\n  min_actionable: 2\n  rerank: true\n"
        "  finescale:\n    enabled: true\n    openai_api_key: test-key\n"
        f"output:\n  digest_path: {tmp_path / 'digest.md'}\n  top_n: {top_n}\n",
        encoding="utf-8",
    )
    return tmp_path


def _papers(n: int) -> list[dict]:
    now = datetime.now(UTC).isoformat()
    return [
        {
            "arxiv_id": f"2401.0000{i}v1",
            "title": f"Retrieval Paper {i}",
            "authors": ["A"],
            "abstract": "retrieval augmented generation for language models",
            "categories": ["cs.CL"],
            "published": now,
            "updated": None,
            "url": f"http://arxiv.org/abs/2401.0000{i}v1",
            "pdf_url": None,
            "matched_query": "all:test",
        }
        for i in range(n)
    ]


class TestUpdateScopesTheBandToTheWindow:
    """End to end through `rr update`, because the defect was in the wiring, not the rule."""

    @patch("reporadar.finescale.score_papers")
    @patch("reporadar.triage.triage_papers")
    @patch("reporadar.pipeline.collect_papers")
    def test_a_band_paper_outside_the_window_is_not_rescored(
        self,
        mock_collect: MagicMock,
        mock_triage: MagicMock,
        mock_score: MagicMock,
        tmp_path: Path,
    ) -> None:
        repo = _repo_with_gate(tmp_path, top_n=2, top_k=5)
        papers = _papers(5)
        mock_collect.return_value = papers
        # Every paper sits exactly at the gate threshold, so the whole triaged set is
        # band. Only the two the digest can show should cost anything.
        mock_triage.return_value = {
            p["arxiv_id"]: {"llm_score": 2, "llm_reason": "r"} for p in papers
        }
        mock_score.return_value = {}

        result = CliRunner().invoke(cli, ["update", "--config", str(repo / ".reporadar.yml")])

        assert result.exit_code == 0, result.output
        assert mock_score.called, "the fine-scale stage did not run at all"
        scored_ids = [p["arxiv_id"] for p in mock_score.call_args.args[0]]
        assert len(scored_ids) == 2, f"rescored {len(scored_ids)} papers for a 2-paper digest"
        assert "Rescoring 2 band papers" in result.output

    @patch("reporadar.finescale.score_papers")
    @patch("reporadar.triage.triage_papers")
    @patch("reporadar.pipeline.collect_papers")
    def test_the_scoped_band_matches_what_the_digest_would_show(
        self,
        mock_collect: MagicMock,
        mock_triage: MagicMock,
        mock_score: MagicMock,
        tmp_path: Path,
    ) -> None:
        """A withdrawn paper in the window must move both callers by the same one slot.

        This is the cross-caller assertion: the ids `cli.update` chose to pay for are
        exactly the ids `categorize_papers` later tiers, withdrawal displacement included.
        """
        repo = _repo_with_gate(tmp_path, top_n=2, top_k=5)
        papers = _papers(5)
        mock_collect.return_value = papers
        mock_triage.return_value = {
            p["arxiv_id"]: {"llm_score": 2, "llm_reason": "r"} for p in papers
        }
        mock_score.return_value = {}

        db = repo / ".reporadar" / "papers.db"
        db.parent.mkdir(parents=True, exist_ok=True)
        with PaperStore(db) as store:
            store.upsert_papers(papers)
            # Flag the paper the ranker will place first, so the window has to reach one
            # deeper than it otherwise would.
            store.save_signals([(papers[0]["arxiv_id"], "withdrawn", "cs.CL", None)])

        result = CliRunner().invoke(cli, ["update", "--config", str(repo / ".reporadar.yml")])
        assert result.exit_code == 0, result.output

        scoped = {p["arxiv_id"] for p in mock_score.call_args.args[0]}
        with PaperStore(db) as store:
            run_id = store.get_last_run()["run_id"]
            scored = store.get_scores_for_run(run_id)
        top, maybe, _ = categorize_papers(scored, top_n=2, triage_threshold=2, rerank=True)
        assert scoped == {p["arxiv_id"] for p in top + maybe}
        assert papers[0]["arxiv_id"] not in scoped, "the withdrawn paper was paid for"


class TestTheKnownResidual:
    """`rr digest --since` can promote a paper past the window `rr update` paid for.

    `rr update` has no `--since`, so it scopes the band to the unfiltered top-`top_n`. A
    later `rr digest --since 7` removes papers from that window, and removal promotes
    whatever sat below it. A band paper promoted that way has no `finescale_p` and
    therefore reaches Maybe rather than Top Picks — where before this change it would have
    carried a score.

    Pinned rather than left to be discovered. The direction is conservative and it is the
    same rule C-13 established for ungated papers (unproven is not endorsed), but it is a
    real reduction in Top Picks for since-filtered digests and belongs in a test, not in a
    bug report six weeks from now. Eliminating it would mean either scoring the whole
    triaged set again — the waste this change exists to remove — or inventing a margin
    constant with nothing to derive it from.
    """

    def test_a_promoted_band_paper_without_a_score_lands_in_maybe(self) -> None:
        scored = [
            _rec("shown_a", 0.9, llm=2),
            _rec("shown_b", 0.8, llm=2),
            _rec("promoted_by_filtering", 0.7, llm=2),
        ]
        # What `rr update` paid for: the top two.
        window, _ = digest_window(scored, top_n=2, triage_threshold=2, rerank=True)
        for rec in window:
            rec["finescale_p"] = 0.9

        # What `rr digest --since` then tiers, after its filter dropped `shown_a`.
        top, maybe, _ = categorize_papers(
            [r for r in scored if r["arxiv_id"] != "shown_a"],
            top_n=2,
            triage_threshold=2,
            rerank=True,
            finescale_threshold=2 / 3,
        )
        assert [p["arxiv_id"] for p in top] == ["shown_b"]
        assert [p["arxiv_id"] for p in maybe] == ["promoted_by_filtering"]
