"""Tests for reporadar.signals.integrity."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import MagicMock, patch

from reporadar.signals.integrity import (
    detect_withdrawal,
    fetch_comments,
    find_withdrawn,
    stale_ids,
)


class TestDetectWithdrawal:
    def test_title_that_is_the_notice(self) -> None:
        assert detect_withdrawal(title="Withdrawn") == "title"
        assert detect_withdrawal(title="A Tight Bound (withdrawn)") == "title"

    def test_bare_withdrawn_comment_counts(self) -> None:
        # The single most common real phrasing: the entire comment is one word.
        # A comment is short curated metadata, so this is unambiguous there.
        assert detect_withdrawal("A Paper", "We study X.", "Withdrawn") == "comment"
        assert detect_withdrawal("A Paper", "We study X.", "WITHDRAWN") == "comment"

    def test_varied_real_comment_phrasings(self) -> None:
        for comment in (
            "This paper has been withdrawn by the author.",
            "The paper is withdrawn because of an error in Theorem 2.",
            "This version is withdrawn as the results do not hold.",
            "This paper as been withdrawn",  # real typo seen on arXiv
            "Withdrawn by the authors due to an incomplete approval process",
            "arXiv admin note: this paper has been withdrawn by arXiv",
        ):
            assert detect_withdrawal("A Paper", "We study X.", comment) == "comment", comment

    def test_retracted_and_withdrew_phrasings(self) -> None:
        """A sample drawn by searching for "withdrawn" cannot contain these.

        Which is exactly how they stayed invisible through two rounds of validation.
        Sampled by the other verbs instead, the matcher now catches 30/36 ("withdrew"),
        53/60 ("retracted") and 12/14 ("has been retracted").
        """
        for comment in (
            "The paper was retracted",
            "retracted due to an error in Section 5",
            "This manuscript has been retracted from publication",
            "Withdrew the paper, resubmit a rewritten version",
            "The authors withdrew this submission",
            "arXiv admin note: removed by arXiv administrators",
        ):
            assert detect_withdrawal("A Paper", "We study X.", comment) == "comment", comment

    def test_a_notice_leading_the_comment_wins_over_a_cross_reference(self) -> None:
        # "Withdrawn; superseded by arXiv:X" is a withdrawal that names its
        # replacement — the xref guard alone would have discarded it.
        for comment in (
            "Withdrawn; superseded by arXiv:2504.01234",
            "Retracted. See arXiv:2504.01234 for the corrected version.",
            "v2: withdrawn, replaced by arXiv:2504.01234",
        ):
            assert detect_withdrawal("A Paper", "We study X.", comment) == "comment", comment

    def test_a_withdrawn_previous_version_is_a_live_paper(self) -> None:
        # A corrected paper explaining that its *earlier* version was withdrawn is
        # not itself withdrawn; flagging it would hide good work.
        for comment in (
            "The previous version was withdrawn because of a proof error; this one fixes it.",
            "Earlier version withdrawn. This revision corrects Theorem 3.",
        ):
            assert detect_withdrawal("A Paper", "We study X.", comment) is None, comment

    def test_an_anchored_notice_still_wins_over_the_prior_version_guard(self) -> None:
        comment = "The previous version had an error; this paper has been withdrawn entirely."
        assert detect_withdrawal("A Paper", "We study X.", comment) == "comment"

    def test_abstract_notice_without_a_comment(self) -> None:
        # Some authors replace the abstract with the notice, so the free local check
        # catches those. It is a bonus, not a substitute for the comment fetch.
        assert (
            detect_withdrawal("A Paper", "This paper has been withdrawn by the authors.")
            == "abstract"
        )

    def test_prose_about_withdrawal_is_not_a_withdrawal(self) -> None:
        # The reason prose gets an anchored matcher: papers legitimately discuss
        # withdrawal. A liberal match here would flag whole fields of research.
        for abstract in (
            "We analyse drugs withdrawn from the market between 1990 and 2020.",
            "Bidders may profit by withdrawing a bid before the deadline.",
            "Withdrawal symptoms are modelled as a stochastic process.",
        ):
            assert detect_withdrawal("A Paper", abstract) is None, abstract

    def test_ordinary_comment_is_not_a_withdrawal(self) -> None:
        assert detect_withdrawal("A Paper", "We study X.", "12 pages, 3 figures") is None
        assert detect_withdrawal("A Paper", "We study X.", "NeurIPS 2026 camera-ready") is None

    def test_withdrawal_of_a_cited_paper_does_not_flag_this_one(self) -> None:
        # The only false positive in a 500-paper control sample.
        comment = 'incorporates arXiv:2011.10199 "Super-clustering", which has been withdrawn'
        assert detect_withdrawal("New version", "We study X.", comment) is None

    def test_self_withdrawal_still_detected_alongside_a_citation(self) -> None:
        comment = "supersedes arXiv:2011.10199; this paper has been withdrawn"
        assert detect_withdrawal("New version", "We study X.", comment) == "comment"

    def test_no_signal_at_all(self) -> None:
        assert detect_withdrawal("A Paper", "We study X.", None) is None
        assert detect_withdrawal() is None


class TestFetchComments:
    def test_skips_non_arxiv_ids_entirely(self) -> None:
        # Synthetic ids from other sources would make arXiv reject the whole request.
        with patch("reporadar.signals.integrity._client") as client:
            assert fetch_comments(["ss:12345", "oa:W1", "biorxiv:10.1101/x"]) == {}
            client.assert_not_called()

    def test_returns_comments_keyed_by_the_id_passed_in(self) -> None:
        result = MagicMock()
        result.entry_id = "http://arxiv.org/abs/1407.6496v2"
        result.comment = "This paper has been withdrawn"
        with patch("reporadar.signals.integrity._client") as client:
            client.return_value.results.return_value = [result]
            got = fetch_comments(["1407.6496v2"])
        assert got == {"1407.6496v2": "This paper has been withdrawn"}

    def test_version_mismatch_still_maps_back(self) -> None:
        # arXiv answers with the LATEST version regardless of the version requested.
        result = MagicMock()
        result.entry_id = "http://arxiv.org/abs/2607.09080v2"
        result.comment = "Withdrawn"
        with patch("reporadar.signals.integrity._client") as client:
            client.return_value.results.return_value = [result]
            got = fetch_comments(["2607.09080v1"])
        assert got == {"2607.09080v1": "Withdrawn"}

    def test_ids_arxiv_drops_are_absent_not_clean(self) -> None:
        # arXiv silently omits unknown ids rather than erroring. A missing key must
        # mean "unknown", never "no withdrawal" — the easiest way to ship the exact
        # absent-is-not-zero bug the house rules warn about.
        result = MagicMock()
        result.entry_id = "http://arxiv.org/abs/2401.00001v1"
        result.comment = ""
        with patch("reporadar.signals.integrity._client") as client:
            client.return_value.results.return_value = [result]
            got = fetch_comments(["2401.00001v1", "9999.99999v1"])
        assert "9999.99999v1" not in got
        assert got["2401.00001v1"] == ""

    def test_network_failure_degrades_to_no_signal(self) -> None:
        with patch("reporadar.signals.integrity._client") as client:
            client.return_value.results.side_effect = RuntimeError("boom")
            assert fetch_comments(["2401.00001v1"]) == {}

    def test_empty_input(self) -> None:
        assert fetch_comments([]) == {}


class TestStaleIds:
    def _stamp(self, days_ago: float) -> str:
        return (datetime.now(UTC) - timedelta(days=days_ago)).isoformat()

    def test_never_checked_papers_come_first(self) -> None:
        got = stale_ids(["new", "old"], {"old": self._stamp(30)})
        assert got[0] == "new"

    def test_recently_checked_papers_are_skipped(self) -> None:
        # Withdrawal is rare and not urgent; arXiv wants 3s between requests, so an
        # unbounded pass over an all-time store would cost minutes per run.
        assert stale_ids(["a"], {"a": self._stamp(1)}) == []

    def test_stale_papers_are_rechecked_oldest_first(self) -> None:
        got = stale_ids(
            ["recent-stale", "very-stale"],
            {"recent-stale": self._stamp(8), "very-stale": self._stamp(90)},
        )
        assert got == ["very-stale", "recent-stale"]

    def test_cap_still_makes_progress_through_a_backlog(self) -> None:
        # Oldest-first ordering means a capped run works through the queue instead of
        # re-checking the same head every time.
        checked = {f"p{i}": self._stamp(100 - i) for i in range(10)}
        got = stale_ids(list(checked), checked, limit=3)
        assert got == ["p0", "p1", "p2"]

    def test_unparseable_timestamp_is_treated_as_unchecked(self) -> None:
        assert stale_ids(["a"], {"a": "not-a-date"}) == ["a"]

    def test_naive_timestamp_does_not_crash(self) -> None:
        # Older rows could lack a timezone; comparing naive to aware raises.
        assert stale_ids(["a"], {"a": "2020-01-01T00:00:00"}) == ["a"]

    def test_empty_input(self) -> None:
        assert stale_ids([], {}) == []


class TestFindWithdrawn:
    def _paper(self, arxiv_id: str, **over: Any) -> dict[str, Any]:
        return {
            "arxiv_id": arxiv_id,
            "title": "A Paper",
            "abstract": "We study X.",
            **over,
        }

    def test_uses_the_fetched_comment(self) -> None:
        papers = [self._paper("2401.1v1"), self._paper("2401.2v1")]
        got = find_withdrawn(papers, {"2401.1v1": "Withdrawn"})
        assert got == {"2401.1v1": "comment"}

    def test_falls_back_to_stored_text_when_no_comment_arrived(self) -> None:
        # Comments may be missing (request failed, id dropped) — the local fields
        # still catch roughly half of real withdrawals and cost nothing.
        papers = [self._paper("2401.1v1", abstract="This paper has been withdrawn.")]
        assert find_withdrawn(papers, {}) == {"2401.1v1": "abstract"}

    def test_clean_papers_are_absent_from_the_result(self) -> None:
        papers = [self._paper("2401.1v1"), self._paper("2401.2v1")]
        assert find_withdrawn(papers, {"2401.1v1": "10 pages", "2401.2v1": ""}) == {}

    def test_paper_without_an_id_is_skipped(self) -> None:
        assert find_withdrawn([{"title": "x", "abstract": "y"}], {}) == {}
