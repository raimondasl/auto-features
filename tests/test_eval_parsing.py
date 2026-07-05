"""Tests for the eval harness's reference parsing (evals/baseline.py, verify.py).

Regression guard for the prose-scraping bug: an honest baseline abstention (an
empty ```json [] block) was overridden by arXiv-looking IDs scraped from the
surrounding "sources reviewed" prose — including a ResearchGate URL path
(``publication/2256929``) that the old-style-ID regex wrongly matched. That
bogus ID then 400'd against arXiv and nuked the baseline's metrics as
``arxiv_unverified`` on every webdev run.
"""

from __future__ import annotations

from baseline import _parse_recommendations
from verify import extract_arxiv_ids


class TestExtractArxivIds:
    def test_new_style(self) -> None:
        assert extract_arxiv_ids("arXiv:2410.14924 and 2301.01261") == [
            "2410.14924",
            "2301.01261",
        ]

    def test_old_style_real_archives(self) -> None:
        assert extract_arxiv_ids("see hep-th/9901001 and cs.LG/0501001") == [
            "hep-th/9901001",
            "cs.LG/0501001",
        ]

    def test_rejects_non_arxiv_url_path(self) -> None:
        # researchgate.net/publication/225692935_... must NOT parse as old-style ID.
        assert extract_arxiv_ids("researchgate.net/publication/225692935_Hardened") == []

    def test_dedupes(self) -> None:
        assert extract_arxiv_ids("2410.14924 ... again 2410.14924") == ["2410.14924"]


class TestParseRecommendations:
    def test_json_block_is_authoritative(self) -> None:
        # An ID mentioned only in prose is discussion, not a recommendation.
        text = (
            "I reviewed arXiv:2304.01982 but it's a replacement model, not an improvement.\n"
            '```json\n[{"arxiv_id": "2409.14683"}]\n```'
        )
        ids, titles = _parse_recommendations(text)
        assert ids == ["2409.14683"]
        assert titles == []

    def test_empty_json_block_is_an_abstention(self) -> None:
        # The webdev bug: an explicit [] plus prose "sources reviewed" must yield NO refs.
        text = (
            "Sources reviewed: https://arxiv.org/pdf/2410.14924 and "
            "researchgate.net/publication/225692935_Foo.\n"
            "My recommendation is to recommend nothing.\n```json\n[]\n```"
        )
        assert _parse_recommendations(text) == ([], [])

    def test_prose_fallback_when_no_json_block(self) -> None:
        # If the baseline ignored the protocol (no JSON block), fall back to prose.
        ids, titles = _parse_recommendations("I recommend arXiv:2409.14683 and 2404.02805.")
        assert ids == ["2409.14683", "2404.02805"]
        assert titles == []

    def test_title_only_recommendations(self) -> None:
        ids, titles = _parse_recommendations(
            '```json\n[{"title": "Some Paper Without An Arxiv Id"}]\n```'
        )
        assert ids == []
        assert titles == ["Some Paper Without An Arxiv Id"]
