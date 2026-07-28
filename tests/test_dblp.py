"""Tests for reporadar.sources.dblp (mocked HTTP)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from reporadar.sources.dblp import collect_papers, search_papers


def _hits(*infos: dict) -> dict:
    return {"result": {"hits": {"hit": [{"info": info} for info in infos]}}}


class TestSearchPapers:
    @patch("reporadar.sources.dblp._request_json")
    def test_normalizes_hit(self, mock_req: MagicMock) -> None:
        mock_req.return_value = _hits(
            {
                "title": "Raft Consensus.",  # trailing period stripped
                "key": "conf/osdi/AliceBob25",
                "year": "2025",
                "venue": "OSDI",
                "ee": "https://doi.org/x",
                "authors": {
                    "author": [{"@pid": "1", "text": "Alice"}, {"@pid": "2", "text": "Bob"}]
                },
            },
            {"title": "", "key": "k"},  # no title → dropped
        )
        out = search_papers("raft")
        assert len(out) == 1
        p = out[0]
        assert p["arxiv_id"] == "dblp:conf/osdi/AliceBob25"
        assert p["title"] == "Raft Consensus"
        assert p["authors"] == ["Alice", "Bob"]
        assert p["abstract"] == ""
        assert p["categories"] == ["OSDI"]
        assert p["url"] == "https://doi.org/x"

    @patch("reporadar.sources.dblp._request_json")
    def test_single_author_dict(self, mock_req: MagicMock) -> None:
        mock_req.return_value = _hits(
            {"title": "T", "key": "k", "year": "2025", "authors": {"author": {"text": "Solo"}}}
        )
        assert search_papers("x")[0]["authors"] == ["Solo"]

    @patch("reporadar.sources.dblp._request_json")
    def test_failure_returns_empty(self, mock_req: MagicMock) -> None:
        mock_req.return_value = None
        assert search_papers("x") == []


class TestDblpJsonQuirks:
    @patch("reporadar.sources.dblp._request_json")
    def test_ee_as_list_or_dict_coerced_to_str(self, mock_req: MagicMock) -> None:
        mock_req.return_value = _hits(
            {"title": "A", "key": "conf/x/y1", "year": "2025", "ee": ["https://a", "https://b"]},
            {
                "title": "B",
                "key": "conf/x/y2",
                "year": "2025",
                "ee": {"@type": "oa", "text": "https://c"},
            },
        )
        out = search_papers("q")
        assert all(isinstance(p["url"], str) for p in out)
        assert out[0]["url"] == "https://a"
        assert out[1]["url"] == "https://c"

    @patch("reporadar.sources.dblp._request_json")
    def test_venue_as_list_coerced(self, mock_req: MagicMock) -> None:
        mock_req.return_value = _hits(
            {"title": "A", "key": "k", "year": "2025", "venue": ["OSDI", "SOSP"]}
        )
        cats = search_papers("q")[0]["categories"]
        assert cats == ["OSDI"] and isinstance(cats[0], str)  # str, not a nested list

    @patch("reporadar.sources.dblp._request_json")
    def test_single_hit_object_not_list(self, mock_req: MagicMock) -> None:
        # DBLP returns `hit` as one object (not a 1-element list) for a single match.
        mock_req.return_value = {
            "result": {"hits": {"hit": {"info": {"title": "Solo", "key": "k", "year": "2025"}}}}
        }
        out = search_papers("q")
        assert len(out) == 1 and out[0]["title"] == "Solo"

    @patch("reporadar.sources.dblp._request_json")
    def test_corr_entry_gets_real_arxiv_id(self, mock_req: MagicMock) -> None:
        mock_req.return_value = _hits(
            {
                "title": "CoRR Paper",
                "key": "journals/corr/abs-2401-12345",
                "year": "2024",
                "doi": "10.48550/arXiv.2401.12345",
            }
        )
        p = search_papers("q")[0]
        assert p["arxiv_id"] == "2401.12345"  # real arXiv id, not dblp:...
        assert p["url"] == "http://arxiv.org/abs/2401.12345"


class TestCollectPapers:
    @patch("reporadar.sources.dblp.search_papers")
    def test_dedups_and_filters_old_years(self, mock_search: MagicMock) -> None:
        def _p(aid: str, year: str) -> dict:
            return {
                "arxiv_id": f"dblp:{aid}",
                "title": aid,
                "authors": [],
                "abstract": "",
                "categories": [],
                "published": f"{year}-01-01T00:00:00+00:00",
                "updated": None,
                "url": "",
                "pdf_url": None,
            }

        recent, old = _p("a", "2099"), _p("b", "2000")
        mock_search.return_value = [recent, old, recent]  # dup + old
        out = collect_papers(["q"], lookback_days=14)
        ids = [p["arxiv_id"] for p in out]
        assert ids == ["dblp:a"]  # old dropped, recent deduped
