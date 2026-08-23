"""Tests for reporadar.sources.openalex."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

from reporadar.sources.openalex import (
    _extract_arxiv_id,
    _normalize_paper,
    collect_papers,
    reconstruct_abstract,
    search_papers,
)


def _mock_response(data: object) -> MagicMock:
    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps(data).encode()
    mock_resp.__enter__ = MagicMock(return_value=mock_resp)
    mock_resp.__exit__ = MagicMock(return_value=False)
    return mock_resp


def _make_oa_work(**overrides) -> dict:
    base = {
        "id": "https://openalex.org/W12345",
        "doi": "https://doi.org/10.48550/arXiv.2401.12345",
        "title": "Test Paper",
        "display_name": "Test Paper",
        "authorships": [
            {"author": {"display_name": "Alice Smith"}},
            {"author": {"display_name": "Bob Jones"}},
        ],
        "abstract_inverted_index": {
            "A": [0],
            "test": [1],
            "abstract": [2],
            "about": [3],
            "machine": [4],
            "learning": [5],
        },
        "primary_topic": {"display_name": "Machine Learning"},
        "publication_date": "2026-01-15",
        "open_access": {"oa_url": "https://arxiv.org/pdf/2401.12345"},
        "ids": {"openalex": "https://openalex.org/W12345"},
    }
    base.update(overrides)
    return base


class TestReconstructAbstract:
    def test_basic_reconstruction(self) -> None:
        inv_index = {"Hello": [0], "world": [1], "of": [2], "ML": [3]}
        result = reconstruct_abstract(inv_index)
        assert result == "Hello world of ML"

    def test_repeated_words(self) -> None:
        inv_index = {"the": [0, 2], "cat": [1], "dog": [3]}
        result = reconstruct_abstract(inv_index)
        assert result == "the cat the dog"

    def test_empty_index(self) -> None:
        assert reconstruct_abstract(None) == ""
        assert reconstruct_abstract({}) == ""


class TestExtractArxivId:
    def test_from_doi(self) -> None:
        work = {"doi": "https://doi.org/10.48550/arXiv.2401.12345", "id": "x", "ids": {}}
        assert _extract_arxiv_id(work) == "2401.12345"

    def test_synthetic_id(self) -> None:
        work = {
            "doi": "",
            "id": "https://openalex.org/W99999",
            "ids": {"openalex": "https://openalex.org/W99999"},
        }
        result = _extract_arxiv_id(work)
        assert result == "oa:W99999"

    def test_no_id_at_all(self) -> None:
        work = {"doi": "", "id": "", "ids": {}}
        assert _extract_arxiv_id(work) == ""


class TestSearchPapers:
    @patch("reporadar.sources.openalex.urllib.request.urlopen")
    def test_returns_normalized_papers(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.return_value = _mock_response({"results": [_make_oa_work()]})

        results = search_papers("machine learning")

        assert len(results) == 1
        assert results[0]["arxiv_id"] == "2401.12345"
        assert results[0]["title"] == "Test Paper"
        assert "Alice Smith" in results[0]["authors"]

    @patch("reporadar.sources.openalex.urllib.request.urlopen")
    def test_query_url_construction(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.return_value = _mock_response({"results": []})

        search_papers("test query", email="user@example.com")

        call_args = mock_urlopen.call_args
        req = call_args[0][0]
        assert "mailto=user%40example.com" in req.full_url
        assert "search=test+query" in req.full_url

    @patch("reporadar.sources.openalex.urllib.request.urlopen")
    def test_api_key_in_url(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.return_value = _mock_response({"results": []})

        search_papers("test query", api_key="secret-key")

        req = mock_urlopen.call_args[0][0]
        assert "api_key=secret-key" in req.full_url

    @patch("reporadar.sources.openalex.urllib.request.urlopen")
    def test_no_api_key_when_absent(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.return_value = _mock_response({"results": []})

        search_papers("test query")

        req = mock_urlopen.call_args[0][0]
        assert "api_key=" not in req.full_url

    @patch("reporadar.sources.openalex.urllib.request.urlopen")
    def test_abstract_reconstruction(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.return_value = _mock_response({"results": [_make_oa_work()]})

        results = search_papers("test")

        assert "test abstract" in results[0]["abstract"]


class TestPaperNormalization:
    def test_full_normalization(self) -> None:
        work = _make_oa_work()
        result = _normalize_paper(work)
        assert result is not None
        assert result["arxiv_id"] == "2401.12345"
        assert result["title"] == "Test Paper"
        assert result["authors"] == ["Alice Smith", "Bob Jones"]
        assert "test abstract" in result["abstract"]
        assert result["published"].startswith("2026-01-15")

    def test_none_on_missing_title(self) -> None:
        work = _make_oa_work(title="", display_name="")
        result = _normalize_paper(work)
        assert result is None


class TestCollectPapers:
    @patch("reporadar.sources.openalex.time.sleep")
    @patch("reporadar.sources.openalex.search_papers")
    def test_dedup_across_queries(self, mock_search: MagicMock, mock_sleep: MagicMock) -> None:
        # Use a recent date (relative to now) so the paper stays inside the
        # default lookback window regardless of when the suite runs.
        recent = (datetime.now(UTC) - timedelta(days=3)).strftime("%Y-%m-%d")
        paper = {
            "arxiv_id": "2401.12345",
            "title": "Test Paper",
            "authors": ["Alice"],
            "abstract": "abstract",
            "categories": [],
            "published": f"{recent}T00:00:00+00:00",
            "updated": None,
            "url": "http://arxiv.org/abs/2401.12345",
            "pdf_url": None,
        }
        mock_search.return_value = [paper]

        results = collect_papers(["query1", "query2"], rate_limit=0.0)
        assert len(results) == 1

    @patch("reporadar.sources.openalex.time.sleep")
    @patch("reporadar.sources.openalex.search_papers")
    def test_date_filtering(self, mock_search: MagicMock, mock_sleep: MagicMock) -> None:
        old_paper = {
            "arxiv_id": "old",
            "title": "Old",
            "authors": [],
            "abstract": "",
            "categories": [],
            "published": "2020-01-01T00:00:00+00:00",
            "updated": None,
            "url": "",
            "pdf_url": None,
        }
        recent = (datetime.now(UTC) - timedelta(days=5)).strftime("%Y-%m-%d")
        new_paper = {
            "arxiv_id": "new",
            "title": "New",
            "authors": [],
            "abstract": "",
            "categories": [],
            "published": f"{recent}T00:00:00+00:00",
            "updated": None,
            "url": "",
            "pdf_url": None,
        }
        mock_search.return_value = [old_paper, new_paper]

        results = collect_papers(["q1"], lookback_days=30, rate_limit=0.0)

        arxiv_ids = [p["arxiv_id"] for p in results]
        assert "new" in arxiv_ids
        assert "old" not in arxiv_ids


class TestArxivIdFromLowercaseDoi:
    """OpenAlex normalises DOIs, so the same work can arrive with a lowercase `arxiv.`.

    The guard was case-insensitive (`"arxiv" in doi.lower()`) but the split was not
    (`doi.split("arXiv.")`), so a lowercase DOI passed the guard, failed the split, and
    fell through to a synthetic `oa:W…` id — the same paper arXiv had already supplied,
    now impossible to dedup against it. The pre-existing test could not catch this: it
    hand-writes the capital-X form on both sides.
    """

    def test_lowercase_doi_yields_the_arxiv_id(self) -> None:
        work = {"doi": "https://doi.org/10.48550/arxiv.2401.12345", "id": "x", "ids": {}}
        assert _extract_arxiv_id(work) == "2401.12345"

    def test_uppercase_doi_still_works(self) -> None:
        work = {"doi": "https://doi.org/10.48550/arXiv.2401.12345", "id": "x", "ids": {}}
        assert _extract_arxiv_id(work) == "2401.12345"

    def test_both_casings_resolve_to_the_same_id(self) -> None:
        # The point of the fix: one paper, one id, whatever OpenAlex chose to send.
        lower = {"doi": "https://doi.org/10.48550/arxiv.2401.12345", "id": "x", "ids": {}}
        upper = {"doi": "https://doi.org/10.48550/ARXIV.2401.12345", "id": "x", "ids": {}}
        assert _extract_arxiv_id(lower) == _extract_arxiv_id(upper)
        assert not _extract_arxiv_id(lower).startswith("oa:")

    def test_a_non_arxiv_doi_yields_no_arxiv_id(self) -> None:
        """Still the point of this case; the id it falls back to has changed.

        It was the OpenAlex handle `oa:W123`, and F15 makes it the DOI — so the same paper
        from Semantic Scholar or bioRxiv now collides with this record instead of joining
        the pool beside it. See `tests/test_nonarxiv_parity.py`.
        """
        work = {
            "doi": "https://doi.org/10.1145/3459637",
            "id": "https://openalex.org/W123",
            "ids": {},
        }
        assert _extract_arxiv_id(work) == "doi:10.1145/3459637"

    def test_a_work_with_no_doi_still_falls_back_to_the_openalex_handle(self) -> None:
        """The coverage the case above used to carry: DOI-first is not DOI-only."""
        work = {"id": "https://openalex.org/W123", "ids": {}}
        assert _extract_arxiv_id(work) == "oa:W123"


class TestArxivIdFromLocations:
    """§39.5: a journal article and its arXiv preprint were two ids and one paper.

    Measured on the matsci OpenAlex arm — the same paper reached Top Picks TWICE in five of
    the five cases the channel contributed to, and zero times in the arXiv-only control. The
    `ids` block never carries an arXiv id for a published version; `locations` sometimes does.
    """

    def _work(self, locations: list[dict] | None) -> dict:
        return {
            "doi": "https://doi.org/10.1038/s41524-020-00406-3",
            "id": "https://openalex.org/W3034141459",
            "ids": {"doi": "https://doi.org/10.1038/s41524-020-00406-3"},
            "locations": locations,
        }

    def test_the_matbench_pair_now_shares_one_id(self) -> None:
        """The real duplicate from `mat-featurize`, both copies judged 1 (§39.5)."""
        work = self._work(
            [
                {"landing_page_url": "https://doi.org/10.1038/s41524-020-00406-3"},
                {"landing_page_url": "http://arxiv.org/abs/2005.00707"},
            ]
        )
        assert _extract_arxiv_id(work) == "2005.00707"

    def test_the_lattice_dynamics_pair_too(self) -> None:
        """`mat-phonon`'s pair, where the two copies were even judged differently (2 and 3)."""
        work = {
            "doi": "https://doi.org/10.1103/physrevb.92.184301",
            "id": "https://openalex.org/W1",
            "ids": {},
            "locations": [{"landing_page_url": "http://arxiv.org/abs/1510.04418"}],
        }
        assert _extract_arxiv_id(work) == "1510.04418"

    def test_a_pdf_url_works_and_the_extension_is_stripped(self) -> None:
        work = self._work([{"pdf_url": "https://arxiv.org/pdf/2005.00707v2.pdf"}])
        assert _extract_arxiv_id(work) == "2005.00707v2"

    def test_pre_2007_ids_survive_the_slash(self) -> None:
        work = self._work([{"landing_page_url": "https://arxiv.org/abs/cs/0602007"}])
        assert _extract_arxiv_id(work) == "cs/0602007"

    def test_no_arxiv_location_falls_back_to_the_doi(self) -> None:
        """CHGNet's case: three of the five duplicates list no arXiv location at all.

        The fix is partial by measurement, and this pins the half that it does not reach so
        a later reader does not assume the defect is closed.
        """
        work = self._work(
            [
                {"landing_page_url": "https://doi.org/10.1038/s42256-023-00716-3"},
                {"landing_page_url": "https://www.repository.cam.ac.uk/handle/1810/357350"},
            ]
        )
        assert _extract_arxiv_id(work) == "doi:10.1038/s41524-020-00406-3"

    def test_missing_or_malformed_locations_do_not_raise(self) -> None:
        for locations in (None, [], [None], ["not-a-dict"], [{}], [{"landing_page_url": None}]):
            assert _extract_arxiv_id(self._work(locations)) == "doi:10.1038/s41524-020-00406-3"

    def test_a_non_arxiv_id_shaped_url_is_rejected_rather_than_used(self) -> None:
        """`is_arxiv_id` guards the capture, so an unanticipated URL shape degrades safely."""
        work = self._work([{"landing_page_url": "https://arxiv.org/abs/not-an-id"}])
        assert _extract_arxiv_id(work) == "doi:10.1038/s41524-020-00406-3"

    def test_an_arxiv_doi_still_wins_over_locations(self) -> None:
        """Order is unchanged where the DOI already answers the question."""
        work = {
            "doi": "https://doi.org/10.48550/arXiv.2401.12345",
            "id": "x",
            "ids": {},
            "locations": [{"landing_page_url": "http://arxiv.org/abs/9999.99999"}],
        }
        assert _extract_arxiv_id(work) == "2401.12345"

    def test_locations_is_requested_from_the_api(self) -> None:
        """A select that omits it makes the whole fix a no-op that still passes unit tests."""
        with patch("reporadar.sources.openalex.urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.return_value = _mock_response({"results": []})
            search_papers("q")
            url = mock_urlopen.call_args[0][0].full_url
        assert "locations" in url
