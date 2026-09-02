"""The pool's candidate enumeration. [PREREG-judge-validity-pool §2.1]

Two properties carry the whole section, and neither is obvious from reading the code.

**GitHub truncates silently.** A search matching 4,000 repositories returns 1,000 rows and
no error, so a truncated slice is indistinguishable from a small one. If that went unnoticed
the "population" would be whatever the API felt like returning, ordered by its own relevance
ranking — which is not a sampling frame at all. So every slice's `total_count` is compared
against the cap, an overflowing slice is **subdivided and its partial rows discarded**, and a
subdivision that still overflows keeps its rows but stamps `TRUNCATED` into each one.

**No URL reaches the CSV.** The benchmark expansion's prior-exposure rule is a live grep for
`github.com/<owner>/<repo>` over this tree, and this file writes thousands of candidate names
into it. A `url` or `description` column would quietly turn the pool's output into that
benchmark's exclusion list.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest

FRAME = Path(__file__).resolve().parent.parent / "evals" / "frame"


def _load(name: str):  # type: ignore[no-untyped-def]
    if str(FRAME) not in sys.path:
        sys.path.insert(0, str(FRAME))
    spec = importlib.util.spec_from_file_location(name, FRAME / f"{name}.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


ep = _load("enumerate_pool")


def _item(name: str, stars: int = 150, created: str = "2019-04-01") -> dict:
    return {
        "full_name": name,
        "created_at": f"{created}T00:00:00Z",
        "pushed_at": "2026-08-01T00:00:00Z",
        "stargazers_count": stars,
        "language": "Python",
        "topics": ["machine-learning", "pytorch"],
        # Present in the real API payload, and deliberately never written out:
        "html_url": f"https://github.com/{name}",
        "description": f"See https://github.com/{name} for details",
    }


class _Api:
    """A fake search API with a fixed corpus per query, so paging and the cap are real."""

    def __init__(self, corpus: dict[str, list[dict]], totals: dict[str, int] | None = None) -> None:
        self.corpus = corpus
        self.totals = totals or {}
        self.calls: list[str] = []

    def __call__(self, url: str, token: str) -> tuple[dict, str]:
        from urllib.parse import parse_qs, urlparse

        params = parse_qs(urlparse(url).query)
        query = params["q"][0]
        page = int(params["page"][0])
        self.calls.append(query)
        items = self.corpus.get(query, [])
        start = (page - 1) * ep.PER_PAGE
        return (
            {
                "total_count": self.totals.get(query, len(items)),
                "items": items[start : start + ep.PER_PAGE],
            },
            "Wed, 02 Sep 2026 12:00:00 GMT",
        )


class TestTheCapIsTreatedAsTruncation:
    def test_an_overflowing_slice_is_subdivided_and_its_partial_rows_discarded(self) -> None:
        """The rows from a truncated query are NOT kept alongside complete slices. They are
        the API's relevance-ranked top 1,000, and mixing them in would put an unmarked
        convenience sample inside a frame that claims to be exhaustive."""
        wide = ep._query("machine-learning", 100, 149, "2024-03-01", None)
        sub = ep._query("machine-learning", 100, 149, "2024-03-01", ("2019-01-01", "2020-12-31"))
        api = _Api(
            {wide: [_item(f"truncated/repo{i}") for i in range(100)], sub: [_item("real/repo")]},
            totals={wide: 4000},
        )
        out = ep.enumerate_universe(["machine-learning"], "2024-03-01", "tok", fetch=api, pause=0)
        assert "real/repo" in out.rows
        assert not any(name.startswith("truncated/") for name in out.rows)

    def test_a_slice_still_truncated_after_subdivision_is_recorded(self) -> None:
        """A frame that silently drops what it could not enumerate is not a frame.

        Both the wide slice and one of its year subdivisions overflow here — a single busy
        year inside a busy slice is exactly the case where subdivision is not enough.
        """
        wide = ep._query("genomics", 100, 149, "2024-03-01", None)
        sub = ep._query("genomics", 100, 149, "2024-03-01", ("2019-01-01", "2020-12-31"))
        api = _Api(
            {wide: [_item("wide/repo")], sub: [_item("big/repo")]},
            totals={wide: 4000, sub: 5000},
        )
        out = ep.enumerate_universe(["genomics"], "2024-03-01", "tok", fetch=api, pause=0)
        assert any(q.query == sub and q.truncated for q in out.queries)
        assert len(out.truncated) == 2  # the wide slice and the year that stayed over

    def test_a_slice_within_the_cap_is_kept_whole(self) -> None:
        query = ep._query("genomics", 100, 149, "2024-03-01", None)
        api = _Api({query: [_item("acme/small")]})
        out = ep.enumerate_universe(["genomics"], "2024-03-01", "tok", fetch=api, pause=0)
        assert "acme/small" in out.rows
        assert not out.truncated

    def test_paging_stops_on_a_short_page_rather_than_asking_ten_times(self) -> None:
        query = ep._query("compiler", 100, 149, "2024-03-01", None)
        api = _Api({query: [_item(f"a/r{i}") for i in range(150)]})
        items, record = ep.run_query(query, "tok", fetch=api, pause=0)
        assert len(items) == 150
        assert record.pages == 2


class TestNoUrlEverReachesTheCsv:
    def test_the_columns_carry_no_url_and_no_description(self) -> None:
        assert not any("url" in c for c in ep.COLUMNS)
        assert "description" not in ep.COLUMNS

    def test_a_written_universe_contains_no_github_urls(self, tmp_path: Path) -> None:
        """Checked on the bytes. The API payload carries `html_url` and a description that
        also contains the URL, so a lazy row builder would leak it twice."""
        query = ep._query("machine-learning", 100, 149, "2024-03-01", None)
        api = _Api({query: [_item("acme/leaky")]})
        out = ep.enumerate_universe(["machine-learning"], "2024-03-01", "tok", fetch=api, pause=0)
        csv_path = tmp_path / "pool-universe-Dp.csv"
        ep.write_universe(out, csv_path)
        text = csv_path.read_text(encoding="utf-8")
        assert "acme/leaky" in text
        assert "github.com" not in text
        assert "for details" not in text


class TestTheCoverageArtefactIsFalsifiable:
    def test_it_records_what_the_api_said_it_had(self, tmp_path: Path) -> None:
        """`total_count` beside `fetched` is what lets a reader check the row count against
        the API's own totals. Without it "we enumerated the population" cannot be checked."""
        query = ep._query("statistics", 100, 149, "2024-03-01", None)
        api = _Api({query: [_item("acme/one")]})
        out = ep.enumerate_universe(["statistics"], "2024-03-01", "tok", fetch=api, pause=0)
        path = tmp_path / "coverage.json"
        ep.write_coverage(out, path)
        cov = json.loads(path.read_text(encoding="utf-8"))
        assert cov["n_rows"] == 1
        assert cov["queries"][0]["total_count"] == 1
        assert cov["queries"][0]["date"].startswith("Wed, 02 Sep 2026")

    def test_a_failed_query_is_an_error_row_not_a_silent_zero(self) -> None:
        """Two benchmark repos once ended up with an empty pool that then scored as a
        legitimate zero. A dropped query here would shrink the population the same way."""

        def boom(url: str, token: str) -> tuple[dict, str]:
            raise TimeoutError("upstream said no")

        items, record = ep.run_query("topic:x", "tok", fetch=boom, pause=0)
        assert items == []
        assert "upstream said no" in record.error

    def test_raw_responses_are_archived(self, tmp_path: Path) -> None:
        query = ep._query("astronomy", 100, 149, "2024-03-01", None)
        api = _Api({query: [_item("acme/star")]})
        archive = tmp_path / "raw"
        ep.run_query(query, "tok", fetch=api, archive=archive, pause=0)
        saved = list(archive.glob("*.json"))
        assert len(saved) == 1
        blob = json.loads(saved[0].read_text(encoding="utf-8"))
        assert blob["query"] == query
        assert blob["payload"]["items"][0]["full_name"] == "acme/star"


class TestTheQueryItself:
    def test_it_pins_the_mechanical_exclusions_in_the_query(self) -> None:
        q = ep._query("machine-learning", 100, 149, "2024-03-01", None)
        assert "fork:false" in q
        assert "archived:false" in q
        assert "created:<=2024-03-01" in q
        assert "stars:100..149" in q

    def test_the_top_slice_is_open_ended(self) -> None:
        assert ep._query("x", 10000, None, "2024-03-01", None).count("stars:>=10000") == 1

    def test_star_slices_are_contiguous_and_start_at_the_floor(self) -> None:
        """A gap between slices is a silently missing stratum of the population."""
        assert ep.STAR_SLICES[0][0] == ep.MIN_STARS
        for (_, hi), (lo_next, _) in zip(ep.STAR_SLICES, ep.STAR_SLICES[1:], strict=False):
            assert hi is not None and lo_next == hi + 1

    def test_the_created_cutoff_is_thirty_months_back(self) -> None:
        assert ep.created_cutoff("2026-09-05") == "2024-03-05"
        assert ep.created_cutoff("2026-01-31") == "2023-07-28"


class TestTheCommittedTopicList:
    def test_it_parses_and_has_no_duplicates(self) -> None:
        topics = json.loads((FRAME / "pool" / "topics.json").read_text(encoding="utf-8"))
        assert isinstance(topics, list)
        assert all(isinstance(t, str) and t for t in topics)
        assert len(topics) == len(set(topics))

    def test_it_spans_the_domains_the_blind_spot_is_measured_on(self) -> None:
        """§6.2 sizes a life-science and materials blind spot rather than closing it. That
        is only a measurement if repositories from those domains are actually enumerated and
        screened — otherwise the DOI covariate has nothing to count."""
        topics = set(json.loads((FRAME / "pool" / "topics.json").read_text(encoding="utf-8")))
        assert {"bioinformatics", "computational-biology", "genomics"} <= topics
        assert {"materials-science", "computational-chemistry"} <= topics
        assert {"machine-learning", "computer-vision"} <= topics


class TestTheWrittenCsvRoundTrips:
    def test_rows_are_sorted_and_complete(self, tmp_path: Path) -> None:
        query = ep._query("compiler", 100, 149, "2024-03-01", None)
        api = _Api({query: [_item("z/last"), _item("a/first")]})
        out = ep.enumerate_universe(["compiler"], "2024-03-01", "tok", fetch=api, pause=0)
        path = tmp_path / "u.csv"
        ep.write_universe(out, path)
        with path.open(encoding="utf-8", newline="") as fh:
            rows = list(csv.DictReader(fh))
        assert [r["full_name"] for r in rows] == ["a/first", "z/last"]
        assert rows[0]["created_at"] == "2019-04-01"
        assert rows[0]["topics"] == "machine-learning|pytorch"


class TestTheTokenIsRequired:
    def test_an_absent_token_fails_loudly(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Unauthenticated search is 10 requests/minute against the hundreds this needs, so
        a missing token must stop the run rather than produce a half-finished snapshot that
        can never be re-taken."""
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        monkeypatch.setattr(
            ep.subprocess,
            "run",
            lambda *a, **k: type("R", (), {"stdout": "", "returncode": 1})(),
        )
        with pytest.raises(SystemExit):
            ep.github_token()

    def test_the_environment_wins_over_gh(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GITHUB_TOKEN", "from-env")
        assert ep.github_token() == "from-env"


class TestAnIncompleteSliceIsVisibleInTheCsvItself:
    def test_rows_from_a_still_truncated_subdivision_are_marked(self) -> None:
        """Kept, because dropping them loses real coverage — but marked, because they are
        the API's relevance-ranked top 1,000 rather than a complete slice. A reader should
        not have to cross-reference `coverage.json` to learn that a row came from an
        incomplete query."""
        wide = ep._query("machine-learning", 100, 149, "2024-03-01", None)
        sub = ep._query("machine-learning", 100, 149, "2024-03-01", ("2019-01-01", "2020-12-31"))
        ok = ep._query("machine-learning", 100, 149, "2024-03-01", ("2021-01-01", "2022-12-31"))
        api = _Api(
            {wide: [_item("w/r")], sub: [_item("partial/repo")], ok: [_item("complete/repo")]},
            totals={wide: 4000, sub: 5000},
        )
        out = ep.enumerate_universe(["machine-learning"], "2024-03-01", "tok", fetch=api, pause=0)
        assert ep.TRUNCATED_MARK in out.rows["partial/repo"]["slice"]
        assert ep.TRUNCATED_MARK not in out.rows["complete/repo"]["slice"]
