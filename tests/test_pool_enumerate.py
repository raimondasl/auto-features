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
import urllib.error
from datetime import date
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

        items, record = ep.run_query("topic:x", "tok", fetch=boom, pause=0, backoff=0)
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

    def test_there_is_never_more_than_one_created_qualifier(self) -> None:
        """The defect that voided the first snapshot. GitHub honours only one `created:`
        clause, so emitting two meant the year subdivision narrowed nothing — every slice
        returned the same total_count — and the surviving clause replaced the cutoff."""
        plain = ep._query("machine-learning", 100, 149, "2024-03-02", None)
        sliced = ep._query("machine-learning", 100, 149, "2024-03-02", ("2023-01-01", "2030-12-31"))
        assert plain.count("created:") == 1
        assert sliced.count("created:") == 1

    def test_a_year_slice_can_never_reach_past_the_cutoff(self) -> None:
        """The measured harm: the 2023..2030 slice admitted 192 repositories created after
        the cutoff, the latest in August 2026, into a universe capped at March 2024."""
        q = ep._query("machine-learning", 100, 149, "2024-03-02", ("2023-01-01", "2030-12-31"))
        assert "created:2023-01-01..2024-03-02" in q

    def test_a_year_slice_wholly_before_the_cutoff_keeps_its_own_bound(self) -> None:
        q = ep._query("genomics", 100, 149, "2024-03-02", ("2016-01-01", "2018-12-31"))
        assert "created:2016-01-01..2018-12-31" in q

    def test_every_generated_query_respects_the_cutoff(self) -> None:
        """Swept over the real slice and year grids rather than asserted on one example."""
        cutoff = "2024-03-02"
        years = (
            ("2008-01-01", "2015-12-31"),
            ("2016-01-01", "2018-12-31"),
            ("2019-01-01", "2020-12-31"),
            ("2021-01-01", "2022-12-31"),
            ("2023-01-01", "2030-12-31"),
        )
        for lo, hi in ep.STAR_SLICES:
            for window in (None, *years):
                q = ep._query("x", lo, hi, cutoff, window)
                assert q.count("created:") == 1
                clause = q.split("created:")[1].split(" ")[0]
                upper = clause[2:] if clause.startswith("<=") else clause.split("..")[1]
                assert upper <= cutoff, f"{q} reaches past the cutoff"

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


class TestDpCannotBeBackdated:
    """Dp is the day the snapshot is taken. GitHub search has no historical index — every
    field but `created_at` is a *current* value, so a backdated Dp yields today's data
    wearing an old date and the pre-registration would describe a frame that never existed."""

    def test_a_past_date_is_refused_with_the_reason(self) -> None:
        with pytest.raises(SystemExit) as exc:
            ep.refuse_a_backdated_snapshot("2025-06-01", today=date(2026, 9, 2))
        assert "no historical index" in str(exc.value)

    def test_today_is_allowed(self) -> None:
        ep.refuse_a_backdated_snapshot("2026-09-02", today=date(2026, 9, 2))

    def test_a_future_date_is_allowed(self) -> None:
        ep.refuse_a_backdated_snapshot("2026-09-09", today=date(2026, 9, 2))

    def test_it_runs_before_any_request_is_made(self) -> None:
        import inspect

        src = inspect.getsource(ep.main)
        assert src.index("refuse_a_backdated_snapshot") < src.index("enumerate_universe(")


class TestRequestsAreSpacedAndRetried:
    """A snapshot that cannot be re-taken must not lose a slice to a throttle, and must not
    provoke one either."""

    def test_the_pause_is_taken_before_every_request_not_only_between_pages(self) -> None:
        """The original bug: the short-page `break` came before the sleep, so a slice that
        fitted in one page — most of them — issued its request and moved straight to the next
        query. Several hundred queries would have gone out back to back against a
        30-per-minute limit."""
        waits: list[float] = []
        query = ep._query("compiler", 100, 149, "2024-03-01", None)
        api = _Api({query: [_item("a/one")]})  # one short page
        # "a request just went out", so the next one must wait its turn.
        ep._last_request_at = ep.time.monotonic()
        real_sleep = ep.time.sleep
        try:
            ep.time.sleep = lambda s: waits.append(s)  # type: ignore[assignment]
            ep.run_query(query, "tok", fetch=api, pause=2.2)
        finally:
            ep.time.sleep = real_sleep  # type: ignore[assignment]
        assert waits, "a single-page query took no pause at all"

    def test_a_throttle_is_retried_rather_than_recorded_as_a_lost_slice(self) -> None:
        calls = {"n": 0}

        def flaky(url: str, token: str) -> tuple[dict, str]:
            calls["n"] += 1
            if calls["n"] == 1:
                err = urllib.error.HTTPError(url, 403, "rate limited", None, None)  # type: ignore[arg-type]
                raise err
            return {"total_count": 1, "items": [_item("a/one")]}, "Wed, 02 Sep 2026 12:00:00 GMT"

        real_sleep = ep.time.sleep
        try:
            ep.time.sleep = lambda s: None  # type: ignore[assignment]
            items, record = ep.run_query("topic:x", "tok", fetch=flaky, pause=0, backoff=0)
        finally:
            ep.time.sleep = real_sleep  # type: ignore[assignment]
        assert record.error == ""
        assert len(items) == 1
        assert calls["n"] == 2

    def test_a_permanent_failure_still_becomes_an_error_row(self) -> None:
        def gone(url: str, token: str) -> tuple[dict, str]:
            raise urllib.error.HTTPError(url, 404, "nope", None, None)  # type: ignore[arg-type]

        items, record = ep.run_query("topic:x", "tok", fetch=gone, pause=0, backoff=0)
        assert items == []
        assert "404" in record.error

    def test_archive_names_are_stable_across_processes(self) -> None:
        """`hash()` on a str is salted per process, so the same query produced a different
        filename on every run — unhelpful in an artefact meant to be cited."""
        import inspect

        src = inspect.getsource(ep.run_query)
        assert "hashlib.sha256(query.encode())" in src
        assert "abs(hash(query))" not in src


class TestTheCommittableArchive:
    """§2.1 asks for the raw responses to be archived, and separately forbids
    `github.com/<owner>/<repo>` strings from entering this tree. At one snapshot's scale those
    two requirements collide: the raw payloads carry `html_url` for every repository, tens of
    thousands of exactly the pattern the prior-exposure grep looks for. The trimmed archive
    keeps what makes the enumeration falsifiable and drops what would break the rule."""

    @staticmethod
    def _raw(tmp_path: Path) -> Path:
        raw = tmp_path / "raw"
        raw.mkdir()
        (raw / "aaa-p1.json").write_text(
            json.dumps(
                {
                    "query": "topic:genomics stars:100..149",
                    "url": "https://api.github.com/search/repositories?q=x",
                    "date": "Wed, 02 Sep 2026 12:00:00 GMT",
                    "payload": {"total_count": 7, "items": [_item("acme/one"), _item("b/two")]},
                }
            ),
            encoding="utf-8",
        )
        return raw

    def test_it_keeps_what_makes_the_snapshot_checkable(self, tmp_path: Path) -> None:
        out = tmp_path / "archive.json"
        summary = ep.trim_archive(self._raw(tmp_path), out)
        entry = summary["responses"][0]
        assert entry["query"] == "topic:genomics stars:100..149"
        assert entry["date"].startswith("Wed, 02 Sep 2026")
        assert entry["total_count"] == 7
        assert entry["returned"] == 2
        assert entry["names"] == ["acme/one", "b/two"]

    def test_no_repository_url_survives_the_trim(self, tmp_path: Path) -> None:
        """Checked on the written bytes, since the leak would be inside a nested payload
        rather than in a column heading."""
        out = tmp_path / "archive.json"
        ep.trim_archive(self._raw(tmp_path), out)
        text = out.read_text(encoding="utf-8")
        assert "acme/one" in text
        assert "github.com/acme/one" not in text
        assert "for details" not in text  # the description, which also carries the URL

    def test_the_only_endpoint_string_names_no_repository(self, tmp_path: Path) -> None:
        """The API endpoint unavoidably contains `github.com/search/repositories`, which the
        prior-exposure grep matches as owner `search`, repo `repositories`. No such repository
        exists, so the one match excludes nothing — recorded here so it is a known constant
        rather than a surprise in a later audit."""
        out = tmp_path / "archive.json"
        ep.trim_archive(self._raw(tmp_path), out)
        import re as _re

        hits = _re.findall(
            r"github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", out.read_text(encoding="utf-8")
        )
        assert hits == ["github.com/search/repositories"]

    def test_it_is_far_smaller_than_the_raw_it_replaces(self, tmp_path: Path) -> None:
        """Measured on a realistically sized page. A two-row fixture would be dominated by
        the archive's own fixed header and would assert nothing about the real ratio."""
        raw = tmp_path / "bigraw"
        raw.mkdir()
        (raw / "aaa-p1.json").write_text(
            json.dumps(
                {
                    "query": "topic:machine-learning stars:100..149",
                    "date": "Wed, 02 Sep 2026 12:00:00 GMT",
                    "payload": {
                        "total_count": 900,
                        "items": [_item(f"owner{i}/repo{i}") for i in range(100)],
                    },
                }
            ),
            encoding="utf-8",
        )
        out = tmp_path / "archive.json"
        ep.trim_archive(raw, out)
        raw_bytes = sum(f.stat().st_size for f in raw.glob("*.json"))
        assert out.stat().st_size < raw_bytes * 0.2
