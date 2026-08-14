"""Tests for the OpenAlex stage-1 yield probe.

Two things in this script decide whether its output is a measurement at all, and both are
mechanisms this project has already been burned by:

* **Refusal is not emptiness.** `openalex.search_papers` returns ``[]`` when the API refused
  and ``[]`` when it honestly found nothing. A first DBLP measurement read "0 vs 0" after 12
  of 18 requests were rate-limited, and a keyless Semantic Scholar probe produced zeros that
  looked like data. `RequestWatch` is what lets this probe tell them apart, so it is tested
  like load-bearing code rather than like instrumentation.
* **"New" by id is not new by content.** OpenAlex mints a synthetic ``oa:`` id for any work
  whose DOI is not an arXiv DOI, which includes the *published* version of a preprint the
  arXiv pool already holds. `_title_key` is the crude check that keeps the coverage claim
  honest.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evals"))

import openalex_yield  # noqa: E402
from openalex_yield import RequestWatch, _title_key, measure, split_appearances  # noqa: E402

from reporadar.sources import openalex  # noqa: E402


def _row(case: str, in_top: int, pool: int = 250) -> dict[str, object]:
    return {"case": case, "oa_in_top10": in_top, "arxiv_pool": pool}


class TestRequestWatch:
    def test_counts_calls_and_failures(self, monkeypatch: pytest.MonkeyPatch) -> None:
        answers = [{"results": []}, None, {"results": [1]}]
        monkeypatch.setattr(openalex_yield, "REQUEST_INTERVAL_S", 0.0)
        monkeypatch.setattr(openalex, "_request_json", lambda url, *a, **k: answers.pop(0))
        with RequestWatch() as watch:
            for _ in range(3):
                openalex._request_json("http://example.invalid")
        assert (watch.calls, watch.failures) == (3, 1)

    def test_an_empty_result_is_not_a_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The distinction the whole probe rests on: `{"results": []}` is an answer."""
        monkeypatch.setattr(openalex_yield, "REQUEST_INTERVAL_S", 0.0)
        monkeypatch.setattr(openalex, "_request_json", lambda url, *a, **k: {"results": []})
        with RequestWatch() as watch:
            openalex._request_json("http://example.invalid")
        assert (watch.calls, watch.failures) == (1, 0)

    def test_restores_the_original_even_when_the_body_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A leaked wrapper would silently instrument every later test in the process."""
        sentinel = object()
        monkeypatch.setattr(openalex_yield, "REQUEST_INTERVAL_S", 0.0)
        monkeypatch.setattr(openalex, "_request_json", sentinel)
        with pytest.raises(RuntimeError), RequestWatch():
            raise RuntimeError("boom")
        assert openalex._request_json is sentinel

    def test_it_spaces_requests(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The politeness is inside the wrapper so it also covers the adapter's retries,
        which is why the probe passes `rate_limit=0.0` to the adapter itself."""
        slept: list[float] = []
        monkeypatch.setattr(openalex_yield, "REQUEST_INTERVAL_S", 1.5)
        monkeypatch.setattr(openalex_yield.time, "sleep", slept.append)
        monkeypatch.setattr(openalex, "_request_json", lambda url, *a, **k: {})
        with RequestWatch():
            openalex._request_json("http://example.invalid")
            openalex._request_json("http://example.invalid")
        assert slept == [1.5, 1.5]


class TestTitleKey:
    def test_ignores_case_punctuation_and_spacing(self) -> None:
        a = _title_key("Attention Is All You Need!")
        b = _title_key("attention   is all you need")
        assert a == b != ""

    def test_different_papers_do_not_collide(self) -> None:
        assert _title_key("Deep Residual Learning") != _title_key("Deep Reinforcement Learning")

    def test_an_empty_title_is_an_empty_key(self) -> None:
        """Callers discard `""` before matching; a paper with no title must not match every
        other paper with no title and be counted as already-in-the-pool."""
        assert _title_key("") == ""


class TestMeasureRefusesToGuess:
    def test_a_missing_clone_is_skipped_not_scored(self, tmp_path: Path) -> None:
        """No repository means no measurement — never a zero for the source."""
        assert measure({"name": "does-not-exist", "expected_categories": []}, "key") is None


class TestWhereTheSlotsLand:
    """The statistic the first verdict did not compute.

    That verdict read `cases_with >= n/4` — 7 of 25 cleared it by three quarters of a case
    — and printed "a judged A/B is justified" while 11 of the 14 appearances sat in
    repositories where placing is worthless. Counting cases *touched* is not counting cases
    where touching is *good*, so the split is now the computation and the verdict reads it.
    """

    def test_control_appearances_do_not_count_as_merit(self) -> None:
        rows = [_row("webdev", 3), _row("systems", 1), _row("rag", 0)]
        split = split_appearances(rows, {"webdev"})
        assert (split["total"], split["in_controls"], split["on_merit"]) == (4, 3, 1)

    def test_a_thin_pool_win_does_not_count_as_merit(self) -> None:
        """Half the median pool is half the competition; an ordinary paper places."""
        rows = [_row("a", 0, 400), _row("b", 0, 300), _row("numerics", 5, 55)]
        split = split_appearances(rows, set())
        assert (split["in_thin"], split["on_merit"]) == (5, 0)

    def test_a_control_is_not_double_counted_as_thin(self) -> None:
        """A thin-pooled negative control must land in one bucket, or `on_merit` goes
        negative and the verdict reads better than the data."""
        rows = [_row("a", 0, 400), _row("b", 0, 300), _row("webdev", 4, 20)]
        split = split_appearances(rows, {"webdev"})
        assert (split["in_controls"], split["in_thin"], split["on_merit"]) == (4, 0, 0)

    def test_the_buckets_always_sum_to_the_total(self) -> None:
        rows = [_row("webdev", 3, 287), _row("numerics", 5, 55), _row("ann", 1, 295)]
        split = split_appearances(rows, {"webdev"})
        assert split["in_controls"] + split["in_thin"] + split["on_merit"] == split["total"]

    def test_the_measured_run_reproduces(self) -> None:
        """The 2026-08-14 numbers, pinned: 14 appearances, 6 control, 5 thin, 3 on merit."""
        stored = Path(__file__).resolve().parents[1] / "evals" / ".work" / "openalex_yield.json"
        if not stored.is_file():
            pytest.skip("no stored probe run in this checkout (.work is gitignored)")
        rows = json.loads(stored.read_text(encoding="utf-8"))
        split = split_appearances(rows, {"webdev", "cli", "http"})
        assert (split["total"], split["in_controls"], split["in_thin"], split["on_merit"]) == (
            14,
            6,
            5,
            3,
        )

    def test_no_rows_is_not_a_crash(self) -> None:
        assert split_appearances([], set())["total"] == 0
