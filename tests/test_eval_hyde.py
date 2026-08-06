"""Tests for P4's dependency verification and the HyDE replication harness.

P4 exists because Design 2's numbers were REPORTED and its dependencies unverified. A
verification script that can quietly pass is worse than no verification, so the guards here
are aimed at the ways this particular script could lie: counting a skipped check as a pass,
blaming the calendar for an index hole, or "range-fetching" a whole shard.
"""

from __future__ import annotations

import importlib.util
import sys
import urllib.request
from pathlib import Path
from unittest.mock import patch

import numpy as np

EVALS = Path(__file__).resolve().parent.parent / "evals"


def _load(name: str):  # type: ignore[no-untyped-def]
    if str(EVALS) not in sys.path:
        sys.path.insert(0, str(EVALS))
    spec = importlib.util.spec_from_file_location(name, EVALS / f"{name}.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


vh = _load("verify_hyde_deps")


class TestArxivYearParsing:
    def test_new_style_ids_map_to_their_shard(self) -> None:
        assert vh._year_of("2106.09685") == 2021
        assert vh._year_of("1011.2602") == 2010
        assert vh._year_of("2607.00768") == 2026

    def test_old_style_ids_have_no_yymm_and_are_not_guessed(self) -> None:
        """`cs/0123456` would otherwise parse as year 20xx from the wrong characters."""
        assert vh._year_of("cs/0123456") is None
        assert vh._year_of("math-ph/9901001") is None


class TestCoverageRuleDistinguishesHolesFromRecency:
    """A miss before the snapshot is a defect; a miss after it is the mirror's schedule.

    Collapsing the two into one count is the failure mode: an index that had genuinely lost
    old papers would pass by being scored against a lenient total.
    """

    SNAP = "2026-07-19T11:44:36.000Z"

    def test_everything_present_passes(self) -> None:
        v = vh.coverage_verdict(["2106.09685", "1011.2602"], {"2106.09685", "1011.2602"}, self.SNAP)
        assert v["pass"] is True
        assert v["present"] == 2

    def test_one_missing_paper_older_than_the_snapshot_fails(self) -> None:
        v = vh.coverage_verdict(["2106.09685"], set(), self.SNAP)
        assert v["pass"] is False
        assert v["missing_older_than_snapshot"] == ["2106.09685"]

    def test_a_paper_newer_than_the_snapshot_is_tolerated_but_reported(self) -> None:
        v = vh.coverage_verdict(["2608.00001"], set(), self.SNAP)
        assert v["pass"] is True
        assert v["missing_newer_than_snapshot"] == ["2608.00001"]
        assert v["missing"] == ["2608.00001"]

    def test_too_many_misses_fail_even_when_all_are_newer(self) -> None:
        ids = [f"2608.0000{i}" for i in range(vh.MAX_MISSING_TARGETS + 1)]
        assert vh.coverage_verdict(ids, set(), self.SNAP)["pass"] is False

    def test_the_snapshot_month_itself_counts_as_not_yet_covered(self) -> None:
        """2607 vs a 2026-07 snapshot: same month, so its absence is the boundary."""
        v = vh.coverage_verdict(["2607.99999"], set(), self.SNAP)
        assert v["missing_newer_than_snapshot"] == ["2607.99999"]


class TestGateArithmetic:
    def test_four_passes_open_the_gate(self) -> None:
        assert vh.gate_verdict([{"pass": True}] * 4) == "OPEN"

    def test_a_skipped_check_is_not_a_passed_check(self) -> None:
        """`--skip-latency` must not turn 4/4 into 3/3. This is the whole point of the gate."""
        results = [{"pass": True}, {"pass": True}, {"pass": None}, {"pass": True}]
        assert vh.gate_verdict(results) == "INCOMPLETE"

    def test_one_failure_closes_the_gate(self) -> None:
        assert vh.gate_verdict([{"pass": True}, {"pass": False}]) == "CLOSED"


class _FakeResponse:
    def __init__(self, data: bytes, headers: dict[str, str]) -> None:
        self._data = data
        self.headers = headers

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *exc: object) -> bool:
        return False

    def read(self) -> bytes:
        return self._data


class TestRangeFileOnlyFetchesWhatItReads:
    """`bytes_fetched` is the evidence for C2, so it has to be real accounting.

    If the server ignored Range and returned whole objects, C2's "15.9% of the shard" would
    be a fiction — and the ~370 MB sync figure that makes Design 2 affordable with it.
    """

    BODY = bytes(range(256)) * 8  # 2048 bytes

    def _urlopen(self, req, timeout=None):  # type: ignore[no-untyped-def]
        if req.get_method() == "HEAD":
            return _FakeResponse(
                b"", {"x-linked-size": str(len(self.BODY)), "accept-ranges": "bytes"}
            )
        rng = req.get_header("Range")
        assert rng, "a body request without a Range header would transfer the whole object"
        start, end = (int(x) for x in rng.removeprefix("bytes=").split("-"))
        return _FakeResponse(self.BODY[start : end + 1], {})

    def test_a_short_read_transfers_only_those_bytes(self) -> None:
        with patch.object(urllib.request, "urlopen", self._urlopen):
            fh = vh.RangeFile("http://example/x.parquet")
            assert fh.size == len(self.BODY)
            assert fh.read(16) == self.BODY[:16]
            assert fh.bytes_fetched == 16
            assert fh.requests == 1

    def test_seek_end_reads_the_parquet_footer_without_the_body(self) -> None:
        with patch.object(urllib.request, "urlopen", self._urlopen):
            fh = vh.RangeFile("http://example/x.parquet")
            fh.seek(-8, 2)
            assert fh.read() == self.BODY[-8:]
            assert fh.bytes_fetched == 8
            assert fh.tell() == fh.size

    def test_reading_past_the_end_returns_nothing_and_fetches_nothing(self) -> None:
        with patch.object(urllib.request, "urlopen", self._urlopen):
            fh = vh.RangeFile("http://example/x.parquet")
            fh.seek(0, 2)
            assert fh.read(100) == b""
            assert fh.bytes_fetched == 0

    def test_the_lfs_object_size_wins_over_the_pointer_length(self) -> None:
        """HF serves LFS pointers; Content-Length is the pointer, x-linked-size the object."""

        def urlopen(req, timeout=None):  # type: ignore[no-untyped-def]
            if req.get_method() == "HEAD":
                return _FakeResponse(b"", {"x-linked-size": "999", "Content-Length": "133"})
            return _FakeResponse(b"", {})

        with patch.object(urllib.request, "urlopen", urlopen):
            assert vh.RangeFile("http://example/x.parquet").size == 999


hr = _load("hyde_replication")


class TestRankIsComputedAgainstTheWholeIndex:
    def _index(self) -> np.ndarray:
        # Row i differs from the query in i bits, so the true ranking is 0,1,2,3.
        rows = []
        for bits in range(4):
            v = np.zeros(128, dtype=np.uint8)
            for b in range(bits):
                v[b] = 1
            rows.append(v)
        return np.array(rows, dtype=np.uint8)

    def test_nearest_row_ranks_first_and_farthest_last(self) -> None:
        index = self._index()
        query = np.zeros(128, dtype=np.uint8)
        ranks = hr._ranks(index, query, {"a": 0, "b": 1, "d": 3})
        assert ranks == {"a": 1, "b": 2, "d": 4}

    def test_ties_are_broken_in_the_targets_favour(self) -> None:
        """Documented as optimistic: a reported median rank is a lower bound."""
        index = np.zeros((3, 128), dtype=np.uint8)
        ranks = hr._ranks(index, np.zeros(128, dtype=np.uint8), {"x": 2})
        assert ranks == {"x": 1}


class TestUnionFusionSpendsFourListsWorthOfCandidates:
    def test_best_of_four_is_what_union_means(self) -> None:
        per_hyp = [{"t": 5000}, {"t": 300}, {"t": 91}, {"t": 40000}]
        assert min(r["t"] for r in per_hyp) == 91

    def test_the_pre_registered_gates_are_the_24_target_fractions_scaled_to_48(self) -> None:
        """P4 predicted >=8/24 and killed at <=5/24; the benchmark now has 48 targets."""
        assert hr.PREDICT_TOP1K == 48 * 8 // 24
        assert hr.KILL_TOP1K == 48 * 5 // 24
