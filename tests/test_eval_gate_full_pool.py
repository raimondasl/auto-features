"""Tests for the full-pool gate measurement.

The experiment fetches each repo's real candidate pool from live arXiv and caches it. arXiv
rate-limited the first run after ~15 cases and the collector swallowed it — every query
returned 429, `collect_live_papers` returned an empty list without raising, and the builder
cached seven empty pools. A cached empty pool is worse than a missing one: the next run skips
it as "already built" and those repos vanish from every downstream number while the report
still prints a confident admit rate.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

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


gfp = _load("gate_full_pool")


@pytest.fixture()
def wired(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A one-case benchmark with a clone present, pools written to tmp_path."""
    pools = tmp_path / "pools"
    monkeypatch.setattr(gfp, "POOLS", pools)
    work = tmp_path / "work"
    (work / "cv").mkdir(parents=True)
    monkeypatch.setattr(gfp, "WORK_DIR", work)
    bench = tmp_path / "benchmark.yaml"
    bench.write_text("cases:\n  - name: cv\n    expected_categories: [cs.CV]\n", encoding="utf-8")
    monkeypatch.setattr(gfp, "BENCH", bench)
    monkeypatch.setattr(gfp, "profile_case_repo", lambda *a, **k: object())
    return pools


class TestAnEmptyFetchIsNeverCached:
    def test_a_rate_limited_case_writes_no_file(
        self, wired: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The collector returns [] on a 429 storm without raising. Caching that would make
        the case silently disappear from every later run."""
        with patch.object(gfp, "collect_live_papers", return_value=[]):
            gfp.build_pools(pause=0)
        assert not (wired / "cv.jsonl").exists()

    def test_a_raising_fetch_writes_no_file(
        self, wired: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        with patch.object(gfp, "collect_live_papers", side_effect=RuntimeError("429")):
            gfp.build_pools(pause=0)
        assert not (wired / "cv.jsonl").exists()

    def test_a_real_fetch_is_cached(self, wired: Path) -> None:
        papers = [{"arxiv_id": "2106.09685", "title": "LoRA", "abstract": "we propose"}]
        with patch.object(gfp, "collect_live_papers", return_value=papers):
            gfp.build_pools(pause=0)
        rows = [json.loads(x) for x in (wired / "cv.jsonl").read_text(encoding="utf-8").split("\n")]
        assert rows == [{"id": "2106.09685", "title": "LoRA", "abstract": "we propose"}]


class TestPoolHygiene:
    def test_versions_are_stripped_and_duplicates_dropped(self, wired: Path) -> None:
        """arXiv returns v1/v2 of the same work; counting both would inflate every pool."""
        papers = [
            {"arxiv_id": "2106.09685v1", "title": "a", "abstract": "x"},
            {"arxiv_id": "2106.09685v3", "title": "a", "abstract": "x"},
            {"arxiv_id": "1706.03762", "title": "b", "abstract": "y"},
        ]
        with patch.object(gfp, "collect_live_papers", return_value=papers):
            gfp.build_pools(pause=0)
        ids = [json.loads(x)["id"] for x in (wired / "cv.jsonl").read_text().split("\n")]
        assert ids == ["2106.09685", "1706.03762"]

    def test_a_paper_with_no_abstract_is_dropped(self, wired: Path) -> None:
        """The gate scores a title plus an abstract; an empty abstract is a different,
        much weaker prompt and would depress the admit rate for a harness reason."""
        papers = [
            {"arxiv_id": "1.1", "title": "a", "abstract": ""},
            {"arxiv_id": "2.2", "title": "b", "abstract": "real"},
        ]
        with patch.object(gfp, "collect_live_papers", return_value=papers):
            gfp.build_pools(pause=0)
        ids = [json.loads(x)["id"] for x in (wired / "cv.jsonl").read_text().split("\n")]
        assert ids == ["2.2"]

    def test_load_pools_ignores_an_empty_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Belt and braces: seven such files already existed on disk before the guard."""
        pools = tmp_path / "pools"
        pools.mkdir()
        (pools / "empty.jsonl").write_text("", encoding="utf-8")
        (pools / "real.jsonl").write_text(
            json.dumps({"id": "1.1", "title": "t", "abstract": "a"}), encoding="utf-8"
        )
        monkeypatch.setattr(gfp, "POOLS", pools)
        assert list(gfp.load_pools()) == ["real"]


class TestPreRegisteredBars:
    def test_the_thresholds_are_the_ones_written_down(self) -> None:
        assert (gfp.PREDICT_ADMIT_LO, gfp.PREDICT_ADMIT_HI) == (0.10, 0.25)
        assert gfp.PREDICT_PRECISION == 0.80
        assert gfp.KILL_ADMIT == 0.40
        assert gfp.KILL_PRECISION == 0.60
