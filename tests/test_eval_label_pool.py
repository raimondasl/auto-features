"""Tests for P5's stratified pool labelling.

P5 spends ~$10 of judge calls to answer two questions — how dense each stratum is, and what
the shipped gate admits from the wild. The guards here cover the ways that sample could be
quietly wrong: a stratum that is really the head of a ranking rather than a random draw from
it, a band boundary off by one, or a confidence interval that collapses at k=0 (which is
precisely where the floor stratum is expected to sit).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

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


lp = _load("label_pool")


class TestWilsonInterval:
    def test_zero_successes_still_has_an_upper_bound(self) -> None:
        """A normal-approximation CI is [0, 0] at k=0, which would read as certainty."""
        lo, hi = lp.wilson(0, 60)
        assert lo == 0.0
        assert 0.04 < hi < 0.08

    def test_the_interval_contains_the_point_estimate(self) -> None:
        lo, hi = lp.wilson(5, 60)
        assert lo < 5 / 60 < hi

    def test_all_successes_is_bounded_at_one(self) -> None:
        lo, hi = lp.wilson(30, 30)
        assert hi == 1.0
        assert lo > 0.8

    def test_an_empty_stratum_does_not_divide_by_zero(self) -> None:
        assert lp.wilson(0, 0) == (0.0, 0.0)


class TestFisherExact:
    def test_matches_the_textbook_tea_tasting_value(self) -> None:
        """Fisher's own 2x2 — the only way to know the implementation is not merely plausible."""
        assert lp.fisher_exact(3, 1, 1, 3) == pytest.approx(0.4857, abs=5e-4)

    def test_identical_rows_cannot_be_significant(self) -> None:
        assert lp.fisher_exact(5, 55, 5, 55) == pytest.approx(1.0)

    def test_swapping_the_rows_does_not_change_the_p_value(self) -> None:
        assert lp.fisher_exact(5, 55, 0, 60) == pytest.approx(lp.fisher_exact(0, 60, 5, 55))

    def test_the_pre_registered_effect_is_detectable_at_the_chosen_n(self) -> None:
        """n=100 was chosen because 8% vs 0% is p=0.007 there and only 0.057 at n=60."""
        assert lp.fisher_exact(8, 92, 0, 100) < 0.01
        assert lp.fisher_exact(5, 55, 0, 60) > 0.05
        assert lp.JUDGE_N["hyde-top100"] == 100
        assert lp.JUDGE_N["random-arxiv"] == 100


class TestSeparationIsReportedClauseByClause:
    """A pre-registered criterion collapsed into one boolean can hide its own result.

    Measured: top stratum 58.0%, floor 2.0% — a 29x separation at p < 0.001. The first
    version of this report printed "BELOW PREDICTION", because the floor clause (<=1%) was
    missed by a single paper and `and` swallowed the other two. That is the bug these guard.
    """

    def test_a_missed_floor_does_not_erase_a_met_effect(self) -> None:
        clauses = lp.separation_clauses(0.58, 0.02)
        assert clauses["top stratum >=8%"] is True
        assert clauses["floor <=1%"] is False
        assert clauses["separation >=6x"] is True

    def test_the_score_three_bar_meets_every_clause(self) -> None:
        """9.0% vs 0.0% — the bar the 48 gold targets were actually drawn at."""
        assert all(lp.separation_clauses(0.09, 0.0).values())

    def test_a_zero_floor_does_not_divide_by_zero(self) -> None:
        assert lp.separation_clauses(0.09, 0.0)["separation >=6x"] is True

    def test_no_effect_fails_the_ratio_clause(self) -> None:
        assert lp.separation_clauses(0.02, 0.02)["separation >=6x"] is False


class TestBandBoundaries:
    def _topk(self, tmp_path: Path, n: int) -> None:
        d = tmp_path / "hyde_topk"
        d.mkdir()
        (d / "cv.json").write_text(json.dumps([f"{i:05d}" for i in range(n)]), encoding="utf-8")
        lp.TOPK_DIR = d

    def test_the_three_bands_partition_the_ranking_without_overlap(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(lp, "TOPK_DIR", tmp_path / "hyde_topk")
        self._topk(tmp_path, 10000)
        monkeypatch.setattr(lp, "TOPK_DIR", tmp_path / "hyde_topk")
        bands = lp._hyde_bands()
        assert len(bands["hyde-top100"]) == 100
        assert len(bands["hyde-100-1k"]) == 900
        assert len(bands["hyde-1k-10k"]) == 9000
        seen = [i for rows in bands.values() for _, i in rows]
        assert len(seen) == len(set(seen))

    def test_a_short_ranking_truncates_rather_than_wrapping(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(lp, "TOPK_DIR", tmp_path / "hyde_topk")
        self._topk(tmp_path, 150)
        monkeypatch.setattr(lp, "TOPK_DIR", tmp_path / "hyde_topk")
        bands = lp._hyde_bands()
        assert len(bands["hyde-top100"]) == 100
        assert len(bands["hyde-100-1k"]) == 50
        assert bands["hyde-1k-10k"] == []


class TestHopDegreeBands:
    def test_degree_two_and_above_is_separated_from_the_bulk(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        d = tmp_path / "hop_pool"
        d.mkdir()
        rows = [
            {"id": "a", "fwd_degree": 0, "back_degree": 1},
            {"id": "b", "fwd_degree": 1, "back_degree": 1},  # degree 2 — in NEITHER band
            {"id": "c", "fwd_degree": 2, "back_degree": 1},
            {"id": "e", "fwd_degree": 7, "back_degree": 0},
            {"id": "d"},  # no degrees recorded at all
        ]
        (d / "rl.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
        monkeypatch.setattr(lp, "HOP_DIR", d)
        bands = lp._hop_bands()
        assert sorted(i for _, i in bands["hop-coupling3+"]) == ["c", "e"]
        assert sorted(i for _, i in bands["hop-coupling1"]) == ["a", "d"]

    def test_the_middle_band_is_excluded_rather_than_folded_into_a_neighbour(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Degree 2 belongs to neither stratum. Folding it into one blurs the contrast that
        the whole comparison rests on, and it would do so silently."""
        d = tmp_path / "hop_pool"
        d.mkdir()
        (d / "rl.jsonl").write_text(
            json.dumps({"id": "mid", "fwd_degree": 1, "back_degree": 1}) + "\n", encoding="utf-8"
        )
        monkeypatch.setattr(lp, "HOP_DIR", d)
        bands = lp._hop_bands()
        assert bands["hop-coupling3+"] == []
        assert bands["hop-coupling1"] == []

    def test_both_directions_count_toward_coupling(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Reading only fwd_degree is the bug the dry-run caught: the field is not `degree`,
        so a `.get("degree", 1)` default put every paper in the bottom band and emptied the
        top one. An empty stratum is a silent, plausible-looking result."""
        d = tmp_path / "hop_pool"
        d.mkdir()
        (d / "rl.jsonl").write_text(
            json.dumps({"id": "back-only", "fwd_degree": 0, "back_degree": 4}) + "\n",
            encoding="utf-8",
        )
        monkeypatch.setattr(lp, "HOP_DIR", d)
        assert [i for _, i in lp._hop_bands()["hop-coupling3+"]] == ["back-only"]


class TestTheSampleIsRandomNotTheHeadOfTheRanking:
    """The single most damaging silent error available here.

    `hyde-1k-10k` holds 9,000 papers per repo and the sample takes 200. If those 200 were
    the *first* 200 of the band rather than a random draw, the stratum's measured density
    would be the density of ranks 1,001-1,200 — a number that looks like a filter result and
    is really a ranking result. The judged subsample is then a prefix of that sample, so the
    same mistake would propagate into the base rate that P5 exists to measure.
    """

    def test_the_drawn_sample_is_not_the_ranking_prefix(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import random

        d = tmp_path / "hyde_topk"
        d.mkdir()
        (d / "cv.json").write_text(json.dumps([f"{i:05d}" for i in range(10000)]), encoding="utf-8")
        monkeypatch.setattr(lp, "TOPK_DIR", d)
        monkeypatch.setattr(lp, "HOP_DIR", tmp_path / "absent")
        monkeypatch.setattr(lp, "INDEX_DIR", tmp_path / "absent")
        monkeypatch.setattr(lp, "WORK_DIR", tmp_path)
        (tmp_path / "cv").mkdir()

        sample = lp.build_sample(200, random.Random(lp.SEED))
        drawn = [i for _, i in sample["hyde-1k-10k"]]
        assert len(drawn) == 200
        assert drawn != sorted(drawn), "the sample is in ranking order — it was not shuffled"
        assert max(drawn) > "05000", "the sample never reaches the tail of the band"

    def test_one_huge_repo_cannot_dominate_a_stratum(self) -> None:
        """`graph` is 42,112 of the hop pool's 109,704 rows and took 139 of 200 slots in a
        flat draw. A stratum that is 70% one repo measures that repo."""
        import random

        rows = [("graph", f"g{i}") for i in range(40000)] + [("rl", f"r{i}") for i in range(500)]
        drawn = lp.balanced_draw(rows, 200, random.Random(1))
        counts = Counter(c for c, _ in drawn)
        assert len(drawn) == 200
        assert counts["graph"] == 100
        assert counts["rl"] == 100

    def test_a_case_that_runs_out_does_not_shrink_the_sample(self) -> None:
        """Round-robin must keep drawing from the repos that still have rows, or a stratum
        with one small repo would silently return fewer papers than asked for."""
        import random

        rows = [("big", f"b{i}") for i in range(500)] + [("small", "s0"), ("small", "s1")]
        drawn = lp.balanced_draw(rows, 100, random.Random(1))
        counts = Counter(c for c, _ in drawn)
        assert len(drawn) == 100
        assert counts["small"] == 2
        assert counts["big"] == 98

    def test_the_same_seed_reproduces_the_same_sample(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import random

        d = tmp_path / "hyde_topk"
        d.mkdir()
        (d / "cv.json").write_text(json.dumps([f"{i:05d}" for i in range(5000)]), encoding="utf-8")
        monkeypatch.setattr(lp, "TOPK_DIR", d)
        monkeypatch.setattr(lp, "HOP_DIR", tmp_path / "absent")
        monkeypatch.setattr(lp, "INDEX_DIR", tmp_path / "absent")
        monkeypatch.setattr(lp, "WORK_DIR", tmp_path)
        (tmp_path / "cv").mkdir()

        a = lp.build_sample(50, random.Random(lp.SEED))
        b = lp.build_sample(50, random.Random(lp.SEED))
        assert a == b
