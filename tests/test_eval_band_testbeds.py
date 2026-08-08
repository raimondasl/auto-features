"""Tests for the frozen testbeds behind the within-band ranking experiments (E1-E5).

The load-bearing piece is Testbed A's *positional* band reconstruction: the judge results
files record per-paper judge scores but NOT per-paper gate scores, so band membership is
recovered from the digest order (a stable sort on (llm_score, score_total) descending)
plus the sweep's returned-counts. Get that wrong by one position and every within-band
metric in every experiment is measured on the wrong papers — so the reconstruction is
pinned both synthetically (known counts -> known bands) and, when the real data files are
present locally, against the run files' own aggregate net values, which were computed
from the true gate scores at run time.
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

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


tb = _load("band_testbeds")


def _entry(case: str, judges: list[int], n3: int, n2: int, n1: int) -> dict:
    def net2(scores: list[int]) -> float:
        good = sum(1 for s in scores if s >= 2)
        return good - 2.0 * (len(scores) - good)

    return {
        "case": case,
        "returned": {
            "reporadar_top10": [
                {"arxiv_id": f"260{i}.0000{i}v1", "title": f"paper {i}", "judge_score": j}
                for i, j in enumerate(judges)
            ]
        },
        "reporadar_toppicks_sweep": {
            "3": {"n_returned": n3, "net_value@2": net2(judges[:n3])},
            "2": {"n_returned": n2, "net_value@2": net2(judges[:n2])},
            "1": {"n_returned": n1, "net_value@2": net2(judges[:n1])},
        },
    }


class TestPositionalReconstruction:
    def test_counts_map_positions_to_bands(self, tmp_path, monkeypatch) -> None:
        entry = _entry("x", judges=[3, 2, 2, 1, 2, 0, 1, 2, 0, 1], n3=2, n2=6, n1=9)
        run = tmp_path / "run.json"
        run.write_text(json.dumps([entry]), encoding="utf-8")
        monkeypatch.setattr(tb, "POOL50", run)
        monkeypatch.setattr(tb, "WORK", tmp_path)  # no pool text on disk -> empty abstracts
        tb.pool_text.cache_clear()
        tb.cached_paper_text.cache_clear()
        band = tb.load_testbed_a()["x"]
        assert [p.gate for p in band.papers] == [3, 3, 2, 2, 2, 2, 1, 1, 1, 0]
        assert len(band.gate3) == 2
        assert len(band.band2) == 4
        assert len(band.admitted) == 6

    def test_empty_band_case(self, tmp_path, monkeypatch) -> None:
        entry = _entry("x", judges=[1, 0, 1, 0, 0, 0, 0, 0, 0, 0], n3=0, n2=0, n1=3)
        run = tmp_path / "run.json"
        run.write_text(json.dumps([entry]), encoding="utf-8")
        monkeypatch.setattr(tb, "POOL50", run)
        monkeypatch.setattr(tb, "WORK", tmp_path)
        tb.pool_text.cache_clear()
        tb.cached_paper_text.cache_clear()
        band = tb.load_testbed_a()["x"]
        assert band.admitted == []
        assert [p.gate for p in band.papers[:4]] == [1, 1, 1, 0]


class TestMetrics:
    def test_auc_separates_and_handles_ties(self) -> None:
        assert tb.auc([0.9, 0.8, 0.2, 0.1], [True, True, False, False]) == 1.0
        assert tb.auc([0.1, 0.2, 0.8, 0.9], [True, True, False, False]) == 0.0
        assert tb.auc([0.5, 0.5, 0.5, 0.5], [True, True, False, False]) == 0.5
        assert math.isnan(tb.auc([0.5, 0.6], [True, True]))

    def test_net2_matches_the_benchmark_metric(self) -> None:
        sys.path.insert(0, str(EVALS))
        from metrics import net_actionable_value

        for judges in ([2, 3, 1, 0], [], [2] * 10, [0, 1]):
            assert tb.net2(list(judges)) == net_actionable_value(list(judges), lam=2.0)

    def test_policy_never_shows_an_unscored_band_paper(self) -> None:
        """A scoring failure must not silently become an admission."""
        band = tb.CaseBand(case="x")
        band.papers = [
            tb.Paper(case="x", id="a", title="", abstract="", judge=3, gate=3),
            tb.Paper(case="x", id="b", title="", abstract="", judge=0, gate=2),
            tb.Paper(case="x", id="c", title="", abstract="", judge=2, gate=2),
        ]
        # b unscored -> not shown; c above threshold -> shown; gate-3 a always shown.
        assert tb.policy_net(band, {"c": 0.9}) == 2.0
        # everything unscored -> only gate-3 shows
        assert tb.policy_net(band, {}) == 1.0

    def test_sign_test_is_symmetric_and_exact(self) -> None:
        assert tb.sign_test([1, 1, 1, 1, 1, 1])["p"] == pytest.approx(2 / 64)
        assert tb.sign_test([1, -1, 1, -1])["p"] == 1.0
        assert tb.sign_test([0, 0])["ties"] == 2

    def test_ece_perfect_and_worst(self) -> None:
        assert tb.ece([0.95] * 10, [True] * 10) < 0.06
        assert tb.ece([0.95] * 10, [False] * 10) > 0.9


exp_select = _load("exp_select")


class TestSelectionParsing:
    def test_parses_labels_and_dedups(self) -> None:
        raw = '{"selected": [{"id": "P03"}, {"id": "P01"}, {"id": "P03"}]}'
        assert exp_select.parse_selected(raw, 5) == [1, 3]

    def test_empty_selection_is_valid(self) -> None:
        assert exp_select.parse_selected('{"selected": []}', 5) == []

    def test_out_of_range_raises(self) -> None:
        with pytest.raises(ValueError):
            exp_select.parse_selected('{"selected": [{"id": "P07"}]}', 5)

    def test_no_json_raises(self) -> None:
        with pytest.raises(ValueError):
            exp_select.parse_selected("I select P01 and P02", 5)


exp_pairwise = _load("exp_pairwise")


class TestBradleyTerry:
    def test_dominant_item_ranks_top(self) -> None:
        wins = {("a", "b"): 2.0, ("a", "c"): 2.0, ("b", "c"): 2.0}
        s = exp_pairwise.bradley_terry(["a", "b", "c"], wins)
        assert s["a"] > s["b"] > s["c"]

    def test_ties_land_between(self) -> None:
        # a beats everyone; b and c split their games -> equal strengths
        wins = {("a", "b"): 2.0, ("a", "c"): 2.0, ("b", "c"): 1.0, ("c", "b"): 1.0}
        s = exp_pairwise.bradley_terry(["a", "b", "c"], wins)
        assert s["b"] == pytest.approx(s["c"], abs=1e-6)


exp_finescale = _load("exp_finescale")


def _tok(token: str, logprob: float, top: list[tuple[str, float]]):  # type: ignore[no-untyped-def]
    return SimpleNamespace(
        token=token,
        logprob=logprob,
        top_logprobs=[SimpleNamespace(token=t, logprob=lp) for t, lp in top],
    )


class TestLogprobExtraction:
    def test_digit_expectation_normalizes_over_digits(self) -> None:
        content = [_tok("7", -0.1, [("7", math.log(0.6)), ("8", math.log(0.3)), ("x", -1.0)])]
        got = exp_finescale._digit_expectation(content)
        assert got is not None
        exp, modal = got
        assert exp == pytest.approx((7 * 0.6 + 8 * 0.3) / 0.9)
        assert modal == pytest.approx(0.6 / 0.9)

    def test_p_true_reads_the_final_answer_token(self) -> None:
        content = [
            _tok("ANSWER", -0.1, [("ANSWER", -0.1)]),
            _tok(":", -0.1, [(":", -0.1)]),
            _tok(" true", -0.2, [(" true", math.log(0.8)), (" false", math.log(0.2))]),
        ]
        assert exp_finescale._p_true(content) == pytest.approx(0.8)

    def test_missing_answer_token_is_none_not_a_guess(self) -> None:
        assert exp_finescale._p_true([_tok("maybe", -0.1, [("maybe", -0.1)])]) is None


exp_features = _load("exp_features")


class TestFeatures:
    def test_age_months_from_new_style_ids(self) -> None:
        assert exp_features.age_months("2601.11557") == 7
        assert exp_features.age_months("2311.00001") == 33

    def test_old_style_ids_yield_none_not_garbage(self) -> None:
        assert exp_features.age_months("1103.3735") == 185  # 2011-03 is fine too
        assert exp_features.age_months("cs/0112017") is None


REAL_DATA = tb.POOL50.is_file() and (tb.WORK / "full_pool").is_dir()


@pytest.mark.skipif(not REAL_DATA, reason="frozen run files not present (gitignored)")
class TestReconstructionAgainstTheRealRun:
    """The run files recorded aggregate net values computed from TRUE gate scores at run
    time; the positional reconstruction must reproduce them exactly, for every case."""

    def test_pool50_shape(self) -> None:
        bands = tb.load_testbed_a()
        assert len(bands) == 22
        assert sum(len(b.papers) for b in bands.values()) == 220
        assert sum(len(b.band2) for b in bands.values()) == 105

    def test_every_case_reproduces_the_sweep_nets(self) -> None:
        bands = tb.load_testbed_a()
        entries = {e["case"]: e for e in json.loads(tb.POOL50.read_text(encoding="utf-8"))}
        for case, band in bands.items():
            sweep = entries[case]["reporadar_toppicks_sweep"]
            assert tb.net2([p.judge for p in band.admitted]) == sweep["2"]["net_value@2"], case
            assert tb.net2([p.judge for p in band.gate3]) == sweep["3"]["net_value@2"], case

    def test_a300_uses_the_reruns_for_db_and_storage(self) -> None:
        bands = tb.load_testbed_a300()
        assert tb.net2([p.judge for p in bands["db"].admitted]) == 10.0
        assert tb.net2([p.judge for p in bands["storage"].admitted]) == 7.0
