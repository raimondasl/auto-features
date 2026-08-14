"""Tests for `--rr-frozen-pool` and the noise-floor instrument.

Freezing the candidate pool removes the benchmark's largest variance term — two runs of
the identical shipped configuration overlap only 0.50 by Jaccard on the ranked top-10 — but
it does so by making a run **not a live measurement**. Everything here defends the one
property that keeps that honest: **a frozen-pool run must be impossible to mistake for a
live one**, and a frozen pool must be impossible to reuse under settings that would have
changed it.

Both failures are silent by nature and both have precedents in this project: a baseline
cache that outlived its flags, and a verdict cache keyed without the prompt.
"""

from __future__ import annotations

import json
import sys
from argparse import Namespace
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evals"))

from ablation_report import pool_mode  # noqa: E402
from noise_floor import provenance, same_pool  # noqa: E402
from run_judge_eval import (  # noqa: E402
    POOL_FLAGS,
    RANKING_FLAGS,
    load_frozen_pool,
    pool_fingerprint,
    save_frozen_pool,
)

CASE = {"name": "demo", "expected_categories": ["cs.LG"]}


def args(**over: Any) -> Namespace:
    base: dict[str, Any] = {
        "sources": ["arxiv"],
        "rr_pool": 50,
        "rr_rerank": True,
        "rr_all_time": True,
        "rr_hybrid": True,
        "rr_prose_chars": 300,
        "rr_ablate_docs": None,
        "rr_hyde": True,
        "rr_hyde_index": "evals/.work/hyde_index",
        "rr_hyde_hypotheses": 4,
        "rr_hyde_top_k": 100,
        "rr_triage_model": "claude-haiku-4-5",
        "rr_bigrams": "verified",
        "rr_absent_category": "omit",
    }
    base.update(over)
    return Namespace(**base)


class TestFingerprint:
    def test_identical_settings_agree(self) -> None:
        assert pool_fingerprint(args(), CASE, None) == pool_fingerprint(args(), CASE, None)

    @pytest.mark.parametrize(
        "change",
        [
            {"rr_hyde": False},
            {"rr_all_time": False},
            {"rr_prose_chars": 0},
            {"rr_ablate_docs": 300},
            {"rr_hyde_top_k": 50},
            {"rr_hyde_hypotheses": 1},
            {"sources": ["arxiv", "dblp"]},
            {"rr_triage_model": "claude-sonnet-4-5"},
            {"rr_bigrams": "none"},
        ],
    )
    def test_every_collection_setting_changes_it(self, change: dict[str, Any]) -> None:
        assert pool_fingerprint(args(**change), CASE, None) != pool_fingerprint(args(), CASE, None)

    @pytest.mark.parametrize(
        "change",
        [
            {"rr_hybrid": False},
            {"rr_pool": 20},
            {"rr_rerank": False},
            {"rr_absent_category": "zero"},
            {"rr_absent_category": "impute"},
        ],
    )
    def test_ranking_settings_do_NOT_change_it(self, change: dict[str, Any]) -> None:
        """Varying a ranking flag against a frozen pool is what freezing is FOR.

        The pool used to store the ranked top-N, so every ranking experiment collected
        live at the 1.04 floor — the opposite of the flag's purpose, since the pool is the
        dominant variance term and ranking is deterministic given it. v2 stores the
        candidates, and these flags moved out of the fingerprint.
        """
        assert pool_fingerprint(args(**change), CASE, None) == pool_fingerprint(args(), CASE, None)

    def test_a_goal_changes_it_because_a_goal_changes_the_hypotheses(self) -> None:
        """A stated-intent experiment is a RETRIEVAL experiment and must collect live."""
        assert pool_fingerprint(args(), CASE, "reduce write amplification") != pool_fingerprint(
            args(), CASE, None
        )

    def test_the_case_changes_it(self) -> None:
        other = {"name": "other", "expected_categories": ["cs.LG"]}
        assert pool_fingerprint(args(), other, None) != pool_fingerprint(args(), CASE, None)

    def test_downstream_settings_are_deliberately_absent(self) -> None:
        """Varying the gate or the rescore must REUSE the pool — that is the whole point."""
        for downstream in (
            "rr_min_actionable",
            "rr_finescale",
            "rr_finescale_threshold",
            "rr_triage",
            "rr_sweep",
            "baseline",
            "model",
        ):
            assert downstream not in POOL_FLAGS

    def test_ranking_flags_and_pool_flags_are_disjoint(self) -> None:
        """A flag in both would be silently un-varyable against a frozen pool."""
        assert not set(POOL_FLAGS) & set(RANKING_FLAGS)


POOL = [{"arxiv_id": "2401.00001", "title": "A"}, {"arxiv_id": "2401.00002", "title": "B"}]


class TestRoundTrip:
    def test_a_saved_pool_reloads_identically(self, tmp_path: Path) -> None:
        fp = pool_fingerprint(args(), CASE, None)
        save_frozen_pool(tmp_path, "demo", fp, POOL)
        assert load_frozen_pool(tmp_path, "demo", fp) == POOL

    def test_absent_is_none_not_empty(self, tmp_path: Path) -> None:
        """None means 'collect'; [] would mean 'this repo has no candidates' and abstain."""
        assert load_frozen_pool(tmp_path, "demo", "abc") is None

    def test_an_empty_pool_is_never_stored(self, tmp_path: Path) -> None:
        """An empty pool and a failed collection are the same bytes on disk.

        A frozen empty would score 0.0 on every later run that reused it — the mistake
        that cost `db` and `storage` in the 2026-08-07 arm.
        """
        save_frozen_pool(tmp_path, "demo", pool_fingerprint(args(), CASE, None), [])
        assert not list(tmp_path.glob("*.json"))

    def test_a_mismatched_fingerprint_refuses(self, tmp_path: Path) -> None:
        save_frozen_pool(tmp_path, "demo", pool_fingerprint(args(), CASE, None), POOL)
        with pytest.raises(SystemExit, match="different retrieval settings"):
            load_frozen_pool(tmp_path, "demo", pool_fingerprint(args(rr_hyde=False), CASE, None))

    def test_a_v1_pool_is_refused_not_reinterpreted(self, tmp_path: Path) -> None:
        """v1 stored the RANKED top-N; reading it as a candidate pool would hand the
        ranker a list already cut by the settings under test, and every metric would be
        computed over it as though it were everything retrieval found."""
        fp = pool_fingerprint(args(), CASE, None)
        (tmp_path / "demo.json").write_text(
            json.dumps({"case": "demo", "fingerprint": fp, "ranked": [[POOL[0], 0.9]]}),
            encoding="utf-8",
        )
        with pytest.raises(SystemExit, match="format v1"):
            load_frozen_pool(tmp_path, "demo", fp)

    def test_the_stored_record_names_when_it_was_collected(self, tmp_path: Path) -> None:
        save_frozen_pool(tmp_path, "demo", "fp", [({"arxiv_id": "1"}, 1.0)])
        data = json.loads((tmp_path / "demo.json").read_text(encoding="utf-8"))
        assert data["collected_at"] and data["n"] == 1 and data["fingerprint"] == "fp"


def run_with(mode: str | None, fp: str = "abc123def456") -> dict[str, dict[str, Any]]:
    prov = None if mode is None else {"mode": mode, "fingerprint": fp}
    return {"a": {"case": "a", "pool_provenance": prov} if prov else {"case": "a"}}


def multi(mode: str, fps: dict[str, str]) -> dict[str, dict[str, Any]]:
    """A run over several cases — each with its OWN fingerprint, as the harness writes."""
    return {
        c: {"case": c, "pool_provenance": {"mode": mode, "fingerprint": f}} for c, f in fps.items()
    }


class TestMultiCaseProvenance:
    """The bug the single-case tests above could not see.

    A pool fingerprint includes its case name, so a real 25-case frozen run carries 25
    *different* fingerprints. The first version of `provenance` folded mode+fingerprint
    per case and took the set, so every genuine frozen run reported 'mixed' — and two runs
    off different pools then compared cleanly, which is precisely the failure the guard
    exists to prevent. It shipped because every test used one case.
    """

    POOL_A = {"cv": "aaa", "rag": "bbb", "graph": "ccc"}
    POOL_B = {"cv": "xxx", "rag": "yyy", "graph": "zzz"}

    def test_a_multi_case_frozen_run_is_not_mixed(self) -> None:
        assert provenance(multi("frozen", self.POOL_A)).startswith("frozen:")

    def test_the_same_pool_gives_the_same_digest(self) -> None:
        assert provenance(multi("frozen", self.POOL_A)) == provenance(
            multi("frozen", dict(reversed(list(self.POOL_A.items()))))
        )

    def test_different_pools_give_different_digests(self) -> None:
        assert provenance(multi("frozen", self.POOL_A)) != provenance(multi("frozen", self.POOL_B))

    def test_same_pool_reports_the_cases_that_differ(self) -> None:
        assert same_pool(multi("frozen", self.POOL_A), multi("frozen", self.POOL_A)) == []
        assert same_pool(multi("frozen", self.POOL_A), multi("frozen", self.POOL_B)) == [
            "cv",
            "graph",
            "rag",
        ]

    def test_a_partial_overlap_is_judged_on_shared_cases_only(self) -> None:
        """A 25-case run and a 22-case run from one pool stay comparable on the 22."""
        subset = {k: v for k, v in self.POOL_A.items() if k != "graph"}
        assert same_pool(multi("frozen", self.POOL_A), multi("frozen", subset)) == []

    def test_a_run_whose_cases_disagree_on_mode_is_mixed(self) -> None:
        run = multi("frozen", self.POOL_A)
        run["rag"]["pool_provenance"]["mode"] = "live"
        assert provenance(run) == "mixed"


class TestProvenanceIsUnmistakable:
    def test_live_and_frozen_are_distinguishable(self) -> None:
        assert provenance(run_with("live")) == "live"
        assert provenance(run_with("frozen")).startswith("frozen:")
        assert pool_mode(run_with("live")) == "live"
        assert pool_mode(run_with("frozen")).startswith("frozen:")

    def test_runs_predating_the_flag_read_as_unlabelled_not_live(self) -> None:
        """They *were* live, but claiming so from absent data is how a wrong assumption
        becomes a published number."""
        assert provenance(run_with(None)) == "unlabelled"
        assert pool_mode(run_with(None)) == "unlabelled"

    def test_two_frozen_runs_from_different_pools_do_not_match(self) -> None:
        assert provenance(run_with("frozen", "aaaaaaaaaaaa")) != provenance(
            run_with("frozen", "bbbbbbbbbbbb")
        )

    def test_a_run_that_seeded_the_pool_is_not_labelled_frozen(self) -> None:
        """The seeding run collected LIVE; calling it frozen would misdate its candidates."""
        assert provenance(run_with("frozen-seeded")) == "frozen-seeded"


class TestReportsRefuseToMixModes:
    def _write(self, path: Path, mode: str | None) -> None:
        rec: dict[str, Any] = {
            "case": "a",
            "pool_size": 10,
            "n_actionable_in_pool": 5,
            "reporadar_toppicks": {"n_returned": 2, "n_actionable": 2, "net_value@2": 2.0},
            "reporadar_top10": {"n_returned": 10, "n_actionable": 5, "net_value@2": 0.0},
            "returned": {"reporadar_toppicks": [], "reporadar_top10": []},
        }
        if mode:
            rec["pool_provenance"] = {"mode": mode, "fingerprint": "abc123def456"}
        path.write_text(json.dumps([rec]), encoding="utf-8")

    def test_ablation_report_refuses(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        import ablation_report

        a, b = tmp_path / "a.json", tmp_path / "b.json"
        self._write(a, "live")
        self._write(b, "frozen")
        monkeypatch.setattr(
            sys, "argv", ["p", f"ctl={a}", f"arm={b}", "--out", str(tmp_path / "o.json")]
        )
        with pytest.raises(SystemExit, match="different pool provenance"):
            ablation_report.main()

    def test_noise_floor_refuses(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        import noise_floor

        a, b = tmp_path / "a.json", tmp_path / "b.json"
        self._write(a, "live")
        self._write(b, "frozen")
        monkeypatch.setattr(sys, "argv", ["p", str(a), str(b)])
        with pytest.raises(SystemExit, match="refusing to compare"):
            noise_floor.main()
