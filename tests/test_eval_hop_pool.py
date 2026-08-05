"""Tests for the P1 citation-hop pool builder and its failure accounting.

The guards here exist because their absence cost a whole run. `hop()` dropped any chunk
whose requests exhausted their retries with a bare `return`, so keyless throttling produced
a *smaller pool and no error*: a rebuild recovered 10,374 of the known 92,014 candidates
(11%), with two cases at exactly zero, and reported success. Filter thresholds swept over
that pool would have looked excellent and meant nothing.
"""

from __future__ import annotations

import importlib.util
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


ch = _load("diagnose_citation_hop")


def _entry(*arxiv_ids: str) -> dict:
    return {"citations": [{"externalIds": {"ArXiv": a}} for a in arxiv_ids]}


class TestFailedChunksAreCounted:
    def test_a_failed_chunk_is_reported_not_swallowed(self) -> None:
        """The bug: `_s2_batch_post` returns None on exhausted retries and the chunk vanished.

        A caller cannot distinguish "these seeds cite nothing" from "the API refused us"
        unless the refusal is counted, and those two have opposite meanings for every
        number computed downstream.
        """
        with patch.object(ch, "_s2_batch_post", return_value=None), patch.object(ch.time, "sleep"):
            result = ch.hop(["1234.5678", "2345.6789"], "citations")
        assert result.failed_chunks == 1
        assert not result.reached

    def test_a_partial_failure_is_still_reported(self) -> None:
        """One good chunk plus one failed chunk is an undercount, not a smaller answer."""
        calls: list = [[_entry("1111.1111")], None]
        with (
            patch.object(ch, "_s2_batch_post", side_effect=lambda *a, **k: calls.pop(0)),
            patch.object(ch.time, "sleep"),
        ):
            result = ch.hop([f"000{i}.0001" for i in range(8)], "citations")
        assert result.failed_chunks == 1
        assert "1111.1111" in result.reached

    def test_a_clean_hop_reports_zero_failures(self) -> None:
        with (
            patch.object(ch, "_s2_batch_post", return_value=[_entry("1111.1111")]),
            patch.object(ch.time, "sleep"),
        ):
            result = ch.hop(["1234.5678"], "citations")
        assert result.failed_chunks == 0
        assert result.reached["1111.1111"] == 1


class TestCouplingDegree:
    def test_degree_counts_distinct_seeds_not_edges(self) -> None:
        """Two seeds both citing a paper give it degree 2 — that is the coupling signal."""
        with (
            patch.object(
                ch,
                "_s2_batch_post",
                return_value=[_entry("9999.0001"), _entry("9999.0001", "9999.0002")],
            ),
            patch.object(ch.time, "sleep"),
        ):
            result = ch.hop(["1111.1111", "2222.2222"], "citations")
        assert result.reached["9999.0001"] == 2
        assert result.reached["9999.0002"] == 1

    def test_one_seed_citing_a_paper_twice_still_scores_one(self) -> None:
        """Degree is co-citation across the repo's seeds; duplicate edges are not evidence.

        S2 can list the same arXiv id twice under one paper (versioned records). Counting
        edges instead of seeds would let a single seed manufacture a high coupling degree.
        """
        with (
            patch.object(ch, "_s2_batch_post", return_value=[_entry("9999.0001", "9999.0001")]),
            patch.object(ch.time, "sleep"),
        ):
            result = ch.hop(["1111.1111"], "citations")
        assert result.reached["9999.0001"] == 1

    def test_versions_are_stripped_before_counting(self) -> None:
        """`1234.5678v2` and `1234.5678` are one paper, or degrees double-count."""
        with (
            patch.object(ch, "_s2_batch_post", return_value=[_entry("9999.0001v2")]),
            patch.object(ch.time, "sleep"),
        ):
            result = ch.hop(["1111.1111"], "citations")
        assert result.reached["9999.0001"] == 1


class TestBuilderRefusesIncompletePools:
    def test_a_case_with_failed_chunks_is_not_persisted(self, tmp_path: Path) -> None:
        """Refusing beats writing an undercount that a later sweep would treat as truth.

        This is the guard that would have stopped 11%-of-the-pool being swept for filter
        thresholds and reported as a finding.
        """
        bp = _load("build_hop_pool")
        with (
            patch.object(bp, "seeds_for", return_value=["1111.1111"]),
            patch.object(bp, "hop", return_value=ch.HopResult(ch.Counter(), 0, 3)),
            patch.object(bp, "OUT_DIR", tmp_path),
        ):
            assert bp.build_case("cv", ["2201.03545"]) is None
        assert not list(tmp_path.glob("*.jsonl"))

    def test_a_clean_case_is_persisted_with_its_degrees(self, tmp_path: Path) -> None:
        import json

        bp = _load("build_hop_pool")
        reached = ch.Counter({"2201.03545": 2, "9999.0001": 1})
        with (
            patch.object(bp, "seeds_for", return_value=["1111.1111"]),
            patch.object(bp, "hop", return_value=ch.HopResult(reached, 0, 0)),
            patch.object(bp, "OUT_DIR", tmp_path),
            patch.object(bp, "META_CACHE", tmp_path / "_meta.json"),
        ):
            row = bp.build_case("cv", ["2201.03545"], with_metadata=False)
        assert row is not None
        assert row["failed_chunks"] == 0
        assert row["targets_in_pool"] == ["2201.03545"]
        written = [
            json.loads(line)
            for line in (tmp_path / "cv.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        target = next(r for r in written if r["id"] == "2201.03545")
        assert target["is_target"] is True
        # hop() is patched for both directions, so the target scores 2 in each.
        assert target["fwd_degree"] == 2
        assert target["back_degree"] == 2


class TestSweepUsesTheCanonicalTargets:
    def test_hardcoded_targets_match_the_derived_list(self) -> None:
        """The 24 targets must come from the caches, never from a file someone dumped.

        A stray `baseline_ids.json` in the repo root held 28 ids — it had skipped the
        judge-score filter and carried 4 phantom entries for `webdev`, a negative control
        with zero actionable papers. Scoring recall against it would have inflated the
        denominator by 17%.
        """
        dp = _load("diagnose_pool")
        if not (EVALS / "cache" / "baseline" / "cli").is_dir():
            pytest.skip("baseline cache not present (evals/cache is gitignored)")
        for case, hardcoded in ch.TARGETS.items():
            assert sorted(hardcoded) == sorted(dp.actionable_baseline_ids(case)), case


class TestTargetsCoverEveryBenchmarkCase:
    """Cohort 2 exposed a hardcoded list that could not see new cases.

    `build_hop_pool` read `diagnose_citation_hop.TARGETS`, a frozen literal covering the
    nine cohort-1 cases that had targets when 18/24 was measured. Every case added later
    raised KeyError — and ten of them failed to build *silently*, because the command that
    ran them piped output through a `grep` that filtered the traceback away. Two failures
    compounding: a stale hardcoded source, and a filter that hid the crash.
    """

    def test_every_benchmark_case_resolves(self) -> None:
        import yaml

        bp = _load("build_hop_pool")
        if not (EVALS / "cache" / "baseline" / "cli").is_dir():
            pytest.skip("baseline cache not present (evals/cache is gitignored)")
        bench = yaml.safe_load((EVALS / "benchmark.yaml").read_text(encoding="utf-8"))
        names = {c["name"] for c in bench["cases"]}
        resolved = bp.resolve_targets()
        assert names - set(resolved) == set(), "benchmark case missing from resolve_targets()"

    def test_the_frozen_literal_is_a_subset_not_the_source(self) -> None:
        """`TARGETS` may lag the benchmark; it must never contradict the derived list.

        Keeping it is fine — it is the record of what the published 18/24 was measured
        against. Reading it as the source of truth is what broke.
        """
        bp = _load("build_hop_pool")
        if not (EVALS / "cache" / "baseline" / "cli").is_dir():
            pytest.skip("baseline cache not present")
        resolved = bp.resolve_targets()
        for case, frozen in ch.TARGETS.items():
            assert sorted(frozen) == sorted(resolved[case]), case
