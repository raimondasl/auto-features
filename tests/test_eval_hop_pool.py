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


class TestResultFilesMergeRatherThanClobber:
    """A `--case` re-run must not replace a whole-set result file with one row.

    This is not hypothetical and it is the second occurrence. `diagnose_triage.py` named its
    output by repo-context and ignored `--model`, so a Sonnet run was overwritten by a Haiku
    one and its per-case numbers no longer exist. Then `synth_seeds.py` wrote its results
    with a plain overwrite, and a single failing retry of `diffusion` replaced an 11-case run
    with `[]` — and printed a KILL verdict against bars scoped to all 27 targets.

    `build_hop_pool` had the correct pattern (read, update, write) the whole time.
    """

    def test_a_single_case_run_preserves_the_others(self, tmp_path: Path) -> None:
        import json

        ss = _load("synth_seeds")
        out = tmp_path / "synth_seeds.json"
        out.write_text(
            json.dumps(
                [
                    {"case": "crypto", "found": ["x"], "pool": 276},
                    {"case": "systems", "found": [], "pool": 425},
                ]
            ),
            encoding="utf-8",
        )
        with (
            patch.object(ss, "OUT", out),
            patch.object(ss, "resolve_targets", return_value={"crypto": ["x"]}),
            patch.object(ss, "run_case", return_value={"case": "crypto", "found": [], "pool": 5}),
            patch.object(ss.qg, "_load_env"),
            patch.object(sys, "argv", ["synth_seeds.py", "--case", "crypto"]),
        ):
            ss.main()
        kept = {r["case"] for r in json.loads(out.read_text(encoding="utf-8"))}
        assert kept == {"crypto", "systems"}, "a --case run destroyed the other cases' results"


class TestSeedRankingIsActuallySelectable:
    """A loop variable named `rank` shadowed the `rank` parameter, so the citation branch
    was unreachable and an entire experiment silently re-ran its own control.

    It produced byte-identical pools on 10 of 11 cases — which is the only reason it was
    caught. "The variant matched the control exactly" is a bug signature, not a finding.
    """

    def test_the_two_rankings_select_different_seeds(self) -> None:
        ss = _load("synth_seeds")
        # 3 phrases x 3 hits; vote order and citation order are deliberately opposed.
        pages = [["a.1", "b.2", "c.3"], ["a.1", "b.2"], ["a.1"]]
        cites = {"a.1": 0, "b.2": 5, "c.3": 900}
        with (
            patch.object(ss.qg, "arxiv_ids", side_effect=lambda *a, **k: pages.pop(0)),
            patch.object(ss, "citation_counts", return_value=cites),
            patch.object(ss.time, "sleep"),
            patch.object(ss, "SEED_CAP", 2),
        ):
            by_votes, _ = ss.seeds_from_phrases(["p1", "p2", "p3"], "votes")
        pages = [["a.1", "b.2", "c.3"], ["a.1", "b.2"], ["a.1"]]
        with (
            patch.object(ss.qg, "arxiv_ids", side_effect=lambda *a, **k: pages.pop(0)),
            patch.object(ss, "citation_counts", return_value=cites),
            patch.object(ss.time, "sleep"),
            patch.object(ss, "SEED_CAP", 2),
        ):
            by_cites, _ = ss.seeds_from_phrases(["p1", "p2", "p3"], "citations")

        assert by_votes[0] == "a.1", "votes ranking should lead with the most-agreed paper"
        assert by_cites[0] == "c.3", "citation ranking should lead with the biggest hub"
        assert set(by_votes) != set(by_cites), "the rank argument had no effect"
