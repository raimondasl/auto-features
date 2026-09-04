"""The boundary around what the judge-validity analysis may write, and what it may claim.

`evals/judge_validity_adoption.json` is the published record of NR-56/57. Its computed numbers
are quoted in `evals/RESULTS.md` (two tables), `evals/README.md` and `PLANS.md`. `report()`
overwrote it unconditionally and `report()` is the DEFAULT action of that script — no `--plan`,
no `--judge` falls through to it — so the first bare invocation against a different positive set
would have replaced the published numbers in place, leaving every citation reading as though
nothing had happened.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evals"))

import judge_validity_pool as jvp  # noqa: E402

FROZEN = ROOT / "evals" / "judge_validity_adoption.json"


class TestTheFrozenRecordIsUnwritable:
    def test_write_artifact_refuses_the_published_path_and_names_it(self) -> None:
        with pytest.raises(SystemExit) as exc:
            jvp.write_artifact(FROZEN, {"anything": 1})
        message = str(exc.value)
        assert "published record" in message
        assert "0.143" in message and "0.243" in message
        assert "RESULTS.md" in message

    def test_the_refusal_happens_before_the_file_is_opened(self) -> None:
        """A guard that truncates first and complains second is not a guard."""
        before = FROZEN.read_bytes()
        with pytest.raises(SystemExit):
            jvp.write_artifact(FROZEN, {"anything": 1})
        assert FROZEN.read_bytes() == before

    def test_the_published_record_is_still_the_one_the_citations_quote(self) -> None:
        """Pinned by value, not by hash: a hash literal tells whoever it fails what changed
        only if they already know what it was."""
        record = json.loads(FROZEN.read_text(encoding="utf-8"))
        assert record["n_positives"] == 35
        assert record["n_controls"] == 140
        assert record["judges"]["gpt-5.5"]["gap"] == pytest.approx(0.1428, abs=1e-4)
        assert record["judges"]["claude-sonnet-5"]["gap"] == pytest.approx(0.2428, abs=1e-4)

    def test_a_write_elsewhere_leaves_the_published_record_alone(self, tmp_path: Path) -> None:
        digest = hashlib.sha256(FROZEN.read_bytes()).hexdigest()
        jvp.write_artifact(tmp_path / "out.json", {"n": 1})
        assert hashlib.sha256(FROZEN.read_bytes()).hexdigest() == digest


class TestOutputPathsAreDerived:
    def test_pool_positives_with_the_pool_scheme_are_refused_citing_section_4(self) -> None:
        """§4 registers the arm-neutral scheme for this population, and there is no data for
        the other one: `pool_papers()` reads `.work/pool-cut100/<case>.json`, which exists only
        for the 37 legacy slugs, so every pool positive would draw ZERO controls."""
        with pytest.raises(SystemExit) as exc:
            jvp.artifact_path("pool", "pool")
        assert "§4" in str(exc.value)
        assert "ZERO controls" in str(exc.value)

    def test_the_registered_combinations_resolve_to_distinct_paths(self) -> None:
        paths = {
            jvp.artifact_path("pool", "arxiv-window"),
            jvp.artifact_path("legacy", "arxiv-window"),
            jvp.artifact_path("legacy", "pool"),
        }
        assert len(paths) == 3
        frozen = {p.resolve() for p in jvp.FROZEN_RECORDS}
        assert not {p.resolve() for p in paths} & frozen

    def test_the_nr57_reproduction_writes_somewhere_untracked(self) -> None:
        """A reproduction that overwrites what it reproduces cannot disagree with it, and
        disagreeing is the only reason to run one."""
        path = jvp.artifact_path("legacy", "pool")
        assert ".work" in path.parts and "repro" in path.parts

    def test_an_unknown_source_raises(self) -> None:
        with pytest.raises(SystemExit) as exc:
            jvp.artifact_path("benchmark", "arxiv-window")
        assert "registered sources" in str(exc.value)


class TestNothingUnparseableIsPublished:
    def test_a_non_finite_number_is_refused_rather_than_written(self, tmp_path: Path) -> None:
        """`json.dumps` writes a bare `NaN`, which no parser outside Python accepts, and §10
        step 7 publishes this artefact as a datasheet. `wilson()` returned `(nan, nan)` at
        n = 0, so a judge with no verdicts produced an unparseable file that looked complete."""
        with pytest.raises(ValueError):
            jvp.write_artifact(tmp_path / "out.json", {"ci": [float("nan"), float("nan")]})

    def test_what_is_written_round_trips_through_a_strict_parser(self, tmp_path: Path) -> None:
        out = jvp.write_artifact(tmp_path / "out.json", {"auc": 0.61, "ci": [0.51, 0.70]})

        def reject(token: str) -> float:
            raise ValueError(f"non-standard token {token!r}")

        loaded = json.loads(out.read_text(encoding="utf-8"), parse_constant=reject)
        assert loaded["auc"] == 0.61


class TestAnUnknownArtefactIsNeverOverwritten:
    def test_a_file_without_the_marker_is_refused(self, tmp_path: Path) -> None:
        target = tmp_path / "someone_elses.json"
        target.write_text(json.dumps({"important": True}), encoding="utf-8")
        with pytest.raises(SystemExit) as exc:
            jvp.write_artifact(target, {"n": 1})
        assert "no _artefact" in str(exc.value)
        assert json.loads(target.read_text(encoding="utf-8")) == {"important": True}

    def test_this_studys_own_artefact_is_rewritable(self, tmp_path: Path) -> None:
        target = tmp_path / "ours.json"
        jvp.write_artifact(target, {"n": 1})
        jvp.write_artifact(target, {"n": 2})
        assert json.loads(target.read_text(encoding="utf-8"))["n"] == 2

    def test_every_artefact_carries_the_marker(self, tmp_path: Path) -> None:
        out = jvp.write_artifact(tmp_path / "ours.json", {"n": 1})
        assert json.loads(out.read_text(encoding="utf-8"))["_artefact"] == jvp.ARTEFACT_MARKER


class TestAPartialRunLeavesTheOldArtefact:
    def test_a_failed_serialisation_does_not_truncate_the_previous_file(
        self, tmp_path: Path
    ) -> None:
        target = tmp_path / "ours.json"
        jvp.write_artifact(target, {"auc": 0.61})
        with pytest.raises(ValueError):
            jvp.write_artifact(target, {"auc": float("inf")})
        assert json.loads(target.read_text(encoding="utf-8"))["auc"] == 0.61


class TestThePublishedRecordIsCheckedOnBothSidesOfEveryWrite:
    def test_a_record_moving_mid_run_is_reported(self, tmp_path: Path, monkeypatch) -> None:
        """The path guard catches the direct overwrite. This catches the indirect one — a
        helper resolving a relative path, a fixture pointed at the real tree."""
        record = tmp_path / "published.json"
        record.write_text('{"gap": 0.143}', encoding="utf-8")
        monkeypatch.setattr(jvp, "FROZEN_RECORDS", frozenset({record}))
        before = jvp.frozen_digests()
        record.write_text('{"gap": 0.999}', encoding="utf-8")
        with pytest.raises(SystemExit) as exc:
            jvp.assert_frozen_records_intact(before)
        assert "PUBLISHED RECORD MOVED" in str(exc.value)
        assert "git checkout" in str(exc.value)

    def test_an_untouched_record_passes(self) -> None:
        jvp.assert_frozen_records_intact(jvp.frozen_digests())
