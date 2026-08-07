"""Tests for P8's stated-wants gate arm.

The arm's whole premise is that the issue titles reach the prompt **verbatim** — the failed
`improvement_areas` arm was LLM-inferred and paraphrased, and paraphrase-vocabulary loss was
the supported diagnosis. Anything here that rewrites, truncates aggressively, or summarises a
title would be re-running the experiment that already failed while calling it a new one.

The other guard is the one that nearly cost this run: `diagnose_triage.py` wrote its per-arm
output unconditionally, so a 3-paper smoke run clobbered the 602-paper file.
"""

from __future__ import annotations

import importlib.util
import json
import sys
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


fw = _load("fetch_wants")
dt = _load("diagnose_triage")


class TestRepoSlug:
    def test_a_github_url_becomes_owner_slash_repo(self) -> None:
        assert fw.repo_slug("https://github.com/pallets/flask") == "pallets/flask"
        assert (
            fw.repo_slug("https://github.com/DLR-RM/stable-baselines3/")
            == "DLR-RM/stable-baselines3"
        )

    def test_a_non_github_url_is_refused_rather_than_mangled(self) -> None:
        assert fw.repo_slug("https://gitlab.com/x/y") is None


class TestTheBlockIsVerbatim:
    TITLES = [
        "Supporting PyTorch GPU compatibility on Apple Silicon chips",
        "[Feature Request] RAINBOW",
        "why is this SO slow??",
    ]

    def test_every_title_appears_unchanged(self) -> None:
        block = fw.as_block(self.TITLES)
        for title in self.TITLES:
            assert title in block, "a title was rewritten — that is the paraphrase arm again"

    def test_the_block_says_where_the_text_came_from(self) -> None:
        block = fw.as_block(self.TITLES)
        assert "open issues" in block
        assert "their own words" in block

    def test_an_empty_tracker_produces_no_header_at_all(self) -> None:
        """A `What users are asking for` heading over nothing reads as "this project has no
        open wants", which is a different claim from "nobody has said". `speech` has zero
        titles and is the run's internal control: its prompt must equal the prose300 one."""
        assert fw.as_block([]) == ""

    def test_the_block_does_not_truncate_what_the_cache_holds(self) -> None:
        """The 120-char cap is applied once, at fetch time. Capping again here would shorten
        already-stored titles and make the prompt depend on which path built it."""
        long = "x" * 500
        assert long in fw.as_block([long])

    def test_the_cap_is_applied_where_the_titles_enter_the_cache(self) -> None:
        import subprocess
        from unittest.mock import patch

        completed = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="y" * 500 + "\nshort one\n", stderr=""
        )
        with patch.object(subprocess, "run", return_value=completed):
            titles = fw.fetch("owner/repo")
        assert titles == ["y" * fw.MAX_TITLE_CHARS, "short one"]

    def test_more_than_the_top_n_is_never_returned(self) -> None:
        import subprocess
        from unittest.mock import patch

        completed = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="\n".join(f"t{i}" for i in range(50)), stderr=""
        )
        with patch.object(subprocess, "run", return_value=completed):
            assert len(fw.fetch("owner/repo")) == fw.TOP_N


class TestWantsLookup:
    def test_a_missing_cache_is_an_error_not_an_empty_block(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Silently scoring 602 papers with no wants at all would be an expensive null run
        reported as a negative result."""
        monkeypatch.setattr(dt, "WANTS_CACHE", tmp_path / "absent.json")
        with pytest.raises(SystemExit):
            dt._wants_for("rag")

    def test_a_case_absent_from_the_cache_yields_no_titles(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        path = tmp_path / "repo_wants.json"
        path.write_text(json.dumps({"rag": ["a want"]}), encoding="utf-8")
        monkeypatch.setattr(dt, "WANTS_CACHE", path)
        assert dt._wants_for("rag") == ["a want"]
        assert dt._wants_for("nosuchcase") == []


class TestPartialRunsDoNotClobberTheArm:
    """`--case rag --limit 3` used to overwrite the arm's 602-paper file with 3 rows."""

    def test_a_smoke_run_preserves_the_rows_it_did_not_touch(self, tmp_path: Path) -> None:
        out = tmp_path / "diag_triage_wants.json"
        full = [{"case": "cv", "id": f"{i}", "judge": 1, "triage": 0} for i in range(50)]
        out.write_text(json.dumps(full), encoding="utf-8")

        merged = dt.merge_rows(out, [{"case": "cv", "id": "7", "judge": 1, "triage": 3}])
        assert len(merged) == 50
        assert {r["id"] for r in merged} == {str(i) for i in range(50)}

    def test_the_rerun_row_wins_over_the_stale_one(self, tmp_path: Path) -> None:
        out = tmp_path / "diag_triage_wants.json"
        out.write_text(
            json.dumps([{"case": "cv", "id": "7", "judge": 1, "triage": 0}]), encoding="utf-8"
        )
        merged = dt.merge_rows(out, [{"case": "cv", "id": "7", "judge": 1, "triage": 3}])
        assert merged == [{"case": "cv", "id": "7", "judge": 1, "triage": 3}]

    def test_a_first_run_writes_cleanly(self, tmp_path: Path) -> None:
        out = tmp_path / "diag_triage_wants.json"
        rows = [{"case": "cv", "id": "1", "judge": 0, "triage": 0}]
        assert dt.merge_rows(out, rows) == rows
        assert json.loads(out.read_text(encoding="utf-8")) == rows
