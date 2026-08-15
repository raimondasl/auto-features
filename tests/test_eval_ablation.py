"""Tests for the thin-docs ablation arm (`--rr-ablate-docs`) and its report.

The arm's whole validity rests on two properties that are easy to get wrong and silent
when wrong:

* it must withhold **prose only** — a repository with no README still declares its
  dependencies, and an ablation that dropped those would model a repo that does not
  exist and would overstate the damage;
* it must never touch the real clone, which gates the shared verdict cache.

The report has a third: arms must cover the same cases, or the means compare different
repositories.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evals"))

from ablation_report import sign_test, summarise  # noqa: E402
from run_judge_eval import MANIFESTS, ablate_docs  # noqa: E402


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    import run_judge_eval

    monkeypatch.setattr(run_judge_eval, "WORK_DIR", tmp_path / "work")
    src = tmp_path / "myrepo"
    (src / "docs" / "guide").mkdir(parents=True)
    (src / "README.md").write_text("A" * 5000, encoding="utf-8")
    (src / "pyproject.toml").write_text('[project]\nname = "x"\n', encoding="utf-8")
    (src / "requirements.txt").write_text("numpy\n", encoding="utf-8")
    (src / "docs" / "guide" / "intro.md").write_text("deep documentation", encoding="utf-8")
    (src / "main.py").write_text("import numpy\n", encoding="utf-8")
    return src


class TestAblateDocs:
    def test_the_readme_is_capped_at_the_budget(self, repo: Path) -> None:
        out = ablate_docs(repo, 300)
        assert len((out / "README.md").read_text(encoding="utf-8")) == 300

    def test_budget_zero_removes_the_readme_entirely(self, repo: Path) -> None:
        out = ablate_docs(repo, 0)
        assert not (out / "README.md").exists()

    def test_docs_are_withheld_at_every_budget(self, repo: Path) -> None:
        for budget in (0, 300, 1500):
            assert not (ablate_docs(repo, budget) / "docs").exists()

    def test_manifests_survive_because_a_thin_repo_still_has_dependencies(self, repo: Path) -> None:
        out = ablate_docs(repo, 0)
        assert (out / "pyproject.toml").read_text(encoding="utf-8").startswith("[project]")
        assert (out / "requirements.txt").read_text(encoding="utf-8") == "numpy\n"

    def test_every_manifest_the_profiler_parses_is_carried(self, repo: Path) -> None:
        """Guards the list drifting away from profiler._extract_anchors."""
        from reporadar import profiler

        src = Path(profiler.__file__).read_text(encoding="utf-8")
        for manifest in MANIFESTS:
            assert f'"{manifest}"' in src, f"{manifest} is copied but the profiler never reads it"

    def test_the_real_clone_is_never_modified(self, repo: Path) -> None:
        before = {p.name: p.read_bytes() for p in repo.rglob("*") if p.is_file()}
        for budget in (0, 300, 1500):
            ablate_docs(repo, budget)
        after = {p.name: p.read_bytes() for p in repo.rglob("*") if p.is_file()}
        assert before == after
        assert (repo / "docs" / "guide" / "intro.md").is_file()

    def test_rerunning_a_budget_does_not_accumulate_stale_files(self, repo: Path) -> None:
        """A second run at a smaller budget must not leave the first run's README behind."""
        out = ablate_docs(repo, 1500)
        assert len((out / "README.md").read_text(encoding="utf-8")) == 1500
        again = ablate_docs(repo, 0)
        assert again == out.parent / "myrepo-b0"
        assert not (ablate_docs(repo, 0) / "README.md").exists()

    def test_budgets_get_separate_directories(self, repo: Path) -> None:
        assert ablate_docs(repo, 300) != ablate_docs(repo, 1500)

    def test_it_refuses_when_the_profiler_would_read_source(self, repo: Path) -> None:
        """Withholding code as well as prose would model a repo that does not exist.

        This used to simulate the trigger by *replacing the ProfilerConfig class*, because
        the guard consulted `ProfilerConfig().scan_source` — a dataclass default that is
        False and always has been. The real trigger arrived on 2026-08-16 as
        `--rr-scan-source`, at which point the guard would have kept passing while the
        incoherence it exists to stop became reachable for the first time. It now reads the
        arm's own setting, and this test passes that setting instead of faking a global.
        """
        with pytest.raises(SystemExit, match="thin-docs repo"):
            ablate_docs(repo, 300, scan_source=True)

    def test_it_permits_the_arm_that_does_not_scan(self, repo: Path) -> None:
        """The other half: a guard that refused both ways would just disable the flag."""
        assert ablate_docs(repo, 300, scan_source=False).exists()

    def test_the_ablated_profile_is_actually_thinner(self, repo: Path) -> None:
        """The treatment has to be a treatment — an ablation that changes nothing is a
        null result manufactured by the harness."""
        from reporadar.config import ProfilerConfig
        from reporadar.profiler import profile_repo

        (repo / "README.md").write_text(
            "vector search over embeddings with approximate nearest neighbours. " * 60,
            encoding="utf-8",
        )
        full = profile_repo(repo, profiler_cfg=ProfilerConfig(prose_chars=300))
        thin = profile_repo(ablate_docs(repo, 0), profiler_cfg=ProfilerConfig(prose_chars=300))
        assert len(thin.keywords) < len(full.keywords) or thin.keywords != full.keywords


def arm(**nets: float) -> dict[str, dict[str, Any]]:
    return {
        case: {
            "case": case,
            "pool_size": 12,
            "reporadar_toppicks": {
                "n_returned": abs(int(v)) or 0,
                "n_actionable": max(0, int(v)),
                "net_value@2": v,
            },
        }
        for case, v in nets.items()
    }


class TestReport:
    def test_precision_is_none_when_nothing_was_shown(self) -> None:
        """Not 0.0 — an arm that abstained everywhere showed no wrong papers."""
        s = summarise(arm(a=0.0, b=0.0))
        assert s["precision"] is None
        assert s["abstained"] == 2

    def test_net_negative_cases_are_counted_separately_from_abstentions(self) -> None:
        s = summarise(arm(a=0.0, b=-4.0, c=5.0))
        assert (s["abstained"], s["net_negative"]) == (1, 1)

    def test_mismatched_arms_are_rejected_rather_than_averaged(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import ablation_report

        a, b = tmp_path / "a.json", tmp_path / "b.json"
        a.write_text(json.dumps([arm(x=1.0)["x"], arm(y=1.0)["y"]]), encoding="utf-8")
        b.write_text(json.dumps([arm(x=1.0)["x"]]), encoding="utf-8")
        monkeypatch.setattr(
            sys, "argv", ["prog", f"control={a}", f"thin={b}", "--out", str(tmp_path / "o.json")]
        )
        with pytest.raises(SystemExit, match="same cases"):
            ablation_report.main()

    def test_sign_test_drops_ties_from_n(self) -> None:
        assert sign_test([1.0, -1.0, 0.0, 0.0]) == (1, 1, 2, 1.0)

    def test_a_clean_sweep_is_significant(self) -> None:
        assert sign_test([-1.0] * 8)[3] < 0.01
