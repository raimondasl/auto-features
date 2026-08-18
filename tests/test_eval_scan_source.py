"""`--rr-scan-source`: one profile, four stages, and a guard that had gone inert.

`profiler.scan_source` lets the profiler read source files as well as prose. It is
shipped, and **no benchmark arm had ever enabled it** — the class of never-measured
default that already produced +1.00 (gate depth) and +1.24 (digest width). NR-26 pointed
at it directly: whatever benefit lived in its richer arm "tracks the extra *information* —
source code the profiler never reads".

Two failure modes are pinned here, both of which this project has paid for before.

* **One invariant, four call sites.** Collection, ranking, the gate and the rescore each
  build a profile. If the flag reaches some and not others, the arm has retrieval reading
  code while the gate does not — two experiments averaged and reported as one. That is
  C-9b's shape, so the profile has a single home (`case_profile`) and an AST check
  forbids a fifth site.
* **A guard that consults a constant.** `ablate_docs` refused to run when source scanning
  was on, because withholding prose from a *copy* would also withhold code that a real
  thin-docs repository has. It asked `ProfilerConfig().scan_source` — the dataclass
  default, which is False and always has been. The instant the flag existed, the guard
  would have kept passing while the incoherence it was written to stop became reachable
  for the first time.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evals"))

RUNNER = ROOT / "evals" / "run_judge_eval.py"
# The profile builders. `case_profile` is the one home; these are what it wraps, and a
# direct call to either from anywhere else is a second implementation.
RAW_BUILDERS = ("profile_repo", "profile_case_repo")


def _calls_to(names: tuple[str, ...], *, inside: str | None = None) -> list[int]:
    """Line numbers of calls to *names*, optionally restricted to one function body."""
    tree = ast.parse(RUNNER.read_text(encoding="utf-8"))
    scopes = [tree]
    if inside is not None:
        scopes = [
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == inside  # type: ignore[attr-defined]
        ]
        assert scopes, f"{inside} is gone from the runner"
    out = []
    for scope in scopes:
        for node in ast.walk(scope):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in names
            ):
                out.append(node.lineno)
    return out


class TestTheProfileHasOneHome:
    def test_nothing_builds_a_profile_outside_the_helper(self) -> None:
        """A fifth stage that builds its own profile silently opts out of the flag."""
        everywhere = set(_calls_to(RAW_BUILDERS))
        allowed = set(_calls_to(RAW_BUILDERS, inside="case_profile"))
        strays = sorted(everywhere - allowed)
        assert strays == [], (
            f"run_judge_eval.py builds a profile outside case_profile at lines {strays} — "
            "that stage will not see --rr-scan-source"
        )

    def test_the_helper_is_actually_used(self) -> None:
        """The other half of C-9b: a shared function nobody calls is not a fix. Four
        stages need a profile — collection, ranking, the gate, the rescore."""
        assert len(_calls_to(("case_profile",))) >= 4

    def test_the_helper_passes_scan_source_through(self) -> None:
        import run_judge_eval

        src = ast.get_source_segment(
            RUNNER.read_text(encoding="utf-8"),
            next(
                n
                for n in ast.walk(ast.parse(RUNNER.read_text(encoding="utf-8")))
                if isinstance(n, ast.FunctionDef) and n.name == "case_profile"
            ),
        )
        assert src is not None and "scan_source=scan_source" in src
        # `typed_anchors` (P9/P10) is the second profiler flag to reach all four stages,
        # and it fails the same way: reaching collection but not the gate would give an
        # arm whose retrieval knows the repo's techniques while its judge does not.
        assert "typed_anchors=typed_anchors" in src, (
            "case_profile must pass typed_anchors through to ProfilerConfig, or the "
            "--rr-typed-anchors arm profiles differently at different stages"
        )
        assert callable(run_judge_eval.case_profile)

    def test_every_call_site_names_the_flag(self) -> None:
        """Passing the helper but defaulting the argument is the same bug wearing a
        different hat — the stage still reads prose only."""
        text = RUNNER.read_text(encoding="utf-8")
        tree = ast.parse(text)
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)):
                continue
            if node.func.id != "case_profile":
                continue
            kwargs = {k.arg for k in node.keywords}
            assert "scan_source" in kwargs, (
                f"case_profile called without scan_source at line {node.lineno}"
            )


class TestThePoolFingerprintCoversIt:
    def test_scan_source_is_a_pool_flag(self) -> None:
        """It changes the profile, therefore the queries, therefore the pool. Left out,
        a frozen pool would be reused across both arms and the treatment would be
        measured over the control's candidates."""
        import run_judge_eval

        assert "rr_scan_source" in run_judge_eval.POOL_FLAGS

    def test_it_is_not_also_a_ranking_flag(self) -> None:
        """RANKING_FLAGS are the ones deliberately excluded from the fingerprint; listing
        it in both would make the exclusion meaningless."""
        import run_judge_eval

        assert "rr_scan_source" not in run_judge_eval.RANKING_FLAGS

    def test_two_arms_get_different_fingerprints(self) -> None:
        import argparse

        import run_judge_eval

        base = {f: None for f in run_judge_eval.POOL_FLAGS}
        case = {"name": "x", "expected_categories": ["cs.LG"]}
        off = argparse.Namespace(**{**base, "rr_scan_source": False})
        on = argparse.Namespace(**{**base, "rr_scan_source": True})
        assert run_judge_eval.pool_fingerprint(off, case, None) != run_judge_eval.pool_fingerprint(
            on, case, None
        )


class TestTheAblationGuardReadsTheArm:
    """It asked the dataclass default — a constant — so it could never fire."""

    def test_the_guard_refuses_the_incoherent_combination(self, tmp_path: Path) -> None:
        import run_judge_eval

        with pytest.raises(SystemExit, match="thin-docs repo"):
            run_judge_eval.ablate_docs(tmp_path, 300, scan_source=True)

    def test_the_guard_does_not_consult_the_dataclass_default(self) -> None:
        """Mutation-proofing the fix: reverting to `ProfilerConfig().scan_source` makes
        the check pass forever, which is exactly how it shipped inert."""
        import inspect

        import run_judge_eval

        assert "if scan_source:" in inspect.getsource(run_judge_eval.ablate_docs)
        # Checked over the AST, not the text: the function's comment *quotes* the old
        # expression to explain the fix, so a substring check fails on the explanation
        # rather than on the defect. Parsed from the module so indentation is intact.
        fn = next(
            n
            for n in ast.walk(ast.parse(RUNNER.read_text(encoding="utf-8")))
            if isinstance(n, ast.FunctionDef) and n.name == "ablate_docs"
        )
        reads_default = [
            n
            for n in ast.walk(fn)
            if isinstance(n, ast.Attribute)
            and n.attr == "scan_source"
            and isinstance(n.value, ast.Call)
        ]
        assert reads_default == [], "the guard consults a constructed default again"

    def test_the_arm_without_scanning_still_ablates(self, tmp_path: Path) -> None:
        (tmp_path / "README.md").write_text("x" * 5000, encoding="utf-8")
        out = run_ablate(tmp_path)
        assert out.exists() and (out / "README.md").read_text(encoding="utf-8") != "x" * 5000


def run_ablate(repo_dir: Path) -> Path:
    import run_judge_eval

    return run_judge_eval.ablate_docs(repo_dir, 300, scan_source=False)
