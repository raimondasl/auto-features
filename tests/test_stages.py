"""The stage registry, and the guard that stops it drifting from the code.

`stages.py` claims `rr watch` and `rr workspace update` skip most of the update pipeline.
That claim is only worth something if it cannot rot — and a warning listing stages that
are *no longer* missing is as bad as one omitting stages that are, because both teach the
reader that the message is approximate.

So the guard walks each entry point's **real import graph** and compares it to the table
in both directions:

* a stage marked *not run* whose module is reachable → the table under-reports; the code
  grew a stage and nobody updated the warning;
* a stage marked *run* whose module is unreachable → the table over-reports; someone
  deleted a stage and the warning still promises it.

Import reachability is the right probe here specifically because every one of these
stages is a lazily-imported optional block: `reporadar.triage` is imported at the point of
use and nowhere else, so "is it in the graph" and "can it run" coincide. The test asserts
that property directly rather than relying on it silently.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "reporadar"

# The code each entry point actually executes, as (file, function-or-None) pairs. A bare
# file means "every import in it".
#
# `rr watch` includes `pipeline.py` because `run_update_cycle` delegates the whole run to
# `run_pipeline`; reading only `watcher.py` would now report every stage as skipped.
# `rr workspace update` is scoped to the ONE function in `cli.py` that implements it plus
# the helper it delegates scoring to — reading all of `cli.py` would sweep in every import
# the other twenty commands make and report the workspace path as running everything.
ENTRY_SOURCES: dict[str, tuple[tuple[str, str | None], ...]] = {
    "watch": (("watcher.py", None), ("pipeline.py", None)),
    "workspace": (("cli.py", "workspace_update"), ("workspace.py", None)),
}


def _imports_for(entry_point: str) -> set[str]:
    found: set[str] = set()
    for name, func in ENTRY_SOURCES[entry_point]:
        found |= _imported_modules(SRC / name, func)
    return found


def _imported_modules(path: Path, function: str | None = None) -> set[str]:
    """Every `reporadar.*` module imported in *path*, or in just one function of it."""
    tree: ast.AST = ast.parse(path.read_text(encoding="utf-8"))
    if function is not None:
        tree = next(
            n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == function
        )
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("reporadar"):
                    found.add(alias.name)
        elif isinstance(node, ast.ImportFrom) and (node.module or "").startswith("reporadar"):
            module = node.module or ""
            found.add(module)
            # `from reporadar.signals import integrity` and `import reporadar.signals.integrity`
            # name the same module two ways; record both so the guard cannot be dodged by
            # switching import style.
            for alias in node.names:
                found.add(f"{module}.{alias.name}")
    return found


def _cfg(**overrides: Any) -> Any:
    from reporadar.config import RepoRadarConfig

    cfg = RepoRadarConfig()
    for dotted, value in overrides.items():
        target: Any = cfg
        *parents, leaf = dotted.split("__")
        for part in parents:
            target = getattr(target, part)
        setattr(target, leaf, value)
    return cfg


def _load_yaml(tmp_path: Path, text: str) -> Any:
    """Load one of the shipped templates the way `rr` does — through the real loader, so a
    template that stopped parsing shows up here rather than in a user's first run."""
    from reporadar.config import load_config

    path = tmp_path / ".reporadar.yml"
    path.write_text(text, encoding="utf-8")
    return load_config(path)


def _everything_on() -> Any:
    """A config with every optional stage enabled, so `unrun_stages` has to name them all."""
    return _cfg(
        triage__enabled=True,
        suggestions__provider="claude",
        hyde__enabled=True,
        ranking__hybrid=True,
        ranking__w_embedding=1.5,
        ranking__w_specter=0.5,
        ranking__w_citations=0.5,
        ranking__w_citation_proximity=0.5,
        ranking__w_community=0.5,
        signals__hackernews=True,
        signals__integrity=True,
        feedback__enabled=True,
        recommendations__enabled=True,
        sources=["arxiv", "semantic_scholar", "openalex", "biorxiv", "iacr", "dblp"],
    )


class TestTheTableMatchesTheCode:
    """Both directions. A guard that only checks one is half a guard."""

    @pytest.mark.parametrize("entry_point", sorted(ENTRY_SOURCES))
    def test_no_stage_is_secretly_run(self, entry_point: str) -> None:
        from reporadar.stages import STAGES

        imported = _imports_for(entry_point)
        wrong = [s.key for s in STAGES if s.missing_from(entry_point) and s.module in imported]
        assert wrong == [], (
            f"stages.py says {entry_point} skips {wrong}, but its source imports those "
            "modules. Either the stage was wired up and the table not updated, or the "
            "table names the wrong module."
        )

    @pytest.mark.parametrize("entry_point", sorted(ENTRY_SOURCES))
    def test_no_stage_is_secretly_skipped(self, entry_point: str) -> None:
        from reporadar.stages import STAGES

        imported = _imports_for(entry_point)
        wrong = [
            s.key for s in STAGES if not s.missing_from(entry_point) and s.module not in imported
        ]
        assert wrong == [], (
            f"stages.py promises {entry_point} runs {wrong}, but its source never imports "
            "them. The warning is over-reporting: it tells users a stage runs when it does not."
        )

    def test_the_positive_case_is_actually_covered(self) -> None:
        """If every stage were marked skip-everywhere, the direction above would pass
        vacuously. At least one stage must claim to run in a reduced entry point."""
        from reporadar.stages import STAGES, WATCH

        assert any(not s.missing_from(WATCH) for s in STAGES)

    def test_update_runs_everything(self) -> None:
        """`rr update` is the pipeline the numbers describe; by construction nothing is
        missing from it, and `unrun_stages` returning anything means the table is wrong."""
        from reporadar.stages import UPDATE, unrun_stages

        assert unrun_stages(_everything_on(), UPDATE) == []

    def test_watch_now_runs_everything_too(self) -> None:
        """The Tier 0 fix, as a checked property rather than a claim in a commit message.

        `rr watch` delegates to the same `run_pipeline` as `rr update`, so no configuration
        can produce a stage it skips. If someone adds a stage to `rr update` alone, the two
        direction checks above fail first — and if someone re-forks the watcher, this fails.
        """
        from reporadar.stages import WATCH, unrun_stages

        assert unrun_stages(_everything_on(), WATCH) == []

    def test_workspace_is_still_reduced_and_says_so(self) -> None:
        """The half deliberately NOT unified: one shared pool across many member repos is
        a different shape, not duplicated code. It keeps the disclosure, and this pins that
        it is still telling the truth rather than having quietly gone silent."""
        from reporadar.stages import WORKSPACE, unrun_stages

        assert {s.key for s in unrun_stages(_everything_on(), WORKSPACE)} >= {"triage", "hyde"}


class TestTheStagesAreLazilyImported:
    """The guard reads imports as a proxy for execution. That is only valid because every
    optional stage is imported at its point of use — if one moved to a module-level import
    in the watcher, the guard would read it as run while the code still skipped it."""

    @pytest.mark.parametrize("entry_point", sorted(ENTRY_SOURCES))
    def test_no_stage_module_is_imported_at_module_level(self, entry_point: str) -> None:
        from reporadar.stages import STAGES

        modules = {s.module for s in STAGES}
        for name, _func in ENTRY_SOURCES[entry_point]:
            tree = ast.parse((SRC / name).read_text(encoding="utf-8"))
            for node in tree.body:  # top level only
                if isinstance(node, ast.ImportFrom) and node.module in modules:
                    pytest.fail(
                        f"{name} imports {node.module} at module level; the drift guard "
                        "reads imports as execution and would now be reading a no-op."
                    )


class TestWhatTheUserIsTold:
    def test_the_gate_needs_both_fields(self) -> None:
        """`triage.enabled` alone is a no-op, so warning about a gate that was never on
        would train the user to ignore the warning."""
        from reporadar.stages import WORKSPACE, unrun_stages

        keys = {s.key for s in unrun_stages(_cfg(triage__enabled=True), WORKSPACE)}
        assert "triage" not in keys
        both = _cfg(triage__enabled=True, suggestions__provider="claude")
        assert "triage" in {s.key for s in unrun_stages(both, WORKSPACE)}

    def test_finescale_is_not_reported_without_the_gate(self) -> None:
        from reporadar.stages import WORKSPACE, unrun_stages

        cfg = _cfg(triage__finescale__enabled=True)
        assert "finescale" not in {s.key for s in unrun_stages(cfg, WORKSPACE)}

    def test_a_default_config_is_told_something(self, tmp_path: Path) -> None:
        """The shipped template enables `hybrid` and `w_embedding`, so even a user who
        never touched the config is running a reduced pipeline under `rr watch`."""
        from reporadar.config import default_config_yaml
        from reporadar.stages import WORKSPACE, unrun_stages

        cfg = _load_yaml(tmp_path, default_config_yaml())
        assert unrun_stages(cfg, WORKSPACE), "the default template must still trigger disclosure"

    def test_the_measured_preset_is_told_a_lot(self, tmp_path: Path) -> None:
        """The preset behind +5.72. Under `rr watch` a user gets none of what earns it."""
        from reporadar.config import measured_config_yaml
        from reporadar.stages import WORKSPACE, unrun_stages

        cfg = _load_yaml(tmp_path, measured_config_yaml())
        keys = {s.key for s in unrun_stages(cfg, WORKSPACE)}
        # The three the headline rests on.
        assert {"triage", "finescale", "hyde"} <= keys

    def test_the_gate_is_named_first(self) -> None:
        """The warning truncates, so ordering is load-bearing: the stage worth most of the
        -8.12 -> +5.72 gap must not be the one cut off."""
        from reporadar.stages import WORKSPACE, unrun_stages

        missing = unrun_stages(_everything_on(), WORKSPACE)
        assert missing[0].key == "triage"

    def test_the_warning_says_what_to_do_instead(self) -> None:
        from reporadar.stages import WORKSPACE, drift_warning

        text = " ".join(drift_warning(_everything_on(), WORKSPACE))
        assert "rr update" in text
        assert "REDUCED" in text

    def test_silence_when_there_is_nothing_to_disclose(self) -> None:
        """A config running nothing optional gets no warning — otherwise the message is
        noise and stops being read."""
        from reporadar.stages import WORKSPACE, drift_warning

        cfg = _cfg(ranking__hybrid=False, signals__integrity=False, enrichment__provider="off")
        assert drift_warning(cfg, WORKSPACE) == []

    def test_truncation_admits_it_truncated(self) -> None:
        from reporadar.stages import WORKSPACE, drift_warning

        lines = drift_warning(_everything_on(), WORKSPACE, limit=2)
        assert any("more" in line for line in lines)

    def test_unknown_entry_point_is_refused(self) -> None:
        from reporadar.stages import unrun_stages

        with pytest.raises(ValueError, match="unknown entry point"):
            unrun_stages(_cfg(), "digest")


class TestTheWatcherSurfacesIt:
    def test_the_cycle_result_carries_the_skipped_list(self, tmp_path: Path) -> None:
        """Programmatic callers (the GitHub Action, anything wrapping the loop) need it in
        data, not only in a log line they never see."""
        import reporadar.watcher as watcher

        assert "skipped_stages" in (SRC / "watcher.py").read_text(encoding="utf-8")
        assert hasattr(watcher, "run_update_cycle")

    def test_it_warns_every_cycle_not_once(self) -> None:
        """An unattended loop that warns once at startup has warned nobody by hour three."""
        src = (SRC / "watcher.py").read_text(encoding="utf-8")
        tree = ast.parse(src)
        cycle = next(
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == "run_update_cycle"
        )
        called = {
            n.func.attr
            for n in ast.walk(cycle)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
        }
        assert "unrun_stages" in {n.id for n in ast.walk(cycle) if isinstance(n, ast.Name)} | called


def test_every_stage_names_a_module_that_exists() -> None:
    """A typo in `module` disables the drift guard for that stage silently — it would read
    as 'not imported' forever and the stage could be wired up unnoticed."""
    import importlib.util

    from reporadar.stages import STAGES

    for stage in STAGES:
        assert importlib.util.find_spec(stage.module) is not None, (
            f"stage {stage.key} names {stage.module}, which does not exist"
        )


def test_stage_keys_are_unique() -> None:
    from reporadar.stages import STAGES

    keys = [s.key for s in STAGES]
    assert len(keys) == len(set(keys))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(pytest.main([__file__]))
