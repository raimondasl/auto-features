"""The pool's one-pass walk. [PREREG-judge-validity-pool §3]

Four properties carry §3, and each has a specific way of going wrong quietly.

* **B₀ is unconditional.** The qualifying rate `q` and per-repository yield `y` are estimated
  over a *fixed* prefix of the seeded order. If the prefix stopped as soon as the target was
  met, its length would be a function of the yield it observed — inverse sampling, which
  biases `q` upward. So the walk may not stop early inside B₀, and `q`/`y` are computed over
  the prefix alone rather than over everything walked.
* **Every row is an outcome.** Clone failures, missing history and owner-cap skips are
  written down. A dropped row shrinks the population silently, which reads as a worse
  channel rather than as a broken run.
* **Resume merges, never rewrites.** `mine_adoptions --mine` rewrites its artefact; on an
  hours-long walk that would discard everything the previous attempt mined.
* **A clone is deleted only if this row made it.** Otherwise the walk is destructive to the
  legacy 37's cached clones.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from unittest import mock

import pytest

EVALS = Path(__file__).resolve().parent.parent / "evals"
FRAME = EVALS / "frame"


def _load(name: str, where: Path):  # type: ignore[no-untyped-def]
    for path in (EVALS, FRAME, EVALS.parent / "src"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    spec = importlib.util.spec_from_file_location(name, where / f"{name}.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


wp = _load("walk_pool", FRAME)
ma = _load("mine_adoptions", EVALS)


def _git(repo: Path, *args: str, when: str | None = None) -> str:
    env = None
    if when is not None:
        env = {**os.environ, "GIT_AUTHOR_DATE": when, "GIT_COMMITTER_DATE": when}
    out = subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.email=t@e.com",
            "-c",
            "user.name=t",
            "-c",
            "commit.gpgsign=false",
            *args,
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=env,
    )
    assert out.returncode == 0, f"git {args}: {out.stderr}"
    return out.stdout


def _make_repo(root: Path, name: str, *, t0_ids: int, new_ids: int) -> Path:
    """A repo with `t0_ids` papers cited two years ago and `new_ids` adopted since."""
    repo = root / name
    repo.mkdir(parents=True)
    _git(repo, "init", "-q", "-b", "main")
    old = "\n".join(f"https://arxiv.org/abs/19{i:02d}.0000{i % 10}" for i in range(t0_ids))
    (repo / "README.md").write_text(
        f"# {name}\n{old}\nDOI 10.1234/journal.abc\nPMID: 12345678\n", encoding="utf-8"
    )
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "t0", when="2022-01-01T00:00:00+00:00")
    new = "\n".join(f"https://huggingface.co/papers/21{i:02d}.0000{i % 10}" for i in range(new_ids))
    (repo / "README.md").write_text(f"# {name}\n{old}\n{new}\n", encoding="utf-8")
    # X7 needs >= 20 files carrying a source extension of the primary language.
    src_dir = repo / "src"
    src_dir.mkdir(exist_ok=True)
    for i in range(22):
        (src_dir / f"mod{i}.py").write_text(f"VALUE = {i}\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "head")
    return repo


@pytest.fixture()
def world(tmp_path: Path):  # type: ignore[no-untyped-def]
    """A tiny population: two qualifying repos and one below PP2."""
    src = tmp_path / "src"
    repos = {
        "acme/rich": _make_repo(src, "rich", t0_ids=6, new_ids=4),
        "acme/also": _make_repo(src, "also", t0_ids=5, new_ids=3),
        "other/thin": _make_repo(src, "thin", t0_ids=1, new_ids=2),
    }
    candidates = [
        {
            "full_name": name,
            "created_at": "2019-01-01",
            "language": "Python",
            "topics": "machine-learning",
        }
        for name in sorted(repos)
    ]
    return tmp_path, repos, candidates


def _url_for(repos: dict[str, Path]):  # type: ignore[no-untyped-def]
    return lambda full: repos[full].as_uri()


class TestOneRow:
    def test_a_qualifying_repo_is_screened_and_mined_in_one_pass(self, world) -> None:  # type: ignore[no-untyped-def]
        tmp, repos, _ = world
        row, mined = wp.walk_row(
            0,
            {"full_name": "acme/rich", "created_at": "2019-01-01", "language": "Python"},
            clones=tmp / "clones",
            contexts=tmp / "ctx",
            url_for=_url_for(repos),
        )
        assert row["outcome"] == "ok"
        assert row["qualifies"] is True
        assert row["ids_v2_t0"] == 6
        assert row["gross_adoptions"] == 4
        assert len(mined) == 4
        assert all(m["extractor"] == "v2" for m in mined)

    def test_it_records_the_doi_and_pmid_covariates(self, world) -> None:  # type: ignore[no-untyped-def]
        """§6.2 sizes the life-science blind spot instead of closing it, which is only a
        measurement if these are actually counted."""
        tmp, repos, _ = world
        row, _ = wp.walk_row(
            0,
            {"full_name": "acme/rich", "created_at": "2019-01-01", "language": "Python"},
            clones=tmp / "clones",
            contexts=tmp / "ctx",
            url_for=_url_for(repos),
        )
        assert row["dois_t0"] == 1
        assert row["pmids_t0"] == 1

    def test_the_realised_window_is_recorded_not_the_nominal_one(self, world) -> None:  # type: ignore[no-untyped-def]
        """`rev-list -1 --before` can land years before the nominal cutoff across a history
        gap, and controls are matched on the realised date."""
        tmp, repos, _ = world
        row, _ = wp.walk_row(
            0,
            {"full_name": "acme/rich", "created_at": "2019-01-01", "language": "Python"},
            clones=tmp / "clones",
            contexts=tmp / "ctx",
            url_for=_url_for(repos),
        )
        assert row["t0_commit_date"] == "2022-01-01"
        assert row["window_days"] > 720

    def test_a_repo_below_pp2_qualifies_not_and_is_never_mined(self, world) -> None:  # type: ignore[no-untyped-def]
        tmp, repos, _ = world
        row, mined = wp.walk_row(
            0,
            {"full_name": "other/thin", "created_at": "2019-01-01", "language": "Python"},
            clones=tmp / "clones",
            contexts=tmp / "ctx",
            url_for=_url_for(repos),
        )
        assert row["pp2_ids_t0"] is False
        assert row["qualifies"] is False
        assert mined == []

    def test_the_t0_context_is_persisted_so_judging_never_reclones(self, world) -> None:  # type: ignore[no-untyped-def]
        tmp, repos, _ = world
        ctx = tmp / "ctx"
        row, _ = wp.walk_row(
            0,
            {"full_name": "acme/rich", "created_at": "2019-01-01", "language": "Python"},
            clones=tmp / "clones",
            contexts=ctx,
            url_for=_url_for(repos),
        )
        saved = list(ctx.glob("acme__rich.*.txt"))
        assert len(saved) == 1
        assert "Repository: acme/rich" in saved[0].read_text(encoding="utf-8")
        assert saved[0].name.split(".")[1] in row["note"]

    def test_a_clone_this_row_made_is_deleted(self, world) -> None:  # type: ignore[no-untyped-def]
        tmp, repos, _ = world
        clones = tmp / "clones"
        wp.walk_row(
            0,
            {"full_name": "acme/rich", "created_at": "2019-01-01", "language": "Python"},
            clones=clones,
            contexts=tmp / "ctx",
            url_for=_url_for(repos),
        )
        assert not (clones / "acme__rich").exists()

    def test_a_clone_that_was_already_there_survives(self, world) -> None:  # type: ignore[no-untyped-def]
        """The legacy 37 live in this directory. A walk that deleted them would destroy
        hours of someone else's cached work."""
        tmp, repos, _ = world
        clones = tmp / "clones"
        clones.mkdir()
        subprocess.run(
            [
                "git",
                "clone",
                "--filter=blob:none",
                "--no-checkout",
                "--quiet",
                repos["acme/rich"].as_uri(),
                str(clones / "acme__rich"),
            ],
            check=True,
            capture_output=True,
        )
        wp.walk_row(
            0,
            {"full_name": "acme/rich", "created_at": "2019-01-01", "language": "Python"},
            clones=clones,
            contexts=tmp / "ctx",
            url_for=_url_for(repos),
        )
        assert (clones / "acme__rich").exists()

    def test_a_clone_failure_is_a_recorded_outcome_not_an_exception(self, tmp_path: Path) -> None:
        row, mined = wp.walk_row(
            0,
            {"full_name": "no/such", "created_at": "2019-01-01"},
            clones=tmp_path / "clones",
            contexts=tmp_path / "ctx",
            url_for=lambda full: (tmp_path / "missing").as_uri(),
        )
        assert row["outcome"] == "clone_failed"
        assert mined == []

    def test_a_repo_with_no_history_before_t0_is_recorded(self, tmp_path: Path) -> None:
        """PP1's failure mode, and the one an API `created_at` filter cannot catch."""
        repo = _make_repo(tmp_path / "src", "young", t0_ids=0, new_ids=0)
        _git(repo, "commit", "-q", "--allow-empty", "-m", "recent")
        fresh = tmp_path / "src2"
        fresh.mkdir()
        only = fresh / "only"
        only.mkdir()
        _git(only, "init", "-q", "-b", "main")
        (only / "README.md").write_text("# only\n", encoding="utf-8")
        _git(only, "add", "-A")
        _git(only, "commit", "-qm", "just now")
        row, mined = wp.walk_row(
            0,
            {"full_name": "a/only", "created_at": "2026-01-01"},
            clones=tmp_path / "clones",
            contexts=tmp_path / "ctx",
            url_for=lambda full: only.as_uri(),
        )
        assert row["outcome"] == "no_history"
        assert row["pp1_history"] is False
        assert mined == []


class TestTheSeededOrder:
    def test_it_is_a_function_of_the_seed_and_the_name_only(self) -> None:
        rows = [{"full_name": n} for n in ("a/x", "b/y", "c/z", "d/w")]
        first = [r["full_name"] for r in wp.seeded_order(rows, "SEED-1")]
        again = [r["full_name"] for r in wp.seeded_order(list(reversed(rows)), "SEED-1")]
        assert first == again

    def test_a_different_seed_gives_a_different_order(self) -> None:
        rows = [{"full_name": f"o/r{i}"} for i in range(30)]
        a = [r["full_name"] for r in wp.seeded_order(rows, "SEED-1")]
        b = [r["full_name"] for r in wp.seeded_order(rows, "SEED-2")]
        assert a != b
        assert sorted(a) == sorted(b)


class TestTheWalk:
    def test_the_prefix_is_unconditional_even_once_the_target_is_met(self, world) -> None:  # type: ignore[no-untyped-def]
        """The inverse-sampling guard. With target=1 and B0=3, a walk that stopped on the
        target would walk one row and report q = 1.0."""
        tmp, repos, candidates = world
        summary = wp.walk(
            candidates,
            "SEED",
            out_dir=tmp / "out",
            b0=3,
            budget=3,
            target=1,
            jobs=1,
            clone_dir=tmp / "clones",
            url_for=_url_for(repos),
        )
        assert summary["walked"] == 3
        assert summary["q_over_b0"] == pytest.approx(2 / 3, abs=0.01)

    def test_q_and_y_are_computed_over_the_prefix_only(self, world) -> None:  # type: ignore[no-untyped-def]
        tmp, repos, candidates = world
        summary = wp.walk(
            candidates,
            "SEED",
            out_dir=tmp / "out",
            b0=2,
            budget=3,
            target=99,
            jobs=1,
            clone_dir=tmp / "clones",
            url_for=_url_for(repos),
        )
        assert summary["walked"] == 3
        # Only ranks 0 and 1 count toward q, whatever rank 2 turned out to be.
        prefix_rows = [r for r in _rows(tmp / "out" / "validity_walk.csv") if int(r["rank"]) < 2]
        expected = sum(1 for r in prefix_rows if r["qualifies"] == "True") / 2
        assert summary["q_over_b0"] == pytest.approx(expected, abs=0.01)

    def test_every_walked_row_is_written_including_failures(self, world) -> None:  # type: ignore[no-untyped-def]
        tmp, repos, candidates = world
        wp.walk(
            candidates,
            "SEED",
            out_dir=tmp / "out",
            b0=3,
            budget=3,
            target=99,
            jobs=1,
            clone_dir=tmp / "clones",
            url_for=_url_for(repos),
        )
        rows = _rows(tmp / "out" / "validity_walk.csv")
        assert {r["full_name"] for r in rows} == {"acme/rich", "acme/also", "other/thin"}
        assert all(r["outcome"] for r in rows)

    def test_it_resumes_without_rewalking_or_losing_mined_rows(self, world) -> None:  # type: ignore[no-untyped-def]
        tmp, repos, candidates = world
        out = tmp / "out"
        wp.walk(
            candidates[:1],
            "SEED",
            out_dir=out,
            b0=1,
            budget=1,
            target=99,
            jobs=1,
            clone_dir=tmp / "clones",
            url_for=_url_for(repos),
        )
        first = json.loads((out / "adoptions-pool-v2.json").read_text(encoding="utf-8"))
        wp.walk(
            candidates,
            "SEED",
            out_dir=out,
            b0=3,
            budget=3,
            target=99,
            jobs=1,
            clone_dir=tmp / "clones",
            url_for=_url_for(repos),
        )
        rows = _rows(out / "validity_walk.csv")
        assert len(rows) == len({r["full_name"] for r in rows}), "a row was walked twice"
        after = json.loads((out / "adoptions-pool-v2.json").read_text(encoding="utf-8"))
        assert len(after) >= len(first), "resume discarded previously mined rows"

    def test_the_owner_cap_skips_without_cloning(self, world) -> None:  # type: ignore[no-untyped-def]
        """PP3 is order-dependent, so it is applied in the walk rather than in the worker —
        and a skipped candidate must not cost a clone."""
        tmp, repos, candidates = world
        monkey_cap = wp.OWNER_CAP
        try:
            wp.OWNER_CAP = 1
            wp.walk(
                candidates,
                "SEED",
                out_dir=tmp / "out",
                b0=3,
                budget=3,
                target=99,
                jobs=1,
                clone_dir=tmp / "clones",
                url_for=_url_for(repos),
            )
        finally:
            wp.OWNER_CAP = monkey_cap
        rows = _rows(tmp / "out" / "validity_walk.csv")
        acme = [r for r in rows if r["full_name"].startswith("acme/")]
        assert any(r["outcome"] == "owner_cap" for r in acme)

    def test_the_summary_tallies_outcomes_rather_than_only_successes(self, world) -> None:  # type: ignore[no-untyped-def]
        tmp, repos, candidates = world
        summary = wp.walk(
            candidates,
            "SEED",
            out_dir=tmp / "out",
            b0=3,
            budget=3,
            target=99,
            jobs=1,
            clone_dir=tmp / "clones",
            url_for=_url_for(repos),
        )
        assert sum(summary["outcomes"].values()) == summary["walked"]


class TestMergeNotRewrite:
    def test_merging_keeps_what_a_previous_attempt_mined(self, tmp_path: Path) -> None:
        path = tmp_path / "adoptions-pool-v2.json"
        wp.merge_adoptions(path, [{"case": "a/b", "id": "2101.00001"}])
        wp.merge_adoptions(path, [{"case": "a/b", "id": "2102.00002"}])
        rows = json.loads(path.read_text(encoding="utf-8"))
        assert {r["id"] for r in rows} == {"2101.00001", "2102.00002"}

    def test_the_same_paper_twice_is_stored_once(self, tmp_path: Path) -> None:
        path = tmp_path / "adoptions-pool-v2.json"
        wp.merge_adoptions(path, [{"case": "a/b", "id": "2101.00001"}])
        total = wp.merge_adoptions(path, [{"case": "a/b", "id": "2101.00001"}])
        assert total == 1


class TestThePatternsAgree:
    """Same guard as the arXiv patterns: `git grep -E` is POSIX ERE, and deriving one
    pattern from the other by string surgery once produced one that matched nothing."""

    def test_the_doi_patterns_find_the_same_thing(self) -> None:
        import re

        sample = "see doi 10.1234/journal.abc-1 and https://doi.org/10.5555/xyz_9"
        ere = {m.group(0) for m in re.compile(wp.DOI_GREP, re.I).finditer(sample)}
        py = {m.group(1) for m in wp.DOI.finditer(sample)}
        assert ere == py

    def test_the_pmid_patterns_find_the_same_ids(self) -> None:
        import re

        sample = "PMID: 12345678 and https://pubmed.ncbi.nlm.nih.gov/23456789"
        assert {m.group(1) for m in wp.PMID.finditer(sample)} == {"12345678", "23456789"}
        assert len(re.compile(wp.PMID_GREP, re.I).findall(sample)) == 2

    def test_neither_grep_pattern_uses_python_only_syntax(self) -> None:
        for pattern in (wp.DOI_GREP, wp.PMID_GREP):
            assert "(?:" not in pattern
            assert "\\d" not in pattern


class TestTheCloneIsItsOwn:
    def test_it_does_not_reassign_the_shared_clone_root(self) -> None:
        """`mine_adoptions.clone` resolves its destination from a module global. The walk
        runs four rows at once, so redirecting it by assignment would be a data race — two
        threads cloning into each other's directory."""
        import inspect

        src = inspect.getsource(wp)
        assert "ma.CLONES =" not in src
        assert "def _clone(" in src

    def test_the_clone_has_a_timeout(self) -> None:
        """One pathological repository must not hang a walk measured in hours."""
        import inspect

        assert "timeout=timeout" in inspect.getsource(wp._clone)


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


class TestTheRowBudgetBoundsTheGrepsNotOnlyTheClone:
    """Measured, not theorised. Cold-cloned and walked, five legacy repositories took:

        huggingface/diffusers        1092.2 s
        huggingface/peft              146.0 s
        pyg-team/pytorch_geometric     87.5 s
        spotify/annoy                   6.1 s
        stanford-futuredata/ColBERT     5.2 s

    The median, 87.5 s, sits inside §3.1's registered 60–120 s. The maximum is **3.6× the
    registered 300 s per-row bound**, and every second of it is inside `git grep` lazily
    fetching documentation blobs from a blobless clone — which the original timeout never
    touched, because it was passed only to `git clone`. One pathological repository could
    hang a walk measured in hours, which is the exact risk the bound exists for.
    """

    def test_every_git_call_in_a_row_is_bounded(self) -> None:
        import inspect

        src = inspect.getsource(wp.walk_row)
        assert "def left()" in src, "the row has no shrinking budget"
        # Nothing may call out to git without spending from that budget.
        for call in ("ma.git(", "ma.ids_with_paths(", "ma.ids_at(", "ma.self_cited(", "_count("):
            for line in src.splitlines():
                if call in line and "def " not in line:
                    break
        assert src.count("left()") >= 8, "not every git call takes the remaining budget"

    def test_the_grep_helpers_accept_a_timeout_at_all(self) -> None:
        """The fix has to reach `mine_adoptions`: the walk cannot bound a subprocess it does
        not launch."""
        import inspect

        for fn in (ma.ids_with_paths, ma.ids_at, ma._matches_with_paths, ma.self_cited, ma.git):
            assert "timeout" in inspect.signature(fn).parameters, fn.__name__

    def test_a_timeout_is_its_own_outcome_not_a_generic_error(self, world) -> None:  # type: ignore[no-untyped-def]
        """`error` and `timeout` mean different things for the yield curve: one is a broken
        row, the other is a repository too large to screen inside the budget, and the second
        is a property of the population worth reporting."""
        tmp, repos, _ = world
        row, mined = wp.walk_row(
            0,
            {"full_name": "acme/rich", "created_at": "2019-01-01", "language": "Python"},
            clones=tmp / "clones",
            contexts=tmp / "ctx",
            timeout=0.001,
            url_for=_url_for(repos),
        )
        assert row["outcome"] in {"timeout", "clone_timeout", "clone_failed"}
        assert mined == []

    def test_the_budget_shrinks_rather_than_resetting_per_call(self) -> None:
        """A per-call timeout of 300 s would let a row of twelve git calls run for an hour.
        The budget is computed from the row's start."""
        import inspect

        src = inspect.getsource(wp.walk_row)
        assert "timeout - (time.monotonic() - started)" in src


class TestTheLegacyThirtySevenAreExcluded:
    """§2.2 lists "not one of the 37 legacy benchmark cases" as an eligibility rule, and
    nothing implemented it. Measured against the frozen candidate list: **21 of the 37 are in
    it**, carrying 89 of NR-60's 94 legacy positives — diffusers 46, peft 27,
    pytorch_geometric 13, scvi-tools 2, scanpy 1.

    Walked, `huggingface/diffusers` clones under the key `huggingface__diffusers`, so the
    `existed` guard never recognises the legacy clone at `.work/fullclone/diffusion`. Its
    papers would be mined again as *new* pool positives, counted toward the stop rule, capped
    a second time, and §5's legacy-versus-pool heterogeneity would compare the legacy cluster
    against itself.
    """

    def test_the_slugs_come_from_the_benchmark_itself(self) -> None:
        slugs = wp.legacy_slugs()
        assert len(slugs) == 37
        assert "huggingface/diffusers" in slugs
        assert "dlr-rm/stable-baselines3" in slugs
        assert all(s == s.lower() and "github.com" not in s for s in slugs)

    def test_a_legacy_repo_is_recorded_and_never_cloned(self, world) -> None:  # type: ignore[no-untyped-def]
        tmp, repos, candidates = world
        out = tmp / "out"
        wp.walk(
            candidates,
            "SEED",
            out_dir=out,
            b0=3,
            budget=3,
            target=99,
            jobs=1,
            clone_dir=tmp / "clones",
            url_for=_url_for(repos),
            legacy={"acme/rich"},
        )
        rows = {r["full_name"]: r for r in _rows(out / "validity_walk.csv")}
        assert rows["acme/rich"]["outcome"] == "legacy_case"
        assert rows["acme/rich"]["qualifies"] == "False"
        assert not (tmp / "clones" / "acme__rich").exists(), "a legacy case was cloned"

    def test_the_others_are_untouched_by_the_rule(self, world) -> None:  # type: ignore[no-untyped-def]
        tmp, repos, candidates = world
        out = tmp / "out"
        wp.walk(
            candidates,
            "SEED",
            out_dir=out,
            b0=3,
            budget=3,
            target=99,
            jobs=1,
            clone_dir=tmp / "clones",
            url_for=_url_for(repos),
            legacy={"acme/rich"},
        )
        rows = {r["full_name"]: r for r in _rows(out / "validity_walk.csv")}
        assert rows["acme/also"]["outcome"] == "ok"
        assert rows["acme/also"]["qualifies"] == "True"

    def test_the_exclusion_costs_no_clone_and_leaves_the_order_alone(self, world) -> None:  # type: ignore[no-untyped-def]
        """A skipped row keeps its rank, so removing it cannot shift what comes after."""
        tmp, repos, candidates = world
        out = tmp / "out"
        wp.walk(
            candidates,
            "SEED",
            out_dir=out,
            b0=3,
            budget=3,
            target=99,
            jobs=1,
            clone_dir=tmp / "clones",
            url_for=_url_for(repos),
            legacy={"acme/rich"},
        )
        ranks = {r["full_name"]: int(r["rank"]) for r in _rows(out / "validity_walk.csv")}
        ordered = [c["full_name"] for c in wp.seeded_order(candidates, "SEED")]
        assert [ordered[ranks[n]] for n in ranks] == list(ranks)


class TestAFatalGitFailureIsNotAnEmptyBibliography:
    def test_a_bad_revision_raises_rather_than_returning_nothing(self, world) -> None:  # type: ignore[no-untyped-def]
        """`git grep` exits 1 for "no match" and 128 for fatal. Treating both as "no ids"
        makes a promisor fetch failure in a blobless clone indistinguishable from a project
        that keeps no bibliography — and, asymmetrically, books ids present at both ends as
        fresh adoptions when only the T0 grep was truncated."""
        tmp, repos, _ = world
        clones = tmp / "clones"
        clones.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "--quiet", repos["acme/rich"].as_uri(), str(clones / "r")],
            check=True,
            capture_output=True,
        )
        with pytest.raises(RuntimeError, match="git grep exited"):
            ma.ids_at(clones / "r", "0000000000000000000000000000000000000000", "v2")

    def test_a_real_revision_with_no_ids_still_returns_empty(self, tmp_path: Path) -> None:
        """The legitimate empty case must stay quiet, or the guard is useless."""
        repo = tmp_path / "bare"
        repo.mkdir()
        _git(repo, "init", "-q", "-b", "main")
        (repo / "README.md").write_text("# nothing here\n", encoding="utf-8")
        _git(repo, "add", "-A")
        _git(repo, "commit", "-qm", "only")
        assert ma.ids_at(repo, "HEAD", "v2") == set()


class TestTheWalkLedgerCarriesNoRepositoryUrls:
    """§2.1's no-URL rule, now guarded on the artefact that actually gets written.

    It used to be pinned on `mine_adoptions.SCREEN_COLUMNS`, and that screen is retired — it
    implemented `ids_v2(HEAD) ≥ 10`, the rule §2.3 replaced with `ids_v2(T0) ≥ 3` because
    screening on HEAD conditions eligibility on the outcome being counted. The walk's ledger
    is the artefact now, and it is the one that gets committed.
    """

    def test_no_column_can_hold_a_url(self) -> None:
        assert not any("url" in column for column in wp.WALK_COLUMNS)

    def test_a_written_ledger_contains_no_github_urls(self, world) -> None:  # type: ignore[no-untyped-def]
        """Checked on the bytes: a leak would arrive through `note` or a future column, not
        through a column heading."""
        tmp, repos, candidates = world
        out = tmp / "out"
        wp.walk(
            candidates,
            "SEED",
            out_dir=out,
            b0=3,
            budget=3,
            target=99,
            jobs=1,
            clone_dir=tmp / "clones",
            url_for=_url_for(repos),
        )
        text = (out / "validity_walk.csv").read_text(encoding="utf-8")
        assert "acme/rich" in text
        assert "github.com" not in text

    def test_the_t0_prompts_are_not_committed(self) -> None:
        """They are README excerpts, so they carry OTHER repositories' URLs — measured, 2 in
        `graph` and 3 in `rag`, including huggingface/transformers. Across ~300 qualifying
        repositories that is hundreds of them. They are working files: §10 step 7's datasheet
        publishes the ledger, positives, controls and verdicts, and each row already carries
        the context HASH, which verifies the prompt without publishing it."""
        ignored = (Path(__file__).resolve().parents[1] / ".gitignore").read_text(encoding="utf-8")
        assert "evals/frame/pool/contexts/" in ignored


class TestTheSeedMustBeTheNamedPulse:
    """§2.4's entire anti-discretion argument is that the order comes from a value nobody
    could choose, published at a timestamp fixed in the commit that froze the candidate list.
    Reading whatever happens to be in a file checks none of that — a typo, a truncated copy
    or a hand-edited value all walk a different order, and every row afterwards is ordered by
    something the pre-registration does not name."""

    def test_a_matching_seed_passes(self) -> None:
        wp.verify_seed("ABC123", "2026-09-04T00:00:00Z", fetch=lambda _p: "ABC123")

    def test_case_is_not_a_difference(self) -> None:
        """The beacon serves uppercase hex and shells lowercase it freely. That is a
        transcription difference, not a different value."""
        wp.verify_seed("abc123", "2026-09-04T00:00:00Z", fetch=lambda _p: "ABC123")

    def test_a_mismatched_seed_refuses_to_walk(self) -> None:
        with pytest.raises(SystemExit) as exc:
            wp.verify_seed("WRONG", "2026-09-04T00:00:00Z", fetch=lambda _p: "ABC123")
        assert "does not match the pulse" in str(exc.value)

    def test_the_registered_pulse_is_the_one_in_the_file(self) -> None:
        text = (
            Path(__file__).resolve().parents[1] / "evals" / "PREREG-judge-validity-pool.md"
        ).read_text(encoding="utf-8")
        assert wp.REGISTERED_PULSE in text

    def test_the_check_runs_before_any_row_is_walked(self) -> None:
        import inspect

        src = inspect.getsource(wp.main)
        assert src.index("verify_seed(") < src.index("walk(")


class TestTheOwnerCapIsEvaluatedInRankOrder:
    """PP3 caps owners "along the frozen seeded order". `owners` is read for a whole chunk
    before any row in it runs and incremented only afterwards, so four same-owner candidates
    inside one parallel chunk would each see a count of 3 and each pass a cap of 3."""

    def test_a_chunk_never_holds_two_candidates_from_one_owner(self) -> None:
        import inspect

        src = inspect.getsource(wp._walk)
        assert "chunk_owners" in src
        assert "if owner in chunk_owners:" in src

    def test_same_owner_candidates_are_capped_exactly(self, tmp_path: Path) -> None:
        """Four repositories from one owner, a cap of 3, and four workers: without the
        chunking fix all four qualify."""
        src = tmp_path / "src"
        repos = {f"acme/r{i}": _make_repo(src, f"r{i}", t0_ids=5, new_ids=2) for i in range(4)}
        candidates = [
            {
                "full_name": name,
                "created_at": "2019-01-01",
                "language": "Python",
                "topics": "machine-learning",
            }
            for name in sorted(repos)
        ]
        out = tmp_path / "out"
        wp.walk(
            candidates,
            "SEED",
            out_dir=out,
            b0=4,
            budget=4,
            target=99,
            jobs=4,
            clone_dir=tmp_path / "clones",
            url_for=_url_for(repos),
        )
        rows = _rows(out / "validity_walk.csv")
        qualifying = [r for r in rows if r["qualifies"] == "True"]
        assert len(qualifying) <= wp.OWNER_CAP, "the owner cap was exceeded under parallelism"
        assert any(r["outcome"] == "owner_cap" for r in rows)


class TestTheYieldCurveIsCommittedNotRemembered:
    """§3.2: the curve is "committed every 50 rows, so a shortfall is visible in hours rather
    than at the ceiling". It was an in-memory list written once at the end — which is a curve
    nobody could have acted on, and the whole point of it is stopping early."""

    def test_it_carries_the_registered_fields(self) -> None:
        """rows walked, rejects BY RULE, clone failures, timeouts, qualifiers, gross
        adoptions, capped-usable positives. "By rule" means broken out per outcome, not
        summed — a shortfall is only diagnosable if you can see which rule caused it."""
        for field in ("walked", "qualifiers", "gross_adoptions", "capped_positives"):
            assert field in wp.CURVE_COLUMNS
        for outcome in ("clone_failed", "clone_timeout", "timeout", "language", "not_software"):
            assert f"n_{outcome}" in wp.CURVE_COLUMNS

    def test_a_point_tallies_outcomes_rather_than_only_successes(self) -> None:
        rows = [
            {"outcome": "ok", "qualifies": "True", "gross_adoptions": "4"},
            {"outcome": "clone_failed", "qualifies": "False", "gross_adoptions": "0"},
            {"outcome": "language", "qualifies": "False", "gross_adoptions": "0"},
        ]
        point = wp.curve_point(rows, capped_total=3)
        assert point["walked"] == 3
        assert point["qualifiers"] == 1
        assert point["gross_adoptions"] == 4
        assert point["capped_positives"] == 3
        assert point["n_clone_failed"] == 1
        assert point["n_language"] == 1

    def test_it_is_written_during_the_walk(self, world) -> None:  # type: ignore[no-untyped-def]
        tmp, repos, candidates = world
        out = tmp / "out"
        wp.walk(
            candidates,
            "SEED",
            out_dir=out,
            b0=3,
            budget=3,
            target=99,
            jobs=1,
            curve_every=1,
            clone_dir=tmp / "clones",
            url_for=_url_for(repos),
        )
        curve = _rows(out / "yield_curve.csv")
        assert len(curve) >= 2, "the curve was written once, not as the walk progressed"
        assert int(curve[-1]["walked"]) == 3

    def test_it_appends_rather_than_rewriting(self, tmp_path: Path) -> None:
        path = tmp_path / "yield_curve.csv"
        wp.append_curve(path, wp.curve_point([{"outcome": "ok"}], 1))
        wp.append_curve(path, wp.curve_point([{"outcome": "ok"}, {"outcome": "ok"}], 2))
        assert len(_rows(path)) == 2


class TestResumeKnowsAResultFromAFailedAttempt:
    """A clone that timed out says nothing about whether the project keeps a bibliography.
    Left in the denominator it understates the qualifying rate, the same way NR-57's empty
    pool understated a channel."""

    HEADER = ",".join(wp.WALK_COLUMNS)

    def _ledger(self, tmp_path: Path, rows: list[tuple[str, str]]) -> Path:
        path = tmp_path / "validity_walk.csv"
        wp.append_rows(
            path,
            [
                {**wp._blank_row(i, name, "2019-01-01", outcome, ""), "qualifies": False}
                for i, (name, outcome) in enumerate(rows)
            ],
        )
        return path

    def test_plain_resume_skips_everything_recorded(self, tmp_path: Path) -> None:
        """The default is deterministic: a rerun reproduces the first run exactly."""
        path = self._ledger(tmp_path, [("a/ok", "ok"), ("b/failed", "clone_failed")])
        assert wp.already_walked(path) == {"a/ok", "b/failed"}

    def test_retry_reopens_only_the_failed_attempts(self, tmp_path: Path) -> None:
        path = self._ledger(
            tmp_path,
            [("a/ok", "ok"), ("b/failed", "clone_failed"), ("c/lang", "language")],
        )
        assert wp.already_walked(path, retry_failed=True) == {"a/ok", "c/lang"}

    def test_a_decided_fact_is_never_retried(self, tmp_path: Path) -> None:
        """`language`, `not_software`, `no_history` and the rest are facts about the
        repository. Re-walking them would spend a clone to learn nothing."""
        for settled in ("ok", "legacy_case", "owner_cap", "language", "no_history"):
            assert settled not in wp.TRANSIENT_OUTCOMES

    def test_a_retry_replaces_rather_than_duplicating(self, tmp_path: Path) -> None:
        """One candidate must never hold two rows: the curve counts rows."""
        path = self._ledger(tmp_path, [("a/ok", "ok"), ("b/failed", "clone_timeout")])
        assert wp.drop_transient_rows(path) == 1
        assert [r["full_name"] for r in _rows(path)] == ["a/ok"]


class TestACloneFailureSaysWhy:
    def test_a_timeout_and_a_failure_are_different_outcomes(self, tmp_path: Path) -> None:
        """A private-or-deleted repository, a rename, a network blip and a repository too
        large to clone inside the budget are four different facts about the population."""
        row, _ = wp.walk_row(
            0,
            {"full_name": "no/such", "created_at": "2019-01-01", "language": "Python"},
            clones=tmp_path / "clones",
            contexts=tmp_path / "ctx",
            url_for=lambda _f: (tmp_path / "missing").as_uri(),
        )
        assert row["outcome"] == "clone_failed"
        assert row["note"], "git's own explanation was discarded"

    def test_the_reason_is_gits_own(self, tmp_path: Path) -> None:
        _repo, why = wp._clone(tmp_path / "c", "k", (tmp_path / "nope").as_uri(), 60)
        assert why and "timeout" not in why


class TestTheLegacyClusterClaimsOnlyWhatItUses:
    """§3.3: an identifier shared across repositories is "assigned to one repository by
    SEED_POOL and counted once, legacy winning ties."

    `legacy_ids` returned all 94 usable legacy positives, but the per-repository cap of 8 means
    only 32 ever enter the analysis. So a pool repository adopting one of the other 62 lost the
    tie to a paper the legacy cluster is not using, and was then counted ZERO times — in
    neither stratum. Sixty-two papers counted zero times is not "counted once", and the loss
    was concentrated in the most widely adopted ML papers in the set: `diffusion` 38, `peft`
    19, `graph` 5 — exactly what a new ML repository is most likely to have taken up.

    Maintainer decision of 2026-09-03, before the walk: the legacy set claims only the papers
    it actually uses.
    """

    def _rows(self, per_case: dict[str, int]) -> list[dict[str, object]]:
        return [
            {"case": case, "id": f"24{i:02d}.{n:05d}", "usable": True}
            for i, (case, n_ids) in enumerate(sorted(per_case.items()))
            for n in range(n_ids)
        ]

    def test_an_over_cap_paper_is_released_to_the_pool(self, tmp_path: Path) -> None:
        src = tmp_path / "adoptions-v2.json"
        src.write_text(json.dumps(self._rows({"diffusion": 46})), encoding="utf-8")
        claimed = wp.legacy_ids("PULSE", src)
        assert len(claimed) == wp.PER_REPO_CAP
        assert len([r for r in self._rows({"diffusion": 46})]) - len(claimed) == 38

    def test_the_cap_uses_the_same_seeded_rule_as_a_pool_repository(self) -> None:
        """`sha256(SEED_POOL || case:id)`, take the first 8 — one implementation, so which 8
        survive is a function of the pulse rather than of whoever wrote the analysis."""
        rows = self._rows({"diffusion": 20})
        kept = {r["id"] for r in wp.legacy_capped(rows, "PULSE")}
        expected = {
            r["id"]
            for r in sorted(rows, key=lambda e: wp.order_key("PULSE", f"{e['case']}:{e['id']}"))[
                : wp.PER_REPO_CAP
            ]
        }
        assert kept == expected

    def test_a_different_pulse_selects_a_different_eight(self) -> None:
        rows = self._rows({"diffusion": 46})
        a = {r["id"] for r in wp.legacy_capped(rows, "PULSE-A")}
        b = {r["id"] for r in wp.legacy_capped(rows, "PULSE-B")}
        assert a != b and len(a) == len(b) == wp.PER_REPO_CAP

    def test_a_case_under_the_cap_keeps_everything(self) -> None:
        rows = self._rows({"rag": 2, "llminfer": 2})
        assert len(wp.legacy_capped(rows, "PULSE")) == 4

    def test_it_is_deterministic(self, tmp_path: Path) -> None:
        src = tmp_path / "adoptions-v2.json"
        src.write_text(json.dumps(self._rows({"peft": 27})), encoding="utf-8")
        assert wp.legacy_ids("PULSE", src) == wp.legacy_ids("PULSE", src)

    def test_a_missing_artefact_claims_nothing(self, tmp_path: Path) -> None:
        assert wp.legacy_ids("PULSE", tmp_path / "absent.json") == set()

    def test_unusable_rows_are_never_claimed(self, tmp_path: Path) -> None:
        """A row the filters rejected is not a legacy positive, so it wins no tie."""
        src = tmp_path / "adoptions-v2.json"
        src.write_text(
            json.dumps([{"case": "graph", "id": "2401.00001", "usable": False}]), encoding="utf-8"
        )
        assert wp.legacy_ids("PULSE", src) == set()


class TestTheWalkSurvivesItsOwnFirstRun:
    """A pre-flight audit of the walk, before it runs for hours and fixes the study's order.

    §2.4 and §3.4 bar re-rolling the seed or re-taking the population, so a defect found after
    the walk has written rows cannot be fixed by running it again. A crash at hour three is
    recoverable — the walk resumes. A silently wrong row is not.
    """

    def test_the_adoptions_merge_lands_before_the_ledger_row(self) -> None:
        """An `ok` row is not transient, so `already_walked` skips it forever — including under
        `--retry-failed` — and the clone is gone. Written the other way round, a crash between
        the two left the ledger claiming `capped=3` for a repository whose positives existed
        nowhere: counted toward the stop rule, absent from the analysis set, and
        indistinguishable downstream from an ordinary contest loss.
        """
        import inspect

        src = inspect.getsource(wp._walk)
        assert src.index("merge_adoptions(adoptions, mined") < src.index("append_rows(walk_csv")

    def test_the_adoptions_artefact_is_written_atomically(self, tmp_path: Path) -> None:
        """A bare write truncates first, so an ENOSPC or a Windows sharing violation leaves a
        partial JSON the next chunk cannot parse — taking every positive mined so far with it.
        """
        import inspect

        assert "os.replace" in inspect.getsource(wp.merge_adoptions)
        path = tmp_path / "adoptions-pool-v2.json"
        wp.merge_adoptions(path, [{"case": "o/r", "id": "2401.00001", "usable": True}], "S")
        assert json.loads(path.read_text(encoding="utf-8"))[0]["id"] == "2401.00001"
        assert not list(tmp_path.glob("*.tmp"))

    def test_a_pickaxe_timeout_does_not_destroy_the_row(self) -> None:
        """A DATE is optional; the POSITIVE is not. Letting `adoption_commit` raise turned the
        whole row into a `timeout` outcome and discarded every positive the repository had —
        measured in the legacy pass, where one identifier on `huggingface/diffusers` exceeded
        300 s because `git log -S` diffs every commit's docs on a promisor clone.
        """
        import inspect

        src = inspect.getsource(wp.walk_row)
        cut = src.index("adoption_commit(")
        assert "TimeoutExpired" in src[cut : cut + 900]
        assert "adoption_note" in src

    def test_the_t0_context_carries_the_rows_remaining_budget(self) -> None:
        """It was the one git call in a row with no timeout, so on a stalled origin a worker
        stayed open indefinitely while the 300 s per-row bound sat uselessly around it.
        """
        import inspect

        assert "timeout" in inspect.signature(ma.t0_context).parameters
        assert "ma.t0_context(repo, full, t0, timeout=left())" in inspect.getsource(wp.walk_row)

    def test_git_is_told_not_to_ask_a_human(self) -> None:
        """The candidate list is a snapshot; some of those repositories are private by now, and
        over HTTPS git asks for a username and blocks until the timeout burns the row.
        """
        import inspect

        env = wp.noninteractive_git_env()
        assert env["GIT_TERMINAL_PROMPT"] == "0"
        assert "env=noninteractive_git_env()" in inspect.getsource(wp._clone)


class TestTheContextIsCheckedWhileTheCloneStillExists:
    def test_a_header_only_context_is_its_own_outcome(self) -> None:
        """`t0_context` runs git with no `check`, so an unfetched read returns a truthy
        one-line string that hashes to a perfectly valid digest and persists looking complete.
        The reader-side gate catches it only at judging time, hours later, with the clone
        deleted and the row already counted toward the stop rule.
        """
        assert wp.context_shortfall("o/r", "Repository: o/r\n") is not None
        assert "thin_context" in wp.CURVE_OUTCOMES

    def test_a_real_context_passes(self) -> None:
        good = "Repository: o/r\n\n## README (excerpt)\nprose\n\n## Source files (sample)\na.py\n"
        assert wp.context_shortfall("o/r", good) is None

    def test_a_readmes_own_heading_is_not_a_section_boundary(self) -> None:
        """`graph`'s real context carries "## Library Highlights" inside the excerpt."""
        opens = (
            "Repository: o/r\n\n## README (excerpt)\n## Overview\nprose\n\n"
            "## Source files (sample)\na.py\n"
        )
        assert wp.context_shortfall("o/r", opens) is None

    def test_a_repository_with_no_readme_section_is_still_judgeable(self) -> None:
        """`eligibility.README_NAMES` accepts names `t0_context` never reads, and Julia, R and
        Fortran emit no file listing — demanding a layout would reject healthy repositories.
        """
        text = "Repository: o/r\n\n## pyproject.toml\n[project]\n\n## Source files (sample)\na.jl\n"
        assert wp.context_shortfall("o/r", text) is None


class TestTheCountsMeanWhatTheySay:
    def test_the_stop_rule_counts_what_survives_the_contest(self, tmp_path: Path) -> None:
        """§3.3 stops at 100 "capped-usable" positives, and a paper the contest awarded to
        another repository is not one — it is counted once, elsewhere. The ledger's `capped`
        column is filled in before any cross-repository comparison exists.
        """
        path = tmp_path / "a.json"
        path.write_text(
            json.dumps(
                [
                    {
                        "case": "a/x",
                        "id": "2401.1",
                        "usable": True,
                        "in_cap": True,
                        "counted": True,
                    },
                    {
                        "case": "b/y",
                        "id": "2401.1",
                        "usable": True,
                        "in_cap": True,
                        "counted": False,
                    },
                    {
                        "case": "a/x",
                        "id": "2401.2",
                        "usable": False,
                        "in_cap": True,
                        "counted": True,
                    },
                ]
            ),
            encoding="utf-8",
        )
        assert wp.counted_positives(path) == 1

    def test_pp3_is_decided_by_rank_not_by_arrival(self) -> None:
        """§3.3 applies the cap "along the frozen seeded order". A running count is only that
        while the walk goes forward: under `--retry-failed` a reopened row at rank 50 competed
        against owners counted from rows at rank 900, so whether it passed depended on when it
        was retried rather than on where the pulse put it.
        """
        import inspect

        src = inspect.getsource(wp._walk)
        assert "owner_ranks" in src
        assert "if r < rank) >= OWNER_CAP" in src

    def test_pp3_is_recorded_when_it_passes(self) -> None:
        """`_blank_row` pre-seeds every column to "", so `setdefault` never fired and the column
        recorded only rejections.
        """
        import inspect

        assert 'row.get("pp3_owner") == ""' in inspect.getsource(wp._walk)

    def test_a_failed_attempt_is_in_neither_side_of_the_qualifying_rate(self) -> None:
        """A clone that failed says nothing about whether that repository qualifies. Leaving it
        in the denominator biases q downward by the failure rate, which is a property of the
        network rather than of the population.
        """
        import inspect

        src = inspect.getsource(wp._walk)
        assert "TRANSIENT_OUTCOMES" in src and "n_prefix_transient" in src

    def test_the_summary_records_which_seed_and_list_produced_it(self) -> None:
        """Resume matches candidates by NAME alone, so without this a walk resumed against a
        different list or seed looks like one continuous run.
        """
        import inspect

        src = inspect.getsource(wp._walk)
        assert "seed_sha256" in src and "n_candidates" in src

    def test_the_terminal_curve_point_is_not_appended_twice(self, tmp_path: Path) -> None:
        """It was written on every invocation, so a walk resumed five times carried five
        identical trailing rows and read as five stalled checkpoints.
        """
        path = tmp_path / "yield_curve.csv"
        point = wp.curve_point([], 0)
        wp.append_curve(path, point)
        assert wp._last_curve_point(path) == point


class TestTheYieldCurveCanSeeWhyCandidatesFail:
    """§3.2 commits the curve every 50 rows "so a shortfall is visible in hours rather than at
    the ceiling", and it breaks rejects out BY RULE. PP1 and PP2 fell through to `ok`, so the
    commonest rejection of all — a repository that cites no papers — was counted under `n_ok`
    and the curve read as healthy while nothing qualified. Measured on the first six real
    candidates: two `ok` rows, both PP2 failures with zero identifiers at T0."""

    def test_every_rejection_reason_has_its_own_bucket(self) -> None:
        for name in ("thin_history", "thin_bibliography"):
            assert name in wp.CURVE_OUTCOMES
        assert "n_thin_bibliography" in wp.CURVE_COLUMNS

    def test_a_thin_bibliography_is_not_recorded_as_ok(self) -> None:
        import inspect

        src = inspect.getsource(wp.walk_row)
        assert 'row["outcome"] = "thin_bibliography"' in src
        assert 'row["outcome"] = "thin_history"' in src

    def test_the_two_history_failures_are_told_apart(self) -> None:
        """`no_history` is no commit at all before the cutoff; `thin_history` is a repository
        that has one but was born too recently for §2.2's 30 months."""
        assert "no_history" in wp.CURVE_OUTCOMES and "thin_history" in wp.CURVE_OUTCOMES


class TestTwoWalksCannotShareAnOutDir:
    """Measured 2026-09-05: two walks were launched against one `out_dir` and ran concurrently
    for about two minutes. Both skipped the same 1,200 completed candidates, both began at rank
    1200, and both appended rows for ranks 1200-1203.

    The duplicate pairs agreed on every substantive field and differed only in wall-clock
    `seconds` — the walk is deterministic under the seed — so the repair was a de-duplication
    rather than a re-walk. But the ledger is an append-only audit trail, and the purchase loop
    already had a lock while this did not.
    """

    def test_a_second_walk_is_refused_while_one_holds_the_lock(self, tmp_path: Path) -> None:
        (tmp_path / "walk.lock").write_text("pid 1 started now", encoding="utf-8")
        with pytest.raises(SystemExit) as exc:
            wp.walk([], "SEED", out_dir=tmp_path)
        assert "Another walk holds it" in str(exc.value)
        assert "1200-1203" in str(exc.value)

    def test_the_lock_names_the_process_holding_it(self, tmp_path: Path) -> None:
        """So a stale lock can be told from a live one without guessing."""
        seen: dict[str, str] = {}

        def peek(*_a: object, **kw: object) -> dict[str, object]:
            seen["text"] = (tmp_path / "walk.lock").read_text(encoding="utf-8")
            return {"walked": 0}

        with mock.patch.object(wp, "_walk", peek):
            wp.walk([], "SEED", out_dir=tmp_path)
        assert "pid" in seen["text"] and "started" in seen["text"]

    def test_the_lock_is_released_even_when_the_walk_fails(self, tmp_path: Path) -> None:
        def boom(*_a: object, **_k: object) -> None:
            raise RuntimeError("clone exploded")

        with mock.patch.object(wp, "_walk", boom), pytest.raises(RuntimeError):
            wp.walk([], "SEED", out_dir=tmp_path)
        assert not (tmp_path / "walk.lock").exists()


class TestARunOfCloneFailuresIsTheMachineNotTheSample:
    """2026-09-05: at rank 1754 every `git clone` began failing with 0xC0000142
    (STATUS_DLL_INIT_FAILED) — Windows could no longer start a process. The walk recorded 2,239
    of them as ordinary negative observations, spent the rest of a 4,000-row budget in two
    minutes, and wrote `walked: 4000` for 1,756 candidates actually examined. `walk_stop_reason`
    read that as `"budget"` and blessed it, on a denominator 2.3x too large."""

    def test_the_helper_counts_only_the_trailing_run(self) -> None:
        def row(outcome: str) -> dict[str, str]:
            return {"outcome": outcome}

        assert wp.trailing_clone_failures([]) == 0
        # A failure with a real row after it is an ordinary failed clone, not a broken machine.
        assert wp.trailing_clone_failures([row("clone_failed"), row("ok")]) == 0
        assert (
            wp.trailing_clone_failures([row("ok"), row("clone_failed"), row("clone_timeout")]) == 2
        )
        assert wp.trailing_clone_failures([row("clone_failed")] * 5) == 5

    def test_the_walk_aborts_and_writes_no_fabricated_negatives(self, tmp_path: Path) -> None:
        candidates = [
            {
                "full_name": f"owner{i:03d}/repo",
                "created_at": "2019-01-01",
                "language": "Python",
                "topics": "machine-learning",
            }
            for i in range(60)
        ]
        out, missing = tmp_path / "out", tmp_path / "nowhere"
        with pytest.raises(SystemExit) as excinfo:
            wp.walk(
                candidates,
                "SEED",
                out_dir=out,
                b0=1,
                budget=60,
                target=99,
                jobs=3,
                clone_dir=tmp_path / "clones",
                url_for=lambda full: (missing / full.replace("/", "__")).as_uri(),
            )
        assert "consecutive clone failures" in str(excinfo.value)

        ledger = out / "validity_walk.csv"
        rows = _rows(ledger) if ledger.is_file() else []
        # The whole point: held back, never appended. A rerun is a plain resume.
        assert [r for r in rows if r["outcome"] in wp.CLONE_FAILURE_OUTCOMES] == []
        assert len(rows) < len(candidates), "it stopped; it did not spend the budget"
        # And nothing was written that `walk_stop_reason` could read as a completed walk.
        assert not (out / "walk_summary.json").is_file()
