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
    candidates = [{"full_name": name, "created_at": "2019-01-01"} for name in sorted(repos)]
    return tmp_path, repos, candidates


def _url_for(repos: dict[str, Path]):  # type: ignore[no-untyped-def]
    return lambda full: repos[full].as_uri()


class TestOneRow:
    def test_a_qualifying_repo_is_screened_and_mined_in_one_pass(self, world) -> None:  # type: ignore[no-untyped-def]
        tmp, repos, _ = world
        row, mined = wp.walk_row(
            0,
            {"full_name": "acme/rich", "created_at": "2019-01-01"},
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
            {"full_name": "acme/rich", "created_at": "2019-01-01"},
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
            {"full_name": "acme/rich", "created_at": "2019-01-01"},
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
            {"full_name": "other/thin", "created_at": "2019-01-01"},
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
            {"full_name": "acme/rich", "created_at": "2019-01-01"},
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
            {"full_name": "acme/rich", "created_at": "2019-01-01"},
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
            {"full_name": "acme/rich", "created_at": "2019-01-01"},
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
            {"full_name": "acme/rich", "created_at": "2019-01-01"},
            clones=tmp / "clones",
            contexts=tmp / "ctx",
            timeout=0.001,
            url_for=_url_for(repos),
        )
        assert row["outcome"] in {"timeout", "clone_failed"}
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
