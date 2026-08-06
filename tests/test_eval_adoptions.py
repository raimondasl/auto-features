"""Tests for P6's git-mined adoption ground truth.

P6 is the one measurement in the plan that does not depend on a model: an arXiv id present
in a repo's docs at HEAD and absent 24 months earlier is a technique the project actually
took up. That only holds if the two reads really are at two different revisions, so these
tests build a real git repository with a real history and check the mechanism end to end.

The failure this guards hardest against: reading the repo at HEAD while calling it T0. The
judge would then be shown the post-adoption repository and asked whether it should adopt what
it already has — a validity test that quietly measures nothing.
"""

from __future__ import annotations

import importlib.util
import subprocess
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


ma = _load("mine_adoptions")


def _git(repo: Path, *args: str) -> str:
    out = subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.email=t@example.com",
            "-c",
            "user.name=t",
            "-c",
            "commit.gpgsign=false",
            *args,
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert out.returncode == 0, f"git {args}: {out.stderr}"
    return out.stdout


@pytest.fixture()
def repo_with_history(tmp_path: Path) -> tuple[Path, str, str]:
    """A repo whose README cites one paper at first, then three, plus a self-citation."""
    repo = tmp_path / "proj"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")

    (repo / "README.md").write_text(
        "# proj\nBuilt on arXiv:1706.03762 for attention.\n", encoding="utf-8"
    )
    (repo / "setup.py").write_text("# v1\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "first")
    old = _git(repo, "rev-parse", "HEAD").strip()

    (repo / "README.md").write_text(
        "# proj rewritten\n"
        "Built on arXiv:1706.03762 for attention.\n"
        "Now also https://arxiv.org/abs/2106.09685 and arXiv:1910.10683.\n"
        "\n## Citation\nIf you use this, cite arXiv:2401.00001.\n",
        encoding="utf-8",
    )
    (repo / "CITATION.cff").write_text("preferred-citation: arXiv:2402.00002\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "second")
    head = _git(repo, "rev-parse", "HEAD").strip()
    return repo, old, head


class TestIdsAreReadAtTheRevisionAsked:
    def test_head_sees_every_id(self, repo_with_history: tuple[Path, str, str]) -> None:
        repo, _old, head = repo_with_history
        assert ma.ids_at(repo, head) == {
            "1706.03762",
            "2106.09685",
            "1910.10683",
            "2401.00001",
            "2402.00002",
        }

    def test_the_older_revision_does_not_see_later_ids(
        self, repo_with_history: tuple[Path, str, str]
    ) -> None:
        """If this read HEAD, every adoption would vanish and P6 would report a clean zero."""
        repo, old, _head = repo_with_history
        assert ma.ids_at(repo, old) == {"1706.03762"}

    def test_the_adoption_set_is_the_difference(
        self, repo_with_history: tuple[Path, str, str]
    ) -> None:
        repo, old, head = repo_with_history
        adopted = ma.ids_at(repo, head) - ma.ids_at(repo, old)
        assert "1706.03762" not in adopted, "a paper cited from the start was not adopted"
        assert {"2106.09685", "1910.10683"} <= adopted


class TestSelfCitationsAreExcluded:
    def test_a_citation_file_and_a_citation_heading_both_count(
        self, repo_with_history: tuple[Path, str, str]
    ) -> None:
        """A reference implementation always cites its own paper, and did not adopt it."""
        repo, _old, head = repo_with_history
        assert ma.self_cited(repo, head) == {"2401.00001", "2402.00002"}

    def test_ordinary_references_are_not_treated_as_self_citations(
        self, repo_with_history: tuple[Path, str, str]
    ) -> None:
        repo, _old, head = repo_with_history
        selfcites = ma.self_cited(repo, head)
        assert "2106.09685" not in selfcites
        assert "1910.10683" not in selfcites


class TestT0ContextIsTheRepoBeforeTheAdoption:
    def test_the_readme_is_the_old_one(self, repo_with_history: tuple[Path, str, str]) -> None:
        """The whole validity test rests on this. Showing the judge the HEAD README asks it
        whether the repo should adopt what it already has."""
        repo, old, head = repo_with_history
        at_t0 = ma.t0_context(repo, "proj", old)
        assert "# proj" in at_t0
        assert "rewritten" not in at_t0
        assert "2106.09685" not in at_t0
        assert "2106.09685" in ma.t0_context(repo, "proj", head)

    def test_manifests_present_then_are_included(
        self, repo_with_history: tuple[Path, str, str]
    ) -> None:
        repo, old, _head = repo_with_history
        assert "## setup.py" in ma.t0_context(repo, "proj", old)


class TestTheTwoPatternsAgree:
    """`git grep -E` is POSIX ERE and the Python regex is not — they cannot be the same string.

    The first version derived the grep pattern from the Python one by replacing "(" with
    "(?:", which produced "(?:?:" and matched nothing at all. Every repo then reported zero
    adoptions, which is indistinguishable from a real negative result.
    """

    SAMPLE = (
        "see arXiv:1706.03762 and https://arxiv.org/abs/2106.09685 plus ARXIV/1910.10683 "
        "and arxiv.org/abs/2401.00001v2 but not 1234.5678 alone"
    )

    def test_the_ere_pattern_finds_what_the_python_pattern_finds(self) -> None:
        import re

        ere = re.compile(ma.GREP_PATTERN, re.I)
        via_ere = {ma.ID.search(m.group(0) + "x").group(1) for m in ere.finditer(self.SAMPLE)}
        via_py = {m.group(1) for m in ma.ID.finditer(self.SAMPLE)}
        assert via_ere == via_py
        assert via_py == {"1706.03762", "2106.09685", "1910.10683", "2401.00001"}

    def test_the_ere_pattern_uses_no_python_only_syntax(self) -> None:
        assert "(?:" not in ma.GREP_PATTERN
        assert "\\d" not in ma.GREP_PATTERN


class TestRetroHopCallsTheRealApi:
    """`hop(ids, direction)` takes S2 endpoint names positionally and validates nothing.

    Calling it with "forward"/"backward" — the English words this project uses in prose —
    fetches nothing. The first version of `retro_hop` omitted the argument entirely and only
    failed at runtime, after a background launch had already been waited on for hours.
    """

    def test_the_directions_are_the_two_s2_endpoint_names(self) -> None:
        assert ma.HOP_DIRECTIONS == ("references", "citations")

    def test_both_directions_are_hopped_and_unioned(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import types

        calls: list[tuple[tuple[str, ...], str]] = []

        class _Result:
            def __init__(self, reached: set[str]) -> None:
                self.reached = reached
                self.failed_chunks = 0

        def fake_hop(ids, direction, cap=60):  # type: ignore[no-untyped-def]
            calls.append((tuple(ids), direction))
            return _Result({"AAA"} if direction == "references" else {"BBB"})

        monkeypatch.setitem(
            sys.modules, "diagnose_citation_hop", types.SimpleNamespace(hop=fake_hop)
        )
        seeds = tmp_path / "adoption_seeds.json"
        seeds.write_text('{"rl": ["1111.1111"]}', encoding="utf-8")
        monkeypatch.setattr(ma, "SEEDS", seeds)

        rows = [
            {"case": "rl", "id": "AAA", "usable": True},
            {"case": "rl", "id": "BBB", "usable": True},
            {"case": "rl", "id": "CCC", "usable": True},
        ]
        out = ma.retro_hop(rows)
        assert [d for _, d in calls] == ["references", "citations"]
        assert [r["hop_reached"] for r in out] == [True, True, False]

    def test_a_throttled_chunk_refuses_to_score(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A silently smaller pool reads as a worse channel rather than a broken run."""
        import types

        class _Result:
            reached: set[str] = set()
            failed_chunks = 1

        monkeypatch.setitem(
            sys.modules,
            "diagnose_citation_hop",
            types.SimpleNamespace(hop=lambda *a, **k: _Result()),
        )
        seeds = tmp_path / "adoption_seeds.json"
        seeds.write_text('{"rl": ["1111.1111"]}', encoding="utf-8")
        monkeypatch.setattr(ma, "SEEDS", seeds)
        out = ma.retro_hop([{"case": "rl", "id": "AAA", "usable": True}])
        assert out[0]["hop_reached"] is None


class TestPaperAgeParsing:
    def test_yymm_becomes_a_real_date(self) -> None:
        assert ma._posted("2106.09685").year == 2021
        assert ma._posted("2106.09685").month == 6
        assert ma._posted("1011.2602").year == 2010

    def test_the_six_month_bar_is_the_pre_registered_one(self) -> None:
        assert ma.MIN_PAPER_AGE_DAYS == 182
        assert ma.WINDOW_MONTHS == 24
