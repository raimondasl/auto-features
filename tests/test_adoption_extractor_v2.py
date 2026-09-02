"""Extractor v2 — the widened adoption label for the judge-validity pool. [frame §6.1/§6.2]

NR-57 exhausted this benchmark's supply of adoptions: 35 usable positives against the ~75
needed for the two judges' discrimination gaps to separate. The pool has to come from a
screened population instead, and the label has to survive one thing v1 does not:

    projects migrate their paper links wholesale.

`diffusers` is the measured case — 99 arXiv-regex ids in its docs at T0 (2024-08-16) and 11
at HEAD, while the Hugging Face form goes 66 → 163. Under v1 that repo reads as one that
stopped citing papers. The dangerous half is the other direction: an id cited at T0 *only*
as an HF link and at HEAD as an arXiv link is, to v1, a brand-new adoption. It is the same
paper and a docs refactor. That is why §6.1 applies the union at **both** ends, and why the
first test below checks the false adoption v1 manufactures rather than only the ones it misses.

The two filters v2 adds are asymmetric on purpose, and the tests pin the asymmetry:

  * the **reverse-citation path filter** drops ids seen only under showcase-shaped paths —
    applied at HEAD only, because removing an id from the T0 bibliography would *create* an
    adoption, the one direction a ground-truth label cannot afford to be wrong in;
  * the **doc-genesis guard** flags repos with no T0 bibliography at all, where every id at
    HEAD would otherwise read as an adoption.
"""

from __future__ import annotations

import importlib.util
import json
import os
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

T0_README = "# proj\nBuilt on arXiv:1706.03762.\nAlso https://huggingface.co/papers/2005.11401.\n"
HEAD_README = (
    "# proj\n"
    # 1706.03762 migrated arXiv -> HF, and 2005.11401 migrated HF -> arXiv. Neither is an
    # adoption; both were cited at T0.
    "Built on https://hf.co/papers/1706.03762.\n"
    "Also https://arxiv.org/abs/2005.11401.\n"
    # The only genuine adoption in this fixture.
    "New in this release: https://huggingface.co/papers/2106.09685.\n"
    "\n## Citation\nIf you use this, cite https://huggingface.co/papers/2402.00002.\n"
)
SHOWCASE = "# Who uses proj\nSee arXiv:2301.00001 for a project built on us.\n"


def _git(repo: Path, *args: str, when: str | None = None) -> str:
    """`when` backdates the COMMITTER date, which is what `rev-list --before` reads — an
    author date alone leaves every commit inside the 24-month window and T0 resolves to
    nothing."""
    env = None
    if when is not None:
        env = {**os.environ, "GIT_AUTHOR_DATE": when, "GIT_COMMITTER_DATE": when}
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
        env=env,
    )
    assert out.returncode == 0, f"git {args}: {out.stderr}"
    return out.stdout


def _write(repo: Path, rel: str, text: str) -> None:
    path = repo / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


@pytest.fixture()
def migrated_repo(tmp_path: Path) -> tuple[Path, str, str]:
    """Links migrate in both directions between T0 and HEAD, plus a showcase and a self-cite."""
    repo = tmp_path / "proj"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _write(repo, "README.md", T0_README)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "first", when="2022-01-01T00:00:00+00:00")
    t0 = _git(repo, "rev-parse", "HEAD").strip()

    _write(repo, "README.md", HEAD_README)
    _write(repo, "docs/showcase/users.md", SHOWCASE)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "second")
    head = _git(repo, "rev-parse", "HEAD").strip()
    return repo, t0, head


class TestTheUnionIsAppliedAtBothEnds:
    def test_v1_misses_the_hf_form_entirely(self, migrated_repo: tuple[Path, str, str]) -> None:
        repo, t0, head = migrated_repo
        assert ma.ids_at(repo, t0, "v1") == {"1706.03762"}
        assert ma.ids_at(repo, head, "v1") == {"2005.11401", "2301.00001"}

    def test_v2_sees_both_forms_at_both_revisions(
        self, migrated_repo: tuple[Path, str, str]
    ) -> None:
        repo, t0, head = migrated_repo
        assert ma.ids_at(repo, t0, "v2") == {"1706.03762", "2005.11401"}
        assert ma.ids_at(repo, head, "v2") == {
            "1706.03762",
            "2005.11401",
            "2106.09685",
            "2301.00001",
            "2402.00002",
        }

    def test_v1_manufactures_an_adoption_out_of_a_docs_refactor(
        self, migrated_repo: tuple[Path, str, str]
    ) -> None:
        """The failure that justifies widening T0 as well as HEAD.

        2005.11401 is cited at T0 as an HF link and at HEAD as an arXiv link. v1 cannot see
        the T0 citation, so it reports the paper as newly adopted — a positive that a
        ground-truth label has no business inventing.
        """
        repo, t0, head = migrated_repo
        adopted_v1 = ma.ids_at(repo, head, "v1") - ma.ids_at(repo, t0, "v1")
        assert "2005.11401" in adopted_v1, "fixture no longer exercises the migration"

        adopted_v2 = ma.ids_at(repo, head, "v2") - ma.ids_at(repo, t0, "v2")
        assert "2005.11401" not in adopted_v2
        assert "1706.03762" not in adopted_v2
        assert adopted_v2 == {"2106.09685", "2301.00001", "2402.00002"}


class TestTheReverseCitationFilter:
    def test_an_id_seen_only_under_a_showcase_path_is_dropped(
        self, migrated_repo: tuple[Path, str, str]
    ) -> None:
        repo, _t0, head = migrated_repo
        paths = ma.ids_with_paths(repo, head, "v2")
        assert paths["2301.00001"] == {"docs/showcase/users.md"}
        assert ma.reverse_cited_only(paths) == {"2301.00001"}

    def test_an_id_also_cited_in_the_readme_survives(self, tmp_path: Path) -> None:
        """`all()` over an empty set is True, and `all()` over a mixed set must be False —
        a paper the docs genuinely rely on does not stop counting because someone also
        listed it on a showcase page."""
        paths = {
            "2106.09685": {"README.md", "docs/showcase/users.md"},
            "2301.00001": {"docs/showcase/users.md"},
            "1706.03762": {"community/index.md", "gallery/demo.md"},
        }
        assert ma.reverse_cited_only(paths) == {"2301.00001", "1706.03762"}

    def test_it_is_never_applied_to_the_t0_side(self) -> None:
        """Pinned as a property of the source, not of a run: `mine` calls
        `reverse_cited_only` on the HEAD path map alone. Filtering T0 would remove ids from
        the 'before' set, and every id removed there becomes a fabricated adoption."""
        import inspect

        body = inspect.getsource(ma.mine)
        assert "reverse_cited_only(head_paths)" in body
        assert "reverse_cited_only(t0" not in body


class TestSelfCitationSpeaksV2:
    def test_a_citation_heading_linking_to_hf_is_caught(
        self, migrated_repo: tuple[Path, str, str]
    ) -> None:
        """Without this, a project's own paper — linked in the HF form under its Citation
        heading — is the strongest-looking adoption in the set."""
        repo, _t0, head = migrated_repo
        assert ma.self_cited(repo, head, "v2") == {"2402.00002"}

    def test_v1_cannot_see_it(self, migrated_repo: tuple[Path, str, str]) -> None:
        repo, _t0, head = migrated_repo
        assert ma.self_cited(repo, head, "v1") == set()


class TestThePatternsAgree:
    """Same guard as v1's: `git grep -E` is POSIX ERE and cannot be the Python pattern."""

    SAMPLE = (
        "see https://huggingface.co/papers/2106.09685 and hf.co/papers/1706.03762 plus "
        "HuggingFace.co/papers/2005.11401 and (https://huggingface.co/papers/2410.15458) "
        "but not huggingface.co/models/2401.00001"
    )

    def test_the_ere_pattern_finds_what_the_python_pattern_finds(self) -> None:
        import re

        ere = re.compile(ma.HF_GREP_PATTERN, re.I)
        via_ere = {ma.HF_ID.search(m.group(0)).group(1) for m in ere.finditer(self.SAMPLE)}
        via_py = {m.group(1) for m in ma.HF_ID.finditer(self.SAMPLE)}
        assert via_ere == via_py
        assert via_py == {"2106.09685", "1706.03762", "2005.11401", "2410.15458"}

    def test_the_ere_pattern_uses_no_python_only_syntax(self) -> None:
        assert "(?:" not in ma.HF_GREP_PATTERN
        assert "\\d" not in ma.HF_GREP_PATTERN

    def test_the_two_extractors_are_the_only_ones_offered(self) -> None:
        assert ma.EXTRACTORS == ("v1", "v2")


class TestTheV1RecordCannotBeOverwritten:
    """§6.1 reports the v1 numbers unchanged as v1. A v2 run that wrote `adoptions.json`
    would destroy the record it is being compared against."""

    def test_the_artefacts_are_per_extractor(self) -> None:
        assert ma.out_path("v1") == ma.OUT
        assert ma.out_path("v2") != ma.OUT
        assert ma.out_path("v2").name == "adoptions-v2.json"
        assert ma.seeds_path("v1") == ma.SEEDS
        assert ma.seeds_path("v2").name == "adoption_seeds-v2.json"


class TestMineEndToEnd:
    """The filters composed, over a real clone of a real history."""

    @staticmethod
    def _run(
        repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, extractor: str = "v2"
    ) -> list[dict[str, object]]:
        monkeypatch.setattr(ma, "CLONES", tmp_path / "clones")
        monkeypatch.setattr(ma, "SEEDS", tmp_path / "seeds.json")
        return ma.mine({"proj": repo.as_uri()}, extractor=extractor)

    def test_only_the_genuine_adoption_is_usable(
        self, migrated_repo: tuple[Path, str, str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        repo, t0, head = migrated_repo
        rows = self._run(repo, tmp_path, monkeypatch)
        by_id = {r["id"]: r for r in rows}
        assert set(by_id) == {"2106.09685", "2301.00001", "2402.00002"}
        assert [r["id"] for r in rows if r["usable"]] == ["2106.09685"]
        assert by_id["2301.00001"]["reverse_cited"] is True
        assert by_id["2402.00002"]["self_cited"] is True
        assert by_id["2106.09685"]["via"] == "hf"

    def test_the_head_revision_is_pinned_to_a_sha(
        self, migrated_repo: tuple[Path, str, str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The pool is a published artefact. A row that says "HEAD" is not reproducible a
        week later, when HEAD has moved and the positive set is silently different."""
        repo, t0, head = migrated_repo
        rows = self._run(repo, tmp_path, monkeypatch)
        assert {r["head"] for r in rows} == {head}
        assert {r["t0"] for r in rows} == {t0}

    def test_the_seeds_go_to_the_v2_file(
        self, migrated_repo: tuple[Path, str, str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        repo, _t0, _head = migrated_repo
        self._run(repo, tmp_path, monkeypatch)
        assert not (tmp_path / "seeds.json").exists()
        seeds = json.loads((tmp_path / "seeds-v2.json").read_text(encoding="utf-8"))
        assert seeds["proj"] == ["1706.03762", "2005.11401"]


class TestTheDocGenesisGuard:
    def test_a_repo_with_no_t0_bibliography_yields_nothing_usable(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Every id at HEAD is "new" when there was no "before". Those rows are kept and
        flagged rather than dropped — how many repos are like this is a property of the
        population §6.2 draws from, and the paper reports it."""
        repo = tmp_path / "genesis"
        repo.mkdir()
        _git(repo, "init", "-q", "-b", "main")
        _write(repo, "README.md", "# genesis\nNo papers here yet.\n")
        _git(repo, "add", "-A")
        _git(repo, "commit", "-qm", "first", when="2022-01-01T00:00:00+00:00")
        _write(repo, "README.md", "# genesis\nNow citing arXiv:1706.03762.\n")
        _git(repo, "add", "-A")
        _git(repo, "commit", "-qm", "second")

        monkeypatch.setattr(ma, "CLONES", tmp_path / "clones")
        monkeypatch.setattr(ma, "SEEDS", tmp_path / "seeds.json")
        rows = ma.mine({"genesis": repo.as_uri()}, extractor="v2")
        assert [r["id"] for r in rows] == ["1706.03762"]
        assert rows[0]["genesis"] is True
        assert rows[0]["usable"] is False


class TestTheQualifyingScreen:
    """§6.2: a row qualifies at ids_v2(HEAD) ≥ 10, from a blobless clone and a doc grep."""

    @staticmethod
    def _repo(tmp_path: Path, name: str, n_ids: int) -> Path:
        repo = tmp_path / name
        repo.mkdir()
        _git(repo, "init", "-q", "-b", "main")
        links = "\n".join(
            f"https://huggingface.co/papers/24{i:02d}.0000{i % 10}" for i in range(n_ids)
        )
        _write(repo, "README.md", f"# {name}\n{links}\n")
        _git(repo, "add", "-A")
        _git(repo, "commit", "-qm", "only")
        return repo

    def test_the_bar_is_on_the_v2_count(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(ma, "CLONES", tmp_path / "clones")
        rich = self._repo(tmp_path, "rich", 12)
        thin = self._repo(tmp_path, "thin", 3)
        rows = ma.screen(
            [
                {"full_name": "acme/rich", "url": rich.as_uri(), "created_at": "2020-01-01"},
                {"full_name": "acme/thin", "url": thin.as_uri(), "created_at": "2020-01-01"},
            ]
        )
        by_name = {r["full_name"]: r for r in rows}
        assert by_name["acme/rich"]["ids_v2_head"] == 12
        assert by_name["acme/rich"]["qualifies"] is True
        # v1 sees none of them: every link in the fixture is in the HF form.
        assert by_name["acme/rich"]["ids_v1_head"] == 0
        assert by_name["acme/thin"]["ids_v2_head"] == 3
        assert by_name["acme/thin"]["qualifies"] is False

    def test_a_clone_it_made_and_did_not_need_is_deleted(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Several hundred candidates at tens of MB each is gigabytes of cache to answer a
        question that fits in one integer, and this runs on a laptop.

        This failed first time on Windows: git leaves its objects and pack files read-only,
        `rmtree(ignore_errors=True)` swallowed the `PermissionError`, and the screen wrote
        "clone removed" for a clone that was entirely still there. The note now reports what
        happened rather than what was intended.
        """
        clones = tmp_path / "clones"
        monkeypatch.setattr(ma, "CLONES", clones)
        thin = self._repo(tmp_path, "thin", 2)
        rows = ma.screen([{"full_name": "acme/thin", "url": thin.as_uri()}])
        assert not (clones / "acme__thin").exists()
        assert rows[0]["note"] == "clone removed"

    def test_a_clone_that_was_already_there_is_left_alone(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A pre-existing clone belongs to the legacy 37 or to an earlier screen. Deleting
        it would make this script destructive to other people's cached work."""
        clones = tmp_path / "clones"
        monkeypatch.setattr(ma, "CLONES", clones)
        thin = self._repo(tmp_path, "thin", 2)
        ma.screen([{"full_name": "acme/thin", "url": thin.as_uri()}], keep_clones=True)
        assert (clones / "acme__thin").exists()
        ma.screen([{"full_name": "acme/thin", "url": thin.as_uri()}])
        assert (clones / "acme__thin").exists()

    def test_a_candidate_created_too_recently_is_never_cloned(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        clones = tmp_path / "clones"
        monkeypatch.setattr(ma, "CLONES", clones)
        rich = self._repo(tmp_path, "rich", 12)
        rows = ma.screen(
            [{"full_name": "acme/rich", "url": rich.as_uri(), "created_at": "2025-06-01"}],
            created_before="2024-03-01",
        )
        assert rows[0]["qualifies"] is False
        assert rows[0]["clone_ok"] is False
        assert not (clones / "acme__rich").exists()

    def test_the_csv_columns_are_fixed(self, tmp_path: Path) -> None:
        out = tmp_path / "validity_screen.csv"
        ma.write_screen([{"full_name": "acme/rich", "qualifies": True}], out)
        header = out.read_text(encoding="utf-8").splitlines()[0]
        assert header.split(",") == list(ma.SCREEN_COLUMNS)

    def test_a_candidate_row_without_a_name_is_refused(self, tmp_path: Path) -> None:
        """A blank `full_name` would be cloned as `https://github.com/` and screened as a
        failure — a silently missing candidate rather than a loud one."""
        path = tmp_path / "universe.csv"
        path.write_text(
            "full_name,created_at\nacme/rich,2020-01-01\n,2020-01-01\n", encoding="utf-8"
        )
        with pytest.raises(SystemExit):
            ma.read_candidates(path)
