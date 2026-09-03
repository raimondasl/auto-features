"""§2.2's remaining eligibility rules, and §3.3's seeded assignment. [pool pre-registration]

Four registered rules had no implementation at all — the walk asked only PP1 (history) and
PP2 (identifiers at T0). Nothing stopped a curated paper list, the artefact §2.2 names
explicitly, from clearing `ids_v2(T0) ≥ 3` on a bibliography of two hundred identifiers and
then dominating the positives.

And two rules of §3.3 were counts rather than decisions: the per-repository cap of 8 recorded
*how many* rather than *which*, and cross-repository identifier assignment did not exist.
Both must be functions of SEED_POOL, fixed before any positive is visible — a cap chosen
afterwards is exactly the discretion the frozen order exists to remove.
"""

from __future__ import annotations

import importlib.util
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


el = _load("eligibility", FRAME)
wp = _load("walk_pool", FRAME)


class TestTheLanguageRule:
    def test_the_committed_set_is_the_frames(self) -> None:
        """§1(v) of the benchmark frame, verbatim."""
        committed = {
            "Python",
            "C",
            "C++",
            "Rust",
            "Go",
            "Julia",
            "JavaScript",
            "TypeScript",
            "R",
            "Fortran",
        }
        assert set(el.LANGUAGES) == committed

    def test_a_blank_language_fails(self) -> None:
        """1,408 of the 17,888 candidate rows report no primary language. "Unknown" is not
        one of the committed languages, so they are out — loudly, not by accident."""
        assert el.language_ok("") is False
        assert el.language_ok("   ") is False

    def test_an_off_set_language_fails(self) -> None:
        assert el.language_ok("Jupyter Notebook") is False
        assert el.language_ok("Java") is False

    def test_a_committed_language_passes(self) -> None:
        assert el.language_ok("Python") is True


class TestX4TheSoftwareProjectRule:
    def test_a_curated_list_is_rejected(self) -> None:
        """The artefact §2.2 names: two hundred identifiers, zero adoption semantics."""
        assert not el.is_software_project("machine-learning", "someone/awesome-nlp", "# hi")
        assert not el.is_software_project("", "x/y", "# A curated list of papers")
        assert not el.is_software_project("paper-list", "x/y", "# hi")

    def test_a_real_project_passes(self) -> None:
        assert el.is_software_project("machine-learning|pytorch", "acme/trainer", "# trainer")

    def test_only_the_first_300_readme_characters_count(self) -> None:
        """X4 reads `README[:300]`, so a project that merely mentions tutorials far down is
        not culled for it."""
        readme = "# a real library\n" + ("x" * 400) + "\nsee our tutorial"
        assert el.is_software_project("", "acme/lib", readme)

    def test_the_description_is_deliberately_not_read(self) -> None:
        """The frame's X4 also reads the repository description; §2.1 forbids it from
        entering the tree because it carries repository URLs, so the enumeration never
        recorded it. A narrowing of the rule's inputs, recorded rather than absorbed — it can
        only let more repositories through, never fewer."""
        import inspect

        doc = el.is_software_project.__doc__ or ""
        assert "description" in doc
        assert "description" not in inspect.signature(el.is_software_project).parameters


class TestX7TheSourceFloor:
    def _git_stub(self, paths: list[str]):  # type: ignore[no-untyped-def]
        def git(repo, *args, check=True, timeout=None):  # type: ignore[no-untyped-def]
            return "\n".join(paths)

        return git

    def test_a_docs_only_repository_fails(self, tmp_path: Path) -> None:
        git = self._git_stub([f"docs/page{i}.md" for i in range(50)])
        assert el.source_file_count(tmp_path, "HEAD", "Python", git) == 0

    def test_source_files_of_the_primary_language_count(self, tmp_path: Path) -> None:
        git = self._git_stub([f"src/mod{i}.py" for i in range(25)] + ["README.md"])
        assert el.source_file_count(tmp_path, "HEAD", "Python", git) == 25

    def test_another_languages_files_do_not(self, tmp_path: Path) -> None:
        git = self._git_stub([f"src/mod{i}.py" for i in range(25)])
        assert el.source_file_count(tmp_path, "HEAD", "Rust", git) == 0

    def test_the_floor_is_the_registered_twenty(self) -> None:
        assert el.MIN_SOURCE_FILES == 20


class TestX5ReadmeProse:
    def test_code_blocks_badges_and_urls_are_stripped(self) -> None:
        text = (
            "# Title\n\n[![build](https://img.shields.io/x)](https://ci.example)\n\n"
            "Real prose here.\n\n```python\nimport this\nprint('code')\n```\n\n"
            "More prose at https://example.com/page\n"
        )
        prose = el.readme_prose(text)
        assert "Real prose here." in prose
        assert "import this" not in prose
        assert "https://" not in prose
        assert "shields.io" not in prose

    def test_a_short_readme_passes_flagged_rather_than_being_culled(self) -> None:
        """Registered explicitly: "shorter READMEs pass with flag `lid_na` (so the rule
        cannot cull the thin proxy)"."""
        ok, flag, chars = el.english_readme("# tiny\nnot much here\n")
        assert ok is True
        assert flag == "lid_na"
        assert chars < el.X5_MIN_PROSE_CHARS

    def test_with_no_detector_the_rule_is_not_applied_and_says_so(self) -> None:
        """`lid.176` is a 126 MB model behind a package this project does not depend on.
        Inventing a substitute would be a different rule wearing X5's name, culling a
        different set with nobody able to tell. So it passes, flagged, and the deviation is
        recorded rather than hidden behind a plausible-looking heuristic."""
        ok, flag, _ = el.english_readme("prose. " * 100)
        assert ok is True
        assert flag == "lid_na_no_detector"

    def test_a_supplied_detector_is_actually_applied(self) -> None:
        long_prose = "prose. " * 100
        assert el.english_readme(long_prose, detector=lambda _t: 0.95)[0] is True
        assert el.english_readme(long_prose, detector=lambda _t: 0.10)[0] is False
        assert el.english_readme(long_prose, detector=lambda _t: 0.10)[1] == "lid"

    def test_the_threshold_is_the_registered_one(self) -> None:
        assert el.X5_MIN_P_EN == 0.8
        assert el.X5_MIN_PROSE_CHARS == 300


class TestTheCapIsASeededSelection:
    """§3.3's cap was an integer in the ledger and nothing marked *which* eight. Whoever
    wrote the analysis would have chosen them after the positives were visible — inside a
    design whose whole anti-discretion argument is an order fixed by an unchoosable pulse."""

    @staticmethod
    def _entries(n: int) -> list[dict]:
        return [{"case": "acme/rich", "id": f"2101.{i:05d}", "usable": True} for i in range(n)]

    def test_the_same_seed_picks_the_same_eight(self) -> None:
        rows = self._entries(20)
        first = sorted(rows, key=lambda e: wp.order_key("S", f"{e['case']}:{e['id']}"))[:8]
        again = sorted(
            list(reversed(rows)), key=lambda e: wp.order_key("S", f"{e['case']}:{e['id']}")
        )[:8]
        assert [e["id"] for e in first] == [e["id"] for e in again]

    def test_a_different_seed_picks_a_different_eight(self) -> None:
        rows = self._entries(20)
        a = sorted(rows, key=lambda e: wp.order_key("S1", f"{e['case']}:{e['id']}"))[:8]
        b = sorted(rows, key=lambda e: wp.order_key("S2", f"{e['case']}:{e['id']}"))[:8]
        assert [e["id"] for e in a] != [e["id"] for e in b]

    def test_the_cap_is_the_registered_eight(self) -> None:
        assert wp.PER_REPO_CAP == 8

    def test_the_walk_stamps_it_rather_than_counting(self) -> None:
        import inspect

        src = inspect.getsource(wp.walk_row)
        assert 'entry["in_cap"] = True' in src
        assert "order_key(seed," in src


class TestCrossRepositoryAssignment:
    """§3.3: an identifier shared across repositories is assigned to one by SEED_POOL and
    counted once, legacy winning ties. Two sibling projects that adopted the same paper would
    otherwise each contribute it — judged twice, counted twice toward the stop rule, and
    entering the cluster bootstrap as two observations, inflating the very count the
    clustered interval exists to be honest about."""

    def test_a_contested_identifier_is_counted_once(self) -> None:
        rows = [
            {"case": "a/one", "id": "2101.00001"},
            {"case": "b/two", "id": "2101.00001"},
            {"case": "a/one", "id": "2102.00002"},
        ]
        wp.assign_across_repos(rows, "SEED")
        contested = [r for r in rows if r["id"] == "2101.00001"]
        assert sum(1 for r in contested if r["counted"]) == 1
        assert len({r["assigned_to"] for r in contested}) == 1

    def test_the_uncontested_identifier_is_untouched(self) -> None:
        rows = [{"case": "a/one", "id": "2102.00002"}]
        wp.assign_across_repos(rows, "SEED")
        assert rows[0]["counted"] is True
        assert rows[0]["assigned_to"] == "a/one"

    def test_the_loser_is_kept_and_marked_not_hidden(self) -> None:
        """§3.1 keeps every row. A dropped loser would make the ledger disagree with itself."""
        rows = [{"case": "a/one", "id": "2101.00001"}, {"case": "b/two", "id": "2101.00001"}]
        wp.assign_across_repos(rows, "SEED")
        assert len(rows) == 2
        assert sorted(r["counted"] for r in rows) == [False, True]

    def test_legacy_wins_the_tie_outright(self) -> None:
        """A paper the legacy cluster already contributes must never be counted again from a
        pool repository, or §5's legacy-versus-pool heterogeneity compares the two clusters
        over an overlapping set of papers."""
        rows = [{"case": "a/one", "id": "2101.00001"}, {"case": "b/two", "id": "2101.00001"}]
        wp.assign_across_repos(rows, "SEED", legacy_ids={"2101.00001"})
        assert all(r["counted"] is False for r in rows)
        assert all(r["assigned_to"] == "legacy" for r in rows)

    def test_the_assignment_does_not_depend_on_arrival_order(self) -> None:
        """A paper mined in an earlier chunk can be contested by one mined later, so the
        merge re-runs over the whole file rather than over the new rows."""
        forward = [{"case": "a/one", "id": "2101.1"}, {"case": "b/two", "id": "2101.1"}]
        backward = [{"case": "b/two", "id": "2101.1"}, {"case": "a/one", "id": "2101.1"}]
        wp.assign_across_repos(forward, "SEED")
        wp.assign_across_repos(backward, "SEED")
        assert {r["case"] for r in forward if r["counted"]} == {
            r["case"] for r in backward if r["counted"]
        }


class TestTheAdoptionCommitDate:
    """§5 splits positives by adoption commit date against each judge's training cutoff, and
    §6 item 6 calls it the only instrument against recognition bias. Every other date on the
    row is repository-level — `head_date` is identical for every positive and post-cutoff by
    construction — so without this the split collapses to one bucket."""

    def test_it_is_taken_while_the_clone_still_exists(self) -> None:
        """The clone is deleted at the end of the row. This is the only cheap window."""
        import inspect

        src = inspect.getsource(wp.walk_row)
        assert src.index("adoption_commit(") < src.index("finally:")

    def test_a_miss_is_none_rather_than_a_blank(self, tmp_path: Path) -> None:
        """A blank reads like a date nobody looked for; None says it was sought and not
        found."""
        import subprocess

        repo = tmp_path / "empty"
        repo.mkdir()
        subprocess.run(
            ["git", "init", "-q", "-b", "main", str(repo)], check=True, capture_output=True
        )
        sha, when = wp.adoption_commit(repo, "2101.00001", "HEAD", "HEAD")
        assert sha is None and when is None

    def test_only_capped_positives_pay_for_it(self) -> None:
        """One `git log -S` per adopted identifier over 24 months of a blobless clone is not
        free. Only the eight that will be judged need the date."""
        import inspect

        src = inspect.getsource(wp.walk_row)
        cap_line = src.index('entry["in_cap"] = True')
        assert src.index("adoption_commit(", cap_line) > cap_line


@pytest.mark.parametrize("language", sorted(el.LANGUAGES))
def test_every_committed_language_has_source_extensions(language: str) -> None:
    """A language with no extension mapping silently scores zero source files and fails X7
    for every repository that reports it — a whole language culled by an omission."""
    assert el.SOURCE_EXTENSIONS.get(language), language
