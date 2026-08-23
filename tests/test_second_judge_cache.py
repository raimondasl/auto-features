"""One rule for what a paper id is called on disk, and a guard that keeps it that way.

`evals/second_judge.second_verdict` used to build its cache path as
``paper['arxiv_id'].replace('/', '_')``. That leaves the **colon** in a synthetic id like
``doi:10.1038/s42256-023-00716-3`` — and on Windows a colon in a path is the NTFS
alternate-data-stream separator, so every non-arXiv verdict this project bought was written
into a stream hanging off a zero-byte file named ``doi``. 93 of them across 11 cases when it
was found (§39.6), 82 of those from the Europe PMC arm of §21.

They read back correctly, because the same expression was used to write and to read. That is
why nothing noticed: the numbers were right, and the files were invisible to ``ls``, ``glob``
and ``find``, and would not survive a copy to any non-NTFS filesystem.

``judge._cache_path`` had sanitised properly since it was written. Two implementations of one
invariant, and the wrong one had been hand-copied into ten eval modules.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
EVALS = ROOT / "evals"
sys.path.insert(0, str(EVALS))

from second_judge import safe_paper_id, second_cache_path  # noqa: E402

# Characters that are illegal in a Windows filename. A colon is the one that bit, because it
# fails silently as an ADS separator instead of raising like the others do.
WINDOWS_ILLEGAL = set('<>:"/\\|?*')


class TestTheIdIsSafeOnEveryPlatform:
    @pytest.mark.parametrize(
        "paper_id",
        [
            "doi:10.1038/s42256-023-00716-3",
            "doi:10.1021/acs.jctc.5c00955.s001",
            "biorxiv:10.1101/2023.01.01.522000",
            "oa:W3114472930",
            "ss:abc123",
            "iacr:2019/1234",
            "dblp:conf/x/Y",
            "cs/0602007v4",
            "2302.14231v2",
        ],
    )
    def test_no_illegal_character_survives(self, paper_id: str) -> None:
        assert not (set(safe_paper_id(paper_id)) & WINDOWS_ILLEGAL)

    def test_the_colon_specifically_is_gone(self) -> None:
        """The one that produced an alternate data stream rather than an error."""
        assert ":" not in safe_paper_id("doi:10.1038/s42256-023-00716-3")

    def test_distinct_ids_stay_distinct(self) -> None:
        """Sanitising must not merge two papers into one cache slot.

        `10.1021/acs.jctc.5c00955` and its `.s001` supporting-information record are two
        different OpenAlex works that reached the same digest (§39.5), so a collision here
        would silently overwrite one verdict with the other's.
        """
        ids = [
            "doi:10.1021/acs.jctc.5c00955",
            "doi:10.1021/acs.jctc.5c00955.s001",
            "doi:10.1038/s42256-023-00716-3",
            "2302.14231v2",
            "oa:W3114472930",
        ]
        assert len({safe_paper_id(i) for i in ids}) == len(ids)

    def test_it_matches_the_gold_cache_rule(self) -> None:
        """The two verdict caches must agree on what a paper is called."""
        import judge as judge_mod

        for paper_id in ("doi:10.1038/x", "2302.14231v2", "biorxiv:10.1101/y"):
            gold = judge_mod._cache_path("m", "case", paper_id).name
            assert gold == f"{safe_paper_id(paper_id)}.json"

    def test_the_path_lands_under_the_second_judge_cache(self, tmp_path: Path) -> None:
        p = second_cache_path("claude-sonnet-5", "mat-chgpot", "doi:10.1038/s42256-023-00716-3")
        assert p.parent.name == "mat-chgpot"
        assert p.parent.parent.name == "claude-sonnet-5"
        assert p.name == "doi_10.1038_s42256-023-00716-3.json"

    def test_it_actually_writes_one_file(self, tmp_path: Path) -> None:
        """The regression, reproduced end to end.

        On Windows the old expression made this create a zero-byte ``doi`` and hide the
        content in a stream; on POSIX it made a file with a colon in the name. Either way the
        directory listing did not show what a reader expects.
        """
        target = tmp_path / f"{safe_paper_id('doi:10.1038/s42256-023-00716-3')}.json"
        target.write_text('{"score": 3}', encoding="utf-8")
        listed = [p.name for p in tmp_path.iterdir()]
        assert listed == [target.name]
        assert target.stat().st_size > 0
        assert list(tmp_path.glob("*.json")) == [target]


class TestNoModuleHandRollsIt:
    """No eval module may build a verdict-cache filename its own way."""

    OFFENDER = ".replace('/', '_')}.json"

    def test_no_cache_path_hand_rolls_the_rule(self) -> None:
        bad: list[str] = []
        for path in sorted(EVALS.glob("*.py")):
            for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                stripped = line.strip()
                if stripped.startswith("#") or self.OFFENDER not in stripped:
                    continue
                # A cache path is one rooted at a CACHE constant. `exp_*.py` build their own
                # slot names from an arXiv-only pool and are out of scope, so this targets
                # the constant rather than the expression alone.
                if "CACHE" in stripped:
                    bad.append(f"{path.name}:{line_no}: {stripped}")
        assert not bad, (
            "verdict-cache paths must call second_judge.safe_paper_id / second_cache_path — "
            "a raw id leaves a colon in the filename, which is an NTFS alternate data stream "
            "and not a file:\n  " + "\n  ".join(bad)
        )
