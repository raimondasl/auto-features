"""The judge prompt's identifier line: labelled by scheme, byte-exact for arXiv.

`evals/judge.py` used to write `arXiv: {arxiv_id}` for every paper. The pipeline's shared
paper shape carries either a real arXiv id or a synthetic `"<source>:<id>"`, so a bioRxiv
preprint reached the judge as `arXiv: biorxiv:10.1101/...` — a false statement about the paper,
and a systematic difference between arXiv and non-arXiv candidates that is not the treatment in
any arm that mixes them. Found while pre-registering the Europe PMC arm
(RESEARCH-scientific-software.md §20.4), before it was run.

The fix has to be conditional, and the reason is the sharp edge here. A verdict is cached under
`sha256(RUBRIC \\0 repo_context)`; the paper is **not** in that hash. So changing this line for a
paper already in the cache leaves the cache key identical while the question changes, and no
guard in the project would notice. The arXiv branch is therefore pinned byte-for-byte by
`TestTheArxivBranchIsFrozen` — if that test ever fails, ~3200 cached verdicts have silently
become answers to a prompt nobody sends any more.

The 61 that *are* affected (`ss:` 55, `iacr:` 6, measured not assumed) are handled by a marker
rather than by deleting files: `evals/cache/` is gitignored and untracked, so a local delete
would fix one machine and propagate to none. `_id_line` is written for non-arXiv verdicts only,
and its absence marks the old ones stale wherever they live.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

EVALS = Path(__file__).resolve().parent.parent / "evals"
if str(EVALS) not in sys.path:
    sys.path.insert(0, str(EVALS))

import judge  # noqa: E402


class TestTheArxivBranchIsFrozen:
    """These strings are a cache-compatibility contract, not a formatting preference."""

    @pytest.mark.parametrize(
        "ident",
        [
            "2506.12986v2",  # new-style, versioned
            "1708.01492v5",
            "2107.03730",  # new-style, unversioned
            "cs/0701001",  # old-style
            "cs/0701001v2",
        ],
    )
    def test_an_arxiv_id_still_renders_exactly_as_before(self, ident: str) -> None:
        assert judge._identifier_line({"arxiv_id": ident}) == f"arXiv: {ident}"

    @pytest.mark.parametrize("missing", [{}, {"arxiv_id": ""}, {"arxiv_id": None}])
    def test_a_missing_id_still_renders_exactly_as_before(self, missing: dict) -> None:
        """The empty case is in the cache too — `arXiv: ` with a trailing space."""
        assert judge._identifier_line(missing) == "arXiv: "

    def test_the_whole_prompt_is_unchanged_for_an_arxiv_paper(self) -> None:
        """Belt and braces: the line is right AND it is still in the same place."""
        prompt = judge._build_user_prompt(
            "REPO", {"title": "T", "arxiv_id": "2506.12986v2", "abstract": "A"}
        )
        assert prompt == (
            "# Repository context\nREPO\n\n"
            "# Candidate paper\n"
            "Title: T\n"
            "arXiv: 2506.12986v2\n"
            "Abstract: A\n\n"
            "Score this paper for the repository above using the rubric."
        )


class TestNonArxivIdsAreLabelledHonestly:
    @pytest.mark.parametrize(
        ("ident", "expected"),
        [
            ("biorxiv:10.1101/2024.01.01.573000", "bioRxiv: 10.1101/2024.01.01.573000"),
            ("medrxiv:10.1101/2024.02.02.24301234", "medRxiv: 10.1101/2024.02.02.24301234"),
            ("doi:10.1101/2024.01.01.573000", "DOI: 10.1101/2024.01.01.573000"),
            (
                "ss:649def34f8be52c8b66281af98ae884c09aef38b",
                "Semantic Scholar: 649def34f8be52c8b66281af98ae884c09aef38b",
            ),
            ("oa:W2741809807", "OpenAlex: W2741809807"),
            ("dblp:conf/vldb/X", "DBLP: conf/vldb/X"),
            ("iacr:2018/367", "IACR ePrint: 2018/367"),
        ],
    )
    def test_known_schemes_get_their_real_name(self, ident: str, expected: str) -> None:
        assert judge._identifier_line({"arxiv_id": ident}) == expected

    def test_an_unknown_scheme_keeps_its_prefix_rather_than_claiming_arxiv(self) -> None:
        """A new adapter must not silently inherit the arXiv label; the wrong-but-honest
        answer here is the scheme it declared."""
        assert judge._identifier_line({"arxiv_id": "zenodo:12345"}) == "zenodo: 12345"

    def test_a_schemeless_non_arxiv_id_is_named_neutrally(self) -> None:
        assert judge._identifier_line({"arxiv_id": "not-an-id"}) == "Identifier: not-an-id"

    def test_no_non_arxiv_paper_is_ever_called_arxiv(self) -> None:
        """The defect this file exists for, stated as an invariant over every scheme."""
        schemes = ("biorxiv:10.1101/x", "ss:abc", "oa:W1", "dblp:a/b", "iacr:2018/1", "doi:10.1/x")
        for ident in schemes:
            assert not judge._identifier_line({"arxiv_id": ident}).startswith("arXiv:")


class TestTheStaleCacheDetectsItself:
    """The 61 pre-existing non-arXiv verdicts must re-judge; the ~3200 arXiv ones must not.

    `evals/cache/` is gitignored and untracked, so deleting the stale entries locally would
    have fixed one machine and propagated to none. The marker travels in the code instead.
    """

    def test_an_arxiv_verdict_with_no_marker_is_still_a_hit(self) -> None:
        """The load-bearing case: every existing verdict predates the marker."""
        assert judge._id_line_matches({"score": 2}, "arXiv: 2506.12986v2")

    def test_a_non_arxiv_verdict_with_no_marker_is_stale(self) -> None:
        """Exactly the 61 `ss:`/`iacr:` entries in the gold cache today."""
        assert not judge._id_line_matches({"score": 2}, "Semantic Scholar: abc")

    def test_a_non_arxiv_verdict_matching_todays_line_is_a_hit(self) -> None:
        assert judge._id_line_matches(
            {"score": 2, "_id_line": "bioRxiv: 10.1101/x"}, "bioRxiv: 10.1101/x"
        )

    def test_a_non_arxiv_verdict_from_a_different_line_is_stale(self) -> None:
        assert not judge._id_line_matches(
            {"score": 2, "_id_line": "arXiv: biorxiv:10.1101/x"}, "bioRxiv: 10.1101/x"
        )
