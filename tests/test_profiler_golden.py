"""The four Tier A fixtures profile to exactly what they profiled yesterday.

`evals/run_eval.py` is this project's free regression gate: it profiles the mini-repos under
`evals/repos/` with the shipping profiler and ranks a frozen pool of labelled arXiv papers,
so the same inputs must give the same numbers. What it cannot do is say *why* a number
moved — a changed profile and a changed ranker look identical in `P@10`.

This pins the other half. Nine profiler fixes landed together for scientific-software
repositories (reading `doc/`, parsing `setup.cfg`, stripping MyST roles and reference-style
badges), each justified by the claim that the ML fixtures are untouched because they have no
doc tree, no `setup.cfg`, no `~=` pin and no reference-style markdown. That claim was
measured once, by hand, before the fixes landed. Here it is CI-enforced instead, which is
the difference between an argument and a guard.

**When this test fails, it is doing its job.** The question it asks is not "is the new
profile worse" — it is "did you mean to move the population every published benchmark
number was measured on". Regenerate the fixture deliberately, in the same commit as the
change, with the Tier A metrics re-measured alongside:

    uv run python -c "import json; from pathlib import Path; \\
        from reporadar.profiler import profile_repo; from reporadar.config import ProfilerConfig; \\
        print(json.dumps({c: (lambda p: {'keywords': [t for t, _ in p.keywords], \\
            'anchors': p.anchors, 'domains': p.domains, 'prose': p.prose, \\
            'corpus_phrases': p.corpus_phrases})(profile_repo(Path('evals/repos')/c, \\
            ProfilerConfig(prose_chars=300))) for c in ('rag','cv','rl','webdev')}, \\
            indent=2, sort_keys=True))" > tests/fixtures/golden_profiles.json
    uv run python evals/run_eval.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from reporadar.config import ProfilerConfig
from reporadar.profiler import profile_repo

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = json.loads(
    (ROOT / "tests" / "fixtures" / "golden_profiles.json").read_text(encoding="utf-8")
)
CASES = sorted(GOLDEN)


def _profile(case: str):
    return profile_repo(ROOT / "evals" / "repos" / case, ProfilerConfig(prose_chars=300))


class TestTheTierAFixturesProfileToTheirGoldenValues:
    @pytest.mark.parametrize("case", CASES)
    def test_keywords(self, case: str) -> None:
        """Keywords are the queries. A moved keyword list is a moved candidate pool, which
        is a moved benchmark — the one change this repository cannot make quietly."""
        assert [term for term, _ in _profile(case).keywords] == GOLDEN[case]["keywords"]

    @pytest.mark.parametrize("case", CASES)
    def test_anchors_and_domains(self, case: str) -> None:
        profile = _profile(case)
        assert profile.anchors == GOLDEN[case]["anchors"]
        assert profile.domains == GOLDEN[case]["domains"]

    @pytest.mark.parametrize("case", CASES)
    def test_prose(self, case: str) -> None:
        """The 300 characters the gate and HyDE read as "what this project is".

        Nothing free can measure whether a change here is an improvement: the offline gate
        is prose-blind, and the fine-scale probability map that consumes this text through
        `triage.repo_context_block` is a frozen logistic whose recalibration costs API
        calls. So the guard is equality, and moving it is a decision to be made with a
        measurement in hand.
        """
        assert _profile(case).prose == GOLDEN[case]["prose"]

    @pytest.mark.parametrize("case", CASES)
    def test_corpus_phrases(self, case: str) -> None:
        assert _profile(case).corpus_phrases == GOLDEN[case]["corpus_phrases"]


class TestTheGuardCanFail:
    """A golden test whose fixture no longer describes anything passes forever."""

    def test_the_fixture_covers_every_mini_repo(self) -> None:
        on_disk = {p.name for p in (ROOT / "evals" / "repos").iterdir() if p.is_dir()}
        assert set(CASES) == on_disk, (
            "evals/repos/ and the golden fixture disagree about which cases exist; "
            "a case added without a golden entry is unguarded"
        )

    def test_a_changed_profile_is_caught(self) -> None:
        """Mutation: the assertion must be sensitive to a single term."""
        mutated = [*GOLDEN[CASES[0]]["keywords"]]
        mutated[0] = "definitely-not-a-real-keyword"
        assert [t for t, _ in _profile(CASES[0]).keywords] != mutated
