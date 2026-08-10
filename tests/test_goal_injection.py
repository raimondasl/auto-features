"""A maintainer's stated goal must reach the query and nothing else.

This is not tidiness. Two separate measured results make the placement non-negotiable:

* **P8**: stated wants fed to the 0-3 **gate** scored net@2 +57 against the shipped +95 —
  the worst arm in the campaign. Precision was untouched and *recall collapsed*, because a
  list of named wants replaces the gate's question ("would this improve the project?") with
  a checklist ("is this on the list?"). That experiment's conclusion was that wants belong
  in the query.
* **The fine-scale calibration** is a frozen two-parameter logistic fitted against the exact
  bytes of :func:`reporadar.triage.repo_context_block`. A goal merged into that block would
  move where P crosses 2/3 with nothing failing loudly.

So the tests below assert an isolation property, not an output: whatever a goal does to
hypothesis generation, it must leave the gate prompt and the rescore prompt byte-identical.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from reporadar import finescale, hyde
from reporadar.profiler import RepoProfile
from reporadar.triage import build_triage_prompt, repo_context_block

GOAL = "reduce write amplification during compaction under write-heavy workloads"


@pytest.fixture
def profile() -> RepoProfile:
    return RepoProfile(
        anchors=["rocksdb", "lz4"],
        domains=["databases"],
        keywords=[("compaction", 0.9), ("key-value", 0.8)],
        prose="A persistent key-value store.",
    )


class _Recorder:
    """Captures the prompt instead of calling a model."""

    def __init__(self) -> None:
        self.prompt = ""

    def __call__(self, prompt: str, cfg: Any, **kw: Any) -> str:
        self.prompt = prompt
        return '["an abstract", "another abstract"]'


def hypothesis_prompt(
    profile: RepoProfile, monkeypatch: pytest.MonkeyPatch, goal: str | None
) -> str:
    rec = _Recorder()
    monkeypatch.setattr(hyde, "complete", rec)
    hyde.generate_hypotheses(profile, SimpleNamespace(), n=2, goal=goal)
    return rec.prompt


class TestTheGoalReachesTheQuery:
    def test_the_goal_text_appears_in_the_hypothesis_prompt(
        self, profile: RepoProfile, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        assert GOAL in hypothesis_prompt(profile, monkeypatch, GOAL)

    def test_without_a_goal_the_prompt_is_unchanged(
        self, profile: RepoProfile, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        assert GOAL not in hypothesis_prompt(profile, monkeypatch, None)

    def test_a_blank_goal_is_a_no_op_not_an_empty_section(
        self, profile: RepoProfile, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An empty heading would tell the model the maintainers stated nothing, which is
        a different claim from not asking."""
        assert hypothesis_prompt(profile, monkeypatch, "   ") == hypothesis_prompt(
            profile, monkeypatch, None
        )

    def test_an_overlong_goal_is_truncated(
        self, profile: RepoProfile, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        prompt = hypothesis_prompt(profile, monkeypatch, "x" * 5000)
        assert "x" * 600 in prompt
        assert "x" * 601 not in prompt


class TestTheGoalReachesNothingElse:
    """The isolation property. If any of these fail, P8's negative has been re-created."""

    def test_the_shared_repo_block_is_byte_identical(self, profile: RepoProfile) -> None:
        before = repo_context_block(profile)
        hyde.GOAL_BLOCK.format(goal=GOAL)  # touching the template must change nothing
        assert repo_context_block(profile) == before
        assert GOAL not in repo_context_block(profile)

    def test_the_gate_prompt_never_carries_the_goal(self, profile: RepoProfile) -> None:
        paper = {"arxiv_id": "2401.00001", "title": "T", "abstract": "A"}
        assert GOAL not in build_triage_prompt(paper, profile)

    def test_the_finescale_prompt_never_carries_the_goal(self, profile: RepoProfile) -> None:
        paper = {"arxiv_id": "2401.00001", "title": "T", "abstract": "A"}
        assert GOAL not in finescale.build_prompt(paper, profile)

    def test_the_hypothesis_prompt_extends_the_shared_block_rather_than_editing_it(
        self, profile: RepoProfile, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The goal is appended after the shared block, so the no-goal prompt's repo
        description survives verbatim inside the goal-bearing one."""
        block = repo_context_block(profile)[:6000]
        assert block in hypothesis_prompt(profile, monkeypatch, GOAL)

    def test_discover_threads_the_goal_to_the_generator(
        self, profile: RepoProfile, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        """Guards the wiring, not the prompt: an unthreaded goal would make the whole arm
        a silent control arm."""
        seen: dict[str, Any] = {}

        def fake(prof: Any, cfg: Any, *, n: int = 4, goal: str | None = None) -> list[str]:
            seen["goal"] = goal
            raise hyde.HydeError("stop here — the wiring is what is under test")

        monkeypatch.setattr(hyde, "index_shards", lambda d: [tmp_path / "s.npy"])
        monkeypatch.setattr(hyde, "generate_hypotheses", fake)
        with pytest.raises(hyde.HydeError):
            hyde.discover(profile, SimpleNamespace(), tmp_path, goal=GOAL)
        assert seen["goal"] == GOAL
