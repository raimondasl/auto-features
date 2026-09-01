"""The augmented arm's wiring: what the agent may see, and what it must never see. [P27]

Arm C is Opus 5 with RepoRadar's MCP server attached. What makes it a controlled arm rather
than a new system is that its tool serves **exactly what arm A returned** — same papers,
same order, same gate scores — so a C-vs-A difference cannot be a different draw of
RepoRadar (NR-54 priced that at sd 1.44/case).

The tests here are almost all about one failure. The frozen results file records
``judge_score`` and ``judge_justification`` beside every pick, and seeding those would hand
the agent the answer key. **That failure leaves no trace**: the run completes, the picks are
plausible, the artifact is well-formed, and every downstream number is meaningless. It is
the only defect in this arm that could not be found after the fact, so it is the one with
the most tests in front of it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EVALS = ROOT / "evals"
if str(EVALS) not in sys.path:
    sys.path.insert(0, str(EVALS))

import baseline as baseline_mod  # noqa: E402
import rr_mcp_arm  # noqa: E402

ARM_A = rr_mcp_arm.ARM_A


# The frozen run files under `evals/results/` are gitignored -- the same asymmetry
# `freeze_opus5_arm.py` exists to close, and why derived facts live in tracked artifacts.
# So the tests that need the raw arm skip on a fresh clone, and every invariant that
# actually guards the experiment is ALSO exercised against a synthetic row, which runs
# everywhere. A leak guard that only fires on the maintainer's laptop is not a guard.
needs_arm_a = pytest.mark.skipif(
    not ARM_A.exists(), reason=f"{ARM_A.name} is gitignored; the synthetic cases still run"
)


@pytest.fixture(scope="module")
def arm_a_rows() -> list[dict]:
    return json.loads(ARM_A.read_text(encoding="utf-8"))


def _synthetic_arm(tmp_path: Path, **extra: object) -> Path:
    """One frozen-results row carrying the judge's verdict, written to a temp file."""
    row = {
        "case": "synthetic",
        "pool_provenance": {"pool_dir": None, "fingerprint": "x"},
        "ranking_config": {},
        "pool_config": {"rr_all_time": True},
        "returned": {
            "reporadar_toppicks": [
                {
                    "arxiv_id": "2401.00001v1",
                    "title": "T",
                    "llm_score": 3,
                    "judge_score": 3,
                    "judge_justification": "the answer",
                    **extra,
                }
            ]
        },
    }
    path = tmp_path / "arm.json"
    path.write_text(json.dumps([row]), encoding="utf-8")
    return path


class TestTheJudgeNeverReachesTheAgent:
    @needs_arm_a
    def test_arm_picks_drops_the_verdict_fields(self, arm_a_rows) -> None:
        for row in arm_a_rows:
            raw = row["returned"]["reporadar_toppicks"]
            if not raw:
                continue
            # The fields ARE in the source — a test that passed because the frozen file
            # happened not to carry them would be pinning nothing.
            assert any("judge_score" in p for p in raw)
            for pick in rr_mcp_arm.arm_picks(ARM_A, row["case"]):
                for forbidden in rr_mcp_arm.FORBIDDEN_FIELDS:
                    assert forbidden not in pick

    def test_it_is_an_allow_list_not_a_block_list(self) -> None:
        """A block-list is one schema change away from leaking. If the results file ever
        gains a third verdict field, an allow-list ignores it and a block-list ships it."""
        source = (EVALS / "rr_mcp_arm.py").read_text(encoding="utf-8")
        assert "{f: p[f] for f in PICK_FIELDS if f in p}" in source
        assert not any(f in rr_mcp_arm.PICK_FIELDS for f in rr_mcp_arm.FORBIDDEN_FIELDS)

    def test_an_unknown_field_is_dropped_rather_than_copied(self, tmp_path: Path) -> None:
        """The behaviour the allow-list buys, exercised rather than asserted about the
        source: a field nobody anticipated does not reach the agent. Synthetic on purpose
        — this is the version of the guard that runs on a fresh clone."""
        path = _synthetic_arm(tmp_path, some_future_verdict_field="also the answer")
        picks = rr_mcp_arm.arm_picks(path, "synthetic")
        assert picks == [{"arxiv_id": "2401.00001v1", "title": "T", "llm_score": 3}]


class TestTheStoreServesArmAsOutput:
    def test_a_missing_case_raises_rather_than_seeding_nothing(self, tmp_path: Path) -> None:
        """An empty store would give arm C an unattached MCP server and call the result a
        measurement — void, not null, in its most expensive form."""
        with pytest.raises(KeyError):
            rr_mcp_arm.arm_picks(_synthetic_arm(tmp_path), "no-such-case")

    @needs_arm_a
    def test_every_arm_a_case_has_picks_to_serve(self, arm_a_rows) -> None:
        for row in arm_a_rows:
            picks = rr_mcp_arm.arm_picks(ARM_A, row["case"])
            assert len(picks) == len(row["returned"]["reporadar_toppicks"])
            assert [p["arxiv_id"] for p in picks] == [
                p["arxiv_id"] for p in row["returned"]["reporadar_toppicks"]
            ]

    def test_the_pool_is_matched_by_fingerprint_not_by_filename(self) -> None:
        """A pool with the right name in the wrong directory would re-rank cleanly and seed
        a store serving papers arm A never saw."""
        source = (EVALS / "rr_mcp_arm.py").read_text(encoding="utf-8")
        assert 'stored["fingerprint"] != row["pool_provenance"]["fingerprint"]' in source


class TestTheToolsetIsAConfigurationAxis:
    def test_the_default_toolset_moves_no_existing_path_or_hash(self) -> None:
        """`web` is every published run. Adding the axis must leave their cache paths and
        discriminators byte-identical — the 2026-08-09 mistake was doing otherwise."""
        assert baseline_mod.DEFAULT_TOOLS == "web"
        assert baseline_mod.TOOLSETS["web"] == ()
        assert baseline_mod._cache_path("cli", "ann") == baseline_mod.CACHE_DIR / "cli" / "ann.json"
        assert baseline_mod._discriminator("cli", "", None) == baseline_mod._discriminator(
            "cli", "", None, tools="web"
        )

    def test_the_augmented_toolset_gets_its_own_path_and_hash(self) -> None:
        plain = baseline_mod._cache_path("cli", "ann", "v2", "claude-opus-5")
        augmented = baseline_mod._cache_path("cli", "ann", "v2", "claude-opus-5", tools="web+rr")
        assert plain != augmented
        assert baseline_mod._discriminator(
            "cli", "", None, "v2", "claude-opus-5"
        ) != baseline_mod._discriminator("cli", "", None, "v2", "claude-opus-5", tools="web+rr")

    def test_the_hash_covers_the_tool_names_not_the_config_path(self) -> None:
        """Two checkouts running the same arm must not look like different configurations
        just because their absolute paths differ."""
        a = baseline_mod.flags_for(tools="web+rr", mcp_config="/one/place/rag.mcp.json")
        b = baseline_mod.flags_for(tools="web+rr", mcp_config="/elsewhere/rag.mcp.json")
        assert a != b  # the flags differ...
        assert baseline_mod._discriminator(
            "cli", "", a, tools="web+rr"
        ) == baseline_mod._discriminator("cli", "", b, tools="web+rr")

    def test_a_typo_fails_loudly_instead_of_running_the_control(self) -> None:
        """A `web+rr` artifact full of plain-web draws is unfalsifiable from the rows: the
        treatment's absence is invisible. Same rule `prompt_for` already applies."""
        with pytest.raises(ValueError):
            baseline_mod.tools_for("web+reporadar")


class TestTheTreatmentCannotBeSilentlyAbsent:
    def test_allowed_tools_without_a_server_is_refused(self) -> None:
        """Claude Code does not report `--allowedTools mcp__…` with no server: the agent
        never sees the tools, answers normally, and the row lands in the augmented artifact
        having had no treatment. The most expensive failure available here."""
        with pytest.raises(ValueError, match="mcp_config"):
            baseline_mod.flags_for(tools="web+rr")

    def test_the_flags_carry_the_server_and_lock_out_every_other_one(self) -> None:
        flags = baseline_mod.flags_for(tools="web+rr", mcp_config="cfg.json")
        assert "--strict-mcp-config" in flags  # no inheriting the developer's own servers
        assert flags[flags.index("--mcp-config") + 1] == "cfg.json"
        allowed = flags[flags.index("--allowedTools") + 1].split(",")
        assert "WebSearch" in allowed and "WebFetch" in allowed  # B keeps everything it had
        assert set(baseline_mod.RR_MCP_TOOLS) <= set(allowed)

    def test_the_config_path_is_not_the_last_flag(self) -> None:
        """`--mcp-config` is VARIADIC. `_run_cli` appends the prompt after this list, so a
        list ending on the config path hands the CLI the PROMPT as a second config file and
        the run dies with "MCP config file not found: <cwd>/Please fetch and summarize...".

        Found on the first smoke run rather than by reading the help text, and pinned here
        because the symptom -- every row failing -- looks like an auth or quota problem."""
        flags = baseline_mod.flags_for(tools="web+rr", mcp_config="cfg.json")
        assert flags[-1] == "--strict-mcp-config"
        assert flags[flags.index("--mcp-config") + 2].startswith("--")

    def test_the_writing_tool_is_withheld(self) -> None:
        """`rate_paper` mutates the store this arm is served from. A treatment that edits
        its own input partway through is not the treatment it is named after."""
        assert not any(t.endswith("rate_paper") for t in baseline_mod.RR_MCP_TOOLS)

    def test_tool_use_is_read_from_the_server_not_inferred(self, tmp_path: Path) -> None:
        """A null result is unreadable without this: "the tool did not help" and "the agent
        never found the tool" call for opposite responses."""
        log = tmp_path / "calls.jsonl"
        log.write_text(
            json.dumps({"t": "2026-09-01T00:00:00+00:00", "tool": "get_ranked_papers"})
            + "\n"
            + json.dumps({"t": "2026-09-01T00:00:01+00:00", "tool": "get_ranked_papers"})
            + "\n"
            + json.dumps({"t": "2026-09-01T00:00:02+00:00", "tool": "explain_relevance"})
            + "\n",
            encoding="utf-8",
        )
        assert rr_mcp_arm.read_call_log(log) == {
            "n": 3,
            "by_tool": {"explain_relevance": 1, "get_ranked_papers": 2},
        }

    def test_no_log_means_zero_calls_which_is_a_real_answer(self, tmp_path: Path) -> None:
        """Zero is a finding about discoverability, not a missing measurement — the run's
        own `status` is what says whether it happened at all."""
        assert rr_mcp_arm.read_call_log(tmp_path / "absent.jsonl") == {"n": 0, "by_tool": {}}

    def test_each_run_gets_its_own_log(self, tmp_path: Path) -> None:
        """Two draws of one case sharing a log would attribute one run's calls to the
        other, which is worse than no log because it looks like data."""
        one, log_one = rr_mcp_arm.write_config("rag", repo_dir=tmp_path, token="aaaa1111")
        two, log_two = rr_mcp_arm.write_config("rag", repo_dir=tmp_path, token="bbbb2222")
        assert one != two and log_one != log_two
        cfg = json.loads(one.read_text(encoding="utf-8"))["mcpServers"]["reporadar"]
        assert cfg["env"]["RR_MCP_CALL_LOG"] == str(log_one)
        assert cfg["args"][0] == "mcp"


class TestTheArtifactsCannotCollide:
    def test_the_augmented_sweep_writes_its_own_file(self) -> None:
        """Rows are keyed `{draw}/{case}`, so a shared file would make `todo_baseline` find
        the augmented arm's work already done and run nothing — silently, with the control
        arm's numbers labelled as the treatment."""
        import gold_spread

        plain = gold_spread.out_path("v2", "claude-opus-5")
        augmented = gold_spread.out_path("v2", "claude-opus-5", "web+rr")
        assert plain != augmented
        assert plain.name == "gold_spread_v2_opus5.json"

    def test_opening_an_artifact_under_the_wrong_toolset_is_refused(self, tmp_path) -> None:
        """The strongest of the three guards, because it is the least visible: a wrongly
        opened prompt shows unfamiliar picks, a wrongly opened toolset shows nothing."""
        import gold_spread

        source = (EVALS / "gold_spread.py").read_text(encoding="utf-8")
        assert "was written with tools {found_tools!r}" in source
        assert gold_spread.load_artifact("v2", "claude-opus-5", "web+rr")["tools"] == "web+rr"
