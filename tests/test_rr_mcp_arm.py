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


class TestTheWideCorpusChangesExactlyOneTool:
    """C-wide seeds the whole frozen pool into `papers` while leaving `paper_scores` and
    `paper_llm_scores` holding only the picks. The two MCP tools then read different
    things, which is the whole trick:

    * `get_ranked_papers` -> `get_scores_for_run` -> only scored papers -> **unchanged**;
    * `search_papers` -> `get_all_papers` -> the whole corpus -> the variable under test.

    The claim is proved by serialising both payloads and comparing bytes, not by comparing
    the fields somebody thought to look at. On the 12 measured cases the corpus grows
    59x-369x and every payload is identical.
    """

    def _store(self, path: Path, extra_corpus: int):
        """A store with one scored+gated pick and *extra_corpus* unscored papers."""
        from reporadar.store import PaperStore

        with PaperStore(path) as store:
            store.upsert_paper(
                {
                    "arxiv_id": "2401.00001v1",
                    "title": "The pick",
                    "authors": ["A"],
                    "abstract": "A concrete method.",
                    "categories": ["cs.LG"],
                    "published": "2024-01-01T00:00:00+00:00",
                    "updated": "2024-01-01T00:00:00+00:00",
                    "url": "http://arxiv.org/abs/2401.00001v1",
                    "pdf_url": "http://arxiv.org/pdf/2401.00001v1",
                }
            )
            for i in range(extra_corpus):
                store.upsert_paper(
                    {
                        "arxiv_id": f"2402.{i:05d}v1",
                        "title": f"Corpus paper {i}",
                        "authors": ["B"],
                        "abstract": "Unranked this run.",
                        "categories": ["cs.LG"],
                        "published": "2024-02-01T00:00:00+00:00",
                        "updated": "2024-02-01T00:00:00+00:00",
                        "url": f"http://arxiv.org/abs/2402.{i:05d}v1",
                        "pdf_url": f"http://arxiv.org/pdf/2402.{i:05d}v1",
                    }
                )
            run_id = store.record_run(["frozen"], 1, 0)
            store.save_scores(run_id, [{"arxiv_id": "2401.00001v1", "score_total": 0.7}])
            store.save_llm_scores(run_id, {"2401.00001v1": {"llm_score": 3, "llm_reason": None}})

    def test_widening_the_corpus_leaves_the_recommendations_byte_identical(self, tmp_path):
        """The synthetic version of `compare_stores`, so the invariant is checked on a
        fresh clone rather than only where the seeded stores happen to exist."""
        from reporadar.mcp_server import ranked_papers_payload
        from reporadar.store import PaperStore

        self._store(tmp_path / "narrow.db", 0)
        self._store(tmp_path / "wide.db", 300)
        out = []
        for name in ("narrow.db", "wide.db"):
            with PaperStore(tmp_path / name) as store:
                payload = ranked_papers_payload(
                    store, limit=50, top_n=15, triage_threshold=2, rerank=True
                )
            payload.pop("run_id", None)
            out.append(json.dumps(payload, sort_keys=True))
        assert out[0] == out[1]

    def test_the_wider_corpus_does_reach_the_search_tool(self, tmp_path):
        """The other half: if `search_papers` did NOT widen, the arm would change nothing
        at all and a null result would be unreadable."""
        from reporadar.mcp_server import search_corpus_payload
        from reporadar.store import PaperStore

        self._store(tmp_path / "narrow.db", 0)
        self._store(tmp_path / "wide.db", 300)
        counts = []
        for name in ("narrow.db", "wide.db"):
            with PaperStore(tmp_path / name) as store:
                counts.append(store.paper_count())
                assert search_corpus_payload(store, "corpus paper", limit=50)["count"] == (
                    0 if name == "narrow.db" else 50
                )
        assert counts == [1, 301]

    def test_a_corpus_paper_is_invisible_to_the_recommendation_tool(self, tmp_path):
        """The load-bearing rule, and it is **stronger than I first wrote it down**.

        I argued that widening the corpus was safe because `categorize_papers` puts an
        unscored paper in Maybe at best, so it could never reach Top Picks. Writing this
        test showed the real reason is one layer earlier and absolute: `get_scores_for_run`
        JOINs `paper_scores`, and a corpus paper has no row there — so it is not merely
        tiered low, it **never enters the payload at all**, in any tier. The tiering rule
        is a second line of defence that this design never reaches.

        Which is why the assertion is that the wide payload has no `maybe_relevant` key
        rather than that its contents are harmless."""
        from reporadar.mcp_server import ranked_papers_payload
        from reporadar.store import PaperStore

        self._store(tmp_path / "wide.db", 40)
        with PaperStore(tmp_path / "wide.db") as store:
            payload = ranked_papers_payload(
                store, limit=50, top_n=15, triage_threshold=2, rerank=True
            )
        assert [p["arxiv_id"] for p in payload["papers"]] == ["2401.00001v1"]
        assert "maybe_relevant" not in payload and "muted" not in payload
        assert "2402." not in json.dumps(payload)

    def test_the_two_stores_are_separate_files(self) -> None:
        """Rebuilding one over the other would make "which corpus is installed" invisible
        state, and verifying either arm would destroy the other."""
        assert rr_mcp_arm.case_db("rag") != rr_mcp_arm.case_db("rag", wide=True)
        assert rr_mcp_arm.case_db("rag", wide=True).name == "papers-wide.db"

    def test_the_server_is_pointed_at_the_right_store(self, tmp_path: Path) -> None:
        cfg, _ = rr_mcp_arm.write_config("rag", repo_dir=tmp_path, wide=True)
        args = json.loads(cfg.read_text(encoding="utf-8"))["mcpServers"]["reporadar"]["args"]
        assert args[args.index("--db") + 1] == str(rr_mcp_arm.case_db("rag", wide=True))
        narrow, _ = rr_mcp_arm.write_config("rag", repo_dir=tmp_path)
        n_args = json.loads(narrow.read_text(encoding="utf-8"))["mcpServers"]["reporadar"]["args"]
        assert n_args[n_args.index("--db") + 1] == str(rr_mcp_arm.case_db("rag"))
        assert cfg != narrow  # and the two configs cannot overwrite each other


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
        out = rr_mcp_arm.read_call_log(log)
        assert out["n"] == 3
        assert out["by_tool"] == {"explain_relevance": 1, "get_ranked_papers": 2}
        # No pid on these lines, so the session count is UNKNOWN rather than 1.
        assert out["n_sessions"] is None

    def test_no_log_means_zero_calls_which_is_a_real_answer(self, tmp_path: Path) -> None:
        """Zero is a finding about discoverability, not a missing measurement — the run's
        own `status` is what says whether it happened at all."""
        assert rr_mcp_arm.read_call_log(tmp_path / "absent.jsonl") == {
            "n": 0,
            "by_tool": {},
            "n_sessions": 0,
        }

    def test_each_run_gets_its_own_log(self, tmp_path: Path) -> None:
        """Two draws of one case sharing a log would attribute one run's calls to the
        other, which is worse than no log because it looks like data."""
        one, log_one = rr_mcp_arm.write_config("rag", repo_dir=tmp_path, token="aaaa1111")
        two, log_two = rr_mcp_arm.write_config("rag", repo_dir=tmp_path, token="bbbb2222")
        assert one != two and log_one != log_two
        cfg = json.loads(one.read_text(encoding="utf-8"))["mcpServers"]["reporadar"]
        assert cfg["env"]["RR_MCP_CALL_LOG"] == str(log_one)
        assert cfg["args"][0] == "mcp"


class TestTheDriverActuallyServesTheArmItWasAskedFor:
    """Every one of these pins a defect an adversarial audit found in this change set
    **before the $86 sweep ran**, and that a `--dry-run` could not have found — a dry run
    returns before any of this code executes."""

    def test_the_corpus_is_read_off_the_toolset_not_guessed(self) -> None:
        """`gold_spread` accepted `--tools web+rrwide` and served the NARROW store: the
        plumbing never landed, the sweep would have run, every row would have looked
        normal, and the artifact would have answered a different question. The mapping now
        lives beside TOOLSETS so there is one source of truth for it."""
        assert baseline_mod.wide_corpus("web+rrwide") is True
        assert baseline_mod.wide_corpus("web+rr") is False
        assert baseline_mod.wide_corpus("web") is False
        with pytest.raises(ValueError):
            baseline_mod.wide_corpus("web+rrwyde")

    def test_each_toolset_points_the_server_at_its_own_store(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """Checked through the DRIVER rather than the helper, because the helper was always
        correct — it was the driver that never passed the bit, and only a test that goes in
        the front door would have caught that.

        Both stores are faked into a tmp tree rather than read from `evals/.work/`, which is
        gitignored: a guard against a $86 mistake that only fires on the maintainer's laptop
        is not a guard. The first version of this test did exactly that and CI caught it.
        """
        import gold_spread

        monkeypatch.setattr(rr_mcp_arm, "WORK_DIR", tmp_path)
        monkeypatch.setattr(rr_mcp_arm, "MCP_DIR", tmp_path / "mcp-arm")
        for wide in (False, True):
            db = rr_mcp_arm.case_db("acase", wide=wide)
            db.parent.mkdir(parents=True, exist_ok=True)
            db.write_bytes(b"")
        for tools, wide in (("web+rrwide", True), ("web+rr", False)):
            cfg, _log = gold_spread.mcp_config_for("acase", tools)
            args = json.loads(cfg.read_text(encoding="utf-8"))["mcpServers"]["reporadar"]["args"]
            assert args[args.index("--db") + 1] == str(rr_mcp_arm.case_db("acase", wide=wide))

    def test_the_guard_checks_the_store_the_toolset_asked_for(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """The audit's secondary: the missing-store guard read the NARROW path whatever the
        toolset, so a wide sweep launched without `--seed --wide` would have sailed past it
        and served the narrow corpus."""
        import gold_spread

        monkeypatch.setattr(rr_mcp_arm, "WORK_DIR", tmp_path)
        monkeypatch.setattr(rr_mcp_arm, "MCP_DIR", tmp_path / "mcp-arm")
        narrow = rr_mcp_arm.case_db("acase")
        narrow.parent.mkdir(parents=True, exist_ok=True)
        narrow.write_bytes(b"")  # only the NARROW store exists
        with pytest.raises(SystemExit, match="papers-wide.db"):
            gold_spread.mcp_config_for("acase", "web+rrwide")

    def test_the_end_of_run_message_names_the_file_it_wrote(self) -> None:
        """`report()` prints no filename and the correct one is printed at the START of the
        sweep, so this is the only end-of-run name a reader sees — and it was announcing
        the CONTROL arm's file after writing the treatment's."""
        source = (EVALS / "gold_spread.py").read_text(encoding="utf-8")
        assert (
            "wrote {out_path(args.prompt_version, args.baseline_model, args.tools).name}" in source
        )


class TestTheCallLogSurvivesARetry:
    def test_sessions_are_partitioned_by_the_server_process(self, tmp_path: Path) -> None:
        """`_run_cli` retries a failed `claude` against the SAME log path and each attempt
        spawns its own server, so the raw total pools a dead attempt's calls with the
        surviving one's. `_run_cli` returns on the first success, so the LAST session is
        the one that produced the answer."""
        log = tmp_path / "calls.jsonl"
        log.write_text(
            "\n".join(
                json.dumps({"t": f"2026-09-01T00:00:0{i}+00:00", "pid": pid, "tool": tool})
                for i, (pid, tool) in enumerate(
                    [(100, "get_ranked_papers"), (100, "search_papers"), (200, "search_papers")]
                )
            )
            + "\n",
            encoding="utf-8",
        )
        out = rr_mcp_arm.read_call_log(log)
        assert out["n"] == 3 and out["n_sessions"] == 2
        assert out["n_last_session"] == 1
        assert out["by_tool_last_session"] == {"search_papers": 1}

    def test_a_single_session_carries_no_partition_keys(self, tmp_path: Path) -> None:
        """Absent, not present-and-equal. A row that says `n_last_session` is a row where
        something went wrong, and it should be greppable."""
        log = tmp_path / "calls.jsonl"
        log.write_text(
            json.dumps({"t": "2026-09-01T00:00:00+00:00", "pid": 7, "tool": "search_papers"})
            + "\n",
            encoding="utf-8",
        )
        out = rr_mcp_arm.read_call_log(log)
        assert out["n_sessions"] == 1 and "n_last_session" not in out

    def test_a_legacy_log_reports_unknown_rather_than_one(self, tmp_path: Path) -> None:
        """Void, not null. Arm C's logs predate the pid; calling them "one session" would
        assert something never measured. (They show no restart signature either — the
        largest inter-call gap on any of the 12 is 172 s, against the minutes a failed
        attempt costs — so the published counts stand, but on evidence, not assumption.)"""
        log = tmp_path / "calls.jsonl"
        log.write_text(
            json.dumps({"t": "2026-09-01T00:00:00+00:00", "tool": "search_papers"}) + "\n",
            encoding="utf-8",
        )
        assert rr_mcp_arm.read_call_log(log)["n_sessions"] is None


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
