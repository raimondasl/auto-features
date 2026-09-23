"""Pin the identifier-line probe: its population, its two arms, and its registered arithmetic.

`evals/sonnet_id_probe.py` executes `evals/PREREG-sonnet-id-probe.md`. Under NR-52's second
judge the baseline's arXiv picks were shown unversioned while the first judge saw them
versioned, so the identifier line differed by arm under one judge only. The probe re-judges
those picks under both id forms and reads the difference by a rule fixed in advance.

Two halves. The first exercises the pure functions on small synthetic fixtures: which picks
are in the population, how each arm's id is built (including the one registered paper with two
versions), the order the calls are made in, the checks made before any call, the cap, retry,
outage and stop rules, and the endpoint arithmetic with the reading rule at its boundaries. The
second pins the tracked artifact once it exists. It is skipped while
`evals/sonnet_id_probe.json` is absent, and only then, because the verdicts it is built from
have not been bought yet.

Everything here is offline. The tests that drive `second_verdict` or `llm_client` replace the
network call.
"""

from __future__ import annotations

import email.message
import io
import itertools
import json
import os
import sys
import urllib.error
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evals"))

import judge as judge_mod  # noqa: E402
import sonnet_id_probe as probe  # noqa: E402

from anonymous import llm_client  # noqa: E402
from anonymous.config import SuggestionsConfig  # noqa: E402
from anonymous.paper_id import dedup_id  # noqa: E402

ARTIFACT = ROOT / "evals" / "sonnet_id_probe.json"

# ── Fixtures ──────────────────────────────────────────────────────────────────────────────

OPUS: dict[str, list[dict[str, Any]]] = {
    "db": [
        {"arxiv_id": "2607.11271", "judge_score": 2},
        {"arxiv_id": "2404.14989", "judge_score": 1},
        {"arxiv_id": "doi:10.1145/3471621.3471846", "judge_score": 3},
    ],
    "old": [{"arxiv_id": "math/0612488", "judge_score": 2}],
}

GOLD_FILES = {
    "db": [
        "2607.11271v2.json",
        "2607.11271v3.json",
        "2404.14989v1.json",
        # Decoys: an unversioned file, a longer id sharing the prefix, and a malformed suffix.
        "2404.14989.json",
        "2404.149891v1.json",
        "2404.14989v1x.json",
        "doi_10.1145_3471621.3471846.json",
    ],
    "old": ["math_0612488v1.json"],
}

NR52 = {
    ("db", "2607.11271"): 3,
    ("db", "2404.14989"): 0,
    ("db", "doi_10.1145_3471621.3471846"): 2,
    ("old", "math_0612488"): 1,
}

MULTI = {("db", "2607.11271"): "2607.11271v3"}


def _gold(root: Path, extra: dict[str, list[str]] | None = None) -> Path:
    gold = root / "gold"
    for case, names in {**GOLD_FILES, **(extra or {})}.items():
        (gold / case).mkdir(parents=True, exist_ok=True)
        for name in names:
            (gold / case / name).write_text('{"score": 2}', encoding="utf-8")
    return gold


def _row(
    case: str,
    pick: str,
    *,
    gpt: int = 2,
    nr52: int = 2,
    v: int | None = 2,
    u: int | None = 2,
    void_v: bool | None = None,
    void_u: bool | None = None,
    early: bool = False,
) -> dict[str, Any]:
    return {
        "case": case,
        "pick": pick,
        "versioned": f"{pick}v1",
        "gpt": gpt,
        "nr52": nr52,
        "nr52_early": early,
        "v": v,
        "u": u,
        "void_v": v is None if void_v is None else void_v,
        "void_u": u is None if void_u is None else void_u,
    }


def _quiet(_: str) -> None:
    return None


# ── Population ────────────────────────────────────────────────────────────────────────────


class TestThePopulation:
    def test_it_is_the_arxiv_picks_with_their_gold_versions(self, tmp_path: Path) -> None:
        rows = probe.build_population(OPUS, OPUS, NR52, gold=_gold(tmp_path), multi=MULTI)
        assert rows == [
            {
                "case": "db",
                "pick": "2404.14989",
                "versioned": "2404.14989v1",
                "gpt": 1,
                "nr52": 0,
            },
            {
                "case": "db",
                "pick": "2607.11271",
                "versioned": "2607.11271v3",
                "gpt": 2,
                "nr52": 3,
            },
            {
                "case": "old",
                "pick": "math/0612488",
                "versioned": "math/0612488v1",
                "gpt": 2,
                "nr52": 1,
            },
        ]

    def test_the_doi_pick_is_not_in_it(self, tmp_path: Path) -> None:
        """Its identifier line is the same under both recipes, so it cannot be tested here."""
        rows = probe.build_population(OPUS, OPUS, NR52, gold=_gold(tmp_path), multi=MULTI)
        assert all(not r["pick"].startswith("doi:") for r in rows)

    def test_the_newer_of_two_registered_versions_is_used(self, tmp_path: Path) -> None:
        gold = _gold(tmp_path)
        assert probe.gold_versions("db", "2607.11271", gold) == ["2607.11271v2", "2607.11271v3"]
        both = ["2607.11271v2", "2607.11271v3"]
        assert probe.choose_versioned("db", "2607.11271", both, MULTI) == "2607.11271v3"
        rows = probe.build_population(OPUS, OPUS, NR52, gold=gold, multi=MULTI)
        assert {r["pick"]: r["versioned"] for r in rows}["2607.11271"] == "2607.11271v3"

    def test_the_decoy_files_are_ignored(self, tmp_path: Path) -> None:
        assert probe.gold_versions("db", "2404.14989", _gold(tmp_path)) == ["2404.14989v1"]

    def test_an_unregistered_second_version_is_refused(self, tmp_path: Path) -> None:
        gold = _gold(tmp_path, {"db": [*GOLD_FILES["db"], "2404.14989v2.json"]})
        with pytest.raises(ValueError, match="unregistered"):
            probe.build_population(OPUS, OPUS, NR52, gold=gold, multi=MULTI)

    def test_the_registered_version_must_be_the_newest(self, tmp_path: Path) -> None:
        wrong = {("db", "2607.11271"): "2607.11271v2"}
        with pytest.raises(ValueError, match="registered"):
            probe.build_population(OPUS, OPUS, NR52, gold=_gold(tmp_path), multi=wrong)

    def test_the_registered_special_case_must_occur(self, tmp_path: Path) -> None:
        """Exactly one special case: a registered one that never appears is refused too."""
        more = {**MULTI, ("old", "math/0612488"): "math/0612488v2"}
        with pytest.raises(ValueError, match="several gold versions"):
            probe.build_population(OPUS, OPUS, NR52, gold=_gold(tmp_path), multi=more)

    def test_a_missing_nr52_verdict_is_refused(self, tmp_path: Path) -> None:
        nr52 = {k: v for k, v in NR52.items() if k != ("db", "2404.14989")}
        with pytest.raises(ValueError, match="NR-52"):
            probe.build_population(OPUS, OPUS, nr52, gold=_gold(tmp_path), multi=MULTI)

    def test_a_pick_without_a_gold_file_is_refused(self, tmp_path: Path) -> None:
        opus = {**OPUS, "old": [{"arxiv_id": "hep-th/9901001", "judge_score": 2}]}
        nr52 = {**NR52, ("old", "hep-th_9901001"): 1}
        with pytest.raises(ValueError, match="no versioned verdict"):
            probe.build_population(opus, opus, nr52, gold=_gold(tmp_path), multi=MULTI)

    def test_a_versioned_pick_is_refused(self, tmp_path: Path) -> None:
        opus = {"old": [{"arxiv_id": "math/0612488v1", "judge_score": 2}]}
        with pytest.raises(ValueError, match="unversioned"):
            probe.build_population(opus, opus, NR52, gold=_gold(tmp_path), multi={})

    def test_the_real_special_case_is_the_registered_one(self) -> None:
        assert probe.MULTI_VERSION == {("db", "2607.11271"): "2607.11271v3"}

    def test_the_registered_counts(self) -> None:
        assert (probe.EXPECTED_CASES, probe.EXPECTED_JUDGED, probe.EXPECTED_POPULATION) == (
            37,
            357,
            237,
        )


class TestTheEarlyNR52Draws:
    """E3's subset: the 11 NR-52 verdicts an earlier probe drew, found by file mtime."""

    def test_they_are_found_by_the_cache_files_mtime(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import second_judge

        monkeypatch.setattr(second_judge, "CACHE", tmp_path)
        rows = [
            {"case": "a", "pick": "2401.00001"},
            {"case": "a", "pick": "math/0612488"},
            {"case": "b", "pick": "2401.00003"},
        ]
        when = [
            datetime(2026, 8, 6, 21, 20, tzinfo=UTC),
            datetime(2026, 8, 30, 23, 59, tzinfo=UTC),
            datetime(2026, 8, 31, 0, 0, tzinfo=UTC),  # the cutoff itself is not early
        ]
        for r, w in zip(rows, when, strict=True):
            f = second_judge.second_cache_path(probe.DEFAULT_MODEL, r["case"], r["pick"])
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text('{"score": 2}', encoding="utf-8")
            os.utime(f, (w.timestamp(), w.timestamp()))
        assert probe.early_nr52(rows) == [("a", "2401.00001"), ("a", "math/0612488")]

    def test_the_registered_eleven(self) -> None:
        cutoff = datetime(2026, 8, 31, tzinfo=UTC)
        assert cutoff == probe.NR52_EARLY_BEFORE
        assert sorted(f"{c}/{p}" for c, p in probe.NR52_EARLY) == [
            "cv/1704.04503",
            "cv/2012.07177",
            "graph/2303.06147",
            "rag/2304.01982",
            "rag/2501.17788",
            "rag/2606.05568",
            "rl/1509.06461",
            "rl/1511.05952",
            "speech/2211.17192",
            "speech/2303.00747",
            "speech/2311.00430",
        ]

    def test_anything_but_exactly_those_is_refused(self) -> None:
        found = sorted(probe.NR52_EARLY)
        assert probe.early_refusal(found) is None
        assert "missing" in str(probe.early_refusal(found[1:]))
        assert "unexpected" in str(probe.early_refusal([*found, ("ann", "2104.03221")]))
        swapped = [*found[1:], ("ann", "2104.03221")]
        assert probe.early_refusal(swapped) is not None


class TestTheArms:
    ROW = {"case": "db", "pick": "2607.11271", "versioned": "2607.11271v3", "gpt": 2, "nr52": 3}
    RECORD = {"status": "ok", "resolved_id": "2607.11271v4", "title": "T", "abstract": "A " * 1200}

    def test_v_is_the_gold_version_and_u_is_the_pick(self) -> None:
        assert probe.arm_id(self.ROW, "v") == "2607.11271v3"
        assert probe.arm_id(self.ROW, "u") == "2607.11271"

    def test_the_shared_rule_says_both_ids_are_one_paper(self) -> None:
        for row in (self.ROW, {**self.ROW, "pick": "math/0612488", "versioned": "math/0612488v1"}):
            assert dedup_id(probe.arm_id(row, "v")) == probe.arm_id(row, "u")

    def test_the_identifier_lines_are_the_two_recipes(self) -> None:
        v = probe.arm_paper(self.ROW, self.RECORD, "v")
        u = probe.arm_paper(self.ROW, self.RECORD, "u")
        assert judge_mod._identifier_line(v) == "arXiv: 2607.11271v3"
        assert judge_mod._identifier_line(u) == "arXiv: 2607.11271"

    def test_the_resolver_version_never_reaches_the_prompt(self) -> None:
        """The stored record's own version is recorded, not sent: V sends the gold version."""
        for arm in probe.ARMS:
            assert "v4" not in probe.arm_paper(self.ROW, self.RECORD, arm)["arxiv_id"]

    def test_the_prompts_differ_only_in_the_identifier_line(self) -> None:
        diff = probe.prompt_difference("CTX", self.ROW, self.RECORD)
        assert diff == [("arXiv: 2607.11271v3", "arXiv: 2607.11271")]
        assert probe.only_the_id_line_differs("CTX", self.ROW, self.RECORD)

    def test_identical_prompts_fail_the_check(self) -> None:
        """A probe whose two arms send the same bytes measures nothing."""
        row = {**self.ROW, "versioned": self.ROW["pick"]}
        assert not probe.only_the_id_line_differs("CTX", row, self.RECORD)

    def test_sent_prompt_is_what_second_verdict_sends(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`sent_prompt` copies an inline expression; this catches the copy drifting.

        It also checks the V namespace round trip: a verdict written under the versioned id
        reads back through the same reader `--report` uses, and not under the pick id.
        """
        import rung1_second_judge
        import second_judge

        sent: list[str] = []

        def fake_complete(prompt: str, cfg: Any, **_: Any) -> str:
            sent.append(prompt)
            return '{"score": 2, "justification": "j", "proposed_change": "c"}'

        monkeypatch.setattr(second_judge, "complete", fake_complete)
        monkeypatch.setattr(second_judge, "CACHE", tmp_path / "second_judge")
        monkeypatch.setattr(rung1_second_judge, "WORK", tmp_path)
        paper = probe.arm_paper(self.ROW, self.RECORD, "v")
        ns = probe.NAMESPACE["v"]
        assert second_judge.second_verdict("db", "CTX", paper, probe.DEFAULT_MODEL, cache_as=ns)
        assert sent == [probe.sent_prompt("CTX", paper)]
        cache = probe.cached_sonnet(ns)
        assert probe.son_of(cache, "db", "2607.11271v3") == 2
        assert probe.son_of(cache, "db", "2607.11271") is None
        assert probe.cached_sonnet(probe.NAMESPACE["u"]) == {}

    def test_the_namespaces_are_outside_nr52s(self) -> None:
        assert probe.NAMESPACE == {
            "v": "claude-sonnet-5#id-versioned",
            "u": "claude-sonnet-5#id-unversioned",
        }
        assert probe.DEFAULT_MODEL not in probe.NAMESPACE.values()


class TestTheCallOrder:
    def test_arms_alternate_by_paper(self) -> None:
        assert [probe.arm_order(i) for i in range(4)] == [
            ("v", "u"),
            ("u", "v"),
            ("v", "u"),
            ("u", "v"),
        ]

    def test_both_arms_of_a_paper_are_back_to_back_and_voids_keep_their_index(self) -> None:
        rows = [_row("a", f"2401.0000{i}") for i in range(4)]
        ok = {"status": "ok", "resolved_id": "x", "title": "T", "abstract": "A"}
        papers = {"a": {r["pick"]: dict(ok) for r in rows}}
        papers["a"]["2401.00001"] = {"status": "no_abstract", "title": "", "abstract": ""}
        tasks = probe.call_plan(rows, papers)
        assert [(t["index"], t["arm"]) for t in tasks] == [
            (0, "v"),
            (0, "u"),
            (2, "v"),
            (2, "u"),
            (3, "u"),
            (3, "v"),
        ]

    def test_both_arms_share_one_stored_text(self) -> None:
        rows = [_row("a", "2401.00001")]
        rec = {"status": "ok", "resolved_id": "2401.00001v2", "title": "T", "abstract": "A"}
        v, u = probe.call_plan(rows, {"a": {"2401.00001": rec}})
        assert (v["paper"]["title"], v["paper"]["abstract"]) == ("T", "A")
        assert (u["paper"]["title"], u["paper"]["abstract"]) == ("T", "A")
        assert v["paper"]["arxiv_id"] != u["paper"]["arxiv_id"]


class TestResolution:
    ROWS = [
        {"case": "a", "pick": "2401.00001"},
        {"case": "a", "pick": "2401.00002"},
        {"case": "b", "pick": "2401.00003"},
    ]

    def test_once_per_paper_grouped_by_case(self) -> None:
        asked: list[list[str]] = []

        def resolver(ids: list[str]) -> list[dict[str, Any]]:
            asked.append(list(ids))
            return [
                {"arxiv_id": "2401.00001v2", "title": "T1", "abstract": "A1"},
                {"arxiv_id": "2401.00003v1", "title": "T3", "abstract": "  "},
            ]

        papers = probe.resolve_missing(self.ROWS, {}, resolver, save=lambda p: None, log=_quiet)
        assert asked == [["2401.00001", "2401.00002"], ["2401.00003"]]
        assert papers["a"]["2401.00001"] == {
            "status": "ok",
            "resolved_id": "2401.00001v2",
            "title": "T1",
            "abstract": "A1",
        }
        assert papers["a"]["2401.00002"] == {"status": "unresolved"}
        assert papers["b"]["2401.00003"]["status"] == "no_abstract"

        # A rerun asks only for what is unresolved. A stored text is never fetched again.
        asked.clear()
        probe.resolve_missing(self.ROWS, papers, resolver, save=lambda p: None, log=_quiet)
        assert asked == [["2401.00002"]]
        assert papers["a"]["2401.00001"]["abstract"] == "A1"

    def test_a_resolver_failure_leaves_papers_unresolved_not_void(self) -> None:
        def broken(ids: list[str]) -> list[dict[str, Any]]:
            raise RuntimeError("arXiv said 503")

        papers = probe.resolve_missing(self.ROWS, {}, broken, save=lambda p: None, log=_quiet)
        assert {r["status"] for c in papers.values() for r in c.values()} == {"unresolved"}


class TestAtomicWrites:
    def test_the_rename_is_retried_while_windows_refuses_it(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        real = os.replace
        refusals = {"left": 2}

        def flaky(src: Any, dst: Any) -> None:
            if refusals["left"]:
                refusals["left"] -= 1
                raise PermissionError("[WinError 5] Access is denied")
            real(src, dst)

        monkeypatch.setattr(probe.os, "replace", flaky)
        monkeypatch.setattr(probe.time, "sleep", lambda s: None)
        target = tmp_path / "ledger.json"
        probe._write_json(target, {"calls": 1})
        assert json.loads(target.read_text(encoding="utf-8")) == {"calls": 1}
        assert refusals["left"] == 0
        assert list(tmp_path.iterdir()) == [target]

    def test_a_refusal_that_never_ends_is_raised(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def never(src: Any, dst: Any) -> None:
            raise PermissionError("[WinError 5] Access is denied")

        monkeypatch.setattr(probe.os, "replace", never)
        monkeypatch.setattr(probe.time, "sleep", lambda s: None)
        with pytest.raises(PermissionError):
            probe._write_json(tmp_path / "x.json", {})


# ── Before any call ───────────────────────────────────────────────────────────────────────


class TestBeforeAnyCall:
    def test_registration_removes_the_temperature_field(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """What NR-52 sent: no temperature field, on the first call as on every other."""
        monkeypatch.setattr(llm_client, "_REJECTS_TEMPERATURE", set())
        sent: list[dict[str, Any]] = []

        def capture(body: dict[str, Any], api_key: str, timeout: int) -> str:
            sent.append(dict(body))
            return "{}"

        monkeypatch.setattr(llm_client, "_post_claude", capture)
        assert probe.temperature_refusal() is not None
        llm_client._call_claude("p", "k", probe.DEFAULT_MODEL, 10, 10)
        assert sent[-1]["temperature"] == 0  # what an unregistered first call sends
        probe.register_no_temperature()
        assert probe.temperature_refusal() is None
        llm_client._call_claude("p", "k", probe.DEFAULT_MODEL, 10, 10)
        assert "temperature" not in sent[-1]

    def test_the_key_check_is_the_clients_own_resolution(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ANONYMOUS_CONFIG_DIR", str(tmp_path))
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        assert probe.anthropic_key_resolves() is False
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        assert probe.anthropic_key_resolves() is True

    def test_nr52_must_reproduce_to_two_places(self) -> None:
        assert probe.NR52_MARGIN == {"sonnet_only": -3.41, "consensus": 0.57}
        assert probe.nr52_refusal({"sonnet_only": -3.4149, "consensus": 0.5712}) is None
        assert probe.nr52_refusal({"sonnet_only": -3.4060, "consensus": 0.5660}) is None
        assert "sonnet_only" in str(probe.nr52_refusal({"sonnet_only": -3.40, "consensus": 0.57}))
        assert "consensus" in str(probe.nr52_refusal({"sonnet_only": -3.41}))

    def test_the_case_list_is_recorded_and_then_fixed(self) -> None:
        assert "no case list" in str(probe.case_list_refusal(None, ["a", "b"]))
        assert probe.case_list_refusal(["a", "b"], ["a", "b"]) is None
        why = probe.case_list_refusal(["a", "b", "c"], ["a", "b"])
        assert why is not None and "['c']" in why


# ── Buying ────────────────────────────────────────────────────────────────────────────────


def _tasks(n: int) -> list[dict[str, Any]]:
    return [
        {"index": i, "case": "c", "pick": f"p{i}", "arm": "v", "paper": {"arxiv_id": f"p{i}v1"}}
        for i in range(n)
    ]


def _ledger(calls: int = 0) -> dict[str, Any]:
    return {"calls": calls, "failures": {}, "errors": []}


def _buy(tasks: list[dict[str, Any]], call: Any, ledger: dict[str, Any], **kw: Any) -> dict:
    kw.setdefault("is_cached", lambda t: False)
    return probe.buy(tasks, call, ledger, save=lambda led: None, log=_quiet, **kw)


def _raises(exc: BaseException) -> Any:
    def call(t: dict[str, Any]) -> int:
        raise exc

    return call


def _http_failure(code: int, body: bytes = b"") -> llm_client.LLMError:
    """An HTTP refusal as `llm_client.complete` hands it on: an LLMError raised from it."""
    http = urllib.error.HTTPError(
        "https://api.anthropic.com/v1/messages",
        code,
        "refused",
        email.message.Message(),
        io.BytesIO(body),
    )
    try:
        raise llm_client.LLMError(f"LLM HTTP {code}: {http}") from http
    except llm_client.LLMError as exc:
        return exc


CREDIT_400 = (
    b'{"type":"error","error":{"type":"invalid_request_error","message":"Your credit balance '
    b'is too low to access the Anthropic API. Please go to Plans & Billing."}}'
)


class TestBuying:
    def test_a_cached_verdict_is_not_a_call(self) -> None:
        made: list[Any] = []
        out = _buy(_tasks(3), made.append, (led := _ledger()), is_cached=lambda t: True)
        assert made == [] and led["calls"] == 0 and out["reason"] == "done"

    def test_two_retries_then_a_verdict(self) -> None:
        attempts = {"n": 0}

        def flaky(t: dict[str, Any]) -> int:
            attempts["n"] += 1
            if attempts["n"] < 3:
                raise ValueError("no JSON object in response")
            return 2

        led = _ledger()
        out = _buy(_tasks(1), flaky, led)
        assert out == {"reason": "done", "bought": 1, "void": 0}
        assert led["calls"] == 3
        assert probe.failures_of(led, "c", "p0", "v") == 2

    def test_three_failures_make_that_arm_void(self) -> None:
        def call(t: dict[str, Any]) -> int:
            if t["pick"] == "p0":
                raise KeyError("score")
            return 1

        led = _ledger()
        out = _buy(_tasks(2), call, led)
        assert out == {"reason": "done", "bought": 1, "void": 1}
        assert led["calls"] == 4
        assert probe.failures_of(led, "c", "p0", "v") == probe.MAX_ATTEMPTS == 3

    def test_a_resumed_run_does_not_refill_spent_retries(self) -> None:
        led = _ledger(calls=3)
        led["failures"] = {"c": {"p0": {"v": 3}}}
        made: list[str] = []
        out = _buy(_tasks(2), lambda t: made.append(t["pick"]) or 2, led)
        assert made == ["p1"]
        assert out["void"] == 1 and led["calls"] == 4

    def test_the_601st_call_is_refused(self) -> None:
        assert probe.CALL_CAP == 600
        made: list[str] = []
        led = _ledger(calls=599)
        out = _buy(_tasks(3), lambda t: made.append(t["pick"]) or 2, led)
        assert out["reason"] == "cap"
        assert made == ["p0"] and led["calls"] == 600

    def test_retries_count_against_the_cap(self) -> None:
        led = _ledger(calls=598)
        out = _buy(_tasks(1), _raises(ValueError("x")), led)
        assert out["reason"] == "cap" and led["calls"] == 600

    def test_twenty_consecutive_failures_stop_the_run_and_are_refunded(self) -> None:
        led = _ledger()
        out = _buy(_tasks(10), _raises(ConnectionError("down")), led)
        assert probe.STOP_AFTER == 20
        # Twenty calls counted against the cap; no paper left void by the outage.
        assert out == {"reason": "stopped", "bought": 0, "void": 0}
        assert led["calls"] == 20 and led["refunded"] == 20 and led["streak"] == []
        assert [probe.failures_of(led, "c", f"p{i}", "v") for i in range(10)] == [0] * 10

        # Resumed after the outage, every paper has its full retries again.
        out = _buy(_tasks(10), lambda t: 2, led)
        assert out == {"reason": "done", "bought": 10, "void": 0}
        assert led["calls"] == 30

    def test_a_shorter_run_of_failures_stays_charged(self) -> None:
        def call(t: dict[str, Any]) -> int:
            if t["pick"] == "p0":
                raise ValueError("score out of range: 7")
            return 2

        led = _ledger()
        out = _buy(_tasks(2), call, led, stop_after=4)
        assert out == {"reason": "done", "bought": 1, "void": 1}
        assert probe.failures_of(led, "c", "p0", "v") == 3
        assert led["streak"] == [] and led.get("refunded", 0) == 0

    def test_the_streak_survives_a_crash_and_a_resume(self) -> None:
        """Consecutive means consecutive calls, not consecutive within one process."""
        seen = {"n": 0}

        def fail_then_crash(t: dict[str, Any]) -> int:
            seen["n"] += 1
            if seen["n"] <= 2:
                raise ValueError("no JSON object in response")
            raise KeyboardInterrupt

        led = _ledger()
        with pytest.raises(KeyboardInterrupt):
            _buy(_tasks(1), fail_then_crash, led, stop_after=3)
        assert len(led["streak"]) == 2 and probe.failures_of(led, "c", "p0", "v") == 2

        out = _buy(_tasks(1), _raises(ValueError("again")), led, stop_after=3)
        assert out["reason"] == "stopped" and led["refunded"] == 3
        assert probe.failures_of(led, "c", "p0", "v") == 0

    def test_a_success_resets_the_streak(self) -> None:
        seen: set[str] = set()

        def first_fails(t: dict[str, Any]) -> int:
            if t["pick"] not in seen:
                seen.add(t["pick"])
                raise ValueError("once")
            return 2

        led = _ledger()
        out = _buy(_tasks(30), first_fails, led)
        assert out["reason"] == "done" and led["calls"] == 60
        assert all(probe.failures_of(led, "c", f"p{i}", "v") == 1 for i in range(30))

    @pytest.mark.parametrize(
        "exc",
        [
            llm_client.LLMUnavailable("No Claude API key."),
            _http_failure(401),
            _http_failure(402),
            _http_failure(403),
            _http_failure(400, CREDIT_400),
        ],
        ids=["no-key", "401", "402", "403", "credit-400"],
    )
    def test_a_failure_not_about_the_paper_stops_uncharged(self, exc: Exception) -> None:
        led = _ledger()
        out = _buy(_tasks(3), _raises(exc), led)
        assert out["reason"] == "unavailable"
        assert led["calls"] == 1
        assert probe.failures_of(led, "c", "p0", "v") == 0 and led["streak"] == []

    @pytest.mark.parametrize(
        "exc",
        [
            ValueError("no JSON object in response: the credit balance of a billing system"),
            KeyError("score"),
            ValueError("score out of range: 5"),
            _http_failure(400, b'{"error":{"message":"prompt is too long"}}'),
            llm_client.LLMError("LLM call failed after 3 attempts: HTTP Error 529: Overloaded"),
            llm_client.LLMRateLimited("LLM call failed after 3 attempts: HTTP Error 429"),
        ],
        ids=["parse", "missing-score", "range", "other-400", "5xx", "429"],
    )
    def test_any_other_failure_is_charged_to_the_paper(self, exc: Exception) -> None:
        assert probe.not_about_the_paper(exc) is None
        led = _ledger()
        out = _buy(_tasks(1), _raises(exc), led)
        assert out == {"reason": "done", "bought": 0, "void": 1}
        assert probe.failures_of(led, "c", "p0", "v") == 3

    def test_the_clients_own_refusals_are_recognised(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Through `llm_client.complete` itself, with the temperature registered as in --judge."""
        monkeypatch.setattr(llm_client, "_REJECTS_TEMPERATURE", {probe.DEFAULT_MODEL})
        cfg = SuggestionsConfig(
            provider="claude", claude_api_key="k", claude_model=probe.DEFAULT_MODEL
        )

        def refuse(code: int, body: bytes) -> Any:
            def post(b: dict[str, Any], api_key: str, timeout: int) -> str:
                raise urllib.error.HTTPError(
                    "https://api.anthropic.com/v1/messages",
                    code,
                    "refused",
                    email.message.Message(),
                    io.BytesIO(body),
                )

            return post

        for code, body, systemic in (
            (401, b'{"error":{"type":"authentication_error"}}', True),
            (400, CREDIT_400, True),
            (400, b'{"error":{"type":"invalid_request_error","message":"bad"}}', False),
        ):
            monkeypatch.setattr(llm_client, "_post_claude", refuse(code, body))
            with pytest.raises(llm_client.LLMError) as info:
                llm_client.complete("p", cfg, max_retries=0)
            assert (probe.not_about_the_paper(info.value) is not None) is systemic, code

    def test_a_refused_check_makes_no_call(self) -> None:
        made: list[str] = []
        led = _ledger()
        out = _buy(_tasks(2), lambda t: made.append(t["pick"]) or 2, led, check=lambda: "no")
        assert out["reason"] == "refused" and made == [] and led["calls"] == 0

    def test_the_check_runs_before_every_call(self) -> None:
        checks: list[int] = []
        led = _ledger()
        _buy(_tasks(3), lambda t: 2, led, check=lambda: checks.append(led["calls"]) or None)
        assert checks == [0, 1, 2]

    def test_a_call_is_counted_before_it_is_made(self) -> None:
        """A crash inside the call must still count against the cap on resume."""
        saved: list[int] = []

        def crash(t: dict[str, Any]) -> int:
            raise KeyboardInterrupt

        led = _ledger()
        with pytest.raises(KeyboardInterrupt):
            probe.buy(
                _tasks(1),
                crash,
                led,
                is_cached=lambda t: False,
                save=lambda x: saved.append(x["calls"]),
                log=_quiet,
            )
        assert saved == [1]


class TestScoringRows:
    def test_voids_and_missing(self) -> None:
        rows = [
            {"case": "a", "pick": "2401.00001", "versioned": "2401.00001v1", "gpt": 2, "nr52": 2},
            {"case": "a", "pick": "2401.00002", "versioned": "2401.00002v2", "gpt": 1, "nr52": 1},
            {"case": "a", "pick": "2401.00003", "versioned": "2401.00003v1", "gpt": 3, "nr52": 3},
        ]
        ok = {"status": "ok", "resolved_id": "x", "title": "T", "abstract": "A"}
        papers = {
            "a": {
                "2401.00001": ok,
                "2401.00002": {"status": "no_abstract", "title": "", "abstract": ""},
                "2401.00003": ok,
            }
        }
        ledger = {"calls": 3, "failures": {"a": {"2401.00003": {"u": 3}}}}
        caches = {
            "v": {("a", "2401.00001v1"): 3, ("a", "2401.00003v1"): 2},
            "u": {("a", "2401.00001"): 1},
        }
        early = [("a", "2401.00003")]
        out, missing = probe.score_rows(rows, papers, ledger, caches, early)
        assert [(r["v"], r["u"], r["void_v"], r["void_u"]) for r in out] == [
            (3, 1, False, False),
            (None, None, True, True),
            (2, None, False, True),
        ]
        assert [r["nr52_early"] for r in out] == [False, False, True]
        assert missing == []

        caches["v"].pop(("a", "2401.00001v1"))
        _, missing = probe.score_rows(rows, papers, ledger, caches, early)
        assert missing == ["a/2401.00001v1 (V)"]


# ── The endpoints ─────────────────────────────────────────────────────────────────────────

# Two cases. Scores chosen so every quantity has a hand-computed value, derived in comments.
# 2401.00004 is void in V, so every endpoint drops it from both arms.
CASES = ["a", "b"]
ROWS = [
    _row("a", "2401.00001", gpt=2, nr52=2, v=2, u=1),
    _row("a", "2401.00002", gpt=2, nr52=3, v=3, u=3),
    _row("a", "2401.00003", gpt=1, nr52=0, v=0, u=2, early=True),
    _row("a", "2401.00004", gpt=2, nr52=2, v=None, u=2),
    _row("b", "2401.00005", gpt=2, nr52=2, v=2, u=1),
    _row("b", "2401.00006", gpt=0, nr52=1, v=1, u=0),
]
FIXED = {
    "a": {"sonnet_only": {"rr": 5, "opus5_rest": 1}, "consensus": {"rr": 5, "opus5_rest": 1}},
    "b": {"sonnet_only": {"rr": -2, "opus5_rest": 0}, "consensus": {"rr": -2, "opus5_rest": 0}},
}


class TestE1:
    def test_the_rate_difference_is_over_papers_both_arms_scored(self) -> None:
        # Paired: a has 3 (the V-void paper leaves), b has 2. V actionable 2 + 1, U 2 + 0.
        e1, diff, _ = probe.e1_actionable(ROWS, CASES)
        assert e1["n_paired"] == 5
        assert e1["rate_v"] == 0.6 and e1["rate_u"] == 0.4
        assert diff == pytest.approx(0.2)

    def test_discordant_counts(self) -> None:
        e1, _, _ = probe.e1_actionable(ROWS, CASES)
        assert e1["actionable_v_only"] == 2
        assert e1["actionable_u_only"] == 1

    def test_every_case_is_a_unit_including_one_with_no_arxiv_pick(self) -> None:
        units = probe.e1_units(ROWS, [*CASES, "none"])
        assert units == [(3, 2, 2), (2, 1, 0), (0, 0, 0)]

    def test_its_interval_is_the_registered_bootstrap_over_all_cases(self) -> None:
        cases = [*CASES, "none"]
        e1, _, ci = probe.e1_actionable(ROWS, cases)
        want = probe.case_bootstrap(probe.e1_units(ROWS, cases), probe._pooled_diff)
        assert tuple(ci) == want
        assert e1["ci95"] == [round(want[0], 4), round(want[1], 4)]


class TestE2AndE4:
    def test_sonnet_only_margins(self) -> None:
        # arXiv net@2 per case (label son >= 2, void if missing). NR-52 uses every paper;
        # V and U use only the papers both scored, so 2401.00004 leaves both.
        #   a: nr52 +1+1-2+1 = 1   v +1+1-2 = 0   u -2+1+1 = 0
        #   b: nr52 +1-2 = -1      v +1-2 = -1    u -2-2 = -4
        # delta = rr - opus5_rest - arxiv:
        #   a: nr52 3, v 4, u 4    b: nr52 -1, v -1, u 2
        e2, shift = probe.margin_shift(ROWS, FIXED, CASES, "sonnet_only")
        assert e2["margin_nr52"] == 1.0
        assert e2["margin_v"] == 1.5
        assert e2["margin_u"] == 3.0
        assert shift == -1.5 and e2["shift_v_minus_u"] == -1.5
        assert e2["ci95_shift"][0] <= shift <= e2["ci95_shift"][1]

    def test_a_paper_void_in_one_arm_leaves_both_margins(self) -> None:
        """Kept in U alone, 2401.00004 (+1) would make a's U delta 3 and margin(U) 2.5."""
        kept = [{**r, "v": 2, "void_v": False} if r["pick"] == "2401.00004" else r for r in ROWS]
        e2_kept, _ = probe.margin_shift(kept, FIXED, CASES, "sonnet_only")
        e2, _ = probe.margin_shift(ROWS, FIXED, CASES, "sonnet_only")
        assert e2_kept["margin_u"] == 2.5 and e2["margin_u"] == 3.0
        assert e2_kept["margin_v"] == 1.0 and e2["margin_v"] == 1.5

    def test_the_margin_intervals_are_nr52s_bootstrap(self) -> None:
        """`bigram_report.paired_bootstrap` over the per-case deltas in case order."""
        e2, _ = probe.margin_shift(ROWS, FIXED, CASES, "sonnet_only")

        def nr52(d: list[float]) -> list[float]:
            lo, hi = probe.paired_bootstrap(d)
            return [round(lo, 4), round(hi, 4)]

        assert e2["ci95_nr52"] == nr52([3.0, -1.0])
        assert e2["ci95_v"] == nr52([4.0, -1.0])
        assert e2["ci95_u"] == nr52([4.0, 2.0])

    def test_the_shift_interval_is_the_e1_bootstrap(self) -> None:
        """Per-case V-U deltas (a 0, b -3) through the registered case bootstrap."""
        e2, _ = probe.margin_shift(ROWS, FIXED, CASES, "sonnet_only")
        lo, hi = probe.case_bootstrap([0.0, -3.0], probe._mean)
        assert e2["ci95_shift"] == [round(lo, 4), round(hi, 4)]

    def test_consensus_margins(self) -> None:
        # Label gpt >= 2 and son >= 1. a: nr52 1, v 0, u 0. b: -1 in every arm.
        e4, shift = probe.margin_shift(ROWS, FIXED, CASES, "consensus")
        assert (e4["margin_nr52"], e4["margin_v"], e4["margin_u"]) == (1.0, 1.5, 1.5)
        assert shift == 0.0

    def test_the_arxiv_part_goes_through_nr52s_net2(self) -> None:
        """A missing verdict is void, not -2, exactly as NR-52 scored it."""
        a = [r for r in ROWS if r["case"] == "a"]
        assert probe.arxiv_net2(a, "v", "sonnet_only", "a") == 0
        zeroed = [{**r, "v": 0 if r["v"] is None else r["v"]} for r in a]
        assert probe.arxiv_net2(zeroed, "v", "sonnet_only", "a") == -2

    def test_reproduction_is_flagged_against_the_registered_figures(self) -> None:
        e2, _ = probe.margin_shift(ROWS, FIXED, CASES, "sonnet_only")
        assert e2["reproduces_nr52"] is False


class TestE3:
    def test_flip_rate_on_all_papers_both_arms_scored(self) -> None:
        # U vs NR-52 on the five paired papers: flips at 2 on 0001 (1 vs 2), 0003 (2 vs 0)
        # and 0005 (1 vs 2). Exact on 0002 only. 0004 is void in V and leaves.
        e3 = probe.replicate_flips(ROWS)
        assert e3["all"] == {
            "n": 5,
            "flips_at_2": 3,
            "flip_rate": 0.6,
            "exact_agreement": 1,
            "exact_rate": 0.2,
        }

    def test_and_on_the_papers_nr52_drew_itself(self) -> None:
        # 0003 is an early draw, so the NR-52-drawn subset is 0001, 0002, 0005, 0006.
        e3 = probe.replicate_flips(ROWS)
        assert e3["nr52_drawn"]["n"] == 4
        assert e3["nr52_drawn"]["flips_at_2"] == 2 and e3["nr52_drawn"]["flip_rate"] == 0.5
        assert e3["nr52_early"] == ["a/2401.00003"]


class TestTheBootstrap:
    def test_it_is_seeded_as_registered(self) -> None:
        assert (probe.BOOT_DRAWS, probe.BOOT_SEED) == (10_000, 20260922)
        units = [1.0, -2.0, 0.5, 3.0]
        assert probe.case_bootstrap(units, probe._mean) == probe.case_bootstrap(units, probe._mean)

    def test_the_interval_is_the_250th_and_9750th_sorted_draw(self) -> None:
        """Counting from 0: each draw's statistic here is its own index, so the interval is
        exactly the two indices read."""
        counter = itertools.count()
        assert probe.case_bootstrap([0.0], lambda xs: float(next(counter))) == (250.0, 9750.0)

    def test_it_resamples_every_case_with_replacement(self) -> None:
        seen: list[list[int]] = []
        probe.case_bootstrap(list(range(5)), lambda xs: seen.append(list(xs)) or 0.0, draws=200)
        assert all(len(x) == 5 for x in seen)
        assert any(len(set(x)) < 5 for x in seen)
        assert set().union(*map(set, seen)) == set(range(5))

    def test_constant_units_give_a_point(self) -> None:
        assert probe.case_bootstrap([0.25] * 7, probe._mean) == (0.25, 0.25)

    def test_the_percentile_convention(self) -> None:
        # Two cases: resampled means are 0, 0.5 or 1, so the 250th and 9750th are 0 and 1.
        assert probe.case_bootstrap([0.0, 1.0], probe._mean) == (0.0, 1.0)

    def test_every_endpoint_sees_the_same_case_draws(self) -> None:
        """E1 and the E2/E4 shifts all pass the same case list, so they resample the same cases."""
        seen_a: list[list[int]] = []
        seen_b: list[list[int]] = []
        probe.case_bootstrap(list(range(5)), lambda xs: seen_a.append(list(xs)) or 0.0, draws=50)
        probe.case_bootstrap(
            list("abcde"), lambda xs: seen_b.append(["abcde".index(x) for x in xs]) or 0.0, draws=50
        )
        assert seen_a == seen_b


# Four cases of five papers each, no voids. V is actionable on two more papers than U in w
# and one fewer in x: E1 is +1/20 = +0.05, and its interval includes zero because a draw
# without w is negative about a quarter of the time. net@2 turns each flip into 3 points
# per case, so the E2 shift is (-6 + 3 + 0 + 0) / 4 = -0.75, well past 0.5.
BIG_CASES = ["w", "x", "y", "z"]
BIG_ROWS = (
    [_row("w", f"2401.1000{i}", v=2, u=1) for i in range(2)]
    + [_row("w", f"2401.1000{i}", v=2, u=2) for i in range(2, 5)]
    + [_row("x", "2401.20000", v=1, u=2)]
    + [_row("x", f"2401.2000{i}", v=0, u=0) for i in range(1, 5)]
    + [_row("y", f"2401.3000{i}", v=3, u=3) for i in range(5)]
    + [_row("z", f"2401.4000{i}", v=1, u=1) for i in range(5)]
)
BIG_FIXED = {
    c: {"sonnet_only": {"rr": 0, "opus5_rest": 0}, "consensus": {"rr": 0, "opus5_rest": 0}}
    for c in BIG_CASES
}


class TestTheReading:
    NARROW = (-0.02, 0.03)

    def test_an_e1_interval_including_zero_is_immaterial(self) -> None:
        assert probe.reading(237, 0, 0, self.NARROW) == "Immaterial"

    def test_an_e1_interval_excluding_zero_is_material(self) -> None:
        assert probe.reading(237, 0, 0, (0.001, 0.08)) == "Material"
        assert probe.reading(237, 0, 0, (-0.08, -0.001)) == "Material"

    def test_an_interval_touching_zero_includes_it(self) -> None:
        assert probe.reading(237, 0, 0, (0.0, 0.08)) == "Immaterial"
        assert probe.reading(237, 0, 0, (-0.08, 0.0)) == "Immaterial"

    def test_a_large_e2_shift_alone_is_immaterial(self) -> None:
        """E2 is reported, never read: no bar on it can make the reading Material."""
        s = probe.summarise(BIG_ROWS, BIG_FIXED, BIG_CASES)
        assert s["e1"]["diff"] == 0.05 and s["e1"]["interval_includes_zero"]
        assert s["e2"]["shift_v_minus_u"] == -0.75
        assert s["reading"]["result"] == "Immaterial"
        assert s["prediction"]["e2_in_range"] is False
        assert set(s["reading"]) == {"result", "unreadable", "e1_interval_excludes_zero"}

    def test_exactly_five_percent_void_is_readable(self) -> None:
        assert not probe.unreadable(100, 5, 5)
        assert probe.unreadable(100, 6, 5)
        assert probe.unreadable(100, 5, 6)

    def test_the_five_percent_rule_on_the_real_population(self) -> None:
        # 11/237 = 4.6%, 12/237 = 5.1%.
        assert not probe.unreadable(237, 11, 11)
        assert probe.unreadable(237, 12, 11)

    def test_a_three_point_void_gap_is_readable(self) -> None:
        assert not probe.unreadable(100, 3, 0)
        assert probe.unreadable(100, 4, 0)
        assert probe.unreadable(100, 0, 4)
        # 7/237 = 2.95 points, 8/237 = 3.38 points.
        assert not probe.unreadable(237, 7, 0)
        assert probe.unreadable(237, 8, 0)

    def test_unreadable_comes_first(self) -> None:
        assert probe.reading(100, 6, 6, (0.1, 0.2)) == "Unreadable"


class TestTheSummary:
    def test_it_survives_a_json_round_trip(self) -> None:
        """The artifact test compares a recomputed summary with a stored one. Tuples, int
        keys or NaN would make that comparison fail for reasons unrelated to the data."""
        for rows, fixed, cases in ((ROWS, FIXED, CASES), (BIG_ROWS, BIG_FIXED, BIG_CASES)):
            s = probe.summarise(rows, fixed, cases)
            assert json.loads(json.dumps(s)) == s

    def test_it_carries_the_registered_prediction(self) -> None:
        p = probe.summarise(ROWS, FIXED, CASES)["prediction"]
        assert p["e1_point"] == -0.04
        assert p["e1_range"] == [-0.10, 0.02]
        assert p["reading"] == "Immaterial"
        assert p["e2_range"] == [-0.5, 0.5]
        assert p["e3_range"] == [0.05, 0.12]

    def test_it_reports_voids_by_arm_and_what_they_drop(self) -> None:
        s = probe.summarise(ROWS, FIXED, CASES)
        assert s["voids"] == {
            "v": 1,
            "u": 0,
            "v_rate": 0.1667,
            "u_rate": 0.0,
            "n_scored_in_both": 5,
            "dropped_from_every_endpoint": ["a/2401.00004"],
        }

    def test_its_reading_and_prediction_check(self) -> None:
        s = probe.summarise(ROWS, FIXED, CASES)
        # One void of six in V: 16.7% is above 5%, so the fixture is Unreadable.
        assert s["reading"]["result"] == "Unreadable" and s["reading"]["unreadable"]
        assert s["prediction"]["reading_as_predicted"] is False
        assert s["prediction"]["e1_in_range"] is False  # +0.2 is outside [-0.10, +0.02]
        assert s["prediction"]["e2_in_range"] is False  # -1.5 is outside [-0.5, +0.5]
        assert s["prediction"]["e3_in_range"] is False  # 60% is outside [5%, 12%]
        assert s["prediction"]["e3_nr52_drawn_in_range"] is False  # 50%


class TestTheRegistration:
    def test_the_constants_are_the_registered_ones(self) -> None:
        # Whitespace is normalised so a rewrapped line in the registration is not a failure.
        text = " ".join(probe.PREREG.read_text(encoding="utf-8").split())
        for needle in (
            "no temperature field in the request",
            "11 of the 237 NR-52 verdicts were drawn on 2026-08-06",
            "Every endpoint uses only papers scored in both arms.",
            "10,000 draws, seed 20260922, resampling all 37 cases with replacement",
            "the 250th and 9,750th of the sorted draws, counting from 0",
            "`bigram_report.paired_bootstrap`",
            "carries the E1 bootstrap's interval over the 37 cases",
            "refuses to report unless they give -3.41 and +0.57",
            "on all 237 and on the 226 that NR-52 drew itself",
            "Material: the E1 interval excludes zero.",
            "Immaterial: the E1 interval includes zero.",
            "A shift in E2 alone does not make the reading Material.",
            "more than 5% of either arm is void",
            "differ by more than 3 points",
            "E2: the shift is between -0.5 and +0.5.",
            "retried up to twice",
            "it is dropped from both arms",
            "refuses to make a 601st call to `second_verdict`, retries included",
            "checks for an API key before its first call and makes no call without one",
            "is not charged to the paper's retries",
            "20 consecutive calls fail",
            "count toward the 600 calls, and are refunded to the papers' retries",
            "`--report` uses the recorded list and refuses if it has changed",
            "`db/2607.11271` as v2 and v3",
            "claude-sonnet-5#id-versioned",
            "claude-sonnet-5#id-unversioned",
        ):
            assert needle in text, needle
        assert (probe.VOID_MAX_PCT, probe.VOID_GAP_PCT) == (5, 3)
        assert probe.MAX_ATTEMPTS == 1 + 2
        assert (probe.BOOT_LO_PER_MILLE, probe.BOOT_HI_PER_MILLE) == (25, 975)
        assert not hasattr(probe, "BAR")


# ── The tracked artifact ──────────────────────────────────────────────────────────────────


@pytest.mark.skipif(
    not ARTIFACT.is_file(),
    reason=(
        "evals/sonnet_id_probe.json is written by `sonnet_id_probe.py --report` after "
        "`--judge` has bought the registered verdicts. It does not exist before then. Once it "
        "exists it is tracked and these tests pin it."
    ),
)
class TestTheTrackedArtifact:
    @pytest.fixture(scope="class")
    def artifact(self) -> dict[str, Any]:
        return dict(json.loads(ARTIFACT.read_text(encoding="utf-8")))

    def test_the_summary_is_recomputed_from_the_rows(self, artifact: dict[str, Any]) -> None:
        """The prose will quote the summary. Recomputing it from the rows is what stops the
        two from drifting apart."""
        again = probe.summarise(artifact["rows"], artifact["per_case_fixed"], artifact["cases"])
        assert again == artifact["summary"]

    def test_the_population_is_the_registered_one(self, artifact: dict[str, Any]) -> None:
        rows = artifact["rows"]
        assert len({(r["case"], r["pick"]) for r in rows}) == len(rows)
        assert len(rows) == probe.EXPECTED_POPULATION
        assert len(artifact["cases"]) == probe.EXPECTED_CASES
        assert set(artifact["per_case_fixed"]) == set(artifact["cases"])

    def test_the_arm_ids_follow_the_shared_rule(self, artifact: dict[str, Any]) -> None:
        for r in artifact["rows"]:
            assert r["versioned"] != r["pick"]
            assert dedup_id(r["versioned"]) == r["pick"]
        by = {(r["case"], r["pick"]): r["versioned"] for r in artifact["rows"]}
        assert by.get(("db", "2607.11271")) == "2607.11271v3"

    def test_every_paper_has_a_verdict_or_is_void(self, artifact: dict[str, Any]) -> None:
        for r in artifact["rows"]:
            for arm in probe.ARMS:
                assert (r[arm] is None) == r[f"void_{arm}"], (r["case"], r["pick"], arm)
            assert r["nr52"] is not None

    def test_the_dropped_papers_are_the_voids(self, artifact: dict[str, Any]) -> None:
        voids = artifact["summary"]["voids"]
        want = [f"{r['case']}/{r['pick']}" for r in artifact["rows"] if r["void_v"] or r["void_u"]]
        assert voids["dropped_from_every_endpoint"] == want
        assert artifact["summary"]["e1"]["n_paired"] == len(artifact["rows"]) - len(want)

    def test_the_early_nr52_draws_are_the_registered_eleven(self, artifact: dict[str, Any]) -> None:
        want = sorted(f"{c}/{p}" for c, p in probe.NR52_EARLY)
        assert artifact["nr52_early"] == want
        assert (
            sorted(f"{r['case']}/{r['pick']}" for r in artifact["rows"] if r["nr52_early"]) == want
        )
        assert artifact["summary"]["e3"]["nr52_early"] == want

    def test_nr52_reproduces_before_either_arm_is_read(self, artifact: dict[str, Any]) -> None:
        s = artifact["summary"]
        assert round(s["e2"]["margin_nr52"], 2) == -3.41 and s["e2"]["reproduces_nr52"]
        assert round(s["e4"]["margin_nr52"], 2) == 0.57 and s["e4"]["reproduces_nr52"]

    def test_the_reading_follows_the_registered_rule(self, artifact: dict[str, Any]) -> None:
        s = artifact["summary"]
        r = s["reading"]
        if r["unreadable"]:
            want = "Unreadable"
        elif not s["e1"]["interval_includes_zero"]:
            want = "Material"
        else:
            want = "Immaterial"
        assert r["result"] == want
        assert r["e1_interval_excludes_zero"] is not s["e1"]["interval_includes_zero"]

    def test_the_design_is_the_registered_one(self, artifact: dict[str, Any]) -> None:
        assert artifact["model"] == "claude-sonnet-5"
        assert artifact["namespaces"] == probe.NAMESPACE
        assert artifact["bootstrap"] == {
            "draws": 10_000,
            "seed": 20260922,
            "unit": "case",
            "interval_indices": [250, 9750],
        }
        assert artifact["margin_bootstrap"]["function"] == "bigram_report.paired_bootstrap"
        assert artifact["pre_registration"] == "evals/PREREG-sonnet-id-probe.md"
        assert artifact["calls"]["made"] <= probe.CALL_CAP
