"""Pin the third judge: its prompt, its populations, its operating rules and its readings.

`evals/third_judge.py` executes `evals/PREREG-third-judge.md`. Everything here is offline and
runs on small synthetic fixtures, except the last class. That one pins the tracked artifact
`evals/third_judge.json`, recomputing its summary from its rows, and it is skipped while that
file is absent, and only then, because the verdicts it is built from have not been bought yet.
"""

from __future__ import annotations

import http.client
import json
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evals"))

import gemini_client as gc  # noqa: E402
import second_judge as sj  # noqa: E402
import third_judge as tj  # noqa: E402

ARTIFACT = ROOT / "evals" / "third_judge.json"

CTX = "## Profile\nA repository that indexes vectors.\n## README (excerpt)\nFast search."
PAPER = {"title": "A Fast Index", "abstract": "We index vectors quickly.", "arxiv_id": "2404.1v2"}


def text(body: str, usage: dict[str, int] | None = None) -> gc.Outcome:
    return gc.Outcome(
        gc.TEXT,
        200,
        text=body,
        finish_reason="STOP",
        model_version="gemini-3.8-flash-001",
        usage=usage or {"promptTokenCount": 2000, "candidatesTokenCount": 60},
    )


def finish(reason: str) -> gc.Outcome:
    return gc.Outcome(gc.FINISH, 200, text='{"score": 2, "justif', finish_reason=reason)


def transport() -> gc.Outcome:
    return gc.Outcome(gc.TRANSPORT, 503, error="unavailable")


def scripted(*outcomes: gc.Outcome) -> tuple[Any, list[dict[str, Any]]]:
    """A fake `send` that answers with *outcomes* in order, then repeats the last one."""
    sent: list[dict[str, Any]] = []
    queue = list(outcomes)

    def send(body: dict[str, Any]) -> gc.Outcome:
        sent.append(body)
        return queue.pop(0) if len(queue) > 1 else queue[0]

    return send, sent


def buyer(tmp_path: Path, send: Any, n: int = 1, **kw: Any) -> tuple[tj.Buyer, list[str]]:
    prompts = {tj.sha256(f"prompt {i}"): f"prompt {i}" for i in range(n)}
    members = {s: [{"population": "band", "case": "c", "id": s[:6]}] for s in prompts}
    b = tj.Buyer(
        prompts, members, send, state=tmp_path, sleep=lambda s: None, log=lambda m: None, **kw
    )
    return b, list(prompts)


# ── The prompt and the request ────────────────────────────────────────────────────────────


class TestThePrompt:
    @pytest.mark.parametrize(
        "paper",
        [
            PAPER,
            {**PAPER, "arxiv_id": "doi:10.1093/bioinformatics/btad074"},
            {**PAPER, "arxiv_id": "2404.14989", "abstract": "x" * 2500},
        ],
    )
    def test_identical_to_what_second_verdict_sends(self, paper: dict[str, Any]) -> None:
        assert tj.build_prompt(CTX, paper) == tj.sonnet_path_prompt("case", CTX, paper)

    def test_the_capture_restores_the_sonnet_path(self) -> None:
        before = sj.CACHE, sj.complete
        tj.sonnet_path_prompt("case", CTX, PAPER)
        assert (sj.CACHE, sj.complete) == before

    def test_the_request_is_the_registered_one(self) -> None:
        prompt = tj.build_prompt(CTX, PAPER)
        body = tj.request_body(prompt)
        assert body["contents"] == [{"role": "user", "parts": [{"text": prompt}]}]
        assert body["generationConfig"] == {
            "maxOutputTokens": 16384,
            "thinkingConfig": {"thinkingLevel": "MEDIUM"},
        }
        assert body["safetySettings"] == [
            {"category": f"HARM_CATEGORY_{c}", "threshold": "BLOCK_NONE"}
            for c in ("HARASSMENT", "HATE_SPEECH", "SEXUALLY_EXPLICIT", "DANGEROUS_CONTENT")
        ]
        flat = json.dumps(body["generationConfig"]).lower()
        for knob in ("temperature", "topp", "topk", "seed"):
            assert knob not in flat

    def test_a_paper_without_text_has_no_prompt(self) -> None:
        assert tj.Item("band", "c", "p", 2, 1, CTX, None).prompt is None
        assert tj.Item("band", "c", "p", 2, 1, CTX, {**PAPER, "abstract": " "}).prompt is None
        drifted = tj.Item("band", "c", "p", 2, 1, None, PAPER)
        assert tj.why_no_prompt(drifted) == "context drifted"
        assert tj.why_no_prompt(tj.Item("band", "c", "p", 2, 1, CTX, None)) == "unresolved"
        item = tj.Item("band", "c", "p", 2, 1, CTX, PAPER)
        assert item.sha == tj.sha256(tj.build_prompt(CTX, PAPER))


class TestTheBaselineIds:
    """The baseline's arXiv picks carry the versioned id GPT-5.5 saw (NR-67)."""

    DOI = "doi:10.1/x"

    def build(self, tmp_path: Path, doi_status: str = "ok") -> tuple[list[Any], list[Any]]:
        gold = tmp_path / "gold"
        (gold / "db").mkdir(parents=True)
        (gold / "db" / "2404.14989v2.json").write_text("{}", encoding="utf-8")
        ship = {"db": [{"arxiv_id": "2301.00001v1", "title": "Ours", "judge_score": 2}]}
        opus = {
            "db": [
                {"arxiv_id": "2404.14989", "judge_score": 2},
                {"arxiv_id": self.DOI, "judge_score": 3},
            ]
        }
        sonnet = {("db", "2404.14989"): 1, ("db", sj.safe_paper_id(self.DOI)): 2}
        sonnet[("db", "2301.00001v1")] = 3
        arxiv = {
            "db": {
                "2404.14989": {
                    "status": "ok",
                    "resolved_id": "2404.14989v2",
                    "title": "Stored",
                    "abstract": "Stored abstract.",
                }
            }
        }
        doi = {"db": {self.DOI: {"status": doi_status, "title": "D", "abstract": "Doi text."}}}
        return tj.comparison_items(
            ship,
            opus,
            ["db"],
            {"db": CTX},
            sonnet,
            arxiv,
            doi,
            meta=lambda case: {"2301.00001v1": {"abstract": "Pool abstract."}},
            gold=gold,
            multi={},
        )

    def test_arxiv_picks_are_asked_under_the_versioned_id(self, tmp_path: Path) -> None:
        ours, base = self.build(tmp_path)
        arxiv = next(i for i in base if i.kind == "arxiv")
        assert arxiv.id == "2404.14989" and arxiv.sonnet == 1 and arxiv.gpt == 2
        assert "\narXiv: 2404.14989v2\n" in arxiv.prompt
        assert "\narXiv: 2404.14989\n" not in arxiv.prompt
        assert "Abstract: Stored abstract." in arxiv.prompt

    def test_doi_picks_use_the_stored_resolution(self, tmp_path: Path) -> None:
        _, base = self.build(tmp_path)
        doi = next(i for i in base if i.kind == "doi")
        assert "\nDOI: 10.1/x\n" in doi.prompt and "Abstract: Doi text." in doi.prompt
        _, base = self.build(tmp_path / "again", doi_status="unresolved")
        assert next(i for i in base if i.kind == "doi").prompt is None

    def test_ours_uses_the_pool_abstract(self, tmp_path: Path) -> None:
        ours, _ = self.build(tmp_path)
        assert ours[0].prompt is not None and "Abstract: Pool abstract." in ours[0].prompt
        assert ours[0].sonnet == 3

    def test_the_sonnet_path_differs_only_in_the_identifier_line(self, tmp_path: Path) -> None:
        _, base = self.build(tmp_path)
        arxiv = next(i for i in base if i.kind == "arxiv")
        theirs = tj.build_prompt(CTX, tj.sonnet_paper(arxiv)).split("\n")
        mine = arxiv.prompt.split("\n")
        diff = [(a, b) for a, b in zip(theirs, mine, strict=True) if a != b]
        assert diff == [("arXiv: 2404.14989", "arXiv: 2404.14989v2")]
        assert tj.sonnet_paper(next(i for i in base if i.kind == "doi")) is None


# ── Parsing and classification ────────────────────────────────────────────────────────────


class TestTheParsingRule:
    @pytest.mark.parametrize(
        ("body", "want"),
        [
            ('{"score": 2, "justification": "j", "proposed_change": "c"}', 2),
            ('Here: {"score": 0} done', 0),
            ('```json\n{"score": 3, "justification": "uses {braces}"}\n```', 3),
            ('{"score": true}', None),
            ('{"score": false}', None),
            ('{"score": 2.0}', None),
            ('{"score": 1.5}', None),
            ('{"score": "2"}', None),
            ('{"score": 4}', None),
            ('{"score": -1}', None),
            ('{"score": null}', None),
            ('{"justification": "no score"}', None),
            ("[2]", None),
            ("no json at all", None),
            ('{"score": 2', None),
            ('} {"score": 2', None),
        ],
    )
    def test_score_must_be_an_int_from_0_to_3(self, body: str, want: int | None) -> None:
        assert tj.parse_score(body) == want

    @pytest.mark.parametrize(
        ("outcome", "want"),
        [
            (text('{"score": 1}'), ("scored", 1)),
            (text('{"score": 1.0}'), ("parse", None)),
            (finish("MAX_TOKENS"), ("truncated", None)),
            (finish("SAFETY"), ("blocked", None)),
            (gc.Outcome(gc.BLOCKED, 200, block_reason="SAFETY"), ("blocked", None)),
            (gc.Outcome(gc.RATE_LIMIT, 429, retry_delay=3.0), ("rate_limit", None)),
            (gc.Outcome(gc.MALFORMED, 200), ("parse", None)),
            (transport(), ("transport", None)),
            (gc.Outcome(gc.KEY, 403), ("key", None)),
            (gc.Outcome(gc.FATAL, 400), ("fatal", None)),
        ],
    )
    def test_classes(self, outcome: gc.Outcome, want: tuple[str, int | None]) -> None:
        assert tj.classify(outcome) == want

    def test_a_cut_off_body_is_a_transport_outcome(self) -> None:
        def cut(req: Any, timeout: float) -> tuple[int, bytes]:
            raise http.client.IncompleteRead(b"partial")

        o = gc.generate({"contents": []}, tj.MODEL, "secret-key", transport=cut)
        assert o.kind == gc.TRANSPORT and "secret-key" not in o.error
        status, body = gc.count_tokens({"contents": []}, tj.MODEL, "secret-key", transport=cut)
        assert status is None and "IncompleteRead" in body


# ── Buying ────────────────────────────────────────────────────────────────────────────────


class TestVoids:
    def test_a_verdict_is_cached_with_its_record(self, tmp_path: Path) -> None:
        send, sent = scripted(text('{"score": 2}'))
        b, (sha,) = buyer(tmp_path, send)
        assert b.run([sha], workers=1) is None
        rec = tj.read_cache(sha, tmp_path)
        assert rec["score"] == 2 and rec["void"] is None and rec["prompt_sha256"] == sha
        assert rec["populations"] == [{"population": "band", "case": "c", "id": sha[:6]}]
        assert rec["finish_reason"] == "STOP" and rec["model_version"].startswith("gemini")
        assert rec["usage"]["promptTokenCount"] == 2000 and rec["text"] == '{"score": 2}'
        assert len(sent) == 1 and b.run([sha], workers=1) is None and len(sent) == 1

    @pytest.mark.parametrize(
        ("bad", "cause"),
        [
            (text("not json"), "parse"),
            (finish("MAX_TOKENS"), "truncated"),
            (finish("SAFETY"), "blocked"),
            (gc.Outcome(gc.MALFORMED, 200), "parse"),  # a 200 the server meant to send
        ],
    )
    def test_a_content_failure_is_retried_once_then_void(
        self, tmp_path: Path, bad: gc.Outcome, cause: str
    ) -> None:
        send, sent = scripted(bad)
        b, (sha,) = buyer(tmp_path, send)
        b.run([sha], workers=1)
        rec = tj.read_cache(sha, tmp_path)
        assert len(sent) == 2 and rec["score"] is None and rec["void"] == cause

    def test_one_content_retry_can_succeed(self, tmp_path: Path) -> None:
        send, sent = scripted(finish("MAX_TOKENS"), text('{"score": 3}'))
        b, (sha,) = buyer(tmp_path, send)
        b.run([sha], workers=1)
        assert len(sent) == 2 and tj.read_cache(sha, tmp_path)["score"] == 3

    def test_transport_is_retried_three_times_then_void(self, tmp_path: Path) -> None:
        send, sent = scripted(transport())
        b, (sha,) = buyer(tmp_path, send)
        b.run([sha], workers=1)
        assert len(sent) == 4 and tj.read_cache(sha, tmp_path)["void"] == "transport"
        send, sent = scripted(transport(), transport(), transport(), text('{"score": 0}'))
        b, (sha,) = buyer(tmp_path / "b", send)
        b.run([sha], workers=1)
        assert len(sent) == 4 and tj.read_cache(sha, tmp_path / "b")["score"] == 0

    def test_a_429_waits_and_is_not_charged(self, tmp_path: Path) -> None:
        limited = gc.Outcome(gc.RATE_LIMIT, 429, retry_delay=7.0)
        send, sent = scripted(*[limited] * 6, text('{"score": 1}'))
        waits: list[float] = []
        b, (sha,) = buyer(tmp_path, send)
        b.sleep = waits.append
        b.run([sha], workers=1)
        assert tj.read_cache(sha, tmp_path)["score"] == 1
        assert waits == [7.0] * 6 and sha not in b.ledger["failures"]
        # Six 429s bought nothing, so only the call that returned a verdict spends the cap.
        assert b.ledger["calls"] == 1 and b.ledger["rate_limited"] == 6
        assert b.ledger["rate_streak"] == 0  # the verdict broke the streak

    def test_a_429_does_not_spend_the_call_cap(self, tmp_path: Path) -> None:
        outcomes = [gc.Outcome(gc.RATE_LIMIT, 429, retry_delay=0.0)] * 5 + [text('{"score": 2}')]
        send, sent = scripted(*(outcomes * 3))
        b, shas = buyer(tmp_path, send, n=3)
        b.ledger["calls"] = tj.CALL_CAP - 3
        assert b.run(shas, workers=1) is None
        assert len(sent) == 18 and b.ledger["calls"] == tj.CALL_CAP
        assert b.ledger["rate_limited"] == 15

    def test_twenty_rate_limits_in_a_row_stop_the_run(self, tmp_path: Path) -> None:
        send, sent = scripted(gc.Outcome(gc.RATE_LIMIT, 429, retry_delay=0.0))
        b, shas = buyer(tmp_path, send, n=3)
        why = b.run(shas, workers=1)
        assert why is not None and why.startswith("rate limit")
        assert "tier" in why and "quota" in why
        assert len(sent) == tj.RATE_LIMIT_STOP and b.ledger["calls"] == 0
        assert b.ledger["rate_limited"] == tj.RATE_LIMIT_STOP
        assert b.ledger["failures"] == {} and not list((tmp_path / tj.MODEL).glob("*.json"))
        assert tj.load_ledger(tmp_path)["stops"][0]["why"] == why

    def test_a_verdict_breaks_the_rate_limit_streak(self, tmp_path: Path) -> None:
        limited = gc.Outcome(gc.RATE_LIMIT, 429, retry_delay=0.0)
        send, sent = scripted(*([limited] * 15 + [text('{"score": 1}')]) * 4)
        b, shas = buyer(tmp_path, send, n=4)
        assert b.run(shas, workers=1) is None and len(sent) == 64

    @pytest.mark.parametrize("kind", [gc.KEY, gc.FATAL])
    def test_a_refusal_stops_the_run_uncharged(self, tmp_path: Path, kind: str) -> None:
        send, sent = scripted(gc.Outcome(kind, 403, error="denied"))
        b, shas = buyer(tmp_path, send, n=3)
        why = b.run(shas, workers=1)
        assert why is not None and why.startswith(kind) and len(sent) == 1
        assert b.ledger["failures"] == {} and not list((tmp_path / tj.MODEL).glob("*.json"))
        assert tj.load_ledger(tmp_path)["stops"][0]["why"] == why

    def test_a_resumed_prompt_keeps_its_retry_budget(self, tmp_path: Path) -> None:
        send, sent = scripted(text("garbled"))
        b, (sha,) = buyer(tmp_path, send)
        b.ledger["failures"][sha] = {"transport": 0, "content": 1, "last": "parse"}
        b.run([sha], workers=1)
        assert len(sent) == 1 and tj.read_cache(sha, tmp_path)["void"] == "parse"

    def test_a_spent_budget_without_a_record_becomes_void_without_a_call(
        self, tmp_path: Path
    ) -> None:
        send, sent = scripted(text('{"score": 2}'))
        b, (sha,) = buyer(tmp_path, send)
        b.ledger["failures"][sha] = {"transport": 4, "content": 0, "last": "transport"}
        b.run([sha], workers=1)
        assert sent == [] and tj.read_cache(sha, tmp_path)["void"] == "transport"


class TestCaps:
    def test_spend_is_computed_from_usage_at_the_registered_prices(self) -> None:
        usage = {
            "promptTokenCount": 1_000_000,
            "candidatesTokenCount": 100_000,
            "thoughtsTokenCount": 900_000,
        }
        assert tj.cost(usage) == pytest.approx(0.75 + 3.75)
        assert tj.cost({}) == 0.0

    def test_it_stops_before_a_call_that_could_pass_the_cap(self, tmp_path: Path) -> None:
        send, sent = scripted(text('{"score": 2}'))
        b, shas = buyer(tmp_path, send, spend_cap=40.0)
        b.ledger["spend_usd"] = 40.0 - tj.worst_cost("prompt 0") + 1e-9
        why = b.run(shas, workers=1)
        assert sent == [] and why is not None and why.startswith("spend cap")

    def test_a_call_that_fits_under_the_cap_is_made(self, tmp_path: Path) -> None:
        usage = {"promptTokenCount": 2000, "candidatesTokenCount": 40, "thoughtsTokenCount": 900}
        send, sent = scripted(text('{"score": 2}', usage))
        b, shas = buyer(tmp_path, send, spend_cap=40.0)
        b.ledger["spend_usd"] = 40.0 - tj.worst_cost("prompt 0") - 1e-9
        assert b.run(shas, workers=1) is None and len(sent) == 1
        assert b.ledger["spend_usd"] == pytest.approx(
            40.0 - tj.worst_cost("prompt 0") - 1e-9 + tj.cost(usage)
        )
        assert b.reserved == pytest.approx(0.0)

    def test_the_call_after_the_cap_is_refused(self, tmp_path: Path) -> None:
        send, sent = scripted(text('{"score": 2}'))
        b, shas = buyer(tmp_path, send, n=3)
        b.ledger["calls"] = tj.CALL_CAP - 2
        why = b.run(shas, workers=1)
        assert len(sent) == 2 and why == f"call cap: call {tj.CALL_CAP + 1} refused"
        assert b.ledger["calls"] == tj.CALL_CAP

    def test_twenty_consecutive_failures_are_an_outage_and_refunded(self, tmp_path: Path) -> None:
        send, sent = scripted(transport())
        b, shas = buyer(tmp_path, send, n=8)
        why = b.run(shas, workers=1)
        assert why is not None and why.startswith("outage") and len(sent) == 20
        assert not list((tmp_path / tj.MODEL).glob("*.json"))  # the four voids are undone
        assert all(f["transport"] == 0 for f in b.ledger["failures"].values())
        assert b.ledger["streak"] == [] and b.ledger["calls"] == 20

    def test_a_verdict_breaks_the_streak(self, tmp_path: Path) -> None:
        outcomes = [transport()] * 3 + [text('{"score": 1}')]
        send, sent = scripted(*(outcomes * 8))
        b, shas = buyer(tmp_path, send, n=8)
        assert b.run(shas, workers=1) is None and len(sent) == 32


# ── Readings ──────────────────────────────────────────────────────────────────────────────


class TestTheReadings:
    G, S = 0.873, 0.494

    @pytest.mark.parametrize(
        ("ci", "want"),
        [
            ((0.494, 0.873), "overlaps both"),
            ((0.40, 0.95), "overlaps both"),  # first match wins over a single overlap
            ((0.50, 0.873), "overlaps GPT-5.5"),
            ((0.873, 0.95), "overlaps GPT-5.5"),
            ((0.30, 0.494), "overlaps Sonnet"),
            ((0.494, 0.60), "overlaps Sonnet"),
            ((0.4941, 0.8729), "between"),
            ((0.60, 0.70), "between"),
            ((0.8731, 0.95), "above both"),
            ((0.10, 0.4939), "below both"),
        ],
    )
    def test_level(self, ci: tuple[float, float], want: str) -> None:
        assert tj.level_reading(ci, self.G, self.S) == want

    def test_level_does_not_depend_on_which_judge_is_higher(self) -> None:
        assert tj.level_reading((0.05, 0.1), 0.068, 0.255) == "overlaps GPT-5.5"
        assert tj.level_reading((0.1, 0.2), 0.068, 0.255) == "between"
        assert tj.level_reading((0.1, 0.2), 0.255, 0.068) == "between"
        assert tj.level_reading((0.26, 0.4), 0.255, 0.068) == "above both"

    @pytest.mark.parametrize(
        ("ci", "want"),
        [
            ((0.5, 0.8), "does not order the band"),
            ((0.5001, 0.8), "orders the band"),
            ((0.3, 0.6), "does not order the band"),
            (None, "no interval"),  # an absent interval is not a failure to order
        ],
    )
    def test_order(self, ci: Any, want: str) -> None:
        assert tj.order_reading(ci) == want

    @pytest.mark.parametrize(
        ("margin", "want"),
        [
            (-3.41, "between"),
            (0.32, "between"),
            (-1.5, "between"),
            (0.3201, "above both"),
            (-3.4101, "below both"),
        ],
    )
    def test_margin(self, margin: float, want: str) -> None:
        assert tj.margin_reading(margin) == want

    @pytest.mark.parametrize(
        ("void", "n", "want"),
        [(16, 324, False), (17, 324, True), (5, 100, False), (6, 100, True), (0, 0, False)],
    )
    def test_unreadable_above_five_percent(self, void: int, n: int, want: bool) -> None:
        assert tj.unreadable(void, n) is want

    @staticmethod
    def arms(ours: tuple[int, int], baseline: tuple[int, int]) -> dict[str, dict[str, Any]]:
        def side(void: int, n: int) -> dict[str, Any]:
            ms = [{"t": None, "void": "truncated"}] * void + [{"t": 2, "void": None}] * (n - void)
            return tj.void_stats(ms)

        return {"ours": side(*ours), "baseline": side(*baseline)}

    @pytest.mark.parametrize(
        ("ours", "baseline", "want"),
        [
            ((0, 300), (0, 360), False),
            ((6, 300), (7, 360), False),  # 2.0% against 1.9%, both under 5% and close
            ((16, 300), (0, 360), True),  # 5.3% on one arm alone
            ((0, 300), (19, 360), True),  # 5.3% on the other arm alone
            ((12, 300), (0, 360), True),  # 4.0% against 0.0%: under 5%, but 4 points apart
            ((0, 300), (11, 360), True),  # 0.0% against 3.06%, the other way round
            ((9, 300), (0, 360), False),  # 3.0% against 0.0% is exactly the limit, not past it
            ((0, 0), (0, 0), False),
        ],
    )
    def test_the_comparison_void_rule_is_per_arm(
        self, ours: tuple[int, int], baseline: tuple[int, int], want: bool
    ) -> None:
        assert tj.arms_unreadable(self.arms(ours, baseline)) is want


# ── The summary from rows ─────────────────────────────────────────────────────────────────


def synthetic(band_voids: int = 0, baseline_voids: int = 0) -> dict[str, Any]:
    """Rows for every population, with one prompt shared by the band and our arm."""
    rng = random.Random(7)
    rows: list[dict[str, Any]] = []

    def row(score: int | None, void: str | None, *members: dict[str, Any]) -> None:
        rows.append(
            {"sha256": f"s{len(rows)}", "score": score, "void": void, "members": list(members)}
        )

    for i in range(100):
        m = {"population": "band", "case": f"c{i % 10}", "id": f"b{i}", "gpt": 3, "sonnet": 2}
        m["finescale"] = rng.random() + (i % 3) * 0.3
        if i < band_voids:
            row(None, "truncated", m)
        elif i == 99:
            ours = {"population": "ours", "case": "c0", "id": "b99", "gpt": 2, "sonnet": 1}
            row(2, None, m, ours)
        else:
            row(2 if i % 3 else 1, None, m)
    for i in range(40):
        case = f"c{i % 4}"
        row(
            2 if i % 4 else 0,
            None,
            {"population": "ours", "case": case, "id": f"o{i}", "gpt": 2, "sonnet": 1},
        )
        row(
            None if i < baseline_voids else (3 if i % 2 else 1),
            "truncated" if i < baseline_voids else None,
            {"population": "baseline", "case": case, "id": f"x{i}", "gpt": 2, "sonnet": 2},
        )
    row(
        None,
        "no prompt",
        {"population": "baseline", "case": "c1", "id": "doi:1", "gpt": 2, "sonnet": 2},
    )
    for i in range(30):
        row(
            2 if i % 5 else 1,
            None,
            {"population": "adopted", "case": f"r{i % 6}", "id": f"a{i}", "gpt": 2, "sonnet": 2},
        )
        row(
            0 if i % 4 else 2,
            None,
            {"population": "crossrepo", "case": f"r{i % 6}", "id": f"k{i}", "gpt": 1, "sonnet": 0},
        )
    return {"rows": rows, "comparison_cases": ["c0", "c1", "c2", "c3"]}


class TestTheSummary:
    def test_a_shared_prompt_counts_in_each_population(self) -> None:
        s = tj.summarise(synthetic())
        assert s["voids"]["band"]["n"] == 100 and s["voids"]["comparison"]["n"] == 82
        assert s["voids"]["comparison"]["by_cause"] == {"no prompt": 1}
        assert s["E1"]["n_scored"] == 100

    def test_e4_is_nr52s_net2_with_voids_excluded(self) -> None:
        data = synthetic()
        s = tj.summarise(data)
        per_case = {}
        for c in data["comparison_cases"]:
            nets, counts = [], []
            for arm in ("ours", "baseline"):
                ms = [m for m in tj.members_of(data["rows"], arm) if m["case"] == c]
                scored = [m for m in ms if m["t"] is not None]
                nets.append(sum(1 if m["t"] >= 2 else -2 for m in scored))
                counts.append(len(scored))
            per_case[c] = {
                "ours": nets[0],
                "baseline": nets[1],
                "delta": nets[0] - nets[1],
                "scored": {"ours": counts[0], "baseline": counts[1]},
            }
        assert s["E4"]["per_case"] == per_case
        deltas = [v["delta"] for v in per_case.values()]
        assert s["E4"]["margin"] == pytest.approx(sum(deltas) / 4)
        assert s["E4"]["reading"] == tj.margin_reading(s["E4"]["margin"])

    def test_e4_keeps_both_arms_scored_counts(self) -> None:
        """A delta alone cannot say whether one arm lost papers the other kept."""
        s = tj.summarise(synthetic(baseline_voids=1))
        c0 = s["E4"]["per_case"]["c0"]
        assert c0["scored"] == {"ours": 11, "baseline": 9}
        assert c0["delta"] == c0["ours"] - c0["baseline"]

    def test_e4_has_no_interval_without_deltas(self) -> None:
        s = tj.summarise({**synthetic(), "comparison_cases": []})
        assert s["E4"]["margin"] is None and s["E4"]["ci"] is None
        assert s["E4"]["per_case"] == {} and s["E4"]["reading"] == "no cases"

    def test_the_comparison_is_unreadable_when_one_arm_voids(self) -> None:
        readable = tj.summarise(synthetic())["voids"]["comparison"]
        assert readable["by_arm"]["ours"]["rate"] == 0.0
        assert not readable["unreadable"]  # 0.0 against 1/41, under both limits
        gap = tj.summarise(synthetic(baseline_voids=1))
        arms = gap["voids"]["comparison"]["by_arm"]
        assert not arms["ours"]["unreadable"] and not arms["baseline"]["unreadable"]
        assert gap["voids"]["comparison"]["unreadable"]  # 0.0 against 4.9%: the 3-point rule
        assert gap["E4"]["reading"] == "Unreadable"
        over = tj.summarise(synthetic(baseline_voids=2))
        assert over["voids"]["comparison"]["by_arm"]["baseline"]["unreadable"]  # 3/41 is 7.3%
        assert over["E4"]["reading"] == "Unreadable"

    def test_the_void_bias_is_descriptive_only(self) -> None:
        s = tj.summarise(synthetic(baseline_voids=1))
        bias = s["void_bias"]["baseline"]
        assert bias["void"]["n"] == 2 and bias["scored"]["n"] == 39
        assert bias["void"]["gpt"] == 1.0 and bias["void"]["sonnet"] == 1.0
        assert bias["scored"]["gpt"] == 1.0
        assert s["void_bias"]["crossrepo"]["void"] == {"n": 0, "gpt": None, "sonnet": None}
        assert set(s["prediction"]) == {
            *tj.PREDICTION,
            "voids under 2% in every population",
            "readings",
        }

    def test_the_bootstrap_resamples_repositories(self) -> None:
        rows = tj.members_of(synthetic()["rows"], "band")
        by: dict[str, list[dict[str, Any]]] = {}
        for r in rows:
            by.setdefault(r["case"], []).append(r)
        repos = sorted(by)
        rng = random.Random(20260923)
        draws = sorted(
            tj.rate([r for _ in repos for r in by[repos[rng.randrange(len(repos))]]])
            for _ in range(10_000)
        )
        assert tj.repo_bootstrap(rows, tj.rate) == [draws[250], draws[9750]]

    def test_more_than_five_percent_void_is_unreadable(self) -> None:
        s = tj.summarise(synthetic(band_voids=6))
        assert s["voids"]["band"]["unreadable"] and s["voids"]["band"]["by_cause"] == {
            "truncated": 6
        }
        assert s["E1"]["reading"] == "Unreadable" and s["E2"]["reading"] == "Unreadable"
        assert s["E1"]["n_scored"] == 94
        s = tj.summarise(synthetic(band_voids=5))
        assert not s["voids"]["band"]["unreadable"] and s["E1"]["reading"] != "Unreadable"

    def test_the_prediction_is_checked_against_the_registered_ranges(self) -> None:
        s = tj.summarise(synthetic())
        p = s["prediction"]
        assert p["E1 band share"]["range"] == [0.55, 0.85]
        assert p["E3 adopted-against-control AUC"]["range"] == [0.75, None]
        want = s["E1"]["rate"] is not None and 0.55 <= s["E1"]["rate"] <= 0.85
        assert p["E1 band share"]["within"] is want
        assert p["voids under 2% in every population"]["value"] == pytest.approx(1 / 82)
        assert p["voids under 2% in every population"]["within"] is True
        assert p["readings"]["E4"]["mode"] == "between"

    def test_the_readings_carry_the_registered_probability_and_point(self) -> None:
        r = tj.summarise(synthetic())["prediction"]["readings"]
        assert r["E1"]["point"] == 0.70 and r["E2"]["point"] == 0.70
        assert r["E3 adopted"]["point"] == 0.75 and r["E3 crossrepo"]["point"] == 0.15
        assert r["E4"]["point"] == -1.5
        for k, v in r.items():
            named = tj.PREDICTED_READING[k]["p"]
            assert v["probability"] == named.get(v["actual"], named["other"])
            assert v["as_predicted"] is (v["actual"] == v["mode"])
        # The registration names three readings for E1 and one for each of the rest.
        assert tj.PREDICTED_READING["E1"]["p"]["between"] == 0.5
        assert tj.PREDICTED_READING["E1"]["p"]["overlaps GPT-5.5"] == 0.25
        assert tj.PREDICTED_READING["E2"]["p"]["orders the band"] == 0.9
        assert tj.PREDICTED_READING["E3 adopted"]["p"]["between"] == 0.45
        assert tj.PREDICTED_READING["E4"]["p"]["between"] == 0.6

    def test_an_unreadable_endpoint_is_not_scored_against_the_prediction(self) -> None:
        """A void rate may not decide a hit: an Unreadable endpoint scores null, not a miss."""
        p = tj.summarise(synthetic(band_voids=6))["prediction"]
        for k in ("E1 band share", "E2 band AUC"):
            assert p[k]["value"] is None and p[k]["within"] is None
            assert p[k]["unreadable"] is True and p[k]["range"] == list(tj.PREDICTION[k])
        assert p["E3 adopted rate"]["unreadable"] is False
        assert p["E3 adopted rate"]["within"] in (True, False)
        assert p["E3 adopted-against-control AUC"]["unreadable"] is False

    def test_an_unreadable_adoption_population_unreads_the_auc(self) -> None:
        data = synthetic()
        for r in data["rows"]:
            if r["members"][0]["id"] in {"a0", "a1", "a2"}:  # 3 of the 30 adopted papers
                r["score"], r["void"] = None, "truncated"
        s = tj.summarise(data)
        assert s["voids"]["adopted"]["unreadable"]  # 3 of 30 is 10%
        assert s["E3"]["auc"]["unreadable"] and s["E3"]["auc"]["auc"] is not None
        assert s["prediction"]["E3 adopted-against-control AUC"] == {
            "value": None,
            "range": [0.75, None],
            "within": None,
            "unreadable": True,
        }


# ── What has to be frozen, and what --judge and --report refuse ───────────────────────────


def git(repo: Path, *args: str) -> str:
    out = subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, check=True, text=True
    )
    return out.stdout.strip()


class TestTheFrozenFiles:
    """The registration is not the whole specification: the script and its client decide too."""

    @pytest.fixture
    def repo(self, tmp_path: Path) -> Path:
        subprocess.run(["git", "init", "-q", str(tmp_path)], check=True, capture_output=True)
        git(tmp_path, "config", "user.email", "t@t")
        git(tmp_path, "config", "user.name", "t")
        for rel in tj.FROZEN:
            path = tmp_path / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"first {rel}\n", encoding="utf-8")
        return tmp_path

    def test_every_frozen_file_must_be_committed(self, repo: Path) -> None:
        assert tj.FROZEN == (
            "evals/PREREG-third-judge.md",
            "evals/third_judge.py",
            "evals/gemini_client.py",
        )
        assert tj.prereg_refusal(repo) == [f"{rel} is not committed" for rel in tj.FROZEN]
        git(repo, "add", "-A")
        git(repo, "commit", "-qm", "freeze")
        assert tj.prereg_refusal(repo) == []

    @pytest.mark.parametrize("rel", tj.FROZEN)
    def test_an_edit_to_any_of_them_refuses(self, repo: Path, rel: str) -> None:
        git(repo, "add", "-A")
        git(repo, "commit", "-qm", "freeze")
        (repo / rel).write_text("second\n", encoding="utf-8")
        assert tj.prereg_refusal(repo) == [f"{rel} differs from HEAD"]

    def test_the_shas_recorded_beside_a_reading(self, repo: Path) -> None:
        git(repo, "add", "-A")
        git(repo, "commit", "-qm", "freeze")
        shas = tj.frozen_shas(repo)
        assert shas["head"] == git(repo, "rev-parse", "HEAD")
        assert shas["files"] == {r: git(repo, "rev-parse", f"HEAD:{r}") for r in tj.FROZEN}
        assert len(set(shas["files"].values())) == len(tj.FROZEN)


TINY = {"title": "T", "abstract": "An abstract.", "arxiv_id": "x"}


def no_count(request: dict[str, Any], model: str, key: str) -> tuple[int, Any]:
    return 200, {"totalTokens": 11}


def loaded(
    *,
    drifted: list[str] | None = None,
    t0_drift: list[str] | None = None,
    missing: dict[str, int] | None = None,
    short: dict[str, int] | None = None,
    cases: int = tj.N_CASES,
) -> tj.Loaded:
    """A `Loaded` at the registered counts, every paper carrying a prompt unless asked otherwise."""
    miss, short = missing or {}, short or {}
    items: dict[str, list[tj.Item]] = {}
    for p in tj.POPULATIONS:
        gone = miss.get(p, 0)
        items[p] = [
            tj.Item(
                p,
                f"c{i % 3}",
                f"{p}-{i}",
                2,
                1,
                CTX,
                None if i < gone else {**TINY, "arxiv_id": f"{p}-{i}"},
            )
            for i in range(tj.EXPECTED[p] - short.get(p, 0))
        ]
    return tj.Loaded(
        items, [f"c{i}" for i in range(cases)], drifted or [], t0_drift or [], {}, {}, {}, {}
    )


class TestWhatBothModesRefuse:
    """`--report` asks `readiness` the same questions as `--judge`, minus the key."""

    @pytest.fixture(autouse=True)
    def _own_state(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """readiness compares the case list with the ledger's. Without its own STATE every test
        here read the real run's ledger, and the clean-load test passed only until the first
        --judge wrote one."""
        monkeypatch.setattr(tj, "STATE", tmp_path)

    def test_a_clean_load_raises_nothing_about_the_populations(self) -> None:
        out = " | ".join(tj.readiness(loaded(), key=False, count=no_count))
        for quiet in ("drifted", "T0 context", "registered", "no prompt", "case list", "Tokens"):
            assert quiet not in out

    def test_report_skips_only_the_key_check(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(tj, "ENV_FILE", tmp_path / "absent.env")
        lo = loaded(drifted=["ann"])
        assert any(tj.KEY_NAME in r for r in tj.readiness(lo, count=no_count))
        without = tj.readiness(lo, key=False, count=no_count)
        assert not any(tj.KEY_NAME in r for r in without)
        assert any("HEAD contexts drifted" in r for r in without)

    def test_drift_and_short_populations_and_case_counts(self) -> None:
        out = tj.readiness(
            loaded(drifted=["ann"], t0_drift=["k"], short={"adopted": 1}, cases=36),
            key=False,
            count=no_count,
        )
        assert any("1 HEAD contexts drifted: ['ann']" in r for r in out)
        assert any("1 stored verdicts are on another T0 context" in r for r in out)
        assert any("adopted has 187 papers, registered 188" in r for r in out)
        assert any("the comparison has 36 cases, registered 37" in r for r in out)

    def test_a_paper_with_no_prompt_refuses_with_its_population_and_cause(self) -> None:
        out = tj.readiness(loaded(missing={"band": 2, "crossrepo": 1}), key=False, count=no_count)
        assert "band has 2 papers with no prompt: {'unresolved': 2}" in out
        assert "crossrepo has 1 papers with no prompt: {'unresolved': 1}" in out
        assert not any("ours has" in r for r in out)

    def test_a_case_list_that_moved_since_judge_refuses(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(tj, "STATE", tmp_path)
        lo = loaded()
        assert not any("case list" in r for r in tj.readiness(lo, key=False, count=no_count))
        ledger = {"cases": [*tj.case_list(lo), "band:gone"]}
        (tmp_path / "ledger.json").write_text(json.dumps(ledger), encoding="utf-8")
        out = tj.readiness(lo, key=False, count=no_count)
        assert any("case list differs from the one the ledger started with" in r for r in out)

    def test_counttokens_validates_the_whole_body_and_a_400_refuses(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        env = tmp_path / ".env"
        env.write_text(f"{tj.KEY_NAME}=not-a-real-key\n", encoding="utf-8")
        monkeypatch.setattr(tj, "ENV_FILE", env)
        seen: list[dict[str, Any]] = []

        def count(request: dict[str, Any], model: str, key: str) -> tuple[int, Any]:
            seen.append(request)
            return 400, {"error": {"message": "Invalid value at generation_config"}}

        out = tj.readiness(loaded(), key=False, count=count)
        assert seen and seen[0].keys() == {"contents", "generationConfig", "safetySettings"}
        assert seen[0]["generationConfig"]["thinkingConfig"] == {"thinkingLevel": "MEDIUM"}
        assert any("countTokens refuses the registered request body with 400" in r for r in out)
        assert any("Invalid value at generation_config" in r for r in out)

    def test_a_countTokens_that_answers_is_not_a_refusal(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        env = tmp_path / ".env"
        env.write_text(f"{tj.KEY_NAME}=not-a-real-key\n", encoding="utf-8")
        monkeypatch.setattr(tj, "ENV_FILE", env)
        for status in (200, None, 503):

            def count(*_: Any, s: int | None = status) -> tuple[int | None, Any]:
                return s, {}

            assert not any(
                "countTokens" in r for r in tj.readiness(loaded(), key=False, count=count)
            )


# ── The tracked artifact ──────────────────────────────────────────────────────────────────


@pytest.mark.skipif(
    not ARTIFACT.is_file(),
    reason="evals/third_judge.json is written by --report once the verdicts exist",
)
class TestTheTrackedArtifact:
    @pytest.fixture(scope="class")
    def data(self) -> dict[str, Any]:
        return json.loads(ARTIFACT.read_text(encoding="utf-8"))

    def test_the_summary_recomputes_from_the_rows(self, data: dict[str, Any]) -> None:
        assert json.loads(json.dumps(tj.summarise(data))) == data["summary"]

    def test_the_registered_populations(self, data: dict[str, Any]) -> None:
        for pop, n in tj.EXPECTED.items():
            assert len(tj.members_of(data["rows"], pop)) == n
        assert len(data["comparison_cases"]) == tj.N_CASES
        shas = [r["sha256"] for r in data["rows"] if r["sha256"]]
        assert len(shas) == len(set(shas))

    def test_every_score_is_an_int_from_0_to_3(self, data: dict[str, Any]) -> None:
        for r in data["rows"]:
            assert (r["score"] is None) != (r["void"] is None)
            assert r["score"] is None or (type(r["score"]) is int and 0 <= r["score"] <= 3)

    def test_the_existing_judges_reproduced(self, data: dict[str, Any]) -> None:
        assert data["reproductions"] and all(r["ok"] for r in data["reproductions"])

    def test_one_model_answered_every_prompt(self, data: dict[str, Any]) -> None:
        assert {r["model_version"] for r in data["rows"]} == {tj.MODEL}
        assert {r["finish_reason"] for r in data["rows"]} == {"STOP"}

    def test_the_run_was_frozen_at_its_start_commit(self, data: dict[str, Any]) -> None:
        """NR-69 quotes the head the run started from. The frozen files are the ones of 1a4914c."""
        assert data["frozen"]["head"] == "f2f9af671c452e60a28e50815e91f5b0c03cf726"
        assert set(data["frozen"]["files"]) == set(tj.FROZEN)

    def test_the_readings_quoted_in_nr69(self, data: dict[str, Any]) -> None:
        """The registered readings and the figures RESULTS.md and the paper quote beside them."""
        s = data["summary"]

        def r3(x: float) -> float:
            return round(x, 3)

        assert (r3(s["E1"]["rate"]), [r3(x) for x in s["E1"]["ci"]]) == (0.278, [0.210, 0.346])
        assert s["E1"]["reading"] == "below both"
        assert (r3(s["E2"]["auc"]), [r3(x) for x in s["E2"]["ci"]]) == (0.672, [0.574, 0.767])
        assert s["E2"]["reading"] == "orders the band"
        adopted, cross = s["E3"]["adopted"], s["E3"]["crossrepo"]
        assert (r3(adopted["rate"]), [r3(x) for x in adopted["ci"]]) == (0.468, [0.355, 0.571])
        assert adopted["reading"] == "below both"
        assert (r3(cross["rate"]), [r3(x) for x in cross["ci"]]) == (0.042, [0.018, 0.071])
        assert cross["reading"] == "overlaps Sonnet"
        assert r3(s["E3"]["auc"]["auc"]) == 0.879
        assert (round(s["E4"]["margin"], 2), [round(x, 2) for x in s["E4"]["ci"]]) == (
            -2.27,
            [-6.59, 2.35],
        )
        assert s["E4"]["reading"] == "between"
        assert {k: r3(v) for k, v in s["E4"]["precision"].items()} == {
            "ours": 0.373,
            "baseline": 0.493,
        }
        assert all(v["void"] == 0 for v in s["voids"].values())
