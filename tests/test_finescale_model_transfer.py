"""Tests for evals/finescale_model_transfer.py, the registered gpt-4.1-mini measurement.

Everything here is offline. The transport is exercised for real down to `_post_json`, which is
replaced, so the request the script validates is the one the shipped code builds. No test reads the
gitignored run files, pool or clones, so these run in CI.
"""

from __future__ import annotations

import io
import json
import math
import urllib.error
from pathlib import Path
from typing import Any, NamedTuple

import finescale_model_transfer as fmt
import pytest

from reporadar import azure_auth, llm_client
from reporadar.llm_client import LLMError, LLMRateLimited, LLMUnavailable
from reporadar.paper_id import dedup_id

PROMPT = "Respond with ONLY a single digit 0-9."


def _body(**over: Any) -> dict[str, Any]:
    body = {
        "model": "gpt-4.1-mini",
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": 4,
        "temperature": 0,
        "logprobs": True,
        "top_logprobs": 20,
    }
    body.update(over)
    return {k: v for k, v in body.items() if v is not None}


class TestTheRegisteredRequest:
    def test_the_body_the_transport_builds_is_valid(self) -> None:
        assert fmt.request_valid(_body(), PROMPT) == (True, "")

    def test_dropping_reasoning_effort_is_the_one_permitted_change(self) -> None:
        assert fmt.request_valid(_body(reasoning_effort="none"), PROMPT)[0]
        assert fmt.request_valid(_body(), PROMPT)[0]
        assert not fmt.request_valid(_body(reasoning_effort="low"), PROMPT)[0]

    @pytest.mark.parametrize(
        "body",
        [
            _body(max_tokens=None, max_completion_tokens=4),
            _body(temperature=None),
            _body(top_logprobs=5),
            _body(logprobs=None),
            _body(seed=1),
            _body(messages=[{"role": "user", "content": "something else"}]),
            _body(max_tokens=4.0),
        ],
    )
    def test_any_other_change_is_invalid(self, body: dict[str, Any]) -> None:
        assert not fmt.request_valid(body, PROMPT)[0]

    def test_no_captured_body_is_invalid(self) -> None:
        assert not fmt.request_valid(None, PROMPT)[0]


class TestModelIdentity:
    @pytest.mark.parametrize("name", ["gpt-4.1-mini-2025-04-14", "GPT-4.1-mini", "gpt-4.1-mini"])
    def test_the_registered_names_pass(self, name: str) -> None:
        assert fmt.model_ok({"model": name})

    @pytest.mark.parametrize("name", ["gpt-4o-mini", "gpt-4.1-mini-2025-05-01", "", None])
    def test_anything_else_stops(self, name: Any) -> None:
        assert not fmt.model_ok({"model": name})
        assert not fmt.model_ok(None)


def _http(
    code: int, body: dict | None = None, headers: dict | None = None
) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        "u", code, "err", headers or {}, io.BytesIO(json.dumps(body or {}).encode())
    )


def _answer(
    tokens: list[tuple[str, list[tuple[str, float]]]],
    model: str = "gpt-4.1-mini-2025-04-14",
    finish: str = "stop",
    content: str | None = None,
) -> dict[str, Any]:
    return {
        "model": model,
        "choices": [
            {
                "finish_reason": finish,
                "message": {
                    "content": "".join(t for t, _ in tokens) if content is None else content
                },
                "logprobs": {
                    "content": [
                        {
                            "token": tok,
                            "logprob": alts[0][1] if alts else 0.0,
                            "top_logprobs": [{"token": a, "logprob": lp} for a, lp in alts],
                        }
                        for tok, alts in tokens
                    ]
                },
            }
        ],
    }


def _digits(weights: dict[str, float]) -> list[tuple[str, float]]:
    return [(d, math.log(p)) for d, p in weights.items()]


class TestClassification:
    def test_the_order_puts_rate_limits_and_unavailable_before_other_errors(self) -> None:
        rl = LLMRateLimited("slow down")
        assert fmt.classify(rl, None) == "rate_limited"
        try:
            raise LLMUnavailable("403") from _http(403)
        except LLMUnavailable as exc:
            assert fmt.classify(exc, None) == "unavailable_http"
        assert fmt.classify(LLMUnavailable("az said no"), None) == "unavailable_token"

    def test_both_content_filter_shapes(self) -> None:
        cap400 = fmt.Capture(body=_body(), response=None, returned=False)
        exc400 = LLMError("Azure OpenAI (x)'s content filter blocked this prompt, so ...")
        assert fmt.classify(exc400, cap400) == "content_filtered"
        cap200 = fmt.Capture(
            body=_body(), response=_answer([], finish="content_filter"), returned=True
        )
        exc200 = LLMError("Azure OpenAI (x)'s content filter withheld the answer, so ...")
        assert fmt.classify(exc200, cap200) == "content_filtered"

    def test_an_empty_answer_is_no_digit_but_a_non_empty_one_without_logprobs_is_an_error(
        self,
    ) -> None:
        exc = LLMError("Azure OpenAI (x) returned no logprobs (model or account may not ...)")
        empty = fmt.Capture(body=_body(), response=_answer([], content=""), returned=True)
        assert fmt.classify(exc, empty) == "empty"
        spoken = fmt.Capture(body=_body(), response=_answer([], content="7"), returned=True)
        assert fmt.classify(exc, spoken) == "error"

    def test_anything_else_is_an_error(self) -> None:
        import http.client

        assert fmt.classify(http.client.IncompleteRead(b""), None) == "error"
        assert fmt.classify(AttributeError("x"), None) == "error"
        assert fmt.classify(LLMError("returned no choices"), None) == "error"


class TestTheTwoParsers:
    def test_a_digit_first_token_reads_the_same_under_both(self) -> None:
        r = fmt.readings(_answer([("7", _digits({"7": 0.6, "8": 0.4}))]))
        assert r["product_exp"] == pytest.approx(7.4)
        assert r["control_exp"] == pytest.approx(7.4)
        assert r["first_token_alternatives"] == 2

    def test_a_leading_non_digit_token_is_no_digit_for_the_product_only(self) -> None:
        """The product reads the first token's alternatives only; the control's parser reads the
        first digit token anywhere. A reply like 'Score: 7' separates them."""
        r = fmt.readings(_answer([("Score", [("Score", 0.0)]), ("7", _digits({"7": 1.0}))]))
        assert r["product_exp"] is None
        assert r["control_exp"] == pytest.approx(7.0)

    def test_a_digit_int_rejects_is_flagged_not_crashed(self) -> None:
        r = fmt.readings(_answer([("₂", [("₂", 0.0)])]))
        assert r["control_exp"] is None
        assert r["control_parse_raised"] is True


@pytest.fixture
def transport(monkeypatch: pytest.MonkeyPatch):  # noqa: ANN201
    """The real `top_logprobs`, `_post_adaptive` and `_post_json`, down to `urlopen`, which plays
    back *responses*: a dict is a 200 body, an HTTPError is raised as the server's refusal."""
    from unittest.mock import MagicMock

    monkeypatch.setattr(azure_auth, "get_token", lambda tenant="": "entra")
    monkeypatch.setattr(llm_client, "_post_adaptive", llm_client._post_adaptive)
    fmt.install_capture()
    for key in list(llm_client._REJECTS_EFFORT):
        if key.startswith("azure:"):
            llm_client._REJECTS_EFFORT.discard(key)
    responses: list[Any] = []

    def urlopen(req: Any, timeout: float = 0) -> Any:
        item = responses.pop(0)
        if isinstance(item, BaseException):
            raise item
        resp = MagicMock()
        resp.read.return_value = json.dumps(item).encode()
        resp.__enter__ = lambda s: s
        resp.__exit__ = MagicMock(return_value=False)
        return resp

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    monkeypatch.setattr(llm_client.time, "sleep", lambda s: None)
    yield responses
    for key in list(llm_client._REJECTS_EFFORT):
        if key.startswith("azure:"):
            llm_client._REJECTS_EFFORT.discard(key)


def _cfg() -> Any:
    cfg = fmt.treatment_cfg()
    cfg.azure_endpoint = "https://example-res.openai.azure.com"
    return cfg


def _paper() -> fmt.BandPaper:
    return fmt.BandPaper(
        case="c",
        id="2401.00001",
        vid="2401.00001v1",
        title="t",
        abstract="a",
        judge=2,
        prompt=PROMPT,
    )


class TestScoringOnePaper:
    def test_a_refused_reasoning_effort_is_dropped_and_the_row_is_valid(
        self, transport: list[Any]
    ) -> None:
        refused = _http(
            400, {"error": {"message": "Unrecognized request argument supplied: reasoning_effort"}}
        )
        transport.extend([refused, _answer([("7", _digits({"7": 1.0}))])])
        row = fmt.score_one(_paper(), _cfg(), sleep=lambda s: None)
        assert row["state"] == "answered"
        assert row["reasoning_effort_dropped"] is True
        assert row["product_exp"] == pytest.approx(7.0)
        assert "reasoning_effort" not in row["body_keys"]

    def test_a_first_token_without_a_digit_is_a_no_digit_row(self, transport: list[Any]) -> None:
        transport.append(_answer([("Score", [("Score", 0.0), ("The", -2.0)])]))
        assert fmt.score_one(_paper(), _cfg(), sleep=lambda s: None)["state"] == "no_digit"

    def test_rate_limits_wait_and_six_in_a_row_stop_resumably(self, transport: list[Any]) -> None:
        waits: list[float] = []
        for _ in range(6):
            transport.append(_http(429, headers={"retry-after": "60"}))
        with pytest.raises(fmt.Stop) as stop:
            fmt.score_one(_paper(), _cfg(), sleep=waits.append)
        assert stop.value.resumable and stop.value.kind == "rate_limit"
        assert waits == [60.0] * 5

    def test_errors_are_retried_five_times_then_become_an_error_row(
        self, transport: list[Any]
    ) -> None:
        # The transport itself makes 3 attempts per call on a 5xx; the script retries 5 times.
        transport.extend([_http(500) for _ in range(3 * 6)])
        row = fmt.score_one(_paper(), _cfg(), sleep=lambda s: None)
        assert row["state"] == "error"

    def test_a_wrong_model_stops_the_run(self, transport: list[Any]) -> None:
        transport.append(_answer([("7", _digits({"7": 1.0}))], model="gpt-4o-mini"))
        with pytest.raises(fmt.Stop) as stop:
            fmt.score_one(_paper(), _cfg(), sleep=lambda s: None)
        assert stop.value.kind == "model_identity" and not stop.value.resumable

    def test_a_403_is_a_final_stop(self, transport: list[Any]) -> None:
        transport.append(_http(403))
        with pytest.raises(fmt.Stop) as stop:
            fmt.score_one(_paper(), _cfg(), sleep=lambda s: None)
        assert stop.value.kind == "llm_unavailable" and not stop.value.resumable

    def test_a_filtered_400_keeps_its_body_and_message(self, transport: list[Any]) -> None:
        transport.append(_http(400, {"error": {"code": "content_filter", "message": "filtered"}}))
        row = fmt.score_one(_paper(), _cfg(), sleep=lambda s: None)
        assert row["state"] == "content_filtered" and row["filter_shape"] == 400
        assert "content filter" in row["filter_message"]


class TestTheReadingsAndTheOutcomeTable:
    def test_interval_indices_are_the_registered_ones(self) -> None:
        values = [float(i) for i in range(4000)]
        assert fmt.interval(values) == (100.0, 3899.0)

    @pytest.mark.parametrize(
        ("ci", "want"),
        [((-0.04, 0.10), "non-inferior"), ((-0.20, -0.06), "worse"), ((-0.08, 0.02), "unresolved")],
    )
    def test_judge_reading(self, ci: tuple[float, float], want: str) -> None:
        assert fmt.judge_reading(ci) == want

    def test_too_many_voids_force_sonnet_to_unresolved(self) -> None:
        assert fmt.judge_reading((0.0, 0.1), forced_unresolved=True) == "unresolved"

    @pytest.mark.parametrize(
        ("gpt", "son", "want"),
        [
            ("worse", "non-inferior", "worse"),
            ("non-inferior", "non-inferior", "non-inferior"),
            ("non-inferior", "unresolved", "split"),
            ("unresolved", "unresolved", "unresolved"),
        ],
    )
    def test_e1_table(self, gpt: str, son: str, want: str) -> None:
        assert fmt.e1_reading(gpt, son) == want

    @pytest.mark.parametrize(
        ("ci", "want"),
        [
            ((-0.05, 0.07), "holds"),
            ((0.09, 0.2), "over-admits"),
            ((-0.3, -0.1), "under-admits"),
            ((-0.02, 0.12), "unresolved"),
        ],
    )
    def test_e2_table(self, ci: tuple[float, float], want: str) -> None:
        assert fmt.e2_reading(ci) == want

    @staticmethod
    def _stats(**ci: Any) -> dict[str, Any]:
        base = {
            "d_auc_gpt": (-0.02, 0.05),
            "d_auc_son": (-0.03, 0.04),
            "auc_t_gpt": (0.6, 0.75),
            "auc_c_gpt": (0.59, 0.75),
            "A": (-0.02, 0.03),
            "A_threshold": (-0.02, 0.03),
        }
        base.update(ci)
        return {"intervals": base}

    def test_p(self) -> None:
        assert fmt.outcome_for(self._stats(), False)["row"] == "P"

    def test_o_over_and_under(self) -> None:
        assert (
            fmt.outcome_for(self._stats(A=(0.1, 0.2), A_threshold=(0.1, 0.2)), False)["row"]
            == "O-over"
        )
        assert (
            fmt.outcome_for(self._stats(A=(-0.3, -0.1), A_threshold=(-0.3, -0.1)), False)["row"]
            == "O-under"
        )

    def test_e2_needs_the_threshold_part_to_agree(self) -> None:
        out = fmt.outcome_for(self._stats(A=(0.1, 0.2), A_threshold=(-0.02, 0.03)), False)
        assert out["e2"] == "unresolved" and out["row"] == "U"

    def test_w1_before_everything(self) -> None:
        out = fmt.outcome_for(self._stats(d_auc_son=(-0.2, -0.06), auc_t_gpt=(0.45, 0.6)), False)
        assert out["row"] == "W1"

    def test_the_floor_is_w2_only_when_the_control_cleared_chance(self) -> None:
        floor = {"auc_t_gpt": (0.45, 0.6), "d_auc_gpt": (-0.1, 0.0)}
        assert fmt.outcome_for(self._stats(**floor), False)["row"] == "W2"
        assert fmt.outcome_for(self._stats(**floor, auc_c_gpt=(0.48, 0.7)), False)["row"] == "U"

    def test_split_and_unresolved_are_u(self) -> None:
        assert fmt.outcome_for(self._stats(d_auc_son=(-0.08, 0.02)), False)["row"] == "U"
        assert fmt.outcome_for(self._stats(), True)["row"] == "U"

    def test_readings_must_agree(self) -> None:
        assert fmt.combine({"control_parser": "P", "product_parser": "P"}) == "P"
        assert fmt.combine({"control_parser": "O-over", "product_parser": "O-under"}) == "U"
        assert fmt.combine({"control_parser": "W1", "product_parser": "W2"}) == "U"


def _items(
    shift: float = 0.0, *, cases: int = 12, per: int = 10, unscored: set | None = None
) -> list[fmt.Item]:
    """A synthetic band: control scores rise with the label, the treatment is control + shift."""
    items = []
    k = 0
    for c in range(cases):
        for i in range(per):
            label = 3 if (i + c) % 3 else 1
            ctrl = 4.0 + (i % 5) * 0.8 + (1.2 if label >= 2 else 0.0)
            t = None if unscored and (c, i) in unscored else ctrl + shift
            items.append(
                fmt.Item(
                    case=f"c{c:02d}",
                    id=f"p{k}",
                    gpt=label,
                    son=label if i % 2 else 1,
                    ctrl=ctrl,
                    state="answered" if t is not None else "no_digit",
                    t_exp={"control_parser": t, "product_parser": t},
                )
            )
            k += 1
    return items


class TestTheStatisticsOnASyntheticBand:
    def test_an_identical_scorer_reads_p(self) -> None:
        items = _items()
        cases = sorted({it.case for it in items})
        stats = fmt.reading_stats(items, cases, "product_parser")
        assert stats["point"]["d_auc_gpt"] == pytest.approx(0.0)
        assert stats["point"]["A"] == pytest.approx(0.0)
        assert fmt.outcome_for(stats, False)["row"] == "P"

    def test_a_uniform_upward_shift_over_admits(self) -> None:
        items = _items(shift=2.0)
        cases = sorted({it.case for it in items})
        stats = fmt.reading_stats(items, cases, "product_parser")
        assert stats["point"]["d_auc_gpt"] == pytest.approx(0.0)
        assert fmt.outcome_for(stats, False)["row"] == "O-over"

    def test_a_case_with_under_half_scored_is_admitted_whole(self) -> None:
        unscored = {(0, i) for i in range(6)}  # 6 of case c00's 10 papers unscored
        items = _items(unscored=unscored)
        cases = sorted({it.case for it in items})
        stats = fmt.reading_stats(items, cases, "product_parser")
        assert stats["fallback_cases"] == ["c00"]
        # The fallback admissions land in the failure part, not the threshold part.
        assert stats["point"]["A_threshold"] == pytest.approx(0.0)
        assert stats["point"]["A_failure"] > 0

    def test_error_rows_leave_both_arms(self) -> None:
        items = _items()
        items[0].state = "error"
        items[0].t_exp = {"control_parser": None, "product_parser": None}
        cases = sorted({it.case for it in items})
        stats = fmt.reading_stats(items, cases, "product_parser")
        assert stats["counts"]["errors_dropped"] == 1
        assert stats["counts"]["kept"] == len(items) - 1


def test_the_registered_bootstrap_reproduces_band_ls_control_interval() -> None:
    """From the tracked control artifact alone: the registration quotes [0.594, 0.747]."""
    import band_testbeds as tb

    path = Path(fmt.EVALS) / "finescale_current_gate_luna.json"
    rows = json.loads(path.read_text(encoding="utf-8"))["rows"]
    cases = sorted({r["case"] for r in rows})
    by_case: dict[str, list[dict[str, Any]]] = {c: [] for c in cases}
    for r in rows:
        by_case[r["case"]].append(r)
    aucs = []
    for draw in fmt.draws_for(cases):
        pool = [r for c in draw for r in by_case[c]]
        labels = [r["judge"] >= tb.ACTIONABLE for r in pool]
        if 0 < sum(labels) < len(labels):
            aucs.append(tb.auc([r["exp09"] for r in pool], labels))
    lo, hi = fmt.interval(aucs)
    assert (round(lo, 3), round(hi, 3)) == (0.594, 0.747)


def test_the_script_never_names_the_azure_resource() -> None:
    """The resource comes from the environment only; the source may not spell any of its names.

    Checked with the script's own helpers, so this file spells no name either."""
    source = Path(fmt.__file__).read_text(encoding="utf-8")
    assert not fmt._AZURE_HOST.search(source)
    low = source.lower()
    assert not any(n.lower() in low for n in fmt._resource_names())


HOST = "zz" + "fake-res"  # assembled, so no real resource name is ever spelled here


class TestTheResourceNeverReachesTheArtifact:
    @pytest.fixture(autouse=True)
    def _endpoint(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("RR_TRANSFER_AZURE_ENDPOINT", f"https://{HOST}.openai.azure.com")
        monkeypatch.setenv("RR_TRANSFER_AZURE_RG", "rg-" + "zzsecret")
        monkeypatch.setenv("RR_TRANSFER_AZURE_ACCOUNT", HOST)

    def _cfg(self) -> Any:
        return fmt.treatment_cfg()

    def test_a_filtered_400_message_is_scrubbed(self, transport: list[Any]) -> None:
        transport.append(_http(400, {"error": {"code": "content_filter", "message": "x"}}))
        row = fmt.score_one(_paper(), self._cfg(), sleep=lambda s: None)
        assert row["state"] == "content_filtered"
        assert HOST not in json.dumps(row)

    @pytest.mark.parametrize("code", [401, 403, 404])
    def test_refusal_stops_carry_only_the_status(self, transport: list[Any], code: int) -> None:
        transport.append(_http(code))
        with pytest.raises(fmt.Stop) as stop:
            fmt.score_one(_paper(), self._cfg(), sleep=lambda s: None)
        assert HOST not in str(stop.value) and stop.value.detail == f"HTTP {code}"

    def test_an_error_row_is_scrubbed(self, transport: list[Any]) -> None:
        transport.extend([_http(418) for _ in range(6)])
        row = fmt.score_one(_paper(), self._cfg(), sleep=lambda s: None)
        assert row["state"] == "error" and HOST not in json.dumps(row)

    def test_save_artifact_scrubs_and_never_writes_the_name(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(fmt, "OUT", tmp_path / "art.json")
        fmt.save_artifact({"x": f"Azure OpenAI ({HOST}.openai.azure.com) said no, rg-zzsecret"})
        text = (tmp_path / "art.json").read_text(encoding="utf-8")
        assert HOST not in text and "zzsecret" not in text


class TestRowsKeepTheCapture:
    def test_the_body_as_sent_and_the_response_are_kept(self, transport: list[Any]) -> None:
        refused = _http(
            400, {"error": {"message": "Unrecognized request argument supplied: reasoning_effort"}}
        )
        transport.extend([refused, _answer([("7", _digits({"7": 1.0}))])])
        row = fmt.score_one(_paper(), _cfg(), sleep=lambda s: None)
        assert row["request_body"]["messages"][0]["content"] == PROMPT
        assert "reasoning_effort" not in row["request_body"]
        assert row["response"]["model"] == "gpt-4.1-mini-2025-04-14"
        assert row["message_sha256"] == fmt.sha(PROMPT)

    def test_the_artifact_copy_hashes_the_prompt(self, transport: list[Any]) -> None:
        transport.append(_answer([("7", _digits({"7": 1.0}))]))
        row = fmt.artifact_row(fmt.score_one(_paper(), _cfg(), sleep=lambda s: None))
        assert row["request_body"]["messages"][0]["content"] == fmt.sha(PROMPT)
        assert row["tokens"] == ["7"]


def _segment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, n: int = 3) -> fmt.Segment:
    monkeypatch.setattr(fmt, "PASS_DIRS", {"first": tmp_path / "first", "retest": tmp_path / "re"})
    monkeypatch.setattr(fmt, "ERROR_LOG", tmp_path / "errors.jsonl")
    papers = [
        fmt.BandPaper(case="c", id=f"p{i}", vid=f"p{i}v1", title="t", abstract="a", judge=2)
        for i in range(n)
    ]
    return fmt.Segment("L-first", "first", papers, papers, None, None)


class TestPassesAreCountedAcrossStopsAndRestarts:
    def test_an_always_failing_paper_is_asked_once_per_pass(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seg = _segment(tmp_path, monkeypatch)
        asked: list[str] = []
        stopped = {"done": False}

        def score_one(paper: Any, cfg: Any, sleep: Any) -> dict[str, Any]:
            asked.append(paper.id)
            if paper.id == "p0":
                return {"case": "c", "id": "p0", "state": "error", "error_type": "X", "error": "x"}
            if paper.id == "p1" and not stopped["done"]:
                stopped["done"] = True
                raise fmt.Stop("rate_limit", "p1", resumable=True)
            return {"case": "c", "id": paper.id, "state": "answered", "product_exp": 7.0}

        monkeypatch.setattr(fmt, "score_one", score_one)
        progress: dict[str, Any] = {"pass": 0, "asked": []}
        with pytest.raises(fmt.Stop):
            fmt.run_segment(seg, None, lambda s: None, progress)
        # Resumed: the stopped pass continues, and p0 is not asked again in it.
        errors = fmt.run_segment(seg, None, lambda s: None, progress)
        assert asked.count("p0") == 1 + fmt.RESUME_PASSES
        assert errors == 1
        assert len((tmp_path / "errors.jsonl").read_text(encoding="utf-8").splitlines()) == 4
        # A restart after the passes are spent asks nothing more.
        before = len(asked)
        fmt.run_segment(seg, None, lambda s: None, progress)
        assert len(asked) == before


class TestAFinishedRunTakesNoMoreRequests:
    @pytest.mark.parametrize(
        "run",
        [
            {"ended": "X", "segments": {}},
            {"ended_after_band_l": "model_identity", "segments": {}},
            {"segments": {"L-first": "complete", "L-retest": "complete", "H-rest": "complete"}},
        ],
    )
    def test_finished(self, run: dict[str, Any]) -> None:
        assert fmt.finished(run)

    def test_a_run_in_progress_is_not_finished(self) -> None:
        assert fmt.finished({"segments": {"L-first": "complete"}}) is None


class TestTheReportNeverReadsAPartialRun:
    def test_x_reports_the_outcome_and_no_endpoint(self) -> None:
        art = {"run": {"ended": "X", "stops": [{"kind": "transport"}], "segments": {}}}
        out = fmt.report_stage({}, art)
        assert out["outcome"] == "X" and "bands" not in out

    def test_an_unfinished_band_l_is_refused(self) -> None:
        art = {"run": {"stops": [], "segments": {}}}
        with pytest.raises(SystemExit):
            fmt.report_stage({}, art)


def test_a_systemic_sonnet_failure_stops_before_anything_is_voided(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import gate_swap_second_judge
    import run_judge_eval
    import second_judge

    papers = [
        fmt.BandPaper(case="c", id=f"p{i}", vid=f"p{i}v1", title="t", abstract="a", judge=2)
        for i in range(5)
    ]
    empty = fmt.sha("")
    band = fmt.BANDS["L"]
    fake = {
        n: fmt.Band(**{**band.__dict__, "name": n, "sonnet_fp": empty, "sonnet_count": 0})
        for n in ("L", "H")
    }
    monkeypatch.setattr(fmt, "BANDS", fake)
    monkeypatch.setattr(fmt, "OUT", tmp_path / "art.json")
    monkeypatch.setattr(fmt, "cached_sonnet", lambda case, vid: None)
    monkeypatch.setattr(fmt.time, "sleep", lambda s: None)
    monkeypatch.setattr(second_judge, "verify_contexts", lambda cases: ({"c": "ctx"}, []))
    monkeypatch.setattr(run_judge_eval, "load_dotenv", lambda path: None)
    monkeypatch.setattr(
        gate_swap_second_judge, "pool_meta", lambda: {("c", p.vid): {"title": "t"} for p in papers}
    )

    def refuse(*a: Any, **k: Any) -> int:
        raise LLMError("upstream is down")

    monkeypatch.setattr(second_judge, "second_verdict", refuse)
    art = fmt.load_artifact()
    with pytest.raises(SystemExit):
        fmt.sonnet_stage({"L": papers, "H": []}, art)
    assert "step0" in art["sonnet"]
    assert "void" not in art["sonnet"] and "label_set" not in art["sonnet"]


def test_papers_still_in_error_are_counted_as_errors_not_missing() -> None:
    """Error rows are never cached, so the report learns about them from the error log. A paper
    in the log that a later pass scored stays scored."""
    papers = [
        fmt.BandPaper(case="c", id=f"p{i}", vid=f"p{i}v1", title="t", abstract="a", judge=2)
        for i in range(3)
    ]
    control = {("c", p.id): {"exp09": 5.0} for p in papers}
    rows = {("c", "p1"): {"state": "answered", "product_exp": 6.0, "control_exp": 6.0}}
    errored = {("c", "p0"), ("c", "p1")}
    items = fmt.items_for(papers, control, rows, {}, errored)
    assert [it.state for it in items] == ["error", "answered", "missing"]
    stats = fmt.reading_stats(items, ["c"], "product_parser")
    assert stats["counts"]["errors_dropped"] == 1 and stats["counts"]["missing"] == 1


# ── NR-65, pinned from tracked files ──────────────────────────────────────────────────────
#
# Everything above runs the script on synthetic input, so none of it could notice the NR-65 entry
# in evals/RESULTS.md drifting from the run it reports. The tests below pin the entry's numbers.
# Each one is recomputed from per-paper rows in tracked files, never read back from the summary
# the script wrote. The recomputed value is then held against that summary too, so neither the
# prose nor the summary can move without the rows moving. The estimators, parsers, map and
# threshold are the script's own (fmt.tb, fmt.ef, fmt.finescale, fmt.admitted), because the point
# is to recompute the run's numbers, not to measure them a second way.

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "evals" / "finescale_model_transfer.json"
# Each band's gpt-4o-mini control, which also carries the band's GPT-5.5 labels.
CONTROLS = {
    "L": ROOT / "evals" / "finescale_current_gate_luna.json",
    "H": ROOT / "evals" / "finescale_current_gate.json",
}
# The artifact's two readings and the row field each one reads.
FIELDS = {"control_parser": "control_exp", "product_parser": "product_exp"}
# The characters RESULTS names: str.isdigit() accepts them and int() rejects them.
INT_REJECTS = {"⑤", "₂", "³"}  # circled five, subscript two, superscript three

Key = tuple[str, str]
Pair = tuple[dict[str, Any], dict[str, Any]]  # (control row, the treatment's reparsed reading)


def _key(case: str, paper_id: str) -> Key:
    return case, dedup_id(paper_id)


def _response(row: dict[str, Any]) -> dict[str, Any]:
    """The response the live run parsed, rebuilt from the tokens and alternatives the artifact
    keeps. A token's own logprob is not kept, and neither parser reads it."""
    content = [
        {"token": tok, "top_logprobs": [{"token": a, "logprob": lp} for a, lp in alts]}
        for tok, alts in zip(row["tokens"], row["alternatives"], strict=True)
    ]
    return {"choices": [{"finish_reason": row["finish_reason"], "logprobs": {"content": content}}]}


def _int_rejects(text: str) -> bool:
    try:
        int(text)
    except ValueError:
        return True
    return False


@pytest.fixture(scope="module")
def transfer() -> dict[str, Any]:
    # Asserted rather than skipped. The artifact is tracked, so its absence is a broken
    # repository, and a skip would leave NR-65 unguarded while the suite stayed green.
    assert ARTIFACT.is_file(), f"{ARTIFACT} is tracked and must be present"
    data: dict[str, Any] = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    return data


@pytest.fixture(scope="module")
def control_files() -> dict[str, dict[str, Any]]:
    out = {}
    for name, path in CONTROLS.items():
        assert path.is_file(), f"{path} is tracked and must be present"
        out[name] = json.loads(path.read_text(encoding="utf-8"))
    return out


@pytest.fixture(scope="module")
def stored(transfer: dict[str, Any]) -> dict[str, dict[Key, dict[str, Any]]]:
    """The artifact's rows per pass, keyed by case and version-stripped id."""
    return {
        pass_name: {_key(r["case"], r["id"]): r for r in rows.values()}
        for pass_name, rows in transfer["rows"].items()
    }


@pytest.fixture(scope="module")
def reparsed(
    stored: dict[str, dict[Key, dict[str, Any]]],
) -> dict[str, dict[Key, dict[str, Any]]]:
    """Both parsers' readings of every row, recomputed from its stored tokens by `fmt.readings`.
    Everything downstream reads these, not the expectations the run stored."""
    return {
        pass_name: {k: fmt.readings(_response(r)) for k, r in rows.items()}
        for pass_name, rows in stored.items()
    }


@pytest.fixture(scope="module")
def band_keys(control_files: dict[str, dict[str, Any]]) -> dict[str, set[Key]]:
    return {
        name: {_key(r["case"], r["id"]) for r in data["rows"]}
        for name, data in control_files.items()
    }


@pytest.fixture(scope="module")
def bands(
    control_files: dict[str, dict[str, Any]], reparsed: dict[str, dict[Key, dict[str, Any]]]
) -> dict[str, list[Pair]]:
    """Each band as (control row, first-pass reading) pairs. Band H's papers shared with band L
    were scored once, in band L's first pass, so both bands read the same first-pass row."""
    first = reparsed["first"]
    return {
        name: [(c, first[_key(c["case"], c["id"])]) for c in data["rows"]]
        for name, data in control_files.items()
    }


def _fallback(pairs: list[Pair], field: str) -> dict[str, bool]:
    """Per case, whether the product would admit it whole: under half its papers scored."""
    scored: dict[str, list[bool]] = {}
    for c, t in pairs:
        scored.setdefault(c["case"], []).append(t[field] is not None)
    return {
        case: not fmt.finescale.enough_scored(sum(s), len(s), 0.5) for case, s in scored.items()
    }


class _Paper(NamedTuple):
    actionable: bool  # under GPT-5.5's label
    treatment: float | None  # gpt-4.1-mini's exp09 under one reading
    control: float  # gpt-4o-mini's exp09
    adm_t: bool
    adm_c: bool
    threshold_part: bool  # scored, in a case where the stage ran


def _papers(pairs: list[Pair], field: str) -> dict[str, list[_Paper]]:
    fallback = _fallback(pairs, field)
    by_case: dict[str, list[_Paper]] = {case: [] for case in sorted(fallback)}
    for c, t in pairs:
        x, whole = t[field], fallback[c["case"]]
        by_case[c["case"]].append(
            _Paper(
                actionable=c["judge"] >= fmt.tb.ACTIONABLE,
                treatment=x,
                control=float(c["exp09"]),
                adm_t=whole or fmt.admitted(x),
                adm_c=fmt.admitted(c["exp09"]),
                threshold_part=x is not None and not whole,
            )
        )
    return by_case


def _points(pairs: list[Pair], field: str) -> dict[str, Any]:
    """One band's GPT-5.5 point estimates under one reading."""
    by_case = _papers(pairs, field)
    papers = [p for group in by_case.values() for p in group]
    scored = [p for p in papers if p.treatment is not None]
    labels = [p.actionable for p in scored]
    # Both arms are ranked on the papers this reading scored. So the control's AUC depends on the
    # reading, and under the control-parser reading it is not the registered 0.675.
    auc_t = fmt.tb.auc([p.treatment for p in scored], labels)
    auc_c = fmt.tb.auc([p.control for p in scored], labels)
    n = len(papers)
    return {
        "scored": len(scored),
        "fallback": sorted(case for case, whole in _fallback(pairs, field).items() if whole),
        "auc_t_gpt": auc_t,
        "auc_c_gpt": auc_c,
        "d_auc_gpt": auc_t - auc_c,
        "A": sum(p.adm_t - p.adm_c for p in papers) / n,
        "A_threshold": sum(p.adm_t - p.adm_c for p in papers if p.threshold_part) / n,
        "admissions": {
            "treatment": sum(p.adm_t for p in papers),
            "control": sum(p.adm_c for p in papers),
            "only_treatment": sum(p.adm_t and not p.adm_c for p in papers),
            "only_control": sum(p.adm_c and not p.adm_t for p in papers),
        },
    }


def _intervals(pairs: list[Pair], field: str) -> dict[str, tuple[float, float] | None]:
    """The registered paired case bootstrap for the GPT-5.5 endpoints. The draws and interval
    indices are the script's; the pooled statistics are recomputed here from the rows."""
    by_case = _papers(pairs, field)
    stats: dict[str, list[float]] = {"A": [], "A_threshold": [], "d_auc_gpt": [], "auc_t_gpt": []}
    for draw in fmt.draws_for(sorted(by_case)):
        pool = [p for case in draw for p in by_case[case]]
        n = len(pool)
        stats["A"].append(sum(p.adm_t - p.adm_c for p in pool) / n)
        stats["A_threshold"].append(sum(p.adm_t - p.adm_c for p in pool if p.threshold_part) / n)
        scored = [p for p in pool if p.treatment is not None]
        labels = [p.actionable for p in scored]
        if 0 < sum(labels) < len(labels):
            auc_t = fmt.tb.auc([p.treatment for p in scored], labels)
            stats["auc_t_gpt"].append(auc_t)
            stats["d_auc_gpt"].append(auc_t - fmt.tb.auc([p.control for p in scored], labels))
    return {key: fmt.interval(values) for key, values in stats.items()}


@pytest.fixture(scope="module")
def band_l_intervals(bands: dict[str, list[Pair]]) -> dict[str, dict[str, Any]]:
    # About two seconds per reading, so computed once for the module.
    return {reading: _intervals(bands["L"], field) for reading, field in FIELDS.items()}


@pytest.fixture(scope="module")
def band_h_intervals(bands: dict[str, list[Pair]]) -> dict[str, dict[str, Any]]:
    return {reading: _intervals(bands["H"], field) for reading, field in FIELDS.items()}


class TestNR65TheArtifactAndItsPopulations:
    def test_the_tracked_artifact_is_the_one_the_script_writes(self) -> None:
        assert fmt.OUT == ARTIFACT
        assert ARTIFACT.is_file(), f"{ARTIFACT} is tracked and must be present"

    def test_the_controls_are_the_registered_bands(
        self, control_files: dict[str, dict[str, Any]]
    ) -> None:
        """Band L's control is the Luna-gated file and band H's the Haiku one (C-37; the
        registration calls it C-36)."""
        for name, data in control_files.items():
            band = fmt.BANDS[name]
            assert band.control == CONTROLS[name].name
            assert data["summary"]["run"] == band.run
            assert len(data["rows"]) == band.size
        gate = control_files["H"]["summary"]["gate"]
        assert (gate["provider"], gate["model"]) == fmt.BANDS["H"].gate

    def test_the_first_pass_scored_both_bands_once(
        self,
        transfer: dict[str, Any],
        stored: dict[str, dict[Key, dict[str, Any]]],
        band_keys: dict[str, set[Key]],
    ) -> None:
        # No two rows collapse to one paper under the shared id rule.
        assert len(stored["first"]) == len(transfer["rows"]["first"]) == 478
        assert (len(band_keys["L"]), len(band_keys["H"])) == (315, 328)
        assert len(band_keys["L"] & band_keys["H"]) == 165
        assert set(stored["first"]) == band_keys["L"] | band_keys["H"]

    def test_the_retest_is_band_l_exactly(
        self,
        transfer: dict[str, Any],
        stored: dict[str, dict[Key, dict[str, Any]]],
        band_keys: dict[str, set[Key]],
    ) -> None:
        assert len(stored["retest"]) == len(transfer["rows"]["retest"]) == 315
        assert set(stored["retest"]) == band_keys["L"]

    def test_all_793_calls_answered(self, stored: dict[str, dict[Key, dict[str, Any]]]) -> None:
        rows = [r for pass_rows in stored.values() for r in pass_rows.values()]
        assert len(rows) == 793
        assert {r["state"] for r in rows} == {"answered"}
        assert {r["response_model"] for r in rows} == {"gpt-4.1-mini-2025-04-14"}


class TestNR65BothParsersFromTheStoredTokens:
    def test_the_product_parser_reproduces_every_stored_expectation(
        self,
        stored: dict[str, dict[Key, dict[str, Any]]],
        reparsed: dict[str, dict[Key, dict[str, Any]]],
    ) -> None:
        """RESULTS: the product's parser scored all 643 band papers, in both passes."""
        wrong = [
            (pass_name, k)
            for pass_name, rows in stored.items()
            for k, r in rows.items()
            if reparsed[pass_name][k]["product_exp"] is None
            or not math.isclose(reparsed[pass_name][k]["product_exp"], r["product_exp"])
        ]
        assert not wrong

    def test_the_control_parser_fails_exactly_where_the_artifact_says(
        self,
        transfer: dict[str, Any],
        stored: dict[str, dict[Key, dict[str, Any]]],
        reparsed: dict[str, dict[Key, dict[str, Any]]],
        bands: dict[str, list[Pair]],
    ) -> None:
        for pass_name, rows in stored.items():
            got = reparsed[pass_name]
            raised = {k for k, r in got.items() if r["control_parse_raised"]}
            assert raised == {k for k, r in rows.items() if r["control_parse_raised"]}
            assert raised == {k for k, r in rows.items() if r["control_exp"] is None}
            assert all(got[k]["control_exp"] is None for k in raised)
            wrong = [
                k
                for k, r in rows.items()
                if k not in raised and not math.isclose(got[k]["control_exp"], r["control_exp"])
            ]
            assert not wrong
        counts = {
            name: sum(t["control_parse_raised"] for _, t in pairs) for name, pairs in bands.items()
        }
        assert counts == {"L": 38, "H": 23}
        summary = transfer["summary"]["bands"]
        assert counts == {name: summary[name]["control_parse_raised"] for name in counts}

    def test_it_fails_on_a_digit_int_rejects_behind_an_ascii_first_token(
        self, stored: dict[str, dict[Key, dict[str, Any]]]
    ) -> None:
        """RESULTS: every first token was an ASCII digit, and the control's parser failed on the
        circled five, subscript two or superscript three among its alternatives."""
        seen: set[str] = set()
        for rows in stored.values():
            for r in rows.values():
                assert r["tokens"][0] in set("0123456789")
                if not r["control_parse_raised"]:
                    continue
                content = _response(r)["choices"][0]["logprobs"]["content"]
                with pytest.raises(ValueError):
                    fmt.ef._digit_expectation(fmt._as_objects(content))
                bad = {
                    a.strip()
                    for a, _ in r["alternatives"][0]
                    if a.strip().isdigit() and _int_rejects(a.strip())
                }
                assert bad, "a row that raised must hold a digit int() rejects"
                seen |= bad
        assert seen == INT_REJECTS


class TestNR65TheGptEndpoints:
    """Band L and band H under GPT-5.5 labels, from the rows and the two tracked controls."""

    def test_band_l_product_reading(self, bands: dict[str, list[Pair]]) -> None:
        pt = _points(bands["L"], "product_exp")
        assert pt["scored"] == 315
        assert pt["auc_t_gpt"] == pytest.approx(0.711, abs=0.001)
        assert pt["auc_c_gpt"] == pytest.approx(0.675, abs=0.001)
        assert pt["d_auc_gpt"] == pytest.approx(0.036, abs=0.001)
        # Every paper is scored, so the control's AUC is the one the registration fixed.
        assert round(pt["auc_c_gpt"], 4) == fmt.BANDS["L"].control_auc

    def test_band_l_control_parser_reading(self, bands: dict[str, list[Pair]]) -> None:
        pt = _points(bands["L"], "control_exp")
        assert pt["scored"] == 315 - 38
        assert pt["d_auc_gpt"] == pytest.approx(0.034, abs=0.001)
        # Case http lost over half its papers to the control's parser, so it is admitted whole.
        assert pt["fallback"] == ["http"]

    def test_band_h_product_reading(self, bands: dict[str, list[Pair]]) -> None:
        pt = _points(bands["H"], "product_exp")
        assert pt["scored"] == 328
        assert pt["d_auc_gpt"] == pytest.approx(-0.026, abs=0.001)
        assert round(pt["auc_c_gpt"], 4) == fmt.BANDS["H"].control_auc

    def test_band_l_admissions(self, bands: dict[str, list[Pair]]) -> None:
        """181 against 173, and 72 papers admitted by one arm only: 8 more of 315."""
        pt = _points(bands["L"], "product_exp")
        assert pt["admissions"] == {
            "treatment": 181,
            "control": 173,
            "only_treatment": 40,
            "only_control": 32,
        }
        assert pt["A"] == pytest.approx(8 / 315)
        assert pt["A"] == pytest.approx(0.025, abs=0.001)

    def test_band_h_admissions(self, bands: dict[str, list[Pair]]) -> None:
        adm = _points(bands["H"], "product_exp")["admissions"]
        assert (adm["treatment"], adm["control"]) == (249, 231)

    @pytest.mark.parametrize("name", ["L", "H"])
    @pytest.mark.parametrize("reading", list(FIELDS))
    def test_the_points_equal_the_artifacts_summary(
        self, transfer: dict[str, Any], bands: dict[str, list[Pair]], name: str, reading: str
    ) -> None:
        pt = _points(bands[name], FIELDS[reading])
        stats = transfer["summary"]["bands"][name]["readings"][reading]
        for key in ("auc_t_gpt", "auc_c_gpt", "d_auc_gpt", "A", "A_threshold"):
            assert pt[key] == pytest.approx(stats["point"][key], abs=1e-12), key
        assert pt["admissions"] == stats["admissions"]
        assert pt["fallback"] == stats["fallback_cases"]
        assert pt["scored"] == stats["counts"]["scored"]


class TestNR65TheRetest:
    def test_band_l_scored_twice(
        self, transfer: dict[str, Any], reparsed: dict[str, dict[Key, dict[str, Any]]]
    ) -> None:
        """Mean absolute exp09 difference 0.086, Spearman 0.993, 9 of 315 admissions flipped."""
        first, retest = reparsed["first"], reparsed["retest"]
        pairs = [(first[k]["product_exp"], r["product_exp"]) for k, r in retest.items()]
        assert len(pairs) == 315
        mad = sum(abs(a - b) for a, b in pairs) / len(pairs)
        rho = fmt.spearman([a for a, _ in pairs], [b for _, b in pairs])
        flips = sum(fmt.admitted(a) != fmt.admitted(b) for a, b in pairs)
        assert mad == pytest.approx(0.086, abs=0.001)
        assert rho == pytest.approx(0.993, abs=0.001)
        assert flips == 9
        recorded = transfer["summary"]["retest"]["product_exp"]
        assert recorded == pytest.approx(
            {"n": 315, "mean_abs_diff": mad, "spearman": rho, "admission_flips": flips}, abs=1e-12
        )


class TestNR65TheRegisteredOutcome:
    def test_the_recorded_outcome_is_u_under_both_readings(self, transfer: dict[str, Any]) -> None:
        summary = transfer["summary"]
        assert summary["outcome_per_reading"] == {"control_parser": "U", "product_parser": "U"}
        assert summary["outcome"] == "U" == fmt.combine(summary["outcome_per_reading"])

    def test_each_recorded_row_follows_from_its_recorded_intervals(
        self, transfer: dict[str, Any]
    ) -> None:
        band_l = transfer["summary"]["bands"]["L"]
        forced = band_l["void"] > fmt.VOID_LIMIT
        for reading in FIELDS:
            stats = band_l["readings"][reading]
            assert fmt.outcome_for(stats, sonnet_forced=forced) == stats["outcome"]

    @pytest.mark.parametrize(
        ("reading", "d_auc", "e2"),
        [
            ("product_parser", (-0.025, 0.100), (-0.038, 0.088)),
            ("control_parser", (-0.041, 0.109), (-0.035, 0.091)),
        ],
    )
    def test_band_l_gpt_intervals_recomputed_from_the_rows(
        self,
        transfer: dict[str, Any],
        band_l_intervals: dict[str, dict[str, Any]],
        reading: str,
        d_auc: tuple[float, float],
        e2: tuple[float, float],
    ) -> None:
        """What makes U without Sonnet. GPT-5.5's half of E1 is non-inferior, the floor did not
        fire, and E2 misses "the threshold holds" by its upper bound. Under the registered table
        that is U unless Sonnet read worse, which would be W1. Sonnet's verdicts are not in this
        artifact, so that half is checked against the record above, not recomputed."""
        got = band_l_intervals[reading]
        assert got["d_auc_gpt"] == pytest.approx(d_auc, abs=0.001)
        assert got["A"] == pytest.approx(e2, abs=0.001)
        recorded = transfer["summary"]["bands"]["L"]["readings"][reading]["intervals"]
        for key, ci in got.items():
            assert ci == pytest.approx(tuple(recorded[key]), abs=1e-12), key
        assert fmt.judge_reading(got["d_auc_gpt"]) == "non-inferior"
        assert not fmt.includes_half(got["auc_t_gpt"])
        assert fmt.e2_reading(got["A"]) == "unresolved"
        assert fmt.e2_reading(got["A_threshold"]) == "unresolved"

    def test_the_treatments_own_auc_interval(
        self, band_l_intervals: dict[str, dict[str, Any]]
    ) -> None:
        """RESULTS: 0.711 [0.648, 0.773] under GPT-5.5, the product-parser reading."""
        got = band_l_intervals["product_parser"]["auc_t_gpt"]
        assert got == pytest.approx((0.648, 0.773), abs=0.001)

    @pytest.mark.parametrize(
        ("reading", "point", "ci"),
        [
            ("product_parser", -0.026, (-0.097, 0.053)),
            ("control_parser", -0.035, (-0.116, 0.052)),
        ],
    )
    def test_band_h_gpt_d_auc_recomputed_from_the_rows(
        self,
        transfer: dict[str, Any],
        bands: dict[str, list[Pair]],
        band_h_intervals: dict[str, dict[str, Any]],
        reading: str,
        point: float,
        ci: tuple[float, float],
    ) -> None:
        """Band H's GPT-5.5 Delta AUC under both readings, by the same registered bootstrap as
        band L's. Both points are negative, and both intervals span zero."""
        pt = _points(bands["H"], FIELDS[reading])
        got = band_h_intervals[reading]
        assert pt["d_auc_gpt"] == pytest.approx(point, abs=0.0005)
        assert got["d_auc_gpt"] == pytest.approx(ci, abs=0.0005)
        recorded = transfer["summary"]["bands"]["H"]["readings"][reading]
        assert pt["d_auc_gpt"] == pytest.approx(recorded["point"]["d_auc_gpt"], abs=1e-12)
        for key, interval in got.items():
            assert interval == pytest.approx(tuple(recorded["intervals"][key]), abs=1e-12), key
