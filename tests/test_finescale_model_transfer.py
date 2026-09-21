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
from typing import Any

import finescale_model_transfer as fmt
import pytest

from reporadar import azure_auth, llm_client
from reporadar.llm_client import LLMError, LLMRateLimited, LLMUnavailable

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
