"""Does the fine-scale rescore behave the same on Azure's gpt-4.1-mini?

Registered in `evals/PREREG-finescale-model-transfer.md`, committed before any gpt-4.1-mini score
or new second-judge verdict existed. Every rule below comes from that document, and where the two
could disagree the document wins and this file is the bug.

    uv run python evals/finescale_model_transfer.py --check    # blocking checks 1-4 and 6, $0
    uv run python evals/finescale_model_transfer.py --sonnet   # Labels steps 0-4, ~$2 of Sonnet
    uv run python evals/finescale_model_transfer.py --score    # the Azure passes
    uv run python evals/finescale_model_transfer.py --report   # the analysis, $0

The Azure resource is named by the environment, never by this file, because the resource is
private: RR_TRANSFER_AZURE_ENDPOINT, RR_TRANSFER_AZURE_RG and
RR_TRANSFER_AZURE_ACCOUNT. Nothing that identifies it is written into the artifact either.
"""

from __future__ import annotations

import argparse
import calendar
import copy
import hashlib
import json
import math
import os
import random
import re
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import band_testbeds as tb  # noqa: E402
import exp_finescale as ef  # noqa: E402

from anonymous import finescale  # noqa: E402

EVALS = Path(__file__).resolve().parent
RESULTS = EVALS / "results"
WORK = EVALS / ".work" / "exp" / "finescale_model_transfer"
PASS_DIRS = {"first": WORK / "azure", "retest": WORK / "azure_retest"}
OUT = EVALS / "finescale_model_transfer.json"
ERROR_LOG = WORK / "errors.jsonl"
RUN_LOCK = WORK / "run.lock"

SONNET = "claude-sonnet-5"
DEPLOYMENT = "gpt-4.1-mini"
ALLOWED_MODELS = {"gpt-4.1-mini-2025-04-14", "gpt-4.1-mini"}
REGISTERED_VERSION = "2025-04-14"

SEED = 20260921
DRAWS = 4000
E1_MARGIN = -0.05
E2_MARGIN = 0.08
VOID_LIMIT = 5  # band L voids above this make E1's Sonnet reading unresolved
ERROR_RETRIES = 5
RESUME_PASSES = 3
RATE_LIMIT_REFUSALS = 6
RATE_LIMIT_DEFAULT_WAIT = 30.0
RATE_LIMIT_MAX_WAIT = 600.0
RESUMABLE_LIMIT = 3  # a fourth resumable stop is final
RESUME_DELAY = 600.0
SONNET_STREAK = 3  # papers in a row failing at the API stop the purchase
P7_SURPRISE = 0.08


@dataclass(frozen=True)
class Band:
    name: str
    run: str
    control: str
    size: int
    cases: int
    control_auc: float
    gate: tuple[str, str]
    population_fp: str
    repo_fp: str
    prompt_fp: str
    label_fp: str
    sonnet_fp: str
    sonnet_count: int
    no_digit_stop: int
    transport_stop: int


BANDS = {
    "L": Band(
        name="L",
        run="judge-gpt-5.5-frozenpool-bigrams_verified-wemb1.5-20260908T163150Z.json",
        control="finescale_current_gate_luna.json",
        size=315,
        cases=35,
        control_auc=0.6749,
        gate=("openai", "gpt-5.6-luna"),
        population_fp="973a571d23708b631069264bb29636adb3e78ba5c68a25eebd121d8b4f9d03c7",
        repo_fp="798213b74f23f22d24f27d37202ab0cfa770e4ad6f440ed513e17e5cb2026209",
        prompt_fp="72336bb88a75c678dff5566c55bdc3011a385bb8df50da9b413b3e5c7a3d49fe",
        label_fp="70ad98d8656e8c02d5b3b78ea2c8d72ca9b87353941d5e8100a03fa6d4968898",
        sonnet_fp="5f5b198048fabe3159ff60cafc71ed54c8bf0048293ca3d9ed3bf2f7c2a49316",
        sonnet_count=212,
        no_digit_stop=16,
        transport_stop=16,
    ),
    "H": Band(
        name="H",
        run="judge-gpt-5.5-frozenpool-bigrams_verified-wemb1.5-20260908T063132Z.json",
        control="finescale_current_gate.json",
        size=328,
        cases=34,
        control_auc=0.7257,
        gate=("claude", "claude-haiku-4-5"),
        population_fp="698bd2caeee0a5803d00b63c9b76ab183f1fba1b872cae9cccff9163ca531222",
        repo_fp="d393f0e37c37b6083302af88e492af309121b4f64efb57dc844ccae4d9c02483",
        prompt_fp="ea26abef5a86b2355f0e2b99257128e99ef53f0e2bcc828f8439b1f3f53eb84d",
        label_fp="233a34d2b26e14f14b2e415f22248a90b15a5e309a86b9f7b8b8cdf3eee88509",
        sonnet_fp="03f6130115fffd311fdd6907e86ad0adca08c48192bebf8a88e217a60b7ab0d2",
        sonnet_count=284,
        no_digit_stop=17,
        transport_stop=17,
    ),
}


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ── Loading and fingerprints ──────────────────────────────────────────────────────────────


@dataclass
class BandPaper:
    case: str
    id: str  # unversioned
    vid: str  # the run file's versioned id
    title: str
    abstract: str
    judge: int
    prompt: str = ""


def load_band(band: Band) -> list[BandPaper]:
    """The band as `finescale_current_gate.load_band` loads it, pointed at the registered run."""
    import finescale_current_gate as fcg

    fcg.RUN = RESULTS / band.run
    papers = fcg.load_band()
    run = json.loads((RESULTS / band.run).read_text(encoding="utf-8"))
    shown = {e["case"]: (e.get("returned") or {}).get("anonymous_top10") or [] for e in run}
    out = [
        BandPaper(
            case=p.case,
            id=p.id,
            vid=str(shown[p.case][p.pos].get("arxiv_id") or ""),
            title=p.title,
            abstract=p.abstract,
            judge=p.judge,
        )
        for p in papers
    ]
    out.sort(key=lambda p: (p.case, p.id))
    blocks = repo_blocks(out)
    for p in out:
        p.prompt = ef.SCALE_PROMPT.format(
            repo=blocks[p.case], title=p.title, abstract=p.abstract[:1500]
        )
    return out


_BLOCKS: dict[str, str] = {}


def repo_blocks(papers: list[BandPaper]) -> dict[str, str]:
    for case in sorted({p.case for p in papers}):
        if case not in _BLOCKS:
            _BLOCKS[case] = tb.repo_block(case)
    return _BLOCKS


def population_fp(papers: list[BandPaper]) -> str:
    return sha("".join(f"{p.case}/{p.id}\n" for p in papers))


def repo_fp(papers: list[BandPaper]) -> str:
    blocks = repo_blocks(papers)
    return sha("".join(f"{c}\n{blocks[c]}\n" for c in sorted({p.case for p in papers})))


def prompt_fp(papers: list[BandPaper]) -> str:
    return sha("".join(f"{p.case}/{p.id}\n{p.prompt}\n" for p in papers))


def label_fp(papers: list[BandPaper]) -> str:
    return sha("".join(f"{p.case}/{p.id}/{p.judge}\n" for p in papers))


def sonnet_fp(papers: list[BandPaper], scores: dict[tuple[str, str], int]) -> str:
    """`case/id/score` for each band paper that has a verdict in *scores*, keyed (case, vid)."""
    return sha(
        "".join(
            f"{p.case}/{p.id}/{scores[(p.case, p.vid)]}\n"
            for p in papers
            if (p.case, p.vid) in scores
        )
    )


# ── Blocking checks 1-4 and 6 ─────────────────────────────────────────────────────────────


def resolved_gate(entry: dict[str, Any]) -> tuple[str, str]:
    """The gate a run used, resolved the way `run_judge_eval.py` resolved it."""
    rc = entry.get("ranking_config") or {}
    pc = entry.get("pool_config") or {}
    return str(rc.get("rr_gate_provider") or ""), str(
        rc.get("rr_gate_model") or pc.get("rr_triage_model") or ""
    )


def blocking_checks(loaded: dict[str, list[BandPaper]], cfg: Any) -> list[str]:
    failures: list[str] = []
    for name, band in BANDS.items():
        papers = loaded[name]
        control = json.loads((EVALS / band.control).read_text(encoding="utf-8"))
        rows = {(r["case"], r["id"]): r for r in control["rows"]}
        # 1. size, population, prompt and label fingerprints; labels equal the artifact's.
        if len(papers) != band.size:
            failures.append(f"1: band {name} loads {len(papers)} papers, registered {band.size}")
        for what, got, want in (
            ("population", population_fp(papers), band.population_fp),
            ("prompt", prompt_fp(papers), band.prompt_fp),
            ("label", label_fp(papers), band.label_fp),
        ):
            if got != want:
                failures.append(f"1: band {name} {what} fingerprint {got[:12]} != {want[:12]}")
        wrong = [p for p in papers if rows.get((p.case, p.id), {}).get("judge") != p.judge]
        if wrong:
            failures.append(f"1: band {name} has {len(wrong)} labels differing from its control")
        # 2. repository blocks.
        if repo_fp(papers) != band.repo_fp:
            failures.append(f"2: band {name} repository-block fingerprint differs")
        # 3. registered run, gate resolved from every entry.
        if control["summary"].get("run") != band.run:
            failures.append(f"3: band {name} control names run {control['summary'].get('run')}")
        run = json.loads((RESULTS / band.run).read_text(encoding="utf-8"))
        gates = {resolved_gate(e) for e in run}
        if gates != {band.gate}:
            failures.append(f"3: band {name} run resolves to gates {sorted(gates)}")
        recorded = control["summary"].get("gate")
        if name == "H" and (recorded or {}) != {"provider": band.gate[0], "model": band.gate[1]}:
            failures.append(f"3: band H summary.gate is {recorded}")
        # 4. control AUC from the artifact's exp09 and the loaded labels.
        scores = [rows[(p.case, p.id)]["exp09"] for p in papers]
        auc = tb.auc(scores, [p.judge >= tb.ACTIONABLE for p in papers])
        if round(auc, 4) != band.control_auc:
            failures.append(f"4: band {name} control AUC {auc:.4f} != {band.control_auc}")
    # 6. the treatment config sends the prompt unredacted.
    if list(getattr(cfg, "redact", []) or []):
        failures.append("6: the treatment config's redact list is not empty")
    return failures


# ── The artifact, which also holds the run's state ────────────────────────────────────────


def load_artifact() -> dict[str, Any]:
    if OUT.is_file():
        data: dict[str, Any] = json.loads(OUT.read_text(encoding="utf-8"))
        return data
    return {
        "registration": "evals/PREREG-finescale-model-transfer.md",
        "sonnet": {},
        "run": {"stops": [], "resumable_stops": 0, "segments": {}},
    }


_AZURE_HOST = re.compile(
    r"[a-z0-9-]+\.(?:openai\.azure\.com|services\.ai\.azure\.com|cognitiveservices\.azure\.com)",
    re.IGNORECASE,
)


def _resource_names() -> list[str]:
    raw = os.environ.get("RR_TRANSFER_AZURE_ENDPOINT", "").strip()
    host = (
        (urlparse(raw if "://" in raw else f"https://{raw}").hostname or "").lower() if raw else ""
    )
    names = {
        raw,
        host,
        host.split(".")[0] if host else "",
        os.environ.get("RR_TRANSFER_AZURE_RG", "").strip(),
        os.environ.get("RR_TRANSFER_AZURE_ACCOUNT", "").strip(),
    }
    return sorted((n for n in names if len(n) >= 3), key=len, reverse=True)


def scrub(text: str) -> str:
    """*text* with every name of the Azure resource replaced. The shipped transport puts the host
    in its messages (`Azure OpenAI (<host>)`), and the repository and artifact are public."""
    text = _AZURE_HOST.sub("<azure-host>", text)
    for name in _resource_names():
        text = re.sub(re.escape(name), "<azure-resource>", text, flags=re.IGNORECASE)
    return text


def save_artifact(art: dict[str, Any]) -> None:
    text = scrub(json.dumps(art, indent=1, sort_keys=True))
    low = text.lower()
    if _AZURE_HOST.search(text) or any(n.lower() in low for n in _resource_names()):
        raise RuntimeError("the artifact would name the Azure resource; nothing was written")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".json.tmp")
    tmp.write_text(text, encoding="utf-8")
    _replace(tmp, OUT)


def _replace(tmp: Path, target: Path) -> None:
    """`tmp.replace(target)`, retried while Windows refuses it.

    On Windows the rename fails with "Access is denied" while another process -- an editor's file
    watcher, the virus scanner -- has *target* open for a moment. The first --score run died on
    exactly that at paper 126, with every row before it safely cached."""
    for attempt in range(40):
        try:
            tmp.replace(target)
            return
        except PermissionError:
            time.sleep(0.25 * min(attempt + 1, 8))
    tmp.replace(target)


def write_atomic(path: Path, data: dict[str, Any]) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=1), encoding="utf-8")
    _replace(tmp, path)


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# ── Labels: the Sonnet steps ──────────────────────────────────────────────────────────────


def cached_sonnet(case: str, vid: str) -> int | None:
    from second_judge import second_cache_path

    path = second_cache_path(SONNET, case, vid)
    if not path.is_file():
        return None
    return int(json.loads(path.read_text(encoding="utf-8"))["score"])


def sonnet_stage(loaded: dict[str, list[BandPaper]], art: dict[str, Any]) -> list[str]:
    """Labels steps 0-4. Returns blocking-check-5 failures; buys verdicts on the first pass only."""
    from second_judge import second_verdict, verify_contexts

    s = art["sonnet"]
    failures: list[str] = []
    first_pass = "label_set" not in s

    # Step 0: the existing-verdict fingerprint, over the cache as it stands or the recorded list.
    if "step0" not in s:
        if not first_pass:  # pragma: no cover - the label set is never written before step 0
            return ["5: a label set exists without a step 0 list"]
        s["step0"] = {
            name: [[p.case, p.vid] for p in papers if cached_sonnet(p.case, p.vid) is not None]
            for name, papers in loaded.items()
        }
        save_artifact(art)  # written before the first purchase
    for name, band in BANDS.items():
        listed = {(c, v) for c, v in s["step0"][name]}
        scores = {}
        for case, vid in listed:
            score = cached_sonnet(case, vid)
            if score is not None:
                scores[(case, vid)] = score
        if len(listed) != band.sonnet_count or sonnet_fp(loaded[name], scores) != band.sonnet_fp:
            failures.append(f"5: band {name} step 0 does not reproduce the registered verdicts")
    if failures:
        return failures

    if first_pass:
        # Step 1: clone drift.
        cases = sorted({p.case for papers in loaded.values() for p in papers})
        contexts, drifted = verify_contexts(cases)
        s["drifted"] = sorted(drifted)
        # Steps 2 and 3: buy what is missing under the versioned id, with up to 3 more tries.
        from gate_swap_second_judge import pool_meta
        from run_judge_eval import load_dotenv

        load_dotenv(EVALS / ".env")
        meta = pool_meta()
        wanted = sorted(
            {
                (p.case, p.vid)
                for papers in loaded.values()
                for p in papers
                if p.case not in drifted and cached_sonnet(p.case, p.vid) is None
            }
        )
        print(f"Sonnet verdicts to buy: {len(wanted)}")
        from anonymous.llm_client import LLMError, LLMRateLimited, LLMUnavailable

        void: list[list[str]] = []
        streak = 0
        for i, (case, vid) in enumerate(wanted, 1):
            record = meta.get((case, vid))
            ok = False
            last: BaseException | None = None
            for attempt in range(4):
                if record is None:
                    break
                try:
                    second_verdict(case, contexts[case], {**record, "arxiv_id": vid}, SONNET)
                    ok = True
                    break
                except LLMUnavailable as exc:
                    # No key or a refused token: every paper would repeat it. Nothing is voided
                    # and step 4 is not written, so a restart is still the first pass.
                    raise SystemExit(
                        f"Sonnet unavailable ({scrub(str(exc))[:120]}); rerun --sonnet"
                    ) from None
                except LLMRateLimited as exc:
                    last = exc
                    time.sleep(rate_limit_wait(exc))
                except Exception as exc:  # noqa: BLE001 -- a failed purchase is retried, then void
                    last = exc
                    print(f"  ! {case}/{vid} attempt {attempt + 1}: {type(exc).__name__}")
                    time.sleep(2.0 * (attempt + 1))
            streak = streak + 1 if (not ok and isinstance(last, LLMError)) else 0
            if streak >= SONNET_STREAK:
                raise SystemExit(
                    f"{streak} papers in a row failed at the API; stopped before step 4 with "
                    "nothing voided. Rerun --sonnet to buy only what is missing."
                )
            if not ok:
                void.append([case, vid])
            if i % 20 == 0:
                print(f"  [{i}/{len(wanted)}]", flush=True)
        s["void"] = void
        # Step 4: fingerprint each band's full label set and record it.
        s["label_set"] = {}
        for name, papers in loaded.items():
            scores = {
                (p.case, p.vid): score
                for p in papers
                if p.case not in drifted and (score := cached_sonnet(p.case, p.vid)) is not None
            }
            s["label_set"][name] = {"fingerprint": sonnet_fp(papers, scores), "count": len(scores)}
        save_artifact(art)
    else:
        for name, papers in loaded.items():
            scores = {
                (p.case, p.vid): score
                for p in papers
                if p.case not in s["drifted"]
                and (score := cached_sonnet(p.case, p.vid)) is not None
            }
            if sonnet_fp(papers, scores) != s["label_set"][name]["fingerprint"]:
                failures.append(f"5: band {name} label set does not reproduce")
    return failures


def sonnet_labels(
    papers: list[BandPaper], art: dict[str, Any]
) -> dict[tuple[str, str], int | None]:
    """Sonnet score per (case, id); None when void, in a drifted case, or never judged."""
    drifted = set(art["sonnet"].get("drifted", []))
    void = {(c, v) for c, v in art["sonnet"].get("void", [])}
    out: dict[tuple[str, str], int | None] = {}
    for p in papers:
        excluded = p.case in drifted or (p.case, p.vid) in void
        out[(p.case, p.id)] = None if excluded else cached_sonnet(p.case, p.vid)
    return out


# ── The treatment: capture, classification, validity ─────────────────────────────────────


@dataclass
class Capture:
    body: dict[str, Any] | None = None
    response: dict[str, Any] | None = None
    returned: bool = False


_LOCAL = threading.local()


def install_capture() -> None:
    """Wrap `llm_client._post_adaptive` so each call's body and response are kept.

    The body is read in `finally`, after the transport has edited it in place, whether the call
    returned or raised. `_call_openai_top_logprobs` looks the function up in the module when it
    runs, so this reaches every request `top_logprobs` makes.
    """
    from anonymous import llm_client

    if getattr(llm_client._post_adaptive, "_transfer_capture", False):
        return
    original = llm_client._post_adaptive

    def wrapped(target: Any, body: dict[str, Any], timeout: int) -> dict[str, Any]:
        cap = Capture()
        calls = getattr(_LOCAL, "calls", None)
        if calls is not None:
            calls.append(cap)
        try:
            response = original(target, body, timeout)
            cap.response = response
            cap.returned = True
            return response
        finally:
            cap.body = copy.deepcopy(body)

    wrapped._transfer_capture = True  # type: ignore[attr-defined]
    llm_client._post_adaptive = wrapped


def request_valid(body: dict[str, Any] | None, prompt: str) -> tuple[bool, str]:
    """The registered request, with dropping `reasoning_effort` the only permitted change."""
    if body is None:
        return False, "no request body was captured"
    allowed = {"model", "messages", "temperature", "max_tokens", "logprobs", "top_logprobs"}
    extra = set(body) - allowed - {"reasoning_effort"}
    if extra:
        return False, f"unexpected keys {sorted(extra)}"
    if body.get("messages") != [{"role": "user", "content": prompt}]:
        return False, "the message is not the registered prompt"
    for key, want in (("temperature", 0), ("max_tokens", 4), ("logprobs", True)):
        if key not in body or body[key] != want or type(body[key]) is not type(want):
            return False, f"{key} is {body.get(key)!r}"
    if body.get("top_logprobs") != 20:
        return False, f"top_logprobs is {body.get('top_logprobs')!r}"
    if "reasoning_effort" in body and body["reasoning_effort"] != "none":
        return False, f"reasoning_effort is {body['reasoning_effort']!r}"
    return True, ""


def model_ok(response: dict[str, Any] | None) -> bool:
    name = str((response or {}).get("model") or "").lower()
    return name in ALLOWED_MODELS


def _first_choice(response: dict[str, Any] | None) -> dict[str, Any]:
    choices = (response or {}).get("choices") or []
    first = choices[0] if choices else {}
    return first if isinstance(first, dict) else {}


def classify(exc: BaseException | None, cap: Capture | None) -> str:
    """One call's outcome: answered, content_filtered, empty, rate_limited, unavailable_http,
    unavailable_token or error. Rate limits and LLMUnavailable are LLMError subclasses, so they
    are tested first."""
    from anonymous.llm_client import LLMError, LLMRateLimited, LLMUnavailable

    if exc is None:
        return "answered"
    if isinstance(exc, LLMRateLimited):
        return "rate_limited"
    if isinstance(exc, LLMUnavailable):
        cause = exc.__cause__
        return (
            "unavailable_http" if isinstance(cause, urllib.error.HTTPError) else "unavailable_token"
        )
    if isinstance(exc, LLMError) and "content filter" in str(exc):
        if cap is None or not cap.returned:
            return "content_filtered"  # the 400 shape: _post_adaptive raised
        if _first_choice(cap.response).get("finish_reason") == "content_filter":
            return "content_filtered"  # the 200 shape
    if (
        isinstance(exc, LLMError)
        and "returned no logprobs" in str(exc)
        and cap is not None
        and cap.returned
    ):
        choice = _first_choice(cap.response)
        text = (choice.get("message") or {}).get("content")
        if choice.get("finish_reason") != "content_filter" and not text:
            return "empty"
    return "error"


def _as_objects(content: list[dict[str, Any]]) -> list[SimpleNamespace]:
    """The raw logprobs content as the attribute objects `_digit_expectation` reads."""
    return [
        SimpleNamespace(
            token=str(tok.get("token", "")),
            logprob=tok.get("logprob"),
            top_logprobs=[
                SimpleNamespace(token=str(a.get("token", "")), logprob=a.get("logprob"))
                for a in tok.get("top_logprobs") or []
                if "logprob" in a
            ],
        )
        for tok in content
    ]


def readings(response: dict[str, Any] | None) -> dict[str, Any]:
    """Both parsers' exp09 for one answered response, plus what the row keeps."""
    choice = _first_choice(response)
    content = (choice.get("logprobs") or {}).get("content") or []
    first = (content[0].get("top_logprobs") or []) if content else []
    alternatives = [(a.get("token", ""), math.exp(a["logprob"])) for a in first if "logprob" in a]
    product = finescale.digit_expectation(alternatives) if alternatives else None
    control: float | None = None
    control_raised = False
    try:
        got = ef._digit_expectation(_as_objects(content))
        control = got[0] if got else None
    except ValueError:  # a digit isdigit() accepts and int() rejects, e.g. a subscript
        control_raised = True
    return {
        "product_exp": product,
        "control_exp": control,
        "control_parse_raised": control_raised,
        "first_token_alternatives": len(first),
        "tokens": [str(t.get("token", "")) for t in content],
        "alternatives": [
            [[str(a.get("token", "")), a.get("logprob")] for a in t.get("top_logprobs") or []]
            for t in content
        ],
        "finish_reason": choice.get("finish_reason"),
    }


# ── Stops ─────────────────────────────────────────────────────────────────────────────────


class Stop(Exception):
    """A registered stop. *resumable* stops are resumed; the rest end the run."""

    def __init__(self, kind: str, detail: str, *, resumable: bool = False) -> None:
        detail = scrub(detail)
        super().__init__(f"{kind}: {detail}")
        self.kind = kind
        self.detail = detail
        self.resumable = resumable


def rate_limit_wait(exc: BaseException) -> float:
    cause = exc.__cause__
    headers = getattr(cause, "headers", None)
    if headers is not None:
        for name, scale in (("retry-after-ms", 0.001), ("retry-after", 1.0)):
            raw = headers.get(name)
            if raw is not None:
                try:
                    return min(max(float(raw) * scale, 1.0), RATE_LIMIT_MAX_WAIT)
                except (TypeError, ValueError):
                    pass
    return RATE_LIMIT_DEFAULT_WAIT


# ── Scoring ───────────────────────────────────────────────────────────────────────────────


def cache_path(pass_name: str, paper: BandPaper) -> Path:
    from second_judge import safe_paper_id

    return PASS_DIRS[pass_name] / f"{paper.case}__{safe_paper_id(paper.id)}.json"


def treatment_cfg() -> Any:
    from anonymous.config import FinescaleConfig

    endpoint = os.environ.get("RR_TRANSFER_AZURE_ENDPOINT", "")
    return FinescaleConfig(
        enabled=True,
        provider="azure_openai",
        azure_deployment=DEPLOYMENT,
        reasoning_effort="none",
        azure_endpoint=endpoint,
        azure_tenant="",
        redact=[],
    )


def score_one(paper: BandPaper, cfg: Any, sleep: Any = time.sleep) -> dict[str, Any]:
    """Score one paper to a row, raising Stop for the registered stops.

    Rows are answered (scored or no digit), empty, content_filtered or error. Only error rows
    are left uncached by the caller.
    """
    from anonymous.llm_client import top_logprobs

    refusals = 0
    errors = 0
    while True:
        _LOCAL.calls = []
        exc: BaseException | None = None
        try:
            top_logprobs(paper.prompt, cfg, top_k=20)
        except Exception as caught:  # noqa: BLE001 -- every exception is classified
            exc = caught
        cap = _LOCAL.calls[-1] if _LOCAL.calls else None
        kind = classify(exc, cap)
        if kind == "rate_limited":
            refusals += 1
            if refusals >= RATE_LIMIT_REFUSALS:
                raise Stop("rate_limit", f"{paper.case}/{paper.id}", resumable=True)
            sleep(rate_limit_wait(exc))  # type: ignore[arg-type]
            continue
        refusals = 0
        if kind == "unavailable_http":
            raise Stop("llm_unavailable", f"HTTP {getattr(exc.__cause__, 'code', '?')}")
        if kind == "unavailable_token":
            # az's own message can carry a tenant id or a user name: keep only the category.
            raise Stop("llm_unavailable_token", "az token fetch failed", resumable=True)
        if kind == "error":
            errors += 1
            if errors > ERROR_RETRIES:
                return {
                    "case": paper.case,
                    "id": paper.id,
                    "state": "error",
                    "error_type": type(exc).__name__,
                    "error": scrub(str(exc))[:300],
                }
            sleep(min(2.0 * errors, 30.0))
            continue
        body = cap.body if cap else None
        valid, why = request_valid(body, paper.prompt)
        if not valid:
            raise Stop("request", f"{paper.case}/{paper.id}: {why}")
        response = cap.response if cap and cap.returned else None
        if response is not None and not model_ok(response):
            raise Stop("model_identity", f"{paper.case}/{paper.id}: {response.get('model')!r}")
        row: dict[str, Any] = {
            "case": paper.case,
            "id": paper.id,
            "vid": paper.vid,
            "state": kind,
            "response_model": (response or {}).get("model"),
            "reasoning_effort_dropped": body is not None and "reasoning_effort" not in body,
            "body_keys": sorted(body or {}),
            "message_sha256": sha(str(((body or {}).get("messages") or [{}])[0].get("content"))),
            # The capture, kept whole in the gitignored cache. The tracked artifact replaces the
            # prompt with its hash (see artifact_row); the prompt fingerprint already fixes it.
            "request_body": copy.deepcopy(body),
            "response": copy.deepcopy(response),
        }
        if kind == "answered":
            row.update(readings(response))
            if row["product_exp"] is None:
                row["state"] = "no_digit"
        elif kind == "empty":
            row.update(readings(response))
            row.update(product_exp=None, control_exp=None, state="empty")
        else:  # content_filtered
            row["filter_shape"] = 200 if cap and cap.returned else 400
            if not (cap and cap.returned):
                row["filter_message"] = scrub(str(exc))[:300]
        return row


@dataclass
class Segment:
    """One registered sweep: band L's first pass, its retest, or band H's remaining papers."""

    name: str
    pass_name: str
    papers: list[BandPaper]
    counted: list[BandPaper]  # the band whose stop counts this segment feeds
    no_digit_stop: int | None
    transport_stop: int | None
    rows: dict[tuple[str, str], dict[str, Any]] = field(default_factory=dict)


def _no_digit(row: dict[str, Any] | None) -> bool:
    return bool(row) and row.get("state") in ("no_digit", "empty")  # type: ignore[union-attr]


def run_segment(
    seg: Segment,
    cfg: Any,
    sleep: Any = time.sleep,
    progress: dict[str, Any] | None = None,
    save: Any = lambda: None,
) -> int:
    """Score a segment: pass 0 over every paper, then up to three resume passes over the papers
    still without a cached row. *progress* records the pass and the papers already asked in it,
    and is saved after every paper, so a resume or a restart continues the pass that stopped and
    adds no pass. Returns the transport count, taken after the resume passes."""
    PASS_DIRS[seg.pass_name].mkdir(parents=True, exist_ok=True)
    progress = progress if progress is not None else {"pass": 0, "asked": []}

    def counted_no_digit() -> int:
        n = 0
        for p in seg.counted:
            path = cache_path(seg.pass_name, p)
            if path.is_file() and _no_digit(json.loads(path.read_text(encoding="utf-8"))):
                n += 1
        return n

    asked = {tuple(k) for k in progress.get("asked", [])}
    while progress["pass"] <= RESUME_PASSES:
        k = progress["pass"]
        if k == 0:
            targets = seg.papers
        else:
            targets = [p for p in seg.papers if not cache_path(seg.pass_name, p).is_file()]
            if not targets:
                progress["pass"], progress["asked"] = RESUME_PASSES + 1, []
                save()
                break
        for i, paper in enumerate(targets, 1):
            path = cache_path(seg.pass_name, paper)
            if path.is_file():
                seg.rows[(paper.case, paper.id)] = json.loads(path.read_text(encoding="utf-8"))
                continue
            if (paper.case, paper.id) in asked:
                continue  # asked earlier in this pass and still in error
            row = score_one(paper, cfg, sleep)
            seg.rows[(paper.case, paper.id)] = row
            if row["state"] == "error":
                ERROR_LOG.parent.mkdir(parents=True, exist_ok=True)
                with ERROR_LOG.open("a", encoding="utf-8") as fh:
                    entry = {
                        **row,
                        "pass": seg.pass_name,
                        "segment": seg.name,
                        "resume": k,
                        "at": utc(),
                    }
                    fh.write(scrub(json.dumps(entry)) + "\n")
            else:
                write_atomic(path, row)
            asked.add((paper.case, paper.id))
            progress["asked"] = sorted([list(a) for a in asked])
            save()
            if seg.no_digit_stop is not None:
                n = counted_no_digit()
                if n >= seg.no_digit_stop:
                    raise Stop("no_digit", f"{n} papers gave no digit")
            if i % 25 == 0:
                print(f"  {seg.name} pass {k}: {i}/{len(targets)}", flush=True)
        progress["pass"], progress["asked"] = k + 1, []
        asked = set()
        save()
    # Counted over the whole band. Error rows are never cached, so a band paper with no cached
    # row is one still in error, including a paper band H shares with band L.
    errors = sum(1 for p in seg.counted if not cache_path(seg.pass_name, p).is_file())
    if seg.transport_stop is not None and errors >= seg.transport_stop:
        raise Stop("transport", f"{errors} papers still in error")
    return errors


def az_reading() -> dict[str, Any]:
    """The deployment's model, version and capacity, read with `az`. Records no names."""
    from anonymous import azure_auth

    az = azure_auth.az_executable()
    rg = os.environ.get("RR_TRANSFER_AZURE_RG", "")
    account = os.environ.get("RR_TRANSFER_AZURE_ACCOUNT", "")
    if not (az and rg and account):
        raise Stop("az", "az, RR_TRANSFER_AZURE_RG or RR_TRANSFER_AZURE_ACCOUNT is missing")
    try:
        done = azure_auth._run(
            [
                az,
                "cognitiveservices",
                "account",
                "deployment",
                "show",
                "-g",
                rg,
                "-n",
                account,
                "--deployment-name",
                DEPLOYMENT,
                "--output",
                "json",
            ],
            timeout=120,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        # The exception's text holds the command line, which names the resource.
        raise Stop("az", f"az deployment show failed ({type(exc).__name__})") from None
    if done.returncode != 0:
        # az's stderr can name the account, a tenant or a user: report only the exit code.
        raise Stop("az", f"az deployment show exited {done.returncode}")
    data = json.loads(done.stdout)
    props = data.get("properties") or {}
    model = props.get("model") or {}
    sku = data.get("sku") or {}
    return {
        "model": model.get("name"),
        "version": model.get("version"),
        "sku": sku.get("name"),
        "capacity": sku.get("capacity"),
        "version_upgrade_option": props.get("versionUpgradeOption"),
        "read_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def resume_wait(art: dict[str, Any], sleep: Any = time.sleep) -> None:
    """Wait out the 10 minutes since the last resumable stop, whether or not this process was the
    one that stopped, and record when the run resumed."""
    run = art["run"]
    last = next(
        (s for s in reversed(run["stops"]) if s.get("resumed") and "resumed_at" not in s), None
    )
    if last is None:
        return
    stopped = calendar.timegm(time.strptime(last["at"], "%Y-%m-%dT%H:%M:%SZ"))
    remaining = stopped + RESUME_DELAY + 1 - time.time()
    if remaining > 0:
        print(f"resuming in {remaining:.0f} s", flush=True)
        sleep(remaining)
    last["resumed_at"] = utc()
    save_artifact(art)


def finished(run: dict[str, Any]) -> str | None:
    """Why the run can take no more requests, or None while it can."""
    if run.get("ended"):
        return str(run["ended"])
    if run.get("ended_after_band_l"):
        return f"after band L: {run['ended_after_band_l']}"
    if all(run["segments"].get(n) == "complete" for n in ("L-first", "L-retest", "H-rest")):
        return "complete"
    return None


def score_stage(
    loaded: dict[str, list[BandPaper]], art: dict[str, Any], sleep: Any = time.sleep
) -> None:
    """Band L's first pass and resume passes, then its retest, then band H's other papers."""
    from anonymous import azure_auth

    run = art["run"]
    if finished(run):
        print(f"the run already finished ({finished(run)})", file=sys.stderr)
        return
    resume_wait(art, sleep)
    cfg = treatment_cfg()
    install_capture()
    reading = az_reading()  # raises Stop("az") before any call if it cannot be read
    if "az" not in run:
        run["az"] = reading
        save_artifact(art)
        if reading["version"] != REGISTERED_VERSION or reading["model"] != DEPLOYMENT:
            run["stops"].append({"kind": "az_version", "band_l_scope": True, "at": utc()})
            run["ended"] = "X"
            save_artifact(art)
            return
    else:
        run.setdefault("az_rereads", []).append(reading)
        save_artifact(art)
    band_l = loaded["L"]
    l_ids = {(p.case, p.id) for p in band_l}
    band_h_rest = [p for p in loaded["H"] if (p.case, p.id) not in l_ids]
    segments = [
        Segment(
            "L-first", "first", band_l, band_l, BANDS["L"].no_digit_stop, BANDS["L"].transport_stop
        ),
        Segment("L-retest", "retest", band_l, band_l, None, None),
        Segment(
            "H-rest",
            "first",
            band_h_rest,
            loaded["H"],
            BANDS["H"].no_digit_stop,
            BANDS["H"].transport_stop,
        ),
    ]
    for seg in segments:
        if run["segments"].get(seg.name) == "complete":
            continue
        band_l_scope = seg.name == "L-first"
        progress = run.setdefault("progress", {}).setdefault(seg.name, {"pass": 0, "asked": []})
        while True:
            try:
                errors = run_segment(seg, cfg, sleep, progress, lambda: save_artifact(art))
                run.setdefault("transport_counts", {})[seg.name] = errors
                run["segments"][seg.name] = "complete"
                save_artifact(art)
                break
            except Stop as stop:
                entry = {"kind": stop.kind, "detail": stop.detail, "segment": seg.name, "at": utc()}
                if stop.resumable and run["resumable_stops"] < RESUMABLE_LIMIT:
                    run["resumable_stops"] += 1
                    entry["resumed"] = True
                    run["stops"].append(entry)
                    save_artifact(art)
                    print(
                        f"resumable stop ({stop}); resuming after {RESUME_DELAY:.0f} s", flush=True
                    )
                    resume_wait(art, sleep)
                    azure_auth.clear_cache()
                    continue
                entry["band_l_scope"] = band_l_scope
                run["stops"].append(entry)
                if band_l_scope:
                    run["ended"] = "S" if stop.kind == "no_digit" else "X"
                else:
                    run["segments"][seg.name] = "incomplete"
                    run["ended_after_band_l"] = stop.kind
                save_artifact(art)
                print(f"stop: {stop}", flush=True)
                return


# ── Analysis ──────────────────────────────────────────────────────────────────────────────


@dataclass
class Item:
    case: str
    id: str
    gpt: int
    son: int | None  # None: void, drifted or never judged
    ctrl: float
    state: str  # answered / no_digit / empty / content_filtered / error / missing
    t_exp: dict[str, float | None]  # per reading


READINGS = ("control_parser", "product_parser")


def items_for(
    papers: list[BandPaper],
    control_rows: dict[tuple[str, str], dict[str, Any]],
    rows: dict[tuple[str, str], dict[str, Any]],
    son: dict[tuple[str, str], int | None],
    errored: frozenset[tuple[str, str]] | set[tuple[str, str]] = frozenset(),
) -> list[Item]:
    out = []
    for p in papers:
        # A cached row wins: a paper that errored once and was scored on a later pass is scored.
        state = "error" if (p.case, p.id) in errored else "missing"
        row = rows.get((p.case, p.id)) or {"state": state}
        out.append(
            Item(
                case=p.case,
                id=p.id,
                gpt=p.judge,
                son=son.get((p.case, p.id)),
                ctrl=float(control_rows[(p.case, p.id)]["exp09"]),
                state=row.get("state", "missing"),
                t_exp={
                    "control_parser": row.get("control_exp"),
                    "product_parser": row.get("product_exp"),
                },
            )
        )
    return out


def admitted(exp: float | None) -> bool:
    return exp is not None and finescale.probability(exp) >= finescale.SHOW_THRESHOLD


def interval(values: list[float]) -> tuple[float, float] | None:
    k = len(values)
    if not k:
        return None
    v = sorted(values)
    return v[int(0.025 * k)], v[int(0.975 * k) - 1]


def draws_for(cases: list[str]) -> list[list[str]]:
    rng = random.Random(SEED)
    return [[rng.choice(cases) for _ in cases] for _ in range(DRAWS)]


def _label(item: Item, judge: str) -> bool | None:
    if judge == "gpt":
        return item.gpt >= tb.ACTIONABLE
    return None if item.son is None else item.son >= tb.ACTIONABLE


def _band_net(items: list[Item], adm: list[bool], judge: str) -> int:
    total = 0
    for it, a in zip(items, adm, strict=True):
        lab = _label(it, judge)
        if a and lab is not None:
            total += 1 if lab else -2
    return total


def reading_stats(items: list[Item], cases: list[str], reading: str) -> dict[str, Any]:
    """Every registered statistic for one reading of one band, on the band's 4,000 draws."""
    kept = [it for it in items if it.state not in ("error", "missing")]
    by_case: dict[str, list[Item]] = {c: [] for c in cases}
    for it in kept:
        by_case[it.case].append(it)
    fallback = {
        c: not finescale.enough_scored(
            sum(1 for it in group if it.t_exp[reading] is not None), len(group), 0.5
        )
        for c, group in by_case.items()
        if group
    }

    def adm_t(it: Item) -> bool:
        return True if fallback.get(it.case) else admitted(it.t_exp[reading])

    def in_threshold_part(it: Item) -> bool:
        return it.t_exp[reading] is not None and not fallback.get(it.case)

    def one(pool: list[Item]) -> dict[str, Any]:
        scored = [it for it in pool if it.t_exp[reading] is not None]
        res: dict[str, Any] = {}
        for judge in ("gpt", "son"):
            lab = [(it, _label(it, judge)) for it in scored]
            lab = [(it, y) for it, y in lab if y is not None]
            ys = [y for _, y in lab]
            if 0 < sum(ys) < len(ys):
                t = tb.auc([it.t_exp[reading] for it, _ in lab], ys)  # type: ignore[misc]
                c = tb.auc([it.ctrl for it, _ in lab], ys)
                res[f"auc_t_{judge}"], res[f"auc_c_{judge}"] = t, c
                res[f"d_auc_{judge}"] = t - c
        n = len(pool)
        if n:
            at = [adm_t(it) for it in pool]
            ac = [admitted(it.ctrl) for it in pool]
            res["A"] = (sum(at) - sum(ac)) / n
            thr = [i for i, it in enumerate(pool) if in_threshold_part(it)]
            res["A_threshold"] = sum(at[i] - ac[i] for i in thr) / n
            res["A_failure"] = res["A"] - res["A_threshold"]
        return res

    point = one(kept)
    stats: dict[str, list[float]] = {}
    e3: dict[str, list[float]] = {"gpt": [], "son": []}
    e4: dict[str, dict[str, list[float]]] = {
        j: {"t_minus_show_all": [], "t_minus_show_none": []} for j in ("gpt", "son")
    }
    for draw in draws_for(cases):
        pool = [it for c in draw for it in by_case[c]]
        for key, value in one(pool).items():
            stats.setdefault(key, []).append(value)
        for judge in ("gpt", "son"):
            diffs, all_diffs, none_diffs = [], [], []
            for c in draw:
                group = by_case[c]
                nt = _band_net(group, [adm_t(it) for it in group], judge)
                nc = _band_net(group, [admitted(it.ctrl) for it in group], judge)
                na = _band_net(group, [True] * len(group), judge)
                diffs.append(nt - nc)
                all_diffs.append(nt - na)
                none_diffs.append(nt)
            e3[judge].append(sum(diffs) / len(draw))
            e4[judge]["t_minus_show_all"].append(sum(all_diffs) / len(draw))
            e4[judge]["t_minus_show_none"].append(sum(none_diffs) / len(draw))

    out: dict[str, Any] = {
        "point": point,
        "intervals": {k: interval(v) for k, v in stats.items()},
        "skipped_draws": {k: DRAWS - len(v) for k, v in stats.items()},
        "e3_interval": {j: interval(v) for j, v in e3.items()},
        "e4_interval": {j: {k: interval(v) for k, v in d.items()} for j, d in e4.items()},
        "fallback_cases": sorted(c for c, f in fallback.items() if f),
        "counts": {
            "kept": len(kept),
            "errors_dropped": sum(1 for it in items if it.state == "error"),
            "missing": sum(1 for it in items if it.state == "missing"),
            "scored": sum(1 for it in kept if it.t_exp[reading] is not None),
            "no_digit": sum(1 for it in kept if it.state in ("no_digit", "empty")),
            "content_filtered": sum(1 for it in kept if it.state == "content_filtered"),
        },
    }
    totals = {}
    for judge in ("gpt", "son"):
        totals[judge] = {
            "treatment": _band_net(kept, [adm_t(it) for it in kept], judge),
            "control": _band_net(kept, [admitted(it.ctrl) for it in kept], judge),
            "show_all": _band_net(kept, [True] * len(kept), judge),
            "show_none": 0,
        }
    out["band_net_totals"] = totals
    n_cases = len(cases)  # the divisor the bootstrap uses, so points and intervals share units
    out["e3_point_per_case"] = {
        j: (totals[j]["treatment"] - totals[j]["control"]) / n_cases for j in totals
    }
    out["e4_point_per_case"] = {
        j: {
            "t_minus_show_all": (totals[j]["treatment"] - totals[j]["show_all"]) / n_cases,
            "t_minus_show_none": totals[j]["treatment"] / n_cases,
        }
        for j in totals
    }
    shifts = [it.t_exp[reading] - it.ctrl for it in kept if it.t_exp[reading] is not None]  # type: ignore[operator]
    if len(shifts) >= 2:
        q1, med, q3 = statistics.quantiles(shifts, n=4, method="inclusive")
        out["exp09_shift"] = {
            "n": len(shifts),
            "mean": statistics.fmean(shifts),
            "median": med,
            "q1": q1,
            "q3": q3,
        }

    def part(select: Any) -> dict[str, Any]:
        chosen = [it for it in kept if select(it)]
        res: dict[str, Any] = {"count": len(chosen)}
        for j in ("gpt", "son"):
            ys = [y for y in (_label(it, j) for it in chosen) if y is not None]
            res[f"{j}_n"] = len(ys)
            res[f"{j}_actionable_rate"] = sum(ys) / len(ys) if ys else None
        return res

    out["e3_split"] = {
        "only_treatment": part(lambda it: adm_t(it) and not admitted(it.ctrl)),
        "only_control": part(lambda it: admitted(it.ctrl) and not adm_t(it)),
    }
    out["admissions"] = {
        "treatment": sum(adm_t(it) for it in kept),
        "control": sum(admitted(it.ctrl) for it in kept),
        "only_treatment": sum(1 for it in kept if adm_t(it) and not admitted(it.ctrl)),
        "only_control": sum(1 for it in kept if admitted(it.ctrl) and not adm_t(it)),
    }
    return out


def judge_reading(ci: tuple[float, float] | None, forced_unresolved: bool = False) -> str:
    if ci is None or forced_unresolved:
        return "unresolved"
    lo, hi = ci
    if lo >= E1_MARGIN:
        return "non-inferior"
    if hi < E1_MARGIN:
        return "worse"
    return "unresolved"


def e1_reading(gpt: str, son: str) -> str:
    if "worse" in (gpt, son):
        return "worse"
    if gpt == son == "non-inferior":
        return "non-inferior"
    if "non-inferior" in (gpt, son):
        return "split"
    return "unresolved"


def e2_reading(ci: tuple[float, float] | None) -> str:
    if ci is None:
        return "unresolved"
    lo, hi = ci
    if lo >= -E2_MARGIN and hi <= E2_MARGIN:
        return "holds"
    if lo > E2_MARGIN:
        return "over-admits"
    if hi < -E2_MARGIN:
        return "under-admits"
    return "unresolved"


def includes_half(ci: tuple[float, float] | None) -> bool:
    return ci is None or ci[0] <= 0.5 <= ci[1]


def outcome_for(stats: dict[str, Any], sonnet_forced: bool) -> dict[str, Any]:
    """Rows W1, W2, U, P, O tested in order; the first match is this reading's outcome."""
    ci = stats["intervals"]
    gpt = judge_reading(ci.get("d_auc_gpt"))
    son = judge_reading(ci.get("d_auc_son"), forced_unresolved=sonnet_forced)
    e1 = e1_reading(gpt, son)
    floor = includes_half(ci.get("auc_t_gpt"))
    control_excludes_half = not includes_half(ci.get("auc_c_gpt"))
    e2_total = e2_reading(ci.get("A"))
    e2_thr = e2_reading(ci.get("A_threshold"))
    e2 = e2_total if e2_total == e2_thr else "unresolved"
    if e1 == "worse":
        row = "W1"
    elif floor and control_excludes_half:
        row = "W2"
    elif (
        e1 in ("split", "unresolved")
        or (e1 == "non-inferior" and e2 == "unresolved")
        or (floor and not control_excludes_half)
    ):
        row = "U"
    elif e2 == "holds":
        row = "P"
    else:
        row = "O-over" if e2 == "over-admits" else "O-under"
    return {
        "e1_gpt": gpt,
        "e1_sonnet": son,
        "e1": e1,
        "floor": floor,
        "control_excludes_half": control_excludes_half,
        "e2": e2,
        "e2_total": e2_total,
        "e2_threshold_part": e2_thr,
        "row": row,
    }


def combine(readings_rows: dict[str, str]) -> str:
    rows = set(readings_rows.values())
    return rows.pop() if len(rows) == 1 else "U"


def spearman(xs: list[float], ys: list[float]) -> float | None:
    def ranks(v: list[float]) -> list[float]:
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2
            i = j + 1
        return r

    if len(xs) < 3:
        return None
    rx, ry = ranks(xs), ranks(ys)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry, strict=True))
    vx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    vy = math.sqrt(sum((b - my) ** 2 for b in ry))
    return cov / (vx * vy) if vx and vy else None


def load_rows(pass_name: str, papers: list[BandPaper]) -> dict[tuple[str, str], dict[str, Any]]:
    out = {}
    for p in papers:
        path = cache_path(pass_name, p)
        if path.is_file():
            out[(p.case, p.id)] = json.loads(path.read_text(encoding="utf-8"))
    return out


def report_stage(loaded: dict[str, list[BandPaper]], art: dict[str, Any]) -> dict[str, Any]:
    run = art["run"]
    segs = run.get("segments", {})
    if run.get("ended") in ("X", "S"):
        # No endpoint is read from a partial run. S's wording needs only its two counts.
        out: dict[str, Any] = {"outcome": run["ended"], "stops": run["stops"]}
        if run["ended"] == "S":
            states = [r.get("state") for r in load_rows("first", loaded["L"]).values()]
            out["s_wording"] = {
                "no_digit": sum(s in ("no_digit", "empty") for s in states),
                "answered": sum(s in ("answered", "no_digit", "empty") for s in states),
            }
        return out
    if segs.get("L-first") != "complete":
        raise SystemExit("band L's first pass is not complete; resume --score before --report")
    summary: dict[str, Any] = {"bands": {}, "stops": run["stops"]}
    if run.get("ended_after_band_l"):
        summary["ended_after_band_l"] = run["ended_after_band_l"]
    errored: set[tuple[str, str]] = set()
    if ERROR_LOG.is_file():
        for line in ERROR_LOG.read_text(encoding="utf-8").splitlines():
            if line.strip():
                e = json.loads(line)
                if e.get("pass") == "first":  # band L's first pass and H-rest share this cache
                    errored.add((e["case"], e["id"]))
    for name, band in BANDS.items():
        papers = loaded[name]
        control = json.loads((EVALS / band.control).read_text(encoding="utf-8"))
        control_rows = {(r["case"], r["id"]): r for r in control["rows"]}
        rows = load_rows("first", papers)
        son = sonnet_labels(papers, art)
        items = items_for(papers, control_rows, rows, son, errored)
        cases = sorted({p.case for p in papers})
        void = [
            v
            for v in art["sonnet"].get("void", [])
            if (v[0], v[1]) in {(p.case, p.vid) for p in papers}
        ]
        band_out: dict[str, Any] = {
            "complete": name == "L" or segs.get("H-rest") == "complete",
            "still_in_error": sorted(f"{it.case}/{it.id}" for it in items if it.state == "error"),
            "void": len(void),
            "drifted": art["sonnet"].get("drifted", []),
            "readings": {},
        }
        for reading in READINGS:
            stats = reading_stats(items, cases, reading)
            if name == "L":
                stats["outcome"] = outcome_for(stats, sonnet_forced=len(void) > VOID_LIMIT)
            band_out["readings"][reading] = stats
        both = [it for it in items if it.t_exp["product_parser"] is not None]
        band_out["between_arm_spearman"] = spearman(
            [it.t_exp["product_parser"] for it in both],  # type: ignore[misc]
            [it.ctrl for it in both],
        )
        band_out["parser_disagreements"] = sum(
            1
            for it in items
            if (it.t_exp["control_parser"] is None) != (it.t_exp["product_parser"] is None)
            or (
                it.t_exp["control_parser"] is not None
                and it.t_exp["product_parser"] is not None
                and abs(it.t_exp["control_parser"] - it.t_exp["product_parser"]) > 0.01
            )
        )
        band_out["control_parse_raised"] = sum(
            1 for r in rows.values() if r.get("control_parse_raised")
        )
        band_out["reasoning_effort_dropped"] = sum(
            1 for r in rows.values() if r.get("reasoning_effort_dropped")
        )
        band_out["response_models"] = sorted({str(r.get("response_model")) for r in rows.values()})
        # Void sensitivity for Sonnet Delta AUC, no bar.
        if void:
            sens = {}
            void_keys = {(p.case, p.id) for p in papers if [p.case, p.vid] in void}
            for value, label in ((3, "void_actionable"), (0, "void_not_actionable")):
                filled = {k: (value if k in void_keys else v) for k, v in son.items()}
                alt = items_for(papers, control_rows, rows, filled, errored)
                sens[label] = reading_stats(alt, cases, "product_parser")["intervals"].get(
                    "d_auc_son"
                )
            band_out["void_sensitivity"] = sens
        summary["bands"][name] = band_out

    # Retest: band L's own noise, against gpt-4o-mini's replicate, under both parsers.
    first = load_rows("first", loaded["L"])
    retest = load_rows("retest", loaded["L"])
    summary["retest"] = {
        "complete": segs.get("L-retest") == "complete",
        "no_digit": sum(1 for r in retest.values() if r.get("state") in ("no_digit", "empty")),
        "content_filtered": sum(1 for r in retest.values() if r.get("state") == "content_filtered"),
        "reference": {"mean_abs_diff": 0.071, "spearman": 0.993, "admission_flips": "4/165"},
    }
    for key in ("control_exp", "product_exp"):
        pairs = [
            (first[k][key], retest[k][key])
            for k in first
            if k in retest and first[k].get(key) is not None and retest[k].get(key) is not None
        ]
        if pairs:
            summary["retest"][key] = {
                "n": len(pairs),
                "mean_abs_diff": sum(abs(a - b) for a, b in pairs) / len(pairs),
                "spearman": spearman([a for a, _ in pairs], [b for _, b in pairs]),
                "admission_flips": sum(1 for a, b in pairs if admitted(a) != admitted(b)),
            }

    # The registered outcome, from band L.
    per = {r: summary["bands"]["L"]["readings"][r]["outcome"]["row"] for r in READINGS}
    summary["outcome_per_reading"] = per
    summary["outcome"] = combine(per)
    gpt_pt = summary["bands"]["L"]["readings"]["product_parser"]["point"].get("d_auc_gpt")
    son_pt = summary["bands"]["L"]["readings"]["product_parser"]["point"].get("d_auc_son")
    if gpt_pt is not None and gpt_pt > P7_SURPRISE:
        confirmed = son_pt is not None and son_pt * gpt_pt > 0 and abs(son_pt) >= gpt_pt / 2
        summary["p7_surprise"] = {"gpt": gpt_pt, "sonnet": son_pt, "confirmed": confirmed}
    return summary


def hold_run_lock() -> Any:
    """An OS lock the operating system releases when the process dies: one run at a time."""
    RUN_LOCK.parent.mkdir(parents=True, exist_ok=True)
    fh = open(RUN_LOCK, "a+")  # noqa: SIM115 -- held for the life of the process
    try:
        if os.name == "nt":
            import msvcrt

            fh.seek(0)
            msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        fh.close()
        raise SystemExit(f"{RUN_LOCK} is held by a live run; the lock ends when it exits") from None
    return fh


def artifact_row(row: dict[str, Any]) -> dict[str, Any]:
    """A cached row as the tracked artifact keeps it: the prompt replaced by its hash, and the
    response without the logprobs content already stored as tokens and alternatives."""
    out = copy.deepcopy(row)
    body = out.get("request_body")
    if isinstance(body, dict):
        for m in body.get("messages") or []:
            m["content"] = sha(str(m.get("content")))
    for choice in (out.get("response") or {}).get("choices") or []:
        if isinstance(choice, dict) and isinstance(choice.get("logprobs"), dict):
            choice["logprobs"] = {"content": "stored as tokens and alternatives"}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--check", action="store_true", help="blocking checks 1-4 and 6, no spend")
    ap.add_argument("--sonnet", action="store_true", help="Labels steps 0-4")
    ap.add_argument("--score", action="store_true", help="the Azure passes")
    ap.add_argument("--report", action="store_true", help="the analysis")
    args = ap.parse_args()
    _lock = hold_run_lock()  # noqa: F841 -- held until exit

    if args.score:
        from anonymous import azure_auth

        try:
            azure_auth.chat_completions_url(os.environ.get("RR_TRANSFER_AZURE_ENDPOINT", ""))
        except azure_auth.AzureAuthError:
            print("RR_TRANSFER_AZURE_ENDPOINT must be the resource URL", file=sys.stderr)
            return 2

    loaded = {name: load_band(band) for name, band in BANDS.items()}
    cfg = treatment_cfg()
    failures = blocking_checks(loaded, cfg)
    for f in failures:
        print(f"BLOCKING CHECK FAILED {f}")
    if failures:
        if not args.check:
            art = load_artifact()
            if "az" not in art["run"]:  # no gpt-4.1-mini call can have happened yet
                art["run"]["ended"] = "X"
                art["run"]["stops"].append(
                    {"kind": "blocking_check", "detail": failures, "at": utc()}
                )
            else:
                art["run"].setdefault("later_check_failures", []).append(
                    {"detail": failures, "at": utc()}
                )
            save_artifact(art)
        return 2
    print("blocking checks 1-4 and 6 pass")
    if args.check:
        return 0

    art = load_artifact()
    if args.report and not (args.sonnet or args.score) and "label_set" not in art["sonnet"]:
        print("no recorded Sonnet label set; run --sonnet first", file=sys.stderr)
        return 2
    if args.sonnet or args.score or args.report:
        failures = sonnet_stage(loaded, art)  # a later pass re-checks the label set, buys nothing
        if failures:
            for f in failures:
                print(f"BLOCKING CHECK FAILED {f}")
            if "az" not in art["run"]:
                art["run"]["ended"] = "X"
                art["run"]["stops"].append(
                    {"kind": "blocking_check", "detail": failures, "at": utc()}
                )
            else:
                art["run"].setdefault("later_check_failures", []).append(
                    {"detail": failures, "at": utc()}
                )
            save_artifact(art)
            return 2
        print(
            "Sonnet label sets: "
            + ", ".join(f"{k} {v['count']}" for k, v in art["sonnet"]["label_set"].items())
            + f"; void {len(art['sonnet'].get('void', []))}"
        )
    if args.score:
        if finished(art["run"]):
            print(f"the run already finished ({finished(art['run'])})", file=sys.stderr)
            return 2
        score_stage(loaded, art)
    if args.report:
        art["summary"] = report_stage(loaded, art)
        everyone = [p for ps in loaded.values() for p in ps]
        art["rows"] = {
            pass_name: {
                f"{k[0]}/{k[1]}": artifact_row(v) for k, v in load_rows(pass_name, everyone).items()
            }
            for pass_name in PASS_DIRS
        }
        if ERROR_LOG.is_file():
            entries = [
                json.loads(line)
                for line in ERROR_LOG.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            for e in entries:
                p = next((q for q in everyone if q.case == e["case"] and q.id == e["id"]), None)
                e["final"] = p is not None and not cache_path(e["pass"], p).is_file()
            art["errors"] = entries
        save_artifact(art)
        print(json.dumps(art["summary"].get("outcome"), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
