"""Did the identifier line move the second judge's verdicts on the baseline's picks?

Pre-registered in `evals/PREREG-sonnet-id-probe.md`, committed before any verdict in this
probe existed. That file is the specification: where a comment here and the registration
disagree, the registration wins. This script only executes it.

NR-52 scored two arms under two judges and found that the sign of the margin depends on the
judge. The judges saw the same rubric but not the same papers. Every prompt carries an
identifier line. The first judge saw the baseline's arXiv picks under the versioned id the
resolver returned (`2404.14989v1`). The second judge saw the unversioned pick id
(`2404.14989`), while the other arm carried versioned ids under both. So under the second
judge, and only there, a version suffix marked which arm a paper came from. This measures
whether that one line moved the verdicts, and how far it moved the second judge's margin.

Two arms re-judge the baseline's 237 arXiv picks with the same model:

    V   arXiv: <versioned id>   what the first judge was shown
    U   arXiv: <pick id>        NR-52's recipe, drawn afresh

Title, abstract, repository context, rubric and call are shared byte for byte. `--plan` checks
on every prompt that the identifier line is the only difference before anything is bought.

    uv run python evals/sonnet_id_probe.py --plan     # no judge call: population, drift, ids
    uv run python evals/sonnet_id_probe.py --judge    # ~$8 of Sonnet, 474 verdicts, resumable
    uv run python evals/sonnet_id_probe.py --report   # no call: E1-E4 and the reading
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import statistics as st
import sys
import time
import urllib.error
from collections.abc import Callable, Iterable, Iterator, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

EVALS = Path(__file__).resolve().parent
sys.path.insert(0, str(EVALS))
sys.path.insert(0, str(EVALS.parent / "src"))

import judge as judge_mod  # noqa: E402
from bigram_report import BOOTSTRAP_N, BOOTSTRAP_SEED, paired_bootstrap  # noqa: E402
from rung1_second_judge import FROZEN as NR52_FROZEN  # noqa: E402
from rung1_second_judge import (  # noqa: E402
    LABELS,
    cached_sonnet,
    net2,
    opus5_arm,
    shipped_arm,
    son_of,
)
from second_judge import (  # noqa: E402
    ACTIONABLE,
    DEFAULT_MODEL,
    GOLD,
    safe_paper_id,
    second_cache_path,
    verify_contexts,
)

from reporadar import llm_client  # noqa: E402
from reporadar.credentials import resolve_api_key  # noqa: E402
from reporadar.paper_id import dedup_id, is_arxiv_id  # noqa: E402

PREREG = EVALS / "PREREG-sonnet-id-probe.md"
FROZEN = EVALS / "sonnet_id_probe.json"
STATE = EVALS / ".work" / "sonnet_id_probe"
PAPERS = STATE / "papers.json"
LEDGER = STATE / "ledger.json"

# Outside the gold cache and outside NR-52's namespace. NR-52's verdicts are read, never
# rewritten, and a fresh U draw must not be answered from NR-52's cache.
ARMS = ("v", "u")
NAMESPACE = {"v": f"{DEFAULT_MODEL}#id-versioned", "u": f"{DEFAULT_MODEL}#id-unversioned"}

# The registered population. A different count is a different experiment.
EXPECTED_CASES = 37
EXPECTED_JUDGED = 357
EXPECTED_POPULATION = 237

# The one pick with two versioned files in the gold cache. The registration names it and names
# the file used, so any other pick with several versions is refused, not silently resolved.
MULTI_VERSION = {("db", "2607.11271"): "2607.11271v3"}

# E1's interval, and the shift's in E2 and E4. The registration fixes all of it: 10,000 draws,
# seed 20260922, every case resampled with replacement, and the 95% interval read off the
# sorted draws at index 250 and index 9,750, counting from 0. `case_bootstrap` computes those
# two indices in integer arithmetic, so no float product decides which draw is read.
BOOT_DRAWS = 10_000
BOOT_SEED = 20260922
BOOT_LO_PER_MILLE = 25  # sorted draw index = draws * 25 // 1000 = 250
BOOT_HI_PER_MILLE = 975  # sorted draw index = draws * 975 // 1000 = 9750

MAX_ATTEMPTS = 3  # the first call plus the two registered retries
CALL_CAP = 600  # the 601st call to second_verdict is refused, retries included
STOP_AFTER = 20  # consecutive failed calls, treated as an outage and refunded
VOID_MAX_PCT = 5  # Unreadable when either arm is void above this
VOID_GAP_PCT = 3  # Unreadable when the arms' void rates differ by more than this

# What E2 and E4 must reproduce from NR-52's own verdicts before the arms are read, to 2 dp.
NR52_MARGIN = {"sonnet_only": -3.41, "consensus": 0.57}

# E3's subset. 11 of the population's NR-52 verdicts were drawn on 2026-08-06 by an earlier
# probe and reused from cache by NR-52, whose own draws are dated 2026-08-31. They are found
# from the cache files' mtimes and must be exactly these, or E3 is not reported.
NR52_EARLY_BEFORE = datetime(2026, 8, 31, tzinfo=UTC)
NR52_EARLY = frozenset(
    {
        ("cv", "1704.04503"),
        ("cv", "2012.07177"),
        ("graph", "2303.06147"),
        ("rag", "2304.01982"),
        ("rag", "2501.17788"),
        ("rag", "2606.05568"),
        ("rl", "1509.06461"),
        ("rl", "1511.05952"),
        ("speech", "2211.17192"),
        ("speech", "2303.00747"),
        ("speech", "2311.00430"),
    }
)

# Recorded in the registration before any call.
PREDICTION = {
    "e1_point": -0.04,
    "e1_range": [-0.10, 0.02],
    "reading": "Immaterial",
    "e2_range": [-0.5, 0.5],
    "e3_range": [0.05, 0.12],
}

PLANNED_CALLS = 474
PLANNED_USD = 8.0


# ── Population ────────────────────────────────────────────────────────────────────────────


def gold_versions(case: str, pick: str, gold: Path = GOLD) -> list[str]:
    """The versioned ids the first judge holds a verdict for, oldest version first.

    A gold file's name is the id the first judge was shown, sanitised by the rule both caches
    share. The version is read off the name and then checked through `dedup_id`, so the one
    rule for "same paper" decides the match rather than a local string split.
    """
    stem = safe_paper_id(pick)
    found: list[tuple[int, str]] = []
    for f in (gold / case).glob(f"{stem}v*.json"):
        m = re.fullmatch(re.escape(stem) + r"v(\d+)", f.stem)
        if m is None:
            continue
        versioned = f"{pick}v{m.group(1)}"
        if dedup_id(versioned) != pick:
            raise ValueError(f"{case}/{f.name}: the shared id rule does not map it to {pick}")
        found.append((int(m.group(1)), versioned))
    return [v for _, v in sorted(found)]


def choose_versioned(
    case: str,
    pick: str,
    versions: Sequence[str],
    multi: dict[tuple[str, str], str] = MULTI_VERSION,
) -> str:
    """The newest versioned id, with the registered special case asserted, not assumed."""
    if not versions:
        raise ValueError(f"{case}/{pick}: no versioned verdict in the gold cache")
    chosen = versions[-1]
    if len(versions) > 1:
        want = multi.get((case, pick))
        if want is None:
            raise ValueError(f"{case}/{pick}: unregistered versions {list(versions)}")
        if chosen != want:
            raise ValueError(f"{case}/{pick}: newest is {chosen}, registered {want}")
    return chosen


def build_population(
    opus: dict[str, list[dict[str, Any]]],
    cases: Iterable[str],
    nr52: dict[tuple[str, str], int],
    *,
    gold: Path = GOLD,
    multi: dict[tuple[str, str], str] = MULTI_VERSION,
) -> list[dict[str, Any]]:
    """The baseline's arXiv picks, in a stable order, each with both arms' ids.

    `is_arxiv_id` decides which picks are arXiv, the same predicate the judge uses to label
    the identifier line. The order is (case, pick), so the V/U alternation in `--judge`
    does not depend on how the frozen arm file happens to list its picks.
    """
    rows: list[dict[str, Any]] = []
    seen_multi: set[tuple[str, str]] = set()
    for case in sorted(cases):
        picks = [p for p in opus.get(case, []) if is_arxiv_id(str(p["arxiv_id"]))]
        for p in sorted(picks, key=lambda q: str(q["arxiv_id"])):
            pick = str(p["arxiv_id"])
            if dedup_id(pick) != pick:
                raise ValueError(f"{case}/{pick}: a pick should be unversioned")
            versions = gold_versions(case, pick, gold)
            if len(versions) > 1:
                seen_multi.add((case, pick))
            score = son_of(nr52, case, pick)
            if score is None:
                raise ValueError(f"{case}/{pick}: no NR-52 Sonnet verdict")
            rows.append(
                {
                    "case": case,
                    "pick": pick,
                    "versioned": choose_versioned(case, pick, versions, multi),
                    "gpt": int(p["judge_score"]),
                    "nr52": score,
                }
            )
    if seen_multi != set(multi):
        raise ValueError(
            f"several gold versions for {sorted(seen_multi)}; registered {sorted(multi)}"
        )
    return rows


def arm_id(row: dict[str, Any], arm: str) -> str:
    return {"v": row["versioned"], "u": row["pick"]}[arm]


def arm_paper(row: dict[str, Any], record: dict[str, Any], arm: str) -> dict[str, Any]:
    """The paper one arm sends. Both arms read one stored record; only the id differs."""
    return {"title": record["title"], "abstract": record["abstract"], "arxiv_id": arm_id(row, arm)}


def arm_order(index: int) -> tuple[str, str]:
    """V first on even papers, U first on odd, so drift in the served model hits both arms."""
    return ("v", "u") if index % 2 == 0 else ("u", "v")


def early_nr52(
    rows: Iterable[dict[str, Any]], *, before: datetime = NR52_EARLY_BEFORE
) -> list[tuple[str, str]]:
    """The papers whose NR-52 verdict file was written before *before*, by its mtime.

    Read from the files NR-52's verdicts live in, the same path `second_verdict` writes, so
    E3's subset is found from the cache rather than typed in. `early_refusal` then checks the
    result against the 11 the registration counts.
    """
    out: list[tuple[str, str]] = []
    for r in rows:
        f = second_cache_path(DEFAULT_MODEL, r["case"], r["pick"])
        if datetime.fromtimestamp(f.stat().st_mtime, tz=UTC) < before:
            out.append((r["case"], r["pick"]))
    return sorted(out)


def early_refusal(found: Sequence[tuple[str, str]]) -> str | None:
    """Why the early-draw set is not the registered one, or None when it is."""
    if len(found) == len(NR52_EARLY) and set(found) == NR52_EARLY:
        return None
    extra = sorted(set(found) - NR52_EARLY)
    lost = sorted(NR52_EARLY - set(found))
    return (
        f"{len(found)} NR-52 verdicts predate {NR52_EARLY_BEFORE.date()}, registered "
        f"{len(NR52_EARLY)}; unexpected {extra}, missing {lost}"
    )


# ── The prompt check ──────────────────────────────────────────────────────────────────────


def sent_prompt(ctx: str, paper: dict[str, Any]) -> str:
    """The prompt `second_verdict` sends, assembled the way it assembles it.

    `second_verdict` builds this inline and exposes no builder. The test suite captures what
    `second_verdict` actually sends and compares it with this, so the copy cannot drift.
    """
    return f"{judge_mod.RUBRIC}\n\n{judge_mod._build_user_prompt(ctx, paper)}"


def prompt_difference(
    ctx: str, row: dict[str, Any], record: dict[str, Any]
) -> list[tuple[str, str]]:
    """Every line where the V prompt and the U prompt differ, as (V line, U line)."""
    a = sent_prompt(ctx, arm_paper(row, record, "v")).split("\n")
    b = sent_prompt(ctx, arm_paper(row, record, "u")).split("\n")
    if len(a) != len(b):
        return [("<line count>", f"{len(a)} != {len(b)}")]
    return [(x, y) for x, y in zip(a, b, strict=True) if x != y]


def only_the_id_line_differs(ctx: str, row: dict[str, Any], record: dict[str, Any]) -> bool:
    """True when the prompts differ in exactly one line and it is the identifier line.

    An empty difference fails too: two identical prompts would make the probe measure nothing.
    """
    want = [
        (
            judge_mod._identifier_line(arm_paper(row, record, "v")),
            judge_mod._identifier_line(arm_paper(row, record, "u")),
        )
    ]
    return prompt_difference(ctx, row, record) == want


# ── Resolution, done once and stored ──────────────────────────────────────────────────────


def load_papers(path: Path = PAPERS) -> dict[str, dict[str, dict[str, Any]]]:
    if not path.is_file():
        return {}
    return dict(json.loads(path.read_text(encoding="utf-8"))["papers"])


def _replace(tmp: Path, target: Path) -> None:
    """`os.replace(tmp, target)`, retried while Windows refuses it.

    On Windows the rename fails with "Access is denied" while another process, an editor's
    file watcher or the virus scanner, holds *target* open for a moment. The same retry as
    `finescale_model_transfer._replace`, which learned it from a run that died on exactly that.
    """
    for attempt in range(40):
        try:
            os.replace(tmp, target)
            return
        except PermissionError:
            time.sleep(0.25 * min(attempt + 1, 8))
    os.replace(tmp, target)


def _write_json(path: Path, data: Any) -> None:
    # Written whole and swapped in, so an interrupted run never leaves half a file behind.
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    _replace(tmp, path)


def save_papers(papers: dict[str, dict[str, dict[str, Any]]], path: Path = PAPERS) -> None:
    _write_json(
        path,
        {
            "_comment": (
                "One resolution per paper through verify.resolve_references, reused byte for "
                "byte by both arms of evals/sonnet_id_probe.py. no_abstract is void in both "
                "arms. unresolved is retried by the next --plan."
            ),
            "papers": papers,
        },
    )


def record_of(paper: dict[str, Any] | None) -> dict[str, Any]:
    if paper is None:
        return {"status": "unresolved"}
    abstract = str(paper.get("abstract") or "")
    return {
        "status": "ok" if abstract.strip() else "no_abstract",
        "resolved_id": str(paper["arxiv_id"]),
        "title": str(paper.get("title") or ""),
        "abstract": abstract,
    }


def record_for(papers: dict[str, dict[str, dict[str, Any]]], row: dict[str, Any]) -> dict:
    return (papers.get(row["case"]) or {}).get(row["pick"]) or {}


def resolve_missing(
    rows: list[dict[str, Any]],
    papers: dict[str, dict[str, dict[str, Any]]],
    resolver: Callable[[list[str]], list[dict[str, Any]]] | None = None,
    *,
    save: Callable[[dict[str, dict[str, dict[str, Any]]]], None] = save_papers,
    log: Callable[[str], None] = print,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Resolve every paper not yet stored, grouped by case, saving after each case.

    Only unresolved papers are asked again. A stored record is never re-fetched, because a
    second resolution could return a newer version's abstract and the two arms would stop
    sharing their text.
    """
    if resolver is None:
        import arxiv
        from verify import resolve_references

        # One client for the whole run. The library spaces requests by a timestamp kept on
        # the client instance, so a fresh client per case would fire each case's first
        # request with no spacing at all. Same settings as the resolver's own default.
        client = arxiv.Client(page_size=25, delay_seconds=3.0, num_retries=2)

        def _resolve(ids: list[str]) -> list[dict[str, Any]]:
            return resolve_references(ids, [], client)[0]

        resolver = _resolve

    todo: dict[str, list[str]] = {}
    for r in rows:
        if record_for(papers, r).get("status") in (None, "unresolved"):
            todo.setdefault(r["case"], []).append(r["pick"])
    for n, (case, picks) in enumerate(sorted(todo.items()), start=1):
        try:
            got = resolver(picks)
        except Exception as exc:  # noqa: BLE001 -- a failed case stays unresolved, not void
            log(f"  ! {case}: resolver failed: {type(exc).__name__}: {str(exc)[:70]}")
            got = []
        # `dedup_id` on both sides: the resolver returns versioned ids and a pick is not.
        by_id = {dedup_id(str(p["arxiv_id"])): p for p in got}
        for pick in picks:
            papers.setdefault(case, {})[pick] = record_of(by_id.get(dedup_id(pick)))
        save(papers)
        ok = sum(1 for p in picks if papers[case][p]["status"] == "ok")
        log(f"  [{n}/{len(todo)}] {case:<16} resolved {ok}/{len(picks)}")
    return papers


# ── Before any call ───────────────────────────────────────────────────────────────────────


def anthropic_key_resolves() -> bool:
    """Whether the client would find a key for the judge's call, found the way it finds it.

    The config is the one `second_verdict` builds, and `resolve_api_key` is the function the
    client's own dispatch calls, so this answers exactly what the first call would find.
    """
    from reporadar.config import SuggestionsConfig

    cfg = SuggestionsConfig(provider="claude", claude_model=DEFAULT_MODEL, timeout=120)
    return bool(resolve_api_key("claude", cfg))


def register_no_temperature() -> None:
    """Make the client send no temperature field for the judge model, as NR-52's requests did.

    `llm_client._call_claude` sends `temperature: 0` to any model not in `_REJECTS_TEMPERATURE`
    and adds a model only after the API refuses the field. Registering it before the first
    call means no request of this probe carries the field, including the first.
    """
    llm_client._REJECTS_TEMPERATURE.add(DEFAULT_MODEL)


def temperature_refusal() -> str | None:
    """Checked before every call: None while the judge model is registered as above."""
    if DEFAULT_MODEL in llm_client._REJECTS_TEMPERATURE:
        return None
    return f"{DEFAULT_MODEL} is not in llm_client._REJECTS_TEMPERATURE; a call would send one"


def nr52_refusal(margins: dict[str, float]) -> str | None:
    """Why NR-52's margins, recomputed from its own verdicts, miss the registered figures."""
    bad = [
        f"{label} {margins.get(label, float('nan')):+.4f} (registered {want:+.2f})"
        for label, want in NR52_MARGIN.items()
        if label not in margins or round(margins[label], 2) != want
    ]
    return f"NR-52's margins do not reproduce: {', '.join(bad)}" if bad else None


def case_list_refusal(recorded: Sequence[str] | None, current: Sequence[str]) -> str | None:
    """Why the context-verified case list differs from the one `--judge` recorded."""
    if recorded is None:
        return "no case list is recorded; --judge records it before its first call"
    if list(recorded) == list(current):
        return None
    lost = sorted(set(recorded) - set(current))
    new = sorted(set(current) - set(recorded))
    return f"the case list changed since it was recorded: lost {lost}, new {new}"


# ── Buying verdicts ───────────────────────────────────────────────────────────────────────

# Refusals every further call would repeat, by the status `llm_client` leaves on the chain.
# `complete` wraps a 4xx as `LLMError("LLM HTTP <code>: ...")` raised from the HTTPError. The
# Anthropic API answers 401 authentication_error, 402 billing_error, 403 permission_error, and
# a spent credit balance as a 400 whose body names it.
SYSTEMIC_HTTP = {401: "authentication refused", 402: "billing refused", 403: "permission refused"}
CREDIT_REFUSAL = re.compile(r"credit balance|billing_error", re.IGNORECASE)


def _http_errors(exc: BaseException) -> Iterator[urllib.error.HTTPError]:
    seen: set[int] = set()
    e: BaseException | None = exc
    while e is not None and id(e) not in seen:
        seen.add(id(e))
        if isinstance(e, urllib.error.HTTPError):
            yield e
        e = e.__cause__ or e.__context__


def _body(http: urllib.error.HTTPError) -> str:
    try:
        return http.read().decode("utf-8", "replace")
    except Exception:  # noqa: BLE001 -- an unreadable body is just an unknown 400
        return ""


def not_about_the_paper(exc: BaseException) -> str | None:
    """Why *exc* is a failure every further call would repeat, or None if it may be the paper's.

    Such a failure stops the run at once and is not charged to the paper's retries: a missing
    key (`LLMUnavailable`), and an authentication, permission, billing or credit refusal. A
    parse failure or a missing or out-of-range score (`ValueError`/`KeyError` from
    `second_verdict`) is the paper's, and so is a transport failure that survived
    `llm_client.complete`'s own retries. Those are charged; an outage of them is caught by the
    consecutive-failure rule in `buy` and refunded there.
    """
    if isinstance(exc, llm_client.LLMUnavailable):
        return f"{type(exc).__name__}: {str(exc)[:160]}"
    for http in _http_errors(exc):
        if http.code in SYSTEMIC_HTTP:
            return f"HTTP {http.code}, {SYSTEMIC_HTTP[http.code]}"
        if http.code == 400 and CREDIT_REFUSAL.search(_body(http)):
            return "HTTP 400, credit refused"
    if isinstance(exc, llm_client.LLMError):
        text = str(exc)
        m = re.match(r"LLM HTTP (\d{3})\b", text)
        if m and int(m.group(1)) in SYSTEMIC_HTTP:
            return f"HTTP {m.group(1)}, {SYSTEMIC_HTTP[int(m.group(1))]}"
        if CREDIT_REFUSAL.search(text):
            return f"credit refused: {text[:160]}"
    return None


def load_ledger(path: Path = LEDGER) -> dict[str, Any]:
    if not path.is_file():
        return {"calls": 0, "failures": {}, "errors": []}
    return dict(json.loads(path.read_text(encoding="utf-8")))


def save_ledger(ledger: dict[str, Any], path: Path = LEDGER) -> None:
    _write_json(path, ledger)


def failures_of(ledger: dict[str, Any], case: str, pick: str, arm: str) -> int:
    return int(((ledger.get("failures") or {}).get(case) or {}).get(pick, {}).get(arm, 0))


def _charge(ledger: dict[str, Any], case: str, pick: str, arm: str, by: int) -> None:
    arms = ledger.setdefault("failures", {}).setdefault(case, {}).setdefault(pick, {})
    arms[arm] = max(0, arms.get(arm, 0) + by)


def call_plan(
    rows: list[dict[str, Any]], papers: dict[str, dict[str, dict[str, Any]]]
) -> list[dict[str, Any]]:
    """Every verdict to buy, in call order: both arms of a paper back to back, alternating.

    The alternation index is the paper's place in the population, voids included, so the
    order of the rest does not shift when one paper turns out to have no abstract.
    """
    tasks: list[dict[str, Any]] = []
    for i, row in enumerate(rows):
        record = record_for(papers, row)
        if record.get("status") != "ok":
            continue
        for arm in arm_order(i):
            tasks.append(
                {
                    "index": i,
                    "case": row["case"],
                    "pick": row["pick"],
                    "arm": arm,
                    "paper": arm_paper(row, record, arm),
                }
            )
    return tasks


def buy(
    tasks: list[dict[str, Any]],
    call: Callable[[dict[str, Any]], int],
    ledger: dict[str, Any],
    *,
    is_cached: Callable[[dict[str, Any]], bool],
    check: Callable[[], str | None] = lambda: None,
    save: Callable[[dict[str, Any]], None] = save_ledger,
    cap: int = CALL_CAP,
    stop_after: int = STOP_AFTER,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Make the registered calls, under the registered cap, retry, outage and stop rules.

    A cached verdict is not a call. Every attempt is counted before it is made and saved at
    once, so a crash in the middle of a call still counts against the cap. *check* runs before
    each count and refuses the call outright when it names a reason.

    A failure `not_about_the_paper` recognises stops the run and is charged to no paper. Any
    other failure is charged to that paper and arm, which is void once it has failed
    `MAX_ATTEMPTS` times. Charges persist, so a resumed run gives a paper only what is left of
    its retries. The current run of consecutive failures persists too, as `streak`, so a crash
    or a resume does not reset what counts as consecutive. When it reaches *stop_after* the
    run stops, and every failure in it is refunded to its paper's retries while still counting
    toward the cap: an outage voids no paper.
    """
    bought = 0
    streak: list[list[str]] = ledger.setdefault("streak", [])

    def finish(reason: str) -> dict[str, Any]:
        void = sum(
            1
            for t in tasks
            if failures_of(ledger, t["case"], t["pick"], t["arm"]) >= MAX_ATTEMPTS
            and not is_cached(t)
        )
        return {"reason": reason, "bought": bought, "void": void}

    for k, t in enumerate(tasks, start=1):
        if is_cached(t):
            continue
        tag = f"{t['case']}/{t['paper']['arxiv_id']} {t['arm'].upper()}"
        while failures_of(ledger, t["case"], t["pick"], t["arm"]) < MAX_ATTEMPTS:
            if ledger["calls"] >= cap:
                log(f"REFUSED: call {cap + 1} would pass the registered cap of {cap}")
                return finish("cap")
            why = check()
            if why:
                log(f"REFUSED before call {ledger['calls'] + 1}: {why}")
                return finish("refused")
            ledger["calls"] += 1
            save(ledger)
            try:
                score = call(t)
            except Exception as exc:  # noqa: BLE001 -- one bad paper must not lose the rest
                msg = f"{tag}: {type(exc).__name__}: {str(exc)[:120]}"
                ledger.setdefault("errors", []).append(msg)
                ledger["errors"] = ledger["errors"][-50:]
                systemic = not_about_the_paper(exc)
                if systemic:
                    save(ledger)
                    log(f"  ! {msg}")
                    log(f"STOPPED: {systemic}. Not charged to {tag}. Resume once it is fixed.")
                    return finish("unavailable")
                _charge(ledger, t["case"], t["pick"], t["arm"], +1)
                streak.append([t["case"], t["pick"], t["arm"]])
                save(ledger)
                log(f"  ! {msg}")
                if len(streak) >= stop_after:
                    for case, pick, arm in streak:
                        _charge(ledger, case, pick, arm, -1)
                    ledger["refunded"] = int(ledger.get("refunded", 0)) + len(streak)
                    n = len(streak)
                    streak.clear()
                    save(ledger)
                    log(
                        f"STOPPED: {n} consecutive failed calls, an outage. They count toward "
                        f"the cap and are refunded to their papers' retries. Resume later."
                    )
                    return finish("stopped")
                continue
            if streak:
                streak.clear()
                save(ledger)
            bought += 1
            log(f"  [{k}/{len(tasks)}] {tag} -> {score}   (paid calls {ledger['calls']})")
            break
    return finish("done")


# ── The endpoints ─────────────────────────────────────────────────────────────────────────


def case_bootstrap(
    units: Sequence[Any],
    stat: Callable[[list[Any]], float],
    *,
    draws: int = BOOT_DRAWS,
    seed: int = BOOT_SEED,
) -> tuple[float, float]:
    """The registered paired case bootstrap, with both arms carried in each unit.

    One `random.Random(seed)`; each draw picks len(units) cases with replacement, every case
    eligible including those with no arXiv pick; the interval is the sorted draws at index
    draws * 25 // 1000 and draws * 975 // 1000, which for the registered 10,000 draws are
    250 and 9,750 counting from 0, exactly as registered. E1 and the E2/E4 shifts pass the
    same case list, so they see the same draws. The E2/E4 margins themselves take NR-52's
    interval instead, `bigram_report.paired_bootstrap`, as the registration says.
    """
    rng = random.Random(seed)
    k = len(units)
    if k == 0:
        return (0.0, 0.0)
    vals = sorted(stat([units[rng.randrange(k)] for _ in range(k)]) for _ in range(draws))
    return vals[draws * BOOT_LO_PER_MILLE // 1000], vals[draws * BOOT_HI_PER_MILLE // 1000]


def nr52_interval(deltas: Sequence[int | float]) -> list[float]:
    """A margin's interval exactly as `rung1_second_judge.report` computed NR-52's.

    The per-case deltas in case order, as floats, through `bigram_report.paired_bootstrap`
    with its own default draws and seed. Rounded to 4 dp here; NR-52 printed 2.
    """
    lo, hi = paired_bootstrap([float(d) for d in deltas])
    return [round(lo, 4), round(hi, 4)]


def _pooled_diff(units: Sequence[tuple[int, int, int]]) -> float:
    n = sum(u[0] for u in units)
    return (sum(u[1] for u in units) - sum(u[2] for u in units)) / n if n else 0.0


def _mean(units: Sequence[float]) -> float:
    return sum(units) / len(units) if units else 0.0


def _group(rows: list[dict[str, Any]], cases: Sequence[str]) -> dict[str, list[dict[str, Any]]]:
    by: dict[str, list[dict[str, Any]]] = {c: [] for c in cases}
    for r in rows:
        by[r["case"]].append(r)
    return by


def scored_in_both(row: dict[str, Any]) -> bool:
    """Every endpoint reads only these. A paper void in either arm leaves V and U alike."""
    return row["v"] is not None and row["u"] is not None


def e1_units(rows: list[dict[str, Any]], cases: Sequence[str]) -> list[tuple[int, int, int]]:
    """Per case, over the papers both arms scored: (papers, actionable in V, actionable in U).

    One unit for every case in *cases*, including a case with no arXiv pick, whose unit is
    (0, 0, 0). The bootstrap resamples all of them, as registered.
    """
    by = _group(rows, cases)
    units: list[tuple[int, int, int]] = []
    for c in cases:
        both = [r for r in by[c] if scored_in_both(r)]
        units.append(
            (
                len(both),
                sum(1 for r in both if r["v"] >= ACTIONABLE),
                sum(1 for r in both if r["u"] >= ACTIONABLE),
            )
        )
    return units


def e1_actionable(rows: list[dict[str, Any]], cases: Sequence[str]) -> tuple[dict, float, list]:
    """V minus U actionable rate over the papers both arms scored, with its interval."""
    units = e1_units(rows, cases)
    n = sum(u[0] for u in units)
    diff = _pooled_diff(units)
    lo, hi = case_bootstrap(units, _pooled_diff)
    paired = [r for r in rows if scored_in_both(r)]
    out = {
        "n_paired": n,
        "rate_v": round(sum(u[1] for u in units) / n, 4) if n else None,
        "rate_u": round(sum(u[2] for u in units) / n, 4) if n else None,
        "diff": round(diff, 4),
        "ci95": [round(lo, 4), round(hi, 4)],
        "interval_includes_zero": bool(lo <= 0 <= hi),
        "actionable_v_only": sum(1 for r in paired if r["v"] >= ACTIONABLE and r["u"] < ACTIONABLE),
        "actionable_u_only": sum(1 for r in paired if r["u"] >= ACTIONABLE and r["v"] < ACTIONABLE),
    }
    return out, diff, [lo, hi]


def arxiv_net2(rows_c: list[dict[str, Any]], verdict: str, label: str, case: str) -> int:
    """One case's net@2 over the baseline's arXiv picks, with one arm's verdicts.

    Routed through NR-52's own `net2`, so a missing verdict is void rather than -2 exactly as
    it was there.
    """
    picks = [{"arxiv_id": r["pick"], "judge_score": r["gpt"]} for r in rows_c]
    cache = {(case, safe_paper_id(r["pick"])): r[verdict] for r in rows_c if r[verdict] is not None}
    return net2(picks, LABELS[label], cache, case)[0]


def margin_shift(
    rows: list[dict[str, Any]],
    fixed: dict[str, dict[str, dict[str, int]]],
    cases: Sequence[str],
    label: str,
) -> tuple[dict[str, Any], float]:
    """NR-52's margin with the baseline's arXiv verdicts from NR-52, from V and from U.

    Everything else keeps its NR-52 verdict: the shipped arm (`rr`) and the baseline's other
    picks (`opus5_rest`). net@2 is a sum over papers, so the case margin splits exactly into
    those fixed parts and the arXiv part this probe re-judges.

    The NR-52 margin uses every row, because it must reproduce NR-52. margin(V) and margin(U)
    use only the papers scored in both arms: a paper void in either arm is dropped from the
    baseline's picks in both, and keeps no verdict in either. Each margin carries NR-52's own
    interval (`nr52_interval`). The shift, margin(V) minus margin(U), carries the E1
    bootstrap's interval over the same cases and the same draws.
    """
    by = _group(rows, cases)
    both = {c: [r for r in by[c] if scored_in_both(r)] for c in cases}

    def deltas(verdict: str, part: dict[str, list[dict[str, Any]]]) -> list[int]:
        return [
            fixed[c][label]["rr"]
            - fixed[c][label]["opus5_rest"]
            - arxiv_net2(part[c], verdict, label, c)
            for c in cases
        ]

    d_nr52, d_v, d_u = deltas("nr52", by), deltas("v", both), deltas("u", both)
    units = [float(a - b) for a, b in zip(d_v, d_u, strict=True)]
    shift = _mean(units)
    lo, hi = case_bootstrap(units, _mean)
    nr52 = _mean([float(d) for d in d_nr52])
    out = {
        "margin_nr52": round(nr52, 4),
        "ci95_nr52": nr52_interval(d_nr52),
        "reproduces_nr52": bool(round(nr52, 2) == NR52_MARGIN[label]),
        "margin_v": round(_mean([float(d) for d in d_v]), 4),
        "ci95_v": nr52_interval(d_v),
        "margin_u": round(_mean([float(d) for d in d_u]), 4),
        "ci95_u": nr52_interval(d_u),
        "shift_v_minus_u": round(shift, 4),
        "ci95_shift": [round(lo, 4), round(hi, 4)],
    }
    return out, shift


def _flips(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    flips = sum(1 for r in rows if (r["u"] >= ACTIONABLE) != (r["nr52"] >= ACTIONABLE))
    exact = sum(1 for r in rows if r["u"] == r["nr52"])
    return {
        "n": n,
        "flips_at_2": flips,
        "flip_rate": round(flips / n, 4) if n else None,
        "exact_agreement": exact,
        "exact_rate": round(exact / n, 4) if n else None,
    }


def replicate_flips(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """E3: U against NR-52 across score 2, on every paper both arms scored, and on those whose
    NR-52 verdict NR-52 drew itself (the early draws, flagged per row, left out)."""
    paired = [r for r in rows if scored_in_both(r)]
    return {
        "all": _flips(paired),
        "nr52_drawn": _flips([r for r in paired if not r["nr52_early"]]),
        "nr52_early": sorted(f"{r['case']}/{r['pick']}" for r in rows if r["nr52_early"]),
    }


def unreadable(n: int, void_v: int, void_u: int) -> bool:
    """More than 5% void in either arm, or void rates more than 3 points apart.

    Integer arithmetic on counts, so exactly 5% and exactly 3 points sit on the readable side
    as registered ("more than"), with no float rounding deciding it.
    """
    return (
        void_v * 100 > VOID_MAX_PCT * n
        or void_u * 100 > VOID_MAX_PCT * n
        or abs(void_v - void_u) * 100 > VOID_GAP_PCT * n
    )


def reading(n: int, void_v: int, void_u: int, e1_ci: Sequence[float]) -> str:
    """The registered reading: Unreadable by the void rules, else Material if and only if the
    E1 interval excludes zero, else Immaterial. E2 is reported, never read."""
    if unreadable(n, void_v, void_u):
        return "Unreadable"
    if not (e1_ci[0] <= 0 <= e1_ci[1]):
        return "Material"
    return "Immaterial"


def _within(x: float | None, bounds: Sequence[float]) -> bool | None:
    return None if x is None else bool(bounds[0] <= x <= bounds[1])


def summarise(
    rows: list[dict[str, Any]],
    fixed: dict[str, dict[str, dict[str, int]]],
    cases: Sequence[str],
) -> dict[str, Any]:
    """Every registered number, from the per-paper rows and the per-case fixed parts alone.

    Kept pure so the tracked artifact can be checked by recomputing its summary from its rows.
    """
    n = len(rows)
    void = {arm: sum(1 for r in rows if r[f"void_{arm}"]) for arm in ARMS}
    dropped = [f"{r['case']}/{r['pick']}" for r in rows if not scored_in_both(r)]
    e1, e1_diff, e1_ci = e1_actionable(rows, cases)
    e2, e2_shift = margin_shift(rows, fixed, cases, "sonnet_only")
    e4, _ = margin_shift(rows, fixed, cases, "consensus")
    e3 = replicate_flips(rows)
    result = reading(n, void["v"], void["u"], e1_ci)
    return {
        "n_population": n,
        "n_cases": len(cases),
        "voids": {
            "v": void["v"],
            "u": void["u"],
            "v_rate": round(void["v"] / n, 4) if n else None,
            "u_rate": round(void["u"] / n, 4) if n else None,
            "n_scored_in_both": n - len(dropped),
            "dropped_from_every_endpoint": dropped,
        },
        "e1": e1,
        "e2": e2,
        "e3": e3,
        "e4": e4,
        "reading": {
            "result": result,
            "unreadable": unreadable(n, void["v"], void["u"]),
            "e1_interval_excludes_zero": not e1["interval_includes_zero"],
        },
        "prediction": {
            **PREDICTION,
            "e1_in_range": _within(e1_diff, PREDICTION["e1_range"]),
            "e1_interval_includes_zero": e1["interval_includes_zero"],
            "reading_as_predicted": result == PREDICTION["reading"],
            "e2_in_range": _within(e2_shift, PREDICTION["e2_range"]),
            "e3_in_range": _within(e3["all"]["flip_rate"], PREDICTION["e3_range"]),
            "e3_nr52_drawn_in_range": _within(
                e3["nr52_drawn"]["flip_rate"], PREDICTION["e3_range"]
            ),
        },
    }


# ── Shared set-up ─────────────────────────────────────────────────────────────────────────


def _setup() -> dict[str, Any]:
    ship, opus = shipped_arm(), opus5_arm()
    both = sorted(set(ship) & set(opus))
    nr52 = cached_sonnet()
    full = build_population(opus, both, nr52)
    contexts, drifted = verify_contexts(both)
    return {
        "ship": ship,
        "opus": opus,
        "both": both,
        "nr52": nr52,
        "full": full,
        "judged": sum(len(opus[c]) for c in both),
        "contexts": contexts,
        "drifted": sorted(drifted),
        "cases": sorted(contexts),
        "rows": [r for r in full if r["case"] in contexts],
    }


def _nr52_deltas(s: dict[str, Any], label: str) -> list[int]:
    """NR-52's per-case deltas exactly as `rung1_second_judge.report` computes them."""
    fn, nr52 = LABELS[label], s["nr52"]
    return [
        net2(s["ship"][c], fn, nr52, c)[0] - net2(s["opus"][c], fn, nr52, c)[0] for c in s["cases"]
    ]


def _nr52_margin(s: dict[str, Any], label: str) -> float:
    return st.mean(_nr52_deltas(s, label))


def fixed_parts(s: dict[str, Any]) -> dict[str, dict[str, dict[str, int]]]:
    keys = {(r["case"], r["pick"]) for r in s["rows"]}
    out: dict[str, dict[str, dict[str, int]]] = {}
    for c in s["cases"]:
        rest = [p for p in s["opus"][c] if (c, str(p["arxiv_id"])) not in keys]
        out[c] = {
            label: {
                "rr": net2(s["ship"][c], LABELS[label], s["nr52"], c)[0],
                "opus5_rest": net2(rest, LABELS[label], s["nr52"], c)[0],
            }
            for label in ("sonnet_only", "consensus")
        }
    return out


# ── Commands ──────────────────────────────────────────────────────────────────────────────


def plan() -> int:
    s = _setup()
    rows, full = s["rows"], s["full"]
    print("population: Opus 5 draw 1 as rung1_second_judge.opus5_arm() loads it")
    print(f"  cases in both arms:  {len(s['both'])} (registered {EXPECTED_CASES})")
    print(f"  judged picks:        {s['judged']} (registered {EXPECTED_JUDGED})")
    print(f"  arXiv picks:         {len(full)} (registered {EXPECTED_POPULATION})")
    print(f"  other picks:         {s['judged'] - len(full)} (keep their NR-52 verdicts)")
    print(f"  NR-52 Sonnet verdict present: {len(full)}/{len(full)}")
    for (case, pick), want in MULTI_VERSION.items():
        print(f"  several gold versions: {case}/{pick} {gold_versions(case, pick)} -> {want}")
    mismatch = []
    for r in full:
        f = GOLD / r["case"] / f"{safe_paper_id(r['versioned'])}.json"
        if int(json.loads(f.read_text(encoding="utf-8"))["score"]) != r["gpt"]:
            mismatch.append(f"{r['case']}/{r['versioned']}")
    # Informational. The gold file is the verdict the frozen arm's score was read from, so a
    # difference would mean the versioned id chosen here is not the one behind that score.
    print(
        f"  gold file score differs from the frozen GPT-5.5 score: {len(mismatch)} {mismatch[:10]}"
    )

    early = early_nr52(full)
    early_bad = early_refusal(early)
    print(f"\nE3's early NR-52 draws (cache file written before {NR52_EARLY_BEFORE.date()}):")
    print(f"  {len(early)} of {len(full)} (registered {len(NR52_EARLY)}); ok={early_bad is None}")
    print("  " + " ".join(f"{c}/{p}" for c, p in early))
    if early_bad:
        print(f"  FAILED: {early_bad}")

    print("\ndrift check (second_judge.verify_contexts):")
    print(f"  context-verified: {len(s['contexts'])}/{len(s['both'])}")
    if s["drifted"]:
        lost = sum(1 for r in full if r["case"] in s["drifted"])
        print(f"  DRIFTED: {s['drifted']} ({lost} papers); --judge refuses while any case drifts")
    else:
        print("  no drift")
    recorded = load_ledger().get("cases")
    if recorded is None:
        print("  case list: not recorded yet; --judge records it before its first call")
    else:
        why = case_list_refusal(recorded, s["cases"])
        print(f"  case list recorded by --judge: {len(recorded)} cases, " + (why or "unchanged"))

    print("\nNR-52 reproduced from its own verdicts (rung1_second_judge.net2):")
    published = {}
    if NR52_FROZEN.is_file():
        published = json.loads(NR52_FROZEN.read_text(encoding="utf-8")).get("labels", {})
    for label, want in NR52_MARGIN.items():
        got = _nr52_margin(s, label)
        ci = nr52_interval(_nr52_deltas(s, label))
        was = (published.get(label) or {}).get("ci95")
        print(
            f"  {label:<12} {got:+.4f}  registered {want:+.2f}  ok={round(got, 2) == want}"
            f"   interval [{ci[0]:+.2f}, {ci[1]:+.2f}] (NR-52 published {was})"
        )

    # All 474 ids, compactly: the U id is the pick and the V id adds the bracketed suffix. The
    # suffix is sliced off the gold id for display only; `gold_versions` already proved, via
    # `dedup_id`, that the gold id is this pick.
    print("\nids per arm (U is the pick id; V is the pick id plus the bracketed gold version):")
    for c in s["cases"]:
        items = [f"{r['pick']}[{r['versioned'][len(r['pick']) :]}]" for r in rows if r["case"] == c]
        print(f"  {c:<16} ({len(items)}) " + " ".join(items))

    print("\nresolution (verify.resolve_references, once per paper, stored):")
    papers = resolve_missing(rows, load_papers())
    status: dict[str, int] = {}
    for r in rows:
        k = record_for(papers, r).get("status", "missing")
        status[k] = status.get(k, 0) + 1
    print(f"  {dict(sorted(status.items()))}   stored in {PAPERS.relative_to(EVALS.parent)}")
    ok_rows = [r for r in rows if record_for(papers, r).get("status") == "ok"]
    newer = sum(1 for r in ok_rows if record_for(papers, r)["resolved_id"] != r["versioned"])
    print(f"  resolver's current version differs from the gold version: {newer}/{len(ok_rows)}")
    pending: dict[str, list[str]] = {}
    for r in rows:
        why = record_for(papers, r).get("status", "missing")
        if why != "ok":
            pending.setdefault(f"{why} in {r['case']}", []).append(r["pick"])
    for where, picks in pending.items():
        print(f"    {where} ({len(picks)}): {' '.join(picks)}")
    if any(k.startswith(("unresolved", "missing")) for k in pending):
        print(
            "  unresolved papers are retried by the next --plan; --judge refuses until none remain"
        )

    bad = [
        f"{r['case']}/{r['pick']}"
        for r in ok_rows
        if not only_the_id_line_differs(s["contexts"][r["case"]], r, record_for(papers, r))
    ]
    print("\nprompt check (RUBRIC + judge._build_user_prompt, as second_verdict sends it):")
    good = len(ok_rows) - len(bad)
    print(f"  prompts differing only in the identifier line: {good}/{len(ok_rows)}")
    if bad:
        print(f"  FAILED: {bad}")

    print("\nsample identifier lines (judge._identifier_line):")
    samples = [r for r in ok_rows if (r["case"], r["pick"]) in MULTI_VERSION]
    samples += [r for r in ok_rows if "/" in r["pick"]][:1]
    samples += [r for r in ok_rows if r not in samples][: max(0, 3 - len(samples))]
    for r in samples[:3]:
        rec = record_for(papers, r)
        print(f"  {r['case']}/{r['pick']}")
        for arm in ARMS:
            line = judge_mod._identifier_line(arm_paper(r, rec, arm))
            print(f"    {arm.upper()}  {line}")

    ledger = load_ledger()
    tasks = call_plan(rows, papers)
    cached = {a: sum(1 for t in tasks if t["arm"] == a and _is_cached(t)) for a in ARMS}
    todo = [t for t in tasks if not _is_cached(t)]
    spent = [t for t in todo if failures_of(ledger, t["case"], t["pick"], t["arm"]) >= MAX_ATTEMPTS]
    todo = [t for t in todo if t not in spent]
    print("\nverdicts:")
    for a in ARMS:
        where = f".work/second_judge/{NAMESPACE[a]}"
        print(f"  {a.upper()} cached {cached[a]}/{len(ok_rows)} in {where}")
    print(f"  calls to second_verdict so far: {ledger['calls']} of a cap of {CALL_CAP}")
    print(f"  refunded after an outage so far: {ledger.get('refunded', 0)}")
    print(f"  void after retries so far: {len(spent)}")
    usd = len(todo) * PLANNED_USD / PLANNED_CALLS
    print(f"  still to buy: {len(todo)} of {PLANNED_CALLS} planned (~${usd:.2f})")
    return 0 if not bad and early_bad is None else 1


def _is_cached(t: dict[str, Any]) -> bool:
    return second_cache_path(NAMESPACE[t["arm"]], t["case"], str(t["paper"]["arxiv_id"])).is_file()


def judge() -> int:
    from run_judge_eval import load_dotenv
    from second_judge import second_verdict

    load_dotenv(EVALS / ".env")
    s = _setup()
    got = (len(s["both"]), s["judged"], len(s["full"]))
    if got != (EXPECTED_CASES, EXPECTED_JUDGED, EXPECTED_POPULATION):
        print(
            f"REFUSED: {got[0]} cases, {got[1]} judged picks, {got[2]} arXiv picks; registered "
            f"{EXPECTED_CASES}, {EXPECTED_JUDGED}, {EXPECTED_POPULATION}"
        )
        return 1
    if s["drifted"]:
        print(f"REFUSED: context drift in {s['drifted']}; no call made")
        return 1
    why = nr52_refusal({label: _nr52_margin(s, label) for label in NR52_MARGIN})
    if why:
        print(f"REFUSED: {why}")
        return 1
    rows, contexts = s["rows"], s["contexts"]
    papers = load_papers()
    pending = [
        f"{r['case']}/{r['pick']}"
        for r in rows
        if record_for(papers, r).get("status") not in ("ok", "no_abstract")
    ]
    if pending:
        print(f"REFUSED: {len(pending)} papers not resolved yet; run --plan again: {pending}")
        return 1
    bad = [
        f"{r['case']}/{r['pick']}"
        for r in rows
        if record_for(papers, r)["status"] == "ok"
        and not only_the_id_line_differs(contexts[r["case"]], r, record_for(papers, r))
    ]
    if bad:
        print(f"REFUSED: prompts differ beyond the identifier line for {bad}")
        return 1
    if not anthropic_key_resolves():
        print("REFUSED: no Anthropic API key resolves (evals/.env, environment, rr auth); no call")
        return 1

    ledger = load_ledger()
    if ledger.get("cases") is not None:
        why = case_list_refusal(ledger["cases"], s["cases"])
        if why:
            print(f"REFUSED: {why}")
            return 1
    ledger["cases"] = s["cases"]
    save_ledger(ledger)

    register_no_temperature()

    def call(t: dict[str, Any]) -> int:
        ns = NAMESPACE[t["arm"]]
        return second_verdict(
            t["case"], contexts[t["case"]], t["paper"], DEFAULT_MODEL, cache_as=ns
        )

    tasks = call_plan(rows, papers)
    print(
        f"{len(tasks)} verdicts in call order over {len(s['cases'])} recorded cases; "
        f"{ledger['calls']} calls already made"
    )
    out = buy(tasks, call, ledger, is_cached=_is_cached, check=temperature_refusal)
    print(
        f"\n{out['reason']}: bought {out['bought']} this run, {out['void']} void after retries, "
        f"{ledger['calls']} calls in total (cap {CALL_CAP}), "
        f"{ledger.get('refunded', 0)} refunded after an outage"
    )
    return 0 if out["reason"] == "done" else 2


def score_rows(
    rows: list[dict[str, Any]],
    papers: dict[str, dict[str, dict[str, Any]]],
    ledger: dict[str, Any],
    caches: dict[str, dict[tuple[str, str], int]],
    early: Iterable[tuple[str, str]] = (),
) -> tuple[list[dict[str, Any]], list[str]]:
    """Per-paper rows with both arms, and what is still missing beyond the voids.

    A void is a paper with no abstract (both arms) or one whose retries ran out (that arm
    only). A paper with no verdict and neither reason is unfinished, not void. `nr52_early`
    marks the papers whose NR-52 verdict an earlier probe drew, for E3's subset.
    """
    early = set(early)
    out: list[dict[str, Any]] = []
    missing: list[str] = []
    for r in rows:
        no_abstract = record_for(papers, r).get("status") == "no_abstract"
        row: dict[str, Any] = {k: r[k] for k in ("case", "pick", "versioned", "gpt", "nr52")}
        row["nr52_early"] = (r["case"], r["pick"]) in early
        flags: dict[str, bool] = {}
        for arm in ARMS:
            score = son_of(caches[arm], r["case"], arm_id(r, arm))
            spent = failures_of(ledger, r["case"], r["pick"], arm) >= MAX_ATTEMPTS
            flags[f"void_{arm}"] = score is None and (no_abstract or spent)
            if score is None and not flags[f"void_{arm}"]:
                missing.append(f"{r['case']}/{arm_id(r, arm)} ({arm.upper()})")
            row[arm] = score
        out.append({**row, **flags})
    return out, missing


def report() -> int:
    ledger = load_ledger()
    s = _setup()
    why = case_list_refusal(ledger.get("cases"), s["cases"])
    if why:
        print(f"REFUSED: {why}")
        return 1
    cases = list(ledger["cases"])
    early = early_nr52(s["full"])
    why = early_refusal(early)
    if why:
        print(f"REFUSED: {why}")
        return 1
    caches = {arm: cached_sonnet(NAMESPACE[arm]) for arm in ARMS}
    rows, missing = score_rows(s["rows"], load_papers(), ledger, caches, early)
    if missing:
        print(f"REFUSED: {len(missing)} verdicts neither bought nor void:")
        for m in missing:
            print(f"  {m}")
        return 1

    fixed = fixed_parts(s)
    summary = summarise(rows, fixed, cases)
    # Two routes to NR-52's margin must agree before either arm is read: the full arms
    # through `net2`, as NR-52 computed it, and the split into fixed parts and arXiv rows
    # that E2 and E4 are built on. Then both must give the registered figures.
    for label, key in (("sonnet_only", "e2"), ("consensus", "e4")):
        direct = _nr52_margin(s, label)
        split = summary[key]["margin_nr52"]
        if round(direct, 4) != split or not summary[key]["reproduces_nr52"]:
            print(
                f"REFUSED: NR-52's {label} margin does not reproduce: direct {direct:+.4f}, "
                f"split {split:+.4f}, registered {NR52_MARGIN[label]:+.2f}"
            )
            return 1

    out = {
        "_comment": (
            "Did the identifier line move the second judge's verdicts on the baseline's arXiv "
            "picks? Pre-registered in evals/PREREG-sonnet-id-probe.md before any verdict in "
            "this probe existed. Arm V shows the versioned id the first judge saw, arm U the "
            "unversioned pick id as NR-52 sent it, drawn afresh. Everything else is shared. "
            "Every endpoint reads only papers scored in both arms. The summary is recomputable "
            "from rows and per_case_fixed with summarise(). Derived by "
            "evals/sonnet_id_probe.py; pinned by tests/test_sonnet_id_probe.py."
        ),
        "pre_registration": "evals/PREREG-sonnet-id-probe.md",
        "model": DEFAULT_MODEL,
        "namespaces": NAMESPACE,
        "bootstrap": {
            "draws": BOOT_DRAWS,
            "seed": BOOT_SEED,
            "unit": "case",
            "interval_indices": [
                BOOT_DRAWS * BOOT_LO_PER_MILLE // 1000,
                BOOT_DRAWS * BOOT_HI_PER_MILLE // 1000,
            ],
        },
        "margin_bootstrap": {
            "function": "bigram_report.paired_bootstrap",
            "draws": BOOTSTRAP_N,
            "seed": BOOTSTRAP_SEED,
        },
        "calls": {"made": ledger["calls"], "refunded_after_outage": ledger.get("refunded", 0)},
        "cases": cases,
        "nr52_early": [f"{c}/{p}" for c, p in early],
        "per_case_fixed": fixed,
        "summary": summary,
        "rows": rows,
    }
    _write_json(FROZEN, out)

    v = summary
    e1, e2, e3, e4 = v["e1"], v["e2"], v["e3"], v["e4"]
    vo = v["voids"]
    print(f"{v['n_population']} papers over {v['n_cases']} cases")
    print(
        f"voids: V {vo['v']} ({vo['v_rate']:.1%}), U {vo['u']} ({vo['u_rate']:.1%}); "
        f"dropped from every endpoint: {len(vo['dropped_from_every_endpoint'])} "
        f"{vo['dropped_from_every_endpoint']}"
    )
    print(
        f"\nE1 actionable rate over {e1['n_paired']} paired papers: V {e1['rate_v']:.3f}, "
        f"U {e1['rate_u']:.3f}, V-U {e1['diff']:+.4f} CI [{e1['ci95'][0]:+.4f}, "
        f"{e1['ci95'][1]:+.4f}]  discordant V-only {e1['actionable_v_only']}, "
        f"U-only {e1['actionable_u_only']}"
    )
    for name, e in (("E2 sonnet_only", e2), ("E4 consensus", e4)):
        print(
            f"{name:<15} margin NR-52 {e['margin_nr52']:+.4f}, "
            f"V {e['margin_v']:+.4f} [{e['ci95_v'][0]:+.2f}, {e['ci95_v'][1]:+.2f}], "
            f"U {e['margin_u']:+.4f} [{e['ci95_u'][0]:+.2f}, {e['ci95_u'][1]:+.2f}]; "
            f"shift V-U {e['shift_v_minus_u']:+.4f} "
            f"CI [{e['ci95_shift'][0]:+.4f}, {e['ci95_shift'][1]:+.4f}]"
        )
    for name, part in (("all", e3["all"]), ("NR-52-drawn", e3["nr52_drawn"])):
        print(
            f"E3 U vs NR-52, {name}: {part['flips_at_2']}/{part['n']} flip at >=2 "
            f"({part['flip_rate']:.1%}), exact agreement {part['exact_agreement']}/{part['n']}"
        )
    print(f"   early NR-52 draws left out of the second: {e3['nr52_early']}")
    print(f"\nREADING: {v['reading']['result']}")
    p = v["prediction"]
    print(
        f"prediction: E1 in {p['e1_range']} {p['e1_in_range']}, interval includes zero "
        f"{p['e1_interval_includes_zero']}, reading as predicted {p['reading_as_predicted']}, "
        f"E2 in {p['e2_range']} {p['e2_in_range']}, E3 in {p['e3_range']} {p['e3_in_range']}"
    )
    print(f"wrote {FROZEN.name}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--plan", action="store_true", help="no judge call (the default)")
    mode.add_argument("--judge", action="store_true", help="buy the verdicts, ~$8")
    mode.add_argument("--report", action="store_true", help="E1-E4 from cached verdicts")
    args = ap.parse_args()
    if args.judge:
        return judge()
    if args.report:
        return report()
    return plan()


if __name__ == "__main__":
    raise SystemExit(main())
