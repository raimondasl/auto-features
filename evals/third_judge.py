"""Where does a third vendor's judge fall between GPT-5.5 and Sonnet?

Pre-registered in `evals/PREREG-third-judge.md`. That file is the specification. Where this
script and the registration disagree, the registration wins.

    uv run python evals/third_judge.py            # --plan: no paid call (the default)
    uv run python evals/third_judge.py --judge    # paid: the registered verdicts, resumable
    uv run python evals/third_judge.py --report   # no paid call: reproductions, E1-E4, readings

`--plan` resolves the baseline's DOI picks once through the free `verify.resolve_references` into
`.work/third_judge/doi_papers.json`, which `--judge` reuses; its only Gemini calls are free
`countTokens` calls on a sample of prompts. One verdict is cached per distinct prompt under
`.work/third_judge/gemini-3.8-flash/<sha256 of the prompt>.json`, with the ledger beside it, and
nothing is written to another judge's cache.

Two operating details the registration leaves open. "A call that would take the total past $40"
is judged against the most that call could cost: every prompt byte a token and the whole output
limit. A request the API refuses as malformed (a 4xx that is neither a 429 nor a key refusal)
stops the run like a key refusal, uncharged, because it would fail for every prompt.

What decides a reading is not only the registration. The reference values, the thresholds, the
prediction and the reading functions live in this file, and the client that carries the request
lives beside it. So `FROZEN` names all three, and `--judge` and `--report` both refuse unless
every one of them is committed and identical to HEAD. `--report` records the HEAD sha and each
file's blob sha in the artifact, so a reading can be tied to the code that produced it.
"""

from __future__ import annotations

import argparse
import contextlib
import difflib
import hashlib
import io
import json
import random
import statistics as st
import subprocess
import sys
import tempfile
import threading
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

EVALS = Path(__file__).resolve().parent
ROOT = EVALS.parent
sys.path.insert(0, str(EVALS))
sys.path.insert(0, str(ROOT / "src"))

import gemini_client as gc  # noqa: E402
import judge as judge_mod  # noqa: E402
import rung1_second_judge as r1  # noqa: E402
import second_judge as sj  # noqa: E402
import sonnet_id_probe as sip  # noqa: E402
from band_testbeds import auc  # noqa: E402
from bigram_report import paired_bootstrap  # noqa: E402

from anonymous.paper_id import is_arxiv_id  # noqa: E402

WORK = EVALS / ".work"
STATE = WORK / "third_judge"
DOI_FILE = STATE / "doi_papers.json"
BAND_FILE = WORK / "second_judge_band.json"
TRIAGE_PAPERS = WORK / "triage_papers.json"
OUT = EVALS / "third_judge.json"
ENV_FILE = EVALS / ".env"
PREREG_REL = "evals/PREREG-third-judge.md"
# Everything that decides a reading: the registration, the script that executes it, the client
# that carries the request. All three must be committed and unchanged before any verdict is read.
FROZEN = (PREREG_REL, "evals/third_judge.py", "evals/gemini_client.py")
KEY_NAME = "GEMINI_API_KEY"

# ── Registered values ─────────────────────────────────────────────────────────────────────

MODEL = "gemini-3.8-flash"
LEVEL = "MEDIUM"  # thinkingConfig.thinkingLevel
MAX_OUTPUT_TOKENS = 16_384
SAFETY = ("HARASSMENT", "HATE_SPEECH", "SEXUALLY_EXPLICIT", "DANGEROUS_CONTENT")
PRICE_IN, PRICE_OUT = 0.75, 3.75  # USD per million tokens; output includes thinking
SPEND_CAP, CALL_CAP = 40.0, 3000  # the 3,001st call is refused
WORKERS = 4
TRANSPORT_RETRIES = 3  # a transport error or a 5xx
CONTENT_RETRIES = 1  # a parse failure, a truncated or a blocked answer
OUTAGE_AFTER = 20
RATE_LIMIT_STOP = 20  # consecutive 429s with no verdict between them
ACTIONABLE = 2
BOOT_DRAWS, BOOT_SEED = 10_000, 20260923
BOOT_LO, BOOT_HI = 250, 9750  # indices into the sorted draws, counting from 0
VOID_MAX = 0.05  # more than this share void makes a population Unreadable
VOID_GAP = 0.03  # the comparison's two arms may not differ by more than this in void rate

POPULATIONS = ("band", "ours", "baseline", "adopted", "crossrepo")
EXPECTED = {"band": 324, "ours": 306, "baseline": 357, "adopted": 188, "crossrepo": 502}
N_CASES = 37
E3_RATES = ("adopted", "crossrepo")
# The registration's four populations. The comparison is our arm and the baseline together.
REGISTERED: dict[str, tuple[str, ...]] = {"comparison": ("ours", "baseline")}
REGISTERED.update({p: (p,) for p in ("band", *E3_RATES)})
# Each level reading's reference pair, (GPT-5.5, Sonnet), as the population table states it.
REFERENCE = {"band": (0.873, 0.494), "adopted": (0.819, 0.644), "crossrepo": (0.255, 0.068)}
MARGINS = (0.32, -3.41)  # NR-52 under GPT-5.5 and under Sonnet
PREDICTION: dict[str, tuple[float | None, float | None]] = {
    "E1 band share": (0.55, 0.85),
    "E2 band AUC": (0.62, 0.78),
    "E3 adopted rate": (0.60, 0.88),
    "E3 cross-repository rate": (0.07, 0.28),
    "E3 adopted-against-control AUC": (0.75, None),
    "E4 margin": (-3.41, 0.32),
}
# The registration's reading prediction: its point estimate, its mode, and the probability it put
# on each reading. `other` is the mass the registration left to every reading it did not name,
# so a reading read off `other` is an upper bound on what was predicted for it, not a point.
PREDICTED_READING: dict[str, dict[str, Any]] = {
    "E1": {
        "point": 0.70,
        "mode": "between",
        "p": {"between": 0.5, "overlaps GPT-5.5": 0.25, "other": 0.25},
    },
    "E2": {"point": 0.70, "mode": "orders the band", "p": {"orders the band": 0.9, "other": 0.1}},
    "E3 adopted": {"point": 0.75, "mode": "between", "p": {"between": 0.45, "other": 0.55}},
    "E3 crossrepo": {"point": 0.15, "mode": "between", "p": {"between": 0.45, "other": 0.55}},
    "E4": {"point": -1.5, "mode": "between", "p": {"between": 0.6, "other": 0.4}},
}
VOID_PREDICTION = 0.02

# Operating values that are not registered numbers.
RATE_WAIT = 60.0  # seconds, for a 429 that carries no RetryInfo delay
COST_SCENARIOS = {"300 visible + 1,000 thinking": 1300, "300 visible + 4,000 thinking": 4300}
CHARS_PER_TOKEN = 2.5  # the fallback when countTokens fails; low, so the estimate runs high

# ── The request and the parsing rule ──────────────────────────────────────────────────────


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_prompt(ctx: str, paper: dict[str, Any]) -> str:
    """The string `second_judge.second_verdict` sends Sonnet, byte for byte."""
    return f"{judge_mod.RUBRIC}\n\n{judge_mod._build_user_prompt(ctx, paper)}"


def request_body(prompt: str) -> dict[str, Any]:
    """One user message. No temperature, topP, topK or seed."""
    cfg = {"maxOutputTokens": MAX_OUTPUT_TOKENS, "thinkingConfig": {"thinkingLevel": LEVEL}}
    block = [{"category": f"HARM_CATEGORY_{c}", "threshold": "BLOCK_NONE"} for c in SAFETY]
    return {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": cfg,
        "safetySettings": block,
    }


def parse_score(text: str) -> int | None:
    """First `{` to last `}` as JSON; `score` must be an int from 0 to 3, never a bool or float."""
    a, b = text.find("{"), text.rfind("}")
    if a < 0 or b < a:
        return None
    try:
        data = json.loads(text[a : b + 1])
    except ValueError:
        return None
    score = data.get("score") if isinstance(data, dict) else None
    return score if type(score) is int and 0 <= score <= 3 else None


def classify(o: gc.Outcome) -> tuple[str, int | None]:
    """An attempt's class: scored, parse/truncated/blocked, rate_limit, transport, key, fatal."""
    if o.kind == gc.TEXT:
        score = parse_score(o.text)
        return ("scored", score) if score is not None else ("parse", None)
    if o.kind == gc.FINISH:
        return ("truncated" if o.finish_reason == "MAX_TOKENS" else "blocked"), None
    # A malformed 200 is an answer the server meant to send, so it is a content failure, not a
    # transport one: one retry, as the registration gives every other unreadable answer.
    kinds = {gc.BLOCKED: "blocked", gc.RATE_LIMIT: "rate_limit", gc.MALFORMED: "parse"}
    return kinds.get(o.kind, {gc.TRANSPORT: "transport", gc.KEY: "key"}.get(o.kind, "fatal")), None


# ── Populations ───────────────────────────────────────────────────────────────────────────

MEMBER_KEYS = ("population", "case", "id", "gpt", "sonnet")


@dataclass
class Item:
    """One paper of one population, and the prompt it is asked with (None without text)."""

    population: str
    case: str
    id: str
    gpt: int
    sonnet: int | None
    ctx: str | None
    paper: dict[str, Any] | None
    kind: str = ""  # "arxiv" or "doi" on the baseline
    finescale: float | None = None
    prompt: str | None = field(init=False, default=None)
    sha: str | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        ctx, paper = self.ctx, self.paper
        if ctx is not None and paper is not None and str(paper.get("abstract") or "").strip():
            self.prompt = build_prompt(ctx, paper)
            self.sha = sha256(self.prompt)

    def member(self) -> dict[str, Any]:
        m = {k: getattr(self, k) for k in MEMBER_KEYS}
        return m if self.finescale is None else {**m, "finescale": self.finescale}


def why_no_prompt(item: Item) -> str:
    """Why a paper carries no prompt: its case drifted, or it has no usable text."""
    if item.ctx is None:
        return "context drifted"
    return "unresolved" if item.paper is None else "no abstract"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def band_items(head: dict[str, str]) -> list[Item]:
    """The 2026-08-20 score-2 band, each paper with its frozen-pool record and stored finescale."""
    import second_judge_band as sjb

    resolved, missing = sjb.resolve_abstracts([dict(r) for r in read_json(BAND_FILE)["rows"]])
    if missing:
        raise SystemExit(f"{len(missing)} band papers are absent from their frozen pool")
    out = []
    for r in resolved:
        g, s, case = int(r["gpt_score"]), int(r["sonnet_score"]), r["case"]
        item = Item("band", case, str(r["arxiv_id"]), g, s, head.get(case), r["paper"])
        item.finescale = float(r["finescale"])
        out.append(item)
    return out


def comparison_items(
    ship: dict[str, list[dict[str, Any]]],
    opus: dict[str, list[dict[str, Any]]],
    cases: Sequence[str],
    head: dict[str, str],
    sonnet: dict[tuple[str, str], int],
    arxiv_papers: dict[str, dict[str, dict[str, Any]]],
    doi_records: dict[str, dict[str, dict[str, Any]]],
    *,
    meta: Callable[[str], dict[str, dict[str, Any]]] = r1.pool_metadata,
    gold: Path = sj.GOLD,
    multi: dict[tuple[str, str], str] = sip.MULTI_VERSION,
) -> tuple[list[Item], list[Item]]:
    """NR-52's two arms as the Sonnet path built them, with the baseline's ids of NR-67.

    Ours: each shown pick with its pool-cut100 abstract. The baseline's arXiv picks: the versioned
    id GPT-5.5 saw, found by `sonnet_id_probe.build_population`, with the text stored in
    `.work/sonnet_id_probe/papers.json`. Its DOI picks: the stored resolution.
    """
    ours: list[Item] = []
    for case in cases:
        m = meta(case)
        for p in ship.get(case, []):
            pid = str(p["arxiv_id"])
            paper = {**p, "abstract": (m.get(pid) or {}).get("abstract", "")}
            g, s = int(p["judge_score"]), r1.son_of(sonnet, case, pid)
            ours.append(Item("ours", case, pid, g, s, head.get(case), paper))
    population = sip.build_population(opus, cases, sonnet, gold=gold, multi=multi)
    rows = {(r["case"], r["pick"]): r for r in population}
    base: list[Item] = []
    for case in cases:
        for p in opus.get(case, []):
            pick = str(p["arxiv_id"])
            kind = "arxiv" if is_arxiv_id(pick) else "doi"
            paper: dict[str, Any] | None = None
            if kind == "arxiv":
                rec = sip.record_for(arxiv_papers, rows[case, pick])
                if rec.get("status") == "ok":
                    paper = sip.arm_paper(rows[case, pick], rec, "v")
            else:
                rec = (doi_records.get(case) or {}).get(pick) or {}
                if rec.get("status") == "ok":
                    paper = {"title": rec["title"], "abstract": rec["abstract"], "arxiv_id": pick}
            g, s = int(p["judge_score"]), r1.son_of(sonnet, case, pick)
            base.append(Item("baseline", case, pick, g, s, head.get(case), paper, kind))
    return ours, base


def adoption_items() -> tuple[list[Item], list[Item], list[str]]:
    """NR-61's adoptions and NR-62's distinct cross-repository pairs, at each T0 context.

    Shaped by `judge_validity_pool.judgeable_items`, the function both judges' items went through,
    with the positives' texts read from the cache it fetched them into. Also returns every stored
    verdict whose context digest is not today's T0 digest.
    """
    import judge_validity_adoption as jva
    import judge_validity_pool as jvp

    with contextlib.redirect_stdout(io.StringIO()):
        positives = jvp.analysis_set(jvp.pool_seed(verify=False))["positives"]
    drawn = read_json(jvp.XREPO_ROWS)
    if drawn["seed"] != jvp.xrepo_seed(verify=False):
        raise SystemExit(f"{jvp.XREPO_ROWS.name} was drawn under another seed")
    payload = read_json(jvp.XREPO_PAYLOAD)["papers"]
    controls = [{**r, "paper": payload[r["id"]]} for r in drawn["controls"]]
    texts = read_json(TRIAGE_PAPERS)
    shaped, missing = jvp.judgeable_items(positives, controls, fetch=lambda ids: texts)
    if missing:
        raise SystemExit(f"{len(missing)} adopted papers have no stored text: {missing[:5]}")
    t0 = jvp.pool_contexts(positives)
    store = read_json(jvp.POOL_VERDICTS)
    drift: list[str] = []

    def score(model: str, case: str, pid: str) -> int:
        key = jvp.verdict_key(model, case, pid)
        if key not in store:
            raise SystemExit(f"no stored verdict {key}")
        if case not in t0 or str(store[key].get("context_digest")) != t0[case][1]:
            drift.append(key)
        return int(store[key]["score"])

    out: dict[str, list[Item]] = {"adopted": [], "crossrepo": []}
    seen: set[tuple[str, str, str]] = set()
    for it in shaped:
        pop = "adopted" if it["arm"] == "adopted" else "crossrepo"
        case, pid = str(it["case"]), str(it["arxiv_id"])
        if (pop, case, pid) in seen:
            continue  # a control drawn for two positives of one repository is one pair
        seen.add((pop, case, pid))
        g, s = score(jva.GPT_MODEL, case, pid), score(jva.SONNET_MODEL, case, pid)
        out[pop].append(Item(pop, case, pid, g, s, t0[case][0] if case in t0 else None, it))
    return out["adopted"], out["crossrepo"], drift


@dataclass
class Loaded:
    items: dict[str, list[Item]]
    cases: list[str]  # the comparison's cases whose context reproduces
    drifted: list[str]  # HEAD cases whose context no longer reproduces its stored hash
    t0_drift: list[str]
    ship: dict[str, list[dict[str, Any]]]
    opus: dict[str, list[dict[str, Any]]]
    sonnet: dict[tuple[str, str], int]
    doi: dict[str, dict[str, dict[str, Any]]]


def load(*, resolve: bool = False) -> Loaded:
    """Every population and its prompts. With *resolve*, unresolved DOI picks are resolved."""
    ship, opus = r1.shipped_arm(), r1.opus5_arm()
    both = sorted(set(ship) & set(opus))
    band_cases = {str(r["case"]) for r in read_json(BAND_FILE)["rows"]}
    head, drifted = sj.verify_contexts(sorted(set(both) | band_cases))
    sonnet = r1.cached_sonnet()
    doi = sip.load_papers(DOI_FILE)
    if resolve:
        note = "The baseline's DOI picks, resolved once through verify.resolve_references."
        ids = [(c, str(p["arxiv_id"])) for c in both for p in opus[c]]
        picks = [{"case": c, "pick": i} for c, i in ids if not is_arxiv_id(i)]

        def store(recs: dict[str, dict[str, dict[str, Any]]]) -> None:
            sip._write_json(DOI_FILE, {"_comment": note, "papers": recs})

        doi = sip.resolve_missing(picks, doi, save=store)
    ours, base = comparison_items(ship, opus, both, head, sonnet, sip.load_papers(), doi)
    adopted, crossrepo, t0_drift = adoption_items()
    items = dict(zip(POPULATIONS, (band_items(head), ours, base, adopted, crossrepo), strict=True))
    for rows in items.values():
        rows.sort(key=lambda i: (i.case, i.id))
    return Loaded(
        items, [c for c in both if c in head], sorted(drifted), t0_drift, ship, opus, sonnet, doi
    )


def distinct_prompts(loaded: Loaded) -> tuple[dict[str, str], dict[str, list[dict[str, Any]]]]:
    """Each distinct prompt once, in population order, with every membership it serves."""
    prompts: dict[str, str] = {}
    members: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for pop in POPULATIONS:
        for i in loaded.items[pop]:
            if i.sha is not None and i.prompt is not None:
                prompts.setdefault(i.sha, i.prompt)
                members[i.sha].append(i.member())
    return prompts, dict(members)


# ── Reproductions ─────────────────────────────────────────────────────────────────────────


def nr52_margin(loaded: Loaded, label: str) -> float:
    """NR-52's margin, recomputed through `rung1_second_judge` from its own labels."""
    fn, son = r1.LABELS[label], loaded.sonnet
    nets = (
        r1.net2(loaded.ship[c], fn, son, c)[0] - r1.net2(loaded.opus[c], fn, son, c)[0]
        for c in loaded.cases
    )
    return st.mean(nets)


def reproductions(loaded: Loaded) -> list[dict[str, Any]]:
    """The registered figures of the two existing judges, from their labels, to the quoted dp."""

    def act(items: Sequence[Item], judge: str) -> list[bool]:
        return [(getattr(i, judge) or 0) >= ACTIONABLE for i in items]

    def share(items: Sequence[Item], judge: str) -> float:
        return sum(act(items, judge)) / len(items)

    def finescale_auc(items: Sequence[Item], judge: str) -> float:
        return auc([float(i.finescale or 0.0) for i in items], act(items, judge))

    band, ad, xr = loaded.items["band"], loaded.items["adopted"], loaded.items["crossrepo"]
    figures = [
        ("NR-66 band actionable, GPT-5.5", share(band, "gpt"), 0.873, 3),
        ("NR-66 band actionable, Sonnet", share(band, "sonnet"), 0.494, 3),
        ("NR-66 band finescale AUC, GPT-5.5", finescale_auc(band, "gpt"), 0.729, 3),
        ("NR-66 band finescale AUC, Sonnet", finescale_auc(band, "sonnet"), 0.702, 3),
        ("NR-52 margin, gpt", nr52_margin(loaded, "gpt"), 0.32, 2),
        ("NR-52 margin, sonnet_only", nr52_margin(loaded, "sonnet_only"), -3.41, 2),
        ("NR-61 adopted actionable, GPT-5.5", share(ad, "gpt"), 0.819, 3),
        ("NR-61 adopted actionable, Sonnet", share(ad, "sonnet"), 0.644, 3),
        ("NR-62 cross-repository, GPT-5.5", share(xr, "gpt"), 0.255, 3),
        ("NR-62 cross-repository, Sonnet", share(xr, "sonnet"), 0.068, 3),
    ]
    return [
        {"figure": n, "got": g, "want": w, "dp": dp, "ok": g == g and round(g, dp) == w}
        for n, g, w, dp in figures
    ]


# ── Endpoints and readings ────────────────────────────────────────────────────────────────


def level_reading(ci: Sequence[float], gpt: float, sonnet: float) -> str:
    """The registered level reading of an interval against the reference pair, first match."""
    lo, hi = ci
    has_g, has_s = lo <= gpt <= hi, lo <= sonnet <= hi
    if has_g and has_s:
        return "overlaps both"
    if has_g or has_s:
        return "overlaps GPT-5.5" if has_g else "overlaps Sonnet"
    if min(gpt, sonnet) < lo and hi < max(gpt, sonnet):
        return "between"
    return "above both" if lo > max(gpt, sonnet) else "below both"


def order_reading(ci: Sequence[float] | None) -> str:
    """E2's reading. No interval is not evidence that it does not order the band."""
    if ci is None:
        return "no interval"
    return "orders the band" if ci[0] > 0.5 else "does not order the band"


def margin_reading(margin: float, refs: tuple[float, float] = MARGINS) -> str:
    if min(refs) <= margin <= max(refs):
        return "between"
    return "above both" if margin > max(refs) else "below both"


def unreadable(n_void: int, n: int) -> bool:
    return n > 0 and n_void / n > VOID_MAX


def void_stats(members: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """One population's void count, rate, causes, and whether that alone makes it Unreadable."""
    void = [m for m in members if m["t"] is None]
    return {
        "n": len(members),
        "void": len(void),
        "rate": len(void) / len(members) if members else None,
        "by_cause": dict(sorted(Counter(str(m["void"]) for m in void).items())),
        "unreadable": unreadable(len(void), len(members)),
    }


def arms_unreadable(arms: dict[str, dict[str, Any]]) -> bool:
    """The comparison's void rule, which is per arm.

    A margin is a difference of two arms, so a void rate that falls on one arm alone moves it
    mechanically: every unscored pick simply leaves that arm's net@2. Pooling the two arms hides
    exactly that. So the comparison is Unreadable when either arm is over the 5% limit on its own,
    and also when the two arms' void rates are more than `VOID_GAP` apart, however low both are.
    """
    if any(a["unreadable"] for a in arms.values()):
        return True
    rates = [a["rate"] for a in arms.values() if a["rate"] is not None]
    return len(rates) > 1 and max(rates) - min(rates) > VOID_GAP


def rate(rows: Sequence[dict[str, Any]]) -> float | None:
    return sum(1 for r in rows if r["t"] >= ACTIONABLE) / len(rows) if rows else None


def band_auc(rows: Sequence[dict[str, Any]]) -> float:
    """E2: the stored finescale expectation against Gemini's labels."""
    return auc([float(r["finescale"]) for r in rows], [r["t"] >= ACTIONABLE for r in rows])


def adoption_auc(rows: Sequence[dict[str, Any]]) -> float:
    """E3: Gemini's score for the adopted papers against the cross-repository controls."""
    return auc([float(r["t"]) for r in rows], [r["population"] == "adopted" for r in rows])


def repo_bootstrap(
    rows: Sequence[dict[str, Any]], stat: Callable[[list[dict[str, Any]]], float | None]
) -> list[float] | None:
    """Percentile interval over 10,000 draws that resample repositories, not papers.

    One generator seeded 20260923 per endpoint. Each draw takes as many repositories as there are,
    with replacement, and pools their papers. None when any draw leaves *stat* undefined.
    """
    by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_case[r["case"]].append(r)
    repos = sorted(by_case)
    if not repos:
        return None
    rng, k = random.Random(BOOT_SEED), len(repos)
    values: list[float] = []
    for _ in range(BOOT_DRAWS):
        v = stat([r for _ in range(k) for r in by_case[repos[rng.randrange(k)]]])
        if v is None or v != v:
            return None
        values.append(v)
    values.sort()
    return [values[BOOT_LO], values[BOOT_HI]]


def gemini_at_2(g: int, t: int | None) -> bool | None:
    """E4's label: actionable when Gemini scores 2 or more, void when it has no verdict."""
    return None if t is None else t >= ACTIONABLE


def arm_net(members: Sequence[dict[str, Any]], case: str) -> int:
    """One arm's net@2 in one case under that label, through NR-52's own `net2`."""
    picks = [{"arxiv_id": m["id"], "judge_score": m["gpt"]} for m in members]
    labels = {(case, sj.safe_paper_id(m["id"])): m["t"] for m in members if m["t"] is not None}
    return r1.net2(picks, gemini_at_2, labels, case)[0]


def members_of(rows: Sequence[dict[str, Any]], population: str) -> list[dict[str, Any]]:
    """Every membership of *population*, with Gemini's score as `t` and the void cause."""
    return [
        {**m, "t": r["score"], "void": r["void"]}
        for r in rows
        for m in r["members"]
        if m["population"] == population
    ]


def level(rows: list[dict[str, Any]], ref: tuple[float, float], bad: bool) -> dict[str, Any]:
    """One actionable-share endpoint: the rate, its interval and the registered reading."""
    ci = repo_bootstrap(rows, rate)
    reading = "no interval" if ci is None else level_reading(ci, *ref)
    return {
        "n_scored": len(rows),
        "rate": rate(rows),
        "ci": ci,
        "reference": {"gpt-5.5": ref[0], "sonnet": ref[1]},
        "reading": "Unreadable" if bad else reading,
    }


def void_bias(members: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Descriptive only: what the two existing judges made of the void members and the scored ones.

    A void that lands on the papers the other judges called actionable is not the same loss as one
    that lands anywhere. This says which it was. It enters no reading and no prediction.
    """

    def share(ms: Sequence[dict[str, Any]], judge: str) -> float | None:
        return sum((m.get(judge) or 0) >= ACTIONABLE for m in ms) / len(ms) if ms else None

    out: dict[str, Any] = {}
    for label, ms in (
        ("void", [m for m in members if m["t"] is None]),
        ("scored", [m for m in members if m["t"] is not None]),
    ):
        out[label] = {"n": len(ms), **{j: share(ms, j) for j in ("gpt", "sonnet")}}
    return out


def summarise(data: dict[str, Any]) -> dict[str, Any]:
    """Every registered endpoint and reading, from the per-prompt rows alone."""
    pop = {p: members_of(data["rows"], p) for p in POPULATIONS}
    voids: dict[str, Any] = {}
    for name, parts in REGISTERED.items():
        v = void_stats([m for p in parts for m in pop[p]])
        if len(parts) > 1:  # the comparison, whose rule is per arm
            v["by_arm"] = {p: void_stats(pop[p]) for p in parts}
            v["unreadable"] = arms_unreadable(v["by_arm"])
        voids[name] = v
    bad = {name: v["unreadable"] for name, v in voids.items()}
    scored = {p: [m for m in pop[p] if m["t"] is not None] for p in POPULATIONS}
    band, adoption = scored["band"], scored["adopted"] + scored["crossrepo"]
    e2_ci = repo_bootstrap(band, band_auc)
    cases = list(data["comparison_cases"])

    def arm(name: str, case: str) -> tuple[int, int]:
        here = [m for m in pop[name] if m["case"] == case]
        return arm_net(here, case), sum(1 for m in here if m["t"] is not None)

    per_case: dict[str, Any] = {}
    for c in cases:
        (o_net, o_n), (b_net, b_n) = arm("ours", c), arm("baseline", c)
        per_case[c] = {
            "ours": o_net,
            "baseline": b_net,
            "delta": o_net - b_net,
            "scored": {"ours": o_n, "baseline": b_n},
        }
    deltas = [float(v["delta"]) for v in per_case.values()]
    margin = st.mean(deltas) if deltas else None
    summary: dict[str, Any] = {
        "voids": voids,
        "void_bias": {p: void_bias(pop[p]) for p in POPULATIONS},
        "E1": level(band, REFERENCE["band"], bad["band"]),
        "E2": {
            "auc": band_auc(band) if band else None,
            "ci": e2_ci,
            "reading": "Unreadable" if bad["band"] else order_reading(e2_ci),
        },
        "E3": {p: level(scored[p], REFERENCE[p], bad[p]) for p in E3_RATES},
        "E4": {
            "n_cases": len(cases),
            "margin": margin,
            "ci": list(paired_bootstrap(deltas)) if deltas else None,
            "reading": "Unreadable"
            if bad["comparison"]
            else ("no cases" if margin is None else margin_reading(margin)),
            "precision": {a: rate(scored[a]) for a in ("ours", "baseline")},
            "per_case": per_case,
        },
    }
    auc_ci = repo_bootstrap(adoption, adoption_auc)
    summary["E3"]["auc"] = {
        "auc": adoption_auc(adoption) if adoption else None,
        "ci": auc_ci,
        # It pools both adoption populations, so either one being Unreadable unreads it.
        "unreadable": bad["adopted"] or bad["crossrepo"],
    }
    summary["prediction"] = prediction_check(summary)
    return summary


def prediction_check(s: dict[str, Any]) -> dict[str, Any]:
    """The registered predictions against the point estimates and the readings.

    An endpoint whose population is Unreadable is not scored against the prediction: its result is
    null and carries an `unreadable` flag. Reading it would let a void rate decide a hit.
    """
    e3 = s["E3"]
    blocks = (s["E1"], s["E2"], e3["adopted"], e3["crossrepo"], s["E4"])
    values = [b.get("rate", b.get("auc", b.get("margin"))) for b in blocks]
    dead = [b["reading"] == "Unreadable" for b in blocks]
    values.insert(4, e3["auc"]["auc"])  # the adopted-against-control AUC, which has no reading
    dead.insert(4, bool(e3["auc"]["unreadable"]))
    out: dict[str, Any] = {}
    for (name, (lo, hi)), v, bad in zip(PREDICTION.items(), values, dead, strict=True):
        if bad:
            out[name] = {"value": None, "range": [lo, hi], "within": None, "unreadable": True}
            continue
        within = v is not None and (lo is None or v >= lo) and (hi is None or v <= hi)
        out[name] = {"value": v, "range": [lo, hi], "within": within, "unreadable": False}
    worst = max((v["rate"] for v in s["voids"].values() if v["rate"] is not None), default=None)
    within = worst is not None and worst < VOID_PREDICTION
    out["voids under 2% in every population"] = {"value": worst, "within": within}
    out["readings"] = {
        k: {
            "mode": w["mode"],
            "point": w["point"],
            "actual": b["reading"],
            "as_predicted": b["reading"] == w["mode"],
            "probability": w["p"].get(b["reading"], w["p"]["other"]),
        }
        for (k, w), b in zip(PREDICTED_READING.items(), blocks, strict=True)
    }
    return out


# ── Buying ────────────────────────────────────────────────────────────────────────────────


class Stop(Exception):
    """The run stops. Nothing that raises this is charged to a prompt."""


def cost(usage: dict[str, Any]) -> float:
    """USD from one response's usageMetadata at the registered prices."""
    out = int(usage.get("candidatesTokenCount") or 0) + int(usage.get("thoughtsTokenCount") or 0)
    return (int(usage.get("promptTokenCount") or 0) * PRICE_IN + out * PRICE_OUT) / 1e6


def worst_cost(prompt: str) -> float:
    """The most one call can cost: every prompt byte a token, and the whole output limit."""
    return (len(prompt.encode("utf-8")) * PRICE_IN + MAX_OUTPUT_TOKENS * PRICE_OUT) / 1e6


def cache_path(sha: str, state: Path = STATE) -> Path:
    return state / MODEL / f"{sha}.json"


def read_cache(sha: str, state: Path = STATE) -> dict[str, Any] | None:
    p = cache_path(sha, state)
    return read_json(p) if p.is_file() else None


def load_ledger(state: Path | None = None) -> dict[str, Any]:
    """The ledger, with every field a ledger written before this version may be missing."""
    state = STATE if state is None else state
    fields: dict[str, Any] = {
        "model": MODEL,
        "calls": 0,  # attempts that count against CALL_CAP; a 429 is never one of them
        "rate_limited": 0,  # attempts the API answered 429, counted apart from `calls`
        "rate_streak": 0,  # consecutive 429s with no verdict between them
        "spend_usd": 0.0,
        "failures": {},
        "streak": [],
        "stops": [],
    }
    p = state / "ledger.json"
    return fields | dict(read_json(p)) if p.is_file() else fields


class Buyer:
    """Asks each prompt through the registered retry rule, under the caps, into the ledger.

    A prompt ends with a cache record: a score, or a void and its cause. Failure charges live in
    the ledger, so a resumed run continues each prompt's retry budget where it stopped.
    """

    def __init__(
        self,
        prompts: dict[str, str],
        members: dict[str, list[dict[str, Any]]],
        send: Callable[[dict[str, Any]], gc.Outcome],
        *,
        state: Path = STATE,
        sleep: Callable[[float], None] = time.sleep,
        spend_cap: float = SPEND_CAP,
        call_cap: int = CALL_CAP,
        log: Callable[[str], None] = print,
    ) -> None:
        self.prompts, self.members, self.send = prompts, members, send
        self.state, self.sleep, self.log = state, sleep, log
        self.spend_cap, self.call_cap = spend_cap, call_cap
        self.ledger = load_ledger(state)
        self.lock = threading.Lock()
        self.reserved = 0.0
        self.halted: str | None = None
        self.opening_spend = float(self.ledger["spend_usd"])

    def _save(self) -> None:
        sip._write_json(self.state / "ledger.json", self.ledger)

    def _halt(self, why: str) -> None:
        """Called with the lock held."""
        self.halted = self.halted or why
        self._save()
        raise Stop(self.halted)

    def _write(self, sha: str, score: int | None, void: str | None, o: gc.Outcome) -> None:
        charges = self.ledger["failures"].get(sha, {})
        record = {"prompt_sha256": sha, "model": MODEL, "populations": self.members.get(sha, [])}
        record.update(score=score, void=void, text=o.text, finish_reason=o.finish_reason)
        record.update(block_reason=o.block_reason, model_version=o.model_version, usage=o.usage)
        record.update(
            http_status=o.http_status,
            error=o.error,
            at=datetime.now(tz=UTC).isoformat(timespec="seconds"),
        )
        record["failures"] = {k: v for k, v in charges.items() if k != "last"}
        sip._write_json(cache_path(sha, self.state), record)

    def _spent(self, sha: str) -> bool:
        f = self.ledger["failures"].get(sha) or {}
        return f.get("transport", 0) > TRANSPORT_RETRIES or f.get("content", 0) > CONTENT_RETRIES

    def _settled(self, sha: str) -> bool:
        with self.lock:
            if cache_path(sha, self.state).is_file():
                return True
            if self._spent(sha):  # a stop between the charge and the record
                cause = str(self.ledger["failures"][sha].get("last", "transport"))
                self._write(sha, None, cause, gc.Outcome(cause, None))
                return True
            return False

    def _begin(self, prompt: str) -> float:
        worst = worst_cost(prompt)
        with self.lock:
            if self.halted:
                raise Stop(self.halted)
            if self.ledger["calls"] >= self.call_cap:
                self._halt(f"call cap: call {self.call_cap + 1} refused")
            if self.ledger["spend_usd"] + self.reserved + worst > self.spend_cap:
                spent = self.ledger["spend_usd"]
                self._halt(f"spend cap: ${spent:.2f} spent; the next call could pass the cap")
            self.ledger["calls"] += 1
            self.reserved += worst
            self._save()
        return worst

    def _end(self, worst: float, o: gc.Outcome) -> None:
        with self.lock:
            self.reserved -= worst
            self.ledger["spend_usd"] += cost(o.usage)
            self._save()

    def _fail(self, sha: str, kind: str, cause: str, o: gc.Outcome) -> bool:
        """Charge one failure; True when the prompt is now void. Twenty in a row is an outage."""
        with self.lock:
            if self.halted:  # the run is stopping; a call still in flight is not charged
                raise Stop(self.halted)
            f = self.ledger["failures"].setdefault(sha, {"transport": 0, "content": 0})
            f[kind] += 1
            f["last"] = cause
            self.ledger["streak"].append([sha, kind])
            if len(self.ledger["streak"]) >= OUTAGE_AFTER:
                for s, k in self.ledger["streak"]:
                    self.ledger["failures"][s][k] -= 1
                    rec = read_cache(s, self.state)
                    if rec is not None and rec["score"] is None:
                        cache_path(s, self.state).unlink()
                self.ledger["streak"] = []
                self._halt(f"outage: {OUTAGE_AFTER} failures in a row, refunded to their prompts")
            void = self._spent(sha)
            if void:
                self._write(sha, None, cause, o)
            self._save()
            return void

    def _rate_limited(self) -> None:
        """Give the call back. A 429 bought nothing, so it may not spend the 3,000-call budget.

        Counting 429s against the cap would let a throttled hour end the run with most prompts
        unasked, which is not what the cap is for. They are counted in their own field instead.
        Twenty in a row with no verdict between them is not throttling to wait out: it is a tier
        or a quota the run cannot talk its way past, so the run stops and can be resumed.
        """
        with self.lock:
            self.ledger["calls"] -= 1
            self.ledger["rate_limited"] += 1
            self.ledger["rate_streak"] += 1
            if self.ledger["rate_streak"] >= RATE_LIMIT_STOP:
                self.ledger["rate_streak"] = 0
                self._halt(
                    f"rate limit: {RATE_LIMIT_STOP} rate-limited calls in a row with no verdict "
                    "between them; the key's project is likely on the free tier or out of quota"
                )
            self._save()

    def ask(self, sha: str) -> None:
        prompt = self.prompts[sha]
        while not self._settled(sha):
            worst = self._begin(prompt)
            try:
                o = self.send(request_body(prompt))
            except Exception as exc:  # noqa: BLE001 -- the client never raises; a fake might
                o = gc.Outcome(gc.TRANSPORT, None, error=type(exc).__name__)
            self._end(worst, o)
            cls, score = classify(o)
            if cls == "scored":
                with self.lock:
                    self._write(sha, score, None, o)
                    self.ledger["streak"] = []
                    self.ledger["rate_streak"] = 0
                    self._save()
                return
            if cls == "rate_limit":  # not charged; waits as long as the error asks
                self._rate_limited()
                self.sleep(o.retry_delay if o.retry_delay is not None else RATE_WAIT)
                continue
            if cls in ("key", "fatal"):
                with self.lock:
                    self._halt(f"{cls} refusal, HTTP {o.http_status}: {o.error[:120]}")
            kind = "transport" if cls == "transport" else "content"
            if self._fail(sha, kind, cls, o):
                return
            if kind == "transport":
                self.sleep(min(2.0 ** self.ledger["failures"][sha]["transport"], 30.0))

    def _projected(self, done: int, total: int) -> float:
        """What the run ends at if the prompts still to ask cost what this run's have.

        The cap stops the run, it does not warn before it. A projection on every progress line is
        the warning: a per-prompt cost above the plan's estimate shows up in the first few lines,
        while there is still a run to abandon, instead of at the call the cap refuses.
        """
        here = self.ledger["spend_usd"] - self.opening_spend
        return self.ledger["spend_usd"] + (here / done * (total - done) if done else 0.0)

    def run(self, shas: Sequence[str], workers: int = WORKERS) -> str | None:
        """Every prompt in *shas* not yet settled, up to *workers* at once. Why it stopped."""
        todo = [s for s in shas if read_cache(s, self.state) is None]
        self.log(f"{len(todo)} of {len(shas)} prompts to ask, {workers} workers")
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(self.ask, s) for s in todo]
            for n, fut in enumerate(as_completed(futures), start=1):
                with contextlib.suppress(Stop):
                    fut.result()
                if n % 50 == 0 or n == len(futures):
                    calls, spend = self.ledger["calls"], self.ledger["spend_usd"]
                    self.log(
                        f"  {n}/{len(futures)}  calls {calls}  spend ${spend:.2f}  "
                        f"projected total ${self._projected(n, len(futures)):.2f}"
                    )
        if self.halted:
            with self.lock:
                self.ledger["stops"].append(
                    {"at": datetime.now(tz=UTC).isoformat(), "why": self.halted}
                )
                self._save()
        return self.halted


# ── Checks before a paid call ─────────────────────────────────────────────────────────────


def _git(cwd: Path, *args: str) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(["git", "-C", str(cwd), *args], capture_output=True)


def prereg_refusal(cwd: Path = ROOT, paths: Sequence[str] = FROZEN) -> list[str]:
    """Why any of the frozen files is not committed and unchanged from HEAD. Empty when all are.

    The registration alone is not the specification a reading rests on. The reference pairs, the
    void thresholds, the prediction and the reading functions are in `third_judge.py`, and the
    request the verdicts answer is assembled by `gemini_client.py`. An uncommitted edit to either
    can change a reading as freely as an edit to the registration, so all three are frozen.
    """
    out: list[str] = []
    for rel in paths:
        if _git(cwd, "cat-file", "-e", f"HEAD:{rel}").returncode:
            out.append(f"{rel} is not committed")
        elif _git(cwd, "diff", "--quiet", "HEAD", "--", rel).returncode:
            out.append(f"{rel} differs from HEAD")
    return out


def frozen_shas(cwd: Path = ROOT, paths: Sequence[str] = FROZEN) -> dict[str, Any]:
    """The HEAD commit and each frozen file's blob sha, recorded beside the readings."""

    def rev(spec: str) -> str | None:
        p = _git(cwd, "rev-parse", spec)
        return p.stdout.decode().strip() if p.returncode == 0 else None

    return {"head": rev("HEAD"), "files": {rel: rev(f"HEAD:{rel}") for rel in paths}}


def api_key(env: Path | None = None) -> str | None:
    """The key from evals/.env. Never printed, never written."""
    env = ENV_FILE if env is None else env
    for line in env.read_text(encoding="utf-8").splitlines() if env.is_file() else []:
        k, sep, v = line.strip().partition("=")
        if sep and k.strip() == KEY_NAME and v.strip().strip("\"'"):
            return v.strip().strip("\"'")
    return None


def case_list(loaded: Loaded) -> list[str]:
    """Every population's cases, the list `--judge` writes into the ledger on its first run."""
    return sorted({f"{i.population}:{i.case}" for rows in loaded.items.values() for i in rows})


Count = Callable[..., tuple[int | None, Any]]


def body_refusal(loaded: Loaded, count: Count = gc.count_tokens) -> str | None:
    """countTokens on one whole registered request body: a 400 means the request is not accepted.

    countTokens is free and takes the same `generateContentRequest` that `generate` is paid for,
    `generationConfig` and `safetySettings` included. Counting only `contents` would validate the
    part that never varies and skip the parts this registration actually chose. A 400 here is a
    refusal, not something to route around with a character estimate: every paid call would carry
    the same body.
    """
    key = api_key()
    prompt = next(
        (i.prompt for p in POPULATIONS for i in loaded.items[p] if i.prompt is not None), None
    )
    if key is None or prompt is None:
        return None
    status, body = count(request_body(prompt), MODEL, key)
    if status != 400:
        return None
    err = body.get("error", {}).get("message") if isinstance(body, dict) else body
    return f"countTokens refuses the registered request body with 400: {str(err)[:200]}"


def readiness(loaded: Loaded, *, key: bool = True, count: Count = gc.count_tokens) -> list[str]:
    """Every reason `--judge` would refuse. `--report` asks the same, without the key check."""
    out = prereg_refusal()
    if key and not api_key():
        out.append(f"no {KEY_NAME} in evals/.env")
    if loaded.drifted:
        out.append(f"{len(loaded.drifted)} HEAD contexts drifted: {loaded.drifted}")
    if loaded.t0_drift:
        out.append(f"{len(loaded.t0_drift)} stored verdicts are on another T0 context")
    if not DOI_FILE.is_file():
        out.append("the DOI picks are not resolved yet (run --plan)")
    n = {p: len(loaded.items[p]) for p in POPULATIONS}
    out += [f"{p} has {n[p]} papers, registered {w}" for p, w in EXPECTED.items() if n[p] != w]
    if len(loaded.cases) != N_CASES:
        out.append(f"the comparison has {len(loaded.cases)} cases, registered {N_CASES}")
    for p in POPULATIONS:
        # A paper the run cannot ask is a hole in a registered population, not a void to report:
        # it never reached the judge, so no retry rule ever applied to it.
        gone = [i for i in loaded.items[p] if i.prompt is None]
        if gone:
            why = dict(sorted(Counter(why_no_prompt(i) for i in gone).items()))
            out.append(f"{p} has {len(gone)} papers with no prompt: {why}")
    cases = case_list(loaded)
    if load_ledger().get("cases", cases) != cases:
        out.append("the case list differs from the one the ledger started with")
    if why := body_refusal(loaded, count):
        out.append(why)
    return out


# ── Modes ─────────────────────────────────────────────────────────────────────────────────


def sonnet_path_prompt(case: str, ctx: str, paper: dict[str, Any]) -> str:
    """What `second_judge.second_verdict` would send, captured with no call and no write."""
    sent: list[str] = []

    class Captured(Exception):
        pass

    def stub(prompt: str, cfg: Any, **_: Any) -> str:
        sent.append(prompt)
        raise Captured

    saved = sj.CACHE, sj.complete
    with tempfile.TemporaryDirectory() as tmp:
        sj.CACHE, sj.complete = Path(tmp), stub  # type: ignore[assignment]
        try:
            sj.second_verdict(case, ctx, paper, sj.DEFAULT_MODEL)
        except Captured:
            pass
        finally:
            sj.CACHE, sj.complete = saved
    return sent[0]


def sonnet_paper(item: Item) -> dict[str, Any] | None:
    """The paper as the existing Sonnet verdict saw it, or None when its text was never kept."""
    if item.population != "baseline":
        return item.paper
    if item.kind == "doi":
        return None  # NR-52 resolved these afresh and stored no text
    return {**(item.paper or {}), "arxiv_id": item.id}  # NR-52 sent the unversioned pick


def no_sonnet_text(loaded: Loaded) -> dict[str, int]:
    """Per population, papers whose Sonnet-side prompt text was never stored.

    Only the baseline's DOI picks: NR-52 resolved them at the time and kept the verdict, not the
    text. Their prompts match the Sonnet path in construction and cannot be shown to match it byte
    for byte, and E4, which is built on the baseline arm, inherits that.
    """
    return {
        p: sum(1 for i in loaded.items[p] if i.prompt is not None and sonnet_paper(i) is None)
        for p in POPULATIONS
    }


def sonnet_verdict_files(loaded: Loaded, model: str = sj.DEFAULT_MODEL) -> dict[str, list[int]]:
    """Per population, [checked, found]: prompts whose paper id names a stored Sonnet verdict.

    The prompt carries a paper id, and the second judge's cache is filed by paper id. So the file
    either is there under the id this prompt sends or it is not, and that is checkable without a
    call. Where it is not, the two judges were not asked about the same id: on the baseline's
    arXiv picks that is the versioned identifier of NR-67, which the registration expects.
    """
    out: dict[str, list[int]] = {}
    for pop in POPULATIONS:
        checked = found = 0
        for i in loaded.items[pop]:
            if i.prompt is None or i.paper is None:
                continue
            checked += 1
            found += sj.second_cache_path(model, i.case, str(i.paper.get("arxiv_id"))).is_file()
        out[pop] = [checked, found]
    return out


def print_reproductions(repro: list[dict[str, Any]]) -> bool:
    for r in repro:
        mark = "ok" if r["ok"] else "DOES NOT REPRODUCE"
        print(f"  {r['figure']:<38} {r['got']:+.4f}  registered {r['want']:+.{r['dp']}f}  {mark}")
    return all(r["ok"] for r in repro)


def estimate_input_tokens(sample: list[str], total_chars: int) -> tuple[float, str]:
    """Input tokens over every distinct prompt, from free countTokens calls when they work.

    Each call sends the whole registered body, `generationConfig` and `safetySettings` included,
    which is what `generate` will be paid for. A 400 is a refusal `readiness` reports; the
    character estimate below is only for a countTokens that could not be reached at all.
    """
    key, counted, chars, why = api_key(), 0, 0, f"no {KEY_NAME}"
    for p in sample if key else []:
        status, body = gc.count_tokens(request_body(p), MODEL, key)
        if status != 200 or not isinstance(body, dict) or "totalTokens" not in body:
            err = body.get("error", {}).get("message") if isinstance(body, dict) else body
            why, counted = f"countTokens answered {status}: {str(err)[:100]}", 0
            break
        counted, chars = counted + int(body["totalTokens"]), chars + len(p)
    if counted:
        how = f"countTokens on {len(sample)} prompts: {counted:,} tokens over {chars:,} characters"
        return total_chars * counted / chars, f"{how}, {chars / counted:.2f} per token"
    return total_chars / CHARS_PER_TOKEN, f"{why}; {CHARS_PER_TOKEN} characters per token assumed"


def print_samples(loaded: Loaded) -> list[Item]:
    """Two prompts per population, each against what the Sonnet path's builder makes of it.

    On the baseline the two are one arXiv pick and one DOI pick, the shapes it is built from.
    The Sonnet side is rebuilt here, not read back from what NR-52 or NR-66 sent, so outside the
    baseline both sides start from the same objects and this compares two assemblies of them.
    """
    picked: list[Item] = []
    for pop in POPULATIONS:
        rows = [i for i in loaded.items[pop] if i.prompt is not None]
        for n, kind in enumerate(("arxiv", "doi") if pop == "baseline" else ("", "")):
            here = [i for i in rows if not kind or i.kind == kind]
            if here:
                picked.append(here[n * len(here) // 2])
    for i in picked:
        assert i.prompt is not None and i.ctx is not None
        print(f"  {i.population}{'/' + i.kind if i.kind else ''}  {i.case}/{i.id}")
        print(f"    {len(i.prompt.encode('utf-8')):,} bytes, sha256 {i.sha}")
        theirs = sonnet_paper(i)
        if theirs is None:
            print("    Sonnet text was never stored: nothing to compare")
            continue
        lines = difflib.unified_diff(
            sonnet_path_prompt(i.case, i.ctx, theirs).split("\n"), i.prompt.split("\n"), n=0
        )
        diff = [x.rstrip("\n") for x in lines if not x.startswith(("---", "+++", "@@"))]
        print("    Sonnet builder: identical" if not diff else "    Sonnet (-) against this (+):")
        for line in diff[:6]:
            print(f"      {line[:96]}")
    return picked


def plan(resolve: bool) -> int:
    print(f"third judge: {MODEL}, thinkingLevel {LEVEL}, maxOutputTokens ", end="")
    print(f"{MAX_OUTPUT_TOKENS}, four safety categories at BLOCK_NONE. No paid call is made.\n")
    loaded = load(resolve=resolve)
    it = loaded.items
    prompts, members = distinct_prompts(loaded)

    print(f"{'population':<20}{'papers':>8}{'registered':>12}{'prompts':>9}{'no prompt':>11}")
    for pop in POPULATIONS:
        rows = it[pop]
        n = sum(1 for i in rows if i.prompt is not None)
        why = Counter(why_no_prompt(i) for i in rows if i.prompt is None)
        print(f"  {pop:<18}{len(rows):>8}{EXPECTED[pop]:>12}{n:>9}{len(rows) - n:>11}", end="")
        print(f"  {dict(why)}" if why else "")
    pops = [sorted({m["population"] for m in ms}) for ms in members.values()]
    repeats = sum(len(m) - len(p) for m, p in zip(members.values(), pops, strict=True))
    shared = Counter(" & ".join(p) for p in pops if len(p) > 1)
    print(f"  {'memberships':<18}{sum(len(v) for v in members.values()):>8}")
    print(f"  distinct prompts {len(prompts)}: an identical prompt is asked once")
    print(f"    shared across populations: {dict(sorted(shared.items()))}")
    print(f"    repeated within one population: {repeats}")

    doi = Counter(
        str(((loaded.doi.get(i.case) or {}).get(i.id) or {}).get("status", "not resolved"))
        for i in it["baseline"]
        if i.kind == "doi"
    )
    heads = len({i.case for p in ("band", "ours", "baseline") for i in it[p]})
    t0 = len({i.case for p in E3_RATES for i in it[p]})
    print(f"\nbaseline DOI picks in {DOI_FILE.name}: {dict(sorted(doi.items()))}")
    print(f"\ncontexts\n  HEAD: {heads - len(loaded.drifted)} of {heads} cases reproduce ", end="")
    print(f"their stored hash; drifted: {loaded.drifted or 'none'}")
    print(f"  T0:   {t0} repositories; stored verdicts on another context: {len(loaded.t0_drift)}")
    print(f"  comparison cases: {len(loaded.cases)}")

    print("\nreproductions (--report refuses unless every one holds)")
    ok = print_reproductions(reproductions(loaded))
    print(f"  {'all reproduce' if ok else 'NOT ALL REPRODUCE'}")

    print("\nSonnet-side text")
    gap = no_sonnet_text(loaded)
    print(f"  papers whose Sonnet prompt text was never stored: {gap}")
    print("    Those prompts match the Sonnet path in construction only. E4 reads the baseline")
    print("    arm, so it inherits that.")
    files = sonnet_verdict_files(loaded)
    print(f"  a stored Sonnet verdict under the id this prompt carries ({sj.DEFAULT_MODEL})")
    for pop, (checked, found) in files.items():
        print(f"    {pop:<12} {found:>4} found of {checked:>4} checked")
    print("    band and ours are filed under the id we send; the baseline's arXiv picks are")
    print("    filed under the unversioned pick, not the versioned id of NR-67; the adoption")
    print("    verdicts were filed outside this cache.")

    print("\nsample prompts, rebuilt through the Sonnet path's own prompt builder")
    print("  Both sides are assembled from the same context and paper object for band, ours,")
    print("  adopted and cross-repository, so this shows the two constructions agree, not that")
    print("  the stored Sonnet verdict saw these bytes.")
    picked = print_samples(loaded)

    n, chars = len(prompts), sum(len(p) for p in prompts.values())
    tokens, how = estimate_input_tokens([i.prompt for i in picked if i.prompt], chars)
    cost_in = tokens * PRICE_IN / 1e6
    print(f"\ncost estimate\n  {how}")
    print(f"  input: ~{tokens:,.0f} tokens over {n} distinct prompts, ${cost_in:.2f}")
    for name, out_tokens in COST_SCENARIOS.items():
        out = n * out_tokens * PRICE_OUT / 1e6
        print(f"  output {name} per call: ${out:.2f}; total ${cost_in + out:.2f}")
    print(f"  caps: ${SPEND_CAP:.0f} and {CALL_CAP:,} calls; {n} first calls before any retry")
    break_even = (SPEND_CAP - cost_in) / (n * PRICE_OUT / 1e6) if n else 0.0
    print(f"  the ${SPEND_CAP:.0f} cap binds at {break_even:,.0f} output tokens per call, ", end="")
    print("thinking included, if no call is retried")

    ledger = load_ledger()
    cached = sum(1 for sha in prompts if cache_path(sha).is_file())
    print(f"\ncache: {cached} of {n} prompts settled; ledger: {ledger['calls']} calls, ", end="")
    print(f"${ledger['spend_usd']:.2f}")
    refusals = readiness(loaded)
    print("--judge would refuse:" if refusals else "--judge would run.")
    print("\n".join(f"  - {r}" for r in refusals))
    return 0


def judge() -> int:
    loaded = load()
    refusals = readiness(loaded)
    if refusals:
        print("REFUSED:\n" + "\n".join(f"  - {r}" for r in refusals))
        return 1
    key = str(api_key())
    prompts, members = distinct_prompts(loaded)
    ledger = load_ledger()
    ledger["cases"] = case_list(loaded)  # readiness has already checked it against the ledger's
    sip._write_json(STATE / "ledger.json", ledger)
    buyer = Buyer(prompts, members, lambda body: gc.generate(body, MODEL, key))
    why = buyer.run(list(prompts))
    settled = sum(1 for sha in prompts if cache_path(sha).is_file())
    calls, spend = buyer.ledger["calls"], buyer.ledger["spend_usd"]
    print(f"\n{settled} of {len(prompts)} prompts settled; {calls} calls, ${spend:.2f}")
    print(f"{buyer.ledger['rate_limited']} rate-limited calls, outside the {CALL_CAP:,} cap")
    if why:
        print(f"STOPPED: {why}")
    return 0 if why is None else 2


def build_rows(loaded: Loaded) -> tuple[list[dict[str, Any]], int]:
    """One row per distinct prompt, and one per paper with no prompt. Also the unsettled count."""
    rows: list[dict[str, Any]] = []
    by_sha: dict[str, dict[str, Any]] = {}
    missing = 0
    for pop in POPULATIONS:
        for i in loaded.items[pop]:
            if i.sha is None:
                rows.append({"sha256": None, "score": None, "void": "no prompt"})
                rows[-1]["members"] = [i.member()]
                continue
            if i.sha not in by_sha:
                rec = read_cache(i.sha)
                missing += rec is None
                keys = ("score", "void", "finish_reason", "model_version", "usage")
                by_sha[i.sha] = {"sha256": i.sha, **{k: (rec or {}).get(k) for k in keys}}
                by_sha[i.sha]["members"] = []
                rows.append(by_sha[i.sha])
            by_sha[i.sha]["members"].append(i.member())
    return rows, missing


def fmt(v: float | None, ci: Sequence[float] | None, dp: int = 3) -> str:
    point = "n/a" if v is None else f"{v:+.{dp}f}"
    return point + (" [no interval]" if ci is None else f" [{ci[0]:+.{dp}f}, {ci[1]:+.{dp}f}]")


def print_void(name: str, v: dict[str, Any], indent: str = "  ") -> None:
    tag = "  UNREADABLE" if v["unreadable"] else ""
    rate_s = "n/a" if v["rate"] is None else f"{v['rate']:.3f}"
    print(f"{indent}{name:<12} {v['void']:>4} of {v['n']:<4} {rate_s}  {v['by_cause']}{tag}")


def print_summary(s: dict[str, Any]) -> None:
    print("\nvoids")
    for name, v in s["voids"].items():
        print_void(name, v)
        for arm, a in v.get("by_arm", {}).items():
            print_void(arm, a, indent="    ")
    print("\nvoid bias, descriptive: actionable share under the two existing judges")
    for pop, v in s["void_bias"].items():
        parts = " ".join(
            f"{k} n={v[k]['n']} gpt={v[k]['gpt']} sonnet={v[k]['sonnet']}"
            for k in ("void", "scored")
        )
        print(f"  {pop:<12} {parts}")
    e3, e4 = s["E3"], s["E4"]
    print()
    print(f"{'E1 band share':<26} {fmt(s['E1']['rate'], s['E1']['ci'])}  {s['E1']['reading']}")
    print(f"{'E2 band AUC':<26} {fmt(s['E2']['auc'], s['E2']['ci'])}  {s['E2']['reading']}")
    for k in E3_RATES:
        print(f"{'E3 ' + k:<26} {fmt(e3[k]['rate'], e3[k]['ci'])}  {e3[k]['reading']}")
    auc_tag = "  Unreadable" if e3["auc"]["unreadable"] else ""
    print(f"{'E3 adopted-vs-control AUC':<26} {fmt(e3['auc']['auc'], e3['auc']['ci'], 4)}{auc_tag}")
    print(f"{'E4 margin':<26} {fmt(e4['margin'], e4['ci'], 2)}  {e4['reading']}")
    print(f"{'E4 precision':<26} {e4['precision']}")
    print("\nprediction")
    for k, v in s["prediction"].items():
        if k != "readings":
            tag = "  (unreadable, not scored)" if v.get("unreadable") else ""
            print(f"  {k:<38} {v['value']}  within: {v['within']}{tag}")
    for k, v in s["prediction"]["readings"].items():
        print(f"  reading {k:<14} mode {v['mode']!r}, point {v['point']}; ", end="")
        print(f"got {v['actual']!r}, registered at probability {v['probability']}")


def report() -> int:
    loaded = load()
    # Everything --judge refuses on, except the key: --report needs none. A reading written from
    # a drifted context, a short population or a changed script is worth no more here than there.
    refusals = readiness(loaded, key=False)
    if refusals:
        print("REFUSED:\n" + "\n".join(f"  - {r}" for r in refusals))
        return 1
    print("reproductions from the existing labels")
    repro = reproductions(loaded)
    if not print_reproductions(repro):
        print("REFUSED: a registered figure of the existing judges does not reproduce")
        return 1
    rows, missing = build_rows(loaded)
    if missing:
        print(f"REFUSED: {missing} distinct prompts have no settled verdict yet (run --judge)")
        return 1
    ledger = load_ledger()
    data: dict[str, Any] = {
        "_comment": (
            "Pre-registered in evals/PREREG-third-judge.md. One row per distinct prompt, and one "
            "per paper with no prompt, each with every population membership it serves. Written "
            "by evals/third_judge.py --report; tests/test_third_judge.py recomputes the summary "
            "from the rows."
        ),
        "model": MODEL,
        "model_versions": dict(Counter(str(r.get("model_version")) for r in rows if r["sha256"])),
        "frozen": frozen_shas(),
        "ledger": {
            "calls": ledger["calls"],
            "rate_limited": ledger["rate_limited"],
            "spend_usd": round(ledger["spend_usd"], 4),
        },
        "reproductions": repro,
        "comparison_cases": loaded.cases,
        "rows": rows,
    }
    data["summary"] = summarise(data)
    sip._write_json(OUT, data)
    print_summary(data["summary"])
    print(f"\nwrote {OUT}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--plan", action="store_true", help="no paid call (the default)")
    mode.add_argument("--judge", action="store_true", help="paid: buy the registered verdicts")
    mode.add_argument("--report", action="store_true", help="no paid call: write the result")
    ap.add_argument("--no-resolve", action="store_true", help="--plan without DOI resolution")
    args = ap.parse_args()
    if args.judge:
        return judge()
    return report() if args.report else plan(resolve=not args.no_resolve)


if __name__ == "__main__":
    raise SystemExit(main())
