"""Frozen testbeds for the within-band ranking experiments (E1-E5).

The research pass (evals/RESEARCH-score2-ranking.md) ends in five pre-registered
experiments, all evaluated OFFLINE against judge labels already on disk — zero new judge
calls. This module is the one place those labels are loaded and the bands reconstructed,
so every experiment scores against the same frozen data:

* **Testbed A** — the live pool-50 run `judge-gpt-5.5-20260807T180938Z.json` (22 cases,
  220 shown papers). Per-paper gate bands are NOT recorded there; they are reconstructed
  positionally: `reporadar_top10` is emitted in `rerank_by_actionability` order — a stable
  sort on (llm_score, score_total) descending — so with the sweep counts n3/n2/n1
  (returned at min_actionable 3/2/1), positions [0,n3) are gate-3, [n3,n2) the score-2
  band, [n2,n1) gate-1, and the rest gate-0/unscored. The counts come from the same file,
  so the reconstruction is exact.
* **Testbed A'** — the pool-300 replication arm: `...164310Z.json` minus its two
  arXiv-throttled zero rows (db, storage), plus their single-case re-runs.
* **Testbed B** — `diag_triage_prose300.json`: 602 (gate, judge) pairs over the original
  12 repos, gate>=2 band of 125 papers, all labelled. Nearly saturated at judge>=2
  (top-stratum bias — see RESEARCH annex pitfall 2), so the target here is judge==3
  ordering only.
* **Testbed C** — the wild-distribution sanity checks: `gate_full_pool.json` (73 judged)
  and `label_pool.json` (320 judged). Pooled metrics only; per-repo counts are noise.

Guardrails enforced here rather than in each experiment: paper text exposes ONLY
gate-time information (title + abstract; never judge text — justifications sit next to
the labels and feeding them to a ranker is leakage), and the policy threshold is the
global 2/3 from net@2's payoff structure, never fitted per repo.
"""

from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import dataclass, field
from functools import cache, lru_cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from reporadar.paper_id import dedup_id  # noqa: E402

EVALS = Path(__file__).resolve().parent
WORK = EVALS / ".work"
RESULTS = EVALS / "results"
EXP = WORK / "exp"

# The frozen runs. These exact files are what the research doc's shortlist pre-registered
# against; pointing an experiment at a fresh run is a different (unregistered) experiment.
POOL50 = RESULTS / "judge-gpt-5.5-20260807T180938Z.json"
POOL300 = RESULTS / "judge-gpt-5.5-20260807T164310Z.json"
POOL300_RERUNS = {
    "db": RESULTS / "judge-gpt-5.5-20260807T191846Z.json",
    "storage": RESULTS / "judge-gpt-5.5-20260807T192453Z.json",
}

# Decision-theoretic show/abstain threshold: under net@2 a shown paper is worth 3p-2, so
# breakeven is p = FP-cost / (FP-cost + TP-gain) = 2/3. Global, never tuned per repo.
SHOW_THRESHOLD = 2.0 / 3.0

ACTIONABLE = 2  # judge >= this counts as genuinely actionable, matching metrics.py


def load_env() -> None:
    """Load evals/.env into the environment (keys only; values never printed)."""
    env = EVALS / ".env"
    if not env.is_file():
        return
    for line in env.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


@dataclass
class Paper:
    """One shown/labelled paper with gate-time text and its frozen judge verdict."""

    case: str
    id: str  # base arxiv id, no version suffix
    title: str
    abstract: str  # "" when no gate-time text is on disk
    judge: int  # 0-3, frozen
    gate: int  # 3 / 2 / 1 / 0; on Testbed A a positional reconstruction
    pos: int = 0  # rank in the shown digest (Testbed A) or 0

    @property
    def actionable(self) -> bool:
        return self.judge >= ACTIONABLE


@dataclass
class CaseBand:
    """One case's shown set, split into gate bands."""

    case: str
    papers: list[Paper] = field(default_factory=list)

    @property
    def gate3(self) -> list[Paper]:
        return [p for p in self.papers if p.gate >= 3]

    @property
    def band2(self) -> list[Paper]:
        return [p for p in self.papers if p.gate == 2]

    @property
    def admitted(self) -> list[Paper]:
        return [p for p in self.papers if p.gate >= 2]


def _base(arxiv_id: str) -> str:
    """Delegates to the one shared rule; see reporadar.paper_id."""
    return dedup_id(arxiv_id)


@cache
def pool_text(case: str) -> dict[str, dict[str, str]]:
    """{base_id: {title, abstract}} from the case's frozen candidate pool."""
    path = WORK / "full_pool" / f"{case}.jsonl"
    out: dict[str, dict[str, str]] = {}
    if not path.is_file():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            out[_base(row["id"])] = {"title": row["title"], "abstract": row["abstract"]}
    return out


@lru_cache(maxsize=1)
def cached_paper_text() -> dict[str, dict[str, str]]:
    """{base_id: {title, abstract}} from the diagnostic's arXiv fetch cache."""
    path = WORK / "triage_papers.json"
    if not path.is_file():
        return {}
    return {_base(k): v for k, v in json.loads(path.read_text(encoding="utf-8")).items()}


def _text_for(case: str, base_id: str) -> tuple[str, str]:
    """(title, abstract) from the frozen gate-time sources, or ("", "")."""
    row = pool_text(case).get(base_id) or cached_paper_text().get(base_id)
    if row:
        return row["title"], row["abstract"]
    return "", ""


def _bands_from_run(entries: list[dict]) -> dict[str, CaseBand]:
    """Positional band reconstruction for one judge-results file (Testbed A shape)."""
    out: dict[str, CaseBand] = {}
    for entry in entries:
        case = entry["case"]
        sweep = entry["reporadar_toppicks_sweep"]
        n3 = sweep["3"]["n_returned"]
        n2 = sweep["2"]["n_returned"]
        n1 = sweep["1"]["n_returned"]
        band = CaseBand(case=case)
        for pos, row in enumerate(entry["returned"]["reporadar_top10"]):
            base_id = _base(row["arxiv_id"])
            gate = 3 if pos < n3 else 2 if pos < n2 else 1 if pos < n1 else 0
            title, abstract = _text_for(case, base_id)
            band.papers.append(
                Paper(
                    case=case,
                    id=base_id,
                    title=title or row.get("title", ""),
                    abstract=abstract,
                    judge=int(row["judge_score"]),
                    gate=gate,
                    pos=pos,
                )
            )
        out[case] = band
    return out


def load_testbed_a() -> dict[str, CaseBand]:
    """The pool-50 arm: 22 cases, 220 shown papers, bands reconstructed."""
    return _bands_from_run(json.loads(POOL50.read_text(encoding="utf-8")))


def load_testbed_a300() -> dict[str, CaseBand]:
    """The pool-300 replication arm, with the throttled db/storage rows replaced by
    their single-case re-runs (the 22-case file's zeroed rows are collection failures,
    not abstentions — RESULTS.md 'Recover db and storage')."""
    entries = [
        e
        for e in json.loads(POOL300.read_text(encoding="utf-8"))
        if e["case"] not in POOL300_RERUNS
    ]
    for case, path in POOL300_RERUNS.items():
        rerun = json.loads(path.read_text(encoding="utf-8"))
        entries.extend(e for e in rerun if e["case"] == case)
    return _bands_from_run(entries)


def load_testbed_b() -> dict[str, CaseBand]:
    """The 12-repo labelled band under the shipped gate config (diag prose-300).

    Every admitted paper carries a judge label. At judge>=2 this set is nearly
    saturated (~0.93 within-band precision — cache papers are a top stratum), so
    experiments target judge==3 ordering here, not judge>=2 selection.
    """
    rows = json.loads((WORK / "diag_triage_prose300.json").read_text(encoding="utf-8"))
    out: dict[str, CaseBand] = {}
    for row in rows:
        case = row["case"]
        band = out.setdefault(case, CaseBand(case=case))
        base_id = _base(row["id"])
        title, abstract = _text_for(case, base_id)
        band.papers.append(
            Paper(
                case=case,
                id=base_id,
                title=title,
                abstract=abstract,
                judge=int(row["judge"]),
                gate=int(row["triage"]),
            )
        )
    return out


def load_testbed_c() -> dict[str, list[Paper]]:
    """The wild-distribution checks, pooled only: {'gate_full_pool': [...], 'label_pool': [...]}.

    gate_full_pool: every judged row (33 admits + 40 sampled non-admits).
    label_pool: every judged gate-admitted row (66; 16 judge==3).
    """
    gfp = [
        Paper(
            case=r["case"],
            id=_base(r["id"]),
            title=_text_for(r["case"], _base(r["id"]))[0],
            abstract=_text_for(r["case"], _base(r["id"]))[1],
            judge=int(r["judge"]),
            gate=int(r["gate"]),
        )
        for r in json.loads((WORK / "gate_full_pool.json").read_text(encoding="utf-8"))
        if r.get("judge") is not None
    ]
    lp = [
        Paper(
            case=r["case"],
            id=_base(r["id"]),
            title=_text_for(r["case"], _base(r["id"]))[0],
            abstract=_text_for(r["case"], _base(r["id"]))[1],
            judge=int(r["judge"]),
            gate=int(r["gate"]),
        )
        for r in json.loads((WORK / "label_pool.json").read_text(encoding="utf-8"))
        if r.get("judge") is not None and int(r["gate"]) >= 2
    ]
    return {"gate_full_pool": gfp, "label_pool": lp}


# ── Repo side of every prompt: the same prose-300 profile the shipped gate saw ─────────


@cache
def repo_block(case: str) -> str:
    """The gate-time repository description (keywords + prose-300), built once per case.

    Delegates to the SHIPPED builder rather than reimplementing it. The fine-scale
    stage's score→probability map was fitted against this exact prompt shape, so a
    benchmark that described repos even slightly differently from the product would
    report a calibration the product does not have. Sharing one function makes that
    class of drift impossible rather than merely unlikely.
    """
    import sys

    sys.path.insert(0, str(EVALS.parent / "src"))

    from reporadar.config import ProfilerConfig
    from reporadar.profiler import profile_repo
    from reporadar.triage import repo_context_block

    repo = WORK / case
    if not repo.is_dir():
        raise FileNotFoundError(f"no clone for case {case!r} under {WORK}")
    return repo_context_block(profile_repo(repo, profiler_cfg=ProfilerConfig(prose_chars=300)))


# ── Metrics ────────────────────────────────────────────────────────────────────────────


def auc(scores: list[float], labels: list[bool]) -> float:
    """Rank-based ROC-AUC (Mann-Whitney with average ranks). NaN if one class."""
    pos = sum(labels)
    neg = len(labels) - pos
    if pos == 0 or neg == 0:
        return float("nan")
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    rank_sum = sum(r for r, y in zip(ranks, labels, strict=True) if y)
    return (rank_sum - pos * (pos + 1) / 2) / (pos * neg)


def net2(judges: list[int]) -> float:
    """net@2 over a shown set, identical to metrics.net_actionable_value(lam=2)."""
    good = sum(1 for j in judges if j >= ACTIONABLE)
    return good - 2.0 * (len(judges) - good)


def brier(probs: list[float], labels: list[bool]) -> float:
    if not probs:
        return float("nan")
    return sum((p - float(y)) ** 2 for p, y in zip(probs, labels, strict=True)) / len(probs)


def ece(probs: list[float], labels: list[bool], bins: int = 10) -> float:
    """Expected calibration error, equal-width bins."""
    if not probs:
        return float("nan")
    total = len(probs)
    err = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        members = [
            (p, y)
            for p, y in zip(probs, labels, strict=True)
            if (lo <= p < hi) or (b == bins - 1 and p == hi)
        ]
        if not members:
            continue
        conf = sum(p for p, _ in members) / len(members)
        acc = sum(1.0 for _, y in members if y) / len(members)
        err += len(members) / total * abs(conf - acc)
    return err


def reliability_table(probs: list[float], labels: list[bool], bins: int = 10) -> list[dict]:
    rows = []
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        members = [
            (p, y)
            for p, y in zip(probs, labels, strict=True)
            if (lo <= p < hi) or (b == bins - 1 and p == hi)
        ]
        if members:
            rows.append(
                {
                    "bin": f"[{lo:.1f},{hi:.1f})",
                    "n": len(members),
                    "mean_p": sum(p for p, _ in members) / len(members),
                    "frac_actionable": sum(1 for _, y in members if y) / len(members),
                }
            )
    return rows


def sign_test(deltas: list[float]) -> dict:
    """Two-sided sign test on nonzero paired deltas (exact binomial)."""
    pos = sum(1 for d in deltas if d > 0)
    neg = sum(1 for d in deltas if d < 0)
    n = pos + neg
    if n == 0:
        return {"pos": 0, "neg": 0, "ties": len(deltas), "p": 1.0}
    k = max(pos, neg)
    p = sum(math.comb(n, i) for i in range(k, n + 1)) / 2 ** (n - 1)
    return {"pos": pos, "neg": neg, "ties": len(deltas) - n, "p": min(1.0, p)}


def policy_net(band: CaseBand, prob_by_id: dict[str, float], thr: float = SHOW_THRESHOLD) -> float:
    """net@2 of the pre-registered policy: show gate-3 + band-2 papers with P >= thr.

    Papers the method failed to score are NOT shown (missing != confident): a scoring
    failure must not silently become an admission.
    """
    shown = [p.judge for p in band.gate3]
    shown += [p.judge for p in band.band2 if prob_by_id.get(p.id, -1.0) >= thr]
    return net2(shown)


def baseline_nets(bands: dict[str, CaseBand]) -> dict[str, dict[str, float]]:
    """The pre-registered comparison points, per case: show-all (min>=2) and score-3-only."""
    out: dict[str, dict[str, float]] = {}
    for case, band in bands.items():
        out[case] = {
            "show_all": net2([p.judge for p in band.admitted]),
            "score3_only": net2([p.judge for p in band.gate3]),
        }
    return out


def pooled_band_auc(
    bands: dict[str, CaseBand], scores: dict[str, dict[str, float]], target: int = ACTIONABLE
) -> float:
    """AUC of a method's scores over all score-2-band papers, pooled across cases.

    *scores* is {case: {paper_id: score}} — keyed per case because the same arXiv id can
    legitimately appear in two repos' bands with different scores.
    """
    xs: list[float] = []
    labels: list[bool] = []
    for case, band in bands.items():
        per_case = scores.get(case, {})
        for p in band.band2:
            if p.id in per_case:
                xs.append(per_case[p.id])
                labels.append(p.judge >= target)
    return auc(xs, labels)
