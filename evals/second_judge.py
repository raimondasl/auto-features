"""P7: how much of the labelled set is noise? A second judge over 200 of the 602 labels.

Every labelled-set decision in this project rests on **single-sample GPT-5.5 verdicts**
deciding differences of ±10 to ±22 net@2, and nothing bounds their noise. §5.2 varied the
*gate* model; the judge has never been varied at all. P6 established that the judge rewards
roughly the right thing on average (61% of verifiably-adopted papers, against a 2% floor).
That is validity. It says nothing about whether an individual verdict is reproducible, which
is what a ±22 difference between two gate arms is actually made of.

So: re-judge a stratified 200 with **Sonnet**, byte-identical rubric, and ask two questions.

  1. **kappa** — how far does a second judge agree, beyond chance?
  2. **does the +22 survive?** The prose-300 headline is a net@2 difference between two gate
     arms scored against judge labels. Swap the labels for the second judge's and recompute
     it on the same papers with the same cached gate verdicts.

**Kappa bounds noise, not validity.** Two LLMs can share a famous-technique halo and agree
enthusiastically on the same wrong answers. P6 is the validity test; these compose, and
neither substitutes for the other.

**The integrity guard that makes this meaningful.** `judge_paper` stores
`_prompt_hash = sha256(RUBRIC \\0 repo_context)[:12]`. If the repo clone has drifted since a
verdict was cached, the stored label answers a different question than any prompt we can
rebuild today, and comparing the two judges would silently compare two questions. Every case
is checked against its stored hash and a mismatch **excludes the case** rather than being
noted in passing.

**Pre-registered, restated before running (2026-08-06).** P7 predicted kappa >=0.6 and that
"the +22 keeps its sign at >=half magnitude", with a kill at kappa <0.4.

  * The +22 was measured over **602** papers. This samples **200**, so the expected magnitude
    on the subset is ~200/602 x 22 ~= 7. The half-magnitude test is therefore applied to the
    delta **recomputed on the same 200 under GPT-5.5 labels**, not to the number 22. Stating
    this here because applying it to 22 directly would fail the test by arithmetic alone.
  * The labelled set has since grown to ~830 verdicts (P5 added 320 wild labels). P7 samples
    only from the 602 that the prose-300 comparison actually used, so the recomputation is
    against the same population as the headline.

  PREDICTION: kappa >=0.6 on the actionable cut AND the delta keeps its sign at >=half of
  the GPT-5.5 delta on the same 200.
  KILL: kappa <0.4 — label noise swamps what the instrument decides, and every labelled-set
  conclusion needs adjudicated labels or noise-adjusted CIs before another arm is run.

    uv run python evals/second_judge.py --dry-run    # sample + hash check, $0
    uv run python evals/second_judge.py              # ~$2 of Sonnet

Verdicts are written to `.work/second_judge/`, **never** to `evals/cache/judge/`. A second
judge's score in the gold cache would be indistinguishable from the first judge's.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import judge as judge_mod  # noqa: E402
from diagnose_triage import _load_env, fetch_papers  # noqa: E402
from harness import WORK_DIR, assemble_repo_context  # noqa: E402

from reporadar.config import SuggestionsConfig  # noqa: E402
from reporadar.llm_client import complete  # noqa: E402

EVALS = Path(__file__).resolve().parent
WORK = EVALS / ".work"
GOLD = EVALS / "cache" / "judge" / "v1" / "gpt-5.5"
OUT = WORK / "second_judge.json"
CACHE = WORK / "second_judge"
CONTROL = WORK / "diag_triage_keywords.json"
TREATMENT = WORK / "diag_triage_prose300.json"

ACTIONABLE = 2
SEED = 20260807
SAMPLE_N = 200
DEFAULT_MODEL = "claude-sonnet-5"
# Pre-registered, see module docstring.
PREDICT_KAPPA = 0.60
KILL_KAPPA = 0.40
HALF_MAGNITUDE = 0.5


def cohens_kappa(a: list[int], b: list[int]) -> float:
    """Unweighted Cohen's kappa. Used on the binary actionable cut, which is what ships."""
    n = len(a)
    if n == 0:
        return float("nan")
    po = sum(1 for x, y in zip(a, b, strict=True) if x == y) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / (n * n)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def quadratic_kappa(a: list[int], b: list[int], k: int = 4) -> float:
    """Quadratic-weighted kappa over the 0-3 scale.

    Supplementary, not the pre-registered statistic: it credits partial agreement (a 2 vs a
    3 is nearly agreement) which is the right reading of an ordinal rubric, but the decision
    the instrument makes is binary at >=2 and that is what the prediction was written about.
    """
    n = len(a)
    if n == 0:
        return float("nan")
    obs = [[0.0] * k for _ in range(k)]
    for x, y in zip(a, b, strict=True):
        obs[x][y] += 1
    ca, cb = Counter(a), Counter(b)
    num = den = 0.0
    for i in range(k):
        for j in range(k):
            w = ((i - j) ** 2) / ((k - 1) ** 2)
            num += w * obs[i][j]
            den += w * ca[i] * cb[j] / n
    return 1.0 if den == 0 else 1 - num / den


def net_at_2(rows: list[dict[str, Any]], labels: dict[tuple[str, str], int]) -> int:
    """net@2 of one gate arm: admitted papers scored (+1 actionable, -2 not)."""
    net = 0
    for r in rows:
        key = (r["case"], r["id"])
        if r["triage"] >= ACTIONABLE and key in labels:
            net += 1 if labels[key] >= ACTIONABLE else -2
    return net


def verify_contexts(cases: list[str]) -> tuple[dict[str, str], list[str]]:
    """Rebuild each case's repo context and check it against the stored prompt hash.

    A mismatch means the clone moved under the cache: the stored GPT-5.5 label answers a
    question we can no longer reconstruct, so re-judging that case would compare two
    different prompts and call the difference noise.
    """
    contexts: dict[str, str] = {}
    drifted: list[str] = []
    for case in cases:
        repo = WORK_DIR / case
        if not repo.is_dir():
            drifted.append(case)
            continue
        ctx = assemble_repo_context(repo)
        want = hashlib.sha256(f"{judge_mod.RUBRIC}\0{ctx}".encode()).hexdigest()[:12]
        stored = {
            json.loads(f.read_text(encoding="utf-8")).get("_prompt_hash")
            for f in (GOLD / case).glob("*.json")
        }
        if want in stored:
            contexts[case] = ctx
        else:
            drifted.append(case)
    return contexts, drifted


def sample(
    labels: dict[tuple[str, str], int], cases: set[str], n: int, rng: random.Random
) -> list[tuple[str, str]]:
    """Stratified by (case, verdict), round-robin so no case or score dominates."""
    strata: dict[tuple[str, int], list[tuple[str, str]]] = defaultdict(list)
    for (case, pid), score in labels.items():
        if case in cases:
            strata[(case, score)].append((case, pid))
    for rows in strata.values():
        rng.shuffle(rows)
    out: list[tuple[str, str]] = []
    order = sorted(strata)
    while len(out) < n and any(strata[k] for k in order):
        for key in order:
            if strata[key] and len(out) < n:
                out.append(strata[key].pop())
    rng.shuffle(out)
    return out


def second_cache_path(model: str, case: str, paper_id: str) -> Path:
    """Where one second-judge verdict lives. Sanitised the way the gold cache already was.

    This used to be ``paper['arxiv_id'].replace('/', '_')``, which leaves the **colon** in a
    synthetic id like ``doi:10.1038/s42256-023-00716-3``. On Windows a colon in a path is the
    NTFS alternate-data-stream separator, so every non-arXiv verdict this project ever bought
    was written into a stream hanging off a zero-byte file named ``doi`` — **93 of them across
    11 cases**, 82 from §21's Europe PMC arm, invisible to ``ls``, ``glob`` and ``find``, and
    silently dropped by any copy to another filesystem. They read back through this same
    function, which is why §21.2's numbers are right and why nothing ever noticed.

    ``judge._cache_path`` has always sanitised with this exact expression. Two implementations
    of one invariant, and the second was wrong — the fourth time this project has paid for that.
    Fixing it orphans the 93, which re-buy for about $1 when a run next needs them.
    """
    return CACHE / model / case / f"{safe_paper_id(paper_id)}.json"


def safe_paper_id(paper_id: str) -> str:
    """A paper id as a filename, for any verdict cache. The rule, in one place.

    Byte-identical to ``judge._cache_path``'s expression, so the two caches agree on what a
    paper is called. Exported because `.work/second_judge` is not the only cache keyed by a
    paper id — `proposes_method`'s classifier cache and `redacted_judge`'s arms share the
    hazard, and a colon is legal in a POSIX filename and is not one on NTFS.
    """
    return re.sub(r"[^A-Za-z0-9_.-]", "_", paper_id)


def second_verdict(
    case: str, ctx: str, paper: dict[str, Any], model: str, *, cache_as: str | None = None
) -> int:
    """One Sonnet verdict, cached OUTSIDE the gold cache.

    *cache_as* overrides only the cache directory, never the model called. It exists so a
    REPLICATE draw of the same model can be stored beside the original instead of reading it
    back: this path sends no temperature, so the Anthropic default (1.0) applies and a second
    call is a genuinely independent sample. Without the override, measuring the judge's
    agreement with itself is impossible — the cache would return the first draw.

    The rubric text is byte-identical to the first judge's. The framing cannot be: the first
    judge sends it as an OpenAI system message and this sends one prompt string. That is a
    real difference between the two conditions and it is not removable while the judges are
    different vendors — it is a limitation of the comparison, reported as one.
    """
    path = second_cache_path(cache_as or model, case, str(paper["arxiv_id"]))
    if path.is_file():
        return int(json.loads(path.read_text(encoding="utf-8"))["score"])
    prompt = f"{judge_mod.RUBRIC}\n\n{judge_mod._build_user_prompt(ctx, paper)}"
    cfg = SuggestionsConfig(provider="claude", claude_model=model, timeout=120)
    # 500 truncated 7 of the first 200 responses mid-justification, and the visible scores
    # in those fragments were 2, 2 and 3 — so the dropped papers skewed ACTIONABLE and the
    # loss would have biased the second judge's base rate downward. A parse failure that
    # correlates with the verdict is not a random 3.5% dropout.
    # The context is ~86% of this prompt and repeats for every item in a case, so the same
    # bytes were being bought once per paper. Measured 2026-09-06: 3521 of 4112 tokens read
    # from cache on the second and third calls of a burst. The prompt itself does not change.
    raw = complete(prompt, cfg, max_tokens=1200, cache_split_on=judge_mod.PAPER_MARKER)
    a, b = raw.find("{"), raw.rfind("}")
    if a < 0 or b < 0:
        raise ValueError(f"no JSON object in response: {raw[:160]}")
    data = json.loads(raw[a : b + 1])
    score = int(data["score"])
    if score not in (0, 1, 2, 3):
        raise ValueError(f"score out of range: {score}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({**data, "score": score}, indent=2), encoding="utf-8")
    return score


def threshold_shift(rows: list[dict[str, Any]]) -> list[tuple[int, float, float]]:
    """Kappa between GPT's >=2 cut and the second judge's cut at each threshold.

    A diagnostic, not a rescoring. If the two judges rank papers the same way but sit at
    different strictness, the disagreement is a calibration offset and moving the second
    judge's threshold recovers the agreement. If they genuinely disagree about which papers
    are good, no threshold helps. Those are very different problems and the pre-registered
    single number cannot tell them apart.
    """
    gpt = [1 if r["gpt"] >= ACTIONABLE else 0 for r in rows]
    out = []
    for cut in (1, 2, 3):
        snt = [1 if r["sonnet"] >= cut else 0 for r in rows]
        agree = sum(1 for x, y in zip(gpt, snt, strict=True) if x == y) / max(len(rows), 1)
        out.append((cut, cohens_kappa(gpt, snt), agree))
    return out


def report(rows: list[dict[str, Any]], control: list[dict], treatment: list[dict]) -> None:
    gpt = [r["gpt"] for r in rows]
    snt = [r["sonnet"] for r in rows]
    gpt_bin = [1 if s >= ACTIONABLE else 0 for s in gpt]
    snt_bin = [1 if s >= ACTIONABLE else 0 for s in snt]
    k_bin = cohens_kappa(gpt_bin, snt_bin)
    k_quad = quadratic_kappa(gpt, snt)
    agree = sum(1 for x, y in zip(gpt, snt, strict=True) if x == y) / max(len(rows), 1)
    agree_bin = sum(1 for x, y in zip(gpt_bin, snt_bin, strict=True) if x == y) / max(len(rows), 1)

    print(f"\n=== P7 — second judge over {len(rows)} labels ===")
    print(f"exact agreement on 0-3      {agree:.0%}")
    print(f"agreement on the >=2 cut    {agree_bin:.0%}")
    print(f"Cohen's kappa (>=2 cut)     {k_bin:.3f}   <- the pre-registered statistic")
    print(f"quadratic-weighted kappa    {k_quad:.3f}   (supplementary, ordinal)")
    print(
        f"\nbase rates: GPT-5.5 {sum(gpt_bin) / len(rows):.0%} actionable, "
        f"Sonnet {sum(snt_bin) / len(rows):.0%}"
    )
    print("\nconfusion (rows GPT-5.5, cols Sonnet):")
    print("      " + "".join(f"{j:>6}" for j in range(4)))
    for i in range(4):
        print(
            f"  {i}   "
            + "".join(
                f"{sum(1 for r in rows if r['gpt'] == i and r['sonnet'] == j):>6}" for j in range(4)
            )
        )

    print("\nis the disagreement a ranking difference or a strictness offset?")
    print("  GPT >=2  vs  Sonnet >=cut :   kappa   agreement")
    for cut, k, agree in threshold_shift(rows):
        mark = "   <- the shipped cut" if cut == ACTIONABLE else ""
        print(f"    cut {cut}                      {k:.3f}     {agree:.0%}{mark}")

    keys = {(r["case"], r["id"]) for r in rows}
    ctl = [r for r in control if (r["case"], r["id"]) in keys]
    trt = [r for r in treatment if (r["case"], r["id"]) in keys]
    gpt_labels = {(r["case"], r["id"]): r["gpt"] for r in rows}
    snt_labels = {(r["case"], r["id"]): r["sonnet"] for r in rows}
    d_gpt = net_at_2(trt, gpt_labels) - net_at_2(ctl, gpt_labels)
    d_snt = net_at_2(trt, snt_labels) - net_at_2(ctl, snt_labels)
    print(
        f"\nprose300 - keywords, recomputed on these {len(rows)} papers:\n"
        f"  under GPT-5.5 labels  {d_gpt:+d}\n"
        f"  under Sonnet labels   {d_snt:+d}"
    )
    print("  (the published +22 is over all 602; the subset delta is the fair comparison)")

    survives = d_gpt != 0 and d_snt * d_gpt > 0 and abs(d_snt) >= HALF_MAGNITUDE * abs(d_gpt)
    print(
        f"\nPRE-REGISTERED: kappa >={PREDICT_KAPPA} AND the delta keeps its sign at "
        f">={HALF_MAGNITUDE:.0%} magnitude; KILL at kappa <{KILL_KAPPA}"
    )
    if k_bin < KILL_KAPPA:
        verdict = "KILL — label noise swamps what the instrument decides"
    elif k_bin >= PREDICT_KAPPA and survives:
        verdict = "MET"
    elif k_bin >= PREDICT_KAPPA:
        verdict = "kappa MET, the prose-300 delta does NOT survive the label swap"
    elif survives:
        verdict = f"kappa BELOW prediction ({k_bin:.2f}) but above the kill bar; delta survives"
    else:
        verdict = f"BELOW PREDICTION on both ({k_bin:.2f}, delta {d_snt:+d} vs {d_gpt:+d})"
    print(f"verdict: {verdict}")
    print(f"\nwritten to {OUT}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=SAMPLE_N)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--dry-run", action="store_true", help="sample + hash check only, $0")
    ap.add_argument("--report", action="store_true", help="re-derive from saved verdicts, $0")
    args = ap.parse_args()
    _load_env()

    control = json.loads(CONTROL.read_text(encoding="utf-8"))
    treatment = json.loads(TREATMENT.read_text(encoding="utf-8"))
    if args.report:
        report(json.loads(OUT.read_text(encoding="utf-8")), control, treatment)
        return 0

    # The population the +22 was measured over: papers with BOTH gate arms and a label.
    both = {(r["case"], r["id"]) for r in control} & {(r["case"], r["id"]) for r in treatment}
    labels = {(r["case"], r["id"]): r["judge"] for r in control if (r["case"], r["id"]) in both}
    print(f"{len(labels)} papers carry both gate arms and a GPT-5.5 label")

    contexts, drifted = verify_contexts(sorted({c for c, _ in labels}))
    print(f"prompt-hash check: {len(contexts)} cases reproduce, {len(drifted)} drifted")
    if drifted:
        print(f"  excluded (clone moved under the cache): {', '.join(sorted(drifted))}")
    usable = {k: v for k, v in labels.items() if k[0] in contexts}
    picked = sample(usable, set(contexts), args.n, random.Random(SEED))
    by_case = Counter(c for c, _ in picked)
    by_score = Counter(usable[k] for k in picked)
    print(f"sampled {len(picked)} from {len(usable)} across {len(by_case)} cases")
    print(f"  score mix: {dict(sorted(by_score.items()))}")
    print(f"  ~${len(picked) * 0.011:.2f} of Sonnet")
    if args.dry_run:
        return 0

    papers = fetch_papers(sorted({pid for _, pid in picked}))
    rows: list[dict[str, Any]] = []
    for case, pid in picked:
        if pid not in papers:
            continue
        paper = {"arxiv_id": pid, **papers[pid]}
        try:
            score = second_verdict(case, contexts[case], paper, args.model)
        except Exception as exc:  # noqa: BLE001
            print(f"    ! {case}/{pid} failed: {exc}")
            continue
        rows.append({"case": case, "id": pid, "gpt": usable[(case, pid)], "sonnet": score})
        if len(rows) % 25 == 0:
            print(f"  {len(rows)}/{len(picked)}", flush=True)
    OUT.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    report(rows, control, treatment)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
