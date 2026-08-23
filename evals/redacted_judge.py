"""P10: is P9's typed-span signal the judge reading its own input back?

P9 measured that abstracts of actionable papers name a repository's typed README spans far
more often than abstracts of non-actionable ones (+27.5pt Mantel-Haenszel, against -0.6pt
for the shipped manifest channel). Every one of those numbers is scored against verdicts
from a judge that `assemble_repo_context` had already shown ``README[:3500]`` plus the
manifests. So the correlation could be a real property of the repository, or it could be
the judge upscoring papers that repeat words it had just read.

**The obvious test is the wrong one.** Blinding the judge to the README removes the
circularity and the ground truth together -- `run_judge_eval.py` says so where it refuses
to ablate the judge's view alongside the treatment: *"Useful for this repository is a
property of the repository."* A judge that does not know what the repo is judges worse, and
a drop under that arm cannot be read as circularity rather than damage.

**So redact the tokens, not the document.** Both arms get the full repo context. In the
treatment arm every typed span is replaced by an opaque `[TERM-k]` placeholder, distinct
per span so sentence structure survives ("we implement [TERM-1] on top of [TERM-2]"). The
judge keeps its understanding of what the project does; what it loses is the literal string
the probe matches on. If the correlation survives that, it is not lexical echo.

**What this can and cannot settle.** It targets one mechanism: the judge repeating a token.
It does not make the judge a source of truth about real usefulness -- the deeper question,
which only model-free ground truth (P6's git-history adoptions) can reach. Redaction is
also imperfect by construction: a README that says "we implement [TERM-1] parameter-
efficient fine-tuning of adapters" still leaks the subject, so this arm UNDERSTATES how
much of the signal is real. A survival here is therefore strong evidence; a collapse is
weaker evidence of echo than it looks.

Both arms are judged fresh by the same model (Sonnet, the P7 second judge), on the same
papers, differing only in the context string. P7's cached verdicts are deliberately NOT
reused as the control: they were sampled for a different question, and a paired design that
re-judges both sides costs one extra call per paper and removes every other difference.

PRE-REGISTERED, written before the first run:

  Primary: the typed-span gap under redaction, as a FRACTION of the same gap under sight.
  Pure lexical echo predicts ~0. A real repository property predicts ~1.

  I predict **0.5 to 0.9** -- a partial drop, because redaction cannot remove the topic,
  only the token, and the judge can still infer "this is a LoRA repository" from the prose
  around the mask.

  **KILL CONDITION: retained fraction < 0.33.** Below that, lexical echo is the dominant
  contributor to P9's headline and the typed channel should not be developed further on
  judge-scored evidence.

  Secondary, a negative control: the MANIFEST channel is scored under both arms too. Its
  terms are not redacted, so its gap should not move. If it does, the arms differ by
  something other than the redaction and the primary is void.

    uv run python evals/redacted_judge.py --dry-run   # $0: mask, verify, print, judge nothing
    uv run python evals/redacted_judge.py             # ~$4: 2 arms x 200 papers of Sonnet
    uv run python evals/redacted_judge.py --report    # $0 once both arms are cached
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import judge as judge_mod  # noqa: E402
from harness import WORK_DIR, assemble_repo_context  # noqa: E402
from nerdme_probe import _spans_as_anchors  # noqa: E402
from relation_probe import (  # noqa: E402
    ACTIONABLE,
    _mentions,
    _pool_abstracts,
    _repo_terms,
    _verdicts,
)
from second_judge import safe_paper_id  # noqa: E402

from reporadar.config import SuggestionsConfig  # noqa: E402
from reporadar.llm_client import complete  # noqa: E402

EVALS = Path(__file__).resolve().parent
WORK = EVALS / ".work"
POOL = WORK / "pool-wemb"
SPANS = WORK / "nerdme_spans.json"
CACHE = WORK / "redacted_judge"  # NEVER evals/cache/judge -- these are not gold verdicts
MODEL = "claude-sonnet-5"
SEED = 20260816
SAMPLE_N = 200
KILL_FRACTION = 0.33


def redact(context: str, spans: set[str]) -> tuple[str, int]:
    """Replace each span with a distinct opaque placeholder. Longest first, so a span
    contained inside another (`lora` inside `qlora`) cannot half-mask it."""
    out, n = context, 0
    for i, span in enumerate(sorted(spans, key=len, reverse=True), start=1):
        pattern = re.compile(rf"\b{re.escape(span)}\b", re.IGNORECASE)
        out, k = pattern.subn(f"[TERM-{i}]", out)
        n += k
    return out, n


def _leaks(context: str, spans: set[str]) -> list[str]:
    """Spans still literally present after redaction. Must be empty, or the arm is void."""
    low = context.lower()
    return [s for s in spans if re.search(rf"\b{re.escape(s)}\b", low)]


def build_arms(cases: list[str]) -> dict[str, dict[str, Any]]:
    spans_by_case = json.loads(SPANS.read_text(encoding="utf-8"))
    arms: dict[str, dict[str, Any]] = {}
    for case in cases:
        repo = WORK_DIR / case
        if not repo.is_dir():
            continue
        sighted = assemble_repo_context(repo)
        spans = _spans_as_anchors(spans_by_case.get(case, {}))
        redacted, n = redact(sighted, spans)
        arms[case] = {
            "sighted": sighted,
            "redacted": redacted,
            "spans": spans,
            "replacements": n,
            "leaks": _leaks(redacted, spans),
        }
    return arms


def sample_papers(cases: list[str], n: int, rng: random.Random) -> list[tuple[str, str]]:
    """Stratified on the EXISTING GPT-5.5 label, oversampling actionable papers.

    A uniform draw would follow the pool's ~19% actionable rate and spend most of the
    budget on papers that cannot move the statistic. Sampling on the gold label is safe
    here because the label is not the outcome -- the outcome is whether Sonnet's verdict
    shifts between two contexts.
    """
    hi: list[tuple[str, str]] = []
    lo: list[tuple[str, str]] = []
    for case in cases:
        abstracts = _pool_abstracts(case)
        for pid, verdict in _verdicts(case).items():
            if pid not in abstracts:
                continue
            (hi if int(verdict.get("score", 0)) >= ACTIONABLE else lo).append((case, pid))
    rng.shuffle(hi)
    rng.shuffle(lo)
    half = n // 2
    return hi[:half] + lo[: n - half]


def verdict(case: str, arm: str, ctx: str, paper: dict[str, Any]) -> int | None:
    path = CACHE / arm / case / f"{safe_paper_id(str(paper['arxiv_id']))}.json"
    if path.is_file():
        return int(json.loads(path.read_text(encoding="utf-8"))["score"])
    prompt = f"{judge_mod.RUBRIC}\n\n{judge_mod._build_user_prompt(ctx, paper)}"
    cfg = SuggestionsConfig(provider="claude", claude_model=MODEL, timeout=120)
    raw = complete(prompt, cfg, max_tokens=1200)
    a, b = raw.find("{"), raw.rfind("}")
    if a < 0 or b < 0:
        return None
    try:
        data = json.loads(raw[a : b + 1])
        score = int(data["score"])
    except (json.JSONDecodeError, KeyError, ValueError, TypeError):
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"score": score, "arm": arm}, indent=1), encoding="utf-8")
    return score


def gap(rows: list[tuple[int, bool]]) -> float:
    hi = [h for s, h in rows if s >= ACTIONABLE]
    lo = [h for s, h in rows if s < ACTIONABLE]
    if not hi or not lo:
        return 0.0
    return 100 * sum(hi) / len(hi) - 100 * sum(lo) / len(lo)


def main() -> int:
    ap = argparse.ArgumentParser(description="P10 span-redacted judge")
    ap.add_argument("--dry-run", action="store_true", help="$0: mask and verify only")
    ap.add_argument("--report", action="store_true", help="$0: read cached verdicts only")
    ap.add_argument("--n", type=int, default=SAMPLE_N)
    args = ap.parse_args()

    cases = sorted(p.stem for p in POOL.glob("*.json"))
    arms = build_arms(cases)
    voided = {c: a["leaks"] for c, a in arms.items() if a["leaks"]}
    if voided:
        raise SystemExit(f"redaction left spans in context, arm is void: {voided}")

    print(f"Redaction over {len(arms)} cases\n")
    total = sum(a["replacements"] for a in arms.values())
    withspans = [c for c, a in arms.items() if a["spans"]]
    print(f"  {total} placeholder substitutions across {len(withspans)} cases with spans")
    unchanged = [c for c, a in arms.items() if a["replacements"] == 0]
    print(f"  {len(unchanged)} case(s) unchanged by redaction: {unchanged}")
    print("  leak check: clean (no span survives in any redacted context)\n")
    if args.dry_run:
        for c in withspans[:3]:
            a = arms[c]
            print(f"  --- {c}: {len(a['spans'])} spans, {a['replacements']} substitutions")
        return 0

    picked = sample_papers(cases, args.n, random.Random(SEED))
    print(f"Judging {len(picked)} papers x 2 arms with {MODEL}\n")
    rows: dict[str, list[tuple[int, bool]]] = {"sighted": [], "redacted": []}
    man_rows: dict[str, list[tuple[int, bool]]] = {"sighted": [], "redacted": []}
    failures: Counter[str] = Counter()
    for i, (case, pid) in enumerate(picked, 1):
        paper = _pool_abstracts(case).get(pid)
        if paper is None or case not in arms:
            continue
        text = f"{paper.get('title', '')} {paper.get('abstract', '')}"
        hit = bool(_mentions(text, arms[case]["spans"]))
        man_hit = bool(_mentions(text, _repo_terms(case)[0]))
        for arm in ("sighted", "redacted"):
            if args.report and not (CACHE / arm / case).is_dir():
                continue
            s = verdict(case, arm, arms[case][arm], paper)
            if s is None:
                failures[arm] += 1
                continue
            rows[arm].append((s, hit))
            man_rows[arm].append((s, man_hit))
        if i % 25 == 0:
            print(f"  {i}/{len(picked)}")

    print("\nP10 RESULT -- does the typed-span signal survive redaction?\n")
    print(f"  {'arm':12} {'n':>5} {'actionable':>11} {'typed gap':>11} {'manifest gap':>13}")
    for arm in ("sighted", "redacted"):
        r = rows[arm]
        act = sum(1 for s, _ in r if s >= ACTIONABLE)
        print(f"  {arm:12} {len(r):>5} {act:>11} {gap(r):>+10.1f}pt {gap(man_rows[arm]):>+12.1f}pt")
    if failures:
        print(f"\n  unparsed verdicts: {dict(failures)}")
    g_s, g_r = gap(rows["sighted"]), gap(rows["redacted"])
    frac = g_r / g_s if g_s else 0.0
    print(
        f"\n  retained fraction: {frac:.2f}  [pre-registered 0.5-0.9; KILL below {KILL_FRACTION}]"
    )
    print(f"  VERDICT: {'echo dominates' if frac < KILL_FRACTION else 'signal survives redaction'}")
    print(
        f"\n  negative control -- manifest gap moved "
        f"{gap(man_rows['redacted']) - gap(man_rows['sighted']):+.1f}pt "
        "(its terms are not redacted; a large move voids the primary)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
