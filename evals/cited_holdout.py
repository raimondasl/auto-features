"""§36: does the second judge price "the repository already has this"? (~$1 of Sonnet.)

    uv run python evals/cited_holdout.py --dry-run   # population + hash check, $0
    uv run python evals/cited_holdout.py             # buy the 95 Sonnet labels
    uv run python evals/cited_holdout.py --report    # re-derive from cache, $0

§32.4 is the live lead: over the ten papers the shipped already-cited rule removes, **Sonnet
scored all five of the repositories' own papers 0 or 1 while GPT-5.5 scored three of them 3.**
If that is real it explains why four separate score-3 arms failed (§9.4 twice, §26, §29, §30) —
the redundancy effect was there and the *outcome variable* did not contain it — and it is worth
more than those four arms combined. If it is not real it is five papers of noise and the whole
line closes.

It was found post-hoc, on ten papers, in the population that produced §16.6's headline. So it is
tested here on data that did not generate it: **24 repo-cited papers that carry a GPT-5.5 gold
label and no Sonnet label at all.** §32.4's ten are excluded by construction — every one of them
already has a Sonnet verdict, which is how they came to be looked at.

PRE-REGISTERED 2026-08-22, before any call was made. Bars and predictions in §36.

**Membership is the product's own rule.** `dedup_id(paper) in cited_arxiv_ids_of(repo)`, the same
test `digest.py:244` ships and §31/§32 audited. No classifier, no title matching — §30 failed
because I sorted papers by reading their titles, and this arm does not get to make that mistake.

**Declared before the fact: two labels I have already seen contradict the sharp form.** While
building this population I read two cached Sonnet verdicts on repositories' own papers outside
it — *OpenMM 8* (`bio-mdsim`, GPT 3 → **Sonnet 3**) and *Implementation strategies in phonopy and
phono3py* (`mat-phonon`, GPT 1 → **Sonnet 3**). Both are papers about the repository itself and
both clear the actionable cut. §32.4's "every one" is therefore already false, and I knew that
before writing the bar below. They sit outside the population and enter no endpoint.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from math import comb
from pathlib import Path
from typing import Any

EVALS = Path(__file__).resolve().parent
sys.path.insert(0, str(EVALS))
sys.path.insert(0, str(EVALS.parent / "src"))

from diagnose_triage import fetch_papers  # noqa: E402
from harness import WORK_DIR  # noqa: E402
from second_judge import (  # noqa: E402
    ACTIONABLE,
    DEFAULT_MODEL,
    second_cache_path,
    second_verdict,
    verify_contexts,
)
from second_judge import _load_env as load_env  # noqa: E402

from reporadar.paper_id import dedup_id  # noqa: E402
from reporadar.profiler import cited_arxiv_ids_of  # noqa: E402

GOLD = EVALS / "cache" / "judge" / "v1" / "gpt-5.5"
OUT = WORK_DIR / "cited_holdout.json"
POP = WORK_DIR / "cited_holdout_population.json"  # frozen on first build, see build()

CONTROLS_PER = 3  # matched non-cited papers per cited paper, same case and same GPT score
PRIMARY_GPT = 3  # the stratum §32.4's observation lives in and the only one with power
CONFIRM_RATIO = 0.67  # cited/control Sonnet-actionable ratio at or below which §36 confirms
KILL_RATIO = 0.90  # at or above which the redundancy line closes for good
ALPHA = 0.05

# §32.3's generating numbers, rebuilt by this script before any of its own output is readable.
PUBLISHED_REMOVED = 10
PUBLISHED_GPT_ACTIONABLE = 7
PUBLISHED_SONNET_ACTIONABLE = 3

# The tertiary, named in advance so it cannot be assembled after the labels arrive. These are
# the papers inside the population that ARE the repository, not merely cited by it.
SELF_PAPERS: dict[tuple[str, str], str] = {
    ("ann", "2401.08281"): "The Faiss library — faiss's own canonical citation",
    ("graph", "1903.02428"): "PyTorch Geometric — pytorch_geometric's own paper",
    ("rag", "2004.12832"): "ColBERT — the repository's own paper",
    ("rag", "2112.01488"): "ColBERTv2 — the repository IS ColBERTv2",
    ("speech", "2212.04356"): "Whisper — openai/whisper's own paper",
}


def _month(paper_id: str) -> int:
    """Months since year 0 for an arXiv id, so 'closest in date' is a total order.

    New-style ids are YYMM.NNNNN and old-style are archive/YYMMNNN; both put the date first,
    which is the only property this needs. Used to match each cited paper against controls of
    the same vintage — cited papers skew old (a repository cites its prior art), and an unmatched
    control arm would let 'Sonnet dislikes old papers' masquerade as the effect under test.
    """
    # dedup_id strips the version; this must not hand-roll that (tests/test_paper_id.py).
    head = dedup_id(paper_id).split("/")[-1].split(".")[0]
    if not head.isdigit() or len(head) < 4:
        return 0
    yy, mm = int(head[:2]), int(head[2:4])
    return (2000 + yy if yy < 91 else 1900 + yy) * 12 + max(1, min(12, mm))


def _gold_rows(case: str) -> dict[str, list[tuple[str, int]]]:
    """base id -> [(cached id, GPT score)]. Versions of one paper collapse to one entry."""
    out: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for path in sorted((GOLD / case).glob("*.json")):
        pid = path.stem
        score = int(json.loads(path.read_text(encoding="utf-8"))["score"])
        out[dedup_id(pid.replace("_", "/"))].append((pid, score))
    return out


def build() -> dict[str, Any]:
    """The whole population, deterministically. No seed, no sampling, no judgement calls.

    **Frozen on first build, and it has to be.** Membership says "carries no Sonnet verdict",
    and this arm's whole job is to CREATE Sonnet verdicts — so a second invocation would
    reclassify its own output: judged treatment papers would migrate into §32.3's generating
    set and blow the reproduction check past ten, while judged controls would vanish. That is
    not hypothetical; the first run of this script died partway through and left eleven
    verdicts behind (§37.1). The snapshot makes the arm resumable and the population fixed.

    A base id whose versions carry DISAGREEING GPT scores is dropped and reported: there is no
    fact of the matter about "the GPT label" for such a paper, and picking one would be a choice
    made by the author rather than by the rule. (`ann/1702.08734` is scored 1 and 3 under two
    ids — it is FAISS's billion-scale paper, and it sits in the excluded generating set anyway.)

    The order matters and the reproduction check caught it. Dropping split labels BEFORE
    classifying threw that FAISS paper out of §32.3's ten, and the rebuild came back 9/6/3
    against a published 10/7/3. The generating set's membership is fixed by §32.3, not by this
    script's tidiness rule, so "already carries a Sonnet verdict" is decided first.
    """
    if POP.is_file():
        return dict(json.loads(POP.read_text(encoding="utf-8")))
    treatment: list[dict[str, Any]] = []
    eligible: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    generating: list[dict[str, Any]] = []
    split_labels: list[str] = []
    no_checkout: list[str] = []

    for case in sorted(d.name for d in GOLD.iterdir() if d.is_dir()):
        repo = WORK_DIR / case
        if not repo.is_dir():
            no_checkout.append(case)
            continue
        cited = cited_arxiv_ids_of(repo)
        for base, versions in _gold_rows(case).items():
            # An existing Sonnet verdict means the paper has been looked at already. For the
            # cited arm that is exactly §32.4's ten; for controls it means an earlier arm chose
            # it for a reason of its own, which is a selection this one does not want to inherit.
            seen = [
                (pid, score)
                for pid, score in versions
                if second_cache_path(DEFAULT_MODEL, case, pid).is_file()
            ]
            if seen:
                if base in cited:
                    pid, score = max(seen, key=lambda v: len(v[0]))
                    generating.append(
                        {"case": case, "base": base, "id": pid, "gpt": score, "cited": True}
                    )
                continue
            scores = {s for _, s in versions}
            if len(scores) > 1:
                split_labels.append(f"{case}/{base}{'*' if base in cited else ''} {sorted(scores)}")
                continue
            # Judge the most specific id: it is the one the gold label was cached against, so
            # the two verdicts land side by side in their two caches.
            row = {
                "case": case,
                "base": base,
                "id": max((p for p, _ in versions), key=len),
                "gpt": scores.pop(),
                "cited": base in cited,
            }
            if row["cited"]:
                treatment.append(row)
            else:
                eligible[(case, row["gpt"])].append(row)

    controls: list[dict[str, Any]] = []
    taken: set[str] = set()
    short: list[str] = []
    for t in sorted(treatment, key=lambda r: (r["case"], r["id"])):
        pool = eligible[(t["case"], t["gpt"])]
        ranked = sorted(
            (r for r in pool if r["id"] not in taken),
            key=lambda r: (abs(_month(r["id"]) - _month(t["id"])), r["id"]),
        )
        picked = ranked[:CONTROLS_PER]
        if len(picked) < CONTROLS_PER:
            short.append(f"{t['case']} GPT{t['gpt']}: {len(picked)} of {CONTROLS_PER}")
        for c in picked:
            taken.add(c["id"])
            controls.append({**c, "matched_to": t["id"]})

    pop = {
        "treatment": treatment,
        "controls": controls,
        "generating": generating,
        "split_labels": split_labels,
        "no_checkout": no_checkout,
        "short_strata": short,
    }
    POP.parent.mkdir(parents=True, exist_ok=True)
    POP.write_text(json.dumps(pop, indent=1), encoding="utf-8")
    return pop


def fisher_one_sided(a: int, b: int, c: int, d: int) -> float:
    """P(cited actionable count <= a) under the null, margins fixed. Exact, tiny n."""
    n, r1, c1 = a + b + c + d, a + b, a + c
    if n == 0 or r1 == 0:
        return 1.0
    lo = max(0, r1 - (n - c1))
    return sum(comb(c1, k) * comb(n - c1, r1 - k) / comb(n, r1) for k in range(lo, a + 1))


def rate(rows: list[dict[str, Any]]) -> tuple[int, int]:
    return sum(1 for r in rows if r["sonnet"] >= ACTIONABLE), len(rows)


def _pct(hit: int, n: int) -> str:
    return f"{hit}/{n} = {hit / n:.3f}" if n else "n/a"


def verdict_for(ratio: float | None, p: float) -> str:
    """Which pre-registered bar the primary lands on. Chosen in §36, not here."""
    if ratio is None:
        return "no verdict — one arm is empty"
    if ratio <= CONFIRM_RATIO and p < ALPHA:
        return "CONFIRMED — the second judge prices redundancy and GPT-5.5 does not"
    if ratio >= KILL_RATIO:
        return "KILL — §32.4 was five papers of noise; the redundancy line closes"
    return "UNRESOLVED — neither bar met; the arm declines to pick a side"


def reproduction_check(generating: list[dict[str, Any]]) -> None:
    """§32.3's ten, rebuilt. Nothing below is readable if this does not match."""
    gpt = sum(1 for r in generating if r["gpt"] >= ACTIONABLE)
    son = 0
    for r in generating:
        path = second_cache_path(DEFAULT_MODEL, r["case"], r["id"])
        son += int(json.loads(path.read_text(encoding="utf-8"))["score"]) >= ACTIONABLE
    ok = (
        len(generating) == PUBLISHED_REMOVED
        and gpt == PUBLISHED_GPT_ACTIONABLE
        and son == PUBLISHED_SONNET_ACTIONABLE
    )
    print("\nreproduction check against §32.3's generating set (excluded from every endpoint)")
    print(f"  removed papers   rebuilt {len(generating):2d}   published {PUBLISHED_REMOVED}")
    print(f"  GPT actionable   rebuilt {gpt:2d}   published {PUBLISHED_GPT_ACTIONABLE}")
    print(f"  Sonnet actionable rebuilt {son:2d}  published {PUBLISHED_SONNET_ACTIONABLE}")
    print(f"  -> {'reproduces' if ok else 'DOES NOT REPRODUCE — nothing below is readable'}")


def report(rows: list[dict[str, Any]], pop: dict[str, Any]) -> None:
    cited = [r for r in rows if r["cited"]]
    ctl = [r for r in rows if not r["cited"]]

    print("\n" + "=" * 92)
    print("PRIMARY — GPT-3 stratum: does the cited arm keep fewer Sonnet-actionable papers?")
    print("=" * 92)
    ca, cn = rate([r for r in cited if r["gpt"] == PRIMARY_GPT])
    la, ln = rate([r for r in ctl if r["gpt"] == PRIMARY_GPT])
    print(f"  cited, GPT {PRIMARY_GPT}      Sonnet >=2  {_pct(ca, cn)}")
    print(f"  matched controls  Sonnet >=2  {_pct(la, ln)}")
    ratio = (ca / cn) / (la / ln) if cn and ln and la else None
    p = fisher_one_sided(ca, cn - ca, la, ln - la)
    print(f"\n  ratio {ratio if ratio is None else f'{ratio:.3f}'}   one-sided Fisher p = {p:.4f}")
    print(f"  bars: CONFIRM ratio <={CONFIRM_RATIO} and p<{ALPHA};  KILL ratio >={KILL_RATIO}")
    print(f"  VERDICT: {verdict_for(ratio, p)}")

    print("\n" + "=" * 92)
    print("SECONDARY — every stratum, reported whether or not it can resolve")
    print("=" * 92)
    print(f"  {'GPT':>3}  {'cited':>16}  {'controls':>16}  {'ratio':>7}  {'p':>7}")
    for g in sorted({r["gpt"] for r in rows}):
        ca_, cn_ = rate([r for r in cited if r["gpt"] == g])
        la_, ln_ = rate([r for r in ctl if r["gpt"] == g])
        rt = f"{(ca_ / cn_) / (la_ / ln_):7.3f}" if cn_ and ln_ and la_ else "    n/a"
        pv = fisher_one_sided(ca_, cn_ - ca_, la_, ln_ - la_)
        print(f"  {g:>3}  {_pct(ca_, cn_):>16}  {_pct(la_, ln_):>16}  {rt}  {pv:7.4f}")
    ca_, cn_ = rate(cited)
    la_, ln_ = rate(ctl)
    print("\n  pooled (NOT a test — the strata have different base rates, §29.2)")
    print(f"    cited {_pct(ca_, cn_)}   controls {_pct(la_, ln_)}")

    print("\n" + "=" * 92)
    print("TERTIARY — the five self-papers, named in §36 before the labels were bought")
    print("=" * 92)
    for r in sorted(cited, key=lambda r: r["id"]):
        key = (r["case"], r["base"])
        if key not in SELF_PAPERS:
            continue
        print(
            f"  {r['case']:10} {r['id']:14} GPT {r['gpt']}  Sonnet {r['sonnet']}"
            f"   {SELF_PAPERS[key]}"
        )
    print(
        "\n  Descriptive, no bar. Four of the five sit at GPT 1, where Sonnet stays <=2 about\n"
        "  94% of the time anyway — a confirmation there would be the base rate, not the effect."
    )

    print("\n" + "=" * 92)
    print("DIAGNOSTIC — is the control arm actually matched?")
    print("=" * 92)
    for label, rs in (("cited", cited), ("controls", ctl)):
        months = sorted(_month(r["id"]) for r in rs)
        mid = months[len(months) // 2] if months else 0
        print(f"  {label:10} n={len(rs):3d}  median vintage {mid // 12}-{mid % 12 or 12:02d}")
    print(f"  cases: {dict(sorted(Counter(r['case'] for r in cited).items()))}")

    reproduction_check(pop["generating"])
    print(f"\nwritten to {OUT}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true", help="population + hash check, $0")
    ap.add_argument("--report", action="store_true", help="re-derive from saved verdicts, $0")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    args = ap.parse_args()
    load_env()

    pop = build()
    treatment, controls = pop["treatment"], pop["controls"]
    if args.report:
        report(json.loads(OUT.read_text(encoding="utf-8")), pop)
        return 0

    print(f"held-out cited papers: {len(treatment)}   matched controls: {len(controls)}")
    print(f"  excluded as §32.4's generating set: {len(pop['generating'])}")
    if pop["split_labels"]:
        # `*` marks a cited one. None so far — every split label is a control candidate, so
        # the drop costs the control pool depth and costs the treatment arm nothing.
        print(f"  dropped, GPT scored two versions differently: {'; '.join(pop['split_labels'])}")
    if pop["short_strata"]:
        print(f"  strata short of {CONTROLS_PER} controls: {'; '.join(pop['short_strata'])}")
    if pop["no_checkout"]:
        print(f"  no checkout on disk: {', '.join(pop['no_checkout'])}")
    by_gpt = Counter(r["gpt"] for r in treatment)
    print(f"  cited by GPT score: {dict(sorted(by_gpt.items()))}")
    print(f"  ~${(len(treatment) + len(controls)) * 0.011:.2f} of Sonnet")

    cases = sorted({r["case"] for r in treatment + controls})
    contexts, drifted = verify_contexts(cases)
    print(f"prompt-hash check: {len(contexts)} cases reproduce, {len(drifted)} drifted")
    if drifted:
        print(f"  EXCLUDED (clone moved under the cache): {', '.join(sorted(drifted))}")
    if args.dry_run:
        reproduction_check(pop["generating"])
        return 0

    work = [r for r in treatment + controls if r["case"] in contexts]
    papers = fetch_papers(sorted({r["id"] for r in work}))
    rows: list[dict[str, Any]] = []
    for r in work:
        # fetch_papers keys its cache by dedup_id, so a versioned id never matches it directly.
        # Looking up the raw id silently resolved only the 11 papers whose gold-cache id happens
        # to carry no version suffix, and excluded the other 84 as "no metadata" (§37.1).
        meta = papers.get(dedup_id(r["id"]))
        if meta is None:
            print(f"    ! {r['case']}/{r['id']} has no metadata — excluded")
            continue
        try:
            score = second_verdict(
                r["case"], contexts[r["case"]], {"arxiv_id": r["id"], **meta}, args.model
            )
        except Exception as exc:  # noqa: BLE001
            print(f"    ! {r['case']}/{r['id']} failed: {exc}")
            continue
        rows.append({**r, "sonnet": score})
        if len(rows) % 20 == 0:
            print(f"  {len(rows)}/{len(work)}", flush=True)
    OUT.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    report(rows, pop)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
