# Open directions for net@2 after the retrieval frontier closed

*Research record, 2026-08-31. Written the day after NR-51 closed item 12 — the last open
evidence-led lead — on the question: what could still raise net@2 enough to give RepoRadar a
statistical advantage over the Opus 5 comparator? Produced by a 13-agent evidence-mining /
generation / adversarial-review pass over the full results record, seeded with a fresh loss
ledger computed from the shipped-arm artifacts. 17 candidate directions were proposed; 7 died
in adversarial review; 10 rungs survive, sequenced cheapest-decisive-first.*

*This is a record and a plan input, not a licence. Every rung's bar is stated with an effect
size (NR-49's lesson), and no product rung below may ever get a solo paid arm — each is
individually below the ±0.78 bootstrap resolution.*

---

## 1. The target, stated honestly

Per-case, shipped RepoRadar minus Opus 5 (from `evals/opus5_arm.json`):

| mean | sd | ~95% CI | w/l/t |
|---|---|---|---|
| **+0.54** | 6.56 | [−1.57, +2.66] | 18/18/1 |

For the CI to exclude zero: mean **≥ ~+2.11 at n = 37**, or ≥ ~+1.66 at n = 60. At the current
mean, **n ≈ 567** would be needed — power alone is dead. The margin decomposition says where
the +0.54 lives: five cases where Opus 5 over-answers contribute +8.4 paired (105% of the whole
margin); the other 32 are level at −0.06. The cases we *lose* — mat-mlip −14, mat-chgpot −12,
cv −9, llminfer −9, mat-toolkit −7, numerics −6 — are literature-dense science cases where
Opus 5 posts +12…+21 with large, nearly-all-actionable digests.

**The advantage is abstention discipline. To make it statistical, the mean must roughly
quadruple, or the program below must run end to end.**

## 2. The loss ledger (computed 2026-08-31 from shipped artifacts)

**False positives.** 34 non-actionable papers shown across 37 cases — ceiling **+1.84/case** if
all removed cleanly. 33 of 34 are judge-score-1: the boundary region, where judge agreement is
worst. Spread across digest positions (five sit at rank 1) and cohorts (17 core / 6 bio /
11 matsci).

**False negatives.** 439 papers in the shipped pool, judged actionable *for that case* by some
past experiment, not shown. 232 sit inside the top-50 gate window (**+6.27/case** ceiling), 88
inside ranks 1–15. But the reject piles run **0.409** actionable (ranks 1–15, n = 215 judged)
and **0.502** (ranks 16–50, n = 287) — and these are exposure-selected numbers that likely
*overstate* the truth. Blanket loosening loses; recall requires a discriminator better than the
current gate on the reject pile.

**A correction that governs the whole precision family.** The scout memo behind this research
stated the removal break-even as 2/3 and the adversarial pass caught the inversion: removing a
shown paper is worth `E[Δ] = q(+2) + (1−q)(−1) = 3q − 1`, so **a removal filter pays at
FP-fraction > 1/3**, not 2/3. (The 2/3 figure is the break-even for *showing*, 3p − 2.)
Removal filters are twice as feasible as the memo implied. Kept per C-17: the wrong figure is
recorded here rather than erased.

**The above-band free pass (verified in shipped code).** `digest.categorize_papers` applies the
calibrated finescale probability only to papers *at* the band (`llm == triage_threshold`);
papers above it are "trusted on the gate's word" — **llm == 3 papers bypass the P ≥ 2/3 check
entirely**, deliberately and untested. Whether the 34 FPs concentrate there is checkable for $0
from the run artifacts.

**The judge-noise cell.** GPT–Sonnet kappa on the n = 324 band is **0.199**; on 65 shown papers
GPT reads precision 0.852 where Sonnet reads 0.537. The 80 gate-withheld band papers are 74%
actionable *per GPT* (+0.21/paper EV, above break-even) and 26% *per Sonnet-consensus*
(−1.21/paper, far below). Every FP, every rescue target, and most of the margin's moving parts
live inside this cell.

## 3. Rung 1 — the validity gate that runs before any spend

**Second-judge re-scoring of the shipped, EPMC, and Opus 5 arms (~$5–10).** Commit the
labelling rules in writing *before* computing any arm result under them, then re-score the
fully-paid arms per-case and report every margin side by side.

**Coverage, measured 2026-08-31 — this corrects the figure the research pass was assembled
with.** The synthesis quoted "837 cached Sonnet verdicts". The real number is **557 unique
(case, paper) Sonnet verdicts** across `second_judge.json` (200), `second_judge_arm.json` (65)
and `second_judge_band.json` (324) — and they are very unevenly distributed:

| arm | shown papers | Sonnet-covered |
|---|---|---|
| shipped RepoRadar | 306 | **159 (52.0%)** |
| Opus 5 (draw 1, judged picks) | 357 | **21 (5.9%)** |

The cross-judge work was aimed at *our own* band and digest, so it barely touches the
comparator. A like-for-like second-judge margin needs **483 fresh Sonnet verdicts** (147
shipped + 336 Opus 5) — roughly **$5–10**, not the $0–5 the synthesis assumed. Cheap either
way, but worth stating plainly: the top-up is not garnish, it is most of the comparator side of
the test.

### Three margins, all pre-registered, all reported

| label | rule | what it asks |
|---|---|---|
| **GPT** | GPT ≥ 2 | the published margin, +0.54/case |
| **consensus** | GPT ≥ 2 **and** Sonnet ≥ 1 | does the margin survive when both judges must agree? |
| **Sonnet-only** | Sonnet ≥ 2 | what is the margin if we simply swap the judge? |

The **Sonnet-only** reading is worth the same top-up and is in some ways the better test. It
needs **no constructed composite** — no rule about how two judges combine, so nothing to argue
over and nothing to label-shop. It is also the harsher one: Sonnet reads precision **0.537**
where GPT reads 0.852 on the same 65 shown papers, so under net@2's −2 penalty **both arms will
very likely go negative** and the question becomes *who loses less* rather than *who wins* — a
more demanding way for an advantage to survive.

All three come from the same 483 verdicts. Pre-register all three before computing any of them
and report all three regardless of which flatters the result; picking afterwards is exactly the
label-shopping this gate exists to prevent.

The sample frame is the *digest of paid arms*, not the judge cache, so NR-51's
exposure-selection trap is structurally absent. The bar, on the **consensus** margin: within
±0.5/case of the GPT margin with sign preserved, and ≥4 of the 6 big science losses persisting
at ≥50% magnitude. The **Sonnet-only** margin is reported and interpreted but carries no kill
condition of its own — both arms are expected to go negative there, and a bar set on a reading
whose absolute level is dominated by one judge's severity would be measuring the judge. **A sign flip or |Δ| > 0.5/case kills every margin-chasing dollar below and reframes
the paper's comparator claim as judge-relative with the cross-check shown.** Two scope notes
from review: the ±0.5 read is descriptive gating, not a test (the consensus margin's own SE is
~1 at n = 37), and a pass means "robust among available judges", not "objective" — a shared
judge cutoff (NR-43) is undetectable by any two-judge construction.

## 4. The product rungs (each bundle-only; none resolves alone)

| # | direction | realistic | ceiling | first probe | bar (effect size + kill) |
|---|---|---|---|---|---|
| 2 | **Bio-scoped arXiv+EPMC routing** — the PLANS 3c conditional, motivated by C-34 (+0.73 own-papers, −0.19 displacement, +4.17/case concentrated in bio) | **+0.45** | +0.76 | $0 counterfactual recompute from `opus5_arm.json` per-case (both arms already paid); settles the NR-40 `w_embedding`-in-bio rider too | scoped composite ≥ +0.5/case on the recompute; park below +0.35. Kill: non-bio moves < −0.2/case (displacement leak), or confirm arm < +0.35 |
| 3 | **Close the above-band free pass** — extend the finescale P ≥ 2/3 display rule to llm == 3 papers | +0.22 | +0.76 | $0: partition the 34 FPs by `finescale_p` presence; <$1 gpt-4o-mini rescore + offline digest rebuild | kill on the spot if < 8 FPs are above-band; screened Δ ≥ +0.2/case at removal FP-fraction ≥ 0.45 (margin over the correct 1/3 break-even) AND ≥ 50% removal purity under the Sonnet cross-check |
| 4 | **Bibliography-seeded hop on the losing science trio** (mat-mlip, mat-chgpot, mat-toolkit) — the held item's stated reopen condition | +0.20 | +0.40 | $0: seed counts (instant kill if < 7 arXiv-indexed bibliography entries), then hop reach against Opus 5's judged actionables — no LLM, no judge, no cache precision | reach ≥ 30% of the fixed labeled targets on ≥ 2 of 3 repos; quote the probe's own science-cohort figure, never the old benchmark's 0.44 |
| 5 | **Per-hypothesis quota pooling at constant volume** — replace the fused HyDE union cut with per-hypothesis quotas; not the closed volume lever (volume held fixed; the *allocation* changes) | +0.35 | +1.50 | ~$0.10 + CPU: re-cut the recorded per-hypothesis rankings, witness-reach + rank simulation | reach gain ≥ +0.03 absolute over the fused cut at equal volume (above NR-46's +0.0577 redraw noise is impossible for one arm — so demand the *rank* profile improves too: quota papers' top-50 share ≥ fused's) |
| 6 | **Density-conditional window** — extend `rr_window` 15 → 25 only where the finescale probability stays ≥ 2/3 deep into the ranking (the global version is killed; this is finescale-governed) | +0.30 | +0.46 | ~$0.50 offline rebuild + ~$5 fresh verdicts on newly exposed papers (never cache-scored) | screened Δ ≥ +0.25/case with every newly shown paper fresh-judged; kill if the fresh-judged precision of extension papers < 0.70 |
| 7 | **FP-filter head-to-head on the shown set** — grounded-claim verification vs boundary-contrastive prompt, same 34-FP pool | +0.25 | +1.84 | ~$3 + ~$5 fresh verdicts | winner removes ≥ 6 of 34 FPs at ≥ 50% purity on fresh judging (comfortably above the 1/3 break-even); kill both if neither clears |
| 8 | **Reject-pile rescue** — the 0-9 finescale read pointed at in-window llm == 1 papers and ranks-16-50 admits; counts-only screen, no cache-precision EV computed anywhere | +0.30 | +2.38 | ~$13–15 with fresh judging of every candidate rescue | rescue set ≥ 8 papers at fresh-judged precision ≥ 0.70 (against reject piles measuring 0.41–0.50); kill below either number |

Shared-pool accounting, adopted ladder-wide: the 34 FPs, the withheld/admitted ranks-16-50
population, and the 66 six-case discovery-dead papers each count **once** in any combined
estimate; rungs claiming the same pool are unioned, not summed.

## 5. The measurement rungs (legitimate, and labelled as what they are)

| # | direction | what it buys | cost | bar |
|---|---|---|---|---|
| 9 | **Comparator draw averaging** | sd 6.56 → ~4.8–5.6 *if* the $0 decomposition shows ≥ 25% of variance is draw noise (C-7: both sides are single draws) | $0 decomposition; ~$100–150 for two pinned Opus 5 draws | kill if reducible share < 25% |
| 10 | **Pre-registered benchmark expansion to n = 60** with a frozen sampling frame written before any case is collected | half-width 2.11 → ~1.66 (× draw averaging → ~1.2–1.4) | ~$1k+ compute plus unpriced curation labor | frame document first; group-sequential futility interim at n = 48 requiring ≥ +0.8/case |

**Legitimate:** the pre-committed consensus label; draw averaging under pinned hypothesis sets
and a fixed retry policy; the frozen sampling frame with a futility boundary; reporting absolute
numbers as judge-relative with the Sonnet cross-check.
**Illegitimate (recorded so nobody has to rediscover it):** post-hoc stratification or cohort
re-cuts; trimmed means (verified a no-op here anyway: +0.586 vs +0.541); case-shopping
abstention-shaped repos into the expansion; label selection after seeing arm results; any
kill/pass bar computed on exposure-selected judge-cache precisions (NR-51: a different
quantity, not a weak estimate).

## 6. What died in review

- **Uncapped calibrated digest** — a union of rungs 3, 6 and 8 wearing one name; its +8.11
  ceiling summed multiply-claimed pools as if disjoint.
- **Finescale-everywhere** — third filing of the same composite; nothing testable not owned by
  an earlier, better-specified rung.
- **Finescale-on-rejects (separate listing)** — folded into rung 8, whose counts-only protocol
  is the strictly better half.
- **Distribution-shape finescale + map refit** — its motivating "miscalibration" (shown FPs at
  mean p 0.793) is a conditioning artifact of any AUC-0.67 filter; its $0 sweep is a threshold
  tuned on the cache.
- **The EPMC + everything "package" arm** — the +1.85 sum added unscreened optimistic
  realistics, double-counted pools, and ignored C-34's measured displacement.
- **Global window extension 15 → 25/50** — killed with evidence already in hand: blanket
  release of the population it reaches is bounded at +0.35–0.46 face value with the binomial CI
  containing break-even; its own pre-registered +1.0 bar fires before the screen runs.
- **Density-conditional window as a standalone** — the null-record adversary's citation of the
  closed adaptive-digest entry was scope-rebutted by the other two (that item predicted
  *precision* from composition signals; this rung is governed by finescale probability), so it
  survives only in its finescale-governed form as rung 6.

## 7. The combined arithmetic, plainly

Deduplicated realistic effects, if every screen passes: bio-EPMC +0.45, FP-removal cell ~+0.3,
rescue∪window ~+0.4 (the most label-fragile term — **negative** under the consensus label),
retrieval union ~+0.4. Sum ~+1.4–1.55; after the C-34 shared-gate-slot discount (a third source
measured net-negative through exactly these 50 slots), **~+1.1–1.3 of new margin, landing mean
+1.6–1.8 best case.**

| landing | n=37, sd 6.56 (need 2.11) | n=37, sd 4.8 (need 1.55) | n=60, sd 6.56 (need 1.66) | n=60, sd 4.8–5.6 (need 1.21–1.41) |
|---|---|---|---|---|
| +1.6–1.8 best case | fails | clears at the top end | borderline | **clears** |
| +1.2–1.3 modal (≈50% screen pass-through) | fails | fails | fails | coin-flip |

**No single surviving direction plausibly puts the CI past zero, and no subset short of the
full sequenced program does.** The only closing arithmetic is joint and strictly ordered:
(1) the consensus-label gate passes; (2) bio-EPMC ships off its recompute + unconditional
confirm (→ ~+1.0 mean); (3) the ~$25–35 of screens run and ≥ 2 of the four bundle cells
survive; (4) survivors confirm in **one** bundled paid arm, pre-registered at package
≥ +1.0/case with the abstention cases byte-identical as an automatic kill; (5) draw averaging
clears its 25% bar and two pinned Opus 5 draws are bought; (6) the frozen-frame n = 60
expansion makes the final claim at ≥ +1.2/case. **Miss any gate and the program stops there,
cheaply.**

The modal outcome — screens killing at their historical rate — is that the margin stays
unresolvable, and the honest fallback is the claim the paper already supports: judge-relative,
cohort-decomposed, with abstention discipline as the measured mechanism.
