"""The pooled witness set (v2), and the size-invariant measures that replace "recall". [P16]

The gold set (`evals/gold_targets.json`) is one draw of one searcher's discovery
distribution, filtered by the judge. Coverage of it was published as "recall", and it is not
recall: the judge marks ~2% of random arXiv papers actionable, so the true positive set is
order 10^3-10^4 papers per repository and recall against it is both unmeasurable and not the
quantity anyone wants. What the set actually provides is per-member *certificates* -- this
paper is findable and judged actionable -- i.e. pooled relevance judgments, TREC-style, with
a pool of one system and one draw.

This module builds the honest version:

* **Pool the sources that are already judged.** At $0: `cli` baseline picks, `api` baseline
  picks (a different system -- P13), RepoRadar's own returned papers from the headline run,
  the git-history adoptions (the only model-free source; judged against the repo as it was
  *before* adoption, which is a different judging context and is recorded as such), and every
  judged `gold_spread` redraw. Every member carries its judge score and full source
  provenance -- a witness found by two sources is one witness with two sources.

* **A source is a CONFIGURATION, not a file.** The redraws arrive labelled by what produced
  them -- `cli-redraw`, `cli-redraw@30`, `cli-v2@30` -- because the prompt and the turn cap
  both change what gets found, and a reach probability pooled across them would describe no
  searcher that exists. `cli` stays exactly the frozen gold-set derivation; nothing is folded
  into it. Labels are DISCOVERED from the data rather than declared, so a configuration that
  has been drawn always appears in the reach table instead of being silently absent.

* **Reach is a probability, not a count ratio.** For each non-self source S,
  ``P(witness in candidate pool | witness drawn from S)`` with a Wilson interval. Growing
  the witness set tightens the interval instead of degrading the number; a *different*
  source scoring lower is the pooling-bias measurement, not a confound. RepoRadar-sourced
  witnesses are excluded from every reach denominator (leave-one-source-out): they are in
  the pool by construction, and grading a system against a pool containing its own finds
  is how pooled evaluation flatters incumbents.

* **At the digest, regret replaces coverage.** A missed witness costs only if it would
  displace something worse in the shown 15: +1 for filling an empty slot with an actionable
  witness, +3 for swapping one in over a shown paper judged < 2. Bounded by the digest
  width, so a growing witness set can only *reveal* headroom, never inflate it.

* **The capture curve says how incomplete the set still is, and the answer is "very".**
  Two estimators, kept apart because their units differ. The P15 probe gives three arms over
  five cases at PICK level (mostly unjudged): Chao1 >= 34.3 against 24 observed. The P17
  redraws give three occasions over 19 cases at WITNESS level, every unit judged: 92 observed,
  68 of them seen in exactly one draw, **Chao1 >= 236.5**. Two and a half times what three
  draws of the same searcher found, from a single configuration -- which is the quantitative
  case for pooling more searchers rather than redrawing this one.

**Reach here is membership in a frozen candidate pool** (`pool-wemb`, the headline's 25-case
pool; `pool-cohort3`, the 37-case pool of the scientific session) -- "did the shipped
configuration's collection step fetch this paper at all". It is deliberately NOT the
published 43/56, which measured the hop/HyDE channels' top-1000 in isolation; the two
numbers answer different questions and must not be compared.

    uv run python evals/witness_set.py            # derive, print, rewrite the artifact
    uv run python evals/witness_set.py --check    # $0, diff against the committed artifact
    uv run python evals/witness_set.py --report   # $0, re-print from the committed artifact
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import baseline as baseline_mod  # noqa: E402
from diagnose_pool import JUDGE, _judge_stem  # noqa: E402

from reporadar.paper_id import canonical_ref, dedup_id  # noqa: E402

EVALS = Path(__file__).resolve().parent
FROZEN = EVALS / "witness_set.json"
WORK = EVALS / ".work"

ACTIONABLE = 2
DIGEST_WINDOW = 15

# The headline run: source of the `reporadar` witnesses and of the shown sets for regret.
# Named, not globbed -- "the most recent run" is how an artifact re-points itself.
HEADLINE_RUN = (
    EVALS / "results" / "judge-gpt-5.5-frozenpool-bigrams_verified-wemb1.5-20260815T225831Z.json"
)
ADOPTIONS = WORK / "adoptions.json"
TURN_PROBE = EVALS / "turn_budget_probe.json"
POOLS = ("pool-wemb", "pool-cohort3")

# Leave-one-source-out: a source may grade RepoRadar's reach unless RepoRadar produced it.
# `reporadar` witnesses are in the pool by construction, and grading a system against a pool
# containing its own finds is how pooled evaluation flatters incumbents. They stay in the set
# — a future system graded against this artifact should face RepoRadar's finds too.
#
# This is a list of the SELF sources rather than an allowlist of the others because the
# source labels are no longer a fixed set: `gold_spread` draws mint one per configuration
# (`cli-redraw`, `cli-redraw@30`, `cli-v2@30`, ...). Under an allowlist a new label would be
# silently dropped from every reach denominator — a witness that grades nothing, invisibly,
# which is the void-not-null failure this project keeps paying for. The cost of the inversion
# is that a future *self* source has to be added here; `tests/test_witness_set.py` pins the
# full label set so that addition cannot be forgotten quietly.
SELF_SOURCES = ("reporadar",)
# Display order for `report`; labels outside it are printed after, discovered from the data.
KNOWN_ORDER = ("cli", "cli-redraw", "cli-v2", "api", "adoption")


def is_non_self(source: str) -> bool:
    return source not in SELF_SOURCES


def non_self_sources(sources: list[str]) -> list[str]:
    return [s for s in sources if is_non_self(s)]


def draw_source_label(row: dict[str, Any]) -> str:
    """The source label for one `gold_spread` row: a searcher, not a file.

    A draw is labelled by the CONFIGURATION that produced it, because that is what a
    discovery distribution is. Two things vary and both change what gets found:

    * the prompt — v2 is allowed to recommend journal and bioRxiv papers, so it searches a
      region v1 structurally cannot reach;
    * the turn cap — a 30-turn agent searches longer and surfaces different papers.

    So they are separate labels, and `reach` reports each separately with its own interval
    rather than pooling them into one number that describes no searcher that exists. Rows
    written before the cap was recorded carry no `max_turns`, and the shipped cap is what
    they ran at — the same default `gold_spread.report` uses, so the two agree by
    construction rather than by coincidence.

    None of these is `cli`. `cli` means the stored gold-set derivation exactly, and
    `tests/test_witness_set.py` asserts that equality — a redraw folded in there would
    silently redefine the set every published recall figure divides by.
    """
    version = row.get("prompt_version", baseline_mod.DEFAULT_PROMPT_VERSION)
    cap = row.get("max_turns", baseline_mod.DEFAULT_MAX_TURNS)
    stem = "cli-redraw" if version == baseline_mod.DEFAULT_PROMPT_VERSION else f"cli-{version}"
    return stem if cap == baseline_mod.DEFAULT_MAX_TURNS else f"{stem}@{cap}"


def _draw_rows() -> list[tuple[str, str, dict[str, Any]]]:
    """(draw, case, row) for every judged `gold_spread` row, across all prompt versions.

    The artifact paths come from `gold_spread.out_path` rather than being rebuilt here: two
    modules deriving the same filename is how this project has repeatedly ended up with two
    answers to one question (C-12, C-14, C-31). The import is deferred because `gold_spread`
    pulls in the arXiv client, which costs ~1.9 s that the rest of this module never needs.

    **Both `ok` and `partial` rows count, which is the opposite of what `gold_spread` does**,
    and the difference is not an inconsistency. `gold_spread` computes a RATIO — reproduced
    over frozen — so a row whose target set is a floor biases its denominator downward and
    manufactures instability that is really an arXiv 429. A witness set is a UNION: a floor
    contributes fewer certificates and claims nothing about what it missed. The set has never
    claimed completeness — that is what the capture curve is for.
    """
    from gold_spread import out_path

    rows: list[tuple[str, str, dict[str, Any]]] = []
    for version in sorted(baseline_mod.PROMPTS):
        path = out_path(version)
        if not path.is_file():
            continue
        artifact = json.loads(path.read_text(encoding="utf-8"))
        stored = artifact.get("prompt_version", baseline_mod.DEFAULT_PROMPT_VERSION)
        if stored != version:
            # `out_path` addressed this file as `version` and the file says otherwise. Reading
            # it anyway would file one searcher's draws under another's label.
            raise SystemExit(f"! {path.name} says prompt {stored!r}, not {version!r}")
        for key, row in artifact.get("results", {}).items():
            if row.get("status") not in ("ok", "partial"):
                continue
            draw, case = key.split("/", 1)
            rows.append((draw, case, row))
    return rows


def _judge_score(case: str, paper_id: str) -> int | None:
    for verdict in (JUDGE / case).glob(f"{_judge_stem(paper_id)}*.json"):
        return int(json.loads(verdict.read_text(encoding="utf-8"))["score"])
    return None


def _baseline_picks(mode: str) -> dict[str, list[str]]:
    """{case: picks} from the cached baselines, replayed the way the harness scores them."""
    out: dict[str, list[str]] = {}
    for cache in sorted((EVALS / "cache" / "baseline" / mode).glob("*.json")):
        data = json.loads(cache.read_text(encoding="utf-8"))
        if data.get("status") != "ok":
            continue
        raw = data.get("raw") or ""
        ids, _ = baseline_mod._parse_recommendations(raw)
        if not ids and not baseline_mod._has_answer_block(raw):
            ids = list(data.get("ids") or [])  # the C-25 fallback
        out[cache.stem] = [dedup_id(i) for i in ids]
    return out


def _wilson(hits: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval -- sane at the small n and extreme p this data actually has."""
    if n == 0:
        return (0.0, 0.0)
    p = hits / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


# ── the set ────────────────────────────────────────────────────────────────


def gather_witnesses() -> dict[str, dict[str, dict[str, Any]]]:
    """{case: {id: {"judge": score, "sources": [...]}}} -- judged >= 2 only.

    Judge scores for `cli`/`api`/`reporadar` witnesses come from the shared verdict cache
    (one verdict per (case, paper), shared across sources -- so multi-source witnesses
    cannot carry contradictory scores). Adoption scores come from the mining artifact and
    were judged against the repository at t0, before the adoption; that is a different
    judging context, which is why the source label matters.
    """
    witnesses: dict[str, dict[str, dict[str, Any]]] = {}

    def add(case: str, paper_id: str, source: str, score: int) -> None:
        if score < ACTIONABLE:
            return
        entry = witnesses.setdefault(case, {}).setdefault(paper_id, {"judge": score, "sources": []})
        if source not in entry["sources"]:
            entry["sources"].append(source)

    # The cli source is THE gold-set derivation, not a reimplementation of it. A first
    # version replayed the caches here and silently dropped `rag`'s two ids-only orphans
    # (73 witnesses against the gold set's 75) -- a second implementation of "what did the
    # baseline pick" drifting from the first, which is the C-12/C-14 shape exactly.
    from build_hop_pool import resolve_targets

    for case, ids in resolve_targets().items():
        for pid in ids:
            # Every gold target has a verdict >= 2 by construction (that is what makes it
            # gold), so a None here would mean the derivation and the verdict cache
            # disagree -- skip loudly rather than fabricate a score.
            score = _judge_score(case, pid)
            if score is None:
                print(f"! gold target {case}/{pid} has no judge verdict; skipped")
                continue
            add(case, dedup_id(pid), "cli", score)

    for case, ids in _baseline_picks("api").items():
        for pid in ids:
            score = _judge_score(case, pid)
            if score is not None:
                add(case, pid, "api", score)

    if HEADLINE_RUN.is_file():
        for entry in json.loads(HEADLINE_RUN.read_text(encoding="utf-8")):
            for paper in entry["returned"]["reporadar_toppicks"]:
                add(entry["case"], dedup_id(paper["arxiv_id"]), "reporadar", paper["judge_score"])

    if ADOPTIONS.is_file():
        for row in json.loads(ADOPTIONS.read_text(encoding="utf-8")):
            if row.get("usable") and not row.get("self_cited") and row.get("judge") is not None:
                add(row["case"], dedup_id(row["id"]), "adoption", int(row["judge"]))

    # The `gold_spread` redraws. Each is an independent draw of a named configuration, and
    # every pick in them was already judged — so they are witnesses that cost nothing more to
    # collect, and the largest single expansion available to this set.
    #
    # The row's own `scores` enumerate the candidates; the SHARED verdict cache decides the
    # value. That asymmetry is the invariant this module is built on: one verdict per
    # (case, paper), so a witness found by four sources cannot carry four opinions. Trusting
    # the row's copy instead would let a witness's score depend on which source reached
    # `add()` first, which is not a property of the paper.
    for _draw, case, row in _draw_rows():
        for pid in row.get("scores") or {}:
            canonical = canonical_ref(pid)
            score = _judge_score(case, canonical)
            if score is None:
                # A judged pick whose verdict the cache cannot produce means the draw and the
                # cache disagree about what was scored. Skip loudly rather than fall back to
                # the row's copy, which would hide the drift behind a plausible number.
                print(f"! {case}/{canonical} was judged in a draw but has no cached verdict")
                continue
            add(case, canonical, draw_source_label(row), score)

    return witnesses


# ── reach: P(witness in pool), per source, leave-one-source-out ────────────


def _pool_ids(pool: str) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for f in sorted((WORK / pool).glob("*.json")):
        data = json.loads(f.read_text(encoding="utf-8"))
        out[f.stem] = {dedup_id(c["arxiv_id"]) for c in data["candidates"]}
    return out


def source_labels(witnesses: dict[str, dict[str, dict[str, Any]]]) -> list[str]:
    """Every source label present, in display order: known ones first, then the rest sorted.

    Discovered rather than declared, so a configuration that has been drawn always appears in
    the reach table. A label the code did not anticipate showing up with a small `n` is
    informative; a label silently absent is not distinguishable from a source that found
    nothing, which is the distinction this project has paid for repeatedly.
    """
    found = {s for papers in witnesses.values() for m in papers.values() for s in m["sources"]}
    return [s for s in KNOWN_ORDER if s in found] + sorted(found - set(KNOWN_ORDER))


def reach(witnesses: dict[str, dict[str, dict[str, Any]]], pool: str) -> dict[str, Any] | None:
    pools = _pool_ids(pool)
    if not pools:
        return None
    labels = [s for s in source_labels(witnesses) if is_non_self(s)]
    rows: dict[str, dict[str, int]] = {s: {"n": 0, "reached": 0} for s in labels}
    rows["pooled_non_self"] = {"n": 0, "reached": 0}
    for case, papers in witnesses.items():
        if case not in pools:
            continue
        for pid, meta in papers.items():
            non_self = non_self_sources(meta["sources"])
            if not non_self:
                continue  # reporadar-only: in the pool by construction, grades nothing
            hit = pid in pools[case]
            for source in non_self:
                rows[source]["n"] += 1
                rows[source]["reached"] += hit
            rows["pooled_non_self"]["n"] += 1
            rows["pooled_non_self"]["reached"] += hit
    out: dict[str, Any] = {"pool": pool, "cases_covered": len(pools), "graded_by": labels}
    for label, r in rows.items():
        lo, hi = _wilson(r["reached"], r["n"])
        out[label] = {
            "n": r["n"],
            "reached": r["reached"],
            "p": round(r["reached"] / r["n"], 3) if r["n"] else None,
            "ci": [round(lo, 3), round(hi, 3)],
        }
    return out


# ── regret: what the misses would actually buy at the digest ───────────────


def regret(witnesses: dict[str, dict[str, dict[str, Any]]]) -> dict[str, Any] | None:
    """Per case: net@2 the shown 15 forgoes, using only unshown non-self witnesses.

    +1 per empty digest slot filled with an actionable witness; +3 per shown paper judged
    < 2 displaced by one (its -2 becomes a +1). Nothing else counts: a witness that would
    displace a shown 2 with a 3 changes net@2 by zero, which matches the product truth.
    """
    if not HEADLINE_RUN.is_file():
        return None
    run = json.loads(HEADLINE_RUN.read_text(encoding="utf-8"))
    per_case: dict[str, dict[str, Any]] = {}
    for entry in sorted(run, key=lambda e: e["case"]):
        case = entry["case"]
        shown = entry["returned"]["reporadar_toppicks"]
        shown_ids = {dedup_id(p["arxiv_id"]) for p in shown}
        shown_scores = [int(p["judge_score"]) for p in shown]
        actual = sum(1 if s >= ACTIONABLE else -2 for s in shown_scores)
        avail = sum(
            1
            for pid, meta in witnesses.get(case, {}).items()
            if pid not in shown_ids and non_self_sources(meta["sources"])
        )
        free = max(0, DIGEST_WINDOW - len(shown_scores))
        fills = min(free, avail)
        swaps = min(sum(1 for s in shown_scores if s < ACTIONABLE), avail - fills)
        gain = fills * 1 + swaps * 3
        per_case[case] = {
            "actual_net2": actual,
            "witnesses_available": avail,
            "fills": fills,
            "swaps": swaps,
            "regret": gain,
        }
    n = len(per_case)
    return {
        "run_file": HEADLINE_RUN.name,
        "window": DIGEST_WINDOW,
        "per_case": per_case,
        "mean_actual_net2": round(sum(r["actual_net2"] for r in per_case.values()) / n, 2),
        "mean_regret": round(sum(r["regret"] for r in per_case.values()) / n, 2),
    }


# ── capture: how incomplete is the witness set itself? ─────────────────────


def capture(witnesses: dict[str, dict[str, dict[str, Any]]]) -> dict[str, Any]:
    """Chao1 over the three cli draws (pick-level), plus the source-overlap histogram.

    The draw-level unit is a PICK, mostly unjudged -- estimating the size of the
    cli-findable pick population, not of the witness set. The overlap histogram is
    witness-level and descriptive only: sources are different distributions, so
    multi-source capture estimators do not apply to it.
    """
    out: dict[str, Any] = {}
    if TURN_PROBE.is_file():
        probe = json.loads(TURN_PROBE.read_text(encoding="utf-8"))
        counts: dict[tuple[str, str], int] = {}
        cases = []
        for r in probe["cases"]:
            if r["cached"].get("status") != "ok":
                continue  # no first draw; the rescue cases have only two occasions
            cases.append(r["case"])
            for arm in ("cached", "control", "treat"):
                for pid in {dedup_id(i) for i in r[arm].get("ids") or []}:
                    counts[(r["case"], pid)] = counts.get((r["case"], pid), 0) + 1
        s_obs = len(counts)
        f1 = sum(1 for c in counts.values() if c == 1)
        f2 = sum(1 for c in counts.values() if c == 2)
        chao1 = s_obs + (f1 * f1 / (2 * f2) if f2 else f1 * (f1 - 1) / 2)
        out["cli_draws"] = {
            "cases": sorted(cases),
            "occasions": 3,
            "unit": "pick (mostly unjudged)",
            "s_obs": s_obs,
            "f1": f1,
            "f2": f2,
            "chao1_lower_bound": round(chao1, 1),
        }
    # The same estimator at WITNESS level, over the `gold_spread` redraws. Strictly better
    # evidence than the block above -- every unit is judged, and it covers the 25-case cohort
    # rather than 5 cases -- but it is a different quantity, so it is a separate entry rather
    # than a replacement. Chao1 over picks estimates how many papers this searcher COULD name;
    # Chao1 over witnesses estimates how many it could name that are actually actionable.
    #
    # Occasions are draws of ONE configuration. Pooling a 12-turn and a 30-turn draw as two
    # occasions of one searcher would understate f1 -- a paper only the longer run can find
    # looks like a rare capture of the shorter one rather than out of its reach entirely --
    # and Chao1 is driven by exactly that count.
    by_config: dict[str, dict[str, set[tuple[str, str]]]] = {}
    for draw, case, row in _draw_rows():
        seen = by_config.setdefault(draw_source_label(row), {}).setdefault(draw, set())
        for pid in row.get("targets") or []:
            seen.add((case, canonical_ref(pid)))
    redraws: dict[str, Any] = {}
    for label, draws in sorted(by_config.items()):
        if len(draws) < 2:
            continue  # one occasion cannot estimate what a second would have added
        counts: dict[tuple[str, str], int] = {}
        for members in draws.values():
            for key in members:
                counts[key] = counts.get(key, 0) + 1
        s_obs = len(counts)
        f1 = sum(1 for c in counts.values() if c == 1)
        f2 = sum(1 for c in counts.values() if c == 2)
        redraws[label] = {
            "occasions": len(draws),
            "unit": "witness (judged >= 2)",
            "cases": len({c for m in draws.values() for c, _ in m}),
            "s_obs": s_obs,
            "f1": f1,
            "f2": f2,
            "chao1_lower_bound": round(
                s_obs + (f1 * f1 / (2 * f2) if f2 else f1 * (f1 - 1) / 2), 1
            ),
        }
    if redraws:
        out["redraws"] = redraws

    overlap: dict[int, int] = {}
    for papers in witnesses.values():
        for meta in papers.values():
            k = len(meta["sources"])
            overlap[k] = overlap.get(k, 0) + 1
    out["source_overlap"] = {str(k): v for k, v in sorted(overlap.items())}
    return out


# ── assembly ───────────────────────────────────────────────────────────────


def build() -> dict[str, Any]:
    witnesses = gather_witnesses()
    by_source: dict[str, int] = {}
    for papers in witnesses.values():
        for meta in papers.values():
            for s in meta["sources"]:
                by_source[s] = by_source.get(s, 0) + 1
    return {
        "_comment": (
            "Pooled witness set (v2) with per-source provenance, reach probabilities "
            "(leave-one-source-out), digest regret, and a capture estimate. Derived by "
            "evals/witness_set.py; pinned by tests/test_witness_set.py. NOT a recall "
            "denominator: see the module docstring for what each number means."
        ),
        "n_witnesses": sum(len(p) for p in witnesses.values()),
        "n_cases": len(witnesses),
        "by_source": dict(sorted(by_source.items())),
        "sources": source_labels(witnesses),
        "n_non_self": sum(
            1
            for papers in witnesses.values()
            for meta in papers.values()
            if non_self_sources(meta["sources"])
        ),
        "witnesses": {
            case: {pid: witnesses[case][pid] for pid in sorted(witnesses[case])}
            for case in sorted(witnesses)
        },
        "reach": [r for pool in POOLS if (r := reach(witnesses, pool))],
        "regret": regret(witnesses),
        "capture": capture(witnesses),
    }


def report(data: dict[str, Any]) -> None:
    print(
        f"witness set: {data['n_witnesses']} witnesses / {data['n_cases']} cases   "
        f"by source: {data['by_source']}   non-self: {data['n_non_self']}"
    )
    for r in data.get("reach") or []:
        print(f"\nreach into {r['pool']} ({r['cases_covered']} case(s) with a pool):")
        for label in (*(r.get("graded_by") or []), "pooled_non_self"):
            row = r.get(label)
            if not row or not row["n"]:
                continue
            print(
                f"  {label:<16} {row['reached']:>3}/{row['n']:<3} "
                f"p={row['p']:.3f}  CI [{row['ci'][0]:.3f}, {row['ci'][1]:.3f}]"
            )
    reg = data.get("regret")
    if reg:
        worst = sorted(reg["per_case"].items(), key=lambda kv: -kv[1]["regret"])[:5]
        print(
            f"\nregret@{reg['window']} vs {reg['run_file'][:40]}...: "
            f"mean {reg['mean_regret']:+.2f} net@2/case on top of {reg['mean_actual_net2']:+.2f}"
        )
        for case, row in worst:
            if row["regret"]:
                print(
                    f"  {case:<17} +{row['regret']} "
                    f"({row['fills']} fill(s), {row['swaps']} swap(s), "
                    f"{row['witnesses_available']} witnesses unshown)"
                )
    cap = data.get("capture") or {}
    if "cli_draws" in cap:
        c = cap["cli_draws"]
        print(
            f"\ncapture, 3 cli draws over {len(c['cases'])} case(s) [{c['unit']}]: "
            f"S_obs={c['s_obs']}  f1={c['f1']}  f2={c['f2']}  "
            f"Chao1 >= {c['chao1_lower_bound']}"
        )
    for label, c in (cap.get("redraws") or {}).items():
        print(
            f"capture, {c['occasions']} {label} draw(s) over {c['cases']} case(s) "
            f"[{c['unit']}]: S_obs={c['s_obs']}  f1={c['f1']}  f2={c['f2']}  "
            f"Chao1 >= {c['chao1_lower_bound']}"
        )
    print(f"source-overlap histogram (witness-level): {cap.get('source_overlap')}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Build/check the pooled witness set (v2).")
    ap.add_argument("--check", action="store_true", help="$0: diff against the artifact.")
    ap.add_argument("--report", action="store_true", help="$0: re-print from the artifact.")
    args = ap.parse_args()

    if args.report:
        if not FROZEN.is_file():
            print(f"no artifact at {FROZEN}")
            return 1
        report(json.loads(FROZEN.read_text(encoding="utf-8")))
        return 0

    built = build()
    if args.check:
        if not FROZEN.is_file():
            print(f"! {FROZEN.name} missing; run without --check to write it.")
            return 1
        stored = json.loads(FROZEN.read_text(encoding="utf-8"))
        if stored.get("witnesses") != built["witnesses"]:
            print("! the witness set moved; re-run without --check and read the diff.")
            return 1
        print(f"{FROZEN.name}: witnesses match.")
        return 0

    FROZEN.write_text(json.dumps(built, indent=1) + "\n", encoding="utf-8")
    report(built)
    print(f"\nwrote {FROZEN.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
