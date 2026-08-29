"""Item 6: do OpenAlex Topics order the gate-admitted band? $0, no LLM, no judge. [NR-44]

From "Topic Is Not Agenda" (arXiv:2605.07158), reduced to its cheapest testable form. The gate
is near-binary and its score-2 band hides a 0.00-1.00 per-repository precision spread; §7
established that ordering that band is where any remaining headroom lives. The fine-scale
rescore orders it at **0.760 on this testbed at the judge==3 target** (its headline 0.841 is
pooled across three testbeds and is not the like-for-like number). This asks whether a free
community signal does anything at all.

**Pre-registered bar: AUC >= 0.65 within the score-2 band.** For reference, the NR-21 metadata
family sits at 0.585.

**The design, and the one thing it must not do.** Scoring a paper by its distance to the
repository's *judged-actionable* papers would measure the leak, not the signal. The community
is therefore defined from the repository itself: `band_testbeds.repo_block` -- the gate-time
description, keywords plus 300 characters of prose -- pushed through OpenAlex's own text
classifier. Paper side: each band paper's `primary_topic`, resolved by arXiv DOI. Nothing on
either side touches a judge label.

**Resolution was checked before anything was computed**, because attaching topics to the wrong
works would produce a healthy-looking AUC over noise -- the failure this project has shipped
twice (the 21% biomedical share that read two null fields; the pool scanner that read 1,250
papers as 0). Over 15 band papers whose titles we already hold, OpenAlex returned 13 correct
matches, 0 mismatches and 2 not-found.

**The testbed cannot resolve the bar, and that is a finding about the item.** PLANS describes
a "602-paper labelled set"; the score-2 band inside it is **108 papers with 41 judge==3
positives**, and three of the twelve cases contribute none at all while `peft` (32) and
`diffusion` (22) supply half. At that size the Hanley-McNeil standard error near AUC 0.65 is
about 0.06, so a 95% interval spans roughly +/-0.11: a measured 0.65 would cover both the bar
and the 0.585 null. The probe still runs -- it costs nothing and a clear miss is still an
answer -- but "passes the bar" was never available to it, and the interval is reported so
nobody reads a point estimate as a verdict.
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

EVALS = Path(__file__).resolve().parent
sys.path.insert(0, str(EVALS))
sys.path.insert(0, str(EVALS.parent / "src"))

import band_testbeds as tb  # noqa: E402

FROZEN = EVALS / "topic_community_probe.json"
CACHE = EVALS / ".work" / "openalex_topics.json"
BAR = 0.65
UA = {"User-Agent": "reporadar-eval (mailto:raimondas@gmail.com)"}
BOOTSTRAP = 4000
SEED = 20260829


def _get(url: str) -> Any:
    key = os.environ.get("OPENALEX_API_KEY")
    if key:
        url += ("&" if "?" in url else "?") + "api_key=" + key
    for attempt in range(3):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=45) as r:
                return json.load(r)
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return None
            if attempt == 2:
                raise
            time.sleep(2 * (attempt + 1))
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2 * (attempt + 1))
    return None


def topic_of(arxiv_id: str, cache: dict[str, Any]) -> dict[str, Any] | None:
    """A paper's OpenAlex primary topic, or None when OpenAlex does not hold it.

    A paper OpenAlex has never seen is **absent**, not un-matched: it is scored as missing and
    excluded from the AUC rather than handed a 0, because a 0 would say "wrong community" when
    what we know is "no data" -- and this project's ledger is mostly that mistake.
    """
    if arxiv_id in cache:
        return cache[arxiv_id]
    w = _get(f"https://api.openalex.org/works/doi:10.48550/arXiv.{arxiv_id}")
    pt = (w or {}).get("primary_topic") or None
    out = (
        None
        if not pt
        else {
            "id": pt.get("id"),
            "display_name": pt.get("display_name"),
            "subfield": (pt.get("subfield") or {}).get("display_name"),
            "field": (pt.get("field") or {}).get("display_name"),
        }
    )
    cache[arxiv_id] = out
    time.sleep(0.12)
    return out


def repo_topics(case: str, cache: dict[str, Any]) -> list[dict[str, Any]]:
    """The repository's own community, from its gate-time description. No labels involved."""
    key = f"__repo__{case}"
    if key in cache:
        return cache[key]
    text = tb.repo_block(case)
    q = urllib.parse.urlencode({"title": text[:900]})
    # OpenAlex returns 500 on some repository blocks (`systems` at every length tried). A
    # case whose community cannot be established is EXCLUDED and counted, never scored 0 --
    # 0 would assert "wrong community" where we only know "no answer".
    try:
        d = _get(f"https://api.openalex.org/text/topics?{q}") or {}
    except urllib.error.HTTPError:
        d = {}
    got = []
    for t in d.get("results") or ([d["primary_topic"]] if d.get("primary_topic") else []):
        got.append(
            {
                "id": t.get("id"),
                "display_name": t.get("display_name"),
                "subfield": (t.get("subfield") or {}).get("display_name"),
                "field": (t.get("field") or {}).get("display_name"),
                "score": float(t.get("score") or 0.0),
            }
        )
    cache[key] = got
    time.sleep(0.12)
    return got


def modal_community(case: str, bands, cache: dict[str, Any]) -> dict[str, float]:
    """The case's community from its OWN papers, excluding the band being scored.

    The second arm, and the one that does not depend on an academic classifier reading
    software prose. Every labelled paper in the case that is NOT in the score-2 band votes for
    its subfield and field; the band is then scored against that distribution. Nothing here
    touches a judge label, and excluding the band keeps the community independent of the thing
    being ordered.
    """
    votes: dict[str, float] = {}
    others = [p for p in bands[case].papers if p.gate != 2]
    for p in others:
        t = topic_of(p.id, cache)
        if not t:
            continue
        for level, weight in (("subfield", 1.0), ("field", 0.4)):
            if t[level]:
                key = f"{level}:{t[level]}"
                votes[key] = votes.get(key, 0.0) + weight
    total = sum(votes.values()) or 1.0
    return {k: v / total for k, v in votes.items()}


def modal_score(paper: dict[str, Any] | None, community: dict[str, float]) -> float | None:
    """A paper's share of the case's own community distribution. None when unknown."""
    if paper is None or not community:
        return None
    best = 0.0
    for level, weight in (("subfield", 1.0), ("field", 0.4)):
        if paper[level]:
            best = max(best, weight * community.get(f"{level}:{paper[level]}", 0.0))
    return best


def match_score(paper: dict[str, Any] | None, repo: list[dict[str, Any]]) -> float | None:
    """How well a paper's community matches the repository's. None when unknown.

    Graded rather than binary: an exact topic match is worth the repo's own confidence in that
    topic, a subfield match half of it, a field match a fifth. A flat yes/no would throw away
    the hierarchy, which is the only structure Topics actually offer.
    """
    if paper is None or not repo:
        return None
    best = 0.0
    for t in repo:
        if paper["id"] and paper["id"] == t["id"]:
            best = max(best, t["score"])
        elif paper["subfield"] and paper["subfield"] == t["subfield"]:
            best = max(best, 0.5 * t["score"])
        elif paper["field"] and paper["field"] == t["field"]:
            best = max(best, 0.2 * t["score"])
    return best


def auc_ci(scores: list[float], labels: list[bool]) -> tuple[float, list[float]]:
    """Point AUC and a seeded bootstrap interval, because at n=108 the point is not enough."""
    point = tb.auc(scores, labels)
    rnd = random.Random(SEED)
    idx = range(len(scores))
    draws = []
    for _ in range(BOOTSTRAP):
        pick = [rnd.choice(idx) for _ in idx]
        s = [scores[i] for i in pick]
        lab = [labels[i] for i in pick]
        if len(set(lab)) == 2:
            draws.append(tb.auc(s, lab))
    draws.sort()
    lo = draws[int(0.025 * len(draws))]
    hi = draws[int(0.975 * len(draws)) - 1]
    return round(point, 4), [round(lo, 4), round(hi, 4)]


def main() -> int:
    from dotenv import load_dotenv

    load_dotenv(EVALS / ".env")
    cache: dict[str, Any] = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.is_file() else {}
    bands = tb.load_testbed_b()

    # Two arms. `repo_text` asks OpenAlex to classify the repository's own description;
    # `own_papers` derives the community from the case's non-band papers and needs no
    # classifier at all. The second exists because the first misreads software prose --
    # `diffusion` classifies as NMR spectroscopy, `cv` as brain-tumour detection -- which is
    # this project's register mismatch reaching a new instrument.
    arms: dict[str, dict[str, dict[str, float]]] = {"repo_text": {}, "own_papers": {}}
    coverage = {"resolved": 0, "unresolved": 0, "cases_without_repo_topic": []}
    per_case: dict[str, Any] = {}
    try:
        for case in sorted(bands):
            band = bands[case].band2
            if not band:
                per_case[case] = {"band": 0, "note": "no score-2 band in this case"}
                continue
            rt = repo_topics(case, cache)
            if not rt:
                coverage["cases_without_repo_topic"].append(case)
            community = modal_community(case, bands, cache)
            arms["repo_text"][case] = {}
            arms["own_papers"][case] = {}
            for p in band:
                t = topic_of(p.id, cache)
                if t is None:
                    coverage["unresolved"] += 1
                    continue
                coverage["resolved"] += 1
                m = match_score(t, rt)
                if m is not None:
                    arms["repo_text"][case][p.id] = m
                o = modal_score(t, community)
                if o is not None:
                    arms["own_papers"][case][p.id] = o
            top = max(community, key=lambda k: community[k]) if community else None
            per_case[case] = {
                "band": len(band),
                "scored_repo_text": len(arms["repo_text"][case]),
                "scored_own_papers": len(arms["own_papers"][case]),
                "judge3": sum(1 for p in band if p.judge >= 3),
                "repo_text_topic": (rt[0]["display_name"] if rt else None),
                "repo_text_field": (rt[0]["field"] if rt else None),
                "own_papers_modal": top,
            }
    finally:
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        CACHE.write_text(json.dumps(cache, indent=0), encoding="utf-8")

    def pooled(scores, target: int) -> tuple[float, list[float], int, int]:
        xs: list[float] = []
        labels: list[bool] = []
        for case, band in bands.items():
            for p in band.band2:
                if p.id in scores.get(case, {}):
                    xs.append(scores[case][p.id])
                    labels.append(p.judge >= target)
        point, ci = auc_ci(xs, labels)
        return point, ci, sum(labels), len(labels)

    results = {}
    for name, sc in arms.items():
        a3, ci3, pos3, n3 = pooled(sc, 3)
        a2, ci2, _pos2, _n2 = pooled(sc, 2)
        results[name] = {
            "auc_judge3": a3,
            "auc_judge3_ci95": ci3,
            "n_scored": n3,
            "n_positive": pos3,
            "auc_judge2": a2,
            "auc_judge2_ci95": ci2,
            "passes_bar": bool(a3 >= BAR),
        }
    best = max(results, key=lambda k: results[k]["auc_judge3"])
    a3 = results[best]["auc_judge3"]
    ci3 = results[best]["auc_judge3_ci95"]

    out = {
        "_comment": (
            "Item 6 / NR-44: OpenAlex Topics as a free community signal for ordering the "
            "gate-admitted band. $0 -- no LLM, no judge. Derived by "
            "evals/topic_community_probe.py; pinned by tests/test_topic_community_probe.py. "
            "Pre-registered bar AUC >= 0.65 at the judge==3 target on testbed B, where the "
            "fine-scale incumbent scores 0.760 and the NR-21 metadata family 0.585."
        ),
        "bar": BAR,
        "incumbent_finescale_testbed_b_judge3": 0.760,
        "nr21_metadata_family": 0.585,
        "testbed": {
            "cases": len(bands),
            "labelled_papers": sum(len(b.papers) for b in bands.values()),
            "band2_papers": sum(len(b.band2) for b in bands.values()),
            "cases_with_no_band": sorted(c for c, b in bands.items() if not b.band2),
        },
        "coverage": coverage,
        "per_case": per_case,
        "arms": results,
        "best_arm": best,
    }
    out["verdict"] = {
        "passes_bar": bool(a3 >= BAR),
        "beats_metadata_family": bool(a3 > 0.585),
        "ci_contains_bar": bool(ci3[0] <= BAR <= ci3[1]),
        "ci_contains_chance": bool(ci3[0] <= 0.5 <= ci3[1]),
        "testbed_could_resolve_the_bar": bool(ci3[1] - ci3[0] < 2 * (BAR - 0.5)),
    }

    FROZEN.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    keys = ("testbed", "coverage", "arms", "best_arm", "verdict")
    print(json.dumps({k: out[k] for k in keys}, indent=1))
    print("per case:")
    for c, v in sorted(per_case.items()):
        print(f"  {c:<12} {json.dumps(v)}")
    print(f"\nwrote {FROZEN.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
