"""How well does the shipped triage gate agree with the judge, on ALL the labelled data?

Triage has only ever been measured incidentally — inside a Tier B run, on the ~10 papers one
case happened to surface. On `speech` that was 50% precision against a 50% base rate, and on
`rag` it found 0 of 3 actionable papers. Those are n=10 samples of a component that decides
what every digest shows.

**428 judged papers are already sitting in `cache/judge/`**, across 12 repos, at a 32%
actionable base rate. That is a real labelled set, and it makes triage falsifiable offline:
score every paper with the shipping prompt, compare to the judge's 0-3, and report precision,
recall and how far the result sits from chance.

    uv run python evals/diagnose_triage.py                  # all cases
    uv run python evals/diagnose_triage.py --case speech    # one
    uv run python evals/diagnose_triage.py --limit 40       # cheap smoke run

Cost: one Haiku call per paper (~$0.10 for all 428). Abstracts are fetched once from arXiv in
`id_list` batches and cached under `.work/triage_papers.json`, so re-runs are LLM-only.

**What this measures, precisely.** Agreement with the GPT-5.5 judge, not ground truth. That
is the same standard the whole Tier B benchmark uses — the judge defines "actionable" there —
so a triage variant that agrees more will score better on Tier B. It is not evidence that the
judge is right.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from reporadar.config import ProfilerConfig, SuggestionsConfig  # noqa: E402
from reporadar.llm_client import complete  # noqa: E402
from reporadar.profiler import (
    _collect_text_corpus,  # noqa: E402
    profile_repo,  # noqa: E402
)
from reporadar.triage import _RUBRIC, _parse_verdict, score_actionability  # noqa: E402

EVALS = Path(__file__).resolve().parent
WORK = EVALS / ".work"
JUDGE = EVALS / "cache" / "judge" / "v1" / "gpt-5.5"
PAPERS_CACHE = WORK / "triage_papers.json"
ACTIONABLE = 2
DEFAULT_MODEL = "claude-haiku-4-5"

_ENTRY = re.compile(r"<entry>(.*?)</entry>", re.DOTALL)
_ID = re.compile(r"<id>https?://arxiv\.org/abs/([^<]+)</id>")
_TITLE = re.compile(r"<title>(.*?)</title>", re.DOTALL)
_SUMMARY = re.compile(r"<summary>(.*?)</summary>", re.DOTALL)


def _load_env() -> None:
    env = EVALS / ".env"
    if not env.is_file():
        return
    for line in env.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


def labels() -> dict[str, dict[str, int]]:
    """{case: {arxiv_id: judge_score}} from the cached verdicts."""
    out: dict[str, dict[str, int]] = {}
    for case in sorted(JUDGE.iterdir()):
        if not case.is_dir():
            continue
        out[case.name] = {
            f.stem.split("v")[0]: json.loads(f.read_text(encoding="utf-8"))["score"]
            for f in case.glob("*.json")
        }
    return out


def fetch_papers(ids: list[str]) -> dict[str, dict[str, str]]:
    """Title + abstract for each id, cached on disk. Batched via arXiv `id_list`."""
    cache: dict[str, dict[str, str]] = {}
    if PAPERS_CACHE.is_file():
        cache = json.loads(PAPERS_CACHE.read_text(encoding="utf-8"))
    todo = [i for i in ids if i not in cache]
    for start in range(0, len(todo), 100):
        chunk = todo[start : start + 100]
        url = "https://export.arxiv.org/api/query?" + urllib.parse.urlencode(
            {"id_list": ",".join(chunk), "max_results": len(chunk)}
        )
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "reporadar-triage-eval/1.0"})
            with urllib.request.urlopen(req, timeout=120) as resp:
                body = resp.read().decode("utf-8", "replace")
        except Exception as exc:  # noqa: BLE001
            print(f"  ! arXiv fetch failed for {len(chunk)} ids: {exc}")
            time.sleep(20)
            continue
        for entry in _ENTRY.findall(body):
            m_id, m_t, m_s = _ID.search(entry), _TITLE.search(entry), _SUMMARY.search(entry)
            if not (m_id and m_t and m_s):
                continue
            cache[m_id.group(1).split("v")[0]] = {
                "title": " ".join(m_t.group(1).split()),
                "abstract": " ".join(m_s.group(1).split()),
            }
        print(f"  fetched {min(start + 100, len(todo))}/{len(todo)} abstracts", flush=True)
        time.sleep(4)
    PAPERS_CACHE.parent.mkdir(parents=True, exist_ok=True)
    PAPERS_CACHE.write_text(json.dumps(cache, indent=2), encoding="utf-8")
    return cache


def repo_section(profile: object, repo: Path, mode: str) -> str:
    """The repo half of the triage prompt, in one of four representations.

    `keywords` is the shipped prompt with `profiler.prose_chars: 0`. For ColBERT it yields
    "functions, indexer, searcher functions, trainer functions, trainer, searcher, colbert,
    file, torch, transformers, master" and "Domains: NLP, deep learning, scientific
    computing, web APIs" — 354 characters that never say retrieval, and say "web APIs"
    because the project depends on flask. The README's own first line is "Efficient and
    Effective Passage Search via Contextualized Late Interaction over BERT".

    `prose` is the shipped prompt with prose on: keywords AND the real README.

    `tagline` is the historical variant that was reported as "+16 net@2 for README
    context". It is kept, and named honestly, because it was NOT the README: it read
    `_collect_text_corpus(repo)[0]`, which is the packaging description on 11 of the 12
    benchmark repos — 23 to 230 characters. It also DROPS the domains and key-topics
    block, so the prompt it produced was *smaller* than `keywords` on 9 of 12 cases.
    Whether its gain came from adding the tagline or from deleting the misleading keyword
    block is exactly what these four modes separate.
    """
    anchors = ", ".join(profile.anchors[:12]) or "none"
    domains = ", ".join(profile.domains[:5]) or "general"
    keywords = ", ".join(t for t, _ in profile.keywords[:12]) or "n/a"
    kw = f"Dependencies/libraries: {anchors}\nDomains: {domains}\nKey topics: {keywords}\n"
    header = "What this project is, in its own words:"
    if mode == "keywords":
        return kw
    if mode == "prose":
        return f"{kw}\n{header}\n{getattr(profile, 'prose', '')}\n"
    # tagline / both: the historical docs[0] read, reproduced exactly.
    docs = _collect_text_corpus(repo)
    tagline = (docs[0] if docs else "")[:1800]
    if mode == "tagline":
        return f"Dependencies/libraries: {anchors}\n\n{header}\n{tagline}\n"
    return f"{kw}\n{header}\n{tagline}\n"


def merge_rows(out: Path, rows: list[dict]) -> list[dict]:
    """Write *rows* into *out*, MERGING by (case, id) rather than clobbering.

    `--case` and `--limit` exist for cheap smoke runs, and until this existed a 3-paper
    smoke run overwrote the 602-paper file for that arm — the same failure that cost
    whole-set results twice in the hop-pool scripts, arriving here by a different door.
    Merging is safe because the arm is already encoded in the filename, so every row in one
    file answers the same question.
    """
    merged: dict[tuple[str, str], dict] = {}
    if out.is_file():
        merged = {(r["case"], r["id"]): r for r in json.loads(out.read_text(encoding="utf-8"))}
    before = len(merged)
    merged.update({(r["case"], r["id"]): r for r in rows})
    if before > len(rows):
        print(f"merged {len(rows)} new rows into {before} existing -> {len(merged)}")
    ordered = [merged[k] for k in sorted(merged)]
    out.write_text(json.dumps(ordered, indent=2), encoding="utf-8")
    return ordered


WANTS_CACHE = WORK / "repo_wants.json"


def _wants_for(case: str) -> list[str]:
    """Verbatim open-issue titles for one repo, from the cache `fetch_wants.py` writes.

    Read from disk rather than fetched here: a tracker moves, and an arm whose prompt
    changes between runs is not a paired comparison.
    """
    if not WANTS_CACHE.is_file():
        raise SystemExit("no .work/repo_wants.json — run `uv run python evals/fetch_wants.py`")
    return list(json.loads(WANTS_CACHE.read_text(encoding="utf-8")).get(case) or [])


def _wants_block(titles: list[str]) -> str:
    from fetch_wants import as_block

    return as_block(titles)


SUMMARY_CACHE = WORK / "repo_summaries.json"


def _summary_for(repo: Path, cfg: object):  # type: ignore[no-untyped-def]
    """One summariser call per repo, cached on disk and keyed by doc content.

    Without the cache a 602-paper arm would re-summarise the same 12 repos 602 times,
    which is both 50x the cost and a different (noisier) experiment than the shipped
    path, where the summary is computed once per run.
    """
    from reporadar.repo_summary import RepoSummary, summarize_repo

    disk = {}
    if SUMMARY_CACHE.is_file():
        disk = json.loads(SUMMARY_CACHE.read_text(encoding="utf-8"))
    entry = disk.get(repo.name, {})
    summary = summarize_repo(repo, cfg, cache=entry)
    disk[repo.name] = entry
    SUMMARY_CACHE.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_CACHE.write_text(json.dumps(disk, indent=2), encoding="utf-8")
    assert isinstance(summary, RepoSummary)
    return summary


def score_with(paper: dict, profile: object, repo: Path, mode: str, cfg: object) -> int:
    """Score one paper. `keywords` and `prose` go through the SHIPPED prompt builder.

    Routing those two through `score_actionability` rather than a local copy is the point:
    a diagnostic that reimplements the prompt measures the diagnostic. Prose is controlled
    by the profile the caller passes (`ProfilerConfig.prose_chars`), not by a flag here.
    """
    if mode in ("keywords", "prose"):
        return score_actionability(paper, profile, cfg)[0]
    if mode.startswith("summary"):
        # A thin shim so the three summary arms differ ONLY in the gap section, and
        # go through the shipped scorer rather than a reimplemented prompt.
        full = _summary_for(repo, cfg)
        if mode == "summary_nogaps":
            full = replace(full, improvement_areas=[])
        return score_actionability(paper, profile, cfg, full)[0]
    if mode == "wants":
        # P8: prose-300 PLUS the repo's own open-issue titles, verbatim. Routed through the
        # shipped scorer via the same shim `extractive` uses, so the only difference from
        # the `prose` arm is one appended block — not a reimplemented prompt.
        titles = _wants_for(repo.name)
        shim = SimpleNamespace(as_prompt_block=lambda **_: _wants_block(titles))
        return score_actionability(paper, profile, cfg, shim)[0]
    if mode == "extractive":
        # Verbatim sentences only: the repo's own words, semantically SELECTED. The
        # direct control for prose300, which is the same words positionally selected.
        sel = _summary_for(repo, cfg)
        shim = SimpleNamespace(as_prompt_block=lambda **_: sel.as_excerpt_block())
        return score_actionability(paper, profile, cfg, shim)[0]
    prompt = (
        f"{_RUBRIC}\n\n# Repository\n{repo_section(profile, repo, mode)}\n"
        f"# Candidate paper\nTitle: {paper.get('title', 'Unknown')}\n"
        f"Abstract: {paper.get('abstract', '')[:1500]}\n\n"
        f"Score this paper for the repository above."
    )
    return _parse_verdict(complete(prompt, cfg, max_tokens=200))[0]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--repo-context",
        choices=(
            "keywords",
            "prose",
            "tagline",
            "both",
            "summary",
            "summary_nogaps",
            "summary_hedged",
            "extractive",
            "wants",
        ),
        default="prose",
        help="how the repo is described to the gate; `prose` is what ships",
    )
    ap.add_argument("--case")
    ap.add_argument("--limit", type=int, help="max papers per case (for a cheap smoke run)")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--prose-chars", type=int, default=300, help="README budget for `prose` mode")
    args = ap.parse_args()

    _load_env()
    by_case = labels()
    if args.case:
        by_case = {args.case: by_case.get(args.case, {})}

    wanted = sorted({i for m in by_case.values() for i in m})
    print(f"{len(wanted)} labelled papers; fetching any missing abstracts...")
    papers = fetch_papers(wanted)
    print(f"have text for {sum(1 for i in wanted if i in papers)}/{len(wanted)}\n")

    cfg = SuggestionsConfig(provider="claude", claude_model=args.model, timeout=60)
    rows: list[dict] = []
    for case, lab in by_case.items():
        repo = WORK / case
        if not repo.is_dir():
            print(f"[{case}] no clone — skipping")
            continue
        # The shipped prompt reads prose off the profile, so the variant is set HERE.
        # Building every mode from one profile would silently put prose in `keywords`.
        budget = 0 if args.repo_context == "keywords" else args.prose_chars
        profile = profile_repo(repo, profiler_cfg=ProfilerConfig(prose_chars=budget))
        ids = [i for i in sorted(lab) if i in papers][: args.limit]
        tp = fp = fn = tn = 0
        failures = 0
        for i in ids:
            paper = {"title": papers[i]["title"], "abstract": papers[i]["abstract"]}
            try:
                got = score_with(paper, profile, repo, args.repo_context, cfg)
            except Exception:  # noqa: BLE001
                failures += 1
                continue
            truth = lab[i] >= ACTIONABLE
            pred = got >= ACTIONABLE
            tp += pred and truth
            fp += pred and not truth
            fn += (not pred) and truth
            tn += (not pred) and not truth
            rows.append({"case": case, "id": i, "judge": lab[i], "triage": got})
        n = tp + fp + fn + tn
        base = (tp + fn) / n if n else 0.0
        prec = tp / (tp + fp) if (tp + fp) else float("nan")
        rec = tp / (tp + fn) if (tp + fn) else float("nan")
        print(
            f"[{case:10}] n={n:3d}  base={base:.0%}  precision={prec:.2f}  recall={rec:.2f}"
            f"  (tp={tp} fp={fp} fn={fn} tn={tn})"
            + (f"  {failures} call failures" if failures else ""),
            flush=True,
        )

    tp = sum(1 for r in rows if r["triage"] >= ACTIONABLE and r["judge"] >= ACTIONABLE)
    fp = sum(1 for r in rows if r["triage"] >= ACTIONABLE and r["judge"] < ACTIONABLE)
    fn = sum(1 for r in rows if r["triage"] < ACTIONABLE and r["judge"] >= ACTIONABLE)
    n = len(rows)
    base = (tp + fn) / n if n else 0.0
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    print("\n=== OVERALL ===")
    print(f"papers scored: {n}")
    print(f"base rate (judge says actionable): {base:.0%}")
    print(f"triage precision: {prec:.2f}   recall: {tp / (tp + fn) if (tp + fn) else 0:.2f}")
    print(f"\nA gate with no signal scores precision == base rate ({base:.0%}).")
    if prec == prec:
        lift = prec - base
        verdict = (
            "ABOVE chance" if lift > 0.05 else ("AT chance" if lift > -0.05 else "BELOW chance")
        )
        print(f"triage is {lift:+.2f} vs chance -> {verdict}")
    # Budget goes in the filename: two `prose` runs at different budgets are different
    # arms, and silently overwriting one with the other would compare a run to itself.
    # Every dimension that changes the arm must be in the filename. `--model` was NOT, and
    # it cost real data: a Sonnet run wrote to diag_triage_keywords.json and was later
    # overwritten by a Haiku run, so the per-case Sonnet numbers no longer exist anywhere
    # and only the aggregates that had already been copied into RESULTS.md survived.
    tag = args.repo_context
    if tag == "prose" and args.prose_chars != 300:
        tag = f"prose{args.prose_chars}"
    if args.model != DEFAULT_MODEL:
        tag = f"{tag}_{args.model}"
    # MERGE, never clobber. `--case` and `--limit` exist for cheap smoke runs, and until now
    # a 3-paper smoke run overwrote the 602-paper file for that arm — the same failure that
    # cost whole-set results twice in the hop-pool scripts. Merging by (case, id) is safe
    # because the arm is already in the filename, so everything here answers one question.
    merge_rows(WORK / f"diag_triage_{tag}.json", rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
