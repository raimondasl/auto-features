"""P6: ground truth that no model produced — papers a repo verifiably adopted after T0.

Every number in this project is agreement with GPT-5.5, and the 48 recall targets are
themselves Opus picks the judge then scored. That is circular twice over, and P5 made the
circularity heavier rather than lighter: "58% of the top band is actionable" is 58% *by the
judge's bar*.

§3.1 supplies the way out. A bibliography is a well-targeted index of what a repo already
does — so an arXiv id that is **in the docs at HEAD and absent 24 months earlier** is a
technique the project demonstrably took up in between. No model is involved in that label.

    ids(HEAD) - ids(T0)   =   what this repo adopted, as judged by the repo

Three things fall out of it, in increasing order of what they are worth:

  1. a count — is there enough signal here to build a retro-benchmark at all
  2. **judge validity** — show GPT-5.5 the repo *as it was at T0* and the papers it went on
     to adopt. If the judge does not call them actionable, then every measurement downstream
     of the judge is measuring something other than the product's goal. This is the single
     highest-value test in the plan.
  3. retro-recall — seed the citation hop from the T0 bibliography and see whether the
     adopted papers were reachable *before* they were adopted

**Pre-registered, restated for the 22-case benchmark before running (2026-08-06).** P6 was
written for 12 cases and predicted ">=30 usable adoptions across >=6 of 7 arXiv-rich repos".
The benchmark now has 22 cases; the arXiv-rich set is whichever repos have ids in their docs
at HEAD, which is measured here rather than assumed. Restated:

  * PREDICTION: >=30 usable adoptions across >=6 repos; the T0 hop reaches >=60% of them;
    the judge scores >=70% of them actionable against the T0 repo.
  * **If judge-actionable <40%, the judge is not measuring the product's goal** — that is
    the outcome that matters most, and it is pre-registered as a threshold, not a vibe.
  * KILL: <10 usable adoptions, or >80% self-citations — ground truth would have to come
    from CHANGELOG/PR mining or from new citation-rich cases instead.

**Clones are blobless and separate.** `evals/.work/<case>` clones are depth-1 and their
working-tree state gates the judge verdict cache; checking out an old commit in one would
silently re-key every cached verdict for that repo. This script never touches them. It
clones into `evals/.work/fullclone/<case>` with `--filter=blob:none --no-checkout` and reads
everything through `git show` / `git grep`, so nothing is ever checked out at all and a
2 GB repo costs a few MB of doc blobs.

    uv run python evals/mine_adoptions.py --mine          # $0, the adoption set
    uv run python evals/mine_adoptions.py --judge         # ~$3, judge validity at T0
    uv run python evals/mine_adoptions.py --report        # re-derive, $0
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import judge as judge_mod  # noqa: E402
import yaml  # noqa: E402
from diagnose_triage import _load_env, fetch_papers  # noqa: E402

EVALS = Path(__file__).resolve().parent
WORK = EVALS / ".work"
CLONES = WORK / "fullclone"
OUT = WORK / "adoptions.json"
SEEDS = WORK / "adoption_seeds.json"
BENCH = EVALS / "benchmark.yaml"

ID = re.compile(r"(?:arxiv\.org/abs/|arXiv[:/])(\d{4}\.\d{4,5})", re.I)
# git grep -E speaks POSIX ERE: no `(?:` and no `\d`. Deriving this from ID.pattern by string
# surgery produced `(?:?:` and silently matched nothing, which reads exactly like a repo with
# no adoptions — so the two patterns are written out separately and pinned to each other by
# a test rather than shared by a clever substitution.
GREP_PATTERN = r"(arxiv\.org/abs/|arXiv[:/])[0-9]{4}\.[0-9]{4,5}"
DOC_GLOBS = ("*.md", "*.rst", "*.cff", "*.bib", "*.txt")
WINDOW_MONTHS = 24
# The two S2 endpoints `hop` dispatches on, spelled the way it expects them.
HOP_DIRECTIONS = ("references", "citations")
MIN_PAPER_AGE_DAYS = 182  # "not younger than 6 months at citing time"
ACTIONABLE = 2
# Pre-registered, see module docstring.
PREDICT_ADOPTIONS = 30
PREDICT_REPOS = 6
PREDICT_JUDGE_RATE = 0.70
JUDGE_INVALIDATES_BELOW = 0.40
KILL_ADOPTIONS = 10
KILL_SELF_CITE_FRACTION = 0.80
# A heading that introduces the project's OWN paper. Ids under it are self-citations.
CITE_HEADING = re.compile(
    r"^#{1,4}\s*.{0,40}\b(citation|cite|bibtex|reference this)\b", re.I | re.M
)


def git(repo: Path, *args: str, check: bool = True) -> str:
    out = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if check and out.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {out.stderr.strip()[:200]}")
    return out.stdout


def clone(case: str, url: str) -> Path | None:
    """Blobless, never checked out. See the module docstring on why this is not `.work/<case>`."""
    path = CLONES / case
    if (path / "HEAD").is_file() or (path / ".git").exists():
        return path
    CLONES.mkdir(parents=True, exist_ok=True)
    print(f"  cloning {url} (blobless)...", flush=True)
    res = subprocess.run(
        ["git", "clone", "--filter=blob:none", "--no-checkout", "--quiet", url, str(path)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if res.returncode != 0:
        print(f"    ! clone failed: {res.stderr.strip()[:160]}")
        return None
    return path


def ids_at(repo: Path, rev: str) -> set[str]:
    """Every arXiv id in the docs at *rev*, without checking anything out."""
    args = ["grep", "-h", "-I", "-i", "-o", "-E", GREP_PATTERN, rev, "--", *DOC_GLOBS]
    out = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    # git grep exits 1 when nothing matches, which is not an error here.
    return {m.group(1) for m in ID.finditer(out.stdout)}


def self_cited(repo: Path, rev: str) -> set[str]:
    """Ids the project cites as ITS OWN work — a CITATION file, or a "Citation" heading.

    A repo that is the reference implementation of a paper always cites that paper, and it
    did not *adopt* it. Without this filter the strongest-looking adoptions would be the ones
    that are definitionally not adoptions.

    Conservative: it catches the conventional places and will miss an unconventional one, so
    the reported self-citation fraction is a lower bound.
    """
    found: set[str] = set()
    listing = subprocess.run(
        ["git", "-C", str(repo), "ls-tree", "-r", "--name-only", rev],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    ).stdout.splitlines()
    for path in listing:
        name = path.rsplit("/", 1)[-1].lower()
        is_citation_file = name.startswith("citation")
        if not (is_citation_file or name.endswith((".md", ".rst"))):
            continue
        blob = subprocess.run(
            ["git", "-C", str(repo), "show", f"{rev}:{path}"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        ).stdout
        if not blob:
            continue
        if is_citation_file:
            found |= {m.group(1) for m in ID.finditer(blob)}
            continue
        for heading in CITE_HEADING.finditer(blob):
            found |= {m.group(1) for m in ID.finditer(blob[heading.end() : heading.end() + 800])}
    return found


def t0_context(repo: Path, case: str, rev: str, max_readme: int = 3500) -> str:
    """`assemble_repo_context` rebuilt from git objects, so nothing is checked out.

    Same three parts as the shipped helper — README excerpt, manifests, a shallow file
    listing — read at *rev* instead of from a working tree. The point is that the judge sees
    the repository as it was BEFORE the adoption, which is the only state at which "would
    this paper improve it" is a real question.
    """
    listing = subprocess.run(
        ["git", "-C", str(repo), "ls-tree", "-r", "--name-only", rev],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    ).stdout.splitlines()

    def read(path: str, limit: int) -> str:
        return subprocess.run(
            ["git", "-C", str(repo), "show", f"{rev}:{path}"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        ).stdout[:limit]

    parts = [f"Repository: {case}", ""]
    for readme in ("README.md", "README.rst", "README.txt", "readme.md"):
        if readme in listing:
            parts += ["## README (excerpt)", read(readme, max_readme), ""]
            break
    for manifest in ("requirements.txt", "pyproject.toml", "package.json", "setup.py"):
        if manifest in listing:
            parts += [f"## {manifest}", read(manifest, 1200), ""]
    exts = {".py", ".js", ".ts", ".cpp", ".c", ".go", ".rs", ".java"}
    files = [f for f in sorted(listing) if Path(f).suffix in exts]
    if files:
        parts += ["## Source files (sample)", "\n".join(files[:60]), ""]
    return "\n".join(parts)


def _posted(arxiv_id: str) -> datetime:
    head = arxiv_id.split(".")[0]
    return datetime(2000 + int(head[:2]), int(head[2:4]), 1, tzinfo=UTC)


def mine(cases: dict[str, str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    # The T0 bibliography, kept because it is the seed set for the retro-recall question:
    # was an adopted paper reachable from what the repo already cited, BEFORE it adopted it?
    seeds: dict[str, list[str]] = {}
    for case, url in sorted(cases.items()):
        repo = clone(case, url)
        if repo is None:
            continue
        head_ts = git(repo, "log", "-1", "--format=%ct", "HEAD").strip()
        if not head_ts:
            continue
        head_date = datetime.fromtimestamp(int(head_ts), tz=UTC)
        cutoff = head_date - timedelta(days=WINDOW_MONTHS * 30)
        t0 = git(
            repo, "rev-list", "-1", f"--before={cutoff.date().isoformat()}", "HEAD", check=False
        ).strip()
        if not t0:
            print(f"[{case:10}] no history before {cutoff.date()} — skipping")
            continue
        at_head, at_t0 = ids_at(repo, "HEAD"), ids_at(repo, t0)
        seeds[case] = sorted(at_t0)
        selfcites = self_cited(repo, "HEAD")
        adopted = sorted(at_head - at_t0)
        usable = [
            a
            for a in adopted
            if a not in selfcites and (head_date - _posted(a)).days >= MIN_PAPER_AGE_DAYS
        ]
        for a in adopted:
            rows.append(
                {
                    "case": case,
                    "id": a,
                    "t0": t0,
                    "t0_date": cutoff.date().isoformat(),
                    "head_date": head_date.date().isoformat(),
                    "self_cited": a in selfcites,
                    "too_new": (head_date - _posted(a)).days < MIN_PAPER_AGE_DAYS,
                    "usable": a in usable,
                    "seeds_at_t0": len(at_t0),
                }
            )
        print(
            f"[{case:10}] HEAD {len(at_head):3} ids, T0 {len(at_t0):3} ({cutoff.date()})  "
            f"adopted {len(adopted):3}  usable {len(usable):3}  "
            f"self-cited {len(adopted) - len([a for a in adopted if a not in selfcites]):2}",
            flush=True,
        )
    SEEDS.write_text(json.dumps(seeds, indent=2), encoding="utf-8")
    return rows


def retro_hop(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Could the citation hop have found these papers from the T0 bibliography alone?

    This is retro-recall, and it is the only recall measurement in the project whose targets
    were not chosen by a model. If the hop reaches an adoption from seeds the repo already
    had, then the channel would have surfaced that paper *before* the maintainers found it.
    """
    from diagnose_citation_hop import hop

    seeds = json.loads(SEEDS.read_text(encoding="utf-8")) if SEEDS.is_file() else {}
    by_case: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if row["usable"]:
            by_case.setdefault(row["case"], []).append(row)
    for case, adopted in sorted(by_case.items()):
        case_seeds = seeds.get(case) or []
        if not case_seeds:
            print(f"[{case:10}] no T0 seeds — cannot hop")
            continue
        # Both directions, as the shipped pool builder does. Forward alone would understate
        # reach — §6 measured that the backward set contributes uniquely — and this is a
        # recall question, so the union is the honest denominator.
        reached: set[str] = set()
        failed = 0
        # "references" is the backward set (what the seeds cite), "citations" the forward
        # set (what cites them) — these are S2 endpoint names, not English directions, and
        # `hop` takes them positionally with no validation. Passing "forward"/"backward"
        # raised nothing useful; the guard for that is in tests/test_eval_adoptions.py.
        for direction in HOP_DIRECTIONS:
            result = hop(case_seeds, direction)
            failed += result.failed_chunks
            reached |= set(result.reached)
        if failed:
            # The same guard as build_hop_pool: a throttled chunk is DATA LOSS, and a
            # silently smaller pool reads as a worse channel rather than a broken run.
            print(f"[{case:10}] {failed} failed chunks — REFUSING to score")
            for row in adopted:
                row["hop_reached"] = None
            continue
        for row in adopted:
            row["hop_reached"] = row["id"] in reached
        hits = sum(1 for row in adopted if row["hop_reached"])
        print(
            f"[{case:10}] {len(case_seeds):3} T0 seeds -> {len(reached):7,} candidates, "
            f"reached {hits}/{len(adopted)} adoptions",
            flush=True,
        )
    return rows


def judge_at_t0(rows: list[dict[str, Any]], model: str) -> list[dict[str, Any]]:
    """Score each adoption against the repo AS IT WAS at T0.

    `use_cache=False` is mandatory and not an optimisation. `judge_paper` keys its cache on
    (model, repo, paper_id) and NOT on the repo context, so a T0 verdict would overwrite the
    HEAD verdict for the same paper in the shared gold cache. That exact write took `rag`
    from 5 targets to 0 once already.
    """
    usable = [r for r in rows if r["usable"]]
    papers = fetch_papers(sorted({r["id"] for r in usable}))
    contexts: dict[str, str] = {}
    for row in usable:
        case = row["case"]
        if case not in contexts:
            contexts[case] = t0_context(CLONES / case, case, row["t0"])
        if row["id"] not in papers:
            row["judge"] = None
            continue
        paper = {"arxiv_id": row["id"], **papers[row["id"]]}
        try:
            verdict = judge_mod.judge_paper(
                case, contexts[case], paper, model=model, use_cache=False
            )
            row["judge"] = int(verdict["score"])
            row["judge_why"] = verdict.get("justification", "")
        except Exception as exc:  # noqa: BLE001
            print(f"    ! judge {row['id']} failed: {exc}")
            row["judge"] = None
        print(
            f"  [{case:10}] {row['id']:12} -> {row['judge']}",
            flush=True,
        )
    return rows


def report(rows: list[dict[str, Any]]) -> None:
    usable = [r for r in rows if r["usable"]]
    repos = {r["case"] for r in usable}
    selfcited = [r for r in rows if r["self_cited"]]
    print(f"\n=== P6 — adoptions mined from git history ({len(rows)} raw) ===")
    print(f"{'case':12} {'adopted':>8} {'self-cite':>10} {'too new':>8} {'usable':>7}")
    for case in sorted({r["case"] for r in rows}):
        sel = [r for r in rows if r["case"] == case]
        print(
            f"{case:12} {len(sel):>8} {sum(1 for r in sel if r['self_cited']):>10} "
            f"{sum(1 for r in sel if r['too_new']):>8} {sum(1 for r in sel if r['usable']):>7}"
        )
    frac = len(selfcited) / max(len(rows), 1)
    print(f"\nusable adoptions: {len(usable)} across {len(repos)} repos")
    print(f"self-citation fraction: {frac:.0%} ({len(selfcited)}/{len(rows)}) — a lower bound")

    hopped = [r for r in usable if r.get("hop_reached") is not None]
    if hopped:
        hit = sum(1 for r in hopped if r["hop_reached"])
        print(
            f"\nretro-recall — T0 bibliography -> one hop: {hit}/{len(hopped)} = "
            f"{hit / len(hopped):.0%} of adoptions were reachable BEFORE they were adopted"
        )
        for case in sorted({r["case"] for r in hopped}):
            sel = [r for r in hopped if r["case"] == case]
            print(f"    {case:12} {sum(1 for r in sel if r['hop_reached'])}/{len(sel)}")

    judged = [r for r in usable if r.get("judge") is not None]
    if judged:
        act = sum(1 for r in judged if r["judge"] >= ACTIONABLE)
        hist = {s: sum(1 for r in judged if r["judge"] == s) for s in (0, 1, 2, 3)}
        print(
            f"\njudge vs T0 repo (n={judged and len(judged)}): "
            f"{act}/{len(judged)} = {act / len(judged):.0%} actionable   "
            f"scores 0:{hist[0]} 1:{hist[1]} 2:{hist[2]} 3:{hist[3]}"
        )

    print(
        f"\nPRE-REGISTERED: >={PREDICT_ADOPTIONS} usable across >={PREDICT_REPOS} repos; "
        f"judge >={PREDICT_JUDGE_RATE:.0%} actionable; KILL <{KILL_ADOPTIONS} usable or "
        f">{KILL_SELF_CITE_FRACTION:.0%} self-cited"
    )
    if len(usable) < KILL_ADOPTIONS or frac > KILL_SELF_CITE_FRACTION:
        print("verdict: KILL — ground truth must come from CHANGELOG/PR mining instead")
    elif len(usable) >= PREDICT_ADOPTIONS and len(repos) >= PREDICT_REPOS:
        print("verdict: yield MET")
    else:
        print(f"verdict: yield BELOW PREDICTION ({len(usable)} across {len(repos)})")
    if judged:
        rate = sum(1 for r in judged if r["judge"] >= ACTIONABLE) / len(judged)
        if rate >= PREDICT_JUDGE_RATE:
            print(f"judge validity: MET at {rate:.0%} — the judge rewards what repos adopt")
        elif rate < JUDGE_INVALIDATES_BELOW:
            print(
                f"judge validity: FAILED at {rate:.0%} (<{JUDGE_INVALIDATES_BELOW:.0%}). "
                "Every number downstream of the judge inherits this."
            )
        else:
            print(f"judge validity: BELOW PREDICTION at {rate:.0%}, above the invalidation bar")
    print(f"\nwritten to {OUT}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mine", action="store_true", help="clone + diff the bibliographies ($0)")
    ap.add_argument("--judge", action="store_true", help="score adoptions against the T0 repo")
    ap.add_argument("--hop", action="store_true", help="retro-recall from the T0 bibliography")
    ap.add_argument("--report", action="store_true", help="re-derive from saved rows ($0)")
    ap.add_argument("--model", default=judge_mod.DEFAULT_JUDGE_MODEL)
    args = ap.parse_args()
    _load_env()

    bench = yaml.safe_load(BENCH.read_text(encoding="utf-8"))
    entries = bench["cases"] if isinstance(bench, dict) else bench
    cases = {
        c["name"]: c["live_repo"]
        for c in entries
        if isinstance(c, dict) and c.get("live_repo") and c.get("name")
    }

    if args.mine:
        rows = mine(cases)
        OUT.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        report(rows)
        return 0
    rows = json.loads(OUT.read_text(encoding="utf-8")) if OUT.is_file() else []
    if not rows:
        print("no mined rows — run with --mine first")
        return 1
    if args.judge:
        # The gold set must not move. `use_cache=False` is what prevents T0 verdicts from
        # overwriting HEAD verdicts in the shared cache; this checks that it worked rather
        # than trusting it, because the same class of write already cost `rag` all 5 of its
        # targets once.
        from build_hop_pool import resolve_targets

        before = resolve_targets()
        rows = judge_at_t0(rows, args.model)
        if resolve_targets() != before:
            print("\n!! THE GOLD SET MOVED — T0 verdicts leaked into the shared cache.")
            return 1
        OUT.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    if args.hop:
        rows = retro_hop(rows)
        OUT.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    report(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
