"""Walk the pool's candidates in seeded order, screening and mining in one pass.
[PREREG-judge-validity-pool §3]

**Screening is mining.** Evaluating PP2 (`ids_v2(T0) ≥ 3`) requires cloning the repository,
resolving T0 and grepping at T0 — which is most of the work of mining it. A two-stage design
would clone every candidate twice for no gain, so this does both in one pass and deletes the
clone before moving on.

**The unconditional prefix is the point of B₀.** The first `B₀ = 300` rows are walked
whatever they yield, and the qualifying rate and per-repository yield are estimated over
exactly those 300. A fixed prefix of a seeded order is a uniform random sample of the
population; a prefix whose *length* is chosen by accumulated yield is inverse sampling, and
it biases the rate upward. Only after B₀ does the walk stop early on reaching its target.

**Every row is an outcome.** A clone failure, a timeout, a repository with no history before
T0 — each is written down with a reason. NR-57's lesson is that a silently smaller population
reads as a worse channel rather than as a broken run, and two benchmark repositories once
scored a legitimate-looking zero from an empty pool.

    uv run python evals/frame/walk_pool.py --candidates evals/frame/pool/pool-universe-Dp.csv \
        --seed-file evals/frame/pool/SEED_POOL --out-dir evals/frame/pool
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

EVALS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EVALS))

import mine_adoptions as ma  # noqa: E402

POOL_DIR = EVALS / "frame" / "pool"
DEFAULT_B0 = 300  # walked unconditionally, to measure q and y without inverse sampling
DEFAULT_B = 1200
DEFAULT_TARGET = 100  # §3.3: 60 is the reporting minimum, not a stopping point
PER_REPO_CAP = 8
OWNER_CAP = 3  # PP3
MIN_IDS_T0 = 3  # PP2
MIN_HISTORY_MONTHS = 30  # PP1
ROW_TIMEOUT_S = 300

# Covariates only — they size the life-science blind spot (§6.2) and never gate anything.
# Written out separately from the arXiv patterns for the same reason `GREP_PATTERN` is:
# `git grep -E` is POSIX ERE and deriving one from the other by string surgery once produced
# a pattern that silently matched nothing.
DOI = re.compile(r"\b(10\.\d{4,9}/[-._;()/:A-Za-z0-9]+)", re.I)
DOI_GREP = r"10\.[0-9]{4,9}/[-._;()/:A-Za-z0-9]+"
PMID = re.compile(r"(?:pubmed\.ncbi\.nlm\.nih\.gov/|PMID:? ?)(\d{7,8})", re.I)
PMID_GREP = r"(pubmed\.ncbi\.nlm\.nih\.gov/|PMID:? ?)[0-9]{7,8}"

WALK_COLUMNS = (
    "rank",
    "full_name",
    "outcome",
    "created_at",
    "head",
    "head_date",
    "t0",
    "t0_commit_date",
    "window_days",
    "ids_v1_head",
    "ids_v2_head",
    "ids_v1_t0",
    "ids_v2_t0",
    "dois_head",
    "dois_t0",
    "pmids_head",
    "pmids_t0",
    "pp1_history",
    "pp2_ids_t0",
    "pp3_owner",
    "qualifies",
    "gross_adoptions",
    "usable",
    "capped",
    "seconds",
    "note",
)


def legacy_slugs(bench: Path | None = None) -> set[str]:
    """The 37 benchmark repositories, lowercased `owner/repo`. Section 2.2 excludes them.

    Nothing implemented this rule, and it is not hypothetical: **21 of the 37 are in the
    frozen candidate list**, and they carry 89 of NR-60's 94 legacy positives -- diffusers
    (46), peft (27), pytorch_geometric (13), scvi-tools (2), scanpy (1).

    Walked, `huggingface/diffusers` clones under the key `huggingface__diffusers`, so the
    `existed` guard never recognises the legacy clone sitting at `.work/fullclone/diffusion`.
    Its papers would be mined a second time as *new* pool positives, counted toward the stop
    rule, capped a second time, and section 5's legacy-versus-pool heterogeneity would compare
    the legacy cluster against itself.
    """
    import yaml

    path = bench or (EVALS / "benchmark.yaml")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    entries = data["cases"] if isinstance(data, dict) else data
    out: set[str] = set()
    for case in entries:
        if isinstance(case, dict) and case.get("live_repo"):
            out.add(case["live_repo"].rstrip("/").split("github.com/")[-1].lower())
    return out


def order_key(seed: str, full_name: str) -> str:
    return hashlib.sha256(f"{seed}{full_name}".encode()).hexdigest()


def seeded_order(candidates: list[dict[str, str]], seed: str) -> list[dict[str, str]]:
    """`sha256(SEED_POOL ‖ full_name)`. The seed is a beacon pulse fixed before any row was
    read, so the order is determined by something nobody could choose."""
    return sorted(candidates, key=lambda row: order_key(seed, row["full_name"]))


def _count(
    repo: Path, rev: str, grep: str, pattern: re.Pattern[str], timeout: float | None = None
) -> int:
    return len(ma._matches_with_paths(repo, rev, grep, pattern, timeout))


def context_hash(rubric_marker: str, context: str) -> str:
    return hashlib.sha256(f"{rubric_marker}\0{context}".encode()).hexdigest()[:12]


def _blank_row(
    rank: int, full_name: str, created_at: str, outcome: str, note: str
) -> dict[str, Any]:
    row: dict[str, Any] = dict.fromkeys(WALK_COLUMNS, "")
    row.update(
        rank=rank,
        full_name=full_name,
        created_at=created_at,
        outcome=outcome,
        note=note,
        qualifies=False,
        usable=0,
        capped=0,
        gross_adoptions=0,
    )
    return row


def github_url(full_name: str) -> str:
    return f"https://github.com/{full_name}"


def _clone(clones: Path, key: str, url: str, timeout: float) -> Path | None:
    """Blobless, never checked out, and with a timeout.

    Deliberately **not** `mine_adoptions.clone`: that resolves its destination from the
    module-level `CLONES`, and the only way to redirect it is to assign to that global. The
    walk runs four rows concurrently, so a global assignment per row is a data race — two
    threads would clone into each other's directory. It also has no timeout, and one
    pathological repository would hang an hours-long walk indefinitely.
    """
    path = clones / key
    if (path / "HEAD").is_file() or (path / ".git").exists():
        return path
    clones.mkdir(parents=True, exist_ok=True)
    try:
        res = subprocess.run(
            ["git", "clone", "--filter=blob:none", "--no-checkout", "--quiet", url, str(path)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        ma._remove_clone(path)
        return None
    return path if res.returncode == 0 else None


def walk_row(
    rank: int,
    candidate: dict[str, str],
    *,
    clones: Path,
    contexts: Path,
    head_ids: Path | None = None,
    timeout: float = ROW_TIMEOUT_S,
    url_for: Any = github_url,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """One candidate: clone, screen, mine if it qualifies, delete. Never raises.

    *head_ids* receives the repository's full identifier set at HEAD. §4 draws controls that
    are "not cited anywhere in the repository at HEAD", and the clone is gone by then — so
    the set has to be persisted here or the control rule cannot be applied at all. Counts
    alone are not enough.
    """
    started = time.monotonic()
    full = candidate["full_name"]
    created = (candidate.get("created_at") or "")[:10]
    key = full.replace("/", "__")
    existed = (clones / key).exists()
    try:
        repo = _clone(clones, key, url_for(full), timeout)
        if repo is None:
            return _blank_row(rank, full, created, "clone_failed", "clone failed or timed out"), []

        def left() -> float:
            """What remains of this row's budget. Each git call is bounded by it, so the
            row as a whole cannot exceed the timeout however many calls it makes."""
            return max(1.0, timeout - (time.monotonic() - started))

        head = ma.git(repo, "rev-parse", "HEAD", check=False, timeout=left()).strip()
        head_ts = ma.git(
            repo, "log", "-1", "--format=%ct", "HEAD", check=False, timeout=left()
        ).strip()
        if not head or not head_ts:
            return _blank_row(rank, full, created, "no_head", "cannot resolve HEAD"), []
        head_date = datetime.fromtimestamp(int(head_ts), tz=UTC)
        cutoff = head_date - timedelta(days=ma.WINDOW_MONTHS * 30)
        t0 = ma.git(
            repo,
            "rev-list",
            "-1",
            f"--before={cutoff.date().isoformat()}",
            head,
            check=False,
            timeout=left(),
        ).strip()
        if not t0:
            row = _blank_row(rank, full, created, "no_history", f"no commit before {cutoff.date()}")
            row.update(head=head, head_date=head_date.date().isoformat(), pp1_history=False)
            return row, []
        t0_ts = ma.git(repo, "log", "-1", "--format=%ct", t0, check=False, timeout=left()).strip()
        t0_date = datetime.fromtimestamp(int(t0_ts), tz=UTC) if t0_ts else cutoff

        head_paths = ma.ids_with_paths(repo, head, "v2", left())
        t0_ids = ma.ids_at(repo, t0, "v2", left())
        row = _blank_row(rank, full, created, "ok", "")
        # PP1 is checked against the CLONE, not the API: T0 is anchored to head_date, so a
        # repository quiet for a year can pass a `created_at` pre-filter and still have no
        # 30 months of history behind its own HEAD.
        history_days = None
        if created:
            born = datetime.fromisoformat(f"{created}T00:00:00+00:00")
            history_days = (head_date - born).days
        row.update(
            head=head,
            head_date=head_date.date().isoformat(),
            t0=t0,
            t0_commit_date=t0_date.date().isoformat(),
            window_days=(head_date - t0_date).days,
            ids_v1_head=_count(repo, head, ma.GREP_PATTERN, ma.ID, left()),
            ids_v2_head=len(head_paths),
            ids_v1_t0=_count(repo, t0, ma.GREP_PATTERN, ma.ID, left()),
            ids_v2_t0=len(t0_ids),
            dois_head=_count(repo, head, DOI_GREP, DOI, left()),
            dois_t0=_count(repo, t0, DOI_GREP, DOI, left()),
            pmids_head=_count(repo, head, PMID_GREP, PMID, left()),
            pmids_t0=_count(repo, t0, PMID_GREP, PMID, left()),
            pp1_history=bool(history_days is not None and history_days >= MIN_HISTORY_MONTHS * 30),
            pp2_ids_t0=len(t0_ids) >= MIN_IDS_T0,
            seconds=round(time.monotonic() - started, 1),
        )
        row["qualifies"] = bool(row["pp1_history"] and row["pp2_ids_t0"])
        if not row["qualifies"]:
            return row, []

        selfcites = ma.self_cited(
            repo, head, "v2", {p for paths in head_paths.values() for p in paths}, left()
        )
        showcase = ma.reverse_cited_only(head_paths)
        arxiv_form = set(ma._matches_with_paths(repo, head, ma.GREP_PATTERN, ma.ID, left()))
        adopted = sorted(set(head_paths) - t0_ids)
        rows: list[dict[str, Any]] = []
        for paper in adopted:
            entry = {
                "case": full,
                "id": paper,
                "extractor": "v2",
                "head": head,
                "t0": t0,
                "t0_date": cutoff.date().isoformat(),
                "t0_commit_date": t0_date.date().isoformat(),
                "head_date": head_date.date().isoformat(),
                "self_cited": paper in selfcites,
                "too_new": ma._too_new(ma._posted(paper), head_date),
                "reverse_cited": paper in showcase,
                "genesis": False,  # PP2 >= 3 subsumes the doc-genesis guard
                "via": "arxiv" if paper in arxiv_form else "hf",
                "paths": sorted(head_paths[paper])[:5],
                "seeds_at_t0": len(t0_ids),
            }
            entry["usable"] = not (
                entry["self_cited"] or entry["too_new"] or entry["reverse_cited"]
            )
            rows.append(entry)

        usable = [r for r in rows if r["usable"]]
        row.update(
            gross_adoptions=len(rows),
            usable=len(usable),
            capped=min(len(usable), PER_REPO_CAP),
        )
        if usable:
            # Persisted so judging never re-clones: the clone is deleted below, and the T0
            # context is what both judges are shown.
            context = ma.t0_context(repo, full, t0)
            contexts.mkdir(parents=True, exist_ok=True)
            digest = context_hash("t0", context)
            (contexts / f"{key}.{digest}.txt").write_text(context, encoding="utf-8")
            row["note"] = f"context {digest}"
            if head_ids is not None:
                head_ids.mkdir(parents=True, exist_ok=True)
                (head_ids / f"{key}.json").write_text(
                    json.dumps(sorted(head_paths), indent=0), encoding="utf-8"
                )
        row["seconds"] = round(time.monotonic() - started, 1)
        return row, rows
    except subprocess.TimeoutExpired:
        # Its own outcome, not lumped under `error`. Measured cold, `huggingface/diffusers`
        # takes 1,092 s -- 3.6x this bound -- and every second of it is inside `git grep`
        # lazily fetching doc blobs, which the clone-only timeout never covered.
        return _blank_row(rank, full, created, "timeout", f"exceeded {timeout:.0f}s"), []
    except Exception as exc:  # noqa: BLE001 - a bad row must not end an hours-long walk
        return _blank_row(rank, full, created, "error", str(exc)[:160]), []
    finally:
        # Delete only what this row created. A clone that was already on disk belongs to the
        # legacy 37 or to an earlier walk, and removing it would make this destructive to
        # someone else's cached work.
        made = clones / key
        if not existed and made.exists():
            ma._remove_clone(made)


def append_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(WALK_COLUMNS))
        if new:
            writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in WALK_COLUMNS})


def already_walked(path: Path) -> set[str]:
    """Resume by name. An hours-long walk that cannot resume is a walk that never finishes."""
    if not path.exists():
        return set()
    with path.open(encoding="utf-8", newline="") as fh:
        return {row["full_name"] for row in csv.DictReader(fh) if row.get("full_name")}


def merge_adoptions(path: Path, new: list[dict[str, Any]]) -> int:
    """Merge, never rewrite. `--mine` rewrites its artefact, which is the NR-57 near-miss;
    a walk that restarts must not discard what the previous attempt mined."""
    existing: list[dict[str, Any]] = []
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
    seen = {(r["case"], r["id"]) for r in existing}
    for row in new:
        if (row["case"], row["id"]) not in seen:
            existing.append(row)
            seen.add((row["case"], row["id"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    return len(existing)


def walk(
    candidates: list[dict[str, str]],
    seed: str,
    *,
    out_dir: Path,
    b0: int = DEFAULT_B0,
    budget: int = DEFAULT_B,
    target: int = DEFAULT_TARGET,
    clone_dir: Path | None = None,
    jobs: int = 4,
    curve_every: int = 50,
    url_for: Any = github_url,
    legacy: set[str] | None = None,
) -> dict[str, Any]:
    ordered = seeded_order(candidates, seed)
    walk_csv = out_dir / "validity_walk.csv"
    adoptions = out_dir / "adoptions-pool-v2.json"
    contexts = out_dir / "contexts"
    head_ids = out_dir / "head_ids"
    clones = clone_dir if clone_dir is not None else EVALS / ".work" / "fullclone"
    done = already_walked(walk_csv)
    legacy = legacy_slugs() if legacy is None else legacy

    owners: dict[str, int] = {}
    for row in _read_rows(walk_csv):
        if row.get("qualifies") == "True":
            owners[row["full_name"].split("/")[0]] = (
                owners.get(row["full_name"].split("/")[0], 0) + 1
            )
    capped_total = sum(int(r.get("capped") or 0) for r in _read_rows(walk_csv))
    walked = len(done)

    pending: list[tuple[int, dict[str, str]]] = []
    for rank, cand in enumerate(ordered):
        if cand["full_name"] in done:
            continue
        pending.append((rank, cand))

    curve: list[dict[str, Any]] = []
    index = 0
    while index < len(pending):
        if walked >= budget:
            break
        if walked >= b0 and capped_total >= target:
            break
        chunk = pending[index : index + max(1, jobs)]
        index += len(chunk)
        # PP3 is order-dependent, so it is applied here rather than inside the worker: a
        # candidate whose owner already has three qualifying repositories earlier in the
        # seeded order is skipped without being cloned.
        prepared: list[tuple[int, dict[str, str]]] = []
        skipped: list[dict[str, Any]] = []
        for rank, cand in chunk:
            owner = cand["full_name"].split("/")[0]
            if cand["full_name"].lower() in legacy:
                # Recorded, never cloned. A legacy repository walked here would enter
                # the pool a second time under a different case key (section 2.2).
                skipped.append(
                    _blank_row(
                        rank,
                        cand["full_name"],
                        (cand.get("created_at") or "")[:10],
                        "legacy_case",
                        "one of the 37 benchmark cases (section 2.2)",
                    )
                )
            elif owners.get(owner, 0) >= OWNER_CAP:
                row = _blank_row(
                    rank, cand["full_name"], (cand.get("created_at") or "")[:10], "owner_cap", ""
                )
                row["pp3_owner"] = False
                skipped.append(row)
            else:
                prepared.append((rank, cand))
        with ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
            results = list(
                pool.map(
                    lambda item: walk_row(
                        item[0],
                        item[1],
                        clones=clones,
                        contexts=contexts,
                        head_ids=head_ids,
                        url_for=url_for,
                    ),
                    prepared,
                )
            )
        rows = skipped + [r for r, _ in results]
        for row in rows:
            row.setdefault("pp3_owner", True)
        mined = [entry for _, entries in results for entry in entries]
        for row, _ in results:
            if row.get("qualifies"):
                owner = str(row["full_name"]).split("/")[0]
                owners[owner] = owners.get(owner, 0) + 1
            capped_total += int(row.get("capped") or 0)
        append_rows(walk_csv, sorted(rows, key=lambda r: r["rank"]))
        if mined:
            merge_adoptions(adoptions, mined)
        walked += len(rows)
        if walked % curve_every < len(rows):
            point = {
                "walked": walked,
                "qualifying": sum(1 for r in _read_rows(walk_csv) if r.get("qualifies") == "True"),
                "capped_positives": capped_total,
            }
            curve.append(point)
            print(
                f"  [{walked:5}/{budget}] qualifying {point['qualifying']:4}  "
                f"capped positives {capped_total:4}",
                flush=True,
            )
    rows_all = _read_rows(walk_csv)
    qualifying = [r for r in rows_all if r.get("qualifies") == "True"]
    prefix = [r for r in rows_all if int(r.get("rank") or 0) < b0]
    prefix_q = [r for r in prefix if r.get("qualifies") == "True"]
    summary = {
        "walked": len(rows_all),
        "b0": b0,
        "budget": budget,
        "target": target,
        "capped_positives": capped_total,
        "qualifying": len(qualifying),
        # q and y are estimated over the unconditional prefix ONLY. Computing them over the
        # whole walk would use a prefix whose length was chosen by the yield it produced.
        "q_over_b0": round(len(prefix_q) / len(prefix), 4) if prefix else None,
        "y_over_b0": (
            round(sum(int(r.get("capped") or 0) for r in prefix_q) / len(prefix_q), 3)
            if prefix_q
            else None
        ),
        "outcomes": _tally(rows_all, "outcome"),
        "curve": curve,
    }
    (out_dir / "walk_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _tally(rows: list[dict[str, str]], field: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for row in rows:
        out[row.get(field, "")] = out.get(row.get(field, ""), 0) + 1
    return dict(sorted(out.items()))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--candidates", type=Path, required=True)
    ap.add_argument("--seed-file", type=Path, required=True, help="file holding SEED_POOL")
    ap.add_argument("--out-dir", type=Path, default=POOL_DIR)
    ap.add_argument("--b0", type=int, default=DEFAULT_B0)
    ap.add_argument("--budget", type=int, default=DEFAULT_B)
    ap.add_argument("--target", type=int, default=DEFAULT_TARGET)
    ap.add_argument("--jobs", type=int, default=4)
    args = ap.parse_args()

    seed = args.seed_file.read_text(encoding="utf-8").strip()
    if not seed:
        raise SystemExit(f"{args.seed_file} is empty — SEED_POOL must be a beacon pulse value")
    with args.candidates.open(encoding="utf-8", newline="") as fh:
        candidates = [r for r in csv.DictReader(fh) if (r.get("full_name") or "").strip()]
    print(f"{len(candidates)} candidates, B0={args.b0} B={args.budget} target={args.target}")
    summary = walk(
        candidates,
        seed,
        out_dir=args.out_dir,
        b0=args.b0,
        budget=args.budget,
        target=args.target,
        jobs=args.jobs,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
