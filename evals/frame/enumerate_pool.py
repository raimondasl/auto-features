"""Enumerate the judge-validity pool's candidate population. [PREREG-judge-validity-pool §2.1]

The pool needs a candidate list that someone else could reproduce, and one snapshot of the
GitHub search API that nobody can ever re-query. Both facts drive the design:

* **Every query is sliced below the 1,000-result cap.** GitHub returns at most 1,000 results
  per search however many pages you ask for, and it does so *silently* — a topic with 4,000
  matching repositories yields 1,000 rows and no error. A truncated slice is therefore
  indistinguishable from a small one, so this compares each slice's reported `total_count`
  against the cap and **subdivides any slice that overflows**, discarding its partial rows.
  A subdivision that *still* overflows keeps its rows — dropping them would lose real
  coverage — but stamps `TRUNCATED` into each row's `slice`, so the CSV alone says which
  rows came from an incomplete query. Nothing is ever quietly kept.
* **The raw response is archived.** Query URL, raw JSON, the response `Date` header and
  `total_count` per query, written before any filtering. A reader can then check the row
  count against the API's own totals; without it, "we enumerated the population" is
  unfalsifiable.

**No URL column, by rule (§2.1).** Repositories are identified by `full_name` alone. The
benchmark expansion's prior-exposure rule is a live grep for `github.com/<owner>/<repo>`
over this tree, and this file writes up to a few thousand candidate names into it. A `url`
column would turn the pool's own output into that benchmark's exclusion list. `description`
is dropped for the same reason: it is free text and frequently contains a repository URL.

    uv run python evals/frame/enumerate_pool.py --date 2026-09-05 \
        --topics evals/frame/pool/topics.json --out evals/frame/pool/pool-universe-Dp.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

SEARCH = "https://api.github.com/search/repositories"
PER_PAGE = 100
RESULT_CAP = 1000  # GitHub returns no more than this per query, without saying so
MAX_PAGES = RESULT_CAP // PER_PAGE
# Authenticated search allows 30 requests/minute. 2.2 s keeps a margin without a token
# bucket; the walk that follows this is hours long, so a slow enumeration costs nothing.
MIN_INTERVAL_S = 2.2
MIN_STARS = 100
CREATED_MONTHS = 30  # §2.2 PP1's API pre-filter
# Stamped into a row's `slice` when even the subdivided query overflowed, so the CSV alone
# says which rows came from an incomplete slice.
TRUNCATED_MARK = "TRUNCATED"

# Star slices, coarse at the bottom where repositories are dense. Subdivision by creation
# year happens on top of these whenever a slice still reports more than the cap.
STAR_SLICES: tuple[tuple[int, int | None], ...] = (
    (100, 149),
    (150, 199),
    (200, 299),
    (300, 499),
    (500, 999),
    (1000, 2499),
    (2500, 9999),
    (10000, None),
)

COLUMNS = (
    "full_name",
    "created_at",
    "pushed_at",
    "stars",
    "language",
    "topics",
    "slice",
)


@dataclass
class QueryRecord:
    """One search query, kept whether or not it produced anything."""

    query: str
    total_count: int
    fetched: int
    truncated: bool
    date_header: str
    pages: int = 0
    error: str = ""


@dataclass
class Enumeration:
    rows: dict[str, dict[str, Any]] = field(default_factory=dict)
    queries: list[QueryRecord] = field(default_factory=list)

    @property
    def truncated(self) -> list[QueryRecord]:
        return [q for q in self.queries if q.truncated]


def github_token() -> str:
    """`GITHUB_TOKEN` if set, else whatever `gh` is already authenticated with.

    Unauthenticated search is 10 requests/minute and this needs hundreds, so a missing token
    is a hard failure rather than a slow run that dies halfway through a snapshot.
    """
    token = (os.environ.get("GITHUB_TOKEN") or "").strip()
    if token:
        return token
    try:
        out = subprocess.run(
            ["gh", "auth", "token"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:  # pragma: no cover - environment
        raise SystemExit(f"no GITHUB_TOKEN and `gh auth token` failed: {exc}") from exc
    token = (out.stdout or "").strip()
    if not token:
        raise SystemExit("no GITHUB_TOKEN in the environment and `gh auth token` returned nothing")
    return token


def _fetch(url: str, token: str) -> tuple[dict[str, Any], str]:
    req = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "reporadar-validity-pool",
        },
    )
    with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310 - fixed https host
        payload = json.loads(resp.read().decode("utf-8"))
        return payload, resp.headers.get("Date", "")


def _query(
    topic: str, lo: int, hi: int | None, created_before: str, years: tuple[str, str] | None
) -> str:
    stars = f"stars:{lo}..{hi}" if hi is not None else f"stars:>={lo}"
    parts = [f"topic:{topic}", stars, f"created:<={created_before}", "fork:false", "archived:false"]
    if years is not None:
        parts.append(f"created:{years[0]}..{years[1]}")
    return " ".join(parts)


def _row(item: dict[str, Any], slice_name: str) -> dict[str, Any]:
    """Deliberately narrow. See the module docstring on why there is no URL and no
    description: both carry `github.com/<owner>/<repo>` strings into this tree."""
    return {
        "full_name": item.get("full_name", ""),
        "created_at": (item.get("created_at") or "")[:10],
        "pushed_at": (item.get("pushed_at") or "")[:10],
        "stars": item.get("stargazers_count", 0),
        "language": item.get("language") or "",
        "topics": "|".join(sorted(item.get("topics") or [])),
        "slice": slice_name,
    }


Fetcher = Callable[[str, str], tuple[dict[str, Any], str]]


def run_query(
    query: str,
    token: str,
    *,
    fetch: Fetcher = _fetch,
    archive: Path | None = None,
    pause: float = MIN_INTERVAL_S,
) -> tuple[list[dict[str, Any]], QueryRecord]:
    """Page one query to exhaustion or to the cap, archiving every raw response."""
    items: list[dict[str, Any]] = []
    total = 0
    date_header = ""
    pages = 0
    for page in range(1, MAX_PAGES + 1):
        url = f"{SEARCH}?{urllib.parse.urlencode({'q': query, 'per_page': PER_PAGE, 'page': page})}"
        try:
            payload, date_header = fetch(url, token)
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
            return items, QueryRecord(
                query, total, len(items), False, date_header, pages, str(exc)[:200]
            )
        pages += 1
        total = int(payload.get("total_count", 0))
        batch = payload.get("items") or []
        items.extend(batch)
        if archive is not None:
            archive.mkdir(parents=True, exist_ok=True)
            stamp = f"{abs(hash(query)) % (10**12):012d}-p{page}"
            (archive / f"{stamp}.json").write_text(
                json.dumps(
                    {"query": query, "url": url, "date": date_header, "payload": payload},
                    indent=2,
                ),
                encoding="utf-8",
            )
        if len(batch) < PER_PAGE:
            break
        if pause:
            time.sleep(pause)
    return items, QueryRecord(query, total, len(items), total > RESULT_CAP, date_header, pages)


def enumerate_universe(
    topics: list[str],
    created_before: str,
    token: str,
    *,
    fetch: Fetcher = _fetch,
    archive: Path | None = None,
    pause: float = MIN_INTERVAL_S,
    year_splits: tuple[tuple[str, str], ...] = (
        ("2008-01-01", "2015-12-31"),
        ("2016-01-01", "2018-12-31"),
        ("2019-01-01", "2020-12-31"),
        ("2021-01-01", "2022-12-31"),
        ("2023-01-01", "2030-12-31"),
    ),
) -> Enumeration:
    """Every topic × star slice, subdivided by creation year whenever a slice overflows."""
    out = Enumeration()
    for topic in topics:
        for lo, hi in STAR_SLICES:
            query = _query(topic, lo, hi, created_before, None)
            items, record = run_query(query, token, fetch=fetch, archive=archive, pause=pause)
            if record.truncated:
                # The slice overflowed: its rows are unusable as a complete set, so they are
                # discarded and the slice is re-run subdivided. Keeping them would mix a
                # truncated slice with complete ones and no reader could tell.
                out.queries.append(record)
                for years in year_splits:
                    sub = _query(topic, lo, hi, created_before, years)
                    sub_items, sub_record = run_query(
                        sub, token, fetch=fetch, archive=archive, pause=pause
                    )
                    out.queries.append(sub_record)
                    # A subdivision that ALSO overflowed cannot be split further here, and
                    # its rows are the API's relevance-ranked top 1,000 rather than a
                    # complete set. They are kept — dropping them would lose real coverage
                    # for no gain — but marked in the row itself, so the CSV alone tells a
                    # reader which rows came from an incomplete slice. `coverage.json`
                    # records the same fact; a reader should not have to cross-reference two
                    # files to find out whether the frame is exhaustive.
                    label = f"{topic}|{lo}|{years[0][:4]}"
                    if sub_record.truncated:
                        label += f"|{TRUNCATED_MARK}"
                    for item in sub_items:
                        out.rows.setdefault(item.get("full_name", ""), _row(item, label))
                continue
            out.queries.append(record)
            for item in items:
                out.rows.setdefault(item.get("full_name", ""), _row(item, f"{topic}|{lo}"))
    out.rows.pop("", None)
    return out


def write_universe(enumeration: Enumeration, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(COLUMNS))
        writer.writeheader()
        for name in sorted(enumeration.rows):
            writer.writerow({k: enumeration.rows[name].get(k, "") for k in COLUMNS})


def write_coverage(enumeration: Enumeration, out: Path) -> None:
    """What the API said it had, beside what we took. The falsifiability half of §2.1."""
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "n_rows": len(enumeration.rows),
                "n_queries": len(enumeration.queries),
                "n_truncated": len(enumeration.truncated),
                "truncated_queries": [q.query for q in enumeration.truncated],
                "errors": [
                    {"query": q.query, "error": q.error} for q in enumeration.queries if q.error
                ],
                "queries": [
                    {
                        "query": q.query,
                        "total_count": q.total_count,
                        "fetched": q.fetched,
                        "pages": q.pages,
                        "truncated": q.truncated,
                        "date": q.date_header,
                        "error": q.error,
                    }
                    for q in enumeration.queries
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def created_cutoff(dp: str, months: int = CREATED_MONTHS) -> str:
    """`Dp − months`, as a date string the GitHub search API accepts."""
    day = date.fromisoformat(dp)
    total = day.year * 12 + (day.month - 1) - months
    return date(total // 12, total % 12 + 1, min(day.day, 28)).isoformat()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--date", required=True, help="Dp, ISO date; the snapshot day")
    ap.add_argument("--topics", type=Path, required=True, help="JSON list of GitHub topics")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--archive", type=Path, help="raw response directory (default: beside --out)")
    ap.add_argument("--pause", type=float, default=MIN_INTERVAL_S)
    args = ap.parse_args()

    topics = json.loads(args.topics.read_text(encoding="utf-8"))
    if not isinstance(topics, list) or not all(isinstance(t, str) for t in topics):
        raise SystemExit(f"{args.topics}: expected a JSON list of topic strings")
    cutoff = created_cutoff(args.date)
    archive = args.archive or args.out.parent / "raw"
    print(
        f"Dp={args.date}  created:<={cutoff}  topics={len(topics)}  "
        f"queries<={len(topics) * len(STAR_SLICES)}  archive={archive}",
        flush=True,
    )
    enumeration = enumerate_universe(
        topics, cutoff, github_token(), archive=archive, pause=args.pause
    )
    write_universe(enumeration, args.out)
    write_coverage(enumeration, args.out.parent / "coverage.json")
    print(
        f"\n{len(enumeration.rows)} distinct repositories from {len(enumeration.queries)} queries; "
        f"{len(enumeration.truncated)} still truncated after subdivision -> {args.out}"
    )
    if enumeration.truncated:
        print("  TRUNCATED (recorded, not hidden):")
        for q in enumeration.truncated[:10]:
            print(f"    {q.query}  total_count={q.total_count} fetched={q.fetched}")
    stamp = datetime.now(tz=UTC).isoformat(timespec="seconds")
    print(f"  snapshot taken {stamp}; raw responses under {archive}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
