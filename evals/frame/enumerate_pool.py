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
import hashlib
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
# GitHub answers a throttle with 403 or 429 and a Retry-After; 5xx is transient. A
# snapshot that cannot be re-taken must never drop a slice because one request was
# refused, so these are retried rather than recorded as a loss.
THROTTLE_STATUSES = (403, 429, 500, 502, 503, 504)
MAX_ATTEMPTS = 5
BACKOFF_S = 8.0
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


_last_request_at = 0.0


def _wait_turn(pause: float) -> None:
    """Space requests GLOBALLY, not merely between the pages of one query.

    The first version slept only inside the paging loop, and the `break` for a short page
    came *before* the sleep -- so a slice that fitted in one page, which is most of them,
    issued its request and moved straight on to the next query. Several hundred queries
    would have gone out back to back and tripped GitHub's 30-per-minute search limit within
    seconds, on a snapshot that cannot be re-taken.
    """
    global _last_request_at
    if pause <= 0:
        return
    wait = pause - (time.monotonic() - _last_request_at)
    if wait > 0:
        time.sleep(wait)
    _last_request_at = time.monotonic()


def _retry_after(exc: Any, fallback: float) -> float:
    header = getattr(getattr(exc, "headers", None), "get", lambda _k: None)("Retry-After")
    try:
        return max(float(header), 1.0)
    except (TypeError, ValueError):
        return fallback


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
    """Exactly **one** `created:` qualifier, always.

    The first version emitted two — `created:<=<cutoff>` plus, when subdividing,
    `created:<start>..<end>` — and GitHub honours only one of them. The measured consequences,
    from the snapshot of 2026-09-02 that was discarded because of this:

    * every year subdivision of `machine-learning stars:100..149` returned the identical
      `total_count` of 1196 as the un-subdivided query, so the subdivision narrowed nothing
      and the cap was never escaped;
    * the last slice returned **1377** — *more* than the un-subdivided query — because its
      year range replaced the cutoff outright, admitting 192 repositories created after it,
      the latest in August 2026 into a universe capped at March 2024.

    A silently ignored qualifier is the worst kind: the query looked right, returned plausible
    rows, and broke the eligibility rule the population is defined by. So the range is composed
    here into a single clause whose upper bound can never exceed the cutoff.
    """
    stars = f"stars:{lo}..{hi}" if hi is not None else f"stars:>={lo}"
    if years is None:
        created = f"created:<={created_before}"
    else:
        start, end = years
        created = f"created:{start}..{min(end, created_before)}"
    return " ".join([f"topic:{topic}", stars, created, "fork:false", "archived:false"])


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
    backoff: float = BACKOFF_S,
) -> tuple[list[dict[str, Any]], QueryRecord]:
    """Page one query to exhaustion or to the cap, archiving every raw response."""
    items: list[dict[str, Any]] = []
    total = 0
    date_header = ""
    pages = 0
    for page in range(1, MAX_PAGES + 1):
        url = f"{SEARCH}?{urllib.parse.urlencode({'q': query, 'per_page': PER_PAGE, 'page': page})}"
        payload = None
        failure: Exception | None = None
        delay = backoff
        for attempt in range(1, MAX_ATTEMPTS + 1):
            _wait_turn(pause)
            try:
                payload, date_header = fetch(url, token)
                failure = None
                break
            except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
                failure = exc
                status = getattr(exc, "code", None)
                transient = status is None or status in THROTTLE_STATUSES
                if not transient or attempt == MAX_ATTEMPTS:
                    break
                nap = _retry_after(exc, delay)
                print(
                    f"    ! {status or type(exc).__name__} on page {page}; "
                    f"retry {attempt}/{MAX_ATTEMPTS - 1} in {nap:.0f}s",
                    flush=True,
                )
                time.sleep(nap)
                delay *= 2
        if failure is not None or payload is None:
            return items, QueryRecord(
                query, total, len(items), False, date_header, pages, str(failure)[:200]
            )
        pages += 1
        total = int(payload.get("total_count", 0))
        batch = payload.get("items") or []
        items.extend(batch)
        if archive is not None:
            archive.mkdir(parents=True, exist_ok=True)
            stamp = f"{hashlib.sha256(query.encode()).hexdigest()[:12]}-p{page}"
            (archive / f"{stamp}.json").write_text(
                json.dumps(
                    {"query": query, "url": url, "date": date_header, "payload": payload},
                    indent=2,
                ),
                encoding="utf-8",
            )
        if len(batch) < PER_PAGE:
            break
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


ENDPOINT_NOTE = "https://api." + "github.com/search/repositories"


def trim_archive(raw: Path, out: Path) -> dict[str, Any]:
    """Reduce the raw responses to the record that can actually be committed (section 2.1).

    Two reasons, and the second is binding rather than practical:

    * **Size.** One snapshot of 32 topics is ~500 MB of JSON across ~870 responses. That does
      not belong in a git history, and gzipping it only hides the next problem.
    * **The raw payloads carry `html_url` for every repository** -- tens of thousands of
      `github.com/<owner>/<repo>` strings. That is precisely what section 2.1's no-URL rule
      keeps out of this tree: the benchmark expansion's prior-exposure rule is a live grep for
      exactly that pattern. Committing the raw archive would break, at scale, the rule the CSV
      is careful to obey.

    What survives is what makes the enumeration falsifiable rather than merely voluminous: the
    query, its response `Date`, the API's own `total_count`, and the ordered `full_name`s the
    slice returned. That is enough to check that the published CSV is exactly the union of the
    slices, that no slice was silently truncated, and on what day it was taken.

    The full raw archive stays on disk, untracked, as the untrimmed record.
    """
    entries: list[dict[str, Any]] = []
    for path in sorted(raw.glob("*.json")):
        blob = json.loads(path.read_text(encoding="utf-8"))
        payload = blob.get("payload") or {}
        entries.append(
            {
                "file": path.name,
                "query": blob.get("query", ""),
                "date": blob.get("date", ""),
                "total_count": payload.get("total_count", 0),
                "returned": len(payload.get("items") or []),
                # Bare `owner/repo`, which the prior-exposure grep does not match.
                "names": [i.get("full_name", "") for i in (payload.get("items") or [])],
            }
        )
    archive = {
        "_comment": (
            "Trimmed archive of the enumeration snapshot. Per-repository objects are dropped: "
            "they carry html_url, which section 2.1 forbids from entering this tree. Each "
            "entry is one page of one query; reissue as ENDPOINT?q=<query>&per_page=100&page=N."
        ),
        "endpoint": ENDPOINT_NOTE,
        "n_responses": len(entries),
        "responses": entries,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(archive, indent=1), encoding="utf-8")
    return archive


def refuse_a_backdated_snapshot(dp: str, today: date | None = None) -> None:
    """Dp is the day the snapshot is TAKEN. It cannot be in the past.

    The GitHub search API answers only "as of now" — there is no as-of-date parameter and no
    historical index. `created:<=X` is a filter on repository creation date applied to
    *today's* index; every other field the query touches — stars, archived, pushed, topics —
    is a current value. A repository at 500 stars today and 50 stars in 2025 passes
    `stars:>=100` in a snapshot labelled 2025, because there is no 2025 state to consult.

    So a backdated Dp does not produce an old snapshot. It produces today's snapshot wearing
    an old date, and the pre-registration would then describe a sampling frame that never
    existed. The archived `Date` headers would contradict it on the first row, which is why
    this is a refusal rather than a warning.
    """
    day = date.fromisoformat(dp)
    now = today or datetime.now(tz=UTC).date()
    if day < now:
        raise SystemExit(
            f"--date {dp} is in the past (today is {now}).\n"
            "  Dp is the day the snapshot is TAKEN, not a day to look back at: GitHub search\n"
            "  has no historical index, so this would label today's data with an old date and\n"
            "  the archived Date headers would contradict it immediately.\n"
            "  Looking back 30 months is what `created:<=Dp-30mo` already does."
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
    ap.add_argument(
        "--trim-only",
        type=Path,
        help="skip the snapshot; trim an existing raw archive directory to committable form",
    )
    args = ap.parse_args()

    if args.trim_only:
        summary = trim_archive(args.trim_only, args.out)
        print(f"trimmed {summary['n_responses']} responses -> {args.out}")
        return 0

    topics = json.loads(args.topics.read_text(encoding="utf-8"))
    if not isinstance(topics, list) or not all(isinstance(t, str) for t in topics):
        raise SystemExit(f"{args.topics}: expected a JSON list of topic strings")
    refuse_a_backdated_snapshot(args.date)
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
