"""What is the OpenAlex channel actually made of? ($0, judge-free, no LLM.)

    uv run python evals/openalex_venue_mix.py                       # the matsci-6
    uv run python evals/openalex_venue_mix.py --cases bio-align     # any benchmark case

`evals/openalex_yield.py` asks whether OpenAlex papers can reach a ranked top-10. It cannot say
**what they are**, because `sources/openalex.py` selects only the fields the pipeline needs and
venue is not one of them. That distinction is the whole open question: §20.12 and §21.6 both
close with *"the matsci half — ChemRxiv via OpenAlex remains unexercised"*, and §6 lists
**journal-only literature** (PRB, JCTC; Bioinformatics, Genome Biology, NAR) as reachable only
through this source and never validated. "OpenAlex competes" does not distinguish *"it found
arXiv papers our own queries missed"* from *"it found the journal literature arXiv cannot
carry"*, and only the second answers the question.

**How it avoids re-implementing the query.** Collection is `sources.openalex.collect_papers`,
unchanged and unwrapped — the same call `evals/harness.collect_live_papers` makes, with the
product's own `build_queries` and `KEYWORD_SOURCE_QUERIES` cap. Venue then comes from a bulk
metadata lookup of the DOIs that come back. Nothing here rebuilds the `search` string or the
`type:article|preprint` filter, so this probe cannot drift away from the source it describes.

Reports, per case and pooled: how much of the channel is an arXiv paper under another name,
how much is peer-reviewed journal literature, how much is a preprint server, and how much is a
repository record — Figshare, Zenodo — which is where supporting information lives.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

EVALS = Path(__file__).resolve().parent
sys.path.insert(0, str(EVALS))
sys.path.insert(0, str(EVALS.parent / "src"))

from harness import WORK_DIR, load_benchmark, profile_case_repo  # noqa: E402
from openalex_yield import ALL_TIME_DAYS, load_key  # noqa: E402

from reporadar.collector import build_queries, to_plain_keywords  # noqa: E402
from reporadar.config import ArxivConfig, QueriesConfig  # noqa: E402
from reporadar.paper_id import is_arxiv_id  # noqa: E402
from reporadar.pipeline import KEYWORD_SOURCE_QUERIES  # noqa: E402
from reporadar.sources import openalex  # noqa: E402

OUT = WORK_DIR / "openalex_venue_mix.json"
MATSCI = ("mat-mlip", "mat-chgpot", "mat-descriptors", "mat-toolkit", "mat-featurize", "mat-phonon")
BATCH = 50  # OpenAlex accepts up to 50 ids in one `filter=doi:a|b|...`


def lookup_venues(dois: list[str]) -> dict[str, dict[str, Any]]:
    """Venue, work type and source type for a list of DOIs. Metadata only, no search."""
    out: dict[str, dict[str, Any]] = {}
    for start in range(0, len(dois), BATCH):
        chunk = dois[start : start + BATCH]
        url = "https://api.openalex.org/works?" + urllib.parse.urlencode(
            {
                "filter": "doi:" + "|".join("https://doi.org/" + d for d in chunk),
                "per_page": str(BATCH),
                "select": "doi,title,type,publication_year,primary_location,primary_topic",
            }
        )
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read())
        for work in data.get("results", []):
            doi = (work.get("doi") or "").replace("https://doi.org/", "").lower()
            src = (work.get("primary_location") or {}).get("source") or {}
            topic = work.get("primary_topic") or {}
            out[doi] = {
                "title": work.get("title") or "",
                "type": work.get("type") or "?",
                "year": work.get("publication_year"),
                "venue": src.get("display_name") or "(no venue)",
                "venue_type": src.get("type") or "?",
                # OpenAlex's OWN field label, not a reading of the venue name. §30 failed by
                # sorting papers on what their titles sounded like, and "this pool is full of
                # cardiology" is exactly the kind of claim that invites the same mistake — so
                # the taxonomy belongs to the index rather than to the author.
                "field": ((topic.get("field") or {}).get("display_name")) or "?",
            }
    return out


# OpenAlex gives bioRxiv, ChemRxiv, Research Square, SSRN and Figshare the SAME source type,
# `repository`, so that field alone cannot tell a preprint server from a data repository —
# and the difference is the whole point here: one is literature this project wants, the other
# is where supporting-information files live. Named explicitly rather than inferred.
PREPRINT_SERVERS = {
    "biorxiv",
    "medrxiv",
    "chemrxiv",
    "research square",
    "ssrn",
    "ssrn electronic journal",
    "preprints.org",
    "authorea",
    "techrxiv",
    "osf preprints",
}


def classify(paper_id: str, venue: dict[str, Any] | None) -> str:
    """One of five buckets. `journal-literature` and `preprint-server` are what §6 called
    unreachable; `repository-record` is the bucket nobody asked for and it is a defect."""
    if is_arxiv_id(paper_id):
        return "arxiv-under-another-name"
    if not paper_id.startswith("doi:"):
        # An `oa:W...` handle is a work OpenAlex holds no DOI for, so there is nothing to
        # look up and no venue to report. Its own bucket rather than "unresolved": 87 of the
        # first run's 93 unclassified rows were these, and calling a structural gap a lookup
        # failure would have made the probe look 15x less reliable than it is.
        return "no-doi-openalex-handle"
    if venue is None:
        return "unresolved"
    vt, name = venue["venue_type"], venue["venue"].lower()
    if vt in {"journal", "conference", "book series", "ebook platform"}:
        return "journal-literature"
    if vt == "repository":
        if venue["type"] == "preprint" or any(s in name for s in PREPRINT_SERVERS):
            return "preprint-server"
        # Figshare/Zenodo records. Supporting information is indexed as a work in its own
        # right, which is how an SI file becomes a candidate paper.
        return "repository-record"
    return f"other:{vt}"


def measure(case: str, key: str | None) -> dict[str, Any] | None:
    repo = WORK_DIR / case
    if not repo.is_dir():
        print(f"  {case:18} no clone at {repo} — skipped")
        return None
    profile = profile_case_repo(repo)
    cfg = ArxivConfig(
        categories=["cs.LG", "cs.CL", "cs.CV", "cs.SE"],
        max_results_per_query=50,
        lookback_days=ALL_TIME_DAYS,
        sort_by="relevance",
    )
    queries = build_queries(profile, QueriesConfig(), cfg)
    papers = openalex.collect_papers(
        [to_plain_keywords(q) for q in queries[:KEYWORD_SOURCE_QUERIES]],
        lookback_days=ALL_TIME_DAYS,
        api_key=key,
    )
    ids = [p["arxiv_id"] for p in papers]
    venues = lookup_venues([i[4:] for i in ids if i.startswith("doi:")])
    rows = []
    for pid in ids:
        v = venues.get(pid[4:].lower()) if pid.startswith("doi:") else None
        rows.append({"id": pid, "bucket": classify(pid, v), **(v or {})})
    counts = collections.Counter(r["bucket"] for r in rows)
    print(
        f"  {case:18} {len(rows):4d} arrived  "
        + "  ".join(f"{k}={v}" for k, v in sorted(counts.items()))
    )
    return {"case": case, "n": len(rows), "counts": dict(counts), "rows": rows}


def report(results: list[dict[str, Any]]) -> None:
    rows = [r for res in results for r in res["rows"]]
    counts = collections.Counter(r["bucket"] for r in rows)
    total = len(rows)
    print("\n" + "=" * 92)
    print("WHAT THE CHANNEL IS MADE OF")
    print("=" * 92)
    for bucket, n in counts.most_common():
        print(f"  {bucket:28} {n:5d}  {n / total:6.1%}")

    journals = [r for r in rows if r["bucket"] == "journal-literature"]
    print(f"\n  top venues among the {len(journals)} journal papers:")
    for venue, n in collections.Counter(r["venue"] for r in journals).most_common(20):
        print(f"    {n:4d}  {venue[:70]}")

    for bucket, note in (
        ("preprint-server", "the route §20.12 expected"),
        ("repository-record", "supporting information enters here — a defect, not a source"),
    ):
        sel = [r for r in rows if r["bucket"] == bucket]
        print(f"\n  {bucket} ({len(sel)}) — {note}:")
        for venue, n in collections.Counter(r["venue"] for r in sel).most_common(10):
            print(f"    {n:4d}  {venue[:70]}")

    other = [r for r in rows if r["bucket"].startswith("other:") or r["bucket"] == "unresolved"]
    if other:
        print(f"\n  unclassified ({len(other)}):")
        for venue, n in collections.Counter(r.get("venue", "?") for r in other).most_common(10):
            print(f"    {n:4d}  {str(venue)[:70]}")

    fields = collections.Counter(r["field"] for r in rows if r.get("field", "?") != "?")
    resolved = sum(fields.values())
    on_topic = fields["Materials Science"] + fields["Chemistry"]
    print(f"\n  OpenAlex's OWN field label, on the {resolved} works it resolved:")
    for field, n in fields.most_common(12):
        print(f"    {n:5d}  {n / resolved:6.1%}  {field}")
    print(
        f"\n  Materials Science + Chemistry = {on_topic}/{resolved} = {on_topic / resolved:.1%}."
        "\n  OpenAlex `search=` carries no domain filter, unlike Europe PMC's `SRC:PPR`, so a"
        "\n  generic word in a query reaches every field that happens to use it."
    )

    print(
        "\n  The question §20.12 and §21.6 left open was ChemRxiv. What this measures is\n"
        "  whichever of the two the channel actually carries — the buckets decide, not the\n"
        "  expectation."
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cases", default=",".join(MATSCI), help="comma-separated benchmark cases")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    key = load_key()
    if not key:
        print("No OPENALEX_API_KEY in evals/.env — a keyless caller's zeros are throttling,")
        print("not measurements (evals/openalex_yield.py says the same). Aborting.")
        return 1

    known = {c["name"] for c in load_benchmark()["cases"]}
    wanted = [c.strip() for c in args.cases.split(",") if c.strip()]
    unknown = set(wanted) - known
    if unknown:
        raise SystemExit(f"Unknown case(s): {', '.join(sorted(unknown))}")

    print("=" * 92)
    print(f"OPENALEX VENUE MIX — {len(wanted)} cases, $0, no LLM")
    print("=" * 92)
    results = [r for c in wanted if (r := measure(c, key)) is not None]
    if not results:
        print("\nNothing measured.")
        return 1
    report(results)
    Path(args.out).write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
