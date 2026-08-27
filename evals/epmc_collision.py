"""Does a keyword source return biology to a compiler? A $0 collision probe. [P22, P23]

Two sources, one probe. `--source europepmc` is P22; `--source openalex` is P23. The query
construction is shared because it is the part that must match what a real run sends; only the
classification differs, and it has to, because the two indexes describe themselves differently
(Europe PMC by MeSH, OpenAlex by its own field taxonomy).

Europe PMC gave **+4.00 net@2** on the six bio cases (P21), and the mechanism was coverage
rather than ranking: 54% of the shown digest came from it, at 0.97 precision. That is the
strongest multi-source evidence this project has, and it is the reason not to generalise it
carelessly — Europe PMC is a *biomedical* index, and the core-25 repositories are compilers,
databases, HTTP servers and linters.

The question this answers, before any judge call: **when a non-biology repository's own
queries are sent to Europe PMC, does anything come back, and is it plausibly on-topic?**

Three outcomes, and they imply different designs:

* **Nothing comes back.** Adding the source is harmless everywhere and useless outside
  biology — routing is justified but nearly free, since only bio repos pay.
* **On-topic results come back.** Europe PMC indexes more than biology (it carries some CS
  venues), so a single `arxiv,europepmc` default may be right.
* **Off-topic results come back, confidently.** The dangerous case. `alignment`, `kernel`,
  `expression`, `translation` and `pruning` all mean something in biology, and net@2 charges
  **2 per false positive** — so a channel that answers off-domain is worse than one that
  abstains. This is the outcome that would make per-domain routing necessary rather than
  merely tidy.

**No LLM calls and no judge calls.** Queries are built by `collector.build_queries` from the
repository profile, which is deterministic; the only network is Europe PMC's free API. Topic
classification is by MeSH/journal metadata that Europe PMC returns anyway, not by a model —
a model asked "is this on-topic" would be the judge, at judge prices, and the point of a
stage-1 probe is to be answerable without one.

    uv run python evals/epmc_collision.py --dry-run              # $0: the queries, no network
    uv run python evals/epmc_collision.py                        # $0: Europe PMC  [P22]
    uv run python evals/epmc_collision.py --source openalex      # $0: OpenAlex    [P23]
    uv run python evals/epmc_collision.py --source openalex --report
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.parse
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from harness import WORK_DIR, profile_case_repo  # noqa: E402
from run_judge_eval import load_dotenv  # noqa: E402

from reporadar.collector import build_queries, to_plain_keywords  # noqa: E402
from reporadar.config import ArxivConfig, QueriesConfig  # noqa: E402
from reporadar.sources.europepmc import EPMC_SEARCH  # noqa: E402
from reporadar.sources.europepmc import _request_json as epmc_request  # noqa: E402

EVALS = Path(__file__).resolve().parent
SOURCES = ("europepmc", "openalex")


def out_path(source: str) -> Path:
    """One artifact per source. They measure different quantities under different taxonomies,
    so a single file would invite exactly the averaging the numbers cannot support."""
    return EVALS / f"{source}_collision.json"


# The same cut the product applies: `pipeline` sends `queries[:KEYWORD_SOURCE_QUERIES]`.
# Imported rather than retyped so the probe cannot measure a different query set than the
# one a real run would send.
from reporadar.pipeline import KEYWORD_SOURCE_QUERIES  # noqa: E402

# How many hits to inspect per query. The question is what the channel WOULD contribute, and
# only the top of each result list can reach a 15-paper digest through the gate.
PER_QUERY = 10


# **Indexed as biomedical**, by Europe PMC's own metadata rather than by a guess of ours.
#
# The first version substring-matched journal names out of `journalTitle` and `pubType`, and
# reported 21% biomedical across the core 25. Both fields are `None` on essentially every
# record -- the real ones are `journalInfo` (nested) and `meshHeadingList` -- so the flag was
# reading empty strings and the 21% measured nothing. Reading the titles is what exposed it:
# the top hit for the linter's own query `lint code` is "Occurrence of postoperative
# pneumoencephalus in posterior fossa surgery".
#
# MeSH is the signal because MeSH IS the biomedical thesaurus: a record carrying MeSH
# headings has been indexed into it by a human cataloguer at the NLM. `subsetList` code "IM"
# (Index Medicus) says the same thing a second way. Neither is a heuristic about words.
def _looks_biomedical(rec: dict[str, Any]) -> bool:
    if (rec.get("meshHeadingList") or {}).get("meshHeading"):
        return True
    subsets = (rec.get("subsetList") or {}).get("subset") or []
    if any((s.get("code") or "").upper() == "IM" for s in subsets):
        return True
    journal = ((rec.get("journalInfo") or {}).get("journal") or {}).get("title") or ""
    return any(m in journal.lower() for m in _JOURNAL_MARKERS)


# Kept as a third fallback for records with neither MeSH nor a subset -- preprints, mostly.
_JOURNAL_MARKERS = (
    "biol",
    "genom",
    "genet",
    "bioinform",
    "medic",
    "clinic",
    "cancer",
    "cell",
    "protein",
    "molecul",
    "neuro",
    "health",
    "patho",
    "immun",
    "microb",
    "plos",
    "lancet",
    "bmc",
    "nucleic",
)


# OpenAlex labels every work with `primary_topic.field`, a 26-field taxonomy of its own.
# That is a far better instrument than the biomedical flag Europe PMC forces on us: it does
# not need a marker list, it cannot silently read an empty field, and it reports WHICH
# discipline came back rather than a yes/no.
#
# "On domain" for this benchmark is the CS-adjacent set below. Every core-25 repository is a
# software project, so a returned work in Medicine or Biochemistry is a collision by the same
# argument P22 made -- the difference is that here the source itself says so.
ON_DOMAIN_FIELDS = frozenset(
    {
        "Computer Science",
        "Engineering",
        "Mathematics",
        "Physics and Astronomy",
        "Decision Sciences",
    }
)


def _oa_field(work: dict[str, Any]) -> str:
    return ((work.get("primary_topic") or {}).get("field") or {}).get("display_name") or "(none)"


def probe_openalex(case: str, queries: list[str], api_key: str | None) -> dict[str, Any]:
    """P23. Uses the adapter's own `search_works`, so the filter and field selection are the
    ones a real run sends rather than a second copy of them."""
    from reporadar.sources.openalex import search_works

    hits: list[dict[str, Any]] = []
    per_query: dict[str, int] = {}
    fields: dict[str, int] = {}
    for q in queries:
        try:
            works = search_works(q, limit=PER_QUERY, api_key=api_key)
        except Exception as exc:  # noqa: BLE001 -- a refusal is not an empty result
            print(f"    ! {case}: OpenAlex failed for {q[:40]!r}: {str(exc)[:80]}")
            per_query[q] = -1
            continue
        per_query[q] = len(works)
        for w in works:
            field = _oa_field(w)
            fields[field] = fields.get(field, 0) + 1
            hits.append(
                {
                    "query": q,
                    "title": (w.get("title") or w.get("display_name") or "")[:120],
                    "field": field,
                    "off_domain": field not in ON_DOMAIN_FIELDS,
                }
            )
    n = len(hits)
    off = sum(1 for h in hits if h["off_domain"])
    return {
        "queries": queries,
        "hits_per_query": per_query,
        "n_hits": n,
        "n_off_domain": off,
        "off_domain_share": round(off / n, 3) if n else None,
        "fields": dict(sorted(fields.items(), key=lambda kv: -kv[1])),
        "sample": hits[:8],
    }


def queries_for(case: str) -> list[str]:
    """The plain-keyword queries a real run would send Europe PMC for this repository."""
    # `profile_case_repo` and `build_queries` are the SHARED implementations the harness and
    # the product use. A probe that re-derived either would measure a query set no run sends,
    # which is the C-3 shape and the reason this file imports rather than reconstructs.
    profile = profile_case_repo(WORK_DIR / case)
    arxiv_q = build_queries(profile, QueriesConfig(bigrams="verified"), ArxivConfig())
    return [to_plain_keywords(q) for q in arxiv_q[:KEYWORD_SOURCE_QUERIES]]


def probe(case: str, queries: list[str]) -> dict[str, Any]:
    hits: list[dict[str, Any]] = []
    per_query: dict[str, int] = {}
    for q in queries:
        params = urllib.parse.urlencode(
            {"query": q, "format": "json", "resultType": "core", "pageSize": PER_QUERY}
        )
        try:
            payload = epmc_request(f"{EPMC_SEARCH}?{params}")
        except Exception as exc:  # noqa: BLE001 — a refusal is not an empty result
            print(f"    ! {case}: Europe PMC failed for {q[:40]!r}: {str(exc)[:80]}")
            per_query[q] = -1
            continue
        results = ((payload or {}).get("resultList") or {}).get("result") or []
        per_query[q] = len(results)
        for r in results:
            hits.append(
                {
                    "query": q,
                    "title": (r.get("title") or "")[:120],
                    "journal": ((r.get("journalInfo") or {}).get("journal") or {}).get("title")
                    or "",
                    "mesh": [
                        h.get("descriptorName")
                        for h in ((r.get("meshHeadingList") or {}).get("meshHeading") or [])[:3]
                    ],
                    "biomedical": _looks_biomedical(r),
                }
            )
    n = len(hits)
    bio = sum(1 for h in hits if h["biomedical"])
    return {
        "queries": queries,
        "hits_per_query": per_query,
        "n_hits": n,
        "n_biomedical": bio,
        "biomedical_share": round(bio / n, 3) if n else None,
        "sample": hits[:8],
    }


def report(data: dict[str, Any]) -> int:
    source = data.get("source", "europepmc")
    flag_n = "n_biomedical" if source == "europepmc" else "n_off_domain"
    flag_s = "biomedical_share" if source == "europepmc" else "off_domain_share"
    label = "biomed" if source == "europepmc" else "off-dom"
    print(f"source: {source}\n")
    print(f"{'case':<14}{'queries':>8}{'hits':>7}{label:>8}{'share':>8}")
    tot_h = tot_b = 0
    for case, row in sorted(data["cases"].items()):
        share = row[flag_s]
        tot_h += row["n_hits"]
        tot_b += row[flag_n]
        print(
            f"{case:<14}{len(row['queries']):>8}{row['n_hits']:>7}{row[flag_n]:>8}"
            f"{(f'{share:.0%}' if share is not None else '-'):>8}"
        )
    print(f"{'TOTAL':<14}{'':>8}{tot_h:>7}{tot_b:>8}{(tot_b / tot_h if tot_h else 0):>7.0%}")
    if source == "openalex":
        agg: dict[str, int] = {}
        for r in data["cases"].values():
            for f, n in (r.get("fields") or {}).items():
                agg[f] = agg.get(f, 0) + n
        print("\nfields returned, all cases:")
        for f, n in sorted(agg.items(), key=lambda kv: -kv[1])[:10]:
            mark = "  " if f in ON_DOMAIN_FIELDS else "!!"
            print(f"  {mark} {f:<40} {n:>5}  {n / max(1, sum(agg.values())):>5.0%}")
    silent = [c for c, r in data["cases"].items() if r["n_hits"] == 0]
    print(f"\nrepositories {source} returned NOTHING for: {len(silent)}/{len(data['cases'])}")
    if silent:
        print(f"  {sorted(silent)}")
    what = "biomedical" if source == "europepmc" else "off-domain"
    print(
        f"\nReading: a high {what} share on a software repository is the COLLISION case —"
        "\nthe channel answering confidently off-domain, which net@2 charges 2 per paper for."
    )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Europe PMC keyword collision on the core 25.")
    ap.add_argument("--source", default="europepmc", choices=SOURCES)
    ap.add_argument("--case", help="Comma-separated subset.")
    ap.add_argument("--dry-run", action="store_true", help="$0: print queries, no network.")
    ap.add_argument("--report", action="store_true", help="$0: re-read the artifact.")
    args = ap.parse_args()

    out = out_path(args.source)
    if args.report:
        if not out.is_file():
            print(f"no artifact at {out}")
            return 1
        return report(json.loads(out.read_text(encoding="utf-8")))

    bench = yaml.safe_load((EVALS / "benchmark.yaml").read_text(encoding="utf-8"))
    cases = [
        c["name"]
        for c in bench["cases"]
        if c.get("live_repo") and not c["name"].startswith(("bio-", "mat-"))
    ]
    if args.case:
        want = set(args.case.split(","))
        cases = [c for c in cases if c in want]

    api_key = None
    if args.source == "openalex":
        load_dotenv(EVALS / ".env")
        api_key = os.environ.get("OPENALEX_API_KEY") or None
        if not api_key:
            # Keyless OpenAlex is throttled to a small daily test allowance, which would
            # silently truncate the result lists and understate whatever it returns.
            print("! OPENALEX_API_KEY is not set; keyless results are throttled. Refusing.")
            return 1

    out: dict[str, Any] = {
        "_comment": (
            f"What {args.source} returns for the core-25 repositories' own queries. No LLM "
            "and no judge calls. Europe PMC is classified by MeSH/Index Medicus (P22); "
            "OpenAlex by its own primary_topic.field taxonomy (P23). Derived by "
            "evals/epmc_collision.py, pinned by tests/test_epmc_collision.py."
        ),
        "source": args.source,
        "per_query_hits": PER_QUERY,
        "cases": {},
    }
    for i, case in enumerate(cases, 1):
        qs = queries_for(case)
        print(f"[{i}/{len(cases)}] {case}: {len(qs)} queries")
        for q in qs[:3]:
            print(f"    {q!r}")
        if args.dry_run:
            out["cases"][case] = {
                "queries": qs,
                "hits_per_query": {},
                "n_hits": 0,
                "n_biomedical": 0,
                "biomedical_share": None,
                "sample": [],
            }
            continue
        row = probe(case, qs) if args.source == "europepmc" else probe_openalex(case, qs, api_key)
        out["cases"][case] = row
        flagged = row.get("n_biomedical", row.get("n_off_domain"))
        print(f"    -> {row['n_hits']} hits, {flagged} off-domain")

    if args.dry_run:
        print("\n(dry run — no network, nothing written)")
        return 0
    out_path(args.source).write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(f"\nwrote {out_path(args.source).name}\n")
    return report(out)


if __name__ == "__main__":
    raise SystemExit(main())
