"""Europe PMC as a keyword-searchable bioRxiv/medRxiv source.

`sources/biorxiv.py` talks to bioRxiv's `details` endpoint, which is a **date-interval
listing, not a search**: it fetches a window and filters locally. Under the measured
all-time configuration that window opens in 2013, the adapter stops after 40 pages, and the
papers it returns are a decade old. There is no keyword parameter to add — the endpoint does
not have one.

Europe PMC does have one, and it indexes both servers. The query is

    (<plain keywords>) AND SRC:PPR AND (PUBLISHER:"bioRxiv" OR PUBLISHER:"medRxiv")

with `FIRST_PDATE:[start TO end]` appended when the window is bounded.

Everything below that is not obvious was measured against the live API on 2026-08-19 before
this file was written, because the last two specs written from API documentation each had a
defect only a probe found. What the probe changed:

* **The keywords must NOT be quoted.** A quoted string is an exact-phrase match, and the
  strings this receives are bags of words from `collector.to_plain_keywords`, not phrases.
  Over eight product-shaped queries, quoting returned 0 or 1 hits for four of them against
  85-2,239 unquoted: `"sequence alignment long reads"` 0 vs 2,239, `"molecular dynamics gpu
  simulation"` 0 vs 148. Quoting would have silently emptied the channel, which is the same
  failure `to_plain_keywords` was written to fix, inverted.
* **Titles carry markup too**, not only abstracts: 18% of 785 sampled records had `<i>` or
  `<sup>` in the *title*, and 36% had `<h4>` in the abstract.
* **The obvious way to strip it destroys the text.** See `_strip_markup`.
* `publisher` and `pubType` come back **null** under `resultType=core` even though
  `PUBLISHER:` filters correctly, so a record cannot be labelled bioRxiv-or-medRxiv from the
  response. Nothing here depends on knowing which.
* Every one of 785 records had a DOI, an abstract and a `firstPublicationDate`. Both DOI
  prefixes are live — 10.1101 (284) and 10.64898 (216), bioRxiv's newer one — and
  :func:`reporadar.paper_id.doi_key` gives both the same shape of id, so a preprint arriving
  here and from OpenAlex is one paper.
* An empty result is `hitCount: 0` with a `resultList` present, so "found nothing" is
  distinguishable from "refused" — and the distinction is kept, because a refusal counted as
  a zero is a mistake this project has published twice.
"""

from __future__ import annotations

import json as json_mod
import logging
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import UTC, datetime, timedelta
from typing import Any

from reporadar.paper_id import doi_key

logger = logging.getLogger(__name__)

EPMC_SEARCH = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"

# `SRC:PPR` is Europe PMC's preprint corpus. The publisher clause narrows it to the two
# servers this source is named for; measured, it drops 1,808 of 8,186 preprints on a
# single-cell query, and every one dropped was Research Square (10.21203), Preprints.org
# (10.20944), F1000 (10.12688) or Authorea (10.22541). That is the intended scope and not a
# defect, but it IS a choice — widening it is a one-line change to this constant.
PREPRINT_FILTER = 'SRC:PPR AND (PUBLISHER:"bioRxiv" OR PUBLISHER:"medRxiv")'

# 100 is the largest page the endpoint served in testing, and it served it reliably.
PAGE_SIZE = 100
# One request per query, spaced. 22 consecutive requests at this spacing completed without a
# refusal; a burst of six with no spacing drew 504s and then a 503.
REQUEST_INTERVAL_S = 1.0
# Beyond roughly a year a date clause stops narrowing anything and only costs query
# complexity: the measured configuration runs `lookback_days: 36500`, and asking for
# "everything since 1926" is a slower way to ask for everything.
_UNBOUNDED_DAYS = 365


class EuropePMCError(RuntimeError):
    """The API refused. Distinct from "the API found nothing" — see the module docstring."""


# A tag is a name, optional whitespace, and a close. Nothing else.
#
# `<[^>]+>` is the obvious pattern and it mutilates biology abstracts: `p < 0.001` opens a
# span that the regex closes at the NEXT real tag, taking every character in between. On 785
# sampled records that pattern removed 9,277 characters where this one removes 5,367 — 3,910
# characters of real abstract, and the abstract is what the gate and the ranker read. One
# measured example lost 240 characters, the whole results sentence, between `p ` and
# `Availability`.
#
# The ten spans it saves are all of this shape: `p < 0.001`, `<<1% have tri-kinetochores`,
# `<Choloepus didactylus>`. Requiring a tag NAME — a letter first, then word characters, no
# spaces — rejects every one while still matching all 1,246 real tags in the sample
# (`h4`, `i`, `sup`, `sub`).
_TAG_RE = re.compile(r"</?[a-zA-Z][\w:-]*\s*/?>")
_WS_RE = re.compile(r"\s+")


def _strip_markup(text: str) -> str:
    """Remove the inline markup Europe PMC embeds in titles and abstracts.

    Whitespace is collapsed afterwards because the markup is padded: the raw title
    ``Mapping  <i>trans</i>  -eQTLs`` leaves ``Mapping  trans  -eQTLs`` behind, and 91 of 400
    sampled titles had a double space after tag removal alone.
    """
    return _WS_RE.sub(" ", _TAG_RE.sub("", text)).strip()


def _request_json(url: str, max_retries: int = 4, base_delay: float = 2.0) -> Any:
    """GET a JSON endpoint, or raise :class:`EuropePMCError`.

    Raises rather than returning ``None`` on purpose. Every other adapter in this package
    returns an empty list when the API refuses, which makes a refusal indistinguishable from
    an honest zero — the defect that made a first DBLP measurement read "0 vs 0" after 12 of
    18 requests were rate-limited. The caller here turns a refusal into a reported failure.
    """
    last: Exception | None = None
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(url, headers={"Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json_mod.loads(resp.read())
        except urllib.error.HTTPError as exc:
            last = exc
            if exc.code == 429 or exc.code >= 500:
                delay = base_delay * (2**attempt)
                logger.warning(
                    "Europe PMC error %d (attempt %d/%d). Retrying in %.1fs...",
                    exc.code,
                    attempt + 1,
                    max_retries,
                    delay,
                )
                time.sleep(delay)
                continue
            raise EuropePMCError(f"Europe PMC error {exc.code}: {exc}") from exc
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
            last = exc
            if attempt < max_retries - 1:
                time.sleep(base_delay * (2**attempt))
    raise EuropePMCError(f"Europe PMC failed after {max_retries} attempts: {last}")


def build_query(keywords: str, lookback_days: int) -> str:
    """The Europe PMC query for one plain keyword string.

    *keywords* is left unquoted — see the module docstring; this is the difference between
    2,239 hits and none.
    """
    query = f"({keywords.strip()}) AND {PREPRINT_FILTER}"
    if 0 < lookback_days <= _UNBOUNDED_DAYS:
        end = datetime.now(UTC).date()
        start = end - timedelta(days=lookback_days)
        query += f" AND FIRST_PDATE:[{start.isoformat()} TO {end.isoformat()}]"
    return query


def _normalize(result: dict[str, Any]) -> dict[str, Any] | None:
    """Convert one Europe PMC record to the internal paper dict, or None if unusable."""
    doi = (result.get("doi") or "").strip()
    title = _strip_markup(result.get("title") or "")
    if not doi or not title:
        return None

    identifier = doi_key(doi)
    if not identifier:
        return None

    date = (result.get("firstPublicationDate") or "").strip()
    published = f"{date}T00:00:00+00:00" if date else datetime.now(UTC).isoformat()

    authors = [a.strip() for a in (result.get("authorString") or "").split(",") if a.strip()]

    return {
        "arxiv_id": identifier,
        "title": title,
        "authors": authors,
        "abstract": _strip_markup(result.get("abstractText") or ""),
        # Europe PMC returns no subject classification for preprints — `publisher` and
        # `pubType` are both null under resultType=core. Left empty deliberately rather than
        # filled with something from another taxonomy, which is the F4 defect: an empty list
        # takes `ranking.absent_category`, which is the policy for exactly this case.
        "categories": [],
        "published": published,
        "updated": None,
        # doi.org resolves both prefixes and both servers. Building
        # `biorxiv.org/content/<doi>` the way `sources/biorxiv.py` does assumes a server this
        # response cannot tell us, and assumes a path shape that 10.64898 dois do not use.
        "url": f"https://doi.org/{doi}",
        "pdf_url": None,
    }


def search_papers(
    keywords: str,
    lookback_days: int = 14,
    page_size: int = PAGE_SIZE,
    email: str | None = None,
) -> list[dict[str, Any]]:
    """One search. Raises :class:`EuropePMCError` if the API refused."""
    params = {
        "query": build_query(keywords, lookback_days),
        "format": "json",
        # `core` is required for `abstractText`; `lite` omits it, and an abstract-less paper
        # cannot be gated, ranked on keywords, or embedded.
        "resultType": "core",
        "pageSize": str(max(1, min(page_size, PAGE_SIZE))),
    }
    if email:
        params["email"] = email

    data = _request_json(f"{EPMC_SEARCH}?{urllib.parse.urlencode(params)}")
    results = (data.get("resultList") or {}).get("result") or []
    out: list[dict[str, Any]] = []
    for result in results:
        normalized = _normalize(result)
        if normalized:
            out.append(normalized)
    return out


def collect_papers(
    queries: list[str],
    lookback_days: int = 14,
    email: str | None = None,
) -> list[dict[str, Any]]:
    """Search Europe PMC for each query and merge, deduplicated by id.

    A query that the API refuses is logged and skipped; if **every** query was refused and
    none succeeded, that is a failure rather than an empty result and it is raised, so the
    caller cannot record "bioRxiv contributed nothing" about a conversation that never
    happened.
    """
    seen: dict[str, dict[str, Any]] = {}
    refused = 0
    attempted = 0
    for i, keywords in enumerate(queries):
        if not keywords.strip():
            continue
        attempted += 1
        if i:
            time.sleep(REQUEST_INTERVAL_S)
        try:
            papers = search_papers(keywords, lookback_days=lookback_days, email=email)
        except EuropePMCError as exc:
            refused += 1
            logger.warning("Europe PMC refused a query: %s", exc)
            continue
        for paper in papers:
            seen.setdefault(paper["arxiv_id"], paper)

    if attempted and refused == attempted:
        raise EuropePMCError(f"Europe PMC refused all {attempted} queries; this is not a zero")
    return list(seen.values())
