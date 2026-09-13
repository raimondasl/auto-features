"""Which retrieval channel contributed a paper, read back from what the pipeline stored.

Every channel adds only papers that no earlier channel found, so the answer is "the channel
that first contributed it". For dense discovery that is the fact worth having: a paper marked
as found by HyDE is one keyword search did not have. Whether a digest's picks came from there
is the difference between "dense discovery found this" and "keyword search would have
anyway" -- and until this was surfaced, answering it meant querying SQLite by hand.

The pipeline records the channel in `matched_query`, a column that predates this module and
doubles as the arXiv query that matched. What each value means:

  "hyde"            collector.collect_by_ids -- dense discovery
  "recommendation"  sources.s2_recommendations -- learned from the user's ratings
  "source:<key>"    pipeline._merge_source, <key> one of pipeline.KEYWORD_SOURCES
  any other text    the arXiv keyword query that matched
  empty or None     not recorded: a non-arXiv source collected before sources were stamped

The prefix cannot collide with an arXiv query, whose field names are `ti:`, `abs:`, `all:`,
`cat:` and the like -- `source:` is not one of them.

One reader, used by every surface that reports it -- `get_ranked_papers`, `explain_relevance`
and `rr why` -- because this project has paid more than once for a rule written out twice.
"""

from __future__ import annotations

HYDE = "hyde"
RECOMMENDATION = "recommendation"
SOURCE_PREFIX = "source:"

# Keyed like `sources:` in the configuration and `pipeline.KEYWORD_SOURCES`. A test holds the
# two in step, so a new source cannot ship reporting itself as "unrecorded".
SOURCE_LABELS = {
    "semantic_scholar": "Semantic Scholar",
    "openalex": "OpenAlex",
    "biorxiv": "bioRxiv",
    "europepmc": "Europe PMC",
    "iacr": "IACR ePrint",
    "dblp": "DBLP",
}


def source_marker(key: str) -> str:
    """What `_merge_source` stores for a paper that source contributed."""
    return f"{SOURCE_PREFIX}{key}"


def found_by(matched_query: str | None) -> str:
    """A stable key for the channel that contributed the paper -- for machines and agents.

    `dense_discovery`, `arxiv_keywords`, `s2_recommendations`, a source key such as
    `openalex`, or `unrecorded`.
    """
    if not matched_query:
        return "unrecorded"
    if matched_query == HYDE:
        return "dense_discovery"
    if matched_query == RECOMMENDATION:
        return "s2_recommendations"
    if matched_query.startswith(SOURCE_PREFIX):
        return matched_query[len(SOURCE_PREFIX) :] or "unrecorded"
    return "arxiv_keywords"


def describe(matched_query: str | None) -> str:
    """The same answer for a person to read, as `rr why` prints it."""
    key = found_by(matched_query)
    if key == "dense_discovery":
        return "dense discovery (HyDE) -- keyword search did not find it"
    if key == "arxiv_keywords":
        return f"arXiv keyword search ({matched_query})"
    if key == "s2_recommendations":
        return "Semantic Scholar recommendations, learned from your ratings"
    if key == "unrecorded":
        return "a source that was not recorded (collected before sources were stamped)"
    return SOURCE_LABELS.get(key, key)
