"""Citation proximity — "this new paper extends work you starred" (Feature 8).

The dormant ``paper_stars`` (and high ratings) become a seed set; a newly-fetched
paper that *references* one of those seeds is surfaced and boosted. Pure logic here
(no network): the reference lists are fetched by :func:`reporadar.citations.fetch_references`
and the digest reads the stored edges back.
"""

from __future__ import annotations

from reporadar.paper_id import dedup_id
from reporadar.store import PaperStore


def base_id(arxiv_id: str) -> str:
    """Strip the version suffix (``2401.00001v2`` -> ``2401.00001``).

    Delegates to :func:`reporadar.paper_id.dedup_id`. The name stays because it reads well
    where it is used and callers import it; what does not stay is a second *implementation*
    of the rule — this one used to truncate at the first lowercase ``v`` anywhere.
    """
    return dedup_id(arxiv_id)


def build_seed_set(store: PaperStore, min_rating: int = 4) -> set[str]:
    """Version-normalized arXiv ids the user starred or rated ``>= min_rating``."""
    seeds: set[str] = set(store.get_starred_papers())
    seeds |= {aid for aid, rating in store.get_all_ratings().items() if rating >= min_rating}
    return {base_id(a) for a in seeds}


def find_citation_links(
    references_by_paper: dict[str, list[str]],
    seeds: set[str],
) -> dict[str, list[str]]:
    """Map each citing paper to the seed ids it references.

    ``references_by_paper`` is ``{citing_id: [cited_id, ...]}``; *seeds* are
    version-normalized. Returns ``{citing_id: [seed_id, ...]}`` for papers that
    reference at least one seed (ids matched version-insensitively, sorted).
    """
    if not seeds:
        return {}
    links: dict[str, list[str]] = {}
    for citing_id, cited_ids in references_by_paper.items():
        hits = sorted({base_id(c) for c in cited_ids} & seeds)
        if hits:
            links[citing_id] = hits
    return links
