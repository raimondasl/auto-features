"""Heuristic scoring of papers against a repo profile."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import Any

from reporadar.config import KNOWN_ARXIV_PREFIXES, QueriesConfig, RankingConfig
from reporadar.profiler import RepoProfile


def has_comparable_categories(paper: dict[str, Any]) -> bool:
    """True when this paper's categories are written in the vocabulary of the target list.

    `ranking.absent_category` decides how the category axis treats a paper carrying **no**
    category signal, and the test for that was `paper.get("categories")` — truthiness. Three
    of the six non-arXiv adapters populate the field from a *different* taxonomy: OpenAlex
    stores its `primary_topic` display name (`Machine Learning`), bioRxiv its subject area
    (`Bioinformatics `, trailing space included) and DBLP the venue key. Those strings cannot
    intersect a list of arXiv categories under any repository's config, so the match is a
    guaranteed 0 that is then averaged in at full `w_category`.

    The consequence is that a paper is ranked by which source happened to find it. At
    keyword 0.6 and the shipped weights: an arXiv paper matching its category scores 0.733,
    the same paper from Semantic Scholar — which leaves `categories` empty and so takes the
    `absent_category` path — scores 0.600, and from bioRxiv or OpenAlex **0.400**. The 0.33
    gap is not a judgement about the paper; it is the taxonomy the adapter happened to read.

    So the question the ranker asks is not "does this paper have categories" but "are its
    categories in the vocabulary the target list is written in". A paper that fails this
    carries no comparable signal and belongs on the `absent_category` path with the rest of
    the non-arXiv pool.

    Deliberately NOT done by blanking `categories` in the three adapters, which is the other
    obvious repair: the field is exported to the digest CSV, so blanking it destroys
    information the user can see to fix a ranking bug they cannot; and it puts one invariant
    in three places plus every adapter written later — the exact shape of the C-12/C-14
    defects that `paper_id.dedup_id` exists to prevent. Europe PMC (B1) is the next adapter,
    and it should be impossible for it to get this wrong.
    """
    for value in paper.get("categories") or ():
        # `hep-th` has no dot and is its own prefix; `cond-mat.mtrl-sci` and `q-bio.QM` are
        # split on the first one. A venue key (`conf/vldb`) and a topic name survive neither.
        if value.strip().split(".", 1)[0] in KNOWN_ARXIV_PREFIXES:
            return True
    return False


def _tokenize(text: str) -> set[str]:
    """Lowercase tokenization of text into word-level tokens."""
    return set(re.findall(r"[a-z][a-z0-9_-]+", text.lower()))


def score_keyword_overlap(
    paper: dict[str, Any],
    profile: RepoProfile,
) -> float:
    """Score based on keyword overlap between paper title+abstract and profile.

    Returns a score in [0, 1]. Each matching profile keyword contributes its
    TF-IDF weight. The result is normalized by the sum of all profile weights.
    """
    if not profile.keywords:
        return 0.0

    paper_tokens = _tokenize(paper["title"] + " " + paper["abstract"])

    matched_weight = 0.0
    total_weight = 0.0
    for term, weight in profile.keywords:
        total_weight += weight
        # Check if any token in the term matches paper tokens
        term_tokens = _tokenize(term)
        if term_tokens & paper_tokens:
            matched_weight += weight

    if total_weight == 0:
        return 0.0

    return min(matched_weight / total_weight, 1.0)


def score_category_match(
    paper: dict[str, Any],
    target_categories: list[str],
    category_weights: dict[str, float] | None = None,
) -> float:
    """Score based on how many of the paper's categories match the target list.

    When *category_weights* is provided and non-empty, each matching category
    contributes its weight (unmentioned categories default to 1.0). The result
    is normalized by the sum of all target category weights.

    Returns a score in [0, 1].
    """
    if not target_categories or not paper.get("categories"):
        return 0.0

    # Stripped: bioRxiv returns `'Bioinformatics '` with a trailing space, and a taxonomy
    # that did pass :func:`has_comparable_categories` should not then miss on whitespace.
    paper_cats = {c.strip() for c in paper["categories"]}

    if category_weights:
        matched_weight = 0.0
        total_weight = 0.0
        for cat in target_categories:
            w = category_weights.get(cat, 1.0)
            total_weight += w
            if cat in paper_cats:
                matched_weight += w
        if total_weight == 0:
            return 0.0
        return min(matched_weight / total_weight, 1.0)

    target_set = set(target_categories)
    overlap = len(target_set & paper_cats)
    return min(overlap / len(target_set), 1.0)


def score_recency(
    paper: dict[str, Any], lookback_days: int = 14, now: datetime | None = None
) -> float:
    """Score based on how recent the paper is.

    Returns 1.0 for papers published today, decaying linearly to 0.0
    at *lookback_days* ago. Papers older than the lookback window get 0.0.

    *now* overrides the reference instant. ``rr eval`` re-scores papers collected
    weeks or months ago, and against today's clock every one of them would score 0
    recency — making the component invisible and the scores unlike anything the user
    actually saw. Passing the paper's first-seen time reproduces the score the ranker
    would have produced at digest time.
    """
    try:
        published = datetime.fromisoformat(paper["published"])
    except (ValueError, KeyError):
        return 0.0

    if published.tzinfo is None:
        published = published.replace(tzinfo=UTC)

    reference = now or datetime.now(UTC)
    if reference.tzinfo is None:
        reference = reference.replace(tzinfo=UTC)
    age_days = (reference - published).total_seconds() / 86400

    if age_days < 0:
        return 1.0
    if age_days >= lookback_days:
        return 0.0

    return 1.0 - (age_days / lookback_days)


def compute_exclude_penalty(
    paper: dict[str, Any],
    exclude_terms: list[str],
) -> float:
    """Compute a penalty multiplier for papers matching exclude terms.

    Returns a value in (0, 1]. Each matched exclude term multiplies the
    score by 0.5, so papers matching many exclude terms get heavily penalized.
    """
    if not exclude_terms:
        return 1.0

    paper_tokens = _tokenize(paper["title"] + " " + paper["abstract"])
    penalty = 1.0
    for term in exclude_terms:
        term_tokens = _tokenize(term)
        if term_tokens & paper_tokens:
            penalty *= 0.5

    return penalty


def score_paper(
    paper: dict[str, Any],
    profile: RepoProfile,
    ranking_cfg: RankingConfig,
    queries_cfg: QueriesConfig,
    arxiv_categories: list[str],
    lookback_days: int = 14,
    embedding_score: float | None = None,
    citation_score: float | None = None,
    citation_proximity_score: float | None = None,
    specter_score: float | None = None,
    community_score: float | None = None,
    attention_score: float | None = None,
    withdrawn: bool = False,
    now: datetime | None = None,
    absent_category_score: float | None = None,
) -> dict[str, Any]:
    """Compute a combined score for a single paper.

    Returns a score dict suitable for PaperStore.save_scores().

    *absent_category_score* is only read under ``ranking.absent_category: impute``: it is a
    property of the pool, not of this paper, so :func:`rank_papers` computes it once and
    passes it in rather than each paper guessing at it.
    """
    kw = score_keyword_overlap(paper, profile)
    cat = score_category_match(
        paper, arxiv_categories, category_weights=ranking_cfg.category_weights or None
    )
    rec = score_recency(paper, lookback_days, now=now)

    raw_total = ranking_cfg.w_keyword * kw + ranking_cfg.w_recency * rec
    weight_sum = ranking_cfg.w_keyword + ranking_cfg.w_recency

    # How the category component treats a paper carrying no COMPARABLE category signal —
    # every paper from every non-arXiv source, whether its adapter leaves the field empty
    # or fills it from another taxonomy (see :func:`has_comparable_categories`; testing
    # truthiness here is what split those two groups by 0.2 for no reason).
    #
    # `omit` (the shipped default) drops the component, which was meant to avoid
    # handicapping those papers and instead *advantages* them: an arXiv paper is averaged
    # over keyword AND category while an uncategorised one is averaged over keyword alone,
    # so at equal keyword relevance the uncategorised paper scores higher (0.600 vs 0.567
    # at the shipped weights; 0.600 vs 0.400 when the arXiv paper matches no target
    # category). The absent-signal rule is correct when missingness is random — a paper
    # SPECTER2 has never seen — and wrong here, because having categories is perfectly
    # correlated with being an arXiv paper.
    #
    # `zero` scores the absence as a real 0. `impute` scores it at *absent_category_score*,
    # the mean category score of the categorised papers in the same pool, so an
    # uncategorised paper is treated as an average paper on this axis rather than as the
    # best or the worst.
    if has_comparable_categories(paper):
        raw_total += ranking_cfg.w_category * cat
        weight_sum += ranking_cfg.w_category
    elif getattr(ranking_cfg, "absent_category", "omit") == "zero":
        weight_sum += ranking_cfg.w_category
    elif getattr(ranking_cfg, "absent_category", "omit") == "impute":
        raw_total += ranking_cfg.w_category * (absent_category_score or 0.0)
        weight_sum += ranking_cfg.w_category

    w_embedding = getattr(ranking_cfg, "w_embedding", 0.0)
    if embedding_score is not None and w_embedding > 0:
        raw_total += w_embedding * embedding_score
        weight_sum += w_embedding

    w_citations = getattr(ranking_cfg, "w_citations", 0.0)
    if citation_score is not None and w_citations > 0:
        raw_total += w_citations * citation_score
        weight_sum += w_citations

    w_prox = getattr(ranking_cfg, "w_citation_proximity", 0.0)
    if citation_proximity_score is not None and w_prox > 0:
        raw_total += w_prox * citation_proximity_score
        weight_sum += w_prox

    w_specter = getattr(ranking_cfg, "w_specter", 0.0)
    if specter_score is not None and w_specter > 0:
        raw_total += w_specter * specter_score
        weight_sum += w_specter

    w_community = getattr(ranking_cfg, "w_community", 0.0)
    if community_score is not None and w_community > 0:
        raw_total += w_community * community_score
        weight_sum += w_community

    w_attention = getattr(ranking_cfg, "w_attention", 0.0)
    if attention_score is not None and w_attention > 0:
        raw_total += w_attention * attention_score
        weight_sum += w_attention

    normalized = raw_total / weight_sum if weight_sum > 0 else 0.0

    penalty = compute_exclude_penalty(paper, queries_cfg.exclude)
    # A withdrawn paper is penalized *multiplicatively*, never as a weighted
    # component: as one more signal it would be outvoted by keyword + category +
    # recency all firing, and a strongly-relevant withdrawn paper would still reach
    # Top Picks. The factor stays above 0 so the paper sinks to Muted while remaining
    # visible with its flag — a reader who already saw the preprint elsewhere is
    # better served by "this was withdrawn" than by its silent disappearance.
    if withdrawn:
        penalty *= getattr(ranking_cfg, "withdrawn_penalty", 0.1)
    total = normalized * penalty

    return {
        "arxiv_id": paper["arxiv_id"],
        "score_total": round(total, 4),
        "keyword_score": round(kw, 4),
        "category_score": round(cat, 4),
        "recency_score": round(rec, 4),
        "embedding_score": round(embedding_score, 4) if embedding_score is not None else None,
        "citation_score": round(citation_score, 4) if citation_score is not None else None,
        "specter_score": round(specter_score, 4) if specter_score is not None else None,
        "community_score": round(community_score, 4) if community_score is not None else None,
        "attention_score": round(attention_score, 4) if attention_score is not None else None,
        "matched_query": paper.get("matched_query"),
    }


def format_score_explanation(score_dict: dict[str, Any], ranking_cfg: RankingConfig) -> str:
    """Return a multi-line string showing weight * component = contribution for each component."""
    lines = [f"  Paper: {score_dict['arxiv_id']}"]
    components = [
        ("keyword", "keyword_score", ranking_cfg.w_keyword),
        ("category", "category_score", ranking_cfg.w_category),
        ("recency", "recency_score", ranking_cfg.w_recency),
    ]
    if "embedding_score" in score_dict and score_dict["embedding_score"] is not None:
        components.append(("embedding", "embedding_score", getattr(ranking_cfg, "w_embedding", 0)))
    if "citation_score" in score_dict and score_dict["citation_score"] is not None:
        components.append(("citation", "citation_score", getattr(ranking_cfg, "w_citations", 0)))
    if "specter_score" in score_dict and score_dict["specter_score"] is not None:
        components.append(("specter", "specter_score", getattr(ranking_cfg, "w_specter", 0)))
    if "community_score" in score_dict and score_dict["community_score"] is not None:
        components.append(("community", "community_score", getattr(ranking_cfg, "w_community", 0)))
    if "attention_score" in score_dict and score_dict["attention_score"] is not None:
        components.append(("attention", "attention_score", getattr(ranking_cfg, "w_attention", 0)))

    for name, key, weight in components:
        val = score_dict.get(key, 0) or 0
        contribution = weight * val
        lines.append(f"    {name:12s}: {weight:.2f} * {val:.4f} = {contribution:.4f}")

    lines.append(f"    {'total':12s}: {score_dict['score_total']:.4f}")
    return "\n".join(lines)


def score_distribution(scores: list[dict[str, Any]]) -> dict[str, float]:
    """Compute distribution stats from a list of score dicts.

    Returns {"mean": ..., "median": ..., "min": ..., "max": ..., "count": ...}.
    """
    if not scores:
        return {"mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0, "count": 0}

    totals = sorted(s["score_total"] for s in scores)
    n = len(totals)
    mean = sum(totals) / n
    median = totals[n // 2] if n % 2 == 1 else (totals[n // 2 - 1] + totals[n // 2]) / 2

    return {
        "mean": round(mean, 4),
        "median": round(median, 4),
        "min": round(totals[0], 4),
        "max": round(totals[-1], 4),
        "count": n,
    }


def rank_papers(
    papers: list[dict[str, Any]],
    profile: RepoProfile,
    ranking_cfg: RankingConfig,
    queries_cfg: QueriesConfig,
    arxiv_categories: list[str],
    lookback_days: int = 14,
    repo_embedding: Any = None,
    citation_scores: dict[str, float] | None = None,
    paper_embeddings: dict[str, Any] | None = None,
    citation_proximity: dict[str, float] | None = None,
    specter: dict[str, float] | None = None,
    community: dict[str, float] | None = None,
    attention: dict[str, float] | None = None,
    withdrawn: set[str] | None = None,
    now_by_id: dict[str, datetime] | None = None,
) -> list[dict[str, Any]]:
    """Score and rank a list of papers. Returns score dicts sorted by score descending.

    *paper_embeddings*, when given, supplies each paper's vector (keyed by arxiv_id)
    from the persistent cache, so vectors are not re-encoded on every run.
    *citation_proximity* rewards papers that cite a starred/highly-rated paper.
    """
    # Under `impute`, an uncategorised paper is scored on this axis as an average
    # categorised paper in the same pool. Computed once over the pool because it is a
    # property of the pool; falls back to 0.0 (i.e. `zero`) when nothing is categorised,
    # since a mean over an empty set is not a number and imputing 1.0 there would hand
    # every non-arXiv paper a perfect category score.
    absent_cat = 0.0
    if getattr(ranking_cfg, "absent_category", "omit") == "impute":
        cats = [
            score_category_match(
                p, arxiv_categories, category_weights=ranking_cfg.category_weights or None
            )
            for p in papers
            if has_comparable_categories(p)
        ]
        absent_cat = sum(cats) / len(cats) if cats else 0.0
    scores = []
    for paper in papers:
        emb_score = None
        if repo_embedding is not None:
            try:
                from reporadar.embeddings import cosine_similarity

                if paper_embeddings is not None:
                    paper_emb = paper_embeddings.get(paper["arxiv_id"])
                else:
                    from reporadar.embeddings import compute_paper_embedding

                    paper_emb = compute_paper_embedding(paper)
                if paper_emb is not None:
                    emb_score = max(0.0, cosine_similarity(repo_embedding, paper_emb))
            except (RuntimeError, ImportError):
                pass

        cit_score = None
        if citation_scores is not None:
            cit_score = citation_scores.get(paper["arxiv_id"])

        prox_score = None
        if citation_proximity is not None:
            prox_score = citation_proximity.get(paper["arxiv_id"])

        spec_score = None
        if specter is not None:
            spec_score = specter.get(paper["arxiv_id"])

        comm_score = None
        if community is not None:
            comm_score = community.get(paper["arxiv_id"])

        att_score = None
        if attention is not None:
            att_score = attention.get(paper["arxiv_id"])

        scores.append(
            score_paper(
                paper,
                profile,
                ranking_cfg,
                queries_cfg,
                arxiv_categories,
                lookback_days,
                embedding_score=emb_score,
                citation_score=cit_score,
                citation_proximity_score=prox_score,
                specter_score=spec_score,
                community_score=comm_score,
                attention_score=att_score,
                withdrawn=bool(withdrawn and paper["arxiv_id"] in withdrawn),
                now=(now_by_id or {}).get(paper["arxiv_id"]),
                absent_category_score=absent_cat,
            )
        )
    # Tie-break on arxiv_id so the order never depends on the order papers were
    # fetched in. Scores are rounded to 4dp and ties are common (many papers share
    # a keyword/category profile), so a stable sort would otherwise silently
    # inherit the input order — which in the eval fixtures means "gold first",
    # flattering every measurement.
    scores.sort(key=lambda s: (-s["score_total"], s["arxiv_id"]))
    return scores
