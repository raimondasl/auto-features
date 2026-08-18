"""Digest generation — render scored papers into a Markdown file."""

from __future__ import annotations

import csv
import io
import json as json_mod
from collections.abc import Collection
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from jinja2 import Environment, PackageLoader

from reporadar.notify import DigestSummary
from reporadar.paper_id import dedup_id
from reporadar.store import PaperStore
from reporadar.suggestions import enrich_papers_with_suggestions

TEMPLATES_DIR = Path(__file__).parent / "templates"

# Score thresholds for categorizing papers into tiers.
# Papers with score >= TOP_THRESHOLD go into "Top Picks",
# >= MAYBE_THRESHOLD into "Maybe Relevant", rest into "Muted".
TOP_THRESHOLD = 0.5
MAYBE_THRESHOLD = 0.2


def _load_template(template_name: str = "digest.md.j2") -> Any:
    """Load a Jinja2 template from the templates directory (no autoescaping).

    Used for the text formats (Markdown, RSS) and the legacy markdown-in-<pre>
    HTML wrapper, where autoescaping would corrupt the intended output.
    """
    env = Environment(
        loader=PackageLoader("reporadar", "templates"),
        keep_trailing_newline=True,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    return env.get_template(template_name)


def _safe_url(url: Any) -> str:
    """Return *url* only if it is an ``http(s)`` link, else ``#``.

    Paper URLs come from external source APIs; autoescaping neutralizes markup
    but not a ``javascript:``/``data:`` scheme (no ``<>&"`` to escape). Since the
    digest is published to GitHub Pages, allow-list the scheme so a hostile URL
    can't become an executable link.
    """
    s = str(url or "")
    return s if s.startswith(("http://", "https://")) else "#"


def _load_html_template(template_name: str) -> Any:
    """Load an HTML template with autoescaping on.

    The rendered-HTML digest and the Pages archive index interpolate live paper
    text (titles, abstracts, author names) into markup, so autoescaping is
    required to stay correct and safe on ``<``/``&`` in that content.
    """
    env = Environment(
        loader=PackageLoader("reporadar", "templates"),
        keep_trailing_newline=True,
        trim_blocks=True,
        lstrip_blocks=True,
        autoescape=True,
    )
    env.filters["safe_url"] = _safe_url
    return env.get_template(template_name)


def filter_since(
    scored_papers: list[dict[str, Any]], since_days: int | None
) -> list[dict[str, Any]]:
    """Keep only papers published within the last *since_days* days.

    A falsy or non-positive *since_days* means no filtering (returns the input
    unchanged). Papers with a missing/unparseable ``published`` date are kept.
    """
    if not since_days or since_days <= 0:
        return scored_papers
    cutoff = (datetime.now(UTC) - timedelta(days=since_days)).strftime("%Y-%m-%d")
    kept: list[dict[str, Any]] = []
    for p in scored_papers:
        pub = (p.get("published") or "")[:10]
        # Keep papers with an unknown date rather than silently hiding them.
        if not pub or pub >= cutoff:
            kept.append(p)
    return kept


def mark_new_papers(
    store: PaperStore, scored: list[dict[str, Any]], run_id: int
) -> list[dict[str, Any]]:
    """Set ``is_new`` on each paper: was it absent from the previous run?

    One implementation for the two renderers that need it. It was two, and they answered
    "is this the same paper" on the *raw* id — so a paper arXiv returned as ``v1`` last
    week and ``v2`` this week read as brand new in the diff. ``dedup_id`` is the same
    normaliser every merge uses; the alternative is a third rule for one question.
    """
    prev_id = store.get_previous_run_id(run_id)
    prev = store.get_scored_paper_ids_for_run(prev_id) if prev_id is not None else set()
    prev_ids = {dedup_id(pid) for pid in prev}
    for paper in scored:
        paper["is_new"] = dedup_id(paper["arxiv_id"]) not in prev_ids
    return scored


def digest_window(
    scored_papers: list[dict[str, Any]],
    top_n: int = 15,
    *,
    triage_threshold: int | None = None,
    rerank: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """The papers a digest can actually show, and the withdrawn ones it removed first.

    Returns ``(window, withdrawn)``. *window* is what reaches a tier; anything outside it
    is dropped before tiering and can never be displayed, whatever its scores say.

    Two rules, in this order, and the order is the whole reason this is one function:

    1. **Rerank before the cut.** With triage scores present, `llm_score` decides the
       ordering (`score_total` breaks ties), so a paper the heuristic ranker buried can
       still enter the window.
    2. **Withdrawn papers leave before the cut, not after.** They are muted regardless, and
       taking them out first means the slot they would have occupied goes to the next
       paper rather than being wasted — so the identity of the *last* paper in the window
       depends on how many withdrawn papers preceded it.

    3. **Only withdrawn papers that were competing for a slot are returned.** The walk stops
       once the window is full, so a retraction sitting at rank 300 is not reported. It used
       to return every withdrawn paper in the run, which is how three off-topic retractions
       -- hydraulic fracturing, opioid use disorders, music sight reading, all fetched by the
       generic queries a thin profile produces -- came to head a materials-science digest
       whose Top Picks were sound. What the section is for is the paper you might otherwise
       have acted on; a paper the digest was never going to show is not that.

    That second rule is why `cli.update` must not re-derive this to decide which papers to
    spend fine-scale calls on. A local reimplementation that forgot it would scope the band
    to a window off by one paper per withdrawal, and the two would disagree only on runs
    where a retraction happened to land in the top slots — the rarest and least testable
    case. One invariant, two implementations, is the shape behind C-9, C-12 and C-14; this
    is the same invariant given one home before it becomes the fourth.
    """
    if rerank and triage_threshold is not None:
        from reporadar.triage import rerank_by_actionability

        scored_papers = rerank_by_actionability(scored_papers)

    window: list[dict[str, Any]] = []
    withdrawn: list[dict[str, Any]] = []
    for paper in scored_papers:
        if len(window) >= top_n:
            break
        if paper.get("withdrawn_in"):
            withdrawn.append(paper)
        else:
            window.append(paper)
    return window, withdrawn


def cited_ids_from(profile: Any | None) -> frozenset[str]:
    """The repository's own citations, off a profile the caller already has.

    Tolerates ``None`` and any object without the attribute, because five call sites pass a
    profile that is optional (`rr notify` and `rr watch` build one only when they can) and a
    digest that silently changed shape depending on which of them ran would be worse than one
    that never excludes anything.
    """
    return frozenset(getattr(profile, "cited_arxiv_ids", None) or ())


def categorize_papers(
    scored_papers: list[dict[str, Any]],
    top_n: int = 15,
    top_threshold: float = TOP_THRESHOLD,
    maybe_threshold: float = MAYBE_THRESHOLD,
    triage_threshold: int | None = None,
    rerank: bool = False,
    finescale_threshold: float | None = None,
    cited_ids: Collection[str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Split scored papers into three tiers: top_picks, maybe_relevant, muted.

    Papers are assumed to be sorted by score descending. Only the first
    *top_n* papers are included across all tiers.

    When *triage_threshold* is set, papers that carry an LLM actionability score
    (``llm_score``) are tiered by it — ``>= triage_threshold`` = Top Pick, ``>= 1``
    = Maybe, ``0`` = Muted — so the digest abstains from "Top Picks" unless a
    paper is judged genuinely applicable. A paper the gate did **not** score reaches
    Maybe at best: it is unproven, not endorsed. That rule used to be a fall-back to the
    heuristic ``score_total`` thresholds, which meant an ungated paper could enter Top
    Picks on the 0.5 threshold — the very threshold Feature 6 replaced after it measured
    net@2 −11. It bites whenever ``output.top_n`` exceeds ``triage.top_k`` (papers past
    the gate's window fill the digest ungated) or a gate call fails. The benchmark has
    always scored the stricter rule, and `evals/audit_product_divergence.py` is what
    found the two disagreeing.

    When *triage_threshold* is ``None`` the gate did not run at all, and the heuristic
    thresholds are the only rule there is — unchanged.

    When *finescale_threshold* is set, papers sitting *exactly at* the triage
    threshold face a second bar: their calibrated ``finescale_p`` must reach it too, or
    they drop to Maybe. Only that band is re-gated, because only that band needed it —
    the gate's score-2 papers ran anywhere from 0% to 100% genuinely actionable while
    its score-3 papers were reliable (evals/RESULTS.md, "Ranking the score-2 band").
    A band paper with **no** ``finescale_p`` does not make Top Picks: an unscored paper
    is unproven, and the benchmark's +2.91 measured it that way. The caller is
    responsible for not passing a threshold at all when the stage broadly failed — see
    :func:`reporadar.finescale.enough_scored`.

    When *rerank* is set (and triage scores are present), papers are reordered by
    ``llm_score`` *before* the *top_n* cut, so an actionable paper the heuristic
    ranker buried below the window isn't dropped before it can be a Top Pick.

    *cited_ids* are arXiv ids the repository's own text already cites
    (:attr:`~reporadar.profiler.RepoProfile.cited_arxiv_ids`). Those papers are marked
    ``already_cited`` and muted instead of tiered: the project has already found them, and
    on five of six scientific repositories measured this way the paper the gate ranked
    first was the repository's **own** publication, which a neutral judge scored 1 -- "the
    method this repository already implements" -- every time. They are muted *inside* the
    window rather than removed before it, so excluding one cannot pull an unseen paper into
    the digest behind it.
    """
    # Ordering and the top_n cut live in `digest_window`, shared with `cli.update` so the
    # fine-scale stage spends only on papers this function can still show. Withdrawn papers
    # are separated out *before* the cut: the triage branch below tiers by llm_score and
    # never reads score_total, so the ranking penalty alone would not stop a withdrawn
    # paper reaching Top Picks, and letting one occupy a slot would evict a good paper.
    limited, withdrawn = digest_window(
        scored_papers, top_n, triage_threshold=triage_threshold, rerank=rerank
    )

    top_picks = []
    maybe_relevant = []
    muted = list(withdrawn)

    for paper in limited:
        if cited_ids and dedup_id(str(paper.get("arxiv_id", ""))) in cited_ids:
            # The repository's own README, CITATION file or bibliography already points at
            # this paper. Recommending it back is not a recommendation. Flagged and muted
            # rather than dropped -- the same shape withdrawn papers use -- so the digest can
            # show it in its own section and a reader can check the call.
            paper["already_cited"] = True
            muted.append(paper)
            continue
        llm = paper.get("llm_score")
        if triage_threshold is not None and llm is None:
            # The gate ran and this paper has no verdict. Unproven is not a Top Pick.
            maybe_relevant.append(paper)
            continue
        if triage_threshold is not None and llm is not None:
            if llm >= triage_threshold:
                # Papers above the band are trusted on the gate's word; papers *at* it
                # must also clear the calibrated probability when that stage ran.
                at_band = llm == triage_threshold and finescale_threshold is not None
                p = paper.get("finescale_p")
                if at_band and (p is None or p < finescale_threshold):
                    maybe_relevant.append(paper)
                else:
                    top_picks.append(paper)
            elif llm >= 1:
                maybe_relevant.append(paper)
            else:
                muted.append(paper)
            continue
        score = paper["score_total"]
        if score >= top_threshold:
            top_picks.append(paper)
        elif score >= maybe_threshold:
            maybe_relevant.append(paper)
        else:
            muted.append(paper)

    return top_picks, maybe_relevant, muted


def annotate_signals(store: PaperStore, papers: list[dict[str, Any]]) -> None:
    """Attach the Hacker News signal to *papers* in place, for badging.

    ``withdrawn_in`` deliberately does **not** come from here — ``get_scores_for_run``
    joins it, so every consumer of a run's scores has it without remembering to
    annotate. Annotating in one caller is what left json/csv/rss, ``rr open`` and
    ``rr gh-issues`` emitting withdrawn papers as recommendations.

    A paper with no signal row simply gains no key, which renders as no badge — never
    as an "unpopular" claim.
    """
    try:
        signals = store.get_signals([p["arxiv_id"] for p in papers], "hn")
    except Exception:
        return
    for paper in papers:
        entry = signals.get(paper["arxiv_id"])
        if entry and entry["value"]:
            paper["hn_points"] = int(entry["value"] or 0)
            paper["hn_url"] = entry["detail"] or ""


def find_extends_starred(store: PaperStore, scored: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return the run's papers that cite a starred/highly-rated paper (Feature 8).

    *scored* is expected best-first; matching papers keep that order and are
    annotated in place with ``cites_starred`` so templates can badge them. The
    seed set is re-read from the store on every call, so unstarring a paper drops
    it from the section on the next digest.

    Shared by the digest context and the notification summary, so the count that
    gets pushed can't drift from the section that gets rendered.
    """
    extends: list[dict[str, Any]] = []
    try:
        from reporadar.citation_graph import build_seed_set

        seeds = build_seed_set(store)
        if not seeds:
            return []
        edges = store.get_citations_for([p["arxiv_id"] for p in scored])
        for paper in scored:
            cites = sorted(c for c in edges.get(paper["arxiv_id"], []) if c in seeds)
            if cites:
                paper["cites_starred"] = cites
                extends.append({"title": paper["title"], "url": paper["url"], "cites": cites})
    except Exception:
        return extends
    return extends


def _build_digest_context(
    store: PaperStore,
    run_id: int,
    top_n: int = 15,
    diff: bool = False,
    suggestions_config: Any | None = None,
    profile: Any | None = None,
    since_days: int | None = None,
    triage_threshold: int | None = None,
    rerank: bool = False,
    finescale_threshold: float | None = None,
) -> dict[str, Any]:
    """Build the shared template context for a run (Markdown and HTML render it)."""
    all_scored = store.get_scores_for_run(run_id)
    scored = filter_since(all_scored, since_days)
    # Resolve the run being rendered (not necessarily the latest — `--run-id N`),
    # so the header stats match the papers below them.
    run = store.get_run(run_id)

    # Diff mode: determine which papers are new vs. carried over
    diff_mode = False
    if diff:
        diff_mode = True
        mark_new_papers(store, scored, run_id)

    # Before tiering: categorize_papers needs the withdrawal flag, since the LLM
    # triage branch decides a paper's tier without consulting score_total.
    annotate_signals(store, scored)

    top_picks, maybe_relevant, muted = categorize_papers(
        scored,
        top_n=top_n,
        triage_threshold=triage_threshold,
        rerank=rerank,
        finescale_threshold=finescale_threshold,
        cited_ids=cited_ids_from(profile),
    )
    already_cited = [p for p in muted if p.get("already_cited")]
    muted = [p for p in muted if not p.get("already_cited")]
    enrich_papers_with_suggestions(top_picks, config=suggestions_config, profile=profile)

    has_embeddings = any(p.get("embedding_score") is not None for p in scored)
    has_citations = any(p.get("citation_score") is not None for p in scored)
    has_enrichments = any(p.get("has_code") or p.get("datasets") for p in scored)

    # Trend detection
    trends: list[dict[str, Any]] = []
    try:
        from reporadar.trends import detect_trends

        trends = detect_trends(store, run_id)
    except Exception:
        pass

    # Recommended papers. Prefer the learned S2 recommendations, but hold them to
    # a real relevance bar: the API is repo-agnostic and happily returns off-topic
    # papers, so anything the local ranker scored below the "maybe" tier is dropped
    # and the keyword-overlap recommender takes over instead. Drawn from the
    # unfiltered scores because recommendations are a user-seeded feed, not a
    # publication-window one (`--since` shouldn't silently empty the section).
    recommended: list[dict[str, Any]] = [
        p
        for p in all_scored
        if p.get("matched_query") == "recommendation"
        and (p.get("score_total") or 0.0) >= MAYBE_THRESHOLD
    ][:3]
    try:
        from reporadar.feedback import find_similar_to_highly_rated

        ratings = store.get_all_ratings()
        if not recommended and ratings:
            rated_papers = {}
            for arxiv_id, rating in ratings.items():
                p = store.get_paper(arxiv_id)
                if p:
                    rated_papers[arxiv_id] = {**p, "rating": rating}
            if rated_papers:
                recommended = find_similar_to_highly_rated(scored, rated_papers)
    except Exception:
        pass

    extends_starred = find_extends_starred(store, scored)

    # Withdrawn papers get their own section rather than relying on a per-card badge:
    # the ranking penalty deliberately demotes them, and the Muted tier renders no
    # badges, so a card-only warning would vanish exactly when it fires.
    #
    # Read off `muted`, which `categorize_papers` seeds from `digest_window`'s withdrawn
    # list, rather than re-scanning `scored`. Scanning `scored` was a SECOND implementation
    # of a rule `digest_window` already owns, and the two disagreed the moment that rule
    # gained its third clause: the window stopped reporting retractions it was never going
    # to show, and this comprehension went on rendering every retraction in the run.
    withdrawn_papers = [
        {"title": p["title"], "url": p["url"], "field": p["withdrawn_in"]}
        for p in muted
        if p.get("withdrawn_in")
    ]

    return {
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
        "run_id": run_id,
        "papers_new": run["papers_new"] if run else 0,
        "papers_seen": run["papers_seen"] if run else 0,
        "total_scored": len(scored),
        "queries_used": run["queries_used"] if run else [],
        "top_picks": top_picks,
        "maybe_relevant": maybe_relevant,
        "muted": muted,
        "diff_mode": diff_mode,
        "has_embeddings": has_embeddings,
        "has_citations": has_citations,
        "has_enrichments": has_enrichments,
        "trends": trends,
        "recommended": recommended,
        "extends_starred": extends_starred,
        "withdrawn_papers": withdrawn_papers,
        "already_cited": already_cited,
    }


def generate_digest(
    store: PaperStore,
    run_id: int,
    top_n: int = 15,
    diff: bool = False,
    suggestions_config: Any | None = None,
    profile: Any | None = None,
    since_days: int | None = None,
    triage_threshold: int | None = None,
    rerank: bool = False,
    finescale_threshold: float | None = None,
) -> str:
    """Generate digest Markdown content for a given run.

    If *diff* is True, marks each paper as new or carried over from the
    previous run by setting ``is_new`` on each paper dict. If *since_days*
    is given, only papers published within that window are included.

    Returns the rendered Markdown string.
    """
    context = _build_digest_context(
        store,
        run_id,
        top_n=top_n,
        diff=diff,
        suggestions_config=suggestions_config,
        profile=profile,
        since_days=since_days,
        triage_threshold=triage_threshold,
        rerank=rerank,
        finescale_threshold=finescale_threshold,
    )
    rendered: str = _load_template().render(**context)
    return rendered


def generate_digest_html(
    store: PaperStore,
    run_id: int,
    top_n: int = 15,
    diff: bool = False,
    suggestions_config: Any | None = None,
    profile: Any | None = None,
    since_days: int | None = None,
    triage_threshold: int | None = None,
    rerank: bool = False,
    finescale_threshold: float | None = None,
) -> str:
    """Generate a fully rendered HTML digest page (not markdown-in-<pre>).

    Renders the same context as :func:`generate_digest` through an autoescaped
    HTML template, producing a browser-ready page suitable for GitHub Pages.
    """
    context = _build_digest_context(
        store,
        run_id,
        top_n=top_n,
        diff=diff,
        suggestions_config=suggestions_config,
        profile=profile,
        since_days=since_days,
        triage_threshold=triage_threshold,
        rerank=rerank,
        finescale_threshold=finescale_threshold,
    )
    rendered: str = _load_html_template("digest_page.html.j2").render(**context)
    return rendered


def markdown_to_html(md_content: str) -> str:
    """Wrap rendered Markdown in a basic HTML page.

    Uses a simple CSS stylesheet for readability. This avoids adding a
    Markdown-to-HTML library dependency — the output is raw Markdown
    inside <pre> tags with some basic styling. For a richer rendering,
    users can pipe the .md through any Markdown renderer.
    """
    template = _load_template("digest.html.j2")
    rendered: str = template.render(markdown_content=md_content)
    return rendered


def generate_digest_json(
    store: PaperStore,
    run_id: int,
    top_n: int = 15,
    diff: bool = False,
    suggestions_config: Any | None = None,
    profile: Any | None = None,
    since_days: int | None = None,
    triage_threshold: int | None = None,
    rerank: bool = False,
    finescale_threshold: float | None = None,
) -> str:
    """Generate digest as a JSON string.

    Returns a JSON string with top_picks, maybe_relevant, and muted tiers.
    """
    scored = filter_since(store.get_scores_for_run(run_id), since_days)
    run = store.get_run(run_id)

    if diff:
        mark_new_papers(store, scored, run_id)

    top_picks, maybe_relevant, muted = categorize_papers(
        scored,
        top_n=top_n,
        triage_threshold=triage_threshold,
        rerank=rerank,
        finescale_threshold=finescale_threshold,
        cited_ids=cited_ids_from(profile),
    )
    already_cited = [p for p in muted if p.get("already_cited")]
    muted = [p for p in muted if not p.get("already_cited")]
    enrich_papers_with_suggestions(top_picks, config=suggestions_config, profile=profile)

    payload: str = json_mod.dumps(
        {
            "generated_at": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
            "run_id": run_id,
            "papers_new": run["papers_new"] if run else 0,
            "papers_seen": run["papers_seen"] if run else 0,
            "top_picks": top_picks,
            "maybe_relevant": maybe_relevant,
            "muted": muted,
            "already_cited": already_cited,
        },
        indent=2,
        default=str,
    )
    return payload


_CSV_FIELDS = [
    "arxiv_id",
    "title",
    "score_total",
    "keyword_score",
    "category_score",
    "recency_score",
    "embedding_score",
    "citation_score",
    "tier",
    # Which field the withdrawal notice was found in, empty for a clean paper. A
    # spreadsheet export of a digest must not hide that a row is retracted work.
    "withdrawn_in",
    "authors",
    "categories",
    "published",
    "url",
    "has_code",
    "datasets",
    "models",
    "upvotes",
]


def generate_digest_csv(
    store: PaperStore,
    run_id: int,
    top_n: int = 15,
    diff: bool = False,
    since_days: int | None = None,
    triage_threshold: int | None = None,
    rerank: bool = False,
    finescale_threshold: float | None = None,
    profile: Any | None = None,
) -> str:
    """Generate digest as a CSV string."""
    scored = filter_since(store.get_scores_for_run(run_id), since_days)
    top_picks, maybe_relevant, muted = categorize_papers(
        scored,
        top_n=top_n,
        triage_threshold=triage_threshold,
        rerank=rerank,
        finescale_threshold=finescale_threshold,
        cited_ids=cited_ids_from(profile),
    )
    muted = [m for m in muted if not m.get("already_cited")]

    # Tag each paper with its tier
    for p in top_picks:
        p["tier"] = "top_pick"
    for p in maybe_relevant:
        p["tier"] = "maybe_relevant"
    for p in muted:
        p["tier"] = "muted"

    all_papers = top_picks + maybe_relevant + muted

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=_CSV_FIELDS, extrasaction="ignore")
    writer.writeheader()
    for p in all_papers:
        row = dict(p)
        # Flatten list fields
        row["authors"] = "; ".join(p.get("authors", []))
        row["categories"] = "; ".join(p.get("categories", []))
        row["datasets"] = "; ".join(p.get("datasets", []))
        row["models"] = "; ".join(p.get("models", []))
        writer.writerow(row)

    return output.getvalue()


def generate_digest_rss(
    store: PaperStore,
    run_id: int,
    top_n: int = 15,
    diff: bool = False,
    since_days: int | None = None,
    triage_threshold: int | None = None,
    rerank: bool = False,
    finescale_threshold: float | None = None,
    profile: Any | None = None,
) -> str:
    """Generate digest as an RSS 2.0 XML string."""
    scored = filter_since(store.get_scores_for_run(run_id), since_days)
    top_picks, maybe_relevant, muted = categorize_papers(
        scored,
        top_n=top_n,
        triage_threshold=triage_threshold,
        rerank=rerank,
        finescale_threshold=finescale_threshold,
        cited_ids=cited_ids_from(profile),
    )
    muted = [m for m in muted if not m.get("already_cited")]
    all_papers = top_picks + maybe_relevant + muted

    template = _load_template("digest.rss.xml.j2")
    rendered: str = template.render(
        generated_at=datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
        run_id=run_id,
        papers=all_papers,
    )
    return rendered


def write_digest(
    store: PaperStore,
    run_id: int,
    output_path: str | Path,
    top_n: int = 15,
    fmt: str = "md",
    diff: bool = False,
    suggestions_config: Any | None = None,
    profile: Any | None = None,
    since_days: int | None = None,
    triage_threshold: int | None = None,
    rerank: bool = False,
    finescale_threshold: float | None = None,
) -> tuple[Path, DigestSummary | None]:
    """Generate and write the digest to a file.

    *fmt* can be ``"md"``, ``"html"``, ``"json"``, ``"csv"``, or ``"rss"``.
    *since_days*, when given, limits output to papers published in that window.
    *triage_threshold*, when given, tiers Top Picks by LLM actionability score.
    Returns a tuple of (output_path, DigestSummary).
    """
    output_path = Path(output_path)

    if fmt == "json":
        content = generate_digest_json(
            store,
            run_id,
            top_n=top_n,
            diff=diff,
            suggestions_config=suggestions_config,
            profile=profile,
            since_days=since_days,
            triage_threshold=triage_threshold,
            rerank=rerank,
            finescale_threshold=finescale_threshold,
        )
        if output_path.suffix in (".md", ".html"):
            output_path = output_path.with_suffix(".json")
    elif fmt == "csv":
        content = generate_digest_csv(
            store,
            run_id,
            top_n=top_n,
            diff=diff,
            since_days=since_days,
            triage_threshold=triage_threshold,
            rerank=rerank,
            finescale_threshold=finescale_threshold,
            profile=profile,
        )
        if output_path.suffix in (".md", ".html"):
            output_path = output_path.with_suffix(".csv")
    elif fmt == "rss":
        content = generate_digest_rss(
            store,
            run_id,
            top_n=top_n,
            diff=diff,
            since_days=since_days,
            triage_threshold=triage_threshold,
            rerank=rerank,
            finescale_threshold=finescale_threshold,
            profile=profile,
        )
        if output_path.suffix in (".md", ".html"):
            output_path = output_path.with_suffix(".xml")
    elif fmt == "html":
        content = generate_digest_html(
            store,
            run_id,
            top_n=top_n,
            diff=diff,
            suggestions_config=suggestions_config,
            profile=profile,
            since_days=since_days,
            triage_threshold=triage_threshold,
            rerank=rerank,
            finescale_threshold=finescale_threshold,
        )
        if output_path.suffix == ".md":
            output_path = output_path.with_suffix(".html")
    else:
        content = generate_digest(
            store,
            run_id,
            top_n=top_n,
            diff=diff,
            suggestions_config=suggestions_config,
            profile=profile,
            since_days=since_days,
            triage_threshold=triage_threshold,
            rerank=rerank,
            finescale_threshold=finescale_threshold,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")

    # Build digest summary (reflects the same since_days filtering as the output)
    scored = filter_since(store.get_scores_for_run(run_id), since_days)
    run = store.get_last_run()
    top_picks, _, _ = categorize_papers(
        scored,
        top_n=top_n,
        cited_ids=cited_ids_from(profile),
        triage_threshold=triage_threshold,
        rerank=rerank,
        finescale_threshold=finescale_threshold,
    )
    summary = DigestSummary(
        digest_path=str(output_path),
        run_id=run_id,
        papers_new=run["papers_new"] if run else 0,
        papers_seen=run["papers_seen"] if run else 0,
        top_picks_count=len(top_picks),
        total_scored=len(scored),
        fmt=fmt,
        extends_starred_count=len(find_extends_starred(store, scored)),
    )

    return output_path, summary
