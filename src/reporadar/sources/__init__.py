"""Paper source adapters.

Each adapter module is a secondary paper source, activated by listing its name in
``sources`` in ``.reporadar.yml`` (``arxiv`` is always the primary). An adapter
exposes:

    collect_papers(queries: list[str], lookback_days: int = 14, **opts)
        -> list[dict]

returning papers in the shared internal dict shape used across the pipeline::

    {
        "arxiv_id": str,   # a real arXiv id, or a synthetic "<source>:<id>" for
                           # non-arXiv papers (e.g. "dblp:...", "biorxiv:...", "ss:...")
        "title": str,
        "authors": list[str],
        "abstract": str,   # "" when the source has no abstracts (e.g. DBLP)
        "categories": list[str],
        "published": str,  # ISO-8601
        "updated": str | None,
        "url": str,
        "pdf_url": str | None,
    }

Adapters deduplicate their own results, degrade gracefully (return ``[]`` on API
failure — never raise), and are merged into the arXiv results with arXiv taking
priority (``cli.update``). Keyword-search sources (Semantic Scholar, OpenAlex,
DBLP) query directly; listing sources (bioRxiv) fetch a recent window and filter
locally.

Adapters: ``semantic_scholar``, ``openalex``, ``dblp``, ``biorxiv``, plus
``hf_papers`` (enrichment, not collection).
"""
