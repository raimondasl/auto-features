"""Suggest domain-matched paper sources from a repo profile (Feature 10).

The extra source adapters (bioRxiv, DBLP) only help repos whose literature isn't
primarily on arXiv, and a user has no way to know that without reading the docs.
This module reads the repo profile and *suggests* the sources that match it.

Deliberately advisory: nothing here activates a source. Both adapters have real
costs — DBLP rate-limits aggressively and indexes publication *year* only, so
silently enabling it would slow runs down and mismatch the recency window — so the
choice stays with the user, with the caveat printed alongside the suggestion.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from reporadar.profiler import RepoProfile


@dataclass(frozen=True)
class SourceSuggestion:
    """A source worth enabling, the evidence for it, and its cost."""

    source: str
    evidence: list[str]
    reason: str
    caveat: str = ""


# Packages that place a repo squarely in biology/medicine, where the relevant
# preprints are on bioRxiv/medRxiv rather than arXiv.
_BIORXIV_ANCHORS = frozenset(
    {
        "biopython",
        "bio",
        "scanpy",
        "anndata",
        "squidpy",
        "scvi-tools",
        "pysam",
        "pybedtools",
        "biotite",
        "mdanalysis",
        "openmm",
        "rdkit",
        "deeptools",
        "scikit-bio",
        "gseapy",
        "lifelines",
        "nibabel",
        "mne",
        "cellprofiler",
        "seurat",
        "bioconductor",
        "deseq2",
        "edger",
        "plink",
        "ensembl",
        "biomart",
    }
)

_BIORXIV_TERMS = frozenset(
    {
        "genomics",
        "genome",
        "genomic",
        "transcriptomics",
        "proteomics",
        "protein",
        "proteins",
        "rna",
        "dna",
        "sequencing",
        "biology",
        "biological",
        "biomedical",
        "bioinformatics",
        "clinical",
        "neuroscience",
        "neuroimaging",
        "molecular",
        "gene",
        "genes",
        "epidemiology",
        "microbiome",
        "crispr",
        "pathology",
        "immunology",
        "pharmacology",
        "cheminformatics",
    }
)

# Packages that place a repo in the systems/PL/DB/SE world, where the venues are
# conferences (SIGMOD, OSDI, PLDI, POPL, ICSE) that DBLP indexes and arXiv
# largely does not.
_DBLP_ANCHORS = frozenset(
    {
        "duckdb",
        "rocksdb",
        "leveldb",
        "lmdb",
        "clickhouse",
        "cassandra",
        "scylla",
        "foundationdb",
        "tikv",
        "flink",
        "pulsar",
        "zookeeper",
        "etcd",
        "llvmlite",
        "llvm",
        "antlr4",
        "tree-sitter",
        "wasmtime",
        "wasmer",
        "z3-solver",
        "cvc5",
        "coq",
        "lean",
        "tla",
        "jepsen",
        "ebpf",
        "bcc",
        "dpdk",
        "spdk",
    }
)

_DBLP_TERMS = frozenset(
    {
        "compiler",
        "compilers",
        "database",
        "databases",
        "transaction",
        "transactions",
        "consensus",
        "raft",
        "paxos",
        "filesystem",
        "concurrency",
        "replication",
        "sharding",
        "serializability",
        "typesystem",
        "serverless",
        "virtualization",
        "hypervisor",
        "microarchitecture",
    }
)

# The profiler's inferred domains (profiler.PACKAGE_DOMAIN_MAP) are deliberately
# NOT used as a signal: its vocabulary has no label specific to either source's
# literature. "containers" and "data pipelines" mark deployment tooling — a repo
# with a Dockerfile is not a systems-research repo — "distributed computing" comes
# from Ray/Dask (an ML stack), and "databases" comes only from SQLAlchemy, an ORM
# that half of all web apps depend on. Matching those would fire on repos arXiv
# already serves. Anchors and keywords carry the signal instead.

# One matched anchor is decisive on its own; loose keyword hits are noisier (a repo
# can mention "protein" once in passing), so they need corroboration.
_ANCHOR_POINTS = 2
_TERM_POINTS = 1
_SUGGEST_THRESHOLD = 2

# Cap the evidence we quote back, so the hint stays one line.
_MAX_EVIDENCE = 4


def _normalize(name: str) -> str:
    """Lowercase a package name and drop separators (``scikit-bio`` -> ``scikitbio``)."""
    return re.sub(r"[-_.\s]", "", name.lower())


def _terms(profile: RepoProfile) -> set[str]:
    """Every word-level token from the profile's keywords and source signals."""
    tokens: set[str] = set()
    for term, _weight in profile.keywords:
        tokens.update(re.findall(r"[a-z][a-z0-9]+", term.lower()))
    for signal in profile.source_signals:
        tokens.update(re.findall(r"[a-z][a-z0-9]+", signal.lower()))
    return tokens


def _score(
    profile: RepoProfile, anchors: frozenset[str], terms: frozenset[str]
) -> tuple[int, list[str]]:
    """Score a source against the profile. Returns (score, matched evidence)."""
    normalized_anchors = {_normalize(a) for a in anchors}
    anchor_hits = sorted({a for a in profile.anchors if _normalize(a) in normalized_anchors})
    term_hits = sorted(_terms(profile) & terms)

    score = _ANCHOR_POINTS * len(anchor_hits) + _TERM_POINTS * len(term_hits)
    # dict.fromkeys dedupes while keeping anchors-before-terms order: a package name
    # can also appear as a keyword token, and quoting it twice reads like a bug.
    evidence = list(dict.fromkeys(anchor_hits + term_hits))
    return score, evidence[:_MAX_EVIDENCE]


def suggest_sources(profile: RepoProfile, active_sources: list[str]) -> list[SourceSuggestion]:
    """Return sources worth adding to ``sources:`` for this repo, best evidence first.

    Sources already listed in *active_sources* are never suggested. Returns an
    empty list for a repo whose literature arXiv already covers, which is the
    common case — the point is to speak up only when there's a real gap.
    """
    active = set(active_sources)
    candidates = (
        (
            "biorxiv",
            _BIORXIV_ANCHORS,
            _BIORXIV_TERMS,
            "biology/medicine preprints land on bioRxiv/medRxiv, not arXiv",
            "bioRxiv has no keyword search, so a run pages through the whole date "
            "window and filters locally (capped at 40 pages — a wide "
            "arxiv.lookback_days may truncate coverage)",
        ),
        (
            "dblp",
            _DBLP_ANCHORS,
            _DBLP_TERMS,
            "systems/PL/DB work is published at conferences DBLP indexes",
            "DBLP is rate-limited and records publication year only, so pair it "
            "with a wide arxiv.lookback_days (or `rr update --foundational`)",
        ),
    )

    suggestions: list[tuple[int, SourceSuggestion]] = []
    for source, anchors, terms, reason, caveat in candidates:
        if source in active:
            continue
        score, evidence = _score(profile, anchors, terms)
        if score >= _SUGGEST_THRESHOLD:
            suggestions.append(
                (score, SourceSuggestion(source, evidence, reason, caveat)),
            )

    suggestions.sort(key=lambda pair: (-pair[0], pair[1].source))
    return [suggestion for _score_value, suggestion in suggestions]


def format_suggestion(suggestion: SourceSuggestion) -> str:
    """Render a suggestion as a single line for the CLI."""
    evidence = ", ".join(suggestion.evidence)
    line = f"Consider adding '{suggestion.source}' to sources: {suggestion.reason}"
    if evidence:
        line += f" (matched: {evidence})"
    if suggestion.caveat:
        line += f". Note: {suggestion.caveat}"
    return line
