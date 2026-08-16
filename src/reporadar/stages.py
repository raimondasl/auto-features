"""Which pipeline stages each entry point actually runs.

`rr update` is the pipeline every published number describes. `rr watch` and
`rr workspace update` re-implement its front half and stop before most of it — no gate, no
HyDE, no fusion, no enrichment. That has been true since the watcher was written, is
recorded in ROADMAP's Tier 0 as "pipeline drift", and until 2026-08-16 nothing told the
user. A configuration saying `triage.enabled: true` produced an ungated digest in silence,
which is worse than a configuration that is merely different: the file asserted something
the run did not do.

This module is the *disclosure*, not the fix. The fix is one shared pipeline; this makes
the gap visible and — more importantly — makes it impossible to close by accident and
leave the warning claiming a stage is still missing.

**Why a registry rather than a hand-written warning string.** A message listing whichever
stages somebody remembered reads exactly like one listing all of them. That is the C-9b
shape (an index covering a remembered subset) and the C-16 shape (an audit scoped by hand
reporting clean over 12 of 79 fields). So each stage names the module its implementation
imports, and `tests/test_stages.py` walks the real import graph of every entry point and
refuses any disagreement in *either* direction: a stage claimed unrun that is in fact
imported, or claimed run that is not. Adding a stage to the watcher without updating this
table fails; updating this table without adding the stage fails too.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

# Entry points that build a candidate pool and rank it. `rr digest` is not one — it reads
# scores a run already produced.
UPDATE = "update"
WATCH = "watch"
WORKSPACE = "workspace"
ENTRY_POINTS = (UPDATE, WATCH, WORKSPACE)


@dataclass(frozen=True)
class Stage:
    """One optional stage of the update pipeline.

    *module* is the import the stage's implementation reaches for, and it is what the drift
    guard matches against an entry point's real import graph. A stage whose module is also
    imported for some unrelated reason would make the guard lie, so each one names the
    module that exists *only* to serve it.
    """

    key: str
    label: str
    module: str
    enabled: Callable[[Any], bool]
    runs_in: frozenset[str]
    note: str = ""

    def missing_from(self, entry_point: str) -> bool:
        return entry_point not in self.runs_in


def _gate_on(cfg: Any) -> bool:
    """The gate needs BOTH fields — `triage.enabled` alone is a no-op that prints one line."""
    return bool(cfg.triage.enabled) and cfg.suggestions.provider in ("ollama", "claude")


# Ordered by how much the measurement record says each one is worth, so a truncated
# warning still leads with the stage whose absence matters most.
STAGES: tuple[Stage, ...] = (
    Stage(
        key="triage",
        label="LLM actionability gate",
        module="reporadar.triage",
        enabled=_gate_on,
        runs_in=frozenset({UPDATE}),
        note="the stage that declines to show a paper; most of the -8.12 -> +5.72 gap",
    ),
    Stage(
        key="finescale",
        label="fine-scale rescore",
        module="reporadar.finescale",
        enabled=lambda cfg: _gate_on(cfg) and bool(cfg.triage.finescale.enabled),
        runs_in=frozenset({UPDATE}),
        note="worth +1.86 -> +3.18 net@2; needs the gate above to have run",
    ),
    Stage(
        key="hyde",
        label="HyDE dense discovery",
        module="reporadar.hyde",
        enabled=lambda cfg: bool(cfg.hyde.enabled),
        runs_in=frozenset({UPDATE}),
        note="+1.36 net@2; reaches papers keyword search structurally cannot",
    ),
    Stage(
        key="hybrid",
        label="BM25 fusion (ranking.hybrid)",
        module="reporadar.retrieval",
        enabled=lambda cfg: bool(cfg.ranking.hybrid),
        runs_in=frozenset({UPDATE}),
        note="in every headline; measured +0.00 on its own (NR-35)",
    ),
    Stage(
        key="embeddings",
        label="embedding similarity (ranking.w_embedding)",
        module="reporadar.embeddings",
        enabled=lambda cfg: cfg.ranking.w_embedding > 0,
        runs_in=frozenset({UPDATE}),
        note="+1.00 net@2 at the measured weight of 1.5 (NR-38)",
    ),
    Stage(
        key="specter",
        label="SPECTER2 similarity (ranking.w_specter)",
        module="reporadar.specter",
        enabled=lambda cfg: cfg.ranking.w_specter > 0,
        runs_in=frozenset({UPDATE}),
    ),
    Stage(
        key="citations",
        label="citation counts (ranking.w_citations)",
        module="reporadar.citations",
        enabled=lambda cfg: cfg.ranking.w_citations > 0,
        runs_in=frozenset({UPDATE}),
    ),
    Stage(
        key="citation_proximity",
        label="extends-work-you-starred (ranking.w_citation_proximity)",
        module="reporadar.citation_graph",
        enabled=lambda cfg: cfg.ranking.w_citation_proximity > 0,
        runs_in=frozenset({UPDATE}),
    ),
    Stage(
        key="community",
        label="community attention (ranking.w_community)",
        module="reporadar.sources.hf_papers",
        enabled=lambda cfg: cfg.ranking.w_community > 0,
        runs_in=frozenset({UPDATE}),
    ),
    Stage(
        key="hackernews",
        label="Hacker News attention (signals.hackernews)",
        module="reporadar.signals.hn",
        enabled=lambda cfg: bool(cfg.signals.hackernews),
        runs_in=frozenset({UPDATE}),
    ),
    Stage(
        key="feedback",
        label="feedback-adjusted ranking weights",
        module="reporadar.feedback",
        enabled=lambda cfg: bool(cfg.feedback.enabled),
        runs_in=frozenset({UPDATE}),
    ),
    Stage(
        key="recommendations",
        label="learned recommendations (recommendations.enabled)",
        module="reporadar.sources.s2_recommendations",
        enabled=lambda cfg: bool(cfg.recommendations.enabled),
        runs_in=frozenset({UPDATE}),
    ),
    Stage(
        key="enrichment",
        label="Hugging Face enrichment (enrichment.provider)",
        module="reporadar.sources.hf_papers",
        enabled=lambda cfg: cfg.enrichment.provider != "off",
        runs_in=frozenset({UPDATE}),
    ),
    # Non-arXiv sources. Both reduced entry points call `collector.collect_papers`, which
    # is arXiv-only, so every other configured source is silently dropped -- a config
    # asking for five literatures gets one.
    Stage(
        key="source_semantic_scholar",
        label="Semantic Scholar source",
        module="reporadar.sources.semantic_scholar",
        enabled=lambda cfg: "semantic_scholar" in cfg.sources,
        runs_in=frozenset({UPDATE}),
    ),
    Stage(
        key="source_openalex",
        label="OpenAlex source",
        module="reporadar.sources.openalex",
        enabled=lambda cfg: "openalex" in cfg.sources,
        runs_in=frozenset({UPDATE}),
    ),
    Stage(
        key="source_biorxiv",
        label="bioRxiv source",
        module="reporadar.sources.biorxiv",
        enabled=lambda cfg: "biorxiv" in cfg.sources,
        runs_in=frozenset({UPDATE}),
    ),
    Stage(
        key="source_iacr",
        label="IACR ePrint source",
        module="reporadar.sources.iacr",
        enabled=lambda cfg: "iacr" in cfg.sources,
        runs_in=frozenset({UPDATE}),
    ),
    Stage(
        key="source_dblp",
        label="DBLP source",
        module="reporadar.sources.dblp",
        enabled=lambda cfg: "dblp" in cfg.sources,
        runs_in=frozenset({UPDATE}),
    ),
    # Runs everywhere, and listed so the guard checks a positive case too. A table that
    # only ever asserts absence cannot tell "correctly absent" from "guard is broken".
    Stage(
        key="integrity",
        label="withdrawal / retraction check (signals.integrity)",
        module="reporadar.signals.integrity",
        enabled=lambda cfg: bool(cfg.signals.integrity),
        runs_in=frozenset({UPDATE, WATCH}),
        note="on by default; `rr watch` is unattended, so it is not an enhancement to skip",
    ),
)


def unrun_stages(cfg: Any, entry_point: str) -> list[Stage]:
    """Stages this config enables that *entry_point* will not execute.

    Empty for `rr update` by construction, which is the point: it is the pipeline the
    numbers describe, and every other entry point is measured against it.
    """
    if entry_point not in ENTRY_POINTS:
        raise ValueError(f"unknown entry point {entry_point!r}; expected one of {ENTRY_POINTS}")
    return [s for s in STAGES if s.missing_from(entry_point) and s.enabled(cfg)]


def drift_warning(cfg: Any, entry_point: str, *, limit: int = 5) -> list[str]:
    """The user-facing disclosure, as lines. Empty when there is nothing to disclose."""
    missing = unrun_stages(cfg, entry_point)
    if not missing:
        return []
    command = {WATCH: "`rr watch`", WORKSPACE: "`rr workspace update`"}.get(
        entry_point, entry_point
    )
    lines = [
        f"{command} runs a REDUCED pipeline. Your config enables "
        f"{len(missing)} stage(s) it will not run:"
    ]
    for stage in missing[:limit]:
        lines.append(f"  - {stage.label}" + (f" -- {stage.note}" if stage.note else ""))
    if len(missing) > limit:
        lines.append(f"  - ...and {len(missing) - limit} more")
    lines.append(
        "Every published number describes the `rr update` pipeline. Run `rr update` on a "
        "schedule (cron, or the GitHub Action) to get the configuration you measured."
    )
    return lines
