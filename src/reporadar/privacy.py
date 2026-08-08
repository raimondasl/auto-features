"""What leaves this machine, and how to stop some of it (Feature 13).

RepoRadar's core mechanic is deriving search queries from your code and docs. That is
exactly what an engineer on a proprietary codebase cannot let leak into arXiv, Semantic
Scholar, OpenAlex or Anthropic query logs — and the outbound surface has grown to a
dozen modules across nine third-party services without anyone being able to answer
"what, precisely, is sent where".

``rr audit`` answers it. Two design commitments make the answer trustworthy:

**The query strings are derived, not described.** The audit calls the real
``build_queries`` on the real profile and prints what it returns. If the audit says a
string leaves, that string leaves — there is no second implementation to drift from the
first.

**The destination list is enforced, not maintained.** :data:`DESTINATIONS` is
declarative, which would normally rot the first time someone adds a source. A privacy
feature that rots is worse than none, because people act on it. So a test walks the
package for modules that make outbound calls and fails if any is undeclared here — see
``tests/test_privacy.py``. Adding an integration without documenting it breaks CI. The
detector is static, so its reach is stated rather than assumed: it recognises a list of
request shapes and propagates through private helpers imported from an already-outbound
module (``specter`` borrows ``citations._s2_batch_post`` and makes no request of its
own). A module reaching the network by some third route would still slip past. This
raises the cost of an undeclared destination; it does not make one impossible.

**Redaction is deliberately modest, and the audit says so.** Removing literal terms
from a query stops a codename reaching a search log. It does *not* hide that you work
on, say, distributed consensus — TF-IDF keywords still encode the domain, and no
denylist changes that. The honest layer is the audit view: read what actually goes out
and decide whether that is acceptable, rather than trusting a filter to make it safe.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# What class of information a destination receives. The distinction that actually
# matters for a proprietary codebase is not "which company" but "does this carry my
# repo's vocabulary, or only public identifiers?".
REPO_DERIVED = "repo-derived"  # search terms inferred from your code and docs
REPO_AND_CONTENT = "repo+paper text"  # profile *and* paper abstracts (LLM prompts)
INTERESTS = "interests"  # which public papers you track, rate or star
NONE = "none"  # nothing repo-derived; a fixed listing or a local process

SENSITIVITY_ORDER = {REPO_AND_CONTENT: 0, REPO_DERIVED: 1, INTERESTS: 2, NONE: 3}


@dataclass(frozen=True)
class Destination:
    """One place data can go, and what reaches it."""

    module: str  # the reporadar module that makes the call
    service: str  # human name
    endpoint: str  # host or URL template
    sends: str  # what actually leaves, in plain words
    sensitivity: str
    enabled_by: str  # the config that turns it on
    # True for destinations only reached by an explicit command (`rr gh-issues`,
    # `rr notify`). They still leave the machine, so the audit reports them in their
    # own section instead of omitting them because `rr update` does not call them.
    on_demand: bool = False
    # Whether this destination is active under a given config. Kept as a lambda so the
    # audit reports *your* posture, not a generic capability list.
    active: Any = field(default=lambda cfg: True)


def _source_enabled(name: str) -> Any:
    return lambda cfg: name in getattr(cfg, "sources", [])


DESTINATIONS: tuple[Destination, ...] = (
    Destination(
        module="collector",
        service="arXiv",
        endpoint="export.arxiv.org/api/query",
        sends="the search queries built from your repo profile (keywords, library names)",
        sensitivity=REPO_DERIVED,
        enabled_by="always (the core source)",
    ),
    Destination(
        module="hyde",
        service="Hugging Face (dense arXiv index)",
        endpoint="huggingface.co/datasets/bluuebunny/arxiv_abstract_embedding_...",
        sends=(
            "nothing about your repository - only HTTP Range requests for index "
            "shards, identical for every user. Matching then happens locally, so "
            "HyDE discovery is a NET REDUCTION in what leaves: the keyword path it "
            "replaces transmits repo-derived queries to arXiv on every run"
        ),
        sensitivity=NONE,
        enabled_by="hyde.enabled (and `rr sync-index`)",
        active=lambda cfg: getattr(getattr(cfg, "hyde", None), "enabled", False),
    ),
    Destination(
        module="sources.semantic_scholar",
        service="Semantic Scholar",
        endpoint="api.semanticscholar.org/graph/v1/paper/search",
        sends="the same repo-derived query strings",
        sensitivity=REPO_DERIVED,
        enabled_by="sources: [semantic_scholar]",
        active=_source_enabled("semantic_scholar"),
    ),
    Destination(
        module="sources.openalex",
        service="OpenAlex",
        endpoint="api.openalex.org/works",
        sends=(
            "repo-derived query strings, plus your api_key - which ties every "
            "query to a billed identity"
        ),
        sensitivity=REPO_DERIVED,
        enabled_by="sources: [openalex]",
        active=_source_enabled("openalex"),
    ),
    Destination(
        module="sources.dblp",
        service="DBLP",
        endpoint="dblp.org/search/publ/api",
        sends="repo-derived query strings",
        sensitivity=REPO_DERIVED,
        enabled_by="sources: [dblp]",
        active=_source_enabled("dblp"),
    ),
    Destination(
        module="sources.biorxiv",
        service="bioRxiv",
        endpoint="api.biorxiv.org/details/{server}/{date-interval}",
        sends="nothing repo-derived: it requests a date window and filters locally",
        sensitivity=NONE,
        enabled_by="sources: [biorxiv]",
        active=_source_enabled("biorxiv"),
    ),
    Destination(
        module="sources.s2_recommendations",
        service="Semantic Scholar (recommendations)",
        endpoint="api.semanticscholar.org/recommendations/v1/papers",
        sends="the arXiv ids of papers you starred or rated - your reading list, not your code",
        sensitivity=INTERESTS,
        enabled_by="recommendations.enabled: true",
        active=lambda cfg: getattr(cfg.recommendations, "enabled", False),
    ),
    Destination(
        module="specter",
        service="Semantic Scholar (SPECTER2 vectors)",
        endpoint="api.semanticscholar.org/graph/v1/paper/batch",
        sends="arXiv ids of candidate and starred papers",
        sensitivity=INTERESTS,
        enabled_by="ranking.w_specter > 0",
        active=lambda cfg: cfg.ranking.w_specter > 0,
    ),
    Destination(
        module="citations",
        service="Semantic Scholar (references)",
        endpoint="api.semanticscholar.org/graph/v1/paper/batch",
        sends="arXiv ids of candidate papers",
        sensitivity=INTERESTS,
        enabled_by="ranking.w_citations > 0 or ranking.w_citation_proximity > 0",
        active=lambda cfg: cfg.ranking.w_citations > 0 or cfg.ranking.w_citation_proximity > 0,
    ),
    Destination(
        module="sources.hf_papers",
        service="Hugging Face Papers",
        endpoint="huggingface.co/api/papers, /api/models, /api/datasets",
        sends="arXiv ids of the papers that made your digest",
        sensitivity=INTERESTS,
        enabled_by="enrichment.provider: huggingface",
        active=lambda cfg: cfg.enrichment.provider != "off",
    ),
    Destination(
        module="signals.hn",
        service="Hacker News (Algolia)",
        endpoint="hn.algolia.com/api/v1/search",
        sends="arXiv ids, and one year-month prefix per run",
        sensitivity=INTERESTS,
        enabled_by="signals.hackernews: true",
        active=lambda cfg: getattr(cfg.signals, "hackernews", False),
    ),
    Destination(
        module="signals.integrity",
        service="arXiv (withdrawal check)",
        endpoint="export.arxiv.org/api/query?id_list=",
        sends="arXiv ids of candidate papers",
        sensitivity=INTERESTS,
        enabled_by="signals.integrity: true (on by default)",
        active=lambda cfg: getattr(cfg.signals, "integrity", True),
    ),
    Destination(
        module="llm_client",
        service="Anthropic",
        endpoint="api.anthropic.com/v1/messages",
        sends=(
            "your repo's libraries and key topics, AND up to profiler.prose_chars "
            "characters of your README, TOGETHER WITH paper abstracts, in every prompt"
        ),
        sensitivity=REPO_AND_CONTENT,
        enabled_by=(
            "suggestions.provider: claude (used by triage and LLM suggestions); "
            "set profiler.prose_chars: 0 to withhold the README"
        ),
        active=lambda cfg: getattr(cfg.suggestions, "provider", "template") == "claude",
    ),
    Destination(
        module="llm_client",
        service="Ollama (local)",
        endpoint="suggestions.ollama_url (localhost by default)",
        sends="the same prompt as above - but to a process you run, so nothing leaves the machine",
        sensitivity=NONE,
        enabled_by="suggestions.provider: ollama",
        active=lambda cfg: getattr(cfg.suggestions, "provider", "template") == "ollama",
    ),
    Destination(
        module="notify",
        service="Slack / Discord / SMTP",
        endpoint="hooks.slack_webhook_url, hooks.discord_webhook_url, hooks.email",
        sends="a one-line run summary including paper counts (no repo text)",
        sensitivity=INTERESTS,
        enabled_by="hooks: configured, and you run `rr notify`",
        on_demand=True,
        active=lambda cfg: bool(
            cfg.hooks.slack_webhook_url
            or cfg.hooks.discord_webhook_url
            or cfg.hooks.email.smtp_host
        ),
    ),
    Destination(
        module="notify",
        service="Your `hooks.on_digest` shell command",
        endpoint="wherever that command sends it; this tool cannot know",
        sends=(
            "the run summary via RR_* environment variables, to an arbitrary command. "
            "Unlike every other row here, the payload and destination are yours, so "
            "read the command itself"
        ),
        # Rated at the ceiling on purpose: an arbitrary shell command can exfiltrate
        # anything, and a privacy audit must not imply a bound it cannot enforce.
        sensitivity=REPO_AND_CONTENT,
        enabled_by="hooks.on_digest: <command>",
        on_demand=True,
        active=lambda cfg: bool(getattr(cfg.hooks, "on_digest", "")),
    ),
    Destination(
        module="gh_issues",
        service="GitHub",
        endpoint="the `gh` CLI, against your configured remote",
        sends="paper titles, abstracts and suggestions, as issue bodies in your repo",
        sensitivity=INTERESTS,
        enabled_by="you run `rr gh-issues`",
        on_demand=True,
    ),
    Destination(
        module="embeddings",
        service="Hugging Face Hub (model download)",
        endpoint="huggingface.co (sentence-transformers/all-MiniLM-L6-v2)",
        # The exception that proves the registry has to be enforced rather than
        # eyeballed: this one downloads *weights* rather than calling an API, so it
        # matches none of the request shapes the drift guard looks for.
        sends="nothing repo-derived - it downloads model weights on first use, then runs offline",
        sensitivity=NONE,
        enabled_by="ranking.w_embedding > 0, or `rr search --semantic/--hybrid`",
        active=lambda cfg: cfg.ranking.w_embedding > 0,
    ),
    Destination(
        module="profiler",
        service="arXiv (via collector)",
        endpoint="n/a - reads your repo locally",
        sends="nothing directly; it BUILDS the queries the sources above transmit",
        sensitivity=NONE,
        enabled_by="always",
    ),
)


REGEX_PREFIX = "re:"


def compile_patterns(patterns: list[str]) -> list[re.Pattern[str]]:
    """Compile redaction entries. **Literal by default; regex only with a ``re:`` prefix.**

    Treating entries as regexes by default is quietly dangerous here. ``C++`` is a
    *valid* Python 3.11 regex — ``++`` is a possessive quantifier — so it compiles
    happily and redacts only the letter ``C``, leaving ``++ parser`` in the query while
    the user believes their codename was scrubbed. Silent partial redaction is the
    worst outcome a privacy filter can produce: it is indistinguishable from working.

    So a denylist entry means exactly what it says, and a user who wants a pattern
    writes ``re:proj-\\d+``. An invalid regex after ``re:`` falls back to literal
    matching rather than being dropped, for the same reason.

    A regex that can match the **empty string** is rejected outright. ``re:x*`` matches
    at every position, so substituting for each match turns ``learning to rank`` into
    ``l e a r n i n g t o r a n k`` — which is then transmitted verbatim. That is worse
    than no redaction: the term is still legible to any reader, the query is destroyed,
    and both happen silently.
    """
    compiled: list[re.Pattern[str]] = []
    for raw in patterns:
        if not raw or not raw.strip():
            continue
        if raw.startswith(REGEX_PREFIX):
            body = raw[len(REGEX_PREFIX) :]
            try:
                pattern = re.compile(body, re.IGNORECASE)
            except re.error:
                compiled.append(re.compile(re.escape(body), re.IGNORECASE))
                continue
            if pattern.match("") is not None:
                logger.warning(
                    "Ignoring redaction pattern %r: it matches the empty string, which "
                    "would mangle every outbound query instead of redacting a term.",
                    raw,
                )
                continue
            compiled.append(pattern)
            continue
        compiled.append(re.compile(re.escape(raw), re.IGNORECASE))
    return compiled


def redact(text: str, patterns: list[re.Pattern[str]]) -> str:
    """Remove every match from *text*, tidying only the whitespace redaction created.

    Two rules, both there because this runs over LLM prompts as well as query strings:
    line breaks are preserved, and a line no pattern touched is passed through byte for
    byte. Collapsing everything would be simpler, but the triage prompt is an indented
    multi-line rubric — flattening it, or even just stripping its indentation, would
    change model behaviour as a silent side effect of turning redaction on. Only the
    lines redaction actually cut are re-spaced, to close the gap the removal left.

    For a query string — one line, and one that matched — this is the same operation
    as before, so nothing is given up by being careful.
    """
    if not text or not patterns:
        return text
    out_lines: list[str] = []
    for line in text.splitlines() or [text]:
        redacted = line
        for pattern in patterns:
            redacted = pattern.sub(" ", redacted)
        if redacted == line:
            out_lines.append(line)  # untouched: leave the author's formatting alone
        else:
            out_lines.append(" ".join(redacted.split()))
    return "\n".join(out_lines) if "\n" in text else "".join(out_lines)


def redact_all(values: list[str], patterns: list[re.Pattern[str]]) -> list[str]:
    """Redact a list, dropping entries that redaction emptied."""
    if not patterns:
        return list(values)
    return [r for r in (redact(v, patterns) for v in values) if r]


def _sorted(dests: list[Destination]) -> list[Destination]:
    return sorted(dests, key=lambda d: (SENSITIVITY_ORDER.get(d.sensitivity, 9), d.service))


def active_destinations(cfg: Any, include_on_demand: bool = False) -> list[Destination]:
    """The destinations a ``rr update`` run reaches, most sensitive first.

    With *include_on_demand*, also returns the ones only an explicit command reaches.
    """
    out = []
    for dest in DESTINATIONS:
        if dest.on_demand and not include_on_demand:
            continue
        try:
            if dest.active(cfg):
                out.append(dest)
        except Exception:
            # A destination whose gate cannot be evaluated is reported, not hidden:
            # under-reporting is the one failure mode this feature must not have.
            out.append(dest)
    return _sorted(out)


def on_demand_destinations(cfg: Any) -> list[Destination]:
    """Destinations reached only by an explicit command, not by ``rr update``."""
    out = []
    for dest in DESTINATIONS:
        if not dest.on_demand:
            continue
        try:
            if dest.active(cfg):
                out.append(dest)
        except Exception:
            out.append(dest)
    return _sorted(out)


def audit_plan(
    cfg: Any,
    profile: Any,
    queries: list[str],
    queries_unredacted: list[str] | None = None,
) -> dict[str, Any]:
    """Everything ``rr audit`` reports, as data.

    *queries* must come from the real ``build_queries`` call — post-redaction, since
    redaction now happens inside it — so the audit shows the strings that would
    genuinely be transmitted rather than a re-derivation that could drift from them.

    *queries_unredacted* is the same call with ``privacy.redact`` emptied. Passing it
    lets the report show what the filter actually removed; without it the audit still
    lists the real outbound strings, just with nothing to diff against.
    """
    patterns = compile_patterns(cfg.privacy.redact)
    before = list(queries_unredacted) if queries_unredacted is not None else list(queries)
    keywords = [term for term, _ in getattr(profile, "keywords", [])]
    return {
        "destinations": active_destinations(cfg),
        "on_demand": on_demand_destinations(cfg),
        "queries": list(queries),
        "queries_before_redaction": before,
        "redaction_changed_anything": before != list(queries),
        "n_patterns": len(patterns),
        "keywords": redact_all(keywords, patterns),
        "anchors": redact_all(list(getattr(profile, "anchors", [])), patterns),
        "scans_source": bool(getattr(cfg.profiler, "scan_source", False)),
    }
