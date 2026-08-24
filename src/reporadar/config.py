"""Load and validate .reporadar.yml configuration."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field, is_dataclass
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_NAME = ".reporadar.yml"

# Matches ``${VAR}`` (braced form only — a bare ``$VAR`` or a lone ``$`` is left
# untouched, so ordinary config values containing ``$`` are never mangled).
_ENV_REF = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _expand_env(value: Any) -> Any:
    """Recursively expand ``${VAR}`` references in string values from os.environ.

    Lets a committed ``.reporadar.yml`` reference secrets by name
    (``slack_webhook_url: ${SLACK_WEBHOOK}``) that a runner (e.g. the GitHub
    Action) injects as environment variables — so secrets never live in the file.
    An unset variable expands to the empty string, which the consumers already
    treat as "not configured".
    """
    if isinstance(value, str):
        return _ENV_REF.sub(lambda m: os.environ.get(m.group(1), ""), value)
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    return value


def _normalize_off(value: Any) -> str:
    """Coerce an off-switch value to a string, treating YAML 1.1 booleans as ``"off"``.

    PyYAML implements YAML 1.1, where a bare ``off`` (also ``no``, ``false``) parses
    as the boolean ``False`` — so the documented ``provider: off`` arrived as
    ``False``, the ``!= "off"`` guard passed, and enrichment kept running: an
    off-switch that did nothing unless the user happened to quote it.
    """
    if value is False:
        return "off"
    if value is True:
        return "huggingface"
    return str(value)


@dataclass
class ArxivConfig:
    categories: list[str] = field(default_factory=lambda: ["cs.LG", "cs.CL"])
    max_results_per_query: int = 50
    # All-time, relevance-first. This used to default to a 14-day submitted-first
    # window, and that default was never the thing the benchmark measured: every
    # headline Tier B number since 2026-07-06 was produced with `--rr-all-time`,
    # which is exactly (lookback 36500, sort relevance, w_recency 0). The shipped
    # default and the validated configuration had drifted apart for a month.
    #
    # It is also structurally unable to do the job. All 48 benchmark targets are
    # >= 11 months old, so a fortnight's window cannot reach any of them, and P5
    # measured where the useful papers actually are: a dense band across the whole
    # corpus, not the last two weeks of it.
    #
    # Repeat runs do not re-show the same work — the store records `first_seen` and
    # the digest marks `is_new` — so an all-time default is a large first digest
    # that tapers, not a weekly re-run of the same seminal papers.
    lookback_days: int = 36500
    # "relevance" = best-match-first, which (with a large lookback_days) surfaces
    # seminal older papers; "submitted" = newest-first, for a strict recency digest.
    sort_by: str = "relevance"


# Phrase-query policies. See collector._generate_bigram_queries for what each emits and
# the measurement that made the choice worth exposing.
BIGRAM_MODES = ("adjacent", "verified", "none")

# How `w_category` treats a paper with no categories. See RankingConfig.absent_category.
ABSENT_CATEGORY_MODES = ("omit", "zero", "impute")


@dataclass
class QueriesConfig:
    """How arXiv query strings are assembled from a repo profile."""

    seed: list[str] = field(default_factory=list)
    exclude: list[str] = field(default_factory=list)
    # Not user-set under `queries:` — populated from `privacy.redact` at load time
    # so `build_queries` strips the terms from every query string it hands out.
    redact: list[str] = field(default_factory=list)
    # How phrase queries are built from the keyword profile:
    #   adjacent — pair each keyword with its TF-IDF neighbour (the original behaviour)
    #   verified — the same pairs, but only those that occur literally in the repo text
    #   none     — no phrase queries; single keywords and anchors carry retrieval
    #
    # Default changed to `verified` 2026-08-12, and **not** because it scores better: on 25
    # cases it is +0.04 net@2/case against `adjacent`, sign p = 0.55, well inside the 1.04
    # noise floor. The reasons are that `adjacent` demonstrably asks for phrases no
    # repository contains ("use page" for duckdb), that the repair costs nothing measurable
    # on arXiv, and that arXiv is the only place it costs nothing — a category clause keeps
    # arXiv results in the right field however meaningless the phrase, and keyword sources
    # have no such fallback. `none` was measured WORSE (-0.48/case, precision 0.914 ->
    # 0.880), so deleting phrase queries is refuted, not merely unsupported.
    # Full three-arm result: evals/RESULTS.md, "Phrase queries".
    bigrams: str = "verified"


@dataclass
class RankingConfig:
    w_keyword: float = 1.0
    w_category: float = 0.5
    # What to do with `w_category` for a paper carrying no category signal comparable to
    # `arxiv.categories` — every paper from every non-arXiv source, whether its adapter
    # leaves the field empty (Semantic Scholar) or fills it from another taxonomy
    # (OpenAlex topics, bioRxiv subjects, DBLP venues). See ranker.has_comparable_categories.
    #   omit   — drop the component (the shipped behaviour since Feature 10)
    #   zero   — score it 0 and keep the component in the average
    #   impute — score it the mean category score of the categorised papers in the pool
    #
    # `omit` was chosen so a missing signal would not handicap non-arXiv papers, and it
    # does the opposite: an arXiv paper is scored on keyword AND category while an
    # uncategorised one is scored on keyword alone, so at equal keyword relevance the
    # uncategorised paper wins (0.600 vs 0.567 at the shipped weights, 0.600 vs 0.400 when
    # the arXiv paper's category match is 0). The absent-signal rule is right when
    # missingness is random; here it is perfectly correlated with the source.
    # See ranker.score_paper and evals/RESULTS.md.
    absent_category: str = "omit"
    # 0.0, not 0.3: with an all-time `arxiv.lookback_days` a recency weight ranks a
    # forgettable preprint from last week above the paper that would actually change
    # the project. This is the third of the three knobs `--foundational` turns, and
    # the benchmark has always measured it at 0.
    w_recency: float = 0.0
    w_embedding: float = 0.0
    w_citations: float = 0.0
    # Boost (and gate) for "extends work you starred": when > 0, RepoRadar fetches
    # each candidate's references and rewards papers that cite a starred/highly-rated
    # paper. 0 disables the feature (no reference lookups).
    w_citation_proximity: float = 0.0
    # SPECTER2 (citation-trained scientific embeddings, served free by Semantic
    # Scholar) similarity to the papers you starred/rated highly. 0 disables the
    # feature entirely (no vector fetches).
    w_specter: float = 0.0
    # Community attention: Hugging Face Papers upvotes, which enrichment already
    # fetches and stores. Uses enrichments cached from *previous* runs, because
    # enrichment happens after ranking (see cli.update) — so a brand-new paper has
    # no community signal on its first run and is simply not scored on it.
    w_community: float = 0.0
    # Hacker News attention (Feature 9). Opt-in and deliberately sparse: measured at
    # 0/340 for random papers from the last two weeks, so it is a badge that
    # occasionally becomes a ranking signal, not a dependable component.
    w_attention: float = 0.0
    # Multiplier applied to a withdrawn paper's total. Not a weighted component: a
    # withdrawn paper must not be able to rank into Top Picks on other strengths.
    # Kept above 0 so it still appears (flagged) in Muted rather than vanishing.
    withdrawn_penalty: float = 0.1
    category_weights: dict[str, float] = field(default_factory=dict)
    # Hybrid retrieval (roadmap #4): fuse the heuristic ranking with a BM25 lexical
    # ranking via RRF, so a paper buried on vocabulary mismatch can still surface.
    hybrid: bool = False


@dataclass
class SemanticScholarConfig:
    api_key: str = ""


@dataclass
class OutputConfig:
    digest_path: str = "./reporadar_digest.md"
    top_n: int = 15


@dataclass
class OpenAlexConfig:
    # OpenAlex requires an API key for real use since 2026-02-13 (keyless
    # callers get only a ~$0.10/day test allowance, then throttling). ``email``
    # is the legacy polite-pool mailto, superseded by keys but still accepted.
    email: str = ""
    api_key: str = ""


@dataclass
class EnrichmentConfig:
    # "huggingface" queries the free HF Papers API for code/model/dataset links
    # and community upvotes; "off" disables enrichment entirely.
    provider: str = "huggingface"  # "huggingface" | "off"
    hf_token: str = ""  # optional; raises the anonymous HF rate limit


@dataclass
class SignalsConfig:
    """Attention and integrity signals about papers already collected (Feature 9)."""

    # Withdrawal detection. Costs a handful of throttled arXiv requests per run and
    # protects against the worst ranking failure — recommending retracted work — so
    # it defaults ON, unlike every other optional integration.
    integrity: bool = True
    # Hacker News discussion lookup (keyless). Off by default: measured coverage is
    # ~0% for papers from the last two weeks, so most runs pay requests for nothing.
    hackernews: bool = False


@dataclass
class PrivacyConfig:
    """Controls on what repo-derived text is allowed to leave (Feature 13).

    Deliberately modest, and `rr audit` says so: removing literal terms stops a
    codename reaching a search log, but TF-IDF keywords still encode the domain you
    work in. The audit view is the honest layer, not this filter.
    """

    # Terms or regexes stripped from every outbound query and LLM prompt. An entry
    # that is not valid regex is matched literally, so `C++` behaves as written.
    redact: list[str] = field(default_factory=list)


@dataclass
class EmailHookConfig:
    smtp_host: str = ""
    smtp_port: int = 587
    from_addr: str = ""
    to: str = ""
    username: str = ""
    password: str = ""
    use_tls: bool = True


@dataclass
class HooksConfig:
    on_digest: str = ""
    slack_webhook_url: str = ""
    discord_webhook_url: str = ""
    email: EmailHookConfig = field(default_factory=EmailHookConfig)


@dataclass
class ProfilerConfig:
    scan_source: bool = False
    # Read the README with an LLM and merge the named entities it finds into `anchors`.
    # OFF by default and measured, not assumed: P9 found the channel discriminates where
    # the manifest channel does not, P10 found that survives redacting the spans from the
    # judge's view -- and neither established that it improves a digest. Costs one Haiku
    # call per repository per profile, cached on the README's hash.
    typed_anchors: bool = False
    typed_anchors_model: str = "claude-haiku-4-5"
    # Mirrored from `suggestions`/`privacy` by the loader so `complete()` can be called
    # with this object. Declared here rather than set loose so mypy and `rr audit` both
    # see that the profiler is an LLM destination when typed_anchors is on.
    provider: str = "claude"
    claude_api_key: str = ""
    claude_model: str = "claude-haiku-4-5"
    timeout: int = 30
    redact: list[str] = field(default_factory=list)
    max_files: int = 100
    source_extensions: list[str] = field(
        default_factory=lambda: [".py", ".js", ".ts", ".tsx", ".jsx"]
    )
    # How much of the README to carry on RepoProfile.prose, for prompts that need to know
    # what the repo is FOR. Set to 0 to send no prose at all: on a proprietary codebase the
    # README is a far larger disclosure than a keyword list, and `rr audit` reports which
    # of the two an LLM destination receives. Budgeted here so it is one setting, checkable
    # in one place, rather than a truncation decided at each call site.
    #
    # 300 scored best of four budgets on 602 labelled papers — net@2 +95 against +73 for no
    # prose (+22, 95% CI [+4, +41]) — but it is the ARGMAX OF NOISY ARMS, not a settled
    # optimum: 2000 scores +86 and 6000 scores +89, and 300-vs-2000 is +9 at P = 0.108.
    # Only "some prose beats none" is established; which budget wins is not.
    #
    # Do not read this as "short is better". A prefix is a lottery on document layout —
    # `graph`'s first 300 characters are link badges and its actual description starts
    # after the cut, while chars 300-2000 of `rag` hold ColBERT's late-interaction
    # explanation. 300 wins here partly by luck of how 12 READMEs happen to open.
    # See evals/RESULTS.md -> "how much prose" for the sweep and its limits.
    prose_chars: int = 300

    # WHERE the window starts, which the comment above already calls "a lottery on document
    # layout". `start` is the shipped behaviour: the first `prose_chars` of the cleaned README.
    # `self_description` re-anchors on the repository's own "<name> is a ..." sentence, and only
    # when that sentence falls OUTSIDE the current window — so a README that already opens with
    # its description is untouched.
    #
    # Measured before it was offered (evals/prose_window_probe.py, §41-42): re-anchoring
    # unconditionally moves 17 of 37 benchmark repositories, most by a few characters at 95% word
    # overlap, for nothing. Conditionally it moves 3. `bio-align` is the case it exists for — its
    # window is a phishing warning and a build recipe while "Minimap2 is a versatile sequence
    # alignment program" sits at line 59 — and on nine repositories outside the benchmark it
    # fires once (LAMMPS, correctly) and harms none.
    #
    # UNVALIDATED as a digest improvement. §42 is the arm; until it reports, `start` ships.
    prose_anchor: str = "start"  # "start" | "self_description"


@dataclass
class SuggestionsConfig:
    provider: str = "template"  # "template" | "ollama" | "claude"
    ollama_model: str = "llama3.2"
    ollama_url: str = "http://localhost:11434"
    claude_api_key: str = ""
    claude_model: str = "claude-haiku-4-5"
    max_suggestions: int = 3
    timeout: int = 30
    # Not user-set under `suggestions:` — populated from `privacy.redact` at load
    # time so `complete()` can strip the terms from any prompt it is about to send.
    redact: list[str] = field(default_factory=list)


@dataclass
class HydeConfig:
    """Dense-index discovery: search by the abstract of the paper this repo wishes existed.

    Off by default because it needs a one-time ~1.1 GB local footprint (a 432 MB index sync
    plus ~670 MB of embedding weights). What it buys is the channel keyword search cannot
    be: measured 27 of 48 benchmark targets, **15 of them unreachable by any other channel**
    including every repository with no arXiv bibliography. See reporadar/hyde.py.
    """

    enabled: bool = False
    index_dir: str = "~/.cache/reporadar/hyde-index"
    model: str = "mixedbread-ai/mxbai-embed-large-v1"
    # Four diverse guesses measured far better than one (42/48 vs 23/48 reachable at an
    # equal candidate budget) — the cost is one LLM call either way.
    n_hypotheses: int = 4
    top_k: int = 100  # candidates per hypothesis; the union feeds the normal ranker
    # Reproduce stored vectors bit-for-bit before searching. Leave this on: a mismatched
    # encoder returns confident nonsense and nothing downstream can detect it.
    verify_encoder: bool = True
    # Warn when the synced index is older than this. The index is a third-party mirror,
    # republished periodically; a stale one silently stops seeing recent work.
    stale_after_days: int = 60


@dataclass
class FinescaleConfig:
    """Second-stage rescore that orders the gate's near-binary score-2 band.

    Off by default because it is the one feature that needs a **different vendor**:
    reading a score distribution requires logprobs, which only the OpenAI API exposes
    (see reporadar/finescale.py). With it off nothing changes; with it on, papers
    sitting exactly at `min_actionable` must also clear `threshold` to be a Top Pick.
    """

    enabled: bool = False
    openai_api_key: str = ""  # falls back to $OPENAI_API_KEY
    openai_model: str = "gpt-4o-mini"
    timeout: int = 30
    # P(actionable) a band paper must clear. The default is not a tuning knob: net@2
    # values a shown paper at 3p-2, so showing pays exactly above p = 2/3. Raising it
    # trades recall for precision; lowering it below 2/3 shows papers with negative
    # expected value.
    threshold: float = 0.6666666666666666
    # Skip the gate (and warn) when fewer than this fraction of attempted calls
    # succeeded, so a broken key cannot quietly empty the digest's Top Picks.
    min_success_fraction: float = 0.5
    # Populated from `privacy.redact` at load time, like SuggestionsConfig.
    redact: list[str] = field(default_factory=list)


@dataclass
class TriageConfig:
    # LLM actionability triage. Uses the `suggestions` provider/model for the LLM
    # call, so it only runs when suggestions.provider is "ollama" or "claude".
    enabled: bool = False
    # 50, not 15. This shipped at 15 and NO experiment had ever included that value: the
    # closest (NR-15, 2026-08-02) compared windows 20 and 50, so the default was shallower
    # than the shallowest arm ever run — and that comparison was confounded with a prompt
    # change and predates the fine-scale rescore, which falsified the mechanism NR-16 used
    # to close pool depth ("nothing orders what the gate admits").
    #
    # Measured 2026-08-14 on one frozen pool, three arms, gate depth the only variable:
    # 15 -> +2.72, 25 -> +3.16, 50 -> +3.72 mean net@2. Paired 50 vs 15 is +1.00/case
    # (95% CI [+0.12, +1.92]) against a frozen floor of 0.48, and +1.23 (CI [+0.27, +2.27],
    # sign p = 0.035) on the 22 non-control cases. A gate-free measure agrees: actionable
    # papers reaching the returned top-10 go 5.00 -> 5.76 -> 6.52 per case.
    #
    # The cost is ~3.3x the gate calls per run (Haiku, and the fine-scale band grows with
    # it), which is still two orders of magnitude below the agentic baseline this system is
    # compared against. Depth 25 is INSIDE the floor and therefore unresolved, not equal.
    top_k: int = 50  # how many top-ranked papers to triage per run
    min_actionable: int = 2  # llm_score >= this qualifies as a Top Pick
    # Reorder papers by llm_score before the Top-N digest window, so an actionable
    # paper the heuristic ranker buried isn't cut off before it can be a Top Pick.
    rerank: bool = True
    finescale: FinescaleConfig = field(default_factory=FinescaleConfig)


@dataclass
class RecommendationsConfig:
    # Learned recommendations from the free Semantic Scholar Recommendations API,
    # seeded by your starred/highly-rated papers (and suppressed by low-rated ones).
    # Off by default: it only helps once you have ratings/stars.
    enabled: bool = False
    limit: int = 20  # how many recommendations to request per run
    max_seeds: int = 50  # cap on positive/negative example papers sent


@dataclass
class FeedbackConfig:
    enabled: bool = False
    min_ratings: int = 10
    learning_rate: float = 0.1


@dataclass
class RepoRadarConfig:
    repo_path: str = "."
    sources: list[str] = field(default_factory=lambda: ["arxiv"])
    arxiv: ArxivConfig = field(default_factory=ArxivConfig)
    queries: QueriesConfig = field(default_factory=QueriesConfig)
    ranking: RankingConfig = field(default_factory=RankingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    semantic_scholar: SemanticScholarConfig = field(default_factory=SemanticScholarConfig)
    openalex: OpenAlexConfig = field(default_factory=OpenAlexConfig)
    enrichment: EnrichmentConfig = field(default_factory=EnrichmentConfig)
    hooks: HooksConfig = field(default_factory=HooksConfig)
    profiler: ProfilerConfig = field(default_factory=ProfilerConfig)
    suggestions: SuggestionsConfig = field(default_factory=SuggestionsConfig)
    triage: TriageConfig = field(default_factory=TriageConfig)
    feedback: FeedbackConfig = field(default_factory=FeedbackConfig)
    recommendations: RecommendationsConfig = field(default_factory=RecommendationsConfig)
    signals: SignalsConfig = field(default_factory=SignalsConfig)
    privacy: PrivacyConfig = field(default_factory=PrivacyConfig)
    hyde: HydeConfig = field(default_factory=HydeConfig)


# The scalar kinds a config leaf can declare. Lists, dicts and nested sections pass
# through untouched.
_SCALARS: tuple[type, ...] = (bool, int, float, str)

# Section name -> its dataclass, read off `RepoRadarConfig`'s own fields rather than
# hand-listed. A hand-listed map checks whatever somebody remembered, which is the C-16
# shape this project has already paid for once — and it would go stale the first time a
# section was added.
_SECTION_TYPES: dict[str, type] = {}

# Leaves that accept a value the type check would otherwise reject, because they define
# their own meaning for it. Keyed "section.key" and applied before the check.
_NORMALIZERS: dict[str, Any] = {"enrichment.provider": _normalize_off}


def _coerce_scalar(section: str, key: str, value: Any, expected: Any) -> Any:
    """One config leaf, checked against its declared type before it reaches a dataclass.

    Two YAML footguns motivate this, and both have already fired here:

    * **A quoted number** (`w_embedding: "1.5"`) parses as a string, and the string then
      flows all the way to `validate_config`'s bounds check, which raises `TypeError:
      '<' not supported between instances of 'str' and 'int'` — from *every* command that
      loads a config, with a traceback and no mention of which field is at fault. Coerced
      here, because a quoted number has exactly one possible meaning.
    * **An unquoted `off`/`on`/`yes`/`no`** parses as a YAML 1.1 *boolean*, so
      `enrichment: provider: off` sets `provider = False`, which is not `"off"` and does
      not disable enrichment. That one shipped: `tests/conftest.py`'s network guard exists
      partly because two tests ran live enrichment while claiming to have it off. There is
      no safe coercion — `False` may have meant `"off"` or `"no"` — so it is refused, by
      name, with the fix in the message.

    Refusing rather than defaulting follows the same rule as `queries.bigrams` and
    `ranking.absent_category` above: a bad value that quietly becomes the default reads
    afterwards as a measurement of the value that was asked for.
    """
    if expected is None or not isinstance(expected, type) or expected not in _SCALARS:
        return value  # lists, dicts, unions, anything unannotated: unchanged
    if isinstance(value, expected) and not (expected is not bool and isinstance(value, bool)):
        return value

    # bool is a subclass of int, so `hybrid: 1` would pass an `isinstance(value, int)`
    # check for an int field and `w_keyword: true` would pass for a float. Caught above.
    if isinstance(value, bool):
        hint = (
            " If you wrote an unquoted off/on/yes/no, YAML 1.1 parses those as booleans — "
            f'quote it: `{key}: "off"`.'
            if expected is str
            else ""
        )
        raise ValueError(
            f"{section}.{key} expects {expected.__name__}, got the boolean {value!r}.{hint}"
        )
    if expected is bool and isinstance(value, str):
        # NOT `bool(value)`: that is truthiness, so `enabled: "false"` would arrive as
        # True — a quoted off-switch silently turning the stage ON, which is worse than
        # the crash this function exists to prevent. Only literals, and only these.
        literal = {"true": True, "false": False, "yes": True, "no": False}.get(
            value.strip().lower()
        )
        if literal is None:
            raise ValueError(
                f"{section}.{key} expects a boolean, got the string {value!r}. "
                f"Use `true` or `false`, unquoted."
            )
        return literal
    if expected is not str and isinstance(value, str):
        try:
            coerced = expected(value.strip())
        except (TypeError, ValueError):
            raise ValueError(
                f"{section}.{key} expects {expected.__name__}, got the string {value!r}, "
                f"which is not a number."
            ) from None
        return coerced
    if expected is float and isinstance(value, int):
        return float(value)
    raise ValueError(
        f"{section}.{key} expects {expected.__name__}, got {type(value).__name__} {value!r}."
    )


def _coerce_section(section: str, raw: Any, cls: type) -> Any:
    """Type-check one config section against its dataclass. Unknown keys pass through so
    the dataclass raises its own `unexpected keyword argument` error, which already names
    the key."""
    if not isinstance(raw, dict):
        raise ValueError(f"{section}: expects a mapping, got {type(raw).__name__} {raw!r}.")
    from dataclasses import fields as dataclass_fields

    declared = {f.name: f.type for f in dataclass_fields(cls)}
    out = {}
    for key, value in raw.items():
        # Leaves that define their own meaning for a YAML boolean run their normaliser
        # FIRST, so the type check sees what the field actually ends up holding. Without
        # this, the check below would reject the documented `enrichment: provider: off` —
        # the exact form `_normalize_off` exists to support.
        normalise = _NORMALIZERS.get(f"{section}.{key}")
        if normalise is not None:
            value = normalise(value)
        expected = declared.get(key)
        # Annotations are strings under `from __future__ import annotations`, so resolve
        # the scalar names directly rather than dragging in `get_type_hints` and its
        # forward-reference machinery for four cases.
        resolved = {"bool": bool, "int": int, "float": float, "str": str}.get(
            expected if isinstance(expected, str) else ""
        )
        out[key] = _coerce_scalar(section, key, value, resolved)
    return out


def _section_types() -> dict[str, type]:
    """Every `RepoRadarConfig` field whose default_factory builds a config dataclass."""
    from dataclasses import MISSING
    from dataclasses import fields as dataclass_fields

    out: dict[str, type] = {}
    for f in dataclass_fields(RepoRadarConfig):
        factory = f.default_factory
        if factory is not MISSING and isinstance(factory, type) and is_dataclass(factory):
            out[f.name] = factory
    return out


def _dict_to_config(data: dict[str, Any]) -> RepoRadarConfig:
    """Build a RepoRadarConfig from a raw dict (parsed YAML)."""
    if not _SECTION_TYPES:
        _SECTION_TYPES.update(_section_types())
    for section, cls in _SECTION_TYPES.items():
        if section in data:
            data = {**data, section: _coerce_section(section, data[section], cls)}
    # `triage.finescale` is the one nested section; the loop above only reaches top level.
    if isinstance(data.get("triage"), dict) and "finescale" in data["triage"]:
        triage_raw = dict(data["triage"])
        triage_raw["finescale"] = _coerce_section(
            "triage.finescale", triage_raw["finescale"], FinescaleConfig
        )
        data = {**data, "triage": triage_raw}

    arxiv = ArxivConfig(**data["arxiv"]) if "arxiv" in data else ArxivConfig()
    queries = QueriesConfig(**data["queries"]) if "queries" in data else QueriesConfig()
    # Fail on an unknown mode rather than silently falling back to `adjacent` — a typo that
    # quietly restores the default would look like a measurement of the mode asked for. Same
    # reasoning as the unknown-source guard in evals/harness.py, which exists because
    # `--sources arxiv,dblp` once ran as arXiv-only and was written up as a DBLP result.
    if queries.bigrams not in BIGRAM_MODES:
        raise ValueError(
            f"queries.bigrams must be one of {', '.join(BIGRAM_MODES)}; got {queries.bigrams!r}"
        )
    if "ranking" in data:
        ranking_data = dict(data["ranking"])
        cat_weights = ranking_data.pop("category_weights", {})
        ranking = RankingConfig(**ranking_data, category_weights=cat_weights or {})
    else:
        ranking = RankingConfig()
    # Same reasoning as queries.bigrams above: an unrecognised mode must not fall back to
    # the default, or a typo reads as a measurement of the policy that was asked for.
    if ranking.absent_category not in ABSENT_CATEGORY_MODES:
        raise ValueError(
            f"ranking.absent_category must be one of {', '.join(ABSENT_CATEGORY_MODES)}; "
            f"got {ranking.absent_category!r}"
        )
    output = OutputConfig(**data["output"]) if "output" in data else OutputConfig()
    semantic_scholar = (
        SemanticScholarConfig(**data["semantic_scholar"])
        if "semantic_scholar" in data
        else SemanticScholarConfig()
    )
    openalex = OpenAlexConfig(**data["openalex"]) if "openalex" in data else OpenAlexConfig()
    enrichment = (
        EnrichmentConfig(**data["enrichment"]) if "enrichment" in data else EnrichmentConfig()
    )
    enrichment.provider = _normalize_off(enrichment.provider)
    signals = SignalsConfig(**data["signals"]) if "signals" in data else SignalsConfig()
    hyde = HydeConfig(**data["hyde"]) if "hyde" in data else HydeConfig()
    privacy = PrivacyConfig(**data["privacy"]) if "privacy" in data else PrivacyConfig()
    sources = data.get("sources", ["arxiv"])

    if "hooks" in data:
        hooks_data = dict(data["hooks"])
        email_data = hooks_data.pop("email", {})
        email_hook = EmailHookConfig(**email_data) if email_data else EmailHookConfig()
        hooks = HooksConfig(**hooks_data, email=email_hook)
    else:
        hooks = HooksConfig()

    profiler = ProfilerConfig(**data["profiler"]) if "profiler" in data else ProfilerConfig()
    suggestions = (
        SuggestionsConfig(**data["suggestions"]) if "suggestions" in data else SuggestionsConfig()
    )
    # Mirror the redaction list onto the two config objects that reach a choke point,
    # rather than threading it through every call site. Both LLM features reach the
    # network via `complete(prompt, suggestions_cfg)`, and every text-search source
    # via `build_queries(..., queries_cfg, ...)`, so putting the terms *there* means
    # redaction applies by construction — a new caller gets it without opting in.
    suggestions.redact = list(privacy.redact)
    queries.redact = list(privacy.redact)
    # Same reasoning for typed anchors: `profile_repo` receives only `cfg.profiler`, and
    # every one of its ten call sites passes exactly that. Mirroring the LLM credentials
    # here means the profiler can reach `complete()` without a signature change rippling
    # through all of them, and redaction still applies because it is copied too.
    profiler.provider = "claude"
    profiler.claude_api_key = suggestions.claude_api_key
    profiler.claude_model = profiler.typed_anchors_model
    profiler.timeout = suggestions.timeout
    profiler.redact = list(privacy.redact)
    if "triage" in data:
        triage_data = dict(data["triage"])
        finescale_data = triage_data.pop("finescale", {})
        triage = TriageConfig(
            **triage_data,
            finescale=FinescaleConfig(**finescale_data) if finescale_data else FinescaleConfig(),
        )
    else:
        triage = TriageConfig()
    # Same choke-point reasoning as `suggestions`: the fine-scale stage reaches the
    # network with a prompt containing the repo profile, so the redaction list has to
    # travel with its config or a new call site would ship un-redacted terms.
    triage.finescale.redact = list(privacy.redact)
    feedback = FeedbackConfig(**data["feedback"]) if "feedback" in data else FeedbackConfig()
    recommendations = (
        RecommendationsConfig(**data["recommendations"])
        if "recommendations" in data
        else RecommendationsConfig()
    )

    return RepoRadarConfig(
        repo_path=data.get("repo_path", "."),
        sources=sources,
        arxiv=arxiv,
        queries=queries,
        ranking=ranking,
        output=output,
        semantic_scholar=semantic_scholar,
        openalex=openalex,
        enrichment=enrichment,
        hooks=hooks,
        profiler=profiler,
        suggestions=suggestions,
        triage=triage,
        feedback=feedback,
        recommendations=recommendations,
        signals=signals,
        privacy=privacy,
        hyde=hyde,
    )


def load_config(config_path: str | Path | None = None) -> RepoRadarConfig:
    """Load config from a YAML file.

    If *config_path* is None, looks for .reporadar.yml in the current directory.
    Raises FileNotFoundError if the file doesn't exist.
    """
    if config_path is None:
        config_path = Path(os.getcwd()) / DEFAULT_CONFIG_NAME

    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    data = _expand_env(data)
    return _dict_to_config(data)


KNOWN_ARXIV_PREFIXES = frozenset(
    {
        "astro-ph",
        "cond-mat",
        "cs",
        "econ",
        "eess",
        "gr-qc",
        "hep-ex",
        "hep-lat",
        "hep-ph",
        "hep-th",
        "math",
        "math-ph",
        "nlin",
        "nucl-ex",
        "nucl-th",
        "physics",
        "q-bio",
        "q-fin",
        "quant-ph",
        "stat",
    }
)


def validate_config(cfg: RepoRadarConfig) -> list[str]:
    """Return a list of warning messages for suspicious config values."""
    warnings: list[str] = []

    # Check sources
    known_sources = {
        "arxiv",
        "semantic_scholar",
        "openalex",
        "biorxiv",
        "europepmc",
        "dblp",
        "iacr",
    }
    for src in cfg.sources:
        if src not in known_sources:
            known = ", ".join(sorted(known_sources))
            warnings.append(f"Unknown source: {src!r}. Known sources: {known}")

    # OpenAlex now requires an API key for its full free allowance; warn if the
    # source is enabled without one (keyless callers get throttled to ~$0.10/day).
    if "openalex" in cfg.sources and not cfg.openalex.api_key:
        warnings.append(
            "openalex source enabled without an api_key. Since 2026-02-13 OpenAlex "
            "throttles keyless requests to a small daily test allowance. Get a free "
            "key at https://openalex.org/ and set openalex.api_key in .reporadar.yml."
        )

    # `biorxiv` is not a keyword search and cannot be made into one: bioRxiv's `details`
    # endpoint takes a date interval, so the adapter fetches a window and filters locally.
    # Under an all-time `lookback_days` that window opens in 2013 and the 40-page cap stops
    # it there, which is why enabling it does not add recent biology. `europepmc` searches
    # the same two servers by keyword. Not an error — someone wanting a category listing
    # still has a use for it — so this warns and does not fail.
    if "biorxiv" in cfg.sources and "europepmc" not in cfg.sources:
        warnings.append(
            "biorxiv source enabled without europepmc. bioRxiv's API is a date-interval "
            "listing, not a keyword search, so this fetches a window and filters locally — "
            f"with lookback_days: {cfg.arxiv.lookback_days} it returns the oldest postings in "
            "that window, not papers about your repository. Add 'europepmc' to sources for a "
            "keyword search over bioRxiv and medRxiv."
        )

    # Check arXiv category prefixes
    for cat in cfg.arxiv.categories:
        prefix = cat.split(".")[0]
        if prefix not in KNOWN_ARXIV_PREFIXES:
            warnings.append(f"Unknown arXiv category prefix: {prefix!r} (in {cat!r})")

    # Numeric bounds
    if cfg.arxiv.max_results_per_query < 1 or cfg.arxiv.max_results_per_query > 500:
        warnings.append(
            f"max_results_per_query={cfg.arxiv.max_results_per_query} is outside range [1, 500]"
        )

    if cfg.arxiv.lookback_days < 1:
        warnings.append(f"lookback_days={cfg.arxiv.lookback_days} should be >= 1")

    # Negative ranking weights
    for name in (
        "w_keyword",
        "w_category",
        "w_recency",
        "w_embedding",
        "w_citations",
        "w_citation_proximity",
        "w_specter",
        "w_community",
        "w_attention",
    ):
        val = getattr(cfg.ranking, name)
        if val < 0:
            warnings.append(f"Negative ranking weight: {name}={val}")

    # The withdrawal penalty is a multiplier, so out-of-range values silently invert
    # the feature: > 1 would *promote* withdrawn papers, < 0 would flip the sign.
    if not 0.0 <= cfg.ranking.withdrawn_penalty <= 1.0:
        warnings.append(
            f"ranking.withdrawn_penalty={cfg.ranking.withdrawn_penalty} should be "
            "between 0 and 1 (it multiplies a withdrawn paper's score; 1 disables "
            "the penalty)"
        )

    if cfg.ranking.w_attention > 0 and not cfg.signals.hackernews:
        warnings.append(
            "ranking.w_attention > 0 but signals.hackernews is false, so no "
            "attention data is ever fetched and the weight does nothing."
        )

    # Category weights
    for cat, weight in cfg.ranking.category_weights.items():
        if weight < 0:
            warnings.append(f"Negative category weight: {cat}={weight}")

    # top_n
    if cfg.output.top_n < 1:
        warnings.append(f"top_n={cfg.output.top_n} should be >= 1")

    # Hooks
    if cfg.hooks.email.smtp_port < 1 or cfg.hooks.email.smtp_port > 65535:
        warnings.append(
            f"hooks.email.smtp_port={cfg.hooks.email.smtp_port} is outside range [1, 65535]"
        )

    # Profiler
    if cfg.profiler.max_files < 1:
        warnings.append(f"profiler.max_files={cfg.profiler.max_files} should be >= 1")

    # Enrichment
    known_enrichment = {"huggingface", "off"}
    if cfg.enrichment.provider not in known_enrichment:
        warnings.append(
            f"Unknown enrichment provider: {cfg.enrichment.provider!r}. "
            f"Known: {', '.join(sorted(known_enrichment))}"
        )

    # Suggestions
    known_providers = {"template", "ollama", "claude"}
    if cfg.suggestions.provider not in known_providers:
        warnings.append(
            f"Unknown suggestions provider: {cfg.suggestions.provider!r}. "
            f"Known: {', '.join(sorted(known_providers))}"
        )
    if cfg.suggestions.max_suggestions < 1:
        warnings.append(
            f"suggestions.max_suggestions={cfg.suggestions.max_suggestions} should be >= 1"
        )
    if cfg.suggestions.timeout < 1:
        warnings.append(f"suggestions.timeout={cfg.suggestions.timeout} should be >= 1")

    # Triage
    if cfg.triage.enabled and cfg.suggestions.provider not in ("ollama", "claude"):
        warnings.append(
            "triage.enabled is true but suggestions.provider is "
            f"{cfg.suggestions.provider!r} — triage needs an LLM provider "
            "(set suggestions.provider to 'ollama' or 'claude'); it will be skipped."
        )
    if cfg.triage.top_k < 1:
        warnings.append(f"triage.top_k={cfg.triage.top_k} should be >= 1")
    if cfg.triage.min_actionable not in (1, 2, 3):
        warnings.append(f"triage.min_actionable={cfg.triage.min_actionable} should be 1, 2, or 3")

    # Recommendations
    if cfg.recommendations.limit < 1:
        warnings.append(f"recommendations.limit={cfg.recommendations.limit} should be >= 1")
    if cfg.recommendations.max_seeds < 1:
        warnings.append(f"recommendations.max_seeds={cfg.recommendations.max_seeds} should be >= 1")

    # Feedback
    if cfg.feedback.min_ratings < 1:
        warnings.append(f"feedback.min_ratings={cfg.feedback.min_ratings} should be >= 1")
    if not (0 < cfg.feedback.learning_rate <= 1.0):
        warnings.append(
            f"feedback.learning_rate={cfg.feedback.learning_rate} should be in (0, 1.0]"
        )

    return warnings


def default_config_yaml() -> str:
    """Return the default .reporadar.yml content as a string."""
    return """\
repo_path: .

sources: [arxiv]
# arXiv carries no biology. A repository whose literature is on bioRxiv or medRxiv
# wants `sources: [arxiv, europepmc]`. Measured 2026-08-21 over six bio repositories:
# it supplied 4.8 of the ~9.7 papers shown per run, at a precision indistinguishable from
# the arXiv papers beside them under two independent judges. It COMPETES for those slots —
# it displaced 44% of what the arXiv-only run showed — and whether it raises net@2 is
# unresolved. `rr profile` suggests it when your dependencies look biological.

arxiv:
  categories: [cs.LG, cs.CL]
  max_results_per_query: 50
  # All-time, relevance-first: the configuration every benchmark number was measured
  # under. Set lookback_days to 14 and sort_by to "submitted" for a strict recency
  # digest — untested, and structurally unable to reach papers older than the window.
  lookback_days: 36500
  sort_by: relevance

queries:
  seed: []
  exclude: []

ranking:
  w_keyword: 1.0
  w_category: 0.5
  w_recency: 0.0
  # Unmeasured as a ranking weight, in either direction: every benchmark number in this
  # project was measured at 0.0, and the one experiment that touched this channel scored
  # README embeddings as a *query*, not as a weight. It also does nothing unless the
  # `embeddings` extra is installed, so the same config ranks two ways. Left as-is rather
  # than swapped for another unmeasured value; see evals/audit_product_divergence.py.
  w_embedding: 1.5

output:
  digest_path: ./reporadar_digest.md
  top_n: 15

# ---------------------------------------------------------------------------
# THIS CONFIGURATION IS THE WEAK ONE, AND WE MEASURED HOW WEAK.
#
# It runs no LLM stage, so papers are ordered by keyword overlap alone. That is the
# configuration the benchmark scores at mean net@2 -8.12 -- i.e. actively worse than
# showing nothing, because the metric charges 2 for each unactionable paper shown.
# It is net-negative on 19 of the 25 repositories. Measured 2026-08-16 on all 25 at
# this width; the measured configuration scores +5.12 on the same 25. It is not enabled
# by default only because each stage needs a credential or a download that a default
# cannot assume, and a tool that fails on first run without a key is worse.
#
# To get the measured configuration:      rr init --measured
# What it needs and what it costs:        README, "The measured configuration"
# ---------------------------------------------------------------------------
"""


def measured_config_yaml() -> str:
    """The configuration every headline number in this project was measured under.

    Written by ``rr init --measured``. This is not a suggestion assembled from taste: it
    is the configuration behind **+5.42 mean net@2 against the Opus 4.8 baseline's +1.62**
    (paired +3.79, 95% CI [+2.17, +5.58], sign p = 0.0001) on the 25-repository benchmark,
    and `evals/audit_product_divergence.py` asserts field-by-field that this file still
    reproduces `BENCHMARK_HEADLINE`. If somebody edits one without the other, that check
    fails — which is the only way a recommended configuration stays true rather than
    becoming a stale copy of one.

    The default from :func:`default_config_yaml` is the *weak* configuration and is
    labelled as such in the file it writes. Both are shipped rather than one, because the
    stages that make up the difference cost money and disk, and choosing to spend those is
    the user's call to make knowingly.
    """
    return """\
# RepoRadar -- the measured configuration.
#
# This is the configuration behind the published result: mean net@2 +5.42 against an
# agentic Claude Opus 4.8 baseline's +1.62 on a 25-repository benchmark (paired +3.79,
# 95% CI [+2.17, +5.58], sign test p = 0.0001). `rr init` without --measured writes a
# keyword-only config that the same benchmark scores at -8.12 (precision 0.379 against
# 0.883 here, and net-negative on 19 of 25 repositories).
#
# WHAT IT NEEDS
#   ANTHROPIC_API_KEY   the actionability gate + HyDE's hypothesis writer (Claude Haiku)
#   OPENAI_API_KEY      the fine-scale rescore. A SECOND vendor is structural, not a
#                       preference: the rescore reads token logprobs and only OpenAI
#                       exposes them. Without it, drop `finescale` and lose ~+1.36.
#   uv pip install -e ".[hyde]"     sentence-transformers + pyarrow
#   rr sync-index                   one time, ~1.1 GB on disk (432 MB index +
#                                   ~670 MB model weights). Offline after that.
#
# WHAT IT COSTS PER `rr update`, PER REPOSITORY
#   gate         ~$0.01   Haiku over `triage.top_k` papers (measured ~$0.02/100)
#   rescore      <$0.01   gpt-4o-mini, one call per paper in the threshold band only
#   HyDE         <$0.01   4 Haiku hypotheses per run; the index search itself is free
#   ------------------------------------------------------------------------------
#   total        ~$0.01-0.02, against ~$0.80/repo for the agentic baseline it beats.
#
# Every value below that carries a measurement cites it. Fields with no citation were
# measured at the value shown but not varied, which is a weaker claim and marked as such.

repo_path: .

sources: [arxiv]
# arXiv carries no biology. A repository whose literature is on bioRxiv or medRxiv
# wants `sources: [arxiv, europepmc]`. Measured 2026-08-21 over six bio repositories:
# it supplied 4.8 of the ~9.7 papers shown per run, at a precision indistinguishable from
# the arXiv papers beside them under two independent judges. It COMPETES for those slots —
# it displaced 44% of what the arXiv-only run showed — and whether it raises net@2 is
# unresolved. `rr profile` suggests it when your dependencies look biological.

arxiv:
  # CHANGE THIS. cs.LG/cs.CL is a guess that fits an ML repository and no other; it is
  # the one field here that no benchmark number justifies, because the benchmark supplies
  # per-repository categories. A wrong category list quietly starves everything downstream.
  categories: [cs.LG, cs.CL]
  max_results_per_query: 50
  # All-time and relevance-first. A 14-day recency window structurally excludes every
  # paper in the gold set (all >= 11 months old) -- NR-5.
  lookback_days: 36500
  sort_by: relevance

queries:
  seed: []
  exclude: []
  # Quoted phrase queries are emitted only when the two words actually co-occur in the
  # repository. Unverified pairs produced `"use page"` and `"data cd"` -- C-10.
  bigrams: verified

ranking:
  w_keyword: 1.0
  w_category: 0.5
  # Recency carries no weight: it is already handled by the all-time discovery above,
  # and weighting it again re-introduces the window NR-5 rules out.
  w_recency: 0.0
  # 1.5, and this line was 0.0 until 2026-08-16 for the stated reason that 1.5 was
  # unmeasured. It is measured now: +1.00 net@2/case over 0.0 across two paired draws on
  # one frozen pool (CI [+0.14, +2.08], sign p = 0.035), and the headline was then re-run
  # here rather than inferred from the delta. REQUIRES the `embeddings` extra --
  # without it the weight is inert and you get the 0.0 configuration, which measured
  # roughly a point per repository worse. `.[hyde]` already pulls it in.
  w_embedding: 1.5
  # BM25 + reciprocal-rank fusion over the heuristic ranking. Present in every headline.
  # Worth keeping ONLY with the gate below: on its own it measured better nDCG and a
  # LOWER headline (NR-11), and the cost was recovered only once the rescore ordered
  # what the gate admits.
  hybrid: true
  # Papers whose arXiv category is absent from the pool are omitted from the category
  # component rather than imputed. Measured +0.00 either way (NR-33); kept because it is
  # what every published number ran.
  absent_category: omit

profiler:
  # Verbatim README prose sent to the gate. An LLM paraphrase of the same text measured
  # BELOW sending no description at all -- paraphrase destroys terms of art (NR-12).
  prose_chars: 300

suggestions:
  # BOTH of these are required for the gate to run. `triage.enabled: true` with
  # provider left at 'template' is a silent no-op that prints one line and gates nothing.
  provider: claude
  claude_api_key: ${ANTHROPIC_API_KEY}
  claude_model: claude-haiku-4-5

triage:
  enabled: true
  # Gate depth. The shipped default was 15 until it was measured: 50 beats it by
  # +1.00 net@2/case, 95% CI [+0.12, +1.92]. Deeper costs proportionally more Haiku.
  top_k: 50
  # A paper is a Top Pick at gate score >= 2. Tightening to 3 was measured to buy
  # nothing and cost recall (P5: the gate's precision is 0.97, its recall 0.60).
  min_actionable: 2
  rerank: true
  finescale:
    enabled: true
    openai_api_key: ${OPENAI_API_KEY}
    openai_model: gpt-4o-mini
    # 60, matching the eval harness rather than the 30-second product default, so that
    # this file reproduces the measured run exactly and the audit needs no exemption.
    timeout: 60
    # DERIVED, not tuned: net@2 values a shown paper at 3p-2, so showing pays exactly
    # above p = 2/3. Moving this moves the metric's own breakeven, not a preference.
    threshold: 0.6666666666666666
    # If fewer than half the band scored, skip the stage rather than let a mostly-failed
    # rescore silently demote the band and pass it off as an abstention.
    min_success_fraction: 0.5

hyde:
  # Dense discovery against a local 3.1M-vector arXiv index. Worth +1.36 mean net@2 end
  # to end. Requires `rr sync-index` first; without the index this raises rather than
  # quietly degrading to the keyword path that reached 0 of 24 targets.
  enabled: true
  n_hypotheses: 4
  top_k: 100

output:
  digest_path: ./reporadar_digest.md
  # 15, not 10. Every number in the paper was measured at 10; widening to the shipped 15
  # is worth +1.24 net@2/case (CI [+0.48, +2.08]) because the papers it adds are 85.5%
  # actionable, well above the 2/3 the metric breaks even at.
  top_n: 15
"""
