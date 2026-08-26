"""The strong baseline: Opus 4.8, in two modes.

- mode="cli"  : Opus 4.8 via Claude Code headless (`claude -p`) with web tools,
                run in the target repo's directory (the agent reads the repo).
- mode="api"  : Opus 4.8 via the Anthropic Messages API with the server-side
                web_search tool. Works without the Claude Code CLI — the repo
                context is passed in the prompt instead of a filesystem mount.

The prompt is **versioned**, not edited. `BASELINE_PROMPT` (v1) is the published comparator:
arXiv-only, and every figure in the paper was measured against it. `BASELINE_PROMPT_V2` opens
the task to journal, conference and bioRxiv papers, which is a *witness-generator* change —
P12 found 79 judged-actionable non-arXiv papers the arXiv-only set is structurally blind to.
The two have separate cache directories, because `_disc` is a staleness check and not an
address: without that, running v2 would have overwritten the v1 answers in place rather than
sitting beside them.

Both parse the recommended papers out of the answer. Every run records a
`status` ("ok" | "missing_cli" | "no_cli_login" | "error" | "timeout" | "no_key")
and its raw output, and **failures are never cached** — so a broken run retries
next time instead of silently poisoning the cache. Caches are keyed by mode, so a
--mock run can never be served to a real run.

The `cli` mode has two ways to authenticate and they are not interchangeable: a
signed-in CLI bills the user's Claude subscription, while a visible
ANTHROPIC_API_KEY bills the API *and*, by the CLI's own warning, disables
connectors — so the agent may not have the same tools under each. `cli_auth_mode`
resolves which one a run uses (default: prefer the subscription when signed in),
and the choice is recorded in the cache entry rather than hashed into its
discriminator, so asking the question later does not invalidate everything.
"""

from __future__ import annotations

import functools
import hashlib
import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any

from verify import extract_arxiv_ids

CACHE_DIR = Path(__file__).resolve().parent / "cache" / "baseline"

# The cli baseline retries transient `claude` failures (and, on the specific
# ANTHROPIC_API_KEY-shadows-claude.ai-login conflict, retries with that key
# stripped from the subprocess so the CLI authenticates via its own login).
_CLI_MAX_RETRIES = 2
_CLI_BACKOFF = 3.0
_CLI_AUTH_CONFLICT_MARKERS = (
    "takes precedence over your claude.ai login",
    "connectors are disabled because ANTHROPIC_API_KEY",
)

# How the `claude` subprocess should authenticate.
#
#   "auto"          -- ask the CLI; if it is signed in, hide ANTHROPIC_API_KEY from the
#                      subprocess so the run bills the subscription. (default)
#   "subscription"  -- always hide the key. Fails loudly when not signed in, which is what
#                      you want for a run you intend to bill to the subscription.
#   "api"           -- always pass the key through, the pre-2026-08-26 behaviour.
#
# This matters beyond billing. The CLI's own warning -- one of the conflict markers above --
# is that a present ANTHROPIC_API_KEY *disables connectors*, so the two auth paths do not
# necessarily give the agent the same tools. Two runs of "the same" baseline under different
# auth are therefore not automatically the same comparator, and `cli_auth_mode` is recorded
# in every cache entry from here on so a future reader can tell which one produced it.
_CLI_AUTH_ENV = "RR_EVAL_CLI_AUTH"
_CLI_AUTH_CHOICES = ("auto", "subscription", "api")

BASELINE_MODEL = "claude-opus-4-8"
# Opus 4.8 pricing ($/1M tokens) for a rough cost estimate (excludes web-search fees).
_PRICE_IN, _PRICE_OUT = 5.0, 25.0

# Headless CLI flags. Configurable because they vary by Claude Code version;
# override the whole list with RR_EVAL_CLAUDE_FLAGS (space-separated) if needed.
# The shipped turn cap. Named so probes can vary it without retyping the flag list, and so
# "what did this run use" is answerable from one place. Every published cache was written
# under this value; changing it changes `_discriminator` and re-runs all of them.
DEFAULT_MAX_TURNS = 12

CLAUDE_FLAGS = [
    "--model",
    BASELINE_MODEL,
    "--output-format",
    "json",
    "--allowedTools",
    "WebSearch,WebFetch",
    "--max-turns",
    str(DEFAULT_MAX_TURNS),
]

BASELINE_PROMPT = (
    "Please fetch and summarize research papers that relate to the code in this "
    "repository and propose methods to improve it. Focus on papers whose methods "
    "could actually be applied to improve this codebase; it is better to recommend "
    "nothing than to recommend papers that are not genuinely useful. "
    "End your response with a fenced ```json code block containing a JSON array of "
    'the papers you recommend, each as {"arxiv_id": "XXXX.XXXXX", "title": "..."}. '
    "Use an empty array [] if you recommend nothing."
)

# v2: the same task, over the whole literature instead of arXiv alone.
#
# The first two sentences and the last are BYTE-IDENTICAL to v1, on purpose. This is meant to
# be a one-variable change -- scope and the identifier schema -- so that if v2 is ever
# promoted to the published comparator, the difference in its score is attributable to what
# it was allowed to search rather than to incidentally better prompting. In particular it
# gains no anti-hallucination coaching that v1 lacks, even though `doi_exists` now makes an
# invented DOI cost something; adding that here would confound the two.
#
# The one sentence that is not pure scope is "Recommend a paper only if you can give one of
# these identifiers", and it earns its place: a pick with no resolvable id is classified
# `unjudgeable` by `verify.resolve_reference`, which means it is dropped without ever being
# scored. Silent disappearance is the failure mode this prompt exists to remove, so the
# schema requirement has to be stated rather than implied. v1 implied it by offering only an
# `arxiv_id` field; v2 offers a choice, so the requirement has to be said out loud.
#
# The field is `id`, not `arxiv_id`. A key called `arxiv_id` holding `10.1038/...` is the
# misnamed-container shape this project has already paid for twice (C-12, C-14): the name
# would have been the documentation, and it would have been false.
BASELINE_PROMPT_V2 = (
    "Please fetch and summarize research papers that relate to the code in this "
    "repository and propose methods to improve it. Focus on papers whose methods "
    "could actually be applied to improve this codebase; it is better to recommend "
    "nothing than to recommend papers that are not genuinely useful. "
    "Papers from anywhere are in scope: arXiv preprints, peer-reviewed journal and "
    "conference papers, and bioRxiv/medRxiv preprints all count equally. "
    "End your response with a fenced ```json code block containing a JSON array of "
    'the papers you recommend, each as {"id": "...", "title": "..."}, where "id" is '
    'the paper\'s arXiv id if it has one (e.g. "2401.12345" or "cond-mat/0403023") '
    'and otherwise its DOI in bare form (e.g. "10.1038/s41586-021-03819-2"). '
    "Recommend a paper only if you can give one of these two identifiers for it. "
    "Use an empty array [] if you recommend nothing."
)

# Selectable rather than swapped. `_cache_path` had no discriminator in it, so editing the
# prompt in place would not invalidate the 34 stored answers -- it would OVERWRITE them, and
# that has happened: `compiler`, `graph` and `storage` still hold a 128-character restoration
# note where their transcript used to be, after a 30-turn re-run displaced the 12-turn entry
# on 2026-08-09. Naming the versions and keying the cache on the name makes the collision
# structurally impossible instead of a thing to remember.
PROMPTS = {"v1": BASELINE_PROMPT, "v2": BASELINE_PROMPT_V2}
# v1 is the published comparator; every figure in the paper was measured against it. Promoting
# v2 is a separate, deliberate decision that requires re-measuring `+1.84 / paired +3.88`.
DEFAULT_PROMPT_VERSION = "v1"


def prompt_for(version: str) -> str:
    """The prompt text for *version*, or a loud failure. Never falls back to v1.

    A typo'd `--prompt-version` that quietly ran v1 would produce a v2-labelled artifact full
    of v1 draws, and nothing downstream could tell: the rows would be well-formed, the picks
    plausible, the arXiv-only distribution unremarkable. That is the void-not-null shape.
    """
    try:
        return PROMPTS[version]
    except KeyError:
        raise ValueError(f"unknown prompt version {version!r}; known: {sorted(PROMPTS)}") from None


# The keys an item may carry its identifier under, in priority order. `arxiv_id` is v1's
# field; `id` is v2's; `doi` is neither, and is accepted because a model told it may answer
# with a DOI is liable to name the field after what it put in it.
#
# Widening this is safe for the stored answers, and that was checked rather than assumed: a
# survey of all 34 caches on 2026-08-26 found 130 recommendation items carrying exactly two
# keys between them, `arxiv_id` and `title`, and nothing else. So no cached run can re-parse
# differently because of the new keys -- which matters, because `run_baseline` replays this
# function over cached `raw` on every hit, and a parser change moves published denominators.
_ID_KEYS = ("arxiv_id", "id", "doi")


def _parse_recommendations(text: str) -> tuple[list[str], list[str]]:
    """Extract (ids, titles) from the baseline's answer.

    The ids are references, not necessarily arXiv ids: under `BASELINE_PROMPT_V2` an entry
    may be a DOI. `verify.resolve_reference` is what decides which registry can answer for
    one, so this function does not classify -- it only refuses to lose them.

    The ```json recommendation block is authoritative: when the baseline emits
    one (even an empty ``[]`` — an explicit abstention), those are its only
    recommendations. Prose is *discussion*, and scraping IDs out of it credits
    the baseline for papers it explicitly reviewed-and-rejected (or, worse, for
    a stray non-arXiv URL path). We only fall back to prose-scraping when the
    baseline emitted no structured block at all.
    """
    ids: list[str] = []
    titles: list[str] = []
    saw_json_block = False

    for block in re.findall(r"```(?:json)?\s*(\[.*?\])\s*```", text, re.DOTALL):
        try:
            items = json.loads(block)
        except json.JSONDecodeError:
            continue
        saw_json_block = True
        for it in items:
            if not isinstance(it, dict):
                continue
            ref = next((str(it[k]).strip() for k in _ID_KEYS if it.get(k)), "")
            if ref:
                ids.append(ref)
            elif it.get("title"):
                titles.append(str(it["title"]))

    if not saw_json_block:
        for rid in extract_arxiv_ids(text):
            if rid not in ids:
                ids.append(rid)

    return ids, titles


def _has_answer_block(text: str) -> bool:
    """Did the model actually answer here, or is this cache holding a note?

    An explicit ``[]`` is an ANSWER -- the abstention `_parse_recommendations` is built to
    respect -- so its absence is what distinguishes "the model recommended nothing" from
    "this `raw` is not a model reply at all". `webdev` is the case that makes the
    distinction load-bearing: it says *"My recommendation is to recommend nothing"*, emits
    ``[]``, and still carries four ids scraped by an older parser from prose. Falling back
    on the mere absence of parsed ids would resurrect exactly those.
    """
    return bool(re.search(r"```(?:json)?\s*\[.*?\]\s*```", text, re.DOTALL))


def _cache_path(mode: str, repo_name: str, prompt_version: str = DEFAULT_PROMPT_VERSION) -> Path:
    """Where a run's answer is stored. **The prompt version is part of the location.**

    `_discriminator` is a staleness check, not an address: a cached entry whose hash no
    longer matches is re-run and the answer is written back to the SAME file. So before this
    parameter existed, a prompt change did not invalidate the 34 stored answers, it destroyed
    them -- three of them for real, on 2026-08-09, and their transcripts are unrecoverable.

    v1 keeps the exact path it has always had (``cache/baseline/cli/ann.json``), so this
    change moves no existing file and needs no migration. Every other version gets a sibling
    directory. That also means the consumers which hardcode ``cache/baseline/cli`` -- the
    gold-set derivation, `diagnose_pool`, `witness_set`, `fill_cli_baseline` -- keep reading
    the comparator only, and cannot absorb v2 picks by accident.
    """
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", repo_name)
    folder = mode if prompt_version == DEFAULT_PROMPT_VERSION else f"{mode}-{prompt_version}"
    return CACHE_DIR / folder / f"{safe}.json"


def _discriminator(
    mode: str,
    repo_context: str,
    flags: list[str] | None,
    prompt_version: str = DEFAULT_PROMPT_VERSION,
) -> str:
    """Hash the inputs that change the baseline's answer, so a model/prompt/flag
    (or, for api mode, repo-context) change invalidates a stale cached run.

    The PROMPT TEXT is hashed, not the version label, so editing `BASELINE_PROMPT_V2` still
    invalidates v2's own caches. Hashing the label would let the text drift underneath a
    stable name -- the version equivalent of a cache key that stops tracking its input.
    """
    parts = [BASELINE_MODEL, prompt_for(prompt_version), mode]
    if mode == "api":
        parts.append(repo_context)
    elif mode == "cli":
        parts.append(" ".join(flags or CLAUDE_FLAGS))
        parts.append(os.environ.get("RR_EVAL_CLAUDE_FLAGS", ""))
    return hashlib.sha256("\0".join(parts).encode()).hexdigest()[:12]


def _empty(status: str, raw: str) -> dict[str, Any]:
    return {"ids": [], "titles": [], "raw": raw, "cost_usd": 0.0, "status": status}


# ── mode: mock ─────────────────────────────────────────────────────────────


def _run_mock() -> dict[str, Any]:
    return {
        "ids": [],
        "titles": [],
        "raw": "[mock baseline: no papers]",
        "cost_usd": 0.0,
        "status": "ok",
    }


# ── mode: cli (Claude Code headless) ───────────────────────────────────────


def _parse_cli_payload(stdout: str | None) -> tuple[dict[str, Any] | None, str]:
    """Parse `claude -p --output-format json` stdout into an ok baseline dict.

    Returns (ok_dict, "") on success or (None, reason). `claude` reports internal
    failures (turn-limit, execution errors) via is_error / subtype while still
    exiting 0; a non-JSON stdout means it printed a raw error. All of these are
    failures — never parse them for "recommendations".

    ``stdout`` is typed optional because `subprocess.run` really can hand back
    ``None``: a reader thread that dies decoding leaves the attribute unset, and it
    prints the traceback rather than raising into the caller. Every other failure in
    this module is a `status` a caller can act on, so this one is too — an exception
    here aborts a whole batch, several frames from the actual cause.
    """
    if not stdout:
        return None, "claude produced no readable stdout (decode failure or empty output)"
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError:
        return None, f"non-JSON output from claude: {stdout.strip()[:400]}"
    if payload.get("is_error") or payload.get("subtype") not in (None, "success"):
        return None, (
            f"claude reported failure (subtype={payload.get('subtype')}, "
            f"is_error={payload.get('is_error')})"
        )
    raw_text = payload.get("result", "")
    if not raw_text.strip():
        return None, "claude returned an empty result"
    cost = float(payload.get("total_cost_usd", 0.0) or 0.0)
    ids, titles = _parse_recommendations(raw_text)
    return {
        "ids": ids,
        "titles": titles,
        "raw": raw_text,
        "cost_usd": cost,
        # How many turns the agent actually used. Kept because it is the only direct
        # evidence of whether `--max-turns` was an active constraint or slack headroom:
        # a run that finishes in 8 turns would behave identically at any higher cap, and
        # a `error_max_turns` failure is the same number pressed against the ceiling.
        # Absent from every cache written before 2026-08-26.
        "num_turns": payload.get("num_turns"),
        "status": "ok",
    }, ""


@functools.lru_cache(maxsize=4)
def cli_logged_in(claude_bin: str = "claude") -> bool | None:
    """Is the `claude` CLI signed in to an Anthropic account?

    Returns None when the question cannot be answered (no CLI, unexpected output) so the
    caller can fall back rather than assert something false.

    Asked with ANTHROPIC_API_KEY removed from the child's environment. The question we need
    answered is "is there a login to fall back on if the key is hidden", and hiding the key
    is the only way to ask it that cannot be confounded by the key. Measured 2026-08-26 on
    Claude Code's current build, the answer happens to be identical either way -- with the
    key visible it still reports ``authMethod: "claude.ai"`` -- so this is a precaution
    against a CLI that reports the key as its auth method, not a workaround for one that
    does. Stated as a precaution because that is what the evidence supports.

    Cached because auth does not change mid-run and a 37-case sweep would otherwise spawn 37
    identical probes. Tests that stub the answer must call ``cli_logged_in.cache_clear()``.
    """
    env = os.environ.copy()
    env.pop("ANTHROPIC_API_KEY", None)
    try:
        proc = subprocess.run(
            [claude_bin, "auth", "status"],
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
            env=env,
            stdin=subprocess.DEVNULL,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None
    try:
        return bool(json.loads(proc.stdout or "").get("loggedIn"))
    except (json.JSONDecodeError, AttributeError):
        return None


def cli_auth_mode(claude_bin: str = "claude") -> str:
    """Resolve `RR_EVAL_CLI_AUTH` to "subscription" or "api" for this run."""
    requested = (os.environ.get(_CLI_AUTH_ENV) or "auto").strip().lower()
    if requested not in _CLI_AUTH_CHOICES:
        raise ValueError(f"{_CLI_AUTH_ENV} must be one of {_CLI_AUTH_CHOICES}, got {requested!r}")
    if requested != "auto":
        return requested
    # Signed in -> prefer the subscription. Not signed in (or unknowable) -> the key is the
    # only thing that can work, so leave it in place rather than guaranteeing a failure.
    return "subscription" if cli_logged_in(claude_bin) else "api"


def _run_cli(
    repo_dir: Path,
    *,
    flags: list[str] | None,
    timeout: int,
    prompt_version: str = DEFAULT_PROMPT_VERSION,
) -> dict[str, Any]:
    claude_bin = os.environ.get("RR_EVAL_CLAUDE_BIN", "claude")
    env_flags = os.environ.get("RR_EVAL_CLAUDE_FLAGS")
    flag_list = flags or (env_flags.split() if env_flags else CLAUDE_FLAGS)
    cmd = [claude_bin, "-p", *flag_list, prompt_for(prompt_version)]

    auth = cli_auth_mode(claude_bin)
    if auth == "subscription" and cli_logged_in(claude_bin) is False:
        return _empty(
            "no_cli_login",
            f"{_CLI_AUTH_ENV}=subscription but `{claude_bin} auth status` reports signed out. "
            f"Run `{claude_bin} auth login` (or `{claude_bin} setup-token`), or set "
            f"{_CLI_AUTH_ENV}=api to bill the API key instead.",
        )
    # Hide the key for the whole run under subscription auth, rather than only after the CLI
    # complains: the complaint is a warning the CLI does not always emit, and a run that
    # silently billed the API key is not one you can tell apart afterwards.
    strip_api_key = auth == "subscription"
    last_err = _empty("error", "claude did not produce a result")
    for attempt in range(_CLI_MAX_RETRIES + 1):
        env = os.environ.copy()
        if strip_api_key:
            # The CLI warned ANTHROPIC_API_KEY shadows its claude.ai login and
            # disabled connectors; drop it here so the CLI uses that login. Only
            # the subprocess env is touched — the judge/triage keys are intact.
            env.pop("ANTHROPIC_API_KEY", None)
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(repo_dir),
                capture_output=True,
                # NOT bare text=True. That decodes with the LOCALE codec, which on this
                # project's Windows box is cp1252 — and `claude --output-format json` emits
                # UTF-8, by contract. One undefined byte (0x81, position 1556 of scanpy's
                # answer, 2026-08-25) makes the reader thread raise UnicodeDecodeError,
                # which `subprocess.run` prints and swallows, handing back `stdout=None`.
                # The TypeError then surfaced four frames away in `json.loads`, uncaught,
                # killing an 11-case batch on its first case after it had already been
                # billed for the answer. The encoding is knowable here, so state it; the
                # decode must never be the thing that fails, so replace rather than raise.
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
                env=env,
                # DEVNULL, not inherited. Without this the CLI inherits the parent's stdin
                # and blocks on it forever when there is no terminal — which is every
                # non-interactive context this benchmark actually runs in: nohup, CI, cron.
                # Measured 2026-08-15: a backgrounded 25-case run sat at 0.0 s of CPU for
                # nine minutes on a single thread, having produced nothing. `timeout=` does
                # not save you, because a process blocked on a read it will never satisfy
                # looks identical to one doing slow work — the failure this project keeps
                # paying for, in its process-control form.
                stdin=subprocess.DEVNULL,
            )
        except FileNotFoundError:
            return _empty(
                "missing_cli",
                f"`{claude_bin}` not found on PATH. Install Claude Code, set RR_EVAL_CLAUDE_BIN "
                "to its full path, or use --baseline api.",
            )
        except subprocess.TimeoutExpired:
            last_err = _empty("timeout", f"claude timed out after {timeout}s")
        else:
            if proc.returncode == 0:
                ok, reason = _parse_cli_payload(proc.stdout)
                if ok is not None:
                    return ok
                last_err = _empty("error", reason)
            else:
                stderr = (proc.stderr or "").strip()
                stdout = (proc.stdout or "").strip()
                # Surface stdout too: a refusal or turn-limit error is often printed
                # there while exiting non-zero with empty stderr. Without it a
                # cybersecurity-guardrail decline is indistinguishable from a crash.
                detail = f"stderr: {stderr[:400]}"
                if stdout:
                    detail += f" | stdout: {stdout[:400]}"
                last_err = _empty("error", f"claude exited {proc.returncode}. {detail}")
                if not strip_api_key and any(m in stderr for m in _CLI_AUTH_CONFLICT_MARKERS):
                    strip_api_key = True  # retry without the shadowing key

        if attempt < _CLI_MAX_RETRIES:
            time.sleep(_CLI_BACKOFF * (2**attempt))
    return last_err


# ── mode: api (Anthropic Messages API + web_search) ────────────────────────


def _estimate_cost(usage: Any) -> float:
    tin = getattr(usage, "input_tokens", 0) or 0
    tout = getattr(usage, "output_tokens", 0) or 0
    return (tin / 1_000_000) * _PRICE_IN + (tout / 1_000_000) * _PRICE_OUT


def _run_api(
    repo_context: str,
    *,
    max_uses: int = 8,
    prompt_version: str = DEFAULT_PROMPT_VERSION,
) -> dict[str, Any]:
    try:
        from anthropic import Anthropic
    except ImportError:
        return _empty(
            "error",
            'anthropic package not installed. Run: uv pip install -e ".[evals]"',
        )
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return _empty("no_key", "ANTHROPIC_API_KEY not set (needed for --baseline api).")

    client = Anthropic()
    tools = [{"type": "web_search_20260209", "name": "web_search", "max_uses": max_uses}]
    messages: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": (f"{prompt_for(prompt_version)}\n\n# Repository context\n{repo_context}"),
        }
    ]

    text_parts: list[str] = []
    cost = 0.0
    final_stop: str | None = None
    try:
        for _ in range(6):  # continue across server-tool pause_turn boundaries
            resp = client.messages.create(
                model=BASELINE_MODEL, max_tokens=8000, tools=tools, messages=messages
            )
            cost += _estimate_cost(resp.usage)
            text_parts += [b.text for b in resp.content if b.type == "text"]
            final_stop = resp.stop_reason
            if resp.stop_reason == "pause_turn":
                messages.append({"role": "assistant", "content": resp.content})
                continue
            break
    except Exception as exc:  # noqa: BLE001 — surface any API error as a failed run
        return _empty("error", f"Anthropic API error: {exc}")

    raw = "\n".join(text_parts)
    ids, titles = _parse_recommendations(raw)
    # Only a clean completion is trustworthy. max_tokens truncation, a refusal,
    # or still being paused after the loop cap means the recommendations block
    # was likely never emitted — mark it failed (and don't cache) rather than
    # reporting a false abstention.
    if final_stop != "end_turn":
        return {
            "ids": ids,
            "titles": titles,
            "raw": f"[incomplete: stop_reason={final_stop}]\n{raw}",
            "cost_usd": cost,
            "status": "error",
        }
    return {"ids": ids, "titles": titles, "raw": raw, "cost_usd": cost, "status": "ok"}


# ── dispatcher ─────────────────────────────────────────────────────────────


def run_baseline(
    repo_dir: Path,
    *,
    repo_name: str,
    repo_context: str = "",
    mode: str = "cli",
    mock: bool = False,
    use_cache: bool = True,
    flags: list[str] | None = None,
    timeout: int = 900,
    prompt_version: str = DEFAULT_PROMPT_VERSION,
) -> dict[str, Any]:
    """Run the baseline for one repo. Returns {ids, titles, raw, cost_usd, status}.

    Only successful runs (status == "ok") are cached, and caches are per-mode **and per
    prompt version**, so a mock run can never be served to a real run, a failure always
    retries, and a v2 run cannot displace the v1 answer the published figures rest on.
    """
    effective_mode = "mock" if mock else mode
    prompt_for(prompt_version)  # fail here, not after paying for a run under a typo'd version
    cache_file = _cache_path(effective_mode, repo_name, prompt_version)
    disc = _discriminator(effective_mode, repo_context, flags, prompt_version)

    if use_cache and cache_file.exists():
        cached = json.loads(cache_file.read_text(encoding="utf-8"))
        if cached.get("_disc") == disc:
            # Re-derive refs from the cached `raw` so a parser fix applies to
            # already-cached runs. The model's answer (the expensive artifact)
            # is what we cache; parsing it is cheap and must not be frozen.
            if cached.get("status") == "ok" and cached.get("raw"):
                ids, titles = _parse_recommendations(cached["raw"])
                # ...unless `raw` is not an answer. Three caches (`compiler`, `graph`,
                # `storage`) hold a 128-character restoration note where their transcript
                # used to be, after a 30-turn re-run displaced the 12-turn entry on
                # 2026-08-09 and only the ids could be recovered from the run record.
                # Replaying a note yields nothing, so those cases scored as ABSTENTIONS in
                # every run since -- while `diagnose_pool.actionable_baseline_ids`, reading
                # the same file's `ids` field, counted their seven targets. One cache, two
                # consumers, opposite answers, and the baseline forfeited +0.28 net@2/case
                # of its own picks.
                #
                # Falling back is deliberately NARROW: only when the replay finds nothing
                # AND stored ids exist. An empty `[]` block is a real abstention and must
                # survive as one -- `webdev` says "I recommend nothing" in prose and stores
                # four ids scraped by an older parser, so a wider rule would resurrect
                # exactly the picks the authoritative-block fix was written to discard.
                if ids or titles or not cached.get("ids") or _has_answer_block(cached["raw"]):
                    cached["ids"], cached["titles"] = ids, titles
            return cached
        # else: model/prompt/flags/context changed -> stale, re-run

    if mock:
        out = _run_mock()
    elif mode == "api":
        out = _run_api(repo_context, prompt_version=prompt_version)
    else:
        out = _run_cli(repo_dir, flags=flags, timeout=timeout, prompt_version=prompt_version)

    out["_disc"] = disc
    # Recorded as well as hashed and pathed. `_disc` is 12 hex characters and says nothing to
    # a reader; the 34 caches written before today carry no version field at all, and the
    # answer for those is v1 by construction (it was the only prompt that existed).
    out["prompt_version"] = prompt_version
    if mode == "cli" and not mock:
        # Recorded, not hashed. A present ANTHROPIC_API_KEY changes which tools the CLI
        # offers the agent, so two `cli` runs under different auth are not self-evidently
        # the same comparator — and the caches written before 2026-08-26 do not say which
        # they were. Hashing it into `_disc` would invalidate all of those at once, which
        # is the 2026-08-09 mistake; recording it lets the question be asked instead.
        auth = cli_auth_mode(os.environ.get("RR_EVAL_CLAUDE_BIN", "claude"))
        out["cli_auth_mode"] = auth
        # `total_cost_usd` is reported either way, but under subscription auth it is what
        # the tokens WOULD have cost on the API, not money spent. Summing it across a sweep
        # and calling the total "spend" would be a fabricated number; label it at the source.
        out["cost_basis"] = "api-dollars" if auth == "api" else "subscription-notional"
    # Only cache clean runs — a failure must retry next time, never be served.
    if use_cache and out.get("status") == "ok":
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out
