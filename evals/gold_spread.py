"""How much of the gold set is a property of the draw? k independent redraws. [P17]

Every published recall denominator -- 21/56, 34/56, 43/56 -- divides by a gold set derived
from **one** run of the agentic baseline. P15 measured what a re-run costs at the PICK level:
a fresh draw of the identical configuration disagrees with the stored answer on ~59% of picks
(mean Jaccard 0.41). What that leaves open is the question the denominators actually depend
on, because picks are not targets -- the judge filters them, and it may absorb most of the
churn or none of it.

This runs the baseline **k = 3 more times over the 25 benchmark cases**, at unchanged flags,
and judges every pick. Two different questions come out, and they must not be conflated:

* **Reproducibility of the frozen set.** For each of the 56 frozen targets, in how many of
  the k fresh draws does it reappear? This is the one that prices the published `/56`.
* **Spread among the fresh draws.** Pairwise agreement between draws that share a session,
  an auth path and a date -- the noise floor with the stored cache's confounds removed.

**Pre-registered, before the first call:**

* **Prediction.** Target-level reproducibility should EXCEED the pick-level 0.41, because the
  judge filter is a stable function applied to a noisy input: a draw that finds a different
  paper on the same topic still yields a target if the judge scores it >= 2. If target-level
  agreement is not meaningfully above 0.41, the judge absorbs nothing and the gold set is as
  noisy as the search.
* **Decision rule.** If a fresh draw reproduces **< 2/3** of the frozen targets, the
  denominator is not a stable quantity and every published recall figure needs an interval
  rather than a point. Between 2/3 and 90%, the point figure stands with the spread quoted
  beside it. Above 90%, the draw is not a material source of error.
* **Kill condition.** If more than 3 of 25 cases fail to produce an `ok` baseline in a given
  draw, that draw is incomplete and is reported as such rather than averaged in -- a missing
  case is not a case that found nothing (void, not null, the failure this project keeps
  paying for).

**Nothing touches the shared caches.** Every baseline runs with `use_cache=False`, so the 34
stored answers are neither read nor overwritten. Judge verdicts DO use the shared cache, on
purpose: a verdict is a function of (case, paper, rubric) and is the same object whoever asks
for it, which is also why repeated picks across draws cost nothing.

**Persistence is incremental and resumable.** The artifact is rewritten after every single
row, under a lock, and a re-invocation skips what it already holds. A 75-run job that loses
three hours of subscription usage to one exception is the C-29 shape, and this is a longer
job than the one that taught it.

**Two phases, because only one of them is rate-limited.** Phase A runs the agentic baselines
and touches nothing we throttle -- `run_baseline` shells out to `claude` and parses the reply.
Phase B verifies against arXiv and judges. The first version interleaved them, which made the
whole job look arXiv-bound when seconds of it were: measured 2026-08-26, four runs at
concurrency 4 compressed 711 s of phase-A work into 275 s of wall clock, with phase B still
strictly serial and no throttling. `--concurrency` therefore applies to phase A only, by
construction rather than by convention.

**The turn cap is a parameter, and draws at different caps are different configurations.**
`--max-turns` is recorded per row and `report` refuses to average across caps -- it analyses
the shipped cap and lists the others separately. Draws are discovered from the artifact, not
assumed to be 1..k, so a trial at a different cap cannot be silently omitted from the figures.
None of this touches `cache/baseline/cli/` or `_discriminator`: the published comparator is a
separate question from the witness generator, and this script only ever does the latter.

**The prompt is a parameter too, and it is a stronger one than the cap.** `--prompt-version
v2` draws with `BASELINE_PROMPT_V2`, which allows non-arXiv papers. That is not a redraw of
this probe's configuration — it is a *different searcher*, so it writes its own artifact
(`gold_spread_v2.json`), its draw numbers cannot collide with these, and `report` refuses to
call its overlap with the frozen set "reproducibility" or to apply the pre-registered
decision rule to it. Those draws exist to widen the witness set (P16), which is the
baseline's witness-generator role and owes no validation; the published comparator is a
separate question and `cache/baseline/cli/` is still never touched here.

    uv run python evals/gold_spread.py --dry-run   # $0, the plan and what is already done
    uv run python evals/gold_spread.py             # k draws at the shipped cap, resumable
    uv run python evals/gold_spread.py --max-turns 30 --concurrency 4    # a faster variant
    uv run python evals/gold_spread.py --report    # $0, re-read the artifact
    uv run python evals/gold_spread.py --prompt-version v2 --max-turns 30 --concurrency 4
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

import baseline as baseline_mod  # noqa: E402
import judge as judge_mod  # noqa: E402
from diagnose_pool import ACTIONABLE, JUDGE, _judge_stem  # noqa: E402
from harness import WORK_DIR, assemble_repo_context, clone_repo  # noqa: E402
from rr_mcp_arm import read_call_log  # noqa: E402
from run_judge_eval import load_dotenv  # noqa: E402
from verify import TIER_SET, resolve_references, tiers_grew  # noqa: E402

from reporadar.paper_id import canonical_ref, dedup_id  # noqa: E402

EVALS = Path(__file__).resolve().parent
OUT = EVALS / "gold_spread.json"
GOLD = EVALS / "gold_targets.json"


def out_path(
    prompt_version: str,
    model: str = baseline_mod.DEFAULT_MODEL,
    tools: str = baseline_mod.DEFAULT_TOOLS,
) -> Path:
    """One artifact per prompt version, rather than one artifact with a version column.

    Rows are keyed `{draw}/{case}`, and every analysis in `report` splits that key. A v2 draw
    sharing the file would collide with v1's draw 1 on the key -- so `todo_baseline` would
    find the work already done and run nothing at all, silently. Separate files make a v2
    draw impossible to mistake for a v1 one, which is the same rule `report` already applies
    to turn caps: different configuration, different figures, never averaged.
    """
    parts = []
    if prompt_version != baseline_mod.DEFAULT_PROMPT_VERSION:
        parts.append(prompt_version)
    if model != baseline_mod.DEFAULT_MODEL:
        parts.append(baseline_mod.model_tag(model))
    if tools != baseline_mod.DEFAULT_TOOLS:
        # The same collision this function exists to prevent, one axis further: a `web+rr`
        # draw 1 and a `web` draw 1 share the key `1/rag`, so sharing the file would make
        # `todo_baseline` find the augmented arm's work already done and run nothing —
        # silently, and with the plain arm's numbers labelled as the treatment.
        parts.append(tools.replace("+", "_"))
    return OUT if not parts else EVALS / f"gold_spread_{'_'.join(parts)}.json"


DRAWS = 3
# The cohort every published denominator is over. The scientific cases were added later and
# no published figure includes them; mixing them in would answer a question nobody asked.
COHORT = "benchmark25"
MAX_FAILED_PER_DRAW = 3  # kill condition

# Run-level statuses that mean **the model was never asked**, so the row is not a
# measurement and must not be treated as one. A `throttled` row is our quota running out; a
# `no_cli_login` row is our session; neither says anything about the searcher, the prompt or
# the repository. Both are recorded (never silently dropped) and both are re-attempted.
#
# `error` and `timeout` stay terminal on purpose: those are runs where the agent WAS asked
# and could not finish, which is a fact about the configuration and belongs in the failure
# rate `report` prints. The distinction is `lookup_failed` vs `unjudgeable` one level up.
#
# Learned the expensive way. The 2026-08-27 Opus 5 sweep exhausted the subscription 21 runs
# in; the other 54 rows were recorded as terminal `error` in ~400 ms each, and `report` then
# showed draws 2 and 3 at a 100% failure rate. That reads as "Opus 5 cannot do this" when it
# means "we ran out of credit", and no re-invocation would ever have revisited them.
UNASKED = ("throttled", "no_cli_login")


# Which repositories a sweep covers. Selectable, because the default was a decision that
# stopped being visible: `benchmark25` is the cohort every published denominator is over, and
# filtering the scientific cases out was right for P17, which was pricing exactly those
# denominators. It is not right for a witness set, which has no denominator to protect and
# was simply leaving 12 repositories' worth of certificates unclaimed -- and it is a live
# choice, not an inherited default, for a comparator re-measurement.
#
# The cohort deliberately does NOT enter `out_path`. Unlike the prompt, the model and the
# turn cap, it changes nothing about what any individual row MEANS -- it only selects which
# rows get made. Rows are keyed `{draw}/{case}`, so widening a sweep adds cases rather than
# colliding with them, and the case set is always recoverable from the rows themselves.
COHORTS: dict[str, Any] = {
    "benchmark25": lambda name: not name.startswith(("bio-", "mat-")),
    "scientific": lambda name: name.startswith(("bio-", "mat-")),
    "all": lambda _name: True,
}


def cohort_cases(bench: dict[str, Any], cohort: str = COHORT) -> list[dict[str, Any]]:
    try:
        keep = COHORTS[cohort]
    except KeyError:
        raise ValueError(f"unknown cohort {cohort!r}; known: {sorted(COHORTS)}") from None
    return [c for c in bench["cases"] if c.get("live_repo") and keep(c["name"])]


def _judge_cached(case: str, paper_id: str) -> int | None:
    for verdict in (JUDGE / case).glob(f"{_judge_stem(paper_id)}*.json"):
        return int(json.loads(verdict.read_text(encoding="utf-8"))["score"])
    return None


def mcp_config_for(case_name: str, tools: str) -> tuple[Path, Path]:
    """The seeded RepoRadar server config for *case_name*, or a loud failure.

    Refuses a missing store rather than running without one. `--allowedTools` naming tools
    no server provides is not an error Claude Code reports: the agent simply never sees
    them, answers normally, and the row lands in the `web+rr` artifact having had no
    treatment. A degraded arm that looks healthy is the most expensive failure available
    here, and this is the cheapest place to catch it.
    """
    from rr_mcp_arm import case_db, write_config

    # Read off the TOOLSET, through the mapping `baseline` owns. Deriving it here from
    # a string comparison is how the driver came to accept `--tools web+rrwide` and
    # serve the narrow store anyway: the sweep would have run, every row would have
    # looked normal, and the artifact would have answered a different question.
    wide = baseline_mod.wide_corpus(tools)
    db = case_db(case_name, wide=wide)
    if not db.exists():
        raise SystemExit(
            f"{case_name}: no seeded RepoRadar store at {db}.\n"
            f"Run `uv run python evals/rr_mcp_arm.py --seed"
            f"{' --wide' if wide else ''}` first ($0)."
        )
    # A token per run, so two draws of one case cannot write into the same call log and
    # attribute one run's tool use to the other. Wrong data is worse than none.
    return write_config(case_name, token=uuid.uuid4().hex[:8], wide=wide)


def _turn_flags(
    max_turns: int,
    model: str = baseline_mod.DEFAULT_MODEL,
    effort: str | None = baseline_mod.DEFAULT_EFFORT,
) -> list[str] | None:
    """The shipped flags with this run's cap and model, or None when both are the defaults.

    The substitution now lives in `baseline.flags_for` rather than here: two places editing
    the same flag list is how the probe and the product drift, and the model axis needed the
    identical operation. Returning None for the default pair keeps `_run_cli` on the shipped
    list exactly, so nothing about the published configuration is rebuilt at all.
    """
    if (
        max_turns == baseline_mod.DEFAULT_MAX_TURNS
        and model == baseline_mod.DEFAULT_MODEL
        and effort == baseline_mod.DEFAULT_EFFORT
    ):
        return None
    # Tools are NOT applied here. `run_baseline` owns that axis, because it also owns the
    # cache path and the discriminator that have to move with it -- handing it a flag list
    # that already carried the MCP flags would let the two disagree.
    return baseline_mod.flags_for(max_turns=max_turns, model=model, effort=effort)


def run_baseline_only(
    case: dict[str, Any],
    *,
    max_turns: int,
    prompt_version: str = baseline_mod.DEFAULT_PROMPT_VERSION,
    model: str = baseline_mod.DEFAULT_MODEL,
    effort: str | None = baseline_mod.DEFAULT_EFFORT,
    cohort: str = COHORT,
    tools: str = baseline_mod.DEFAULT_TOOLS,
) -> dict[str, Any]:
    """Phase A: the agentic run alone. **Touches no network service we rate-limit.**

    This is what makes concurrency safe. `run_baseline` shells out to `claude` and parses the
    reply; arXiv and the judge are not involved until phase B. The earlier serial design
    interleaved them, which made the whole job look arXiv-bound when only seconds of it were.
    """
    name = case["name"]
    dest = clone_repo(case["live_repo"], WORK_DIR / name, reuse=True)
    if dest is None:
        return {"status": "clone_failed", "phase": "judged"}
    # Resolved BEFORE the clock starts and before anything is billed: a missing store is a
    # setup mistake, and finding it out after paying for the run is finding it out late.
    mcp_config, call_log = (
        mcp_config_for(name, tools) if tools != baseline_mod.DEFAULT_TOOLS else (None, None)
    )
    started = time.monotonic()
    result = baseline_mod.run_baseline(
        dest,
        repo_name=name,
        repo_context=assemble_repo_context(dest),
        mode="cli",
        use_cache=False,  # never read, never write -- the stored gold set is untouchable here
        flags=_turn_flags(max_turns, model, effort),
        prompt_version=prompt_version,
        model=model,
        effort=effort,
        tools=tools,
        mcp_config=mcp_config,
    )
    elapsed = round(time.monotonic() - started, 1)
    if result.get("status") != "ok":
        # Nothing to judge, so this row is finished; `phase` says so and resume skips it.
        return {
            "status": result.get("status", "error"),
            "phase": "judged",
            "raw": (result.get("raw") or "")[:160],
            "max_turns": max_turns,
            "prompt_version": prompt_version,
            "model": model,
            "effort": effort,
            "tools": tools,
            **({"mcp": read_call_log(call_log)} if call_log is not None else {}),
            "cohort": cohort,
            "duration_s": elapsed,
        }
    return {
        "status": "baseline_ok",
        "phase": "baseline",
        # `canonical_ref`, not `dedup_id`: under v2 a pick can be a DOI, and the resolver
        # hands the same paper back in the prefixed `doi:` form. Two spellings of one id
        # would break this artifact's own `targets <= picks` invariant for every non-arXiv
        # paper. For an arXiv id the two functions are identical, so no stored row moves.
        "picks": [canonical_ref(i) for i in result.get("ids") or []],
        "raw_ids": list(result.get("ids") or []),
        "raw_titles": list(result.get("titles") or []),
        "num_turns": result.get("num_turns"),
        # What the agent did with RepoRadar, read off the server's own log rather than
        # inferred from the answer. Without it a null result is unreadable: "the tool did
        # not help" and "the agent never found the tool" are opposite findings.
        **({"mcp": read_call_log(call_log)} if call_log is not None else {}),
        # WHICH server this row was served by. `run_baseline` records it and this function
        # was dropping it, so an artifact could carry `tools: "web+rrwide"` on every row
        # with no way to check that the wide store was ever opened -- which is exactly the
        # failure an audit found in the driver a few hours earlier, in its other half.
        **({"mcp_config": str(mcp_config)} if mcp_config is not None else {}),
        "cost_usd": result.get("cost_usd", 0.0),
        "max_turns": max_turns,
        "prompt_version": prompt_version,
        "model": model,
        "effort": effort,
        "tools": tools,
        "cohort": cohort,
        "duration_s": elapsed,
    }


def judge_row(case_name: str, row: dict[str, Any], *, model: str) -> dict[str, Any]:
    """Phase B: verify against arXiv and judge. **Serial, because both are rate-limited.**"""
    dest = WORK_DIR / case_name
    context = assemble_repo_context(dest)
    papers, hallucinated, lookup_failed, unjudgeable = resolve_references(
        row.get("raw_ids") or [], row.get("raw_titles") or []
    )
    scores: dict[str, int] = {}
    judge_failed = 0
    for paper in papers:
        pid = canonical_ref(paper["arxiv_id"])
        try:
            scores[pid] = int(
                judge_mod.judge_paper(case_name, context, paper, model=model)["score"]
            )
        except Exception as exc:  # noqa: BLE001 -- an unjudged paper is never scored 0
            judge_failed += 1
            print(f"        ! judge failed for {pid}: {str(exc)[:100]}")
    return {
        **{k: v for k, v in row.items() if k not in ("raw_ids", "raw_titles")},
        # `partial` when something could not be verified or judged: the draw's picks are
        # real, but its TARGET set is a floor rather than a count.
        "status": "partial" if (lookup_failed or judge_failed) else "ok",
        "phase": "judged",
        "targets": sorted(p for p, s in scores.items() if s >= ACTIONABLE),
        "scores": dict(sorted(scores.items())),
        "n_hallucinated": hallucinated,
        "n_lookup_failed": lookup_failed,
        # Not part of the `partial` condition above: an existing-but-abstractless DOI never
        # resolves, so retrying would strand the row forever (C-30).
        "n_unjudgeable": unjudgeable,
        "n_judge_failed": judge_failed,
        # Which sources produced the verdicts above. An `unjudgeable` means "none of these
        # had an abstract", which stops being true the moment the list grows; recorded so
        # `retryable` can tell, rather than inferred from the date the row was written.
        "tier_set": list(TIER_SET),
    }


def unscored_picks(row: dict[str, Any]) -> list[str]:
    """Picks a judged row never scored — its retry set, recovered from `picks`.

    `judge_row` drops `raw_ids` once it has judged, so for a while a failed lookup left a
    COUNT and no identity. It turns out nothing was lost: `picks` holds the same references
    already canonicalised, and `resolve_reference` takes either scheme, so the difference
    against `scores` is exactly what could not be resolved or judged.
    """
    scored = row.get("scores") or {}
    return [p for p in row.get("picks") or [] if p not in scored]


def retryable(row: dict[str, Any]) -> bool:
    """Is this partial row worth another network call?

    The four outcomes `verify` classifies exist precisely so this question has an answer.
    A `lookup_failed` is OUR infrastructure failing — transient, and retrying is how the row
    stops being a floor. An `unjudgeable` is a real paper no source carries an abstract for:
    permanent, so retrying it forever is the C-30 trap run in reverse, a row that can never
    be finished being asked again on every invocation.

    So the retry condition reads `n_lookup_failed`, never `status` alone. The 2026-08-26 v2
    sweep is why this exists: 19 of its 22 lookup failures landed in draw 1, inside one
    window of serial judging, and every one of them was recoverable — while 41 unjudgeable
    references in the same sweep were not, given the sources we had.

    **Given the sources we had** is the second clause. `unjudgeable` is permanent relative to
    a fixed tier set, not absolutely: adding a source turns every stored one into a claim its
    evidence no longer supports. So a row also comes back when `verify.TIER_SET` has *grown*
    since it was judged — once, because the row then records the new set. Without this the
    31 references OpenAlex can supply would sit behind a predicate correctly refusing to ask
    a question that had already been settled.

    Note the asymmetry: a `status == "ok"` row can be retryable on the tier clause but never
    on the lookup clause. A finished row has no backlog; it may still have a verdict that
    only held because we could not look anywhere else.
    """
    if not unscored_picks(row):
        return False
    if (row.get("n_lookup_failed") or 0) > 0 and row.get("status") == "partial":
        return True
    return bool((row.get("n_unjudgeable") or 0) > 0 and tiers_grew(row.get("tier_set")))


def repair_row(case_name: str, row: dict[str, Any], *, model: str) -> dict[str, Any]:
    """Re-resolve and judge only what a partial row missed, then merge.

    Deliberately narrow. It never re-resolves a pick that already has a verdict — those cost
    an arXiv call to re-fetch and cannot change — and it never re-runs the baseline, because
    a redraw would be a different draw wearing this one's number.
    """
    context = assemble_repo_context(WORK_DIR / case_name)
    papers, hallucinated, lookup_failed, unjudgeable = resolve_references(unscored_picks(row), [])
    scores = dict(row.get("scores") or {})
    judge_failed = 0
    for paper in papers:
        pid = canonical_ref(paper["arxiv_id"])
        try:
            scores[pid] = int(
                judge_mod.judge_paper(case_name, context, paper, model=model)["score"]
            )
        except Exception as exc:  # noqa: BLE001 -- an unjudged paper is never scored 0
            judge_failed += 1
            print(f"        ! judge failed for {pid}: {str(exc)[:100]}")
    return {
        **row,
        "status": "partial" if (lookup_failed or judge_failed) else "ok",
        "targets": sorted(p for p, s in scores.items() if s >= ACTIONABLE),
        "scores": dict(sorted(scores.items())),
        # ALL THREE are replaced, not accumulated. `unscored_picks` is every open question
        # on the row, so this pass re-asked all of them and its answers are the current
        # state. Accumulating instead double-counts a reference that fails twice, and the
        # first version of this function did exactly that for two of the three: after two
        # repair passes `1/linter` reported 31 unjudgeable references against 12 unscored
        # picks, and the sweep's total read 71 where the truth was 44. A counter that grows
        # every time you look at it is a history wearing a backlog's name.
        #
        # `test_gold_spread.py` pins the identity this restores — h + l + u equals the
        # number of picks with no verdict — so the next version cannot drift either way.
        "n_hallucinated": hallucinated,
        "n_lookup_failed": lookup_failed,
        "n_unjudgeable": unjudgeable,
        "n_judge_failed": judge_failed,
        # Stamped here too, and this is the line that makes the tier clause terminate. A row
        # re-asked because the tiers grew records the set it was re-asked under, so the same
        # growth cannot trigger it twice. Without it every invocation would re-ask every
        # unjudgeable reference forever — C-30 arriving through the door built to prevent it.
        "tier_set": list(TIER_SET),
    }


def load_artifact(
    prompt_version: str = baseline_mod.DEFAULT_PROMPT_VERSION,
    model: str = baseline_mod.DEFAULT_MODEL,
    tools: str = baseline_mod.DEFAULT_TOOLS,
) -> dict[str, Any]:
    path = out_path(prompt_version, model, tools)
    if path.is_file():
        stored = json.loads(path.read_text(encoding="utf-8"))
        found_model = stored.get("model", baseline_mod.DEFAULT_MODEL)
        if found_model != model:
            raise SystemExit(f"! {path.name} was written by {found_model!r}, not {model!r}.")
        # An artifact that does not say which prompt produced it is v1: it predates the
        # versions. Anything that says otherwise is a file being opened under the wrong flag,
        # and merging the two would mix configurations under one set of draw numbers.
        found = stored.get("prompt_version", baseline_mod.DEFAULT_PROMPT_VERSION)
        if found != prompt_version:
            raise SystemExit(
                f"! {path.name} was written under prompt {found!r}, not {prompt_version!r}."
            )
        # Same rule for the toolset, and it matters more: an artifact opened under the wrong
        # prompt would at least show unfamiliar picks, while one opened under the wrong
        # toolset looks completely ordinary -- the treatment's absence is invisible in the
        # rows. An artifact that does not say is `web`, because it predates the axis.
        found_tools = stored.get("tools", baseline_mod.DEFAULT_TOOLS)
        if found_tools != tools:
            raise SystemExit(
                f"! {path.name} was written with tools {found_tools!r}, not {tools!r}."
            )
        return stored
    return {
        "_comment": (
            "k independent redraws of the cli baseline over the benchmark25 cohort, judged, "
            "to price how much of the gold set is a property of the draw. Derived by "
            "evals/gold_spread.py; pinned by tests/test_gold_spread.py. Written incrementally "
            "-- a partial artifact is expected and resumable."
        ),
        "cohort": COHORT,
        "draws": DRAWS,
        "prompt_version": prompt_version,
        "model": model,
        "tools": tools,
        "results": {},
    }


def save(artifact: dict[str, Any]) -> None:
    path = out_path(
        artifact.get("prompt_version", baseline_mod.DEFAULT_PROMPT_VERSION),
        artifact.get("model", baseline_mod.DEFAULT_MODEL),
        artifact.get("tools", baseline_mod.DEFAULT_TOOLS),
    )
    path.write_text(json.dumps(artifact, indent=1) + "\n", encoding="utf-8")


# ── analysis ───────────────────────────────────────────────────────────────


def _draw_targets(artifact: dict[str, Any], draw: int) -> dict[str, set[str]] | None:
    """{case: targets} for one draw -- **`ok` rows only**.

    A `partial` row is one where arXiv could not resolve a pick or the judge could not score
    one, so its target set is a FLOOR, not a count. Reading it as a count would bias every
    fresh draw downward and manufacture exactly the instability this probe exists to measure:
    the arXiv 429s of 2026-08-26 would have been reported as the gold set moving. Partial
    rows are kept in the artifact and excluded from the arithmetic, and `report` says how
    many were dropped.
    """
    out: dict[str, set[str]] = {}
    for key, row in artifact["results"].items():
        d, case = key.split("/", 1)
        if int(d) == draw and row["status"] == "ok":
            out[case] = set(row.get("targets") or [])
    return out or None


def report(artifact: dict[str, Any]) -> int:
    frozen_all = json.loads(GOLD.read_text(encoding="utf-8"))["targets"]
    frozen = {
        c: {dedup_id(i) for i in ids}
        for c, ids in frozen_all.items()
        if not c.startswith(("bio-", "mat-"))
    }
    n_frozen = sum(len(v) for v in frozen.values())

    ok_by_draw: dict[int, list[str]] = {}
    partial_by_draw: dict[int, list[str]] = {}
    failed_by_draw: dict[int, list[str]] = {}
    unasked_by_draw: dict[int, list[str]] = {}
    for key, row in artifact["results"].items():
        d, case = key.split("/", 1)
        if row["status"] in UNASKED:
            # Void, not null: the run never happened, so it is neither a success nor a
            # failure of this configuration. Reported separately so a reader can see the
            # draw is incomplete rather than concluding the searcher failed on it.
            unasked_by_draw.setdefault(int(d), []).append(case)
            continue
        bucket = (
            ok_by_draw
            if row["status"] == "ok"
            else partial_by_draw
            if row["status"] == "partial"
            else failed_by_draw
        )
        bucket.setdefault(int(d), []).append(case)

    version = artifact.get("prompt_version", baseline_mod.DEFAULT_PROMPT_VERSION)
    print(f"prompt: {version}")
    if version != baseline_mod.DEFAULT_PROMPT_VERSION:
        # The pre-registered decision rule at the bottom of this report was registered for
        # redraws of the SAME configuration; it prices sampling noise. A different prompt is
        # a different searcher, so the same arithmetic answers a different question and the
        # rule must not be applied to it. Say so here rather than trusting a reader to notice.
        print(
            "  NOTE: a different searcher, not a redraw. The overlap figures below are\n"
            "  COVERAGE of the frozen set by this configuration, not reproducibility, and\n"
            "  the pre-registered decision rule does NOT apply to them."
        )
    print(f"frozen {COHORT} gold set: {n_frozen} targets / {len(frozen)} cases")
    all_draws = sorted(
        set(ok_by_draw) | set(partial_by_draw) | set(failed_by_draw) | set(unasked_by_draw)
    )
    for d in all_draws:
        good = ok_by_draw.get(d, [])
        part = partial_by_draw.get(d, [])
        bad = failed_by_draw.get(d, [])
        n = len(good) + len(part) + len(bad)
        # The baseline failing to complete is not the baseline finding nothing. It is its own
        # source of denominator movement, so it is reported as a rate rather than hidden.
        flag = (
            "  [INCOMPLETE — figures below are over the cases it covered]"
            if len(bad) > MAX_FAILED_PER_DRAW
            else ""
        )
        # A rate needs attempts in its denominator. When a draw was entirely throttled there
        # are none, and printing "0% failure rate" would say the draw went perfectly — the
        # void-as-null error arriving in the reporting layer, after the artifact got it right.
        rate = f"{len(bad) / n:.0%} failure rate" if n else "NOT RUN — no attempts to rate"
        print(
            f"  draw {d}: {len(good)} ok, {len(part)} partial (excluded from counts), "
            f"{len(bad)} failed = {rate}{flag}"
        )
        if bad:
            print(f"           failed: {sorted(bad)}")
        if part:
            print(f"           partial: {sorted(part)}")
        if never := unasked_by_draw.get(d, []):
            print(f"           NEVER RUN (quota/login, not a result): {sorted(never)}")

    # Draws are DISCOVERED, not assumed to be 1..DRAWS: a `--draws 4` trial would otherwise
    # be recorded and silently omitted from every figure below.
    present = sorted({int(k.split("/", 1)[0]) for k in artifact["results"]})
    caps = {}
    for d in present:
        seen = {
            row.get("max_turns", baseline_mod.DEFAULT_MAX_TURNS)
            for key, row in artifact["results"].items()
            if int(key.split("/", 1)[0]) == d
        }
        caps[d] = seen.pop() if len(seen) == 1 else "mixed"
    if any(c != baseline_mod.DEFAULT_MAX_TURNS for c in caps.values()):
        print(f"\nturn cap per draw: {caps}")
        print("  Draws at different caps are DIFFERENT CONFIGURATIONS. The aggregate below")
        print(f"  covers only draws at the shipped cap ({baseline_mod.DEFAULT_MAX_TURNS});")
        print("  others are listed separately so the two are never averaged together.")

    default_draws = [d for d in present if caps[d] == baseline_mod.DEFAULT_MAX_TURNS]
    draws = {d: t for d in default_draws if (t := _draw_targets(artifact, d))}
    other = {d: t for d in present if d not in default_draws and (t := _draw_targets(artifact, d))}
    for d, targets in sorted(other.items()):
        shared = set(targets) & set(frozen)
        hit = sum(len(frozen[c] & targets[c]) for c in shared)
        tot = sum(len(frozen[c]) for c in shared)
        n_t = sum(len(targets[c]) for c in shared)
        print(
            f"  draw {d} @ {caps[d]} turns: {n_t} target(s) over {len(shared)} case(s); "
            f"reproduces {hit}/{tot} of the frozen set"
            if tot
            else f"  draw {d} @ {caps[d]} turns: {n_t} target(s), no frozen overlap"
        )
    if not draws:
        print("\nno complete draws at the shipped cap yet.")
        return 0

    # --- the number the published denominators depend on -------------------------
    print("\nreproducibility of the frozen targets, per draw (shared cases only):")
    per_draw_repro = []
    for d, targets in sorted(draws.items()):
        shared = set(targets) & set(frozen)
        hit = sum(len(frozen[c] & targets[c]) for c in shared)
        tot = sum(len(frozen[c]) for c in shared)
        if not tot:
            continue
        per_draw_repro.append(hit / tot)
        print(f"  draw {d}: {hit}/{tot} = {hit / tot:.2f}   over {len(shared)} case(s)")

    # --- how big is each draw's own gold set? ------------------------------------
    print("\ngold-set size per draw (the denominator, had we used that draw):")
    for d, targets in sorted(draws.items()):
        shared = set(targets) & set(frozen)
        print(
            f"  draw {d}: {sum(len(targets[c]) for c in shared)} targets"
            f"  vs frozen {sum(len(frozen[c]) for c in shared)} on the same {len(shared)} case(s)"
        )

    # --- union growth: does the target set saturate? -----------------------------
    common = set.intersection(*[set(t) for t in draws.values()]) & set(frozen)
    if common:
        union: set[tuple[str, str]] = set()
        print(f"\nunion growth over {len(common)} case(s) present in every draw:")
        union |= {(c, i) for c in common for i in frozen[c]}
        print(f"  frozen alone:          {len(union)}")
        for d, targets in sorted(draws.items()):
            union |= {(c, i) for c in common for i in targets[c]}
            print(f"  + draw {d}:              {len(union)}")

        # Chao1 at TARGET level (P16's was pick-level), counting the frozen set as an
        # occasion alongside the fresh draws.
        counts: dict[tuple[str, str], int] = {}
        for source in [frozen, *draws.values()]:
            for c in common:
                for i in source.get(c, set()):
                    counts[(c, i)] = counts.get((c, i), 0) + 1
        f1 = sum(1 for v in counts.values() if v == 1)
        f2 = sum(1 for v in counts.values() if v == 2)
        s_obs = len(counts)
        chao1 = s_obs + (f1 * f1 / (2 * f2) if f2 else f1 * (f1 - 1) / 2)
        print(
            f"  Chao1 (target level, {1 + len(draws)} occasions): "
            f"S_obs={s_obs} f1={f1} f2={f2} -> >= {chao1:.1f}"
        )

    if per_draw_repro:
        mean = sum(per_draw_repro) / len(per_draw_repro)
        label = "reproducibility" if version == baseline_mod.DEFAULT_PROMPT_VERSION else "coverage"
        print(f"\nmean {label} of the frozen set: {mean:.2f}")
        if version != baseline_mod.DEFAULT_PROMPT_VERSION:
            print("  (different searcher — no verdict; see the note at the top.)")
            return 0
        print("  pick-level agreement measured by P15 for comparison: 0.41")
        if mean < 2 / 3:
            verdict = "the denominator is NOT stable — published recall needs an interval"
        elif mean < 0.90:
            verdict = "the point figure stands, with this spread quoted beside it"
        else:
            verdict = "the draw is not a material source of error"
        print(f"  VERDICT (pre-registered): {verdict}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="k independent redraws of the gold set.")
    ap.add_argument("--draws", type=int, default=DRAWS)
    ap.add_argument("--case", help="Comma-separated subset (for smoke tests).")
    ap.add_argument("--dry-run", action="store_true", help="$0: plan and progress.")
    ap.add_argument("--report", action="store_true", help="$0: analyse the artifact.")
    ap.add_argument("--model", default=judge_mod.DEFAULT_JUDGE_MODEL)
    ap.add_argument(
        "--max-turns",
        type=int,
        default=baseline_mod.DEFAULT_MAX_TURNS,
        help="Agent turn cap. Recorded per row; does NOT touch the shipped caches.",
    )
    ap.add_argument(
        "--prompt-version",
        default=baseline_mod.DEFAULT_PROMPT_VERSION,
        choices=sorted(baseline_mod.PROMPTS),
        help=(
            "Which baseline prompt to draw with. v2 allows non-arXiv papers. "
            "Each version writes its OWN artifact; draws are never mixed."
        ),
    )
    ap.add_argument(
        "--effort",
        default=baseline_mod.DEFAULT_EFFORT,
        choices=["low", "medium", "high", "xhigh", "max"],
        help=(
            "Pin the agent's reasoning effort. Omit to take the CLI default (measured as "
            "`high` on 2026-08-27). Recorded on every row either way; pinning also changes "
            "the cache path and discriminator, so a pinned sweep cannot overwrite an "
            "unpinned one."
        ),
    )
    ap.add_argument(
        "--cohort",
        default=COHORT,
        choices=sorted(COHORTS),
        help=(
            "Which repositories to sweep. benchmark25 = the 25 the published denominators "
            "cover; scientific = the 12 bio-/mat- cases; all = 37. Recorded per row."
        ),
    )
    ap.add_argument(
        "--baseline-model",
        default=baseline_mod.DEFAULT_MODEL,
        help=(
            "Which model the baseline runs. Named `--baseline-model` because `--model` is "
            "already the JUDGE's, and one flag silently setting the other would be a probe "
            "measuring something other than its name. Each model writes its OWN artifact."
        ),
    )
    ap.add_argument(
        "--tools",
        default=baseline_mod.DEFAULT_TOOLS,
        choices=sorted(baseline_mod.TOOLSETS),
        help=(
            "Which tools the agent gets. web = WebSearch+WebFetch, every published run. "
            "web+rr adds RepoRadar's MCP server on top (P27) and needs the per-case stores "
            "from `evals/rr_mcp_arm.py --seed` first. Each toolset writes its OWN artifact."
        ),
    )
    ap.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Concurrent PHASE-A baselines. Phase B stays serial (arXiv + judge are gated).",
    )
    args = ap.parse_args()

    artifact = load_artifact(args.prompt_version, args.baseline_model, args.tools)
    if args.report:
        return report(artifact)

    bench = yaml.safe_load((EVALS / "benchmark.yaml").read_text(encoding="utf-8"))
    cases = cohort_cases(bench, args.cohort)
    if args.case:
        want = set(args.case.split(","))
        if unknown := want - {c["name"] for c in cases}:
            print(f"! unknown case(s): {sorted(unknown)}")
            return 1
        cases = [c for c in cases if c["name"] in want]

    by_key = artifact["results"]
    todo_baseline = [
        (d, c)
        for d in range(1, args.draws + 1)
        for c in cases
        # Absent, or present but never actually asked. Keying resume purely on presence is
        # what would have frozen 54 quota failures into a permanent 100% failure rate.
        if by_key.get(f"{d}/{c['name']}", {}).get("status", "__absent__")
        in ("__absent__", *UNASKED)
    ]
    todo_judge = [k for k, row in by_key.items() if row.get("phase") == "baseline"]
    # A partial row used to be partial forever: it carries `phase: "judged"`, so no
    # re-invocation ever looked at it again, and the transient failures inside it were
    # indistinguishable from permanent ones. `retryable` reads the outcome counts rather
    # than the status, so only the rows with a recoverable gap come back.
    todo_repair = sorted(k for k, row in by_key.items() if retryable(row))

    print(f"cohort: {args.cohort} — {len(cases)} case(s) x {args.draws} draw(s)")
    print(
        f"recorded: {len(by_key)}   need a baseline: {len(todo_baseline)}   "
        f"need judging: {len(todo_judge)}   partial and retryable: {len(todo_repair)}"
    )
    print(
        f"turn cap: {args.max_turns}   prompt: {args.prompt_version}"
        f"   baseline model: {args.baseline_model}"
        f"   effort: {args.effort or 'cli-default'}"
        f"   tools: {args.tools}"
        f"   phase-A concurrency: {args.concurrency}"
    )
    print(f"artifact: {out_path(args.prompt_version, args.baseline_model, args.tools).name}")
    if args.dry_run:
        print("\n(dry run)")
        return 0
    if not (todo_baseline or todo_judge or todo_repair):
        print("\nnothing to do; use --report.")
        return report(artifact)

    load_dotenv(EVALS / ".env")
    if not os.environ.get("OPENAI_API_KEY"):
        # Picks without verdicts are not targets, and this whole probe is about targets.
        print("! OPENAI_API_KEY is not set; picks would be unjudgeable. Refusing.")
        return 1
    print(f"baseline auth: {baseline_mod.cli_auth_mode()}   judge: {args.model}")

    lock = threading.Lock()

    def record(key: str, row: dict[str, Any]) -> None:
        # After EVERY row, under a lock. This job is too long to lose to one exception, and
        # concurrent writers make a last-writer-wins save a data-loss bug rather than a race
        # nobody notices (C-29 taught the serial version of this lesson).
        with lock:
            artifact["results"][key] = row
            save(artifact)

    # ── phase A: agentic runs, concurrent, no rate-limited service involved ──
    if todo_baseline:
        print(f"\n=== phase A: {len(todo_baseline)} baseline run(s) ===")
        done = 0
        with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as pool:
            futures = {
                pool.submit(
                    run_baseline_only,
                    case,
                    max_turns=args.max_turns,
                    prompt_version=args.prompt_version,
                    model=args.baseline_model,
                    effort=args.effort,
                    cohort=args.cohort,
                    tools=args.tools,
                ): (draw, case)
                for draw, case in todo_baseline
            }
            for fut in as_completed(futures):
                draw, case = futures[fut]
                key = f"{draw}/{case['name']}"
                try:
                    row = fut.result()
                except Exception as exc:  # noqa: BLE001 -- one bad case must not end the job
                    print(f"  !! {key} crashed: {type(exc).__name__}: {str(exc)[:140]}")
                    row = {"status": "crashed", "phase": "judged", "raw": str(exc)[:160]}
                record(key, row)
                done += 1
                if row["status"] == "baseline_ok":
                    print(
                        f"  [{done}/{len(todo_baseline)}] {key:<18} {len(row['picks'])} pick(s)"
                        f"  turns={row['num_turns']}  {row['duration_s']}s"
                    )
                else:
                    print(
                        f"  [{done}/{len(todo_baseline)}] {key:<18} !! {row['status']}"
                        f"  {row.get('duration_s', '?')}s"
                    )
        todo_judge = [k for k, row in artifact["results"].items() if row.get("phase") == "baseline"]

    # ── phase B: verify + judge, serial, because arXiv and the judge are gated ──
    if todo_judge:
        print(f"\n=== phase B: judging {len(todo_judge)} run(s) (serial) ===")
        for n, key in enumerate(sorted(todo_judge), start=1):
            case_name = key.split("/", 1)[1]
            try:
                row = judge_row(case_name, artifact["results"][key], model=args.model)
            except Exception as exc:  # noqa: BLE001
                print(f"  !! {key} crashed while judging: {type(exc).__name__}: {str(exc)[:140]}")
                continue
            record(key, row)
            print(
                f"  [{n}/{len(todo_judge)}] {key:<18} {len(row['targets'])} target(s)"
                f"  [{row['status']}]"
            )
        todo_repair = sorted(k for k, row in artifact["results"].items() if retryable(row))

    # ── phase C: re-ask only the questions a transient failure left open ──
    if todo_repair:
        print(f"\n=== phase C: retrying {len(todo_repair)} partial row(s) (serial) ===")
        for n, key in enumerate(todo_repair, start=1):
            case_name = key.split("/", 1)[1]
            before = len(artifact["results"][key].get("targets") or [])
            try:
                row = repair_row(case_name, artifact["results"][key], model=args.model)
            except Exception as exc:  # noqa: BLE001
                print(f"  !! {key} crashed while retrying: {type(exc).__name__}: {str(exc)[:140]}")
                continue
            record(key, row)
            print(
                f"  [{n}/{len(todo_repair)}] {key:<18} {before} -> {len(row['targets'])} "
                f"target(s)  [{row['status']}]"
            )

    # Every axis, every time. The prompt version reached the read path and the write path but
    # not this line, so a v2 run that correctly wrote `gold_spread_v2.json` announced that it
    # had written `gold_spread.json` -- the file holding every published denominator. Adding
    # the model axis missed the SAME line again, which is the argument for taking the name
    # from `out_path` rather than assembling it: the function that owns the answer is the one
    # that should be asked, and a status line that has to be updated per axis will be wrong
    # once per axis. C-29 was this omission one path over.
    # ALL THREE axes. This line has been the last one patched for each new axis in
    # turn, and it is the only end-of-run filename a reader sees -- `report()` prints
    # none, and the correct name at the START of the run is ~$86 of scrollback
    # earlier. An announcement naming `gold_spread_v2_opus5.json` after writing
    # `gold_spread_v2_opus5_web_rrwide.json` sends the reader to the control arm.
    print(f"\nwrote {out_path(args.prompt_version, args.baseline_model, args.tools).name}\n")
    return report(artifact)


if __name__ == "__main__":
    raise SystemExit(main())
