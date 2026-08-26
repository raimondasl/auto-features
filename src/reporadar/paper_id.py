"""One answer to "are these two records the same paper".

This module exists because that question had **eight** answers. `dedup_id` began life in
`collector.py`, beside `to_plain_keywords`, on the reasoning that keeping it next to the
function whose output it consumes would stop it drifting. It did not: a survey on
2026-08-15 found three competing rules across the product —

* ``arxiv_id.split("v")[0]`` — truncates at the **first** lowercase ``v``, wherever it is.
  ``solv-int/9801001`` becomes ``sol`` and ``dblp:conf/vldb/X`` becomes ``dblp:conf/``.
* ``re.sub(r"v\\d+$", "", arxiv_id)`` — anchored at the end, so it survives the above, but
  it will happily edit a synthetic id that merely ends in a version-shaped suffix.
* ``dedup_id`` — anchored at both ends against the two arXiv id eras, so anything that is
  not an arXiv id passes through untouched.

They agree on ``2401.12345v2``, which is why the disagreement stayed invisible: every rule
is correct on the ids anyone looks at. Nothing but a survey was ever going to find it.

It lives in its own module rather than in `collector` because **importing `collector` costs
~1.9 seconds and pulls in 1,250 modules** (it imports the `arxiv` client). Eight callers —
the MCP server, two signal collectors, three source adapters, the citation graph — would
have paid that to normalise a string, and the ones that are lazily imported to keep CLI
startup fast would have stopped being lazy. A shared rule nobody can afford to import is a
shared rule that grows local copies again.
"""

from __future__ import annotations

import re

# The two arXiv id eras, written once. Pre-2007 ids are `cs/0602007`, `math.GT/0309136`,
# `cond-mat.supr-con/9501001`; five of them sit in this project's judged pools, and the first
# version of `dedup_id` left their versions on — which is how it came to disagree with the
# `split("v")[0]` rule used elsewhere for the same job.
_ARXIV_NEW = r"\d{4}\.\d{4,5}"

# The real arXiv archive stems. The old-style pattern used to be `[a-z-]+/\d{7}`, which
# accepts any lowercase word before the slash — so `publication/2256929`, the ResearchGate
# URL path that C-25 was written about, answered TRUE to `is_arxiv_id`. `evals/verify.py`
# had already been hardened with this allowlist for `extract_arxiv_ids`, which left two
# rules for "is this an arXiv id" disagreeing about the one id the project knows is bogus:
# the C-14 shape, inside the module that exists to prevent it. The list lives here, and
# verify.py imports it, so there is one rule again. ("math-ph" precedes "math" so the
# longer stem wins.)
ARXIV_ARCHIVES = (
    "astro-ph|cond-mat|gr-qc|hep-ex|hep-lat|hep-ph|hep-th|math-ph|math|nlin|"
    "nucl-ex|nucl-th|quant-ph|q-bio|q-fin|cs|econ|eess|physics|stat|"
    "alg-geom|funct-an|dg-ga|chao-dyn|solv-int|patt-sol|adap-org|cmp-lg|mtrl-th|supr-con"
)
_ARXIV_OLD = rf"(?:{ARXIV_ARCHIVES})(?:\.[A-Za-z-]+)?/\d{{7}}"

# With a version suffix, for version-insensitive cross-source dedup. Both are anchored and
# neither can match a synthetic `ss:`/`dblp:`/`oa:`/`iacr:`/`biorxiv:`/`doi:` id, which must
# pass through untouched.
_ARXIV_VER_RE = re.compile(rf"^({_ARXIV_NEW})v\d+$")
_ARXIV_OLD_VER_RE = re.compile(rf"^({_ARXIV_OLD})v\d+$")
# Version optional: "is this an arXiv id at all", which is a different question.
_ARXIV_ANY_RE = re.compile(rf"^(?:{_ARXIV_NEW}|{_ARXIV_OLD})(?:v\d+)?$")


def is_arxiv_id(value: str) -> bool:
    """True when *value* is an arXiv id rather than one of the synthetic ids.

    Callers used to answer this by exclusion — Semantic Scholar's adapter built an abstract
    URL for anything that did not start with ``ss:``, which was true of every id it could
    produce until :func:`doi_key` gave it a second one. By-exclusion tests are correct
    exactly until the next id scheme is added, so this asks the question positively.

    It also replaces the copy of these two eras that `sources/s2_recommendations.py` kept for
    deciding which ids S2 can resolve as seeds. That is the same predicate, and this module
    exists because this project has already paid three times for the same rule living in more
    than one place (see :func:`dedup_id`).
    """
    return bool(_ARXIV_ANY_RE.match(value))


# A DOI as it arrives: bare, prefixed, or as any of the resolver URLs. OpenAlex returns
# `https://doi.org/10.1101/...`, Semantic Scholar and DBLP the bare form, bioRxiv the bare
# form in its own `doi` field.
_DOI_PREFIX_RE = re.compile(r"^(?:https?://(?:dx\.)?doi\.org/|doi:)", re.IGNORECASE)


def doi_key(doi: str | None) -> str:
    """Canonical cross-source id for a paper with a DOI, or ``""`` if there is none.

    Every non-arXiv adapter minted its own synthetic id from whatever handle its API
    happened to use — ``oa:W2741809807``, ``ss:649def34f8be52c8b66281af98ae884c09aef38b``,
    ``biorxiv:10.1101/2024.01.01.123456``, ``dblp:conf/vldb/X``. Three of them can return
    the SAME preprint, and under three different ids it survives every dedup this project
    has, so it enters the pool three times, is gated three times (three API calls), and can
    occupy three slots of a ten-paper digest.

    A DOI is the identifier all of those sources already agree on, so when one is known it
    is the id. Sources that supply no DOI keep their synthetic ids, which is why this
    returns ``""`` rather than raising: the caller falls back to what it used before.

    Normalisation is lowercase because DOI names are case-insensitive by specification and
    the sources disagree in practice — OpenAlex lowercases, Crossref does not — which is the
    same trap `_extract_arxiv_id` fell into for arXiv DOIs (see its comment: a lowercase DOI
    passed a case-insensitive guard, failed a case-sensitive split, and fell through to a
    synthetic id for a paper arXiv had already supplied).

    Not applied to arXiv papers even though they have DOIs: their arXiv id is the id the
    rest of this project is keyed on — the HyDE index, the judge cache, every stored score —
    and :func:`dedup_id` is what reconciles versions there.
    """
    if not doi:
        return ""
    cleaned = _DOI_PREFIX_RE.sub("", doi.strip()).strip().lower()
    # A DOI is `10.<registrant>/<suffix>`. Anything else is a handle of some other kind and
    # must not be minted into an id that claims cross-source authority.
    if not cleaned.startswith("10.") or "/" not in cleaned:
        return ""
    return f"doi:{cleaned}"


def dedup_id(arxiv_id: str) -> str:
    """Version-strip an arXiv id for cross-source dedup; leave every other id as-is.

    Sources disagree about versions for the same paper — arXiv hands back ``2605.23815v1``
    where Semantic Scholar says ``2605.23815`` — so a merge on raw equality admits both
    copies and the digest shows one paper twice.

    The history is the argument for a single implementation. It drifted three times:

    * **C-12** — ``cli.py`` was fixed to version-strip before merging a non-arXiv source
      while ``evals/harness.py`` kept merging on raw ids, and the 2026-08-13 Semantic
      Scholar A/B consequently showed **6 duplicate papers across 4 cases** in the
      treatment arm and none in the control.
    * **C-12b** — the guard written to prevent that read one file by name, so a third
      runner two files away kept the identical bug.
    * **C-14** — a bare ``split("v")[0]`` was doing the same job at eight further call
      sites, and this function was itself too narrow: it left versions on the pre-2007 ids,
      five of which sit in this project's judged pools. So a source merge and the judge pool
      five steps later disagreed about whether two records were one paper.

    Handling both id eras here, anchored, makes the shared rule strictly better than every
    local copy it replaces — which is the only way a consolidation actually sticks.
    """
    match = _ARXIV_VER_RE.match(arxiv_id) or _ARXIV_OLD_VER_RE.match(arxiv_id)
    return match.group(1) if match else arxiv_id


def canonical_ref(ref: str) -> str:
    """One id for a reference, whichever of the two schemes it is written in.

    Composed rather than new: a DOI in any accepted form becomes :func:`doi_key`'s ``doi:``
    id, and everything else falls to :func:`dedup_id`. Both are idempotent, so applying this
    to an already-canonical id is a no-op — which is what lets a caller apply it on both
    sides of a comparison without tracking which side has been through it.

    It exists because the baseline's v2 prompt made the two schemes meet. A pick arrives as
    the model wrote it (``10.1038/s41586-021-03819-2``), while the same paper comes back from
    `verify.resolve_by_doi_s2` carrying the prefixed form in its ``arxiv_id`` field — so
    `gold_spread`'s stored picks and its judged targets would have disagreed about identity
    for every non-arXiv paper, and its own ``targets <= picks`` invariant would have failed.
    For an arXiv id this is exactly ``dedup_id`` and changes nothing about the stored runs.
    """
    return doi_key(ref) or dedup_id(ref)
