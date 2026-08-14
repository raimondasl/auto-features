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

# Modern arXiv id with a version suffix, for version-insensitive cross-source dedup.
_ARXIV_VER_RE = re.compile(r"^(\d{4}\.\d{4,5})v\d+$")
# Pre-2007 ids: `cs/0602007v4`, `math.GT/0309136v2`, `cond-mat.supr-con/9501001v1`. Five of
# them sit in this project's judged pools, and the first version of this function left
# their versions on — which is how it came to disagree with the `split("v")[0]` rule used
# elsewhere for the same job. Both halves are anchored and neither can match a synthetic
# `ss:`/`dblp:`/`oa:`/`iacr:`/`biorxiv:` id, which must pass through untouched.
_ARXIV_OLD_VER_RE = re.compile(r"^([a-z-]+(?:\.[A-Za-z-]+)?/\d{7})v\d+$")


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
