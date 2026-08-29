"""Whether there is enough of a paper on hand to judge it at all. [NR-42]

Both LLM stages describe a paper to the model the same way — a title and
``paper["abstract"][:1500]`` — and until now both read that field with no guard. A paper
whose abstract never arrived was still sent, still scored 0-3, and still admitted to a
digest on the strength of its title.

That is not a strict judgement made on thin evidence. It is a **void read as signal**: the
prompt says ``Abstract:`` followed by nothing, and the model answers anyway. This project has
a ledger of the same shape — a keyword filter applied to ``(term, weight)`` pairs that
discarded every keyword and reported 0.0% in all three strata (C-30's neighbourhood), a pool
scanner that read 1,250 papers as 0, a probe whose biomedical flag read two fields that are
``None`` on every record and measured 21% of nothing. Each looked like a finding.

The measurement that opened this: **26.5% of OpenAlex candidates carry no abstract**, against
0 of 17,511 from Europe PMC. Among papers that reached a digest, 4 of 17 non-actionable ones
had no abstract against 1 of 51 actionable. Those intervals are barely disjoint at n=17, so
the *size* of the effect is not established and this module does not claim one — the defect
stands on its own: **a paper we cannot read is unmeasured, not unactionable**, and the two
must not produce the same score.

Deliberately NOT a minimum length. A short abstract is evidence, merely less of it, and
picking a character threshold would be tuning the gate against net@2 through a back door —
which is the thing NR-42 declined to do when it closed the relevance-filter item. The rule
here is absence, and absence is not a matter of degree.

Also deliberately not configurable. Every other stage's failure policy is an invariant, not a
setting, and a flag whose off-position restores "score papers you cannot read" is a footgun
rather than a choice. `triage_papers` already promised this in prose — "a paper whose scoring
fails is omitted (never scored 0), so downstream tiering treats 'couldn't judge' as 'not a
confident Top Pick', not as a confident rejection" — and a missing abstract is the clearest
case of not being able to judge there is. This module makes the promise true before the call
rather than only after one fails.
"""

from __future__ import annotations

from typing import Any

__all__ = ["has_abstract", "partition_by_evidence"]


def has_abstract(paper: dict[str, Any]) -> bool:
    """True when this paper carries abstract text an LLM stage could actually read.

    Whitespace is not evidence: several adapters emit ``" "`` or ``"\\n"`` where a field was
    absent upstream, and a prompt built from those is indistinguishable from one built from
    ``""``. `None` and a missing key are the same answer as an empty string, for the same
    reason — the question is what reaches the prompt, not which of four ways it went missing.
    """
    return bool(str(paper.get("abstract") or "").strip())


def partition_by_evidence(
    papers: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split into (scoreable, skipped), preserving order within each.

    Returned as a partition rather than a filter so callers can *report* what they dropped.
    A guard that silently shortens a list is how "we scored 50 papers" quietly becomes "we
    scored 37" with nothing in the log — the same silence that let the damaged baseline
    caches read as abstentions for six weeks (C-25).
    """
    keep = [p for p in papers if has_abstract(p)]
    skip = [p for p in papers if not has_abstract(p)]
    return keep, skip
