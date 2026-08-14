"""One rule for "are these two records the same paper", and a guard over every module.

`dedup_id` is the shared answer. The tests below are in two halves and the second is the
one with teeth.

**The rule.** Both arXiv id eras version-strip; everything else passes through untouched.
The "untouched" half is not defensive padding — the two rules this one replaced would edit
a synthetic ``ss:``/``dblp:`` id, and merging two different papers is a worse failure than
failing to merge one.

**The wiring.** A unit test of `dedup_id` passes whether or not anyone calls it, which is
exactly how `to_plain_keywords` sat correct and unused through C-9. So the guard reads the
source of **every** module in `src/reporadar/` and `evals/`, not the handful where the bug
was last found. That scoping is the point: the previous version of this guard listed five
pipeline files, and a survey on 2026-08-15 then turned up three competing rules across
eight product modules it had never looked at — the same "guard scoped to where you found
it" mistake the guard exists to teach.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from reporadar.paper_id import dedup_id

ROOT = Path(__file__).resolve().parents[1]

# Rules that answer "is this the same paper" and are NOT the shared one. Matched against
# live source lines, so a comment quoting a rule to explain it does not fail the guard.
COMPETING_RULES = (
    ('.split("v")[0]', "truncates at the first lowercase v, wherever it is"),
    ('re.sub(r"v\\d+$"', "edits any id ending in a version-shaped suffix, synthetic ones too"),
    ("re.sub(r'v\\d+$'", "edits any id ending in a version-shaped suffix, synthetic ones too"),
)

# Lines that hold a competing rule ON PURPOSE. Declared by their exact source text rather
# than by line number, so an edit elsewhere in the file cannot silently widen the hole, and
# each carries the reason it is not a defect — the same discipline the divergence audit uses
# for its raw-id lookups. Adding an entry is a visible decision; a stale one fails below.
DECLARED_COMPETING_RULES: tuple[tuple[str, str, str], ...] = (
    (
        "evals/audit_product_divergence.py",
        'return arxiv_id.split("v")[0] if "v" in arxiv_id else arxiv_id',
        "the audit reports where dedup_id and the old rule DISAGREE, so it has to be able "
        "to compute the old one; deleting it would delete the comparison",
    ),
)


def _product_and_eval_modules() -> list[Path]:
    """Every module either side of the product/benchmark line.

    `evals/.work/` holds cloned benchmark repositories — other people's source, with their
    own conventions — so it is excluded. Everything this project actually wrote is in.
    """
    files = sorted(ROOT.joinpath("src", "reporadar").rglob("*.py"))
    files += sorted(p for p in ROOT.joinpath("evals").glob("*.py"))
    return [p for p in files if ".work" not in p.parts]


class TestTheRule:
    def test_versioned_and_unversioned_ids_collapse(self) -> None:
        assert dedup_id("2605.23815v1") == dedup_id("2605.23815") == "2605.23815"

    def test_five_digit_ids_are_handled(self) -> None:
        assert dedup_id("2502.08832v2") == "2502.08832"

    @pytest.mark.parametrize(
        "other", ["ss:abc123", "dblp:conf/x/Y", "iacr:2026/1373", "cs/0301001"]
    )
    def test_non_arxiv_ids_are_left_alone(self, other: str) -> None:
        """A synthetic id has no version to strip, and mangling one would merge two
        genuinely different papers into one."""
        assert dedup_id(other) == other

    @pytest.mark.parametrize(
        ("versioned", "base"),
        [
            ("cs/0602007v4", "cs/0602007"),
            ("cs/0007008v1", "cs/0007008"),
            ("math.GT/0309136v2", "math.GT/0309136"),
            ("cond-mat.supr-con/9501001v1", "cond-mat.supr-con/9501001"),
        ],
    )
    def test_old_style_ids_version_strip_too(self, versioned: str, base: str) -> None:
        """Pre-2007 ids, which the first version of this function left versioned.

        Five of them sit in this project's judged pools. Leaving their versions on was not
        merely incomplete — it made this function DISAGREE with the copies doing the same
        job elsewhere, so which rule a merge happened to use decided whether one paper
        counted once or twice.
        """
        assert dedup_id(versioned) == dedup_id(base) == base

    @pytest.mark.parametrize(
        "tricky", ["solv-int/9801001", "ss:vector-db-7", "dblp:journals/vldb/Abc"]
    )
    def test_an_id_containing_a_v_is_not_truncated(self, tricky: str) -> None:
        """The failure mode of the first rule this one replaced.

        ``"solv-int/9801001".split("v")[0]`` is ``"sol"`` and ``"dblp:journals/vldb/Abc"``
        becomes ``"dblp:journals/"`` — which would merge every such paper into a single
        phantom. Consolidating is only safe because this rule is anchored; that is the claim.
        """
        assert dedup_id(tricky) == tricky
        assert dedup_id(tricky) != tricky.split("v")[0]

    @pytest.mark.parametrize("synthetic", ["ss:abcv2", "oa:W123v1", "dblp:conf/x/Yv3"])
    def test_a_synthetic_id_ending_in_a_version_is_not_edited(self, synthetic: str) -> None:
        """The failure mode of the *second* rule this one replaced.

        ``re.sub(r"v\\d+$", "", ...)`` survives the truncation bug above by anchoring at the
        end — and then edits opaque ids that merely end that way. An S2 or OpenAlex id is
        not ours to reinterpret.
        """
        assert dedup_id(synthetic) == synthetic
        assert dedup_id(synthetic) != re.sub(r"v\d+$", "", synthetic)

    def test_it_is_idempotent(self) -> None:
        """Merges apply it to both sides and to already-normalised sets."""
        for aid in ("2605.23815v1", "cs/0602007v4", "ss:abc", "2605.23815"):
            assert dedup_id(dedup_id(aid)) == dedup_id(aid)


class TestOneRuleEverywhere:
    """No module anywhere may answer this question its own way."""

    @pytest.mark.parametrize(
        "path", _product_and_eval_modules(), ids=lambda p: f"{p.parent.name}/{p.name}"
    )
    def test_no_module_hand_rolls_the_rule(self, path: Path) -> None:
        if path.name == "paper_id.py":
            return  # this module IS the rule
        rel = path.relative_to(ROOT).as_posix()
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if any(rel == d and stripped == text for d, text, _ in DECLARED_COMPETING_RULES):
                continue
            for rule, why in COMPETING_RULES:
                assert rule not in line, (
                    f"{rel}:{line_no} answers 'same paper?' with a rule that {why}, "
                    f"instead of calling reporadar.paper_id.dedup_id: {stripped}"
                )

    def test_the_guard_covers_more_than_where_the_bug_was_found(self) -> None:
        """The scoping *is* the lesson.

        Its predecessor listed five pipeline modules — the files C-9 and C-12 happened to
        live in. A survey then found three competing rules across eight product modules it
        had never looked at. If this ever narrows back to a handful, the next drift is
        invisible again.
        """
        covered = {p.name for p in _product_and_eval_modules()}
        for beyond_the_pipeline in ("mcp_server.py", "citations.py", "hn.py", "integrity.py"):
            assert beyond_the_pipeline in covered
        assert len(covered) > 40

    def test_every_declared_exemption_still_points_at_real_code(self) -> None:
        """A stale exemption is a hole nobody can see.

        If the declared line is edited or deleted, this fails and the next reader has to
        re-decide whether the exemption is still earned — rather than inheriting a blanket
        pass on a file that has since changed.
        """
        for rel, text, reason in DECLARED_COMPETING_RULES:
            lines = {
                ln.strip() for ln in ROOT.joinpath(rel).read_text(encoding="utf-8").splitlines()
            }
            assert text in lines, f"{rel} no longer contains the declared line: {text}"
            assert reason.strip(), rel


class TestTheCallersActuallyCallIt:
    """Forbidding copies is half a guard; a file that deleted the call passes it happily."""

    @pytest.mark.parametrize(
        "rel",
        [
            "src/reporadar/cli.py",
            "src/reporadar/digest.py",
            "src/reporadar/citations.py",
            "src/reporadar/citation_graph.py",
            "src/reporadar/mcp_server.py",
            "src/reporadar/sources/dblp.py",
            "src/reporadar/sources/hf_papers.py",
            "src/reporadar/signals/hn.py",
            "src/reporadar/signals/integrity.py",
            "evals/harness.py",
            "evals/run_eval.py",
            "evals/run_judge_eval.py",
        ],
    )
    def test_it_is_imported_and_used(self, rel: str) -> None:
        text = ROOT.joinpath(rel).read_text(encoding="utf-8")
        assert "dedup_id" in text, f"{rel} no longer normalises ids at all — did a merge move?"

    def test_the_shared_rule_stays_cheap_to_import(self) -> None:
        """Why it is not in `collector`.

        Importing `reporadar.collector` costs ~1.9 s and 1,250 modules — it pulls in the
        arXiv client. Eight callers would have paid that to normalise a string, and the
        lazily-imported ones would have stopped being lazy. A shared rule nobody can afford
        to import grows local copies again, which is how this started.
        """
        source = ROOT.joinpath("src", "reporadar", "paper_id.py").read_text(encoding="utf-8")
        imports = [
            ln
            for ln in source.splitlines()
            if ln.startswith(("import ", "from ")) and "__future__" not in ln
        ]
        assert imports == ["import re"], imports
