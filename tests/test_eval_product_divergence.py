"""Tests for the product/benchmark divergence audit.

The audit exists because two of this project's published corrections — C-9 (the query
bridge) and C-12 (the version-strip before a source merge) — are the same defect: **one
invariant, two implementations, one of them fixed**. Both were found by accident while
looking for something else, months apart.

So the tests here are not about the script's formatting. They pin the two things that
decide whether it can catch the next one:

* its **checkers actually see the defect** — each is run against a file that has it,
  because a checker that cannot fail is worse than no checker (it reads as a clean bill);
* its **declared exemptions stay honest** — a difference between the shipped default and
  the measured configuration is fine, an *undeclared* one is the bug, and a declaration
  left behind for a field nobody compares any more is a lie in the audit's own output.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evals"))

from audit_product_divergence import (  # noqa: E402
    BENCHMARK_HEADLINE,
    DECLARED,
    PIPELINE_MODULES,
    SAME_VINTAGE_CONTAINERS,
    _norm_calls,
    _product_defaults,
    _raw_merges,
    config_divergences,
)


class TestTheCheckersSeeTheDefect:
    """Each pass, run against a file that has the bug it is supposed to name."""

    def test_a_hand_rolled_version_strip_is_found(self, tmp_path: Path) -> None:
        f = tmp_path / "offender.py"
        f.write_text('known = {p["arxiv_id"].split("v")[0] for p in papers}\n', encoding="utf-8")
        assert [(rule) for _, rule, _ in _norm_calls(f)] == ["split-v"]

    def test_the_shared_helper_is_recognised_under_both_names(self, tmp_path: Path) -> None:
        """`cli.py` re-exports it as `_dedup_id`; missing the alias would report it dirty."""
        f = tmp_path / "ok.py"
        f.write_text(
            'a = dedup_id(p["arxiv_id"])\nb = _dedup_id(p["arxiv_id"])\n', encoding="utf-8"
        )
        assert [rule for _, rule, _ in _norm_calls(f)] == ["dedup_id", "dedup_id"]

    def test_a_raw_id_merge_is_found(self, tmp_path: Path) -> None:
        f = tmp_path / "merge.py"
        f.write_text('if p["arxiv_id"] not in seen:\n    pass\n', encoding="utf-8")
        assert [kind for _, kind, _ in _raw_merges(f)] == ["RAW MERGE"]

    def test_a_repaired_merge_is_not_reported(self, tmp_path: Path) -> None:
        f = tmp_path / "fixed.py"
        f.write_text('if dedup_id(p["arxiv_id"]) not in seen:\n    pass\n', encoding="utf-8")
        assert _raw_merges(f) == []

    def test_a_same_vintage_lookup_is_declared_not_flagged(self, tmp_path: Path) -> None:
        """A dict built from the very list being iterated needs no normalising."""
        f = tmp_path / "lookup.py"
        f.write_text('x = s["arxiv_id"] in papers_by_id\n', encoding="utf-8")
        assert [kind for _, kind, _ in _raw_merges(f)] == ["declared"]


class TestTheCurrentTreeIsClean:
    """The audit's own verdict, asserted — so a regression fails CI, not a later reader."""

    @pytest.mark.parametrize("parts", PIPELINE_MODULES, ids=lambda p: p[-1])
    def test_no_module_hand_rolls_a_version_strip(self, parts: tuple[str, ...]) -> None:
        path = Path(__file__).resolve().parents[1].joinpath(*parts)
        offenders = [(line, text) for line, rule, text in _norm_calls(path) if rule == "split-v"]
        assert offenders == [], f"{path.name}: {offenders}"

    @pytest.mark.parametrize("parts", PIPELINE_MODULES, ids=lambda p: p[-1])
    def test_no_module_merges_on_a_raw_id(self, parts: tuple[str, ...]) -> None:
        path = Path(__file__).resolve().parents[1].joinpath(*parts)
        offenders = [(line, text) for line, kind, text in _raw_merges(path) if kind == "RAW MERGE"]
        assert offenders == [], f"{path.name}: {offenders}"

    def test_every_config_difference_is_declared(self) -> None:
        """The durable half of the audit.

        `arxiv.lookback_days` shipped at 14 days for a month while every headline was
        measured all-time — nobody was lying, the two numbers simply lived in different
        files. This fails the moment a shipped default and the measured configuration
        part company without a written reason.
        """
        undeclared = [d for d in config_divergences() if d.status == "DIVERGENT"]
        assert undeclared == [], (
            "the benchmark is no longer measuring the shipped default for: "
            + ", ".join(d.name for d in undeclared)
        )


class TestTheDeclarationsStayHonest:
    def test_every_declared_field_is_actually_compared(self) -> None:
        """A declaration for a field the audit no longer looks at reads as coverage."""
        assert set(DECLARED) <= set(BENCHMARK_HEADLINE)

    def test_every_declared_field_actually_differs(self) -> None:
        """If a difference is closed, the exemption must go with it — otherwise the next
        person to change that default gets no warning."""
        product = _product_defaults()
        for key in DECLARED:
            assert product[key] != BENCHMARK_HEADLINE[key], (
                f"{key} no longer differs; remove it from DECLARED so the guard can bite"
            )

    def test_every_compared_field_resolves_against_the_product(self) -> None:
        """A renamed config field must fail loudly, not drop out of the comparison."""
        assert set(_product_defaults()) == set(BENCHMARK_HEADLINE)

    def test_each_exempt_container_carries_its_reason(self) -> None:
        assert all(reason.strip() for reason in SAME_VINTAGE_CONTAINERS.values())


class TestBothRunnersFailTheSameWay:
    """A collection failure must never be scoreable, in either eval runner.

    `evals/harness.py` raises on `CollectionError` because scoring a throttled fetch as an
    honest result once supplied −17 of a −21 delta (C-4). `evals/run_eval.py` printed a
    warning and carried on with whatever it had — which, with a second source enabled,
    means a domain-purity number computed on a pool missing its arXiv half, indexed and
    printed like any other. Two runners, one invariant, one of them fixed: the shape this
    whole audit is looking for.
    """

    def test_collect_live_raises_instead_of_returning_a_partial_pool(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import run_eval

        from reporadar.collector import CollectionError

        def boom(*_args: object, **_kwargs: object) -> list[dict[str, object]]:
            raise CollectionError("429 from arXiv")

        monkeypatch.setattr("reporadar.collector.collect_papers", boom)
        profile = run_eval.profile_case_repo(Path(__file__).resolve().parents[1])
        case = {"expected_categories": ["cs.LG"], "name": "x"}
        with pytest.raises(CollectionError):
            run_eval.collect_live(profile, case, ["arxiv", "semantic_scholar"], {}, 90)
