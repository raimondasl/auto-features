"""The $0 join that predicts the headline at `ranking.w_embedding = 1.5`.

The join pairs RepoRadar's per-case net@2 from two `--baseline none` draws against the
baseline column of a run that never saw those draws' pool. That is only legitimate because
of one invariant:

    net@2 is a function of a system's OWN returned papers. Pool composition drives
    `ndcg@k` and `pool_has_relevant`; it does not touch net@2.

If that stops being true, the join silently produces a plausible wrong number — the exact
failure mode of C-17, where a mean assembled from the wrong cases was published as a
headline. So the first test below is not about the join at all; it pins the invariant in
`metrics.summarize_system` directly, by feeding the same returned scores against wildly
different pools and demanding net@2 not move.

The rest pin the guards, because a comparison script's value is entirely in what it
REFUSES. Each one is checked by breaking it: a guard that cannot be shown to fire is
indistinguishable from a guard that never fires.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evals"))


def _case(
    name: str,
    *,
    net: float = 4.0,
    w: float | None = 1.5,
    fp: str = "fp0",
    window: int = 15,
    baseline_status: str = "skipped",
) -> dict[str, Any]:
    return {
        "case": name,
        "w_embedding": w,
        "digest_window": window,
        "baseline_status": baseline_status,
        "pool_provenance": {"fingerprint": fp},
        "reporadar_toppicks": {
            "net_value@2": net,
            "n_returned": 10,
            "n_actionable": 8,
        },
        "baseline": {"net_value@2": 1.0, "n_returned": 2, "n_actionable": 2},
    }


def _run(**kw: Any) -> dict[str, dict[str, Any]]:
    return {n: _case(n, **kw) for n in ("a", "b")}


class TestTheInvariantTheJoinRestsOn:
    """net@2 must not depend on the pool. Everything else here is bookkeeping."""

    def test_net_at_2_ignores_the_pool(self) -> None:
        from metrics import summarize_system

        returned = [3, 2, 0, 1]
        empty = summarize_system(returned, [])
        rich = summarize_system(returned, [3] * 200)
        poor = summarize_system(returned, [0] * 200)
        assert empty["net_value@2"] == rich["net_value@2"] == poor["net_value@2"]

    def test_and_the_pool_does_move_the_metrics_that_should_move(self) -> None:
        """The complement: if NOTHING moved, the fixture would be proving nothing."""
        from metrics import summarize_system

        returned = [3, 2, 0, 1]
        rich = summarize_system(returned, [3] * 200)
        poor = summarize_system(returned, [0] * 4)
        assert rich["pool_has_relevant"] != poor["pool_has_relevant"]
        assert rich["ndcg@k"] != poor["ndcg@k"]

    def test_volume_columns_are_pool_free_too(self) -> None:
        """`shown`/`actionable` are counted off the returned list, not the pool."""
        from metrics import summarize_system

        returned = [3, 2, 0]
        a = summarize_system(returned, [])
        b = summarize_system(returned, [3] * 50)
        assert (a["n_returned"], a["n_actionable"]) == (b["n_returned"], b["n_actionable"])


class TestItRefusesAMislabelledArm:
    def test_accepts_the_arm_it_says_it_is(self) -> None:
        import join_wemb_headline as j

        j.check_draw("treat", _run(w=1.5), 1.5)

    def test_refuses_a_swapped_arm(self) -> None:
        import join_wemb_headline as j

        with pytest.raises(SystemExit, match="w_embedding"):
            j.check_draw("treat", _run(w=0.0), 1.5)

    def test_refuses_the_wrong_digest_width(self) -> None:
        """Every paired-vs-Opus number in RESULTS.md predating 2026-08-15 is width 10.
        Joining one of those against a width-15 arm compares two different products."""
        import join_wemb_headline as j

        with pytest.raises(SystemExit, match="digest_window"):
            j.check_draw("treat", _run(w=1.5, window=10), 1.5)

    def test_refuses_an_arm_that_already_ran_a_baseline(self) -> None:
        """A draw with its own baseline needs no join; silently joining it would pair
        RepoRadar against one baseline run while reporting another's numbers."""
        import join_wemb_headline as j

        with pytest.raises(SystemExit, match="baseline none"):
            j.check_draw("treat", _run(w=1.5, baseline_status="ok"), 1.5)


class TestItRefusesArmsFromDifferentPools:
    def test_accepts_one_shared_pool(self) -> None:
        import join_wemb_headline as j

        j.check_same_pool({"c": _run(fp="same"), "t": _run(fp="same")})

    def test_refuses_a_fingerprint_split(self) -> None:
        import join_wemb_headline as j

        with pytest.raises(SystemExit, match="different pools"):
            j.check_same_pool({"c": _run(fp="one"), "t": _run(fp="two")})

    def test_refuses_a_case_set_split(self) -> None:
        import join_wemb_headline as j

        short = {"a": _case("a")}
        with pytest.raises(SystemExit, match="case set differs"):
            j.check_same_pool({"c": _run(), "t": short})


class TestItRefusesTheWrongHeadlineArtifact:
    def test_refuses_a_headline_that_is_not_the_published_one(self) -> None:
        """The mean is asserted so that pointing this at a neighbouring artifact — there
        are a dozen with near-identical names — fails loudly instead of republishing."""
        import join_wemb_headline as j

        with pytest.raises(SystemExit, match="expected"):
            j.check_headline(_run(w=0.0, net=9.0))

    def test_refuses_a_headline_measured_at_a_treatment_value(self) -> None:
        import join_wemb_headline as j

        with pytest.raises(SystemExit, match="w_embedding"):
            j.check_headline(_run(w=1.5))


class TestAFailedBaselineIsNamedNotScored:
    def test_excluded_and_reported(self) -> None:
        run = {
            "ok1": _case("ok1", baseline_status="ok"),
            "bad": _case("bad", baseline_status="error"),
        }
        import join_wemb_headline as j

        ok, failed = j.baseline_net(run)
        assert set(ok) == {"ok1"}
        assert failed == ["bad"], "a failed baseline must be named, never read as a 0.0"


class TestItRunsOnTheRealArtifacts:
    """The join's own numbers are a result, not a unit test. What is checkable is that it
    still parses today's artifacts and still reproduces the baseline column already
    published — 48 shown, 45 actionable, +1.62/case."""

    @pytest.fixture(autouse=True)
    def _skip_without_artifacts(self) -> None:
        import join_wemb_headline as j

        if not (j.RESULTS / j.HEADLINE_RUN).exists():
            pytest.skip("run artifacts are gitignored; present only on the eval machine")

    def test_reproduces_the_published_baseline_column(self) -> None:
        import statistics

        import join_wemb_headline as j

        headline = j.load(j.HEADLINE_RUN)
        b_net, failed = j.baseline_net(headline)
        assert failed == ["thin-lang"]
        shown, good, prec = j.volume(headline, sorted(b_net), key="baseline")
        assert (shown, good) == (48, 45)
        assert round(prec, 3) == 0.938
        assert round(statistics.mean(b_net.values()), 2) == 1.62

    def test_main_is_clean(self) -> None:
        import join_wemb_headline as j

        assert j.main() == 0


class TestTheDocumentedContract:
    def test_it_does_not_print_a_pool_dependent_statistic(self) -> None:
        """The docstring promises the join never prints anything reading `pool_gains`.
        A later edit adding an ndcg or pool-recall line would make it a lie."""
        src = (ROOT / "evals" / "join_wemb_headline.py").read_text(encoding="utf-8")
        body = src.split('"""', 2)[2]
        for banned in ("ndcg", "n_actionable_in_pool", "pool_has_relevant", "pool_gains"):
            assert banned not in body, f"{banned} is pool-dependent and cannot be joined"

    def test_the_named_artifacts_are_pinned_not_globbed(self) -> None:
        """Globbing for 'the latest wemb run' would silently re-point at a future draw."""
        import join_wemb_headline as j

        assert all(n.endswith(".json") for n, _ in j.DRAWS.values())
        assert len({n for n, _ in j.DRAWS.values()}) == 4
        assert sorted(w for _, w in j.DRAWS.values()) == [0.0, 0.0, 1.5, 1.5]


def test_the_artifact_names_resolve_or_the_suite_says_why() -> None:
    """Documents the split: the script is committed, the artifacts it reads are not."""
    import join_wemb_headline as j

    missing = [n for n, _ in j.DRAWS.values() if not (j.RESULTS / n).exists()]
    if missing:
        pytest.skip(f"gitignored run artifacts absent: {len(missing)} of 4")
    for name, _ in j.DRAWS.values():
        assert json.loads((j.RESULTS / name).read_text(encoding="utf-8"))
