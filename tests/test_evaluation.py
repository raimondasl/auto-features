"""Tests for reporadar.evaluation — the `rr eval` harness."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from reporadar.config import QueriesConfig, RankingConfig, RepoRadarConfig
from reporadar.evaluation import (
    Judgments,
    bootstrap_delta,
    compare_configs,
    compare_to_baseline,
    evaluate,
    format_baseline_check,
    format_comparison,
    format_report,
    load_judgments,
    rank_judged_papers,
    stored_signals,
)
from reporadar.profiler import RepoProfile
from reporadar.store import PaperStore

ON_TOPIC = "retrieval augmented generation with transformers and vector search"
OFF_TOPIC = "combinatorial optimization of scheduling heuristics on graphs"


def _profile() -> RepoProfile:
    return RepoProfile(
        keywords=[("retrieval", 0.8), ("transformers", 0.7), ("generation", 0.6)],
        anchors=["torch"],
        domains=["NLP"],
    )


def _cfg(**ranking: float) -> RepoRadarConfig:
    cfg = RepoRadarConfig()
    cfg.queries = QueriesConfig()
    if ranking:
        cfg.ranking = RankingConfig(**ranking)  # type: ignore[arg-type]
    return cfg


def _paper(arxiv_id: str, *, on_topic: bool, published: str | None = None) -> dict:
    return {
        "arxiv_id": arxiv_id,
        "title": ("RAG" if on_topic else "Scheduling") + f" {arxiv_id}",
        "authors": ["A"],
        "abstract": ON_TOPIC if on_topic else OFF_TOPIC,
        # Same category either way, so only keyword matching can separate them —
        # otherwise a "ranking change" test passes on the category signal alone.
        "categories": ["cs.CL"],
        "published": published or "2026-07-20T00:00:00+00:00",
        "updated": None,
        "url": f"http://arxiv.org/abs/{arxiv_id}",
        "pdf_url": None,
    }


def _seed(store: PaperStore, n: int = 20) -> None:
    for i in range(n):
        on = i % 2 == 0
        arxiv_id = f"2607.{i:05d}v1"
        store.upsert_paper(_paper(arxiv_id, on_topic=on))
        store.save_rating(arxiv_id, 5 if on else 1)


class TestLoadJudgments:
    def test_rating_thresholds_match_the_feedback_loop(self, tmp_path: Path) -> None:
        # Must agree with feedback.compute_adjusted_weights, or the thing being
        # measured and the thing being tuned disagree about what "good" means.
        with PaperStore(tmp_path / "p.db") as store:
            for i, rating in enumerate((5, 4, 3, 2, 1), start=1):
                arxiv_id = f"2607.0000{i}v1"
                store.upsert_paper(_paper(arxiv_id, on_topic=True))
                store.save_rating(arxiv_id, rating)
            judgments = load_judgments(store)
        assert judgments.labels == {
            "2607.00001v1": 1,
            "2607.00002v1": 1,
            "2607.00004v1": 0,
            "2607.00005v1": 0,
        }
        assert judgments.ignored_neutral == 1  # the 3-star is dropped, not forced

    def test_stars_count_as_relevant(self, tmp_path: Path) -> None:
        with PaperStore(tmp_path / "p.db") as store:
            store.upsert_paper(_paper("2607.00001v1", on_topic=True))
            store.star_paper("2607.00001v1")
            judgments = load_judgments(store)
        assert judgments.labels == {"2607.00001v1": 1}
        assert judgments.from_stars == 1

    def test_an_explicit_rating_beats_an_implicit_star(self, tmp_path: Path) -> None:
        # `rr open` stars papers as a side effect of opening them, so a star is much
        # weaker evidence than a rating. Same precedence the recommender uses.
        with PaperStore(tmp_path / "p.db") as store:
            store.upsert_paper(_paper("2607.00001v1", on_topic=True))
            store.star_paper("2607.00001v1")
            store.save_rating("2607.00001v1", 1)
            judgments = load_judgments(store)
        assert judgments.labels == {"2607.00001v1": 0}
        assert judgments.from_stars == 0

    def test_a_neutral_rating_does_not_become_a_star_label(self, tmp_path: Path) -> None:
        # A 3-star paper that was also opened must stay excluded, not sneak back in
        # as relevant through the star table.
        with PaperStore(tmp_path / "p.db") as store:
            store.upsert_paper(_paper("2607.00001v1", on_topic=True))
            store.star_paper("2607.00001v1")
            store.save_rating("2607.00001v1", 3)
            judgments = load_judgments(store)
        assert judgments.labels == {}
        assert judgments.ignored_neutral == 1

    def test_empty_store(self, tmp_path: Path) -> None:
        with PaperStore(tmp_path / "p.db") as store:
            judgments = load_judgments(store)
        assert judgments.labels == {} and judgments.n_judged == 0


class TestCondensedRanking:
    def test_unjudged_papers_are_removed_not_scored_irrelevant(self, tmp_path: Path) -> None:
        """The core methodological choice.

        Judgments are incomplete — a user rates tens of papers out of thousands. If
        unjudged papers counted as irrelevant, precision@10 would be dominated by
        papers the user never looked at, and a perfect ranker would score near zero.
        """
        with PaperStore(tmp_path / "p.db") as store:
            # Two judged relevant papers plus 50 unjudged ones that score *higher*
            # (they are on-topic too, they just were never rated).
            for i in range(50):
                store.upsert_paper(_paper(f"2607.9{i:04d}v1", on_topic=True))
            for i in range(2):
                arxiv_id = f"2607.0000{i}v1"
                store.upsert_paper(_paper(arxiv_id, on_topic=True))
                store.save_rating(arxiv_id, 5)
            judgments = load_judgments(store)
            result = evaluate(store, judgments, _profile(), _cfg(), k=10)
        assert result["n_judged"] == 2  # only the judged papers are in the list
        assert result["precision@k"] == 1.0  # both judged papers are relevant

    def test_only_judged_papers_are_ranked(self, tmp_path: Path) -> None:
        with PaperStore(tmp_path / "p.db") as store:
            store.upsert_paper(_paper("2607.00001v1", on_topic=True))
            store.upsert_paper(_paper("2607.00002v1", on_topic=True))
            store.save_rating("2607.00001v1", 5)
            judgments = load_judgments(store)
            ranked = rank_judged_papers(store, judgments, _profile(), _cfg())
        assert [r["arxiv_id"] for r in ranked] == ["2607.00001v1"]

    def test_recency_uses_each_paper_first_seen(self, tmp_path: Path) -> None:
        """Scored against today's clock, every stored paper is old.

        Recency would collapse to 0 for all of them and the component would vanish
        from every comparison. Scoring as of ``first_seen`` reproduces what the ranker
        produced at digest time. A paper published 200 days ago but *collected the day
        after it appeared* was fresh then, so its recency must be high — which is only
        true if the reference instant is actually threaded through.
        """
        published = datetime.now(UTC) - timedelta(days=200)
        collected = published + timedelta(days=1)
        with PaperStore(tmp_path / "p.db") as store:
            store.upsert_paper(
                _paper("2607.00001v1", on_topic=True, published=published.isoformat())
            )
            store.save_rating("2607.00001v1", 5)
            # upsert_paper stamps first_seen as "now"; backdate it to when this paper
            # would really have been collected.
            store._conn.execute(
                "UPDATE papers SET first_seen = ? WHERE arxiv_id = ?",
                (collected.isoformat(), "2607.00001v1"),
            )
            store._conn.commit()
            judgments = load_judgments(store)
            ranked = rank_judged_papers(store, judgments, _profile(), _cfg())
        # ~1 day old within a 14-day window -> ~0.93. Against today's clock it would
        # be exactly 0.0, so this assertion fails if `now=` is not plumbed through.
        assert ranked[0]["recency_score"] > 0.8


class TestBootstrapDelta:
    def _labels(self, n: int = 10) -> dict[str, int]:
        return {f"p{i}": (1 if i < n // 2 else 0) for i in range(n)}

    def test_a_perfect_and_a_reversed_ranking_are_distinguishable(self) -> None:
        """Pins the bug that made this function a no-op.

        Resampling rank *positions* (the obvious shortcut) scrambles the order, so
        both configs collapse to their base rate and every interval straddles zero.
        The first implementation did exactly that and reported "cannot tell them
        apart" for a perfect ranking versus its exact reverse.
        """
        labels = self._labels()
        perfect = [f"p{i}" for i in range(10)]
        result = bootstrap_delta(perfect, list(reversed(perfect)), labels, "ndcg@k", 10)
        assert result["delta"] < 0
        assert result["high"] < 0  # the whole interval is below zero
        assert result["significant"] == 1.0

    def test_identical_rankings_report_no_difference(self) -> None:
        labels = self._labels()
        ranked = [f"p{i}" for i in range(10)]
        result = bootstrap_delta(ranked, ranked, labels, "ndcg@k", 10)
        assert result["delta"] == 0.0
        assert result["significant"] == 0.0

    def test_is_deterministic(self) -> None:
        # A CI gate built on a flapping interval is worse than none.
        labels = self._labels()
        a = [f"p{i}" for i in range(10)]
        b = list(reversed(a))
        assert bootstrap_delta(a, b, labels) == bootstrap_delta(a, b, labels)

    def test_empty_inputs(self) -> None:
        assert bootstrap_delta([], [], {})["significant"] == 0.0

    def test_supports_each_metric(self) -> None:
        labels = self._labels()
        a = [f"p{i}" for i in range(10)]
        b = list(reversed(a))
        for metric in ("ndcg@k", "precision@k", "recall@k", "mrr"):
            result = bootstrap_delta(a, b, labels, metric, 5)
            assert result["low"] <= result["high"]


class TestCompareConfigs:
    def test_detects_a_real_ranking_regression(self, tmp_path: Path) -> None:
        """The acceptance test for the whole feature.

        Turning off keyword matching must be measurably worse on a corpus where only
        keywords separate relevant from irrelevant.
        """
        with PaperStore(tmp_path / "p.db") as store:
            _seed(store, n=30)
            judgments = load_judgments(store)
            comparison = compare_configs(
                store,
                judgments,
                _profile(),
                _cfg(),
                RankingConfig(w_keyword=1.0, w_category=0.5, w_recency=0.3),
                RankingConfig(w_keyword=0.0, w_category=1.0, w_recency=1.0),
                k=10,
            )
        assert comparison["a"]["ndcg@k"] > comparison["b"]["ndcg@k"]
        assert comparison["bootstrap"]["significant"] == 1.0

    def test_re_scores_rather_than_reading_stored_scores(self, tmp_path: Path) -> None:
        # If it read back paper_scores, both sides would be identical and --compare
        # would be a no-op that always says "not shown".
        with PaperStore(tmp_path / "p.db") as store:
            _seed(store, n=20)
            run_id = store.record_run(["q"], 20, 0)
            store.save_scores(
                run_id,
                [{"arxiv_id": f"2607.{i:05d}v1", "score_total": 0.5} for i in range(20)],
            )
            judgments = load_judgments(store)
            a = evaluate(store, judgments, _profile(), _cfg(), RankingConfig(w_keyword=1.0), 10)
            b = evaluate(
                store,
                judgments,
                _profile(),
                _cfg(),
                RankingConfig(w_keyword=0.0, w_recency=1.0),
                10,
            )
        assert a["ndcg@k"] != b["ndcg@k"]


class TestReports:
    def test_sparsity_warning_below_the_threshold(self, tmp_path: Path) -> None:
        with PaperStore(tmp_path / "p.db") as store:
            _seed(store, n=6)
            judgments = load_judgments(store)
            result = evaluate(store, judgments, _profile(), _cfg(), k=10)
        report = format_report(result, judgments)
        assert "mostly noise" in report

    def test_no_sparsity_warning_when_there_is_enough_data(self, tmp_path: Path) -> None:
        with PaperStore(tmp_path / "p.db") as store:
            _seed(store, n=40)
            judgments = load_judgments(store)
            result = evaluate(store, judgments, _profile(), _cfg(), k=10)
        assert "mostly noise" not in format_report(result, judgments)

    def test_selection_bias_is_always_stated(self, tmp_path: Path) -> None:
        # It cannot be corrected offline, so it must not be left to the docs.
        with PaperStore(tmp_path / "p.db") as store:
            _seed(store, n=40)
            judgments = load_judgments(store)
            result = evaluate(store, judgments, _profile(), _cfg(), k=10)
        report = format_report(result, judgments)
        assert "only cover what an earlier ranking already surfaced" in report

    def test_precision_ceiling_is_explained(self, tmp_path: Path) -> None:
        # A perfect ranking of 5 relevant papers reads as precision@10 = 0.500;
        # without the note that looks like being half wrong.
        with PaperStore(tmp_path / "p.db") as store:
            _seed(store, n=10)
            judgments = load_judgments(store)
            result = evaluate(store, judgments, _profile(), _cfg(), k=10)
        report = format_report(result, judgments)
        assert result["precision@k"] < 1.0
        assert "maxes out at" in report

    def test_report_is_ascii_only(self, tmp_path: Path) -> None:
        # Windows consoles are cp1252; a stray em dash raises UnicodeEncodeError.
        with PaperStore(tmp_path / "p.db") as store:
            _seed(store, n=30)
            judgments = load_judgments(store)
            result = evaluate(store, judgments, _profile(), _cfg(), k=10)
            comparison = compare_configs(
                store, judgments, _profile(), _cfg(), RankingConfig(), RankingConfig(), k=10
            )
        format_report(result, judgments).encode("ascii")
        format_comparison(comparison, judgments).encode("ascii")

    def test_comparison_says_not_shown_when_the_interval_spans_zero(self, tmp_path: Path) -> None:
        with PaperStore(tmp_path / "p.db") as store:
            _seed(store, n=20)
            judgments = load_judgments(store)
            comparison = compare_configs(
                store, judgments, _profile(), _cfg(), RankingConfig(), RankingConfig(), k=10
            )
        assert "NOT SHOWN" in format_comparison(comparison, judgments)


class TestJudgmentsDataclass:
    def test_counts(self) -> None:
        judgments = Judgments(
            labels={"a": 1, "b": 0, "c": 1}, from_ratings=3, from_stars=0, ignored_neutral=2
        )
        assert judgments.n_judged == 3
        assert judgments.n_relevant == 2


class TestStoredSignals:
    """The optional components must be measurable, or the harness answers nothing.

    Without rebuilding these from the store, every optional score arrives as None,
    absent-is-not-zero (correctly) drops it from the weighted sum, and a --compare
    that changes only `w_specter` reports NOT SHOWN on every corpus -- a harness that
    looks like it works.
    """

    def test_only_builds_signals_the_config_asks_for(self, tmp_path: Path) -> None:
        with PaperStore(tmp_path / "p.db") as store:
            _seed(store, n=6)
            ids = list(load_judgments(store).labels)
            store.save_signals([(ids[0], "hn", "800", "u")])
            off = stored_signals(store, ids, RankingConfig())
            on = stored_signals(store, ids, RankingConfig(w_attention=2.0))
        assert "attention" not in off  # w_attention == 0 -> no work done
        assert on["attention"] == {ids[0]: 1.0}

    def test_withdrawal_flags_are_always_applied(self, tmp_path: Path) -> None:
        # Integrity is not an opt-in weight; a withdrawn paper must be penalized in
        # the eval too, or the eval scores a ranking the digest would never show.
        with PaperStore(tmp_path / "p.db") as store:
            _seed(store, n=6)
            ids = list(load_judgments(store).labels)
            store.save_signals([(ids[0], "withdrawn", "comment", None)])
            signals = stored_signals(store, ids, RankingConfig())
        assert signals["withdrawn"] == {ids[0]}

    def test_a_checked_clean_paper_is_not_treated_as_withdrawn(self, tmp_path: Path) -> None:
        with PaperStore(tmp_path / "p.db") as store:
            _seed(store, n=6)
            ids = list(load_judgments(store).labels)
            store.save_signals([(ids[0], "withdrawn", None, None)])
            signals = stored_signals(store, ids, RankingConfig())
        assert "withdrawn" not in signals

    def test_citation_proximity_comes_from_stored_edges(self, tmp_path: Path) -> None:
        with PaperStore(tmp_path / "p.db") as store:
            _seed(store, n=6)
            store.upsert_paper(_paper("2401.00099v1", on_topic=True))
            store.star_paper("2401.00099v1")
            store.save_citations([("2607.00000v1", "2401.00099")])
            ids = list(load_judgments(store).labels)
            signals = stored_signals(store, ids, RankingConfig(w_citation_proximity=1.0))
        assert signals["citation_proximity"] == {"2607.00000v1": 1.0}

    def test_detects_that_over_trusting_a_stored_signal_hurts(self, tmp_path: Path) -> None:
        # Put all the Hacker News buzz on papers the user rated badly. A config that
        # leans on it must measure WORSE -- which only happens if the eval reads it.
        with PaperStore(tmp_path / "p.db") as store:
            _seed(store, n=20)
            store.save_signals([(f"2607.{i:05d}v1", "hn", "800", "u") for i in range(1, 20, 2)])
            judgments = load_judgments(store)
            comparison = compare_configs(
                store,
                judgments,
                _profile(),
                _cfg(),
                RankingConfig(),
                RankingConfig(w_attention=25.0),
                k=10,
            )
        assert comparison["b"]["ndcg@k"] < comparison["a"]["ndcg@k"]
        assert comparison["bootstrap"]["significant"] == 1.0

    def test_a_store_failure_degrades_to_no_signal(self, tmp_path: Path) -> None:
        with PaperStore(tmp_path / "p.db") as store:
            _seed(store, n=4)
            ids = list(load_judgments(store).labels)
            store.close()  # every read now raises
            signals = stored_signals(store, ids, RankingConfig(w_attention=1.0))
        assert signals == {}


class TestBootstrapCalibration:
    def test_false_positive_rate_is_near_nominal(self) -> None:
        """A 90% interval must be wrong about 10% of the time, not 40%.

        Users gate ranking decisions on "interval excludes zero", so a miscalibrated
        interval is worse than no interval at all. Two independent random orderings of
        the same labels are of equal expected quality, so every claimed difference
        between them is a false positive.
        """
        import random as _random

        rng = _random.Random(7)
        ids = [f"p{i}" for i in range(30)]
        labels = {aid: (1 if i < 15 else 0) for i, aid in enumerate(ids)}
        trials, hits = 120, 0
        for _ in range(trials):
            a, b = ids[:], ids[:]
            rng.shuffle(a)
            rng.shuffle(b)
            if bootstrap_delta(a, b, labels, "ndcg@k", 10)["significant"]:
                hits += 1
        assert hits / trials < 0.25, f"false-positive rate {hits / trials:.0%} is too high"


class TestBaselineRegressionGate:
    """`--baseline` without a comparison is a diary, not a gate.

    The roadmap and README promise "CI catches a ranking regression"; that requires
    something to actually read a stored snapshot back.
    """

    def _snapshot(self, ndcg: float = 0.80, k: int = 10, n_judged: int = 30) -> dict:
        return {
            "metrics": {"ndcg@k": ndcg},
            "k": k,
            "n_judged": n_judged,
            "label": "before",
        }

    def test_improvement_is_not_a_regression(self) -> None:
        check = compare_to_baseline({"ndcg@k": 0.85, "k": 10, "n_judged": 30}, self._snapshot())
        assert check["comparable"] and not check["regressed"]

    def test_a_real_drop_is_a_regression(self) -> None:
        check = compare_to_baseline({"ndcg@k": 0.60, "k": 10, "n_judged": 30}, self._snapshot())
        assert check["regressed"]
        assert "REGRESSION" in format_baseline_check(check)

    def test_noise_sized_movement_is_tolerated(self) -> None:
        # Metrics shift slightly whenever a new rating lands; an exact-equality gate
        # would fail constantly and get switched off.
        check = compare_to_baseline({"ndcg@k": 0.79, "k": 10, "n_judged": 30}, self._snapshot())
        assert not check["regressed"]

    def test_a_different_k_is_refused_not_differenced(self) -> None:
        check = compare_to_baseline({"ndcg@k": 0.60, "k": 5, "n_judged": 30}, self._snapshot())
        assert not check["comparable"]
        assert "k=" in format_baseline_check(check)

    def test_a_changed_judged_set_is_flagged(self) -> None:
        check = compare_to_baseline({"ndcg@k": 0.81, "k": 10, "n_judged": 44}, self._snapshot())
        assert check["judgments_changed"]
        assert "new ratings" in format_baseline_check(check)

    def test_a_baseline_missing_the_metric_is_refused(self) -> None:
        check = compare_to_baseline(
            {"ndcg@k": 0.8, "k": 10, "n_judged": 30},
            {"metrics": {"mrr": 1.0}, "k": 10, "n_judged": 30},
        )
        assert not check["comparable"]


class TestUnmodelledKnobs:
    def test_a_hybrid_only_difference_is_flagged_not_reported_as_equal(
        self, tmp_path: Path
    ) -> None:
        """Hybrid RRF reorders with a corpus-wide BM25 index at digest time.

        A condensed list of judged papers cannot reproduce it, so two configs differing
        only in `hybrid` rank identically here — and reporting that as "cannot tell them
        apart" is indistinguishable from a genuine null result.
        """
        with PaperStore(tmp_path / "p.db") as store:
            _seed(store, n=20)
            judgments = load_judgments(store)
            comparison = compare_configs(
                store,
                judgments,
                _profile(),
                _cfg(),
                RankingConfig(hybrid=False),
                RankingConfig(hybrid=True),
                k=10,
            )
        assert comparison["unmodelled"] == ["hybrid"]
        rendered = format_comparison(comparison, judgments)
        assert "does not reproduce" in rendered

    def test_no_warning_when_the_configs_differ_in_modelled_knobs(self, tmp_path: Path) -> None:
        with PaperStore(tmp_path / "p.db") as store:
            _seed(store, n=20)
            judgments = load_judgments(store)
            comparison = compare_configs(
                store,
                judgments,
                _profile(),
                _cfg(),
                RankingConfig(w_keyword=1.0),
                RankingConfig(w_keyword=0.5),
                k=10,
            )
        assert comparison["unmodelled"] == []
        assert "does not reproduce" not in format_comparison(comparison, judgments)


class TestCeilingNotes:
    def test_printed_precision_never_exceeds_its_printed_ceiling(self, tmp_path: Path) -> None:
        """The ceiling must be computed the way precision actually is.

        `precision_at_k` divides by the number of items in the list, which may be fewer
        than k. Dividing the ceiling by k instead made the report contradict itself in
        the very next line.
        """
        for n in (4, 6, 10, 24):
            with PaperStore(tmp_path / f"p{n}.db") as store:
                _seed(store, n=n)
                judgments = load_judgments(store)
                result = evaluate(store, judgments, _profile(), _cfg(), k=10)
            window = min(result["k"], result["n_judged"]) or 1
            ceiling = min(1.0, judgments.n_relevant / window)
            assert result["precision@k"] <= ceiling + 1e-9, (
                f"n={n}: precision {result['precision@k']} exceeds ceiling {ceiling}"
            )

    def test_recall_ceiling_is_explained_too(self, tmp_path: Path) -> None:
        # Same trap as precision: 15 relevant papers cannot all fit in a top-10.
        with PaperStore(tmp_path / "p.db") as store:
            _seed(store, n=30)
            judgments = load_judgments(store)
            result = evaluate(store, judgments, _profile(), _cfg(), k=10)
        report = format_report(result, judgments)
        assert result["recall@k"] < 1.0
        assert "recall@10 maxes out" in report
