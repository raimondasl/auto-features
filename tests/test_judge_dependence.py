"""Pin the judge-dependence figures: which band claims move with the judge, scorer or band.

`evals/judge_dependence.json` holds every paper of the three dual-judged score-2 bands (aug20, H
and L) with both judges' labels and both scorers' exp09. These tests recompute the quoted figures
from those per-paper rows, with the estimator the script used, rather than reading back the
summary the script wrote. They then check the summary against the same recomputation, so the
summary cannot drift from its rows either.

What is pinned, because prose depends on it:

* Which judge a scorer looks better under is not stable. On band H, gpt-4o-mini's point is higher
  under GPT-5.5 (0.726 against 0.693), and gpt-4.1-mini's is higher under Sonnet (0.700 against
  0.743). No overall ordering gap has an interval that excludes zero, so this is noise around a
  shared ordering, not a judge preference.
* Level depends on the judge far more than ordering does. On aug20, GPT-5.5 calls 38 points more
  of the band actionable than Sonnet, while the two AUCs differ by 0.027.
* The stage's value against showing nothing changes sign with the judge, on all three bands.
* gpt-4.1-mini compresses the band toward 7, just above the frozen cut, rather than spreading it.
  NR-65 said the opposite, and C-39 corrects it from these figures.
* gpt-4o-mini's legacy-minus-scientific gap is a GPT-5.5 reading. Its interval excludes zero on
  bands H and L under GPT-5.5 and spans zero under Sonnet. The paired contrast between the two
  judges excludes zero on both bands. C-38 withdraws the split as a property of the cohort.

Intervals are recomputed from the rows with the script's own draws, so every stored endpoint must
come back exactly. The sign of every interval the prose relies on is pinned as well.
"""

from __future__ import annotations

import hashlib
import json
import statistics
from pathlib import Path
from typing import Any

# The estimator the experiment used, not a second implementation. evals/ is on pytest's
# pythonpath, so this is the same module object the script imports.
import band_testbeds as tb
import finescale_model_transfer as fmt
import pytest

from anonymous import finescale

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "evals" / "judge_dependence.json"
TRANSFER = ROOT / "evals" / "finescale_model_transfer.json"
# The tracked file each run band's papers, GPT-5.5 labels and gpt-4o-mini scores were joined from.
CONTROLS = {
    "H": ROOT / "evals" / "finescale_current_gate.json",
    "L": ROOT / "evals" / "finescale_current_gate_luna.json",
}

SCIENTIFIC = {
    "bio-align",
    "bio-kmer",
    "bio-mdsim",
    "bio-mdtraj",
    "bio-scvi",
    "bio-singlecell",
    "mat-chgpot",
    "mat-descriptors",
    "mat-featurize",
    "mat-mlip",
    "mat-phonon",
    "mat-toolkit",
}
BANDS = ("aug20", "H", "L")
SIZES = {"aug20": 324, "H": 328, "L": 315}
COHORTS = ("overall", "legacy", "scientific")
DIVISORS = {"overall": 37, "legacy": 25, "scientific": 12}
JUDGES = ("gpt", "sonnet")
SCORERS = {"4o": ("s4o", "adm4o"), "41": ("s41", "adm41")}


@pytest.fixture(scope="module")
def artifact() -> dict:
    # Asserted rather than skipped. The artifact is tracked, so its absence is a broken
    # repository, and a skip here would be this project's own void-not-null failure: the
    # suite would stay green while the figures it guards went unchecked.
    assert ARTIFACT.is_file(), f"{ARTIFACT} is tracked and must be present"
    return json.loads(ARTIFACT.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def recomputed(artifact: dict) -> dict[str, dict[str, Any]]:
    """Each band's summary, rebuilt from the rows by the script's own analyse and band_summary.

    Same seed, same draws and the same interval indices, so every stored estimate must come back
    exactly. It takes about 30 seconds for the three bands, so it runs once for the module.
    """
    import judge_dependence as jd

    out = {}
    for band in BANDS:
        rows, cases, scorers = _rows(artifact, band), artifact["cases"][band], list(_scorers(band))
        point, boot = jd.analyse(rows, cases, scorers)
        out[band] = jd.band_summary(rows, point, boot, cases, scorers)
    return out


@pytest.fixture(scope="module")
def transfer() -> dict:
    assert TRANSFER.is_file(), f"{TRANSFER} is tracked and must be present"
    return json.loads(TRANSFER.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def controls() -> dict[str, dict]:
    out = {}
    for band, path in CONTROLS.items():
        assert path.is_file(), f"{path} is tracked and must be present"
        out[band] = json.loads(path.read_text(encoding="utf-8"))
    return out


def _rows(artifact: dict, band: str, cohort: str = "overall") -> list[dict]:
    rows = [r for r in artifact["rows"] if r["band"] == band]
    return rows if cohort == "overall" else [r for r in rows if r["cohort"] == cohort]


def _ok(r: dict, judge: str) -> bool:
    return bool(r[judge] >= tb.ACTIONABLE)


def _auc(rows: list[dict], scorer: str, judge: str) -> float:
    key = SCORERS[scorer][0]
    return float(tb.auc([r[key] for r in rows], [_ok(r, judge) for r in rows]))


def _base(rows: list[dict], judge: str) -> float:
    return sum(_ok(r, judge) for r in rows) / len(rows)


def _net(rows: list[dict], judge: str, adm: str | None = None) -> int:
    """+1 per actionable paper, -2 per other one, over the admitted papers or all of them."""
    return sum(1 if _ok(r, judge) else -2 for r in rows if adm is None or r[adm])


def _scorers(band: str) -> tuple[str, ...]:
    return ("4o",) if band == "aug20" else ("4o", "41")


# The five band-scorer readings: gpt-4o-mini on all three bands, gpt-4.1-mini on H and L.
PAIRS = [(band, scorer) for band in BANDS for scorer in _scorers(band)]


def _lo(e: dict) -> float:
    return float(e["ci"][0])


def _hi(e: dict) -> float:
    return float(e["ci"][1])


def _spans_zero(e: dict) -> bool:
    return _lo(e) < 0 < _hi(e)


def _excludes_zero(e: dict) -> bool:
    return _lo(e) > 0 or _hi(e) < 0


class TestTheArtifactIsComplete:
    def test_three_bands_of_the_declared_sizes(self, artifact: dict) -> None:
        for band, size in SIZES.items():
            assert len(_rows(artifact, band)) == size, band
            assert artifact["summary"]["counts"][band]["papers"] == size

    def test_every_row_has_both_labels(self, artifact: dict) -> None:
        """A paper without a Sonnet verdict is a void row, never a null one."""
        for r in artifact["rows"]:
            for judge in JUDGES:
                assert r[judge] in (0, 1, 2, 3), f"{r['band']} {r['case']}/{r['id']} {judge}"

    def test_every_row_has_its_scores(self, artifact: dict) -> None:
        for r in artifact["rows"]:
            assert isinstance(r["s4o"], float)
            assert isinstance(r["adm4o"], bool)
            if r["band"] == "aug20":
                assert r["s41"] is None and r["adm41"] is None
            else:
                assert isinstance(r["s41"], float)
                assert isinstance(r["adm41"], bool)

    def test_papers_are_unique_within_a_band(self, artifact: dict) -> None:
        for band in BANDS:
            keys = [(r["case"], r["id"]) for r in _rows(artifact, band)]
            assert len(keys) == len(set(keys)), band

    def test_every_band_divides_by_the_37_benchmark_cases(self, artifact: dict) -> None:
        """Per-repository values divide by every case, including those with no band paper."""
        for band in BANDS:
            cases = artifact["cases"][band]
            assert len(cases) == len(set(cases)) == 37
            assert len([c for c in cases if c in SCIENTIFIC]) == 12
            for r in _rows(artifact, band):
                assert r["case"] in cases
                assert r["cohort"] == ("scientific" if r["case"] in SCIENTIFIC else "legacy")

    def test_admission_is_the_frozen_map(self, artifact: dict) -> None:
        m = artifact["summary"]["map"]
        assert m["slope"] == finescale.SLOPE
        assert m["intercept"] == finescale.INTERCEPT
        assert m["threshold"] == finescale.SHOW_THRESHOLD
        assert artifact["summary"]["aug20_admission"]["disagreements_with_recomputed"] == 0
        for r in artifact["rows"]:
            for scorer, (score, adm) in SCORERS.items():
                if r[score] is None:
                    continue
                want = finescale.probability(r[score]) >= finescale.SHOW_THRESHOLD
                assert r[adm] == want, f"{r['band']} {r['case']}/{r['id']} {scorer}"


class TestTheRowsAreTheirSources:
    """Band H and L rows against the tracked files the script joined them from.

    The run files and the Sonnet cache the script also read are untracked, so these are the joins
    a checkout can still verify: papers, GPT-5.5 labels and gpt-4o-mini scores from each band's
    control file, gpt-4.1-mini scores from NR-65's first pass, and the Sonnet labels against the
    label-set fingerprint NR-65 recorded.
    """

    @pytest.mark.parametrize("band", ["H", "L"])
    def test_papers_labels_and_4o_scores_are_the_control_files(
        self, artifact: dict, controls: dict[str, dict], band: str
    ) -> None:
        control = controls[band]
        assert control["summary"]["run"] == artifact["summary"]["inputs"][band]["run"]
        rows = {(r["case"], r["id"]): r for r in _rows(artifact, band)}
        by_key = {(c["case"], c["id"]): c for c in control["rows"]}
        assert set(rows) == set(by_key)
        for key, r in rows.items():
            c = by_key[key]
            assert (r["gpt"], r["s4o"]) == (c["judge"], c["exp09"]), key

    @pytest.mark.parametrize("band", ["H", "L"])
    def test_4_1_scores_are_nr65s_product_parser_first_pass(
        self, artifact: dict, transfer: dict, band: str
    ) -> None:
        first = transfer["rows"]["first"]
        for r in _rows(artifact, band):
            assert r["s41"] == first[f"{r['case']}/{r['id']}"]["product_exp"], (r["case"], r["id"])

    @pytest.mark.parametrize("band", ["H", "L"])
    def test_sonnet_labels_are_the_label_set_nr65_fingerprinted(
        self, artifact: dict, transfer: dict, band: str
    ) -> None:
        """NR-65's step 4 recipe, through its own sonnet_fp: papers sorted by (case, id), each
        with a verdict and outside a drifted case. None drifted and none was void, so that is
        every band paper. sonnet_fp keys scores by the versioned id, and a paper's id stands in
        for it here, since the fingerprinted text uses only the unversioned one."""
        sonnet = transfer["sonnet"]
        assert sonnet["drifted"] == [] and sonnet["void"] == []
        rows = sorted(_rows(artifact, band), key=lambda r: (r["case"], r["id"]))
        papers = [fmt.BandPaper(r["case"], r["id"], r["id"], "", "", r["gpt"]) for r in rows]
        scores = {(r["case"], r["id"]): r["sonnet"] for r in rows}
        recorded = sonnet["label_set"][band]
        assert fmt.sonnet_fp(papers, scores) == recorded["fingerprint"]
        assert len(scores) == recorded["count"] == SIZES[band]
        # The same fingerprint as a bare hash, so a changed sonnet_fp cannot pass by agreeing
        # with itself.
        text = "".join(f"{r['case']}/{r['id']}/{r['sonnet']}\n" for r in rows)
        assert hashlib.sha256(text.encode("utf-8")).hexdigest() == recorded["fingerprint"]


# The figures the log quotes, recomputed from the rows. Each tolerance is half the last digit the
# log prints, so a misprint of one in that digit fails.
DP3 = 0.0005
DP2 = 0.005
AUC = {  # (band, scorer): overall AUC under (GPT-5.5, Sonnet)
    ("aug20", "4o"): (0.729, 0.702),
    ("H", "4o"): (0.726, 0.693),
    ("H", "41"): (0.700, 0.743),
    ("L", "4o"): (0.675, 0.694),
    ("L", "41"): (0.711, 0.743),
}
BASE = {"aug20": (0.873, 0.494), "H": (0.832, 0.445), "L": (0.737, 0.273)}  # GPT-5.5, Sonnet
LEVEL = {"aug20": 0.380, "H": 0.387, "L": 0.463}
COHORT_GAP = {  # (band, scorer): legacy AUC minus scientific AUC under (GPT-5.5, Sonnet)
    ("aug20", "4o"): (0.077, -0.068),
    ("H", "4o"): (0.199, 0.003),
    ("H", "41"): (0.026, -0.027),
    ("L", "4o"): (0.229, 0.037),
    ("L", "41"): (0.069, -0.004),
}
DID_GPT = {"H": 0.173, "L": 0.160}  # gpt-4o-mini's cohort gap minus gpt-4.1-mini's, GPT-5.5
ADMITTED = {
    ("aug20", "4o"): 244,
    ("H", "4o"): 231,
    ("H", "41"): 249,
    ("L", "4o"): 173,
    ("L", "41"): 181,
}
# (band, scorer, judge): stage minus none and stage minus all per repository over 37, and the
# precision of the admitted papers.
VALUE = {
    ("aug20", "4o", "gpt"): (4.97, -0.46, 0.918),
    ("aug20", "4o", "sonnet"): (-1.92, 2.62, 0.570),
    ("H", "4o", "gpt"): (4.38, -0.03, 0.900),
    ("H", "4o", "sonnet"): (-2.68, 3.22, 0.524),
    ("H", "41", "gpt"): (4.46, 0.05, 0.888),
    ("H", "41", "sonnet"): (-2.51, 3.38, 0.542),
    ("L", "4o", "gpt"): (2.24, 0.46, 0.827),
    ("L", "4o", "sonnet"): (-4.08, 5.97, 0.376),
    ("L", "41", "gpt"): (2.62, 0.84, 0.845),
    ("L", "41", "sonnet"): (-4.03, 6.03, 0.392),
}


class TestTheQuotedNumbersAreRecomputable:
    """Recompute from the rows so prose cannot drift from the artifact."""

    @pytest.mark.parametrize(("band", "scorer"), list(AUC))
    def test_auc_under_each_judge(self, artifact: dict, band: str, scorer: str) -> None:
        """On band H, gpt-4o-mini reads higher under GPT-5.5 and gpt-4.1-mini under Sonnet."""
        rows = _rows(artifact, band)
        got = tuple(_auc(rows, scorer, judge) for judge in JUDGES)
        assert got == pytest.approx(AUC[(band, scorer)], abs=DP3)

    @pytest.mark.parametrize("band", BANDS)
    def test_base_rate_and_level(self, artifact: dict, band: str) -> None:
        rows = _rows(artifact, band)
        base = tuple(_base(rows, judge) for judge in JUDGES)
        assert base == pytest.approx(BASE[band], abs=DP3)
        assert base[0] - base[1] == pytest.approx(LEVEL[band], abs=DP3)

    def test_aug20_judges_differ_in_level_not_ordering(self, artifact: dict) -> None:
        """GPT-5.5 calls 38 points more of the band actionable, and the AUCs differ by 0.027."""
        rows = _rows(artifact, "aug20")
        level = _base(rows, "gpt") - _base(rows, "sonnet")
        ordering = _auc(rows, "4o", "gpt") - _auc(rows, "4o", "sonnet")
        assert level == pytest.approx(0.380, abs=DP3)
        assert ordering == pytest.approx(0.027, abs=DP3)

    def test_band_h_cohorts_under_both_judges(self, artifact: dict) -> None:
        """The NR-64 legacy/scientific split is a GPT-5.5 finding. Under Sonnet it is gone."""
        legacy, sci = _rows(artifact, "H", "legacy"), _rows(artifact, "H", "scientific")
        assert (len(legacy), len(sci)) == (224, 104)
        assert _auc(legacy, "4o", "gpt") == pytest.approx(0.785, abs=DP3)
        assert _auc(sci, "4o", "gpt") == pytest.approx(0.587, abs=DP3)
        assert _auc(legacy, "4o", "sonnet") == pytest.approx(0.699, abs=DP3)
        assert _auc(sci, "4o", "sonnet") == pytest.approx(0.695, abs=DP3)

    @pytest.mark.parametrize(("band", "scorer"), list(COHORT_GAP))
    def test_cohort_gap(self, artifact: dict, band: str, scorer: str) -> None:
        legacy, sci = _rows(artifact, band, "legacy"), _rows(artifact, band, "scientific")
        got = tuple(_auc(legacy, scorer, j) - _auc(sci, scorer, j) for j in JUDGES)
        assert got == pytest.approx(COHORT_GAP[(band, scorer)], abs=DP3)

    @pytest.mark.parametrize("band", list(DID_GPT))
    def test_difference_in_differences_under_gpt(self, artifact: dict, band: str) -> None:
        legacy, sci = _rows(artifact, band, "legacy"), _rows(artifact, band, "scientific")
        gap = {s: _auc(legacy, s, "gpt") - _auc(sci, s, "gpt") for s in ("4o", "41")}
        assert gap["4o"] - gap["41"] == pytest.approx(DID_GPT[band], abs=DP3)

    @pytest.mark.parametrize(("band", "scorer", "judge"), list(VALUE))
    def test_stage_value_per_repository(
        self, artifact: dict, band: str, scorer: str, judge: str
    ) -> None:
        rows, adm = _rows(artifact, band), SCORERS[scorer][1]
        minus_none, minus_all, precision = VALUE[(band, scorer, judge)]
        stage, every = _net(rows, judge, adm), _net(rows, judge)
        shown = [r for r in rows if r[adm]]
        assert stage / 37 == pytest.approx(minus_none, abs=DP2)
        assert (stage - every) / 37 == pytest.approx(minus_all, abs=DP2)
        assert len(shown) == ADMITTED[(band, scorer)]
        assert sum(_ok(r, judge) for r in shown) / len(shown) == pytest.approx(precision, abs=DP3)

    def test_the_stage_changes_sign_with_the_judge(self, artifact: dict) -> None:
        """Both scorers' stages beat showing nothing under GPT-5.5 and lose to it under Sonnet,
        on every band. The sign follows the judge's strictness, not the band or the scorer."""
        for band, scorer in PAIRS:
            rows, adm = _rows(artifact, band), SCORERS[scorer][1]
            assert _net(rows, "gpt", adm) / 37 > 0, (band, scorer)
            assert _net(rows, "sonnet", adm) / 37 < 0, (band, scorer)

    def test_aug20_cohort_stage_values_per_repository(self, artifact: dict) -> None:
        """second_judge_band.py's stage value against showing all: -1.25 and +3.75 per scientific
        repository. Against showing nothing, Sonnet's is -1.17 per scientific repository."""
        for cohort, want in (("scientific", (-1.25, 3.75)), ("legacy", (-0.08, 2.08))):
            rows = _rows(artifact, "aug20", cohort)
            got = tuple((_net(rows, j, "adm4o") - _net(rows, j)) / DIVISORS[cohort] for j in JUDGES)
            assert got == pytest.approx(want, abs=DP2), cohort
        sci = _rows(artifact, "aug20", "scientific")
        assert _net(sci, "sonnet", "adm4o") / DIVISORS["scientific"] == pytest.approx(
            -1.17, abs=DP2
        )


class TestTheGateIsRecorded:
    """C-37: a band's gate is read from ranking_config, never from pool_config alone."""

    def test_band_h_is_the_shipped_haiku_gate(self, artifact: dict) -> None:
        gate = artifact["summary"]["gate"]["H"]
        assert (gate["provider"], gate["model"]) == ("claude", "claude-haiku-4-5")

    def test_band_l_is_the_luna_gate(self, artifact: dict) -> None:
        gate = artifact["summary"]["gate"]["L"]
        assert (gate["provider"], gate["model"]) == ("openai", "gpt-5.6-luna")

    def test_aug20_says_its_gate_was_not_recorded(self, artifact: dict) -> None:
        gate = artifact["summary"]["gate"]["aug20"]
        assert gate["model"] == "claude-haiku-4-5"
        assert gate["read_from"].startswith("not recorded")


class TestTheSummaryIsTheRows:
    """Every stored point equals the recomputation from the rows."""

    def test_auc_points(self, artifact: dict) -> None:
        bands = artifact["summary"]["bands"]
        for band in BANDS:
            for scorer in _scorers(band):
                for judge in JUDGES:
                    for cohort in COHORTS:
                        rows = _rows(artifact, band, cohort)
                        e = bands[band]["auc"][scorer][judge][cohort]
                        where = f"{band} {scorer} {judge} {cohort}"
                        assert e["point"] == pytest.approx(_auc(rows, scorer, judge), abs=1e-12)
                        assert e["n"] == len(rows), where
                        assert e["base_rate"] == pytest.approx(_base(rows, judge), abs=1e-12)

    def test_judge_gap_points(self, artifact: dict) -> None:
        bands = artifact["summary"]["bands"]
        for band in BANDS:
            for scorer in _scorers(band):
                for cohort in COHORTS:
                    rows = _rows(artifact, band, cohort)
                    e = bands[band]["judge_gap"][scorer][cohort]
                    ordering = _auc(rows, scorer, "gpt") - _auc(rows, scorer, "sonnet")
                    level = _base(rows, "gpt") - _base(rows, "sonnet")
                    assert e["ordering"]["point"] == pytest.approx(ordering, abs=1e-12)
                    assert e["level"]["point"] == pytest.approx(level, abs=1e-12)

    def test_cohort_gap_and_difference_in_differences(self, artifact: dict) -> None:
        bands = artifact["summary"]["bands"]
        for band in BANDS:
            legacy = _rows(artifact, band, "legacy")
            sci = _rows(artifact, band, "scientific")
            gaps = {}
            for scorer in _scorers(band):
                stored = bands[band]["cohort_gap"][scorer]
                # One cell per judge, and the paired contrast between them.
                assert set(stored) == {*JUDGES, "gpt_minus_sonnet"}
                for judge in JUDGES:
                    gap = _auc(legacy, scorer, judge) - _auc(sci, scorer, judge)
                    gaps[(scorer, judge)] = gap
                    assert stored[judge]["point"] == pytest.approx(gap, abs=1e-12)
                contrast = gaps[(scorer, "gpt")] - gaps[(scorer, "sonnet")]
                assert stored["gpt_minus_sonnet"]["point"] == pytest.approx(contrast, abs=1e-12)
            if band != "aug20":
                for judge in JUDGES:
                    did = gaps[("4o", judge)] - gaps[("41", judge)]
                    assert bands[band]["did"][judge]["point"] == pytest.approx(did, abs=1e-12)

    def test_value_points(self, artifact: dict) -> None:
        bands = artifact["summary"]["bands"]
        for band in BANDS:
            for scorer in _scorers(band):
                adm = SCORERS[scorer][1]
                for judge in JUDGES:
                    for cohort in COHORTS:
                        rows = _rows(artifact, band, cohort)
                        n = DIVISORS[cohort]
                        v = bands[band]["value"][scorer][judge][cohort]
                        stage, every = _net(rows, judge, adm), _net(rows, judge)
                        shown = [r for r in rows if r[adm]]
                        assert v["n_cases"] == n
                        assert v["totals"] == {"stage": stage, "show_all": every, "show_none": 0}
                        assert v["admitted"] == len(shown)
                        precision = sum(_ok(r, judge) for r in shown) / len(shown)
                        assert v["precision"] == pytest.approx(precision, abs=1e-12)
                        got_none = v["stage_minus_none"]["point"]
                        got_all = v["stage_minus_all"]["point"]
                        assert got_none == pytest.approx(stage / n, abs=1e-12)
                        assert got_all == pytest.approx((stage - every) / n, abs=1e-12)

    def test_every_interval_brackets_its_point(self, artifact: dict) -> None:
        found = []

        def walk(node: Any, path: str) -> None:
            if isinstance(node, dict):
                if "point" in node and "ci" in node:
                    found.append(path)
                    lo, hi = node["ci"]
                    assert lo <= node["point"] <= hi, path
                    assert 0 <= node["skipped_draws"] < artifact["summary"]["bootstrap"]["draws"]
                    return
                for key, child in node.items():
                    walk(child, f"{path}/{key}")

        walk(artifact["summary"]["bands"], "")
        assert len(found) > 100, "the walk found too few estimates to have checked the summary"


class TestTheIntervalsAreTheDraws:
    """Every stored interval, recomputed from the rows by the script's own bootstrap."""

    def test_the_bootstrap_is_the_scripts(self, artifact: dict) -> None:
        import judge_dependence as jd

        boot = artifact["summary"]["bootstrap"]
        assert (boot["seed"], boot["draws"]) == (jd.SEED, jd.DRAWS) == (20260921, 4000)

    @pytest.mark.parametrize("band", BANDS)
    def test_every_estimate_is_the_recomputation(
        self, artifact: dict, recomputed: dict[str, dict[str, Any]], band: str
    ) -> None:
        """Walk the stored band and its recomputation together. Every leaf must agree: each
        point, interval endpoint and skipped-draw count, and the counts beside them. The score
        distribution is not part of band_summary and is checked against the rows below."""
        stored, fresh = artifact["summary"]["bands"][band], recomputed[band]
        extra = {"score_distribution"} if band != "aug20" else set()
        assert set(stored) == set(fresh) | extra
        found: list[str] = []

        def walk(got: Any, want: Any, path: str) -> None:
            if isinstance(want, dict):
                assert isinstance(got, dict) and set(got) == set(want), path
                if "point" in want and "ci" in want:
                    found.append(path)
                for key in want:
                    walk(got[key], want[key], f"{path}/{key}")
            elif isinstance(want, list):
                assert isinstance(got, list) and len(got) == len(want), path
                for i, (g, w) in enumerate(zip(got, want, strict=True)):
                    walk(g, w, f"{path}[{i}]")
            elif isinstance(want, float):
                assert got == pytest.approx(want, abs=1e-12), path
            else:
                assert got == want, path

        for key in fresh:
            walk(stored[key], fresh[key], f"{band}/{key}")
        # Per scorer: 6 AUCs, 3 cohort-gap cells, 6 judge gaps and 12 stage values; and 2
        # differences in differences on a band with both scorers.
        assert len(found) == (27 if band == "aug20" else 56), found


class TestTheIntervalSigns:
    """The sign of every interval the log's prose relies on.

    Read from the stored summary, which the recomputation above proves is the rows'. A draw
    change that moved one of these across zero would change what the prose may say.
    """

    def test_no_overall_ordering_gap_excludes_zero(self, artifact: dict) -> None:
        """Which judge a scorer looks better under is noise around a shared ordering."""
        bands = artifact["summary"]["bands"]
        for band, scorer in PAIRS:
            e = bands[band]["judge_gap"][scorer]["overall"]["ordering"]
            assert _spans_zero(e), (band, scorer)

    def test_two_within_cohort_ordering_gaps_exclude_zero(self, artifact: dict) -> None:
        """Of the ten cohort cells, only gpt-4o-mini's band H legacy and band L scientific."""
        bands = artifact["summary"]["bands"]
        cells = {
            (band, scorer, cohort): bands[band]["judge_gap"][scorer][cohort]["ordering"]
            for band, scorer in PAIRS
            for cohort in ("legacy", "scientific")
        }
        assert len(cells) == 10
        excluded = {key for key, e in cells.items() if _excludes_zero(e)}
        assert excluded == {("H", "4o", "legacy"), ("L", "4o", "scientific")}
        assert all(_spans_zero(e) for key, e in cells.items() if key not in excluded)

    def test_the_level_gap_is_above_0_3_on_every_band(self, artifact: dict) -> None:
        bands = artifact["summary"]["bands"]
        for band, scorer in PAIRS:
            assert _lo(bands[band]["judge_gap"][scorer]["overall"]["level"]) > 0.3, band

    def test_the_stage_against_showing_nothing_changes_sign_with_the_judge(
        self, artifact: dict
    ) -> None:
        bands = artifact["summary"]["bands"]
        for band, scorer in PAIRS:
            value = bands[band]["value"][scorer]
            assert _lo(value["gpt"]["overall"]["stage_minus_none"]) > 0, (band, scorer)
            assert _hi(value["sonnet"]["overall"]["stage_minus_none"]) < 0, (band, scorer)

    def test_the_stage_against_showing_all(self, artifact: dict) -> None:
        """No different from showing all under GPT-5.5, and better than it under Sonnet."""
        bands = artifact["summary"]["bands"]
        for band, scorer in PAIRS:
            value = bands[band]["value"][scorer]
            assert _spans_zero(value["gpt"]["overall"]["stage_minus_all"]), (band, scorer)
            assert _lo(value["sonnet"]["overall"]["stage_minus_all"]) > 0, (band, scorer)

    def test_aug20_scientific_stage_against_showing_all(self, artifact: dict) -> None:
        """Worse than showing all under GPT-5.5 and better than it under Sonnet."""
        value = artifact["summary"]["bands"]["aug20"]["value"]["4o"]
        assert _hi(value["gpt"]["scientific"]["stage_minus_all"]) < 0
        assert _lo(value["sonnet"]["scientific"]["stage_minus_all"]) > 0

    def test_the_cohort_gap_is_a_gpt_4o_mini_and_gpt_5_5_reading(self, artifact: dict) -> None:
        bands = artifact["summary"]["bands"]
        for band in ("H", "L"):
            assert _lo(bands[band]["cohort_gap"]["4o"]["gpt"]) > 0, band
        assert _spans_zero(bands["aug20"]["cohort_gap"]["4o"]["gpt"])
        assert len(PAIRS) == 5
        for band, scorer in PAIRS:
            assert _spans_zero(bands[band]["cohort_gap"][scorer]["sonnet"]), (band, scorer)
        for band in ("H", "L"):
            for judge in JUDGES:
                assert _spans_zero(bands[band]["cohort_gap"]["41"][judge]), (band, judge)

    def test_the_judges_differ_on_gpt_4o_minis_cohort_gap(self, artifact: dict) -> None:
        """The paired contrast, GPT-5.5's gap minus Sonnet's on the same draws."""
        bands = artifact["summary"]["bands"]
        for band in ("H", "L"):
            assert _lo(bands[band]["cohort_gap"]["4o"]["gpt_minus_sonnet"]) > 0, band
        assert _spans_zero(bands["aug20"]["cohort_gap"]["4o"]["gpt_minus_sonnet"])

    def test_the_difference_in_differences(self, artifact: dict) -> None:
        bands = artifact["summary"]["bands"]
        for band in ("H", "L"):
            assert _lo(bands[band]["did"]["gpt"]) > 0, band
            assert _spans_zero(bands[band]["did"]["sonnet"]), band

    def test_admitted_precision_straddles_the_threshold(self, artifact: dict) -> None:
        """Above 2/3 under GPT-5.5 and below it under Sonnet, for every reading."""
        bands = artifact["summary"]["bands"]
        for band, scorer in PAIRS:
            value = bands[band]["value"][scorer]
            assert value["gpt"]["overall"]["precision"] > 2 / 3, (band, scorer)
            assert value["sonnet"]["overall"]["precision"] < 2 / 3, (band, scorer)


class TestTheScoreDistribution:
    """C-39's figures: gpt-4.1-mini is narrower than gpt-4o-mini, not wider, and piles up above
    the cut. Recomputed from the rows against the cut the frozen map implies."""

    @staticmethod
    def _cut() -> float:
        import math

        t = finescale.SHOW_THRESHOLD
        return (math.log(t / (1 - t)) - finescale.INTERCEPT) / finescale.SLOPE

    @staticmethod
    def _scores(artifact: dict, band: str, scorer: str) -> list[float]:
        return sorted(r[SCORERS[scorer][0]] for r in _rows(artifact, band))

    def test_the_cut_is_where_the_map_crosses_two_thirds(self) -> None:
        cut = self._cut()
        assert cut == pytest.approx(6.721, abs=0.001)
        assert finescale.probability(cut) == pytest.approx(finescale.SHOW_THRESHOLD, abs=1e-9)

    @pytest.mark.parametrize(
        ("band", "pile_4o", "pile_41", "top_4o", "top_41"),
        [("L", 61, 112, 27, 10), ("H", 60, 129, 69, 17)],
    )
    def test_gpt_4_1_mini_piles_up_just_above_the_cut(
        self, artifact: dict, band: str, pile_4o: int, pile_41: int, top_4o: int, top_41: int
    ) -> None:
        cut = self._cut()
        for scorer, pile, top in (("4o", pile_4o, top_4o), ("41", pile_41, top_41)):
            v = self._scores(artifact, band, scorer)
            assert sum(1 for x in v if cut <= x < 7.5) == pile
            assert sum(1 for x in v if x >= 8.0) == top
        assert max(self._scores(artifact, band, "41")) == pytest.approx(8.011, abs=0.001)

    @pytest.mark.parametrize(
        ("band", "iqr_4o", "iqr_41"), [("L", 2.262, 1.771), ("H", 1.473, 1.146)]
    )
    def test_gpt_4_1_mini_is_narrower_not_wider(
        self, artifact: dict, band: str, iqr_4o: float, iqr_41: float
    ) -> None:
        for scorer, want in (("4o", iqr_4o), ("41", iqr_41)):
            q1, _, q3 = statistics.quantiles(
                self._scores(artifact, band, scorer), n=4, method="inclusive"
            )
            assert q3 - q1 == pytest.approx(want, abs=0.001)
        sd4o = statistics.stdev(self._scores(artifact, band, "4o"))
        sd41 = statistics.stdev(self._scores(artifact, band, "41"))
        assert sd41 <= sd4o

    @pytest.mark.parametrize("band", ["H", "L"])
    def test_the_summary_is_the_rows(self, artifact: dict, band: str) -> None:
        """Every stored field, recomputed from the rows. The field set is fixed too, so a field
        added to the script cannot go unchecked here."""
        dist = artifact["summary"]["bands"][band]["score_distribution"]
        cut = self._cut()
        assert set(dist) == {"cut", "n", "4o", "41", "between_scorer_spearman"}
        assert dist["cut"] == pytest.approx(cut, abs=1e-12)
        assert dist["n"] == SIZES[band]
        for scorer in ("4o", "41"):
            v = self._scores(artifact, band, scorer)
            q1, median, q3 = statistics.quantiles(v, n=4, method="inclusive")
            want = {
                "mean": statistics.fmean(v),
                "sd": statistics.stdev(v),
                "q1": q1,
                "median": median,
                "q3": q3,
                "iqr": q3 - q1,
                "max": v[-1],
                "below_6": sum(1 for x in v if x < 6.0),
                "from_6_to_cut": sum(1 for x in v if 6.0 <= x < cut),
                "from_cut_to_7_5": sum(1 for x in v if cut <= x < 7.5),
                "from_7_5_to_8": sum(1 for x in v if 7.5 <= x < 8.0),
                "at_or_above_8": sum(1 for x in v if x >= 8.0),
            }
            got = dist[scorer]
            assert set(got) == set(want), scorer
            for key, value in want.items():
                if isinstance(value, int):
                    assert got[key] == value, f"{scorer} {key}"
                else:
                    assert got[key] == pytest.approx(value, abs=1e-12), f"{scorer} {key}"
            assert sum(got[k] for k in want if isinstance(want[k], int)) == len(v)

    def test_the_scorers_agree_less_than_nr65_predicted(self, artifact: dict) -> None:
        """NR-65 predicted a between-scorer Spearman of 0.75 to 0.85. It landed below on both
        bands, and C-39 scores that prediction, which NR-65's own list left out."""
        from scipy.stats import spearmanr

        for band, want in (("L", 0.725), ("H", 0.745)):
            rows = _rows(artifact, band)
            rho = spearmanr([r["s4o"] for r in rows], [r["s41"] for r in rows]).statistic
            assert rho == pytest.approx(want, abs=0.001)
            assert rho < 0.75
            stored = artifact["summary"]["bands"][band]["score_distribution"]
            assert stored["between_scorer_spearman"] == pytest.approx(rho, abs=1e-9)
