"""E5 — the free-feature logistic chassis, leave-one-repo-out (LORO) everywhere.

The research pass's fifth experiment (evals/RESEARCH-score2-ranking.md §4.5): does
repo-independent metadata carry any within-band signal, and does an L2 logistic
regression over a handful of features — optionally including E1-E4's scores as columns —
give an honest pooled P(actionable)?

Features per Testbed A paper (gate-time information only):
  age_months        — from the arXiv id's YYMM prefix
  log_citations     — log1p(S2 citationCount), one keyless batch call, cached
  log_influential   — log1p(S2 influentialCitationCount)
  hyde_rank         — fractional rank in the case's HyDE top-k list (1.0 = absent)
  hop_coupling      — fwd+back citation-coupling degree from the hop pool (0 = absent)

The expected outcome is the null (features-only AUC ≈ 0.5): the practitioner-relevance
literature found citation counts uncorrelated with what engineers act on, and the
coupling/HyDE features were derived on these same labels (RESEARCH annex, pitfall 7) —
which is why every fitted number here is LORO cross-fitted (fit on 21 repos, predict the
held-out one, pool the 22 held-out predictions; regularization chosen by inner LORO on
the training repos, never by benchmark score). A confirmed null is a useful result: it
kills the feature family and E1-E4's raw scores ship un-combined.

    uv run python evals/exp_features.py            # features only
    uv run python evals/exp_features.py --combined # + E1-E4 score columns where present
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import band_testbeds as tb  # noqa: E402

from reporadar.paper_id import dedup_id  # noqa: E402

S2_CACHE = tb.EXP / "s2_meta.json"
S2_URL = (
    "https://api.semanticscholar.org/graph/v1/paper/batch"
    "?fields=citationCount,influentialCitationCount"
)
C_GRID = (0.01, 0.1, 1.0, 10.0)


def s2_metadata(ids: list[str]) -> dict[str, dict]:
    """{base_id: {citationCount, influentialCitationCount}} — batched, cached, keyless."""
    cache: dict[str, dict] = {}
    if S2_CACHE.is_file():
        cache = json.loads(S2_CACHE.read_text(encoding="utf-8"))
    todo = [i for i in ids if i not in cache]
    for start in range(0, len(todo), 400):
        chunk = todo[start : start + 400]
        body = json.dumps({"ids": [f"ARXIV:{i}" for i in chunk]}).encode()
        req = urllib.request.Request(
            S2_URL, data=body, headers={"Content-Type": "application/json"}, method="POST"
        )
        for attempt in range(5):
            try:
                with urllib.request.urlopen(req, timeout=60) as resp:
                    rows = json.loads(resp.read().decode())
                for pid, row in zip(chunk, rows, strict=True):
                    cache[pid] = {
                        "citations": (row or {}).get("citationCount"),
                        "influential": (row or {}).get("influentialCitationCount"),
                    }
                break
            except Exception as exc:  # noqa: BLE001
                print(f"  ! S2 batch failed (attempt {attempt + 1}): {exc}")
                time.sleep(10 * (attempt + 1))
        time.sleep(2)
    S2_CACHE.parent.mkdir(parents=True, exist_ok=True)
    S2_CACHE.write_text(json.dumps(cache, indent=1), encoding="utf-8")
    return cache


def age_months(base_id: str) -> float | None:
    """Months since submission, read off the id's YYMM prefix (new-style ids only)."""
    head = base_id.split(".")[0]
    if len(head) != 4 or not head.isdigit():
        return None
    year, month = 2000 + int(head[:2]), int(head[2:])
    if not 1 <= month <= 12:
        return None
    return (2026 - year) * 12 + (8 - month)


def hyde_rank(case: str) -> dict[str, float]:
    path = tb.WORK / "hyde_topk" / f"{case}.json"
    if not path.is_file():
        return {}
    ids = json.loads(path.read_text(encoding="utf-8"))
    return {pid: (i + 1) / len(ids) for i, pid in enumerate(ids)}


def hop_coupling(case: str) -> dict[str, float]:
    path = tb.WORK / "hop_pool" / f"{case}.jsonl"
    if not path.is_file():
        return {}
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            out[dedup_id(row["id"])] = float(
                (row.get("fwd_degree") or 0) + (row.get("back_degree") or 0)
            )
    return out


def method_scores() -> dict[str, dict[str, dict[str, float]]]:
    """E1-E4 per-paper scores from their result files, where already run."""
    out: dict[str, dict[str, dict[str, float]]] = {}
    e1 = tb.EXP / "select_sonnet-5_a.json"
    if e1.is_file():
        data = json.loads(e1.read_text(encoding="utf-8"))["results"]
        out["e1_share"] = {
            c: {i: v for i, v in r["shares"].items() if v is not None} for c, r in data.items()
        }
    e2 = tb.EXP / "finescale_a.json"
    if e2.is_file():
        data = json.loads(e2.read_text(encoding="utf-8"))["scored"]
        out["e2_exp09"] = {
            c: {i: r["exp09"] for i, r in per.items() if "exp09" in r} for c, per in data.items()
        }
        out["e2_p_true"] = {
            c: {i: r["p_true"] for i, r in per.items() if r.get("p_true") is not None}
            for c, per in data.items()
        }
    e3 = tb.EXP / "ensemble_a.json"
    if e3.is_file():
        data = json.loads(e3.read_text(encoding="utf-8"))["scored"]
        out["e3_p_hat"] = {
            c: {i: r["p_hat"] for i, r in per.items() if r} for c, per in data.items()
        }
    e4 = tb.EXP / "pairwise_a.json"
    if e4.is_file():
        data = json.loads(e4.read_text(encoding="utf-8"))["results"]
        out["e4_p_beat"] = {c: r.get("p_beats_borderline", {}) for c, r in data.items()}
    return out


def build_matrix(combined: bool) -> tuple[list[dict], list[str]]:
    """Rows for every Testbed A shown paper; feature columns + case/id/labels."""
    import math

    bands = tb.load_testbed_a()
    all_ids = sorted({p.id for b in bands.values() for p in b.papers})
    s2 = s2_metadata(all_ids)
    methods = method_scores() if combined else {}
    rows = []
    for case, band in bands.items():
        hyde = hyde_rank(case)
        hop = hop_coupling(case)
        for p in band.papers:
            meta = s2.get(p.id) or {}
            row: dict = {
                "case": case,
                "id": p.id,
                "judge": p.judge,
                "gate": p.gate,
                "age_months": age_months(p.id),
                "log_citations": math.log1p(meta["citations"])
                if meta.get("citations") is not None
                else None,
                "log_influential": math.log1p(meta["influential"])
                if meta.get("influential") is not None
                else None,
                "hyde_rank": hyde.get(p.id, 1.0),
                "hop_coupling": hop.get(p.id, 0.0),
            }
            for name, per in methods.items():
                row[name] = per.get(case, {}).get(p.id)
            rows.append(row)
    features = ["age_months", "log_citations", "log_influential", "hyde_rank", "hop_coupling"]
    features += list(methods.keys())
    return rows, features


def loro_fit(rows: list[dict], features: list[str]) -> dict:
    """LORO logistic: pooled held-out probabilities, AUC and Brier, chosen C per fold."""
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    usable = [r for r in rows if all(r.get(f) is not None for f in features)]
    dropped = len(rows) - len(usable)
    cases = sorted({r["case"] for r in usable})
    x = np.array([[float(r[f]) for f in features] for r in usable])
    y = np.array([1 if r["judge"] >= tb.ACTIONABLE else 0 for r in usable])
    case_of = np.array([r["case"] for r in usable])

    def fit_predict(train_mask, test_mask, c):  # type: ignore[no-untyped-def]
        model = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=1000))
        model.fit(x[train_mask], y[train_mask])
        return model.predict_proba(x[test_mask])[:, 1]

    pooled_p = np.zeros(len(usable))
    chosen: dict[str, float] = {}
    for held in cases:
        test_mask = case_of == held
        train_mask = ~test_mask
        if len(set(y[train_mask])) < 2:
            pooled_p[test_mask] = float(y[train_mask].mean())
            continue
        # Inner LORO over the training repos picks C — never the benchmark score.
        best_c, best_auc = C_GRID[0], -1.0
        for c in C_GRID:
            inner_scores, inner_labels = [], []
            for inner_held in cases:
                if inner_held == held:
                    continue
                inner_test = case_of == inner_held
                inner_train = train_mask & ~inner_test
                if len(set(y[inner_train])) < 2 or not inner_test.any():
                    continue
                inner_scores.extend(fit_predict(inner_train, inner_test, c))
                inner_labels.extend(bool(v) for v in y[inner_test])
            a = tb.auc([float(s) for s in inner_scores], inner_labels)
            if a == a and a > best_auc:
                best_auc, best_c = a, c
        chosen[held] = best_c
        pooled_p[test_mask] = fit_predict(train_mask, test_mask, best_c)

    probs = [float(v) for v in pooled_p]
    labels = [bool(v) for v in y]
    band_mask = np.array([r["gate"] == 2 for r in usable])
    result = {
        "n_rows": len(usable),
        "n_dropped_missing_feature": dropped,
        "features": features,
        "auc_all": tb.auc(probs, labels),
        "brier_all": tb.brier(probs, labels),
        "auc_band_judge2": tb.auc(
            [probs[i] for i in range(len(usable)) if band_mask[i]],
            [labels[i] for i in range(len(usable)) if band_mask[i]],
        ),
        "chosen_C": chosen,
    }
    # Held-out probabilities keyed for the policy eval and for later joins.
    result["probs"] = {}
    for i, r in enumerate(usable):
        result["probs"].setdefault(r["case"], {})[r["id"]] = probs[i]
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--combined", action="store_true", help="add E1-E4 score columns")
    args = ap.parse_args()

    tb.load_env()
    rows, features = build_matrix(args.combined)
    print(f"matrix: {len(rows)} rows, features: {features}")

    fit = loro_fit(rows, features)
    bands = tb.load_testbed_a()
    baselines = tb.baseline_nets(bands)
    per_case = {}
    for case, band in bands.items():
        policy = tb.policy_net(band, fit["probs"].get(case, {}))
        per_case[case] = {
            "policy": policy,
            "show_all": baselines[case]["show_all"],
            "delta": policy - baselines[case]["show_all"],
        }
    summary = {
        "combined": args.combined,
        **{k: v for k, v in fit.items() if k != "probs"},
        "policy_mean": round(sum(v["policy"] for v in per_case.values()) / len(per_case), 3),
        "show_all_mean": round(sum(v["show_all"] for v in per_case.values()) / len(per_case), 3),
        "sign_test": tb.sign_test([v["delta"] for v in per_case.values()]),
        "per_case": per_case,
    }
    out = tb.EXP / ("features_combined.json" if args.combined else "features_a.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({"summary": summary, "probs": fit["probs"]}, indent=1), encoding="utf-8"
    )
    print("\n=== E5 ===")
    for k, v in summary.items():
        if k not in ("per_case", "probs", "chosen_C"):
            print(f"  {k}: {v}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
