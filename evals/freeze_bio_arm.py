"""Freeze P21's per-case figures: the run files they derive from are gitignored.

`evals/results/` is not in version control, so a document quoting these numbers would cite
something nothing in the repository can reproduce -- the failure `gold_targets.json` and
`restated_runs.json` were built against. Same remedy: derive once, commit the derivation,
pin it with a test.
"""

import json
import pathlib
import statistics as st

RES = pathlib.Path("evals/results")
ARMS = {
    "w15_epmc_wemb15": "judge-gpt-5.5-frozenpool-bigrams_verified-wemb1.5-20260827T163811Z.json",
    "w30_epmc_wemb0": "judge-gpt-5.5-frozenpool-bigrams_verified-20260821T152858Z.json",
    "w15_arxiv_wemb0": "judge-gpt-5.5-frozenpool-bigrams_verified-20260820T060917Z.json",
}


def net(picks):
    return sum(1 if int(p["judge_score"]) >= 2 else -2 for p in picks)


out = {
    "_comment": (
        "P21: RepoRadar on the six bio cases across three configurations, plus the Opus 5 "
        "draw for the same cases. Derived by evals/freeze_bio_arm.py from run files under "
        "evals/results/, which are gitignored; pinned by tests/test_bio_matched_arm.py. "
        "net@2 = #actionable - 2 x #non-actionable over what each system RETURNED."
    ),
    "arms": {},
}
for label, fname in ARMS.items():
    run = json.loads((RES / fname).read_text(encoding="utf-8"))
    e0 = next(e for e in run if e["case"].startswith("bio-"))
    arm = {
        "run_file": fname,
        "digest_window": e0["digest_window"],
        "sources": e0["sources"],
        "w_embedding": e0["w_embedding"],
        "per_case": {},
    }
    for e in run:
        if not e["case"].startswith("bio-"):
            continue
        picks = e["returned"]["reporadar_toppicks"]
        arm["per_case"][e["case"]] = {"n": len(picks), "net2": net(picks)}
        if arm["digest_window"] == 30:
            arm["per_case"][e["case"]]["net2_truncated_15"] = net(picks[:15])
    arm["mean_net2"] = round(st.mean(v["net2"] for v in arm["per_case"].values()), 2)
    if arm["digest_window"] == 30:
        arm["mean_net2_truncated_15"] = round(
            st.mean(v["net2_truncated_15"] for v in arm["per_case"].values()), 2
        )
    out["arms"][label] = arm

rows = json.loads(pathlib.Path("evals/gold_spread_v2_opus5.json").read_text(encoding="utf-8"))[
    "results"
]
o5 = {}
for k, r in rows.items():
    c = k.split("/", 1)[1]
    if c.startswith("bio-") and r.get("status") == "ok":
        sc = r.get("scores") or {}
        o5[c] = {"n": len(sc), "net2": sum(1 if s >= 2 else -2 for s in sc.values())}
out["opus5"] = {
    "source": "gold_spread_v2_opus5.json draw 1",
    "prompt_version": "v2",
    "max_turns": 30,
    "per_case": o5,
    "mean_net2": round(st.mean(v["net2"] for v in o5.values()), 2),
}
shared = sorted(set(o5) & set(out["arms"]["w15_epmc_wemb15"]["per_case"]))
d = [out["arms"]["w15_epmc_wemb15"]["per_case"][c]["net2"] - o5[c]["net2"] for c in shared]
out["matched_comparison"] = {
    "reporadar_arm": "w15_epmc_wemb15",
    "paired_delta": round(st.mean(d), 2),
    "wins": sum(1 for x in d if x > 0),
    "losses": sum(1 for x in d if x < 0),
    "n_cases": len(shared),
}
a = out["arms"]
out["decomposition"] = {
    "window_30_to_15": round(
        a["w30_epmc_wemb0"]["mean_net2_truncated_15"] - a["w30_epmc_wemb0"]["mean_net2"], 2
    ),
    "add_europepmc_at_wemb0": round(
        a["w30_epmc_wemb0"]["mean_net2_truncated_15"] - a["w15_arxiv_wemb0"]["mean_net2"], 2
    ),
    "wemb_0_to_1p5_at_w15": round(
        a["w15_epmc_wemb15"]["mean_net2"] - a["w30_epmc_wemb0"]["mean_net2_truncated_15"], 2
    ),
}
pathlib.Path("evals/bio_matched_arm.json").write_text(
    json.dumps(out, indent=1) + "\n", encoding="utf-8"
)
print(json.dumps({k: out[k] for k in ("matched_comparison", "decomposition")}, indent=1))
print(
    "means:",
    {k: v["mean_net2"] for k, v in out["arms"].items()},
    "opus5:",
    out["opus5"]["mean_net2"],
)
