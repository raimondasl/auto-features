"""Freeze P24: the matched arxiv vs arxiv+europepmc arms. Run files are gitignored."""

import json
import pathlib
import statistics as st
import sys

sys.path.insert(0, "evals")
from reporadar.paper_id import is_arxiv_id

RES = pathlib.Path("evals/results")
ARMS = {
    "control_arxiv": "judge-gpt-5.5-frozenpool-bigrams_verified-wemb1.5-20260827T213701Z.json",
    "treat_arxiv_epmc": "judge-gpt-5.5-frozenpool-bigrams_verified-wemb1.5-20260827T234024Z.json",
}
HEADLINE = "judge-gpt-5.5-frozenpool-bigrams_verified-wemb1.5-20260815T225831Z.json"


def net(p):
    return sum(1 if int(x["judge_score"]) >= 2 else -2 for x in p)


out = {
    "_comment": (
        "P24: arxiv vs arxiv+europepmc, matched same-day frozen pools, 37 cases, everything "
        "but `sources` held fixed. Derived by evals/freeze_multisource_arm.py from run files "
        "under evals/results/ (gitignored); pinned by tests/test_multisource_arm.py."
    ),
    "arms": {},
}
for label, f in ARMS.items():
    run = json.loads((RES / f).read_text(encoding="utf-8"))
    e0 = run[0]
    arm = {
        "run_file": f,
        "sources": e0["sources"],
        "digest_window": e0["digest_window"],
        "w_embedding": e0["w_embedding"],
        "pool_config": e0.get("pool_config", {}),
        "per_case": {},
    }
    for e in run:
        picks = e["returned"]["reporadar_toppicks"]
        na = [p for p in picks if not is_arxiv_id(str(p["arxiv_id"]))]
        arm["per_case"][e["case"]] = {
            "n": len(picks),
            "net2": net(picks),
            "n_non_arxiv": len(na),
            "n_non_arxiv_actionable": sum(1 for p in na if int(p["judge_score"]) >= 2),
        }
    out["arms"][label] = arm

head = json.loads((RES / HEADLINE).read_text(encoding="utf-8"))
h = {e["case"]: net(e["returned"]["reporadar_toppicks"]) for e in head}
ctrl = out["arms"]["control_arxiv"]["per_case"]
shared = sorted(set(h) & set(ctrl))
out["control_reproduces_headline"] = {
    "headline_run": HEADLINE,
    "n_cases": len(shared),
    "headline_mean": round(st.mean(h[c] for c in shared), 2),
    "control_mean": round(st.mean(ctrl[c]["net2"] for c in shared), 2),
    "delta": round(st.mean(ctrl[c]["net2"] - h[c] for c in shared), 2),
}

COHORTS = {
    "core25": lambda c: not c.startswith(("bio-", "mat-")),
    "scientific12": lambda c: c.startswith(("bio-", "mat-")),
    "all37": lambda _c: True,
}
treat = out["arms"]["treat_arxiv_epmc"]["per_case"]
out["cohorts"] = {}
for name, pred in COHORTS.items():
    cases = sorted(c for c in ctrl if pred(c))
    d = [treat[c]["net2"] - ctrl[c]["net2"] for c in cases]
    shown = sum(treat[c]["n"] for c in cases)
    na = sum(treat[c]["n_non_arxiv"] for c in cases)
    good = sum(treat[c]["n_non_arxiv_actionable"] for c in cases)
    out["cohorts"][name] = {
        "n_cases": len(cases),
        "control_mean": round(st.mean(ctrl[c]["net2"] for c in cases), 2),
        "treatment_mean": round(st.mean(treat[c]["net2"] for c in cases), 2),
        "paired_delta": round(st.mean(d), 2),
        "wins": sum(1 for x in d if x > 0),
        "losses": sum(1 for x in d if x < 0),
        "non_arxiv_shown": na,
        "non_arxiv_share": round(na / shown, 3),
        "non_arxiv_actionable": good,
    }
pathlib.Path("evals/multisource_arm.json").write_text(
    json.dumps(out, indent=1) + "\n", encoding="utf-8"
)
print(
    json.dumps(
        {
            "control_reproduces_headline": out["control_reproduces_headline"],
            "cohorts": out["cohorts"],
        },
        indent=1,
    )
)
