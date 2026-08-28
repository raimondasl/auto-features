"""Freeze P24: the matched arxiv vs arxiv+europepmc arms. Run files are gitignored."""

import json
import pathlib
import statistics as st
import sys

sys.path.insert(0, "evals")
from reporadar.paper_id import is_arxiv_id

RES = pathlib.Path("evals/results")
picks_by_arm: dict[str, dict[str, list[tuple[str, int]]]] = {}
ARMS = {
    "control_arxiv": "judge-gpt-5.5-frozenpool-bigrams_verified-wemb1.5-20260827T213701Z.json",
    "treat_arxiv_epmc": "judge-gpt-5.5-frozenpool-bigrams_verified-wemb1.5-20260827T234024Z.json",
    # P25. Same control, same flags, only `sources` differs -- so it drops into the same
    # comparison without a second control being collected.
    "treat_arxiv_openalex": (
        "judge-gpt-5.5-frozenpool-bigrams_verified-wemb1.5-20260828T052915Z.json"
    ),
}
HEADLINE = "judge-gpt-5.5-frozenpool-bigrams_verified-wemb1.5-20260815T225831Z.json"


def net(p):
    return sum(1 if int(x["judge_score"]) >= 2 else -2 for x in p)


def pt(score):
    return 1 if int(score) >= 2 else -2


def decompose(label, cases):
    """Split a treatment's delta into what the SOURCE contributed and what it DISPLACED.

    The control is arXiv-only, so every non-arXiv paper in the treatment is a paper the
    source supplied and the first term is exactly their net@2. Everything else is arXiv
    churn: papers the treatment shows that the control did not, minus papers the control
    showed that the treatment dropped. The two terms sum to the delta by construction.

    This exists because P25 read OpenAlex's -0.76 off its 17 non-actionable papers (17 x -2
    over 37 cases is -0.92, which looked close enough to be an explanation). It is not one.
    Those 17 sit alongside 51 actionable ones, so OpenAlex's own papers are net POSITIVE;
    the loss is 142 arXiv papers leaving the digest and 100 different ones arriving.
    """
    src = swap = ax_in = ax_out = 0
    for case in cases:
        base = picks_by_arm["control_arxiv"][case]
        treat = picks_by_arm[label][case]
        base_ids = {pid for pid, _ in base}
        treat_ids = {pid for pid, _ in treat}
        for pid, score in treat:
            if not is_arxiv_id(pid):
                src += pt(score)
            elif pid not in base_ids:
                ax_in += 1
                swap += pt(score)
        for pid, score in base:
            if pid not in treat_ids:
                ax_out += 1
                swap -= pt(score)
    return src, swap, ax_in, ax_out


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
    picks_by_arm[label] = {
        e["case"]: [
            (str(p["arxiv_id"]), int(p["judge_score"])) for p in e["returned"]["reporadar_toppicks"]
        ]
        for e in run
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
    # Split out for C-34: it is the only cohort where a source contributes NO papers, so the
    # displacement term stands alone with nothing to confound it.
    "matsci6": lambda c: c.startswith("mat-"),
    "all37": lambda _c: True,
}
out["cohorts"] = {}
for label in ("treat_arxiv_epmc", "treat_arxiv_openalex"):
    treat = out["arms"][label]["per_case"]
    out["cohorts"][label] = {}
    for name, pred in COHORTS.items():
        cases = sorted(c for c in ctrl if pred(c))
        d = [treat[c]["net2"] - ctrl[c]["net2"] for c in cases]
        shown = sum(treat[c]["n"] for c in cases)
        na = sum(treat[c]["n_non_arxiv"] for c in cases)
        good = sum(treat[c]["n_non_arxiv_actionable"] for c in cases)
        src, swap, ax_in, ax_out = decompose(label, cases)
        out["cohorts"][label][name] = {
            "n_cases": len(cases),
            "control_mean": round(st.mean(ctrl[c]["net2"] for c in cases), 2),
            "treatment_mean": round(st.mean(treat[c]["net2"] for c in cases), 2),
            "paired_delta": round(st.mean(d), 2),
            "wins": sum(1 for x in d if x > 0),
            "losses": sum(1 for x in d if x < 0),
            "non_arxiv_shown": na,
            "non_arxiv_share": round(na / shown, 3),
            "non_arxiv_actionable": good,
            "non_arxiv_precision": round(good / na, 3) if na else None,
            "non_arxiv_missed": na - good,
            # C-34. The two terms the delta is actually made of. Adding a source does two
            # separate things: it contributes its own papers, and it perturbs the pool so a
            # different set of ARXIV papers surfaces. Only the first was being measured, and
            # for OpenAlex the two have OPPOSITE signs -- its own papers are worth +0.46/case
            # and the displacement costs -1.22, so reading the loss off the 17 misses gets
            # the mechanism backwards while landing near the right number by coincidence.
            "delta_from_source_papers": round(src / len(cases), 2),
            "delta_from_arxiv_swap": round(swap / len(cases), 2),
            "arxiv_added": ax_in,
            "arxiv_dropped": ax_out,
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
