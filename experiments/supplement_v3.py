"""Post-freeze supplement V3: optional low-weight sanad and context branches fused with the frozen consensus.

Two stages, mirroring final_test.py:
  --stage val   recompute the frozen CONS-classical validation distributions with the frozen code,
                fit the branch score models on train, select (w_sanad, w_context, tau) on validation
                only, and write SUPPLEMENT_V3_MANIFEST.json. Never reads test labels.
  --stage test  score the selected configuration on test once (supplement_v3.lock prevents a rerun),
                using the frozen test distributions saved by final_test.py.

Fusion, per item:  q = (1 - ws*gs - wc) * p_cons + ws*gs * p_sanad + wc * p_context
gs = 1 only when the item has an isnad with >= 2 linked narrators; otherwise the sanad branch is
silent and its weight returns to the frozen consensus. Weights are capped at 0.3 (low weight).
FREEZE_MANIFEST.json, final_test.lock and the frozen results are read, never written.
"""
import os, sys, json, argparse
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

EXP = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(EXP)
sys.path.insert(0, os.path.join(ROOT, "src_v3"))
sys.path.insert(0, EXP)
from common import CACHE, load_split
from sanad_branch import FEATURES, rule_score
import final_test as ft

GRID = [0.0, 0.1, 0.2, 0.3]
MAN = os.path.join(EXP, "SUPPLEMENT_V3_MANIFEST.json")
LOCK = os.path.join(EXP, "supplement_v3.lock")


def branch_inputs(task, split):
    S = pd.read_csv(os.path.join(CACHE, f"sanad_{task}_{split}.csv"))
    C = pd.read_csv(os.path.join(CACHE, f"context_{task}_{split}.csv"))
    return S, C


def fit_branches(task):
    tr = load_split(task, "train")
    S, C = branch_inputs(task, "train")
    ms, mc = S.worst_level.notna().values, C.ctx_max.notna().values
    sanad = LogisticRegression(max_iter=2000, class_weight="balanced").fit(S[FEATURES].values[ms], tr.label.values[ms])
    ctx = LogisticRegression(max_iter=2000, class_weight="balanced").fit(C.values[mc], tr.label.values[mc])
    return sanad, ctx


def branch_probs(task, split, sanad, ctx):
    S, C = branch_inputs(task, split)
    gs = S.worst_level.notna().values
    K = len(sanad.classes_)
    ps = np.full((len(S), K), 1.0 / K)
    ps[gs] = sanad.predict_proba(S[FEATURES].values[gs])
    pc = ctx.predict_proba(C.values)
    return ps, pc, gs, S


def fuse(p, ps, pc, gs, ws, wc):
    g = gs.astype(float)[:, None]
    return (1 - ws * g - wc) * p + ws * g * ps + wc * pc


def summary(task, y, D, keep=None):
    keep = np.ones(len(y), bool) if keep is None else keep
    v = [ft.metrics(task, y[keep], D[s][keep].argmax(1))["primary"] for s in ft.SEEDS]
    return float(np.mean(v)), float(np.std(v, ddof=1))  # sample std, as in FREEZE_MANIFEST


def stage_val():
    man = json.load(open(os.path.join(EXP, "FREEZE_MANIFEST.json")))
    out = {"note": "post-freeze supplement V3; selection on validation only", "tasks": {}}
    rows = []
    for task in ("binary", "ternary"):
        t = man["tasks"][task]
        tr, va = ft.load(task, "train"), ft.load(task, "val")
        cache = os.path.join(CACHE, f"val_cons_{task}.npz")
        if os.path.exists(cache):
            z = np.load(cache); P = {s: z[f"s{s}"] for s in ft.SEEDS}
        else:
            M = ft.member_probs(task, tr, va, os.path.join(ROOT, "colab", "outputs"), "val")
            P = ft.candidates(task, M, np.array(t["prior"]), t["alpha"])["CONS-classical"]
            np.savez_compressed(cache, **{f"s{s}": P[s] for s in ft.SEEDS})
        y = va.label.values
        frozen = summary(task, y, P)
        ref = t["val_primary_mean_std"]["CONS-classical"][0]
        assert abs(frozen[0] - ref) < 1e-6, f"frozen consensus not reproduced on val: {frozen[0]} vs {ref}"
        sanad, ctx = fit_branches(task)
        ps, pc, gs, S = branch_probs(task, "val", sanad, ctx)
        res = {}
        for ws in GRID:
            for wc in GRID:
                D = {s: fuse(P[s], ps, pc, gs, ws, wc) for s in ft.SEEDS}
                m, sd = summary(task, y, D)
                ms, _ = summary(task, y, D, gs)
                res[(ws, wc)] = (m, sd)
                rows.append(dict(task=task, ws=ws, wc=wc, val_primary=m, val_std=sd, val_primary_isnad_subset=ms))
        base = res[(0.0, 0.0)][0]
        best = max(res, key=lambda k: (round(res[k][0], 4), -(k[0] + k[1])))
        ws, wc = best
        D = {s: fuse(P[s], ps, pc, gs, ws, wc) for s in ft.SEEDS}
        cov = {tau: np.mean([(D[s].max(1) >= tau).mean() for s in ft.SEEDS]) for tau in ft.TAUS}
        tau = max(tau for tau in ft.TAUS if cov[tau] >= 0.90)
        # diagnostics: each branch alone and the classical weakest-link rule, on rows where it applies
        diag = {"isnad_rows": float(gs.mean()),
                "sanad_lr_alone_on_isnad_rows": float(ft.metrics(task, y[gs], ps[gs].argmax(1))["primary"]),
                "context_lr_alone": float(ft.metrics(task, y, pc.argmax(1))["primary"])}
        if task == "binary":
            rule = np.array([rule_score(r) for r in S[FEATURES].to_dict("records")])
            diag["weakest_link_rule_on_isnad_rows"] = float(ft.metrics(task, y[gs], (1 - rule[gs]).astype(int))["primary"])
        out["tasks"][task] = {"w_sanad": ws, "w_context": wc, "tau": tau, "val_primary_frozen": frozen,
                              "val_primary_selected": res[best], "improves_on_val": res[best][0] > base,
                              "diagnostics": diag}
        print(task, json.dumps(out["tasks"][task]), flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(EXP, "v3_val_grid.csv"), index=False)
    json.dump(out, open(MAN, "w"), indent=1)


def stage_test():
    if os.path.exists(LOCK):
        sys.exit("Refusing: supplement V3 has already been scored on test (supplement_v3.lock exists).")
    man = json.load(open(MAN))
    open(LOCK, "w").write("supplement V3 test scored once\n")
    rows = []
    for task in ("binary", "ternary"):
        t = man["tasks"][task]
        te = ft.load(task, "test")
        z = np.load(os.path.join(EXP, f"final_test_probs_{task}.npz"))
        y = z["y"]
        assert (y == te.label.values).all(), "frozen test probabilities are not row-aligned with the test split"
        P = {s: z[f"CONS-classical__s{s}"] for s in ft.SEEDS}
        sanad, ctx = fit_branches(task)
        ps, pc, gs, _ = branch_probs(task, "test", sanad, ctx)
        D = {s: fuse(P[s], ps, pc, gs, t["w_sanad"], t["w_context"]) for s in ft.SEEDS}
        for name, X in (("CONS-classical", P), ("V3-fused", D)):
            for s in ft.SEEDS:
                for k, v in ft.metrics(task, y, X[s].argmax(1)).items():
                    rows.append(dict(task=task, run=f"S3-{name}", seed=s, subset="all", metric=k, value=float(v)))
                for k, v in ft.metrics(task, y[gs], X[s][gs].argmax(1)).items():
                    rows.append(dict(task=task, run=f"S3-{name}", seed=s, subset="isnad_rows", metric=k, value=float(v)))
                keep = X[s].max(1) >= t["tau"]
                rows.append(dict(task=task, run=f"S3-{name}@tau{t['tau']}", seed=s, subset="all", metric="coverage", value=float(keep.mean())))
                rows.append(dict(task=task, run=f"S3-{name}@tau{t['tau']}", seed=s, subset="all", metric="covered_primary",
                                 value=float(ft.metrics(task, y[keep], X[s][keep].argmax(1))["primary"])))
    r = pd.DataFrame(rows)
    r.to_csv(os.path.join(EXP, "v3_test_results.csv"), index=False)
    print(r[r.metric.isin(["primary", "coverage", "covered_primary"])].groupby(["task", "run", "subset", "metric"]).value.agg(["mean", "std"]).round(4))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["val", "test"], required=True)
    a = ap.parse_args()
    stage_val() if a.stage == "val" else stage_test()
