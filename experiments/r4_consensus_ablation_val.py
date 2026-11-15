"""
R4: consensus, ablations and error analysis on the VALIDATION split only (binary v2 and three-class benchmarks).
Test splits are never read. Members: LR (word), RF (word), calibrated LinearSVC (char), ComplementNB (word).
Weights for the weighted schemes come from 3-fold out-of-fold predictions on the TRAINING set, never from validation.
Outputs: r4_val_results.csv (long format), r4_error_analysis.json, r4_val_errors_{task}.csv
"""
import os, json
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import ComplementNB
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEEDS = [0, 1, 2, 3, 4]
word = lambda: TfidfVectorizer(ngram_range=(1, 2), max_features=10000)
char = lambda: TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), max_features=50000, sublinear_tf=True)
MEMBERS = {  # name -> (vectoriser, classifier factory); all expose predict_proba
    "LR": (word, lambda s: LogisticRegression(max_iter=2000, random_state=s)),
    "RF": (word, lambda s: RandomForestClassifier(n_estimators=100, random_state=s, n_jobs=-1)),
    "SVM": (char, lambda s: CalibratedClassifierCV(LinearSVC(class_weight="balanced", random_state=s), cv=3, method="sigmoid")),
    "CNB": (word, lambda s: ComplementNB()),
}
rows = []
def add(task, run, seed, metric, value, note=""):
    rows.append(dict(task=task, run=run, seed=seed, metric=metric, value=float(value), note=note))

def score(task, y, p):
    labs = [0, 1] if task == "binary" else [0, 1, 2]
    f = f1_score(y, p, labels=labs, average=None, zero_division=0)
    out = {"macro_f1": f.mean(), "accuracy": accuracy_score(y, p)}
    if task == "binary":
        out["mawdu_f1"] = f[0]          # binary: label 0 = Mawdu
    else:
        out.update(f1_mawdu=f[0], f1_daif=f[1], f1_sahih=f[2])
    return out

def consensus(P, w):
    """P: list of (n, K) probability arrays; w: member weights. Returns normalised class distribution."""
    num = sum(wi * p for wi, p in zip(w, P))
    return num / num.sum(axis=1, keepdims=True)

for task, d in [("binary", "processed_v2"), ("ternary", "processed_3class")]:
    D = os.path.join(ROOT, "data", d)
    tr = pd.read_csv(os.path.join(D, "processed_train.csv")); va = pd.read_csv(os.path.join(D, "processed_val.csv"))
    for df in (tr, va):
        for c in ["normalized_text", "isnad", "matn"]:
            df[c] = df[c].fillna("").astype(str)
    ytr, yva = tr.label.values, va.label.values
    K = 2 if task == "binary" else 3
    primary = "mawdu_f1" if task == "binary" else "macro_f1"
    print(task, len(tr), len(va), flush=True)

    # out-of-fold macro-F1 on train -> accuracy-type weights without touching validation
    oof_w = {}
    for name, (vf, cf) in MEMBERS.items():
        pred = np.zeros(len(tr), int)
        for a, b in StratifiedKFold(3, shuffle=True, random_state=0).split(tr, ytr):
            v = vf(); Xa = v.fit_transform(tr.normalized_text.iloc[a]); pred[b] = cf(0).fit(Xa, ytr[a]).predict(v.transform(tr.normalized_text.iloc[b]))
        oof_w[name] = score(task, ytr, pred)
        print(" oof", name, {k: round(x, 4) for k, x in oof_w[name].items()}, flush=True)
        for k, x in oof_w[name].items():
            add(task, f"OOF-{name}", 0, k, x, "3-fold out-of-fold on train")

    # fit members on full train, predict validation probabilities for each seed
    P = {s: {} for s in SEEDS}
    for name, (vf, cf) in MEMBERS.items():
        v = vf(); Xtr = v.fit_transform(tr.normalized_text); Xva = v.transform(va.normalized_text)
        for s in SEEDS:
            P[s][name] = cf(s).fit(Xtr, ytr).predict_proba(Xva)
            for k, x in score(task, yva, P[s][name].argmax(1)).items():
                add(task, f"M-{name}", s, k, x)
    schemes = {"uniform": {n: 1.0 for n in MEMBERS},
               "acc_oof": {n: oof_w[n]["accuracy"] for n in MEMBERS},
               "f1_oof": {n: oof_w[n][primary] for n in MEMBERS}}

    def run_consensus(names, scheme, s):
        w = [schemes[scheme][n] for n in names]
        return consensus([P[s][n] for n in names], w)

    # consensus with the three weighting schemes (all four members)
    for sch in schemes:
        for s in SEEDS:
            for k, x in score(task, yva, run_consensus(list(MEMBERS), sch, s).argmax(1)).items():
                add(task, f"CONS-all-{sch}", s, k, x)
    # leave-one-out member ablation (uniform weights; same scheme for all rows)
    for drop in MEMBERS:
        names = [n for n in MEMBERS if n != drop]
        for s in SEEDS:
            for k, x in score(task, yva, run_consensus(names, "uniform", s).argmax(1)).items():
                add(task, f"CONS-minus-{drop}", s, k, x)
    # tau sweep: abstain when max class probability < tau (uniform consensus, all members)
    for s in SEEDS:
        C = run_consensus(list(MEMBERS), "uniform", s); conf = C.max(1); pred = C.argmax(1)
        for tau in [0.0, 0.5, 0.6, 0.7, 0.8, 0.9]:
            keep = conf >= tau
            add(task, f"TAU-{tau}", s, "coverage", keep.mean())
            if keep.sum() > 10:
                for k, x in score(task, yva[keep], pred[keep]).items():
                    add(task, f"TAU-{tau}", s, "covered_" + k, x)

    # isnad / matn shortcut ablation with the strongest single configuration (char SVM, balanced)
    for view in ["normalized_text", "isnad", "matn"]:
        v = char(); Xa = v.fit_transform(tr[view]); Xb = v.transform(va[view])
        pred = LinearSVC(class_weight="balanced", random_state=0).fit(Xa, ytr).predict(Xb)
        for k, x in score(task, yva, pred).items():
            add(task, f"VIEW-{view}", 0, k, x)
        print(" view", view, {k: round(x, 4) for k, x in score(task, yva, pred).items()}, flush=True)

    # near-duplicate exposure: 5-gram Jaccard between each validation item and its closest training item
    cv = CountVectorizer(ngram_range=(5, 5), binary=True, token_pattern=r"\S+", lowercase=False)
    Atr = cv.fit_transform(tr.normalized_text); Ava = cv.transform(va.normalized_text)
    inter = (Ava @ Atr.T).tocsr()
    sa = np.asarray(Ava.sum(1)).ravel(); sb = np.asarray(Atr.sum(1)).ravel()
    jmax = np.zeros(len(va))
    for i in range(len(va)):
        r = inter.getrow(i)
        if r.nnz:
            jmax[i] = (r.data / (sa[i] + sb[r.indices] - r.data)).max()
    near = jmax >= 0.8
    best = run_consensus(list(MEMBERS), "uniform", 0).argmax(1)
    add(task, "NEARDUP", 0, "fraction_val_ge_0.8", near.mean())
    for lab, mask in [("near", near), ("rest", ~near)]:
        if mask.sum() > 10:
            for k, x in score(task, yva[mask], best[mask]).items():
                add(task, f"NEARDUP-{lab}", 0, k, x)
    print(" near-dup fraction", near.mean().round(4), flush=True)

    # error analysis (seed 0, uniform consensus)
    cm = confusion_matrix(yva, best, labels=list(range(K)))
    wrong = va.assign(pred=best, true=yva, conf=run_consensus(list(MEMBERS), "uniform", 0).max(1), jmax=jmax)
    wrong = wrong[wrong.pred != wrong.true]
    wrong[["source", "true", "pred", "conf", "jmax", "raw_text"]].to_csv(os.path.join(ROOT, "experiments", f"r4_val_errors_{task}.csv"), index=False, encoding="utf-8-sig")
    json.dump({"confusion_matrix": cm.tolist(), "n_errors": int(len(wrong)),
               "mean_conf_errors": float(wrong.conf.mean()), "mean_conf_correct": float(run_consensus(list(MEMBERS), "uniform", 0).max(1)[best == yva].mean()),
               "mean_words_errors": float(wrong.raw_text.str.split().str.len().mean()), "mean_words_all": float(va.raw_text.str.split().str.len().mean())},
              open(os.path.join(ROOT, "experiments", f"r4_error_analysis_{task}.json"), "w"), indent=1)

df = pd.DataFrame(rows)
df.to_csv(os.path.join(ROOT, "experiments", "r4_val_results.csv"), index=False)
print(df.groupby(["task", "run", "metric"]).value.agg(["mean", "std"]).round(4).loc[lambda x: x.index.get_level_values(2).isin(["mawdu_f1", "macro_f1"])].to_string())
