"""
R7: selection of the abstention threshold tau for the frozen consensus design (VALIDATION split only; test never read).
Rule (fixed before running): among tau in {0.35, 0.40, ..., 0.90} (grid was {0.50..0.90} in the first run; extended because the three-class task had no admissible tau), choose the largest tau whose mean validation coverage over
seeds 0-4 is >= 0.90 (at most 10% of narrations referred to scholars). The primary metric on the covered subset is reported for every tau.
Consensus: uniform weights, prior-correction exponent alpha from R6 (selected on training out-of-fold predictions).
Also checks that src_v2/consensus_soft.py reproduces the R6 consensus exactly.
"""
import os, sys
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import ComplementNB
from sklearn.metrics import f1_score, accuracy_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src_v2"))
from consensus_soft import consensus_distribution

word = lambda: TfidfVectorizer(ngram_range=(1, 2), max_features=10000)
char = lambda: TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), max_features=50000, sublinear_tf=True)
MEMBERS = {  # identical to R6
    "LR": (word, lambda s: LogisticRegression(max_iter=2000, class_weight="balanced", random_state=s)),
    "RF": (word, lambda s: RandomForestClassifier(n_estimators=100, class_weight="balanced_subsample", random_state=s, n_jobs=-1)),
    "SVM": (char, lambda s: CalibratedClassifierCV(LinearSVC(class_weight="balanced", random_state=s), cv=3, method="sigmoid")),
    "CNB": (word, lambda s: ComplementNB()),
}
TAUS = [round(0.35 + 0.05 * i, 2) for i in range(12)]  # grid extended from 0.50 to 0.35 after the first run found no tau with coverage >= 0.90 in the three-class task (deviation logged)
r6 = pd.read_csv(os.path.join(ROOT, "experiments", "r6_val_results.csv"))
rows = []
def add(task, run, seed, metric, v): rows.append(dict(task=task, run=run, seed=seed, metric=metric, value=float(v), note=""))

for task, d in [("binary", "processed_v2"), ("ternary", "processed_3class")]:
    D = os.path.join(ROOT, "data", d)
    tr = pd.read_csv(os.path.join(D, "processed_train.csv")); va = pd.read_csv(os.path.join(D, "processed_val.csv"))
    for df in (tr, va): df["normalized_text"] = df.normalized_text.fillna("").astype(str)
    ytr, yva = tr.label.values, va.label.values
    K = int(ytr.max()) + 1
    prior = np.bincount(ytr, minlength=K) / len(ytr)
    alpha = float(r6[(r6.task == task) & (r6.run == "CHOSEN-alpha")].value.iloc[0])
    labs = list(range(K))
    def prim(y, p):
        f = f1_score(y, p, labels=labs, average=None, zero_division=0)
        return f[0] if task == "binary" else f.mean()
    P = {s: [] for s in range(5)}
    for name, (vf, cf) in MEMBERS.items():
        v = vf(); Xtr = v.fit_transform(tr.normalized_text); Xva = v.transform(va.normalized_text)
        for s in range(5): P[s].append(cf(s).fit(Xtr, ytr).predict_proba(Xva))
        print(task, "fitted", name, flush=True)
    cov = {t: [] for t in TAUS}; met = {t: [] for t in TAUS}
    for s in range(5):
        q = consensus_distribution(P[s], prior, alpha)
        c = sum(P[s]) / 4; c = c / (prior ** alpha); c = c / c.sum(1, keepdims=True)   # R6 formula, re-derived
        assert np.allclose(q, c), "consensus_soft deviates from the R6 formula"
        conf, pred = q.max(1), q.argmax(1)
        add(task, "R7-nocut", s, "primary", prim(yva, pred))
        for t in TAUS:
            keep = conf >= t
            cov[t].append(keep.mean()); add(task, f"R7-tau{t}", s, "coverage", keep.mean())
            m = prim(yva[keep], pred[keep]) if keep.sum() > 10 else np.nan
            met[t].append(m); add(task, f"R7-tau{t}", s, "covered_primary", m)
    ok = [t for t in TAUS if np.mean(cov[t]) >= 0.90]
    tau = max(ok)
    add(task, "CHOSEN-tau", 0, "tau", tau)
    print(f"{task}: alpha={alpha}  chosen tau={tau}")
    for t in TAUS: print(f"  tau={t:.2f} coverage={np.mean(cov[t]):.4f} covered_primary={np.nanmean(met[t]):.4f}")
pd.DataFrame(rows).to_csv(os.path.join(ROOT, "experiments", "r7_val_results.csv"), index=False)
