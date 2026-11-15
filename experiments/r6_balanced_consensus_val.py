"""
R6: class-imbalance-aware consensus on the VALIDATION split (binary v2, three-class).
Change under test: (a) members trained with class weighting, (b) consensus distribution divided by the class prior raised to a power alpha
(prior correction). alpha is chosen from {0, .25, .5, .75, 1} on 3-fold OUT-OF-FOLD predictions of the TRAINING set (primary metric there),
never on validation. Compared against the R4 uniform consensus and the best single member. Test splits are not read.
Members: LR(balanced), RF(balanced_subsample), calibrated LinearSVC(balanced, char), ComplementNB.
"""
import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import ComplementNB
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
word = lambda: TfidfVectorizer(ngram_range=(1, 2), max_features=10000)
char = lambda: TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), max_features=50000, sublinear_tf=True)
MEMBERS = {
    "LR": (word, lambda s: LogisticRegression(max_iter=2000, class_weight="balanced", random_state=s)),
    "RF": (word, lambda s: RandomForestClassifier(n_estimators=100, class_weight="balanced_subsample", random_state=s, n_jobs=-1)),
    "SVM": (char, lambda s: CalibratedClassifierCV(LinearSVC(class_weight="balanced", random_state=s), cv=3, method="sigmoid")),
    "CNB": (word, lambda s: ComplementNB()),
}
ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]
rows = []
def add(task, run, seed, metric, v, note=""): rows.append(dict(task=task, run=run, seed=seed, metric=metric, value=float(v), note=note))
def score(task, y, p):
    labs = [0, 1] if task == "binary" else [0, 1, 2]
    f = f1_score(y, p, labels=labs, average=None, zero_division=0)
    o = {"macro_f1": f.mean(), "accuracy": accuracy_score(y, p)}
    if task == "binary": o["mawdu_f1"] = f[0]
    else: o.update(f1_mawdu=f[0], f1_daif=f[1], f1_sahih=f[2])
    return o
def cons(Ps, prior, a):
    c = sum(Ps) / len(Ps)
    c = c / (prior ** a)
    return c / c.sum(1, keepdims=True)

for task, d in [("binary", "processed_v2"), ("ternary", "processed_3class")]:
    D = os.path.join(ROOT, "data", d)
    tr = pd.read_csv(os.path.join(D, "processed_train.csv")); va = pd.read_csv(os.path.join(D, "processed_val.csv"))
    for df in (tr, va): df["normalized_text"] = df.normalized_text.fillna("").astype(str)
    ytr, yva = tr.label.values, va.label.values
    K = int(ytr.max()) + 1
    prior = np.bincount(ytr, minlength=K) / len(ytr)
    primary = "mawdu_f1" if task == "binary" else "macro_f1"
    oof = {n: np.zeros((len(tr), K)) for n in MEMBERS}
    for name, (vf, cf) in MEMBERS.items():
        for a, b in StratifiedKFold(3, shuffle=True, random_state=0).split(tr, ytr):
            v = vf(); Xa = v.fit_transform(tr.normalized_text.iloc[a])
            oof[name][b] = cf(0).fit(Xa, ytr[a]).predict_proba(v.transform(tr.normalized_text.iloc[b]))
        print(task, "oof", name, flush=True)
    oof_scores = {}
    for a in ALPHAS:
        s = score(task, ytr, cons(list(oof.values()), prior, a).argmax(1))
        oof_scores[a] = s[primary]; add(task, f"OOF-alpha{a}", 0, primary, s[primary], "train OOF")
    alpha = max(oof_scores, key=oof_scores.get)
    print(task, "OOF primary by alpha", {k: round(v, 4) for k, v in oof_scores.items()}, "-> chosen", alpha, flush=True)
    add(task, "CHOSEN-alpha", 0, "alpha", alpha, "selected on train OOF")
    P = {s: {} for s in range(5)}
    for name, (vf, cf) in MEMBERS.items():
        v = vf(); Xtr = v.fit_transform(tr.normalized_text); Xva = v.transform(va.normalized_text)
        for s in range(5):
            P[s][name] = cf(s).fit(Xtr, ytr).predict_proba(Xva)
            for k, x in score(task, yva, P[s][name].argmax(1)).items(): add(task, f"M6-{name}", s, k, x)
    for s in range(5):
        for a, tag in [(0.0, "CONS6-alpha0"), (alpha, "CONS6-chosen")]:
            for k, x in score(task, yva, cons(list(P[s].values()), prior, a).argmax(1)).items(): add(task, tag, s, k, x, f"alpha={a}")
df = pd.DataFrame(rows); df.to_csv(os.path.join(ROOT, "experiments", "r6_val_results.csv"), index=False)
t = df[df.metric.isin(["mawdu_f1", "macro_f1"])].groupby(["task", "run", "metric"]).value.agg(["mean", "std"]).round(4)
print(t.to_string())
