"""
R1: Baselines on the corrected v2 benchmark (official MAHADDAT split, fixed labels,
leakage removed). VALIDATION SET ONLY - the test set is locked until the Day-3 freeze.
Primary metric: F1 of the Mawdu (fabricated, label 0) class. Seeds 0-4; mean +/- std.
"""
import os, sys, json, time
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import ComplementNB
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, "data", "processed_v2")
tr = pd.read_csv(os.path.join(D, "processed_train.csv"))
va = pd.read_csv(os.path.join(D, "processed_val.csv"))
Xtr, ytr = tr.normalized_text.fillna("").astype(str), tr.label.values
Xva, yva = va.normalized_text.fillna("").astype(str), va.label.values

def m(y, p):
    return {"mawdu_f1": f1_score(y, p, pos_label=0), "mawdu_p": precision_score(y, p, pos_label=0, zero_division=0),
            "mawdu_r": recall_score(y, p, pos_label=0), "macro_f1": f1_score(y, p, average="macro"),
            "weighted_f1": f1_score(y, p, average="weighted"), "accuracy": accuracy_score(y, p)}

word = lambda: TfidfVectorizer(ngram_range=(1, 2), max_features=10000)
char = lambda: TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), max_features=50000, sublinear_tf=True)
configs = {
    "B0_majority":       (word, lambda s: DummyClassifier(strategy="most_frequent")),
    "B1_lr_word_orig":   (word, lambda s: LogisticRegression(max_iter=1000, random_state=s)),
    "B2_rf_word_orig":   (word, lambda s: RandomForestClassifier(n_estimators=100, random_state=s, n_jobs=-1)),
    "B3_cnb_word":       (word, lambda s: ComplementNB()),
    "B4_svm_word":       (word, lambda s: LinearSVC(random_state=s)),
    "B5_lr_word_bal":    (word, lambda s: LogisticRegression(max_iter=1000, class_weight="balanced", random_state=s)),
    "B6_svm_char_bal":   (char, lambda s: LinearSVC(class_weight="balanced", random_state=s)),
}
SEEDS = [0, 1, 2, 3, 4]
rows, probs = [], {}
for name, (vec_f, clf_f) in configs.items():
    vec = vec_f(); Vtr = vec.fit_transform(Xtr); Vva = vec.transform(Xva)
    for s in SEEDS:
        t = time.time(); clf = clf_f(s).fit(Vtr, ytr); p = clf.predict(Vva)
        r = {"run": name, "seed": s, **m(yva, p), "train_sec": round(time.time() - t, 1)}
        rows.append(r)
        if hasattr(clf, "predict_proba") and name in ("B1_lr_word_orig", "B2_rf_word_orig"):
            probs.setdefault(name, []).append(clf.predict_proba(Vva))
    print(name, "done", flush=True)

# E1: LR+RF confidence-weighted consensus (src_v2/ensemble_consensus formula), weights = mean val macro-F1
df = pd.DataFrame(rows)
w = df.groupby("run").macro_f1.mean()
for i, s in enumerate(SEEDS):
    pl, pr = probs["B1_lr_word_orig"][i], probs["B2_rf_word_orig"][i]
    s1 = w["B1_lr_word_orig"] * pl[:, 1] * (pl[:, 1] >= .5) + w["B2_rf_word_orig"] * pr[:, 1] * (pr[:, 1] >= .5)
    s0 = w["B1_lr_word_orig"] * pl[:, 0] * (pl[:, 0] > .5) + w["B2_rf_word_orig"] * pr[:, 0] * (pr[:, 0] > .5)
    p = (s1 >= s0).astype(int)
    rows.append({"run": "E1_ens_lr_rf_weighted", "seed": s, **m(yva, p), "train_sec": 0})

df = pd.DataFrame(rows)
df.to_csv(os.path.join(ROOT, "experiments", "r1_val_per_seed.csv"), index=False)
agg = df.groupby("run").agg(["mean", "std"])[["mawdu_f1", "mawdu_p", "mawdu_r", "macro_f1", "weighted_f1", "accuracy"]]
agg.to_csv(os.path.join(ROOT, "experiments", "r1_val_summary.csv"))
pd.set_option("display.width", 250)
print(agg.round(4))
