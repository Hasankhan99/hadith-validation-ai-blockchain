"""
R3: Baselines on the three-class benchmark (0 = Mawdu, 1 = Da'if, 2 = Sahih). VALIDATION ONLY.
Primary metric: macro-F1 over the three classes; per-class F1 reported. Seeds 0-4.
Also reports Da'if recall per source (LK vs NAH) as a source-confound check.
"""
import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.dummy import DummyClassifier
from sklearn.metrics import f1_score, accuracy_score, recall_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, "data", "processed_3class")
tr = pd.read_csv(os.path.join(D, "processed_train.csv")); va = pd.read_csv(os.path.join(D, "processed_val.csv"))
Xtr, ytr = tr.normalized_text.fillna("").astype(str), tr.label.values
Xva, yva = va.normalized_text.fillna("").astype(str), va.label.values

def m(y, p):
    f = f1_score(y, p, labels=[0, 1, 2], average=None, zero_division=0)
    r = {"macro_f1": f.mean(), "f1_mawdu": f[0], "f1_daif": f[1], "f1_sahih": f[2], "accuracy": accuracy_score(y, p)}
    for src in ["LK-Daif", "NAH-Daif"]:
        mask = (va.source == src).values
        r[f"daif_recall_{src.split('-')[0]}"] = recall_score(y[mask], p[mask], labels=[1], average="micro", zero_division=0)
    return r

word = lambda: TfidfVectorizer(ngram_range=(1, 2), max_features=10000)
char = lambda: TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), max_features=50000, sublinear_tf=True)
configs = {
    "T0_majority": (word, lambda s: DummyClassifier(strategy="most_frequent")),
    "T1_lr_word": (word, lambda s: LogisticRegression(max_iter=2000, random_state=s)),
    "T2_rf_word": (word, lambda s: RandomForestClassifier(n_estimators=100, random_state=s, n_jobs=-1)),
    "T3_svm_word": (word, lambda s: LinearSVC(random_state=s)),
    "T4_lr_word_bal": (word, lambda s: LogisticRegression(max_iter=2000, class_weight="balanced", random_state=s)),
    "T5_svm_char_bal": (char, lambda s: LinearSVC(class_weight="balanced", random_state=s)),
}
rows = []
for name, (vf, cf) in configs.items():
    vec = vf(); Vtr = vec.fit_transform(Xtr); Vva = vec.transform(Xva)
    for s in range(5):
        rows.append({"run": name, "seed": s, **m(yva, cf(s).fit(Vtr, ytr).predict(Vva))})
    print(name, "done", flush=True)
df = pd.DataFrame(rows)
df.to_csv(os.path.join(ROOT, "experiments", "r3_val_per_seed.csv"), index=False)
a = df.groupby("run", sort=False).agg(["mean", "std"])
for r in a.index:
    print(f"{r:18s}", "  ".join(f"{c}={a.loc[r,(c,'mean')]:.4f}±{a.loc[r,(c,'std')]:.4f}" for c in
                                ["macro_f1", "f1_mawdu", "f1_daif", "f1_sahih", "accuracy", "daif_recall_LK", "daif_recall_NAH"]))
