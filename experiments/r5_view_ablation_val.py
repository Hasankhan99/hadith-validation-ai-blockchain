"""
R5: input-view ablation on the VALIDATION split, with a diagnostic of the diacritics shortcut found in R4.
R4 fed un-normalised isnad/matn columns (segmented from raw text) to the classifier; in the three-class data the Da'if texts
(LK) keep Arabic diacritics whereas MAHADDAT texts do not, so un-normalised views are separable by formatting alone.
R5 (i) quantifies diacritic prevalence per class/source, (ii) repeats the view ablation with every view normalised,
(iii) keeps the raw-text view as an explicit shortcut diagnostic. Classifier: char_wb TF-IDF + balanced LinearSVC (deterministic).
"""
import os, re, json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score, accuracy_score
import sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src_v2"))
from preprocessing import normalize_arabic

DIAC = re.compile("[ً-ْٰ]")
rows, diag = [], []
def add(task, run, metric, v): rows.append(dict(task=task, run=run, seed=0, metric=metric, value=float(v)))
def score(task, y, p):
    labs = [0, 1] if task == "binary" else [0, 1, 2]
    f = f1_score(y, p, labels=labs, average=None, zero_division=0)
    o = {"macro_f1": f.mean(), "accuracy": accuracy_score(y, p)}
    if task == "binary": o["mawdu_f1"] = f[0]
    else: o.update(f1_mawdu=f[0], f1_daif=f[1], f1_sahih=f[2])
    return o

for task, d in [("binary", "processed_v2"), ("ternary", "processed_3class")]:
    D = os.path.join(ROOT, "data", d)
    tr = pd.read_csv(os.path.join(D, "processed_train.csv")); va = pd.read_csv(os.path.join(D, "processed_val.csv"))
    for df in (tr, va):
        for c in ["raw_text", "normalized_text", "isnad", "matn"]:
            df[c] = df[c].fillna("").astype(str)
        df["isnad_n"] = df.isnad.map(normalize_arabic); df["matn_n"] = df.matn.map(normalize_arabic)
    lab = {0: "Mawdu", 1: "Daif", 2: "Sahih"} if task == "ternary" else {0: "Mawdu", 1: "Sahih"}
    g = tr.assign(has_diac=tr.raw_text.map(lambda t: bool(DIAC.search(t))), cls=tr.label.map(lab)).groupby(["cls", "source"]).has_diac.agg(["size", "mean"])
    for (c, s), r in g.iterrows():
        diag.append(dict(task=task, cls=c, source=s, n=int(r["size"]), frac_with_diacritics=float(r["mean"])))
    for view in ["raw_text", "normalized_text", "isnad_n", "matn_n"]:
        v = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), max_features=50000, sublinear_tf=True)
        Xa = v.fit_transform(tr[view]); Xb = v.transform(va[view])
        p = LinearSVC(class_weight="balanced", random_state=0).fit(Xa, tr.label.values).predict(Xb)
        s = score(task, va.label.values, p)
        for k, x in s.items(): add(task, f"VIEW5-{view}", k, x)
        print(task, view, {k: round(float(x), 4) for k, x in s.items()}, flush=True)
pd.DataFrame(rows).to_csv(os.path.join(ROOT, "experiments", "r5_val_results.csv"), index=False)
pd.DataFrame(diag).to_csv(os.path.join(ROOT, "experiments", "r5_diacritics_diag.csv"), index=False)
print(pd.DataFrame(diag).to_string())
