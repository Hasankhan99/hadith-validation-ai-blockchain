"""
R0: Reproduce the numbers reported in the defense paper from the saved model artifacts
(models/*.joblib) on the original processed test split (data/processed/processed_test.csv).
No retraining. Reports weighted (as in the paper) and macro / per-class metrics.
"""
import os, sys, json, hashlib
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from ensemble_consensus import EnsembleConsensusEngine

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def metrics(y, p):
    acc = accuracy_score(y, p)
    wp, wr, wf, _ = precision_recall_fscore_support(y, p, average="weighted", zero_division=0)
    mp, mr, mf, _ = precision_recall_fscore_support(y, p, average="macro", zero_division=0)
    cp, cr, cf, cs = precision_recall_fscore_support(y, p, labels=[0, 1], zero_division=0)
    return {
        "accuracy": acc, "w_precision": wp, "w_recall": wr, "w_f1": wf,
        "macro_f1": mf, "f1_label0": cf[0], "f1_label1": cf[1],
        "p_label1": cp[1], "r_label1": cr[1], "support": cs.tolist(),
        "cm": confusion_matrix(y, p, labels=[0, 1]).tolist(),
    }

test = pd.read_csv(os.path.join(ROOT, "data/processed/processed_test.csv"))
X = test["normalized_text"].fillna("").astype(str)
y = test["label"].values

out = {}
preds, confs = {}, {}
for mid, fname in [("model_lr", "model_logistic_regression.joblib"), ("model_rf", "model_random_forest.joblib")]:
    path = os.path.join(ROOT, "models", fname)
    d = joblib.load(path)
    Xv = d["vectorizer"].transform(X)
    p = d["model"].predict(Xv)
    pr = d["model"].predict_proba(Xv)
    preds[mid] = p
    confs[mid] = pr[np.arange(len(p)), p]
    out[mid] = metrics(y, p)
    out[mid]["sha256"] = sha256(path)

# Ensemble exactly as main_pipeline.py: weights = reported test accuracy, conf = predicted-class prob
eng = EnsembleConsensusEngine({"model_lr": 0.9719, "model_rf": 0.9780})
ens = []
for i in range(len(y)):
    v = {m: int(preds[m][i]) for m in preds}
    c = {m: float(confs[m][i]) for m in preds}
    ens.append(eng.weighted_majority_vote(v, c)[0])
out["ensemble_weighted"] = metrics(y, np.array(ens))
out["majority_class"] = metrics(y, np.zeros_like(y))

json.dump(out, open(os.path.join(ROOT, "experiments/r0_results.json"), "w"), indent=2, default=float)
for k, v in out.items():
    print(f"{k:18s} acc={v['accuracy']:.4f} wP={v['w_precision']:.4f} wR={v['w_recall']:.4f} wF1={v['w_f1']:.4f} "
          f"macroF1={v['macro_f1']:.4f} F1(lbl1)={v['f1_label1']:.4f} cm={v['cm']}")
