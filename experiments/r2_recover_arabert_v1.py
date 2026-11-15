"""
R2: Historical AraBERT run (user's Colab log, 2026-09-24, v1 data). The log reports only
support-weighted metrics. With the v1 validation composition known (2,385 vs 247), the
confusion matrix is recovered exactly by solving for (FN, FP) of the minority class such that
weighted precision and weighted F1 match the log (zero residual = unique, exact solution).
Under the v1 label inversion the minority class is the fabricated (Mawdu) class.
"""
import os, re, ast, json
import pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
v = pd.read_csv(os.path.join(ROOT, "data/processed/processed_val.csv"))
n0, n1 = int((v.label == 0).sum()), int((v.label == 1).sum()); N = n0 + n1
evals = [ast.literal_eval(l) for l in open(os.path.join(ROOT, "experiments/logs/colab_arabert_v1_2026-09-24.txt"), encoding="utf-8")
         if l.startswith("{'eval_loss'")]
out = []
for e in evals:
    err = round(N * (1 - e["eval_accuracy"])); sols = []
    for a in range(err + 1):
        b = err - a; tp1, tp0 = n1 - a, n0 - b
        p1, r1 = tp1 / (tp1 + b), tp1 / n1; f1 = 2 * p1 * r1 / (p1 + r1)
        p0, r0 = tp0 / (tp0 + a), tp0 / n0; f0 = 2 * p0 * r0 / (p0 + r0)
        res = abs((n0 * p0 + n1 * p1) / N - e["eval_precision"]) + abs((n0 * f0 + n1 * f1) / N - e["eval_f1"])
        sols.append((res, a, b, p1, r1, f1, (f0 + f1) / 2))
    sols.sort(); res, a, b, p1, r1, f1, mf = sols[0]
    assert res < 1e-9 and sols[1][0] > 1e-6, "solution not exact/unique"
    out.append({"epoch": e["epoch"], "accuracy": e["eval_accuracy"], "weighted_f1": e["eval_f1"], "errors": err,
                "mawdu_fn": a, "mawdu_fp": b, "mawdu_p": p1, "mawdu_r": r1, "mawdu_f1": f1, "macro_f1": mf})
json.dump(out, open(os.path.join(ROOT, "experiments/r2_results.json"), "w"), indent=2)
for o in out: print({k: (round(x, 4) if isinstance(x, float) else x) for k, x in o.items()})

# Same-split reference: the saved v1 LR / RF artefacts (trained on v1 train) scored on the v1 validation split.
import joblib
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
X = v.normalized_text.fillna("").astype(str); y = v.label.values
ref = {}
for mid, fn in [("LR", "model_logistic_regression.joblib"), ("RF", "model_random_forest.joblib")]:
    d = joblib.load(os.path.join(ROOT, "models", fn)); p = d["model"].predict(d["vectorizer"].transform(X))
    ref[mid] = {"mawdu_f1": f1_score(y, p, pos_label=1), "mawdu_p": precision_score(y, p, pos_label=1),
                "mawdu_r": recall_score(y, p, pos_label=1), "macro_f1": f1_score(y, p, average="macro"),
                "accuracy": accuracy_score(y, p), "weighted_f1": f1_score(y, p, average="weighted")}
    print("v1-val", mid, {k: round(x, 4) for k, x in ref[mid].items()})
json.dump({"arabert": out, "saved_baselines_v1_val": ref}, open(os.path.join(ROOT, "experiments/r2_results.json"), "w"), indent=2)
