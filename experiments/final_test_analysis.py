"""
Post-freeze analysis of the single test evaluation (thesis Section 5.x). Reads only the frozen test distributions saved by
final_test.py --stage test (experiments/final_test_probs_{task}.npz); it fits nothing, selects nothing and never re-reads the test CSV.

Outputs:
  experiments/final_test_analysis.json   seed-mean metrics, 95% bootstrap CIs, paired difference headline vs runner-up, confusion matrices
  thesis/figures/fig_pr_binary.png        precision-recall curves for the Mawdu class (seed 0, probabilistic candidates)
  thesis/figures/fig_coverage_test.png    coverage vs covered primary metric for the consensus, frozen tau marked
"""
import os, json
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix, precision_recall_curve, average_precision_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP, FIG = os.path.join(ROOT, "experiments"), os.path.join(ROOT, "thesis", "figures")
SEEDS, B, RNG = [0, 1, 2, 3, 4], 1000, np.random.default_rng(20261030)
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
COLORS = {"M-LR": "#2a78d6", "M-RF": "#eb6834", "M-SVM": "#1baf7a", "M-CNB": "#eda100", "CONS-classical": "#e87ba4"}
STYLES = {"M-LR": ":", "M-RF": "-.", "M-SVM": "--", "M-CNB": (0, (1, 3)), "CONS-classical": "-"}
LABELS = {"M-LR": "LR (word)", "M-RF": "RF (word)", "M-SVM": "Calibrated SVM (char)", "M-CNB": "Complement NB (word)",
          "CONS-classical": "Classical consensus", "M-B6": "Char-SVM, uncalibrated (B6)"}

def primary(task, y, p):
    f = f1_score(y, p, labels=list(range(2 if task == "binary" else 3)), average=None, zero_division=0)
    return f[0] if task == "binary" else f.mean()

def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8.5)
    ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)

man = json.load(open(os.path.join(EXP, "FREEZE_MANIFEST.json")))
out = {}
fig_cov, axes_cov = plt.subplots(1, 2, figsize=(8.6, 3.4))
for ti, task in enumerate(["binary", "ternary"]):
    Z = np.load(os.path.join(EXP, f"final_test_probs_{task}.npz"))
    y = Z["y"]; n = len(y)
    names = sorted({k.split("__s")[0] for k in Z.files if k != "y"})
    preds = {c: np.stack([Z[f"{c}__s{s}"].argmax(1) for s in SEEDS]) for c in names}  # (seeds, n)
    idx = RNG.integers(0, n, size=(B, n))  # the same resamples for every candidate (paired)
    res = {}
    boot = {}
    for c in names:
        point = [primary(task, y, preds[c][s]) for s in SEEDS]
        boot[c] = np.array([np.mean([primary(task, y[i], preds[c][s][i]) for s in SEEDS]) for i in idx])
        res[c] = {"mean": float(np.mean(point)), "std": float(np.std(point, ddof=1)),
                  "ci95": [float(np.percentile(boot[c], 2.5)), float(np.percentile(boot[c], 97.5))]}
    head = man["tasks"][task]["headline"]
    val = {k: v[0] for k, v in man["tasks"][task]["val_primary_mean_std"].items()}
    runner = max((k for k in val if k != head), key=lambda k: val[k])
    d = boot[head] - boot[runner]
    out[task] = {"n_test": int(n), "headline": head, "runner_up_on_val": runner, "candidates": res,
                 "diff_headline_minus_runner": {"mean": float(d.mean()), "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
                                                "share_resamples_headline_better": float((d > 0).mean())},
                 "confusion_seed0": {c: confusion_matrix(y, preds[c][0], labels=list(range(int(y.max()) + 1))).tolist() for c in [head, runner]}}

    # coverage curve of the consensus on test (descriptive; tau was fixed on validation)
    D = [Z[f"CONS-classical__s{s}"] for s in SEEDS]
    grid = np.round(np.arange(0.35, 0.951, 0.05), 2)
    cov = [np.mean([(q.max(1) >= t).mean() for q in D]) for t in grid]
    met = [np.mean([primary(task, y[q.max(1) >= t], q.argmax(1)[q.max(1) >= t]) for q in D]) for t in grid]
    tau = man["tasks"][task]["tau"]["CONS-classical"]
    ax = axes_cov[ti]; style(ax)
    ax.plot(cov, met, color=COLORS["CONS-classical"], lw=2, marker="o", ms=4)
    j = int(np.argmin(np.abs(grid - tau)))
    ax.plot([cov[j]], [met[j]], marker="o", ms=9, mfc="none", mec=INK, mew=1.4)
    ax.annotate(f"frozen tau = {tau}\ncoverage {cov[j]:.3f}", (cov[j], met[j]), xytext=(10, -28), textcoords="offset points", fontsize=8, color=INK)
    ax.set_xlabel("Coverage (share of test items not referred)", fontsize=9, color=INK)
    ax.set_ylabel("Mawdu F1 on covered items" if task == "binary" else "Macro-F1 on covered items", fontsize=9, color=INK)
    ax.set_title("Binary task" if task == "binary" else "Three-class task", fontsize=10, color=INK, loc="left")
    out[task]["coverage_curve"] = {"tau": grid.tolist(), "coverage": [float(v) for v in cov], "covered_primary": [float(v) for v in met]}

    if task == "binary":
        fig, ax = plt.subplots(figsize=(5.6, 4.2)); style(ax)
        ap = {}
        for c in ["M-LR", "M-RF", "M-SVM", "M-CNB", "CONS-classical"]:
            s_ = Z[f"{c}__s0"][:, 0]
            pr, rc, _ = precision_recall_curve((y == 0).astype(int), s_)
            ap[c] = float(average_precision_score((y == 0).astype(int), s_))
            ax.plot(rc, pr, color=COLORS[c], ls=STYLES[c], lw=2, label=f"{LABELS[c]} (AP {ap[c]:.3f})")
        ax.set_xlabel("Recall (Mawdu)", fontsize=9, color=INK); ax.set_ylabel("Precision (Mawdu)", fontsize=9, color=INK)
        ax.set_xlim(0, 1.01); ax.set_ylim(0, 1.02)
        ax.legend(fontsize=7.8, frameon=False, loc="lower left", labelcolor=INK)
        fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_pr_binary.png"), dpi=220); plt.close(fig)
        out[task]["average_precision_seed0"] = ap

fig_cov.tight_layout(); fig_cov.savefig(os.path.join(FIG, "fig_coverage_test.png"), dpi=220)
json.dump(out, open(os.path.join(EXP, "final_test_analysis.json"), "w"), indent=1)
print(json.dumps({t: {"headline": out[t]["headline"], "runner": out[t]["runner_up_on_val"],
                      "cands": {c: [round(v["mean"], 4), round(v["std"], 4), [round(x, 4) for x in v["ci95"]]] for c, v in out[t]["candidates"].items()},
                      "diff": out[t]["diff_headline_minus_runner"]} for t in out}, indent=1))
