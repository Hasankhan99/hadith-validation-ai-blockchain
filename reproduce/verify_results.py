"""
Independent check of the frozen thesis results. It never writes to experiments/ and never touches
experiments/final_test.lock or experiments/FREEZE_MANIFEST.json.

  python reproduce/verify_results.py            # fast (~10 s): digests + metrics recomputed from the saved test distributions
  python reproduce/verify_results.py --refit    # full (~10-30 min on CPU): refit every candidate on train, re-score test

Checks
  1. SHA-256 of the train/val/test splits and of the frozen scripts match experiments/FREEZE_MANIFEST.json.
  2. Metrics recomputed from experiments/final_test_probs_{task}.npz equal experiments/final_test_results.csv.
  3. (--refit) Every candidate is refitted from scratch with the frozen settings (same code path as
     experiments/final_test.py) and its test metrics equal the frozen CSV. Output goes to reproduce/out/.
"""
import os, sys, json, argparse
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "experiments"))
sys.path.insert(0, os.path.join(ROOT, "src_v2"))
import final_test as ft  # importing does not run anything; main() is guarded

EXP = os.path.join(ROOT, "experiments")
OUT = os.path.join(ROOT, "reproduce", "out")
TOL = 1e-9


def check_digests(man):
    ok = True
    for p, h in {**man["inputs"], **man["scripts"]}.items():
        good = os.path.exists(os.path.join(ROOT, p)) and ft.sha(os.path.join(ROOT, p)) == h
        ok &= good
        print(f"  [{'OK ' if good else 'BAD'}] {p}")
    return ok


def compare(rows, frozen, label):
    new = pd.DataFrame(rows)
    m = frozen.merge(new, on=["task", "run", "seed", "metric"], how="inner", suffixes=("_frozen", "_new"))
    diff = (m.value_frozen - m.value_new).abs()
    bad = m[diff > TOL]
    print(f"  {label}: {len(m)} metric values compared, {len(bad)} differ (tolerance {TOL})")
    if len(bad):
        print(bad.head(20).to_string())
    return len(m) > 0 and len(bad) == 0


def rows_from_dists(task, y, C, taus):
    rows = []
    for name, D in C.items():
        for s in ft.SEEDS:
            for k, x in ft.metrics(task, y, D[s].argmax(1)).items():
                rows.append(dict(task=task, run=f"FT-{name}", seed=s, metric=k, value=float(x)))
            if name in taus:
                conf, pred = D[s].max(1), D[s].argmax(1); keep = conf >= taus[name]
                rows.append(dict(task=task, run=f"FT-{name}@tau{taus[name]}", seed=s, metric="coverage", value=float(keep.mean())))
                for k, x in ft.metrics(task, y[keep], pred[keep]).items():
                    rows.append(dict(task=task, run=f"FT-{name}@tau{taus[name]}", seed=s, metric="covered_" + k, value=float(x)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refit", action="store_true", help="refit all candidates from the training split and re-score test")
    a = ap.parse_args()
    man = json.load(open(os.path.join(EXP, "FREEZE_MANIFEST.json")))
    frozen = pd.read_csv(os.path.join(EXP, "final_test_results.csv"))
    results = {}

    print("1. Data and script digests vs FREEZE_MANIFEST.json")
    results["digests"] = check_digests(man)

    print("2. Metrics recomputed from the saved frozen test distributions")
    rows = []
    for task in ft.TASKS:
        z = np.load(os.path.join(EXP, f"final_test_probs_{task}.npz"))
        names = sorted({k.split("__s")[0] for k in z.files if "__s" in k})
        C = {n: {s: z[f"{n}__s{s}"] for s in ft.SEEDS} for n in names}
        rows += rows_from_dists(task, z["y"], C, man["tasks"][task]["tau"])
    results["saved_distributions"] = compare(rows, frozen, "saved distributions")

    if a.refit:
        print("3. Full refit from the training split (frozen alpha, prior and tau from the manifest)")
        os.makedirs(OUT, exist_ok=True)
        rows = []
        for task in ft.TASKS:
            t = man["tasks"][task]
            tr, te = ft.load(task, "train"), ft.load(task, "test")
            P = ft.member_probs(task, tr, te, os.path.join(ROOT, "colab", "outputs_absent"), "test")
            C = ft.candidates(task, P, np.array(t["prior"]), t["alpha"])
            rows += rows_from_dists(task, te.label.values, C, t["tau"])
        pd.DataFrame(rows).to_csv(os.path.join(OUT, "refit_test_results.csv"), index=False)
        results["refit"] = compare(rows, frozen, "refit")

    print("\nSummary (mean over seeds 0-4, frozen test set)")
    s = frozen[frozen.metric.isin(["primary", "coverage", "covered_primary"])]
    print(s.groupby(["task", "run", "metric"]).value.agg(["mean", "std"]).round(4).to_string())
    print("\nRESULT:", "ALL CHECKS PASSED" if all(results.values()) else f"FAILED {[k for k, v in results.items() if not v]}")
    sys.exit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    main()
