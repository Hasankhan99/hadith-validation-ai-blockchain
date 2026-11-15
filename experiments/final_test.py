"""
Frozen final evaluation (thesis Section 3.7). Two stages; the test split is read only in stage `test`, once.

  python experiments/final_test.py --stage val
      Fits every candidate on the training split, scores VALIDATION, selects tau per consensus candidate with the R7 rule
      (largest tau with mean validation coverage >= 0.90), picks the headline system (highest mean validation primary metric),
      and writes experiments/FREEZE_MANIFEST.json with SHA-256 digests of the inputs. No test file is opened.
  python experiments/final_test.py --stage test --confirm-freeze
      Verifies the manifest, then scores ALL candidates on the test split exactly once and writes experiments/final_test_results.csv.
      A lock file (experiments/final_test.lock) prevents a second run. Results are reported whether or not they flatter the system.

Candidates (per task): the four classical members, the calibrated char-SVM, the classical consensus (alpha from R6, uniform weights),
each transformer found in transformer_dir (mean over seeds), and the consensus over classical members + transformers (same alpha).
Transformer outputs come from colab/train_transformer_v2.py: val_probs_*.npy and sealed_test_preds/test_probs_*.npy (seed-aligned).
"""
import os, sys, json, glob, hashlib, argparse
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import ComplementNB
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src_v2"))
from consensus_soft import consensus_distribution

EXP = os.path.join(ROOT, "experiments")
SEEDS = [0, 1, 2, 3, 4]
TAUS = [round(0.35 + 0.05 * i, 2) for i in range(12)]
word = lambda: TfidfVectorizer(ngram_range=(1, 2), max_features=10000)
char = lambda: TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), max_features=50000, sublinear_tf=True)
MEMBERS = {  # identical to R6/R7
    "LR": (word, lambda s: LogisticRegression(max_iter=2000, class_weight="balanced", random_state=s)),
    "RF": (word, lambda s: RandomForestClassifier(n_estimators=100, class_weight="balanced_subsample", random_state=s, n_jobs=-1)),
    "SVM": (char, lambda s: CalibratedClassifierCV(LinearSVC(class_weight="balanced", random_state=s), cv=3, method="sigmoid")),
    "CNB": (word, lambda s: ComplementNB()),
}
EXTRA = ["B6"]  # uncalibrated char-SVM (hard labels, one-hot); best single model of R1/R3, evaluated alone and never inside the consensus
TASKS = {"binary": "processed_v2", "ternary": "processed_3class"}

def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()

def metrics(task, y, p):
    K = 2 if task == "binary" else 3
    f = f1_score(y, p, labels=list(range(K)), average=None, zero_division=0)
    o = {"macro_f1": f.mean(), "accuracy": accuracy_score(y, p),
         "mawdu_p": precision_score(y, p, labels=[0], average="micro", zero_division=0),
         "mawdu_r": recall_score(y, p, labels=[0], average="micro", zero_division=0)}
    o["primary"] = f[0] if task == "binary" else f.mean()
    o["f1_mawdu"] = f[0]
    if task == "ternary":
        o["f1_daif"], o["f1_sahih"] = f[1], f[2]
    return o

def load(task, split):
    df = pd.read_csv(os.path.join(ROOT, "data", TASKS[task], f"processed_{split}.csv"))
    df["normalized_text"] = df.normalized_text.fillna("").astype(str)
    return df

def member_probs(task, tr, ev, tdir, split):
    """Returns {candidate_member: {seed: (n,K) probs}} on the evaluation frame ev (validation or test)."""
    ytr = tr.label.values
    out = {}
    for name, (vf, cf) in MEMBERS.items():
        v = vf(); Xtr = v.fit_transform(tr.normalized_text); Xev = v.transform(ev.normalized_text)
        out[name] = {s: cf(s).fit(Xtr, ytr).predict_proba(Xev) for s in SEEDS}
        print(task, split, "fitted", name, flush=True)
    v = char(); Xtr = v.fit_transform(tr.normalized_text); Xev = v.transform(ev.normalized_text)
    K = int(ytr.max()) + 1
    out["B6"] = {s: np.eye(K)[LinearSVC(class_weight="balanced", random_state=s).fit(Xtr, ytr).predict(Xev)] for s in SEEDS}
    for f in sorted(glob.glob(os.path.join(tdir, f"val_probs_{task}_*_s0.npy"))):
        short = os.path.basename(f)[len("val_probs_"):-len("_s0.npy")]
        name = short.split("_", 1)[1]
        seeds = {}
        for s in SEEDS:
            pth = (os.path.join(tdir, f"val_probs_{short}_s{s}.npy") if split == "val"
                   else os.path.join(tdir, "sealed_test_preds", f"test_probs_{short}_s{s}.npy"))
            if os.path.exists(pth):
                seeds[s] = np.load(pth)
        if len(seeds) == len(SEEDS):
            out[name] = seeds
    return out

def candidates(task, P, prior, alpha):
    """Maps candidate name -> {seed: (n,K) distribution}."""
    C = {f"M-{n}": P[n] for n in P}
    classical = list(MEMBERS)
    trans = [n for n in P if n not in MEMBERS and n not in EXTRA]
    C["CONS-classical"] = {s: consensus_distribution([P[n][s] for n in classical], prior, alpha) for s in SEEDS}
    if trans:
        C["CONS-all"] = {s: consensus_distribution([P[n][s] for n in classical + trans], prior, alpha) for s in SEEDS}
    return C

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["val", "test"], required=True)
    ap.add_argument("--confirm-freeze", action="store_true")
    ap.add_argument("--transformer_dir", default=os.path.join(ROOT, "colab", "outputs"))
    a = ap.parse_args()
    r6 = pd.read_csv(os.path.join(EXP, "r6_val_results.csv"))
    man_path = os.path.join(EXP, "FREEZE_MANIFEST.json"); lock = os.path.join(EXP, "final_test.lock")
    rows = []

    if a.stage == "val":
        manifest = {"inputs": {}, "tasks": {}}
        for task in TASKS:
            tr, va = load(task, "train"), load(task, "val")
            K = int(tr.label.max()) + 1; prior = np.bincount(tr.label.values, minlength=K) / len(tr)
            alpha = float(r6[(r6.task == task) & (r6.run == "CHOSEN-alpha")].value.iloc[0])
            P = member_probs(task, tr, va, a.transformer_dir, "val")
            C = candidates(task, P, prior, alpha)
            summ, taus = {}, {}
            for name, D in C.items():
                m = [metrics(task, va.label.values, D[s].argmax(1))["primary"] for s in SEEDS]
                summ[name] = (float(np.mean(m)), float(np.std(m, ddof=1)))
                for s, x in zip(SEEDS, m):
                    rows.append(dict(task=task, run=f"FV-{name}", seed=s, metric="primary", value=x))
                if name.startswith("CONS"):
                    cov = {t: np.mean([(D[s].max(1) >= t).mean() for s in SEEDS]) for t in TAUS}
                    taus[name] = max(t for t in TAUS if cov[t] >= 0.90)
            head = max(summ, key=lambda k: summ[k][0])
            manifest["tasks"][task] = {"alpha": alpha, "prior": prior.tolist(), "val_primary_mean_std": summ,
                                       "tau": taus, "headline": head,
                                       "transformers_present": sorted(n for n in P if n not in MEMBERS and n not in EXTRA)}
            print(task, "validation primary:", {k: round(v[0], 4) for k, v in summ.items()}, "-> headline", head, "tau", taus)
            for pth in [f"data/{TASKS[task]}/processed_{s}.csv" for s in ("train", "val", "test")]:
                manifest["inputs"][pth] = sha(os.path.join(ROOT, pth))
        manifest["scripts"] = {p: sha(os.path.join(ROOT, p)) for p in ["experiments/final_test.py", "src_v2/consensus_soft.py", "experiments/r6_val_results.csv"]}
        json.dump(manifest, open(man_path, "w"), indent=1)
        pd.DataFrame(rows).to_csv(os.path.join(EXP, "final_val_results.csv"), index=False)
        print("manifest written:", man_path)
        return

    # ---- stage test ----
    if not a.confirm_freeze:
        sys.exit("Refusing: pass --confirm-freeze after reading FREEZE_PROTOCOL.md.")
    if os.path.exists(lock):
        sys.exit("Refusing: the test split has already been scored once (final_test.lock exists).")
    if not os.path.exists(man_path):
        sys.exit("Refusing: run --stage val first.")
    man = json.load(open(man_path))
    for p, h in man["inputs"].items():
        if sha(os.path.join(ROOT, p)) != h:
            sys.exit(f"Refusing: {p} changed after the freeze.")
    for p, h in man["scripts"].items():
        if sha(os.path.join(ROOT, p)) != h:
            sys.exit(f"Refusing: {p} changed after the freeze.")
    open(lock, "w").write("test split scored once\n")
    for task in TASKS:
        t = man["tasks"][task]
        tr, te = load(task, "train"), load(task, "test")
        P = member_probs(task, tr, te, a.transformer_dir, "test")
        assert sorted(n for n in P if n not in MEMBERS and n not in EXTRA) == t["transformers_present"], "transformer set differs from the freeze"
        C = candidates(task, P, np.array(t["prior"]), t["alpha"])
        # Frozen test distributions are saved so that confidence intervals and confusion matrices need no second test pass.
        np.savez_compressed(os.path.join(EXP, f"final_test_probs_{task}.npz"), y=te.label.values,
                            **{f"{name}__s{s}": D[s] for name, D in C.items() for s in SEEDS})
        for name, D in C.items():
            for s in SEEDS:
                for k, x in metrics(task, te.label.values, D[s].argmax(1)).items():
                    rows.append(dict(task=task, run=f"FT-{name}", seed=s, metric=k, value=float(x)))
                if name in t["tau"]:
                    conf, pred = D[s].max(1), D[s].argmax(1); keep = conf >= t["tau"][name]
                    rows.append(dict(task=task, run=f"FT-{name}@tau{t['tau'][name]}", seed=s, metric="coverage", value=float(keep.mean())))
                    for k, x in metrics(task, te.label.values[keep], pred[keep]).items():
                        rows.append(dict(task=task, run=f"FT-{name}@tau{t['tau'][name]}", seed=s, metric="covered_" + k, value=float(x)))
    df = pd.DataFrame(rows); df.to_csv(os.path.join(EXP, "final_test_results.csv"), index=False)
    print(df[df.metric.isin(["primary", "macro_f1"])].groupby(["task", "run", "metric"]).value.agg(["mean", "std"]).round(4).to_string())

if __name__ == "__main__":
    main()
