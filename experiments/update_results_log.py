"""Regenerates thesis/RESULTS_LOG.csv from experiment outputs, so every thesis number traces to a row."""
import os, json, csv
import pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rows = []
add = lambda **r: rows.append(r)

a0 = json.load(open(os.path.join(ROOT, "experiments/a0_audit.json")))
for k, v in a0.items():
    if k == "v2":
        for split, st in v.items():
            for m, x in st.items():
                add(run_id="A0-V2", change="v2 benchmark construction (src_v2/dataset_loader.py)", config="official split; min 15 chars; dedup; cross-split leak removal",
                    seed="n/a", split=f"v2-{split}", metric=m, result=x, date="2026-10-24", artifact="experiments/a0_audit.json")
    else:
        add(run_id="A0-AUDIT", change="audit of preliminary pipeline / official files", config="count", seed="n/a", split="data",
            metric=k, result=v, date="2026-10-24", artifact="experiments/a0_audit.json")

r0 = json.load(open(os.path.join(ROOT, "experiments/r0_results.json")))
desc = {"model_lr": ("R0-LR", "LR word TF-IDF(1-2,10k), saved artefact"), "model_rf": ("R0-RF", "RF 100 trees word TF-IDF, saved artefact"),
        "ensemble_weighted": ("R0-ENS", "weighted consensus LR+RF, weights .9719/.9780 (test acc), conf=pred prob"),
        "majority_class": ("R0-MAJ", "predict label 0 for all")}
for k, (rid, cfg) in desc.items():
    for met, key in [("accuracy", "accuracy"), ("weighted_precision", "w_precision"), ("weighted_recall", "w_recall"),
                     ("weighted_f1", "w_f1"), ("macro_f1", "macro_f1"), ("minority_class_f1", "f1_label1")]:
        add(run_id=rid, change="Reproduction of defense-paper numbers, no retraining (v1 data, inverted labels)", config=cfg,
            seed=42, split="v1-test", metric=met, result=round(r0[k][key], 4), date="2026-10-24", artifact="experiments/r0_results.json")

r1 = pd.read_csv(os.path.join(ROOT, "experiments/r1_val_per_seed.csv"))
cfgs = {"B0_majority": "most_frequent", "B1_lr_word_orig": "LR lbfgs max_iter=1000; word TF-IDF(1-2,10k)",
        "B2_rf_word_orig": "RF 100 trees; word TF-IDF(1-2,10k)", "B3_cnb_word": "ComplementNB alpha=1; word TF-IDF(1-2,10k)",
        "B4_svm_word": "LinearSVC C=1; word TF-IDF(1-2,10k)", "B5_lr_word_bal": "LR class_weight=balanced; word TF-IDF(1-2,10k)",
        "B6_svm_char_bal": "LinearSVC C=1 class_weight=balanced; char_wb TF-IDF(2-5,50k,sublinear)",
        "E1_ens_lr_rf_weighted": "weighted consensus B1+B2, weights=mean val macro-F1, conf=pred prob"}
mets = ["mawdu_f1", "mawdu_p", "mawdu_r", "macro_f1", "weighted_f1", "accuracy"]
for run, g in r1.groupby("run", sort=False):
    rid = "R1-" + run.split("_")[0]
    for _, r in g.iterrows():
        for m in mets:
            add(run_id=rid, change="v2 benchmark: official split, fixed labels, leakage removed", config=cfgs[run], seed=int(r.seed),
                split="v2-val", metric=m, result=round(r[m], 4), date="2026-10-24", artifact="experiments/r1_val_per_seed.csv")
    for m in mets:
        add(run_id=rid, change="aggregate over seeds", config=cfgs[run], seed="0-4", split="v2-val", metric=m + "_mean±std",
            result=f"{g[m].mean():.4f} ± {g[m].std():.4f}", date="2026-10-24", artifact="experiments/r1_val_per_seed.csv")

r2 = json.load(open(os.path.join(ROOT, "experiments/r2_results.json")))
for e in r2["arabert"]:
    for m in ["accuracy", "weighted_f1", "mawdu_f1", "mawdu_p", "mawdu_r", "macro_f1", "mawdu_fn", "mawdu_fp"]:
        add(run_id=f"R2-ARABERT-ep{int(e['epoch'])}", change="Historical Colab run (user log 2026-09-24); per-class metrics recovered exactly from weighted metrics",
            config="aubmindlab/bert-base-arabertv02, max_len 128, lr 2e-5, bs 16, 3 ep, best-on-val weighted F1 (final = ep3)", seed=42,
            split="v1-val", metric=m, result=round(e[m], 4) if isinstance(e[m], float) else e[m], date="2026-09-24",
            artifact="experiments/logs/colab_arabert_v1_2026-09-24.txt; experiments/r2_results.json")
for mid, r in r2["saved_baselines_v1_val"].items():
    for m, x in r.items():
        add(run_id=f"R2-{mid}", change="Saved v1 artefact scored on v1 validation split (same-split reference for R2)", config="saved models/*.joblib",
            seed=42, split="v1-val", metric=m, result=round(x, 4), date="2026-10-24", artifact="experiments/r2_results.json")

r3 = pd.read_csv(os.path.join(ROOT, "experiments/r3_val_per_seed.csv"))
cfg3 = {"T0_majority": "most_frequent", "T1_lr_word": "LR lbfgs max_iter=1000; word TF-IDF(1-2,10k)", "T2_rf_word": "RF 100 trees; word TF-IDF(1-2,10k)",
        "T3_svm_word": "LinearSVC C=1; word TF-IDF(1-2,10k)", "T4_lr_word_bal": "LR class_weight=balanced; word TF-IDF(1-2,10k)",
        "T5_svm_char_bal": "LinearSVC C=1 class_weight=balanced; char_wb TF-IDF(2-5,50k,sublinear)"}
m3 = ["macro_f1", "f1_mawdu", "f1_daif", "f1_sahih", "accuracy", "daif_recall_LK", "daif_recall_NAH"]
for run, g in r3.groupby("run", sort=False):
    rid = "R3-" + run.split("_")[0]
    for _, r in g.iterrows():
        for m in m3:
            add(run_id=rid, change="three-class benchmark (src_v2/dataset_loader_3class.py)", config=cfg3.get(run, run), seed=int(r.seed),
                split="3class-val", metric=m, result=round(r[m], 4), date="2026-10-24", artifact="experiments/r3_val_per_seed.csv")
    for m in m3:
        add(run_id=rid, change="aggregate over seeds", config=cfg3.get(run, run), seed="0-4", split="3class-val", metric=m + "_mean±std",
            result=f"{g[m].mean():.4f} ± {g[m].std():.4f}", date="2026-10-24", artifact="experiments/r3_val_per_seed.csv")

def long_csv(fname, prefix, change, date):
    f = os.path.join(ROOT, "experiments", fname)
    if not os.path.exists(f): return
    d = pd.read_csv(f)
    for _, r in d.iterrows():
        add(run_id=f"{prefix}-{r.run}", change=change, config=f"task={r.task}" + (f"; {r.note}" if isinstance(r.get('note'), str) else ""),
            seed=int(r.seed), split=(("3class-test" if r.task == "ternary" else "v2-test") if prefix == "FT" else "train-OOF" if str(r.run).startswith("OOF") or "OOF" in str(r.get('note', '')) else ("3class-val" if r.task == "ternary" else "v2-val")),
            metric=r.metric, result=round(r.value, 4), date=date, artifact=f"experiments/{fname}")
    for (t, run, m), g in d.groupby(["task", "run", "metric"], sort=False):
        if len(g) > 1:
            add(run_id=f"{prefix}-{run}", change="aggregate over seeds", config=f"task={t}", seed="0-4", split=(("3class-test" if t == "ternary" else "v2-test") if prefix == "FT" else "3class-val" if t == "ternary" else "v2-val"),
                metric=m + "_mean±std", result=f"{g.value.mean():.4f} ± {g.value.std():.4f}", date=date, artifact=f"experiments/{fname}")
long_csv("r4_val_results.csv", "R4", "consensus, member ablation, tau sweep, view and near-duplicate analyses (validation only)", "2026-10-25")
long_csv("r5_val_results.csv", "R5", "input-view ablation with normalised views and raw-text diagnostic (validation only)", "2026-10-25")
long_csv("r6_val_results.csv", "R6", "balanced members + prior-corrected consensus (validation only)", "2026-10-25")
long_csv("r7_val_results.csv", "R7", "abstention threshold selection for the frozen consensus (validation only)", "2026-10-25")
long_csv("final_val_results.csv", "FV", "frozen-candidate validation scores before the test (validation only)", "2026-10-30")
long_csv("final_test_results.csv", "FT", "FINAL test evaluation, scored once at the freeze", "2026-10-30")
fa = os.path.join(ROOT, "experiments", "final_test_analysis.json")
if os.path.exists(fa):
    for t, A in json.load(open(fa)).items():
        sp = "3class-test" if t == "ternary" else "v2-test"; art = "experiments/final_test_analysis.json"
        for c, v in A["candidates"].items():
            add(run_id=f"FA-{c}", change="post-freeze analysis of saved test distributions (no refit)", config=f"task={t}; 1000 paired bootstrap resamples, seed-mean primary metric",
                seed="0-4", split=sp, metric="primary_ci95", result=f"[{v['ci95'][0]:.4f}, {v['ci95'][1]:.4f}]", date="2026-10-30", artifact=art)
        d = A["diff_headline_minus_runner"]
        add(run_id="FA-DIFF", change="paired bootstrap difference, headline minus validation runner-up", config=f"task={t}; {A['headline']} - {A['runner_up_on_val']}",
            seed="0-4", split=sp, metric="diff_mean [ci95]; P(diff>0)", result=f"{d['mean']:.4f} [{d['ci95'][0]:.4f}, {d['ci95'][1]:.4f}]; {d['share_resamples_headline_better']:.3f}", date="2026-10-30", artifact=art)
        for c, cm in A["confusion_seed0"].items():
            add(run_id=f"FA-CM-{c}", change="confusion matrix (rows true, cols predicted; class order Mawdu, [Daif,] Sahih)", config=f"task={t}", seed=0, split=sp,
                metric="confusion_matrix", result=json.dumps(cm), date="2026-10-30", artifact=art)
        for c, ap in A.get("average_precision_seed0", {}).items():
            add(run_id=f"FA-AP-{c}", change="average precision, Mawdu class", config=f"task={t}", seed=0, split=sp, metric="average_precision", result=round(ap, 4), date="2026-10-30", artifact=art)
        for tau, cv, m in zip(*[A["coverage_curve"][k] for k in ("tau", "coverage", "covered_primary")]):
            add(run_id=f"FA-COV-tau{tau}", change="descriptive coverage curve of CONS-classical on test (tau fixed on validation)", config=f"task={t}", seed="0-4", split=sp,
                metric="coverage | covered_primary", result=f"{cv:.4f} | {m:.4f}", date="2026-10-30", artifact=art)
dg = os.path.join(ROOT, "experiments", "r5_diacritics_diag.csv")
if os.path.exists(dg):
    for _, r in pd.read_csv(dg).iterrows():
        add(run_id="R5-DIACRITICS", change="fraction of training texts containing Arabic diacritics", config=f"task={r.task}; class={r.cls}; source={r.source}; n={r.n}",
            seed="n/a", split="train", metric="frac_with_diacritics", result=round(r.frac_with_diacritics, 4), date="2026-10-25", artifact="experiments/r5_diacritics_diag.csv")
for t in ["binary", "ternary"]:
    ef = os.path.join(ROOT, "experiments", f"r4_error_analysis_{t}.json")
    if os.path.exists(ef):
        for k, v in json.load(open(ef)).items():
            add(run_id="R4-ERR", change="error analysis, uniform consensus seed 0", config=f"task={t}", seed=0, split="3class-val" if t == "ternary" else "v2-val",
                metric=k, result=json.dumps(v) if isinstance(v, list) else round(v, 4), date="2026-10-25", artifact=ef.split("research paper/")[-1] if False else f"experiments/r4_error_analysis_{t}.json")

lp = os.path.join(ROOT, "experiments/a1_length_stats.json")
if os.path.exists(lp):
    for k, v in json.load(open(lp)).items():
        for m, x in v.items():
            add(run_id="A1-LEN", change="text length in words per class (descriptive)", config=k, seed="n/a", split="all splits",
                metric=m, result=x, date="2026-10-30", artifact="experiments/a1_length_stats.json")

a2p = os.path.join(ROOT, "experiments/a2_diagnostics.json")
if os.path.exists(a2p):
    a2 = json.load(open(a2p)); art = "experiments/a2_diagnostics.json"
    for group, recs in [(g, r) for g, r in a2.items() if g != "binary_val_error_length_words"]:
        for name, rec in recs.items():
            for m, x in rec.items():
                add(run_id="A2-" + group, change="post-freeze descriptive diagnostic; no model fitted, no test file opened", config=name, seed="n/a",
                    split="train/val", metric=m, result=x, date="2026-10-31", artifact=art)
    for m, x in a2["binary_val_error_length_words"].items():
        add(run_id="A2-ERRLEN", change="post-freeze descriptive diagnostic of R4 binary errors; no model fitted", config="task=binary; words in raw_text", seed=0,
            split="v2-val", metric=m, result=x, date="2026-10-31", artifact=art)

b1p = os.path.join(ROOT, "experiments/b1_fabric/b1_fabric_results.json")
if os.path.exists(b1p):
    b1 = json.load(open(b1p)); d = "2026-10-25"; art = "experiments/b1_fabric/b1_fabric_results.json"
    cfgb = "Fabric 2.5.16 test network, 2 orgs, 1 Raft orderer, default endorsement policy, synthetic payloads, Docker Desktop 4 CPU / 8 GB"
    for t in b1["security"]:
        add(run_id="B1-SEC", change="chaincode v2 functional/tamper tests", config=t["test"], seed="n/a", split="fabric", metric="behaved_as_expected",
            result=bool(t["expected_message_found"]), date=d, artifact=art)
    for k, v in b1["sequential_submit"].items():
        add(run_id="B1-SEQ", change="sequential submit latency", config=cfgb, seed="n/a", split="fabric", metric=k, result=round(v, 2), date=d, artifact=art)
    for c in b1["concurrent"]:
        for k in ["submitted", "committed", "failed", "elapsed_s", "tps"]:
            add(run_id=f"B1-C{c['concurrency']}", change="concurrent submit throughput", config=cfgb, seed="n/a", split="fabric", metric=k, result=round(c[k], 2), date=d, artifact=art)
        for k, v in c["latency"].items():
            add(run_id=f"B1-C{c['concurrency']}", change="concurrent submit latency", config=cfgb, seed="n/a", split="fabric", metric=k, result=round(v, 2), date=d, artifact=art)
    for k, v in b1["query"].items():
        add(run_id="B1-QRY", change="GetValidation evaluate latency", config=cfgb, seed="n/a", split="fabric", metric=k, result=round(v, 2), date=d, artifact=art)

out = os.path.join(ROOT, "thesis/RESULTS_LOG.csv")
with open(out, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["run_id", "change", "config", "seed", "split", "metric", "result", "date", "artifact"])
    w.writeheader(); w.writerows(rows)
print(len(rows), "rows ->", out)
