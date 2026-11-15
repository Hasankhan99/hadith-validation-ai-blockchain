"""A0: Data audit of the preliminary (v1) pipeline and statistics of the corrected (v2) benchmark."""
import os, sys, json
import pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src_v2"))
from preprocessing import normalize_arabic
from dataset_loader import build_benchmark_dataset

res = {}
v1 = pd.concat([pd.read_csv(os.path.join(ROOT, f"data/processed/processed_{s}.csv")) for s in ["train", "val", "test"]])
fab = set(normalize_arabic(str(x)) for x in pd.read_csv(os.path.join(ROOT, "data/MAHADDAT/MAHADDAT.csv"), encoding="utf-8-sig")["Matan"].dropna())
hit = v1[v1.normalized_text.isin(fab)]
res["v1_total"] = len(v1)
res["v1_known_fabricated_found"] = len(hit)
res["v1_known_fabricated_labelled_sahih"] = int((hit.label == 1).sum())
lk = v1[v1.source == "LK-Hadith"]
res["v1_lk_records"] = len(lk)
res["v1_lk_mean_words"] = round(lk.normalized_text.str.split().str.len().mean(), 1)
for s in ["train", "dev", "test"]:
    df = pd.read_csv(os.path.join(ROOT, f"data/MAHADDAT/{s}Final.csv"), encoding="utf-8-sig")
    res[f"official_{s}_rows"] = len(df)
    res[f"official_{s}_degree1"] = int((df.Degree == 1).sum())
    res[f"official_{s}_degree1_in_MAHADDATcsv"] = int(df[df.Degree == 1].Matan.astype(str).map(normalize_arabic).isin(fab).sum())
res["v2"] = build_benchmark_dataset(os.path.join(ROOT, "data"), os.path.join(ROOT, "data", "processed_v2"))
json.dump(res, open(os.path.join(ROOT, "experiments/a0_audit.json"), "w"), indent=2)
print(json.dumps(res, indent=1))
