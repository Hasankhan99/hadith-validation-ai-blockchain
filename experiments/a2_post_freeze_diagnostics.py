"""A2: post-freeze descriptive diagnostics on the training and validation splits (run 2026-10-31).

No model is fitted, nothing is selected, and no test file is opened. These counts qualify statements in the thesis:
  - how the isnad/matn views of R4/R5 were actually produced (formula match, 35% positional fallback, or whole text);
  - whether Arabic marks or punctuation that normalisation keeps still separate the Da'if sources;
  - the length of the binary validation errors of R4 (median as well as mean).
Output: experiments/a2_diagnostics.json
"""
import os, re, json
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FORMULAS = [r'أن رسول الله', r'قال رسول الله', r'سمعت رسول الله', r'عن النبي', r'أن النبي', r'قال النبي']
NAMES = {"binary": {0: "Mawdu", 1: "Sahih"}, "ternary": {0: "Mawdu", 1: "Daif", 2: "Sahih"}}
RESIDUAL_MARKS = re.compile(r'[ؐ-ؖٓ-ٰٟۖ-ۭ]')
AR_PUNCT = re.compile(r'[،؛؟]')  # Arabic comma, semicolon, question mark

def view_kind(t):
    if any(re.search(p + r'\b', t) for p in FORMULAS):
        return "formula"
    return "positional" if len(t.split()) > 15 else "whole_text"

def load(task, split):
    d = {"binary": "processed_v2", "ternary": "processed_3class"}[task]
    df = pd.read_csv(os.path.join(ROOT, "data", d, f"processed_{split}.csv"))
    df["raw_text"] = df.raw_text.fillna("").astype(str)
    df["normalized_text"] = df.normalized_text.fillna("").astype(str)
    return df

out = {"segmenter_views": {}, "residual_characters_train": {}, "binary_val_error_length_words": {}}
for task in ["binary", "ternary"]:
    for split in ["train", "val"]:
        df = load(task, split)
        kinds = df.raw_text.map(view_kind)
        rec = {"n": len(df), **{k: round(float((kinds == k).mean()), 4) for k in ["formula", "positional", "whole_text"]}}
        for lab, name in NAMES[task].items():
            rec[f"formula_share_{name}"] = round(float((kinds[df.label == lab] == "formula").mean()), 4)
        out["segmenter_views"][f"{task}-{split}"] = rec

tr = load("ternary", "train")
for src, g in tr.groupby(["source", "label"]):
    key = f"{src[0]}-label{src[1]}"
    txt = g.normalized_text
    out["residual_characters_train"][key] = {
        "n": len(g),
        "frac_with_residual_marks": round(float(txt.map(lambda t: bool(RESIDUAL_MARKS.search(t))).mean()), 4),
        "frac_with_arabic_punctuation": round(float(txt.map(lambda t: bool(AR_PUNCT.search(t))).mean()), 4),
    }

va = load("binary", "val")
all_w = va.raw_text.str.split().map(len)
err = pd.read_csv(os.path.join(ROOT, "experiments", "r4_val_errors_binary.csv"))
err_w = err.raw_text.fillna("").astype(str).str.split().map(len)
out["binary_val_error_length_words"] = {
    "errors_n": int(len(err)), "errors_mean": round(float(err_w.mean()), 1), "errors_median": float(err_w.median()),
    "errors_max": int(err_w.max()), "errors_mean_without_longest": round(float(np.sort(err_w.values)[:-1].mean()), 1),
    "all_n": int(len(va)), "all_mean": round(float(all_w.mean()), 1), "all_median": float(all_w.median()),
}
json.dump(out, open(os.path.join(ROOT, "experiments", "a2_diagnostics.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
