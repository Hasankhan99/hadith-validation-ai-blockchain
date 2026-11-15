"""
Unified Dataset Loader for the Hadith authenticity benchmark (v2).

Changes from v1 (see CHANGELOG.md):
  - Uses the official MAHADDAT train/dev/test split (Gaanoun & Alsuhaibani, 2022)
    instead of pooling all files and re-splitting.
  - Fixes the label inversion: in *Final.csv, Degree=1 denotes Mawdu (2,452 rows,
    identical to MAHADDAT.csv) and Degree=0 denotes Sahih (LK corpus).
  - Removes exact (normalized-text) duplicates within splits and any dev/test item
    already present in a preceding split, so no evaluation text is seen in training.
  - LK-Hadith is not loaded separately: the MAHADDAT authentic class already is the
    LK corpus, and v1's column auto-detection read chapter titles, not hadiths.
Labels:
  1 = Sahih / Authentic
  0 = Mawdu / Fabricated
"""

import os
import pandas as pd
from preprocessing import normalize_arabic
from sanad_segmenter import segment_isnad_matn

MIN_CHARS = 15

def load_mahaddat_split(base_dir: str, split: str) -> pd.DataFrame:
    path = os.path.join(base_dir, "MAHADDAT", f"{split}Final.csv")
    df = pd.read_csv(path, encoding="utf-8-sig")
    df = df[df["Matan"].notna()].copy()
    df["raw_text"] = df["Matan"].astype(str).str.strip()
    df = df[df["raw_text"].str.len() >= MIN_CHARS]
    df["normalized_text"] = df["raw_text"].map(normalize_arabic)
    seg = df["raw_text"].map(segment_isnad_matn)
    df["isnad"] = seg.map(lambda s: s["isnad"])
    df["matn"] = seg.map(lambda s: s["matn"])
    df["label"] = (df["Degree"].astype(int) == 0).astype(int)  # Degree 1 = Mawdu -> label 0
    df["source"] = "MAHADDAT"
    return df[["source", "raw_text", "normalized_text", "isnad", "matn", "label"]].reset_index(drop=True)

def build_benchmark_dataset(base_dir: str, output_dir: str) -> dict:
    os.makedirs(output_dir, exist_ok=True)
    seen = set()
    stats = {}
    for split, name in [("train", "train"), ("dev", "val"), ("test", "test")]:
        df = load_mahaddat_split(base_dir, split)
        n0 = len(df)
        df = df.drop_duplicates(subset=["normalized_text"])
        n_dup = n0 - len(df)
        leak = df["normalized_text"].isin(seen)
        df = df[~leak]
        seen.update(df["normalized_text"])
        df.to_csv(os.path.join(output_dir, f"processed_{name}.csv"), index=False, encoding="utf-8-sig")
        stats[name] = {"raw": n0, "dup_removed": n_dup, "cross_split_removed": int(leak.sum()),
                       "final": len(df), "sahih": int(df.label.sum()), "mawdu": int((df.label == 0).sum())}
        print(name, stats[name])
    return stats

if __name__ == "__main__":
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    build_benchmark_dataset(os.path.join(root, "data"), os.path.join(root, "data", "processed_v2"))
