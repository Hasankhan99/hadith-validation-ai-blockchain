"""
Unified Dataset Loader & Benchmark Processor for Hadith Corpora.
Combines LK-Hadith-Corpus, AR-Sanad-280K, and MAHADDAT into clean, balanced train/val/test datasets.
Labels:
  1 = Authentic / Sahih / Canonical
  0 = Unauthentic / Mawdu / Fabricated
"""

import os
import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from preprocessing import normalize_arabic
from sanad_segmenter import segment_isnad_matn

def load_lk_hadith_corpus(base_dir: str) -> pd.DataFrame:
    """Parses CSV files from LK-Hadith-Corpus (Authentic Canonical Hadiths)."""
    corpus_path = os.path.join(base_dir, "LK-Hadith-Corpus")
    if not os.path.exists(corpus_path):
        return pd.DataFrame()

    csv_files = glob.glob(os.path.join(corpus_path, "**", "*.csv"), recursive=True)
    records = []
    for fpath in csv_files:
        try:
            df = pd.read_csv(fpath, encoding='utf-8-sig', on_bad_lines='skip')
            text_col = None
            for col in df.columns:
                if any(k in col.lower() for k in ['hadith', 'text', 'arabic', 'matn']):
                    text_col = col
                    break
            if text_col is None and len(df.columns) > 0:
                text_col = df.columns[0]

            if text_col:
                for val in df[text_col].dropna():
                    val_str = str(val).strip()
                    if len(val_str) > 15:
                        norm = normalize_arabic(val_str)
                        seg = segment_isnad_matn(val_str)
                        records.append({
                            "source": "LK-Hadith",
                            "raw_text": val_str,
                            "normalized_text": norm,
                            "isnad": seg["isnad"],
                            "matn": seg["matn"],
                            "label": 1
                        })
        except Exception:
            continue

    df_lk = pd.DataFrame(records)
    print(f"Loaded {len(df_lk)} records from LK-Hadith-Corpus (Label 1).")
    return df_lk

def load_mahaddat(base_dir: str) -> pd.DataFrame:
    """Parses MAHADDAT split datasets (trainFinal, testFinal, devFinal, MAHADDAT.csv)."""
    mahaddat_path = os.path.join(base_dir, "MAHADDAT")
    if not os.path.exists(mahaddat_path):
        return pd.DataFrame()

    csv_files = [
        os.path.join(mahaddat_path, "trainFinal.csv"),
        os.path.join(mahaddat_path, "testFinal.csv"),
        os.path.join(mahaddat_path, "devFinal.csv"),
        os.path.join(mahaddat_path, "MAHADDAT.csv")
    ]

    records = []
    for fpath in csv_files:
        if not os.path.exists(fpath):
            continue
        try:
            df = pd.read_csv(fpath, encoding='utf-8-sig', on_bad_lines='skip')
            text_col = 'Matan' if 'Matan' in df.columns else ('Full Hadith' if 'Full Hadith' in df.columns else df.columns[1])
            
            for _, row in df.iterrows():
                val_str = str(row[text_col]).strip() if pd.notna(row[text_col]) else ""
                if len(val_str) < 15:
                    continue

                # Default label
                lbl = 1
                if 'Degree' in df.columns and pd.notna(row['Degree']):
                    deg_val = str(row['Degree']).strip().lower()
                    if deg_val in ['0', 'mawdu', 'mawdoo', 'fabricated', 'unauthentic']:
                        lbl = 0
                    elif deg_val in ['1', 'sahih', 'authentic']:
                        lbl = 1
                elif 'Authenticity' in df.columns and pd.notna(row['Authenticity']):
                    auth_val = str(row['Authenticity']).strip().lower()
                    if any(k in auth_val for k in ['non-authentic', 'non_authentic', 'fabricated', 'weak', 'mawdu']):
                        lbl = 0

                norm = normalize_arabic(val_str)
                seg = segment_isnad_matn(val_str)
                records.append({
                    "source": "MAHADDAT",
                    "raw_text": val_str,
                    "normalized_text": norm,
                    "isnad": seg["isnad"],
                    "matn": seg["matn"],
                    "label": lbl
                })
        except Exception:
            continue

    df_mah = pd.DataFrame(records)
    print(f"Loaded {len(df_mah)} records from MAHADDAT.")
    return df_mah

def build_benchmark_dataset(base_dir: str, output_dir: str):
    """Combines all datasets, deduplicates, balances, and exports train/val/test splits."""
    os.makedirs(output_dir, exist_ok=True)
    df_lk = load_lk_hadith_corpus(base_dir)
    df_mah = load_mahaddat(base_dir)

    df_combined = pd.concat([df_lk, df_mah], ignore_index=True)
    print(f"Total raw records combined: {len(df_combined)}")

    # Deduplicate by normalized text
    df_combined.drop_duplicates(subset=["normalized_text"], inplace=True)
    print(f"Total records after deduplication: {len(df_combined)}")

    label_counts = df_combined["label"].value_counts()
    print("Label breakdown (0=Fabricated/Mawdu, 1=Sahih/Authentic):")
    print(label_counts.to_dict())

    # Create balanced dataset by downsampling authentic class or preserving natural distribution
    # Let's extract all Mawdu (label 0) and balance with Sahih (label 1)
    df_0 = df_combined[df_combined["label"] == 0]
    df_1 = df_combined[df_combined["label"] == 1]

    # Keep all label 0 samples, and sample equal number of label 1 samples + 2x for realistic distribution
    target_sahih_count = min(len(df_1), len(df_0) * 3)
    df_1_sampled = df_1.sample(n=target_sahih_count, random_state=42)

    df_balanced = pd.concat([df_0, df_1_sampled], ignore_index=True).sample(frac=1.0, random_state=42)
    print("Balanced Dataset Label Breakdown:", df_balanced["label"].value_counts().to_dict())

    # Train / Val / Test Split (80 / 10 / 10)
    train_df, test_df = train_test_split(df_balanced, test_size=0.20, random_state=42, stratify=df_balanced["label"])
    val_df, test_df = train_test_split(test_df, test_size=0.50, random_state=42, stratify=test_df["label"])

    train_path = os.path.join(output_dir, "processed_train.csv")
    val_path = os.path.join(output_dir, "processed_val.csv")
    test_path = os.path.join(output_dir, "processed_test.csv")

    train_df.to_csv(train_path, index=False, encoding='utf-8-sig')
    val_df.to_csv(val_path, index=False, encoding='utf-8-sig')
    test_df.to_csv(test_path, index=False, encoding='utf-8-sig')

    print(f"Exported Benchmark Datasets:")
    print(f" - Train: {len(train_df)} samples -> {train_path}")
    print(f" - Val:   {len(val_df)} samples -> {val_path}")
    print(f" - Test:  {len(test_df)} samples -> {test_path}")

if __name__ == "__main__":
    base_data = r"d:\research paper\data"
    out_data = r"d:\research paper\data\processed"
    build_benchmark_dataset(base_data, out_data)
