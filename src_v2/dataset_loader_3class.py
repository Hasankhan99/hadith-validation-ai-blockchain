"""
Three-class benchmark (Sahih / Da'if / Mawdu) built on top of the v2 binary benchmark.

  - Sahih and Mawdu rows: exactly the v2 binary splits (official MAHADDAT split, cleaned).
  - Da'if rows (new): LK-Hadith-Corpus narrations graded "Daif - Weak" (Arabic_Matn column)
    and NAH Plus narrations with Degree == "Daif" (Matan column). Other grades (Hasan,
    "Da'if in chain", Munkar, ...) are excluded as ambiguous for a three-way scheme.
  - Da'if rows are deduplicated, removed if their normalised text occurs anywhere in the
    v2 benchmark, and split 80/10/10 (seed 42) within each source.
Labels (ordinal): 0 = Mawdu, 1 = Da'if, 2 = Sahih
"""

import os
import glob
import pandas as pd
from sklearn.model_selection import train_test_split
from preprocessing import normalize_arabic
from sanad_segmenter import segment_isnad_matn

MIN_CHARS = 15
LABELS = {0: "Mawdu", 1: "Daif", 2: "Sahih"}

def _frame(texts, source):
    df = pd.DataFrame({"raw_text": pd.Series(texts, dtype=str).str.strip()})
    df = df[df.raw_text.str.len() >= MIN_CHARS].copy()
    df["normalized_text"] = df.raw_text.map(normalize_arabic)
    seg = df.raw_text.map(segment_isnad_matn)
    df["isnad"], df["matn"] = seg.map(lambda s: s["isnad"]), seg.map(lambda s: s["matn"])
    df["source"], df["label"] = source, 1
    return df

def load_daif(data_dir: str) -> pd.DataFrame:
    lk = pd.concat([pd.read_csv(f, encoding="utf-8-sig", on_bad_lines="skip")
                    for f in glob.glob(os.path.join(data_dir, "LK-Hadith-Corpus", "*", "*.csv"))])
    lk = lk[lk.English_Grade.astype(str).str.strip() == "Daif - Weak"]
    nah = pd.read_csv(os.path.join(data_dir, "MAHADDAT", "NAH_plus.csv"), encoding="utf-8-sig")
    nah = nah[nah.Degree.astype(str).str.strip() == "Daif"]
    return pd.concat([_frame(lk.Arabic_Matn.dropna(), "LK-Daif"), _frame(nah.Matan.dropna(), "NAH-Daif")], ignore_index=True)

def build_3class(data_dir: str, v2_dir: str, out_dir: str) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    v2 = {s: pd.read_csv(os.path.join(v2_dir, f"processed_{s}.csv")) for s in ["train", "val", "test"]}
    seen = set(pd.concat(v2.values()).normalized_text)
    daif = load_daif(data_dir)
    n_raw = len(daif)
    daif = daif.drop_duplicates(subset=["normalized_text"])
    n_dup = n_raw - len(daif)
    clash = daif.normalized_text.isin(seen)
    daif = daif[~clash]
    parts = {"train": [], "val": [], "test": []}
    for _, g in daif.groupby("source"):
        tr, rest = train_test_split(g, test_size=0.2, random_state=42)
        va, te = train_test_split(rest, test_size=0.5, random_state=42)
        parts["train"].append(tr); parts["val"].append(va); parts["test"].append(te)
    stats = {"daif_raw": n_raw, "daif_dup_removed": n_dup, "daif_overlap_with_v2_removed": int(clash.sum()),
             "daif_by_source": daif.source.value_counts().to_dict()}
    for s, df in v2.items():
        df = df.copy()
        df["label"] = df["label"].map({0: 0, 1: 2})  # binary 0 = Mawdu, 1 = Sahih
        out = pd.concat([df] + parts[s], ignore_index=True).sample(frac=1.0, random_state=42)
        out.to_csv(os.path.join(out_dir, f"processed_{s}.csv"), index=False, encoding="utf-8-sig")
        stats[s] = {LABELS[k]: int(v) for k, v in out.label.value_counts().sort_index().items()} | {"total": len(out)}
        print(s, stats[s])
    print({k: v for k, v in stats.items() if k.startswith("daif")})
    return stats

if __name__ == "__main__":
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    build_3class(os.path.join(root, "data"), os.path.join(root, "data", "processed_v2"), os.path.join(root, "data", "processed_3class"))
