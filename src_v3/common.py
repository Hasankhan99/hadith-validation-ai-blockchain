"""Shared paths and Arabic normalisation for the post-freeze sanad/context extension (src_v3)."""
import os, re, glob
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
CACHE = os.path.join(ROOT, "experiments", "v3_cache")
os.makedirs(CACHE, exist_ok=True)
TASKS = {"binary": "processed_v2", "ternary": "processed_3class"}

DIAC = re.compile(r"[ؐ-ًؚ-ٰٟۖ-ۭـ]")
NONLETTER = re.compile(r"[^ء-ي ]")


def norm(s):
    """Diacritic-free, alef/ya/ta-marbuta-unified Arabic with single spaces."""
    s = DIAC.sub("", str(s))
    s = re.sub("[إأآٱا]", "ا", s).replace("ى", "ي").replace("ة", "ه")
    return re.sub(r"\s+", " ", s).strip()


def key(s):
    """Letters-only join key used to align split rows with their source records."""
    return NONLETTER.sub("", norm(s)).replace(" ", "")


def load_split(task, split):
    df = pd.read_csv(os.path.join(DATA, TASKS[task], f"processed_{split}.csv"))
    df["normalized_text"] = df.normalized_text.fillna("").astype(str)
    return df


def lk_corpus():
    """All LK-Hadith rows with book, grade class, Arabic isnad and matn."""
    rows = []
    for f in glob.glob(os.path.join(DATA, "LK-Hadith-Corpus", "*", "*.csv")):
        d = pd.read_csv(f, usecols=["Arabic_Isnad", "Arabic_Matn", "English_Grade"])
        d["book"] = os.path.basename(os.path.dirname(f))
        rows.append(d)
    d = pd.concat(rows, ignore_index=True)
    g = d.English_Grade.astype(str).str.lower()
    d["grade"] = "other"
    d.loc[g.str.contains("sahih"), "grade"] = "sahih"
    d.loc[g.str.contains("hasan") & (d.grade == "other"), "grade"] = "hasan"
    d.loc[g.str.contains("da'if|daif|weak", regex=True) & (d.grade == "other"), "grade"] = "daif"
    d.loc[g.str.contains("maudu|mawdu|fabricat", regex=True), "grade"] = "mawdu"
    d.loc[d.book.isin(["Bukhari", "Muslim"]), "grade"] = "sahih"
    return d
