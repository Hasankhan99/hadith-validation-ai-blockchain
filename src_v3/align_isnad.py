"""Attach the real isnad of each split row by matching its matn to the source corpora.

The frozen split files hold only the matn (MAHADDAT ships matn only). The isnad is recovered from
MAHADDAT.csv, NAH_plus.csv and the LK-Hadith corpus by an exact letters-only key. Rows without a
match keep an empty isnad, so the sanad branch stays optional by construction.
Output: experiments/v3_cache/isnad_{task}_{split}.csv (row-aligned with the split file).
"""
import os
import pandas as pd
from common import DATA, CACHE, key, load_split, lk_corpus


def source_table():
    parts = []
    for f in ("MAHADDAT.csv", "NAH_plus.csv"):
        d = pd.read_csv(os.path.join(DATA, "MAHADDAT", f), usecols=["Isnad", "Matan"])
        parts.append(pd.DataFrame({"k": d.Matan.map(key), "isnad": d.Isnad.astype(str), "src": f}))
    lk = lk_corpus()
    parts.append(pd.DataFrame({"k": lk.Arabic_Matn.map(key), "isnad": lk.Arabic_Isnad.astype(str), "src": "LK"}))
    t = pd.concat(parts, ignore_index=True)
    t = t[(t.k.str.len() > 10) & (t.isnad.str.len() > 5) & (t.isnad != "nan")]
    return t.drop_duplicates("k")  # first source wins; one isnad per matn key


def main():
    t = source_table().set_index("k")
    for task in ("binary", "ternary"):
        for split in ("train", "val", "test"):
            df = load_split(task, split)
            k = df.raw_text.map(key)
            out = pd.DataFrame({"isnad": k.map(t.isnad).fillna(""), "isnad_src": k.map(t.src).fillna("")})
            out.to_csv(os.path.join(CACHE, f"isnad_{task}_{split}.csv"), index=False)
            cov = (out.isnad != "").groupby(df.label).mean().round(3).to_dict()
            print(task, split, len(df), "coverage by label:", cov)


if __name__ == "__main__":
    main()
