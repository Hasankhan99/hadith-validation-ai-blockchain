"""Context branch: corroboration of a matn by semantically similar authentic narrations.

QulBERT (FDSRashid/QulBERT; CAMeLBERT-CA fine-tuned on matns grouped by report) embeds the query
matn and a reference library of Sahih/Hasan matns from the LK-Hadith corpus. The branch returns
the maximum and the mean top-5 cosine similarity, and the top matches as evidence.
Self-exclusion: a library entry is skipped when its letters-only key equals the query's key or
its cosine is >= 0.995, so an item never corroborates itself. This also removes verbatim
parallels in other books, which makes the signal conservative.
Training rows are subsampled (all minority-class rows, 3,000 majority rows per task, seed 0) and
used only to fit the branch's score model; validation and test rows are encoded in full.
"""
import os
import numpy as np
import pandas as pd
from common import CACHE, key, load_split, lk_corpus

MODEL = "FDSRashid/QulBERT"
SELF_COS = 0.995
TOPK = 5


def encoder():
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(MODEL, device="cpu")
    m.max_seq_length = 128
    return m


def library():
    lk = lk_corpus()
    lk = lk[lk.grade.isin(["sahih", "hasan"])].copy()
    lk["k"] = lk.Arabic_Matn.map(key)
    lk = lk[lk.k.str.len() > 10].drop_duplicates("k").reset_index(drop=True)
    return lk


def train_subset(df, seed=0):
    maj = df.label.max()
    keep = df[df.label != maj].index.tolist()
    keep += df[df.label == maj].sample(min(3000, (df.label == maj).sum()), random_state=seed).index.tolist()
    return sorted(keep)


def features(q_emb, q_keys, lib_emb, lib_keys):
    sims = q_emb @ lib_emb.T
    out = np.zeros((len(q_emb), 2), dtype=np.float32)
    lk = np.asarray(lib_keys)
    for i in range(len(q_emb)):
        s = sims[i].copy()
        s[(lk == q_keys[i]) | (s >= SELF_COS)] = -1
        top = np.partition(s, -TOPK)[-TOPK:]
        out[i] = (top.max(), top.mean())
    return out


def main():
    m = encoder()
    lib = library()
    lp = os.path.join(CACHE, "ctx_library_emb.npy")
    if not os.path.exists(lp):
        # encoded in checkpointed chunks so an interrupted run resumes where it stopped
        texts, parts, CH = lib.Arabic_Matn.astype(str).tolist(), [], 2000
        for c in range(0, len(texts), CH):
            cp = os.path.join(CACHE, f"ctx_library_chunk{c // CH:03d}.npy")
            if not os.path.exists(cp):
                np.save(cp, m.encode(texts[c:c + CH], batch_size=64, normalize_embeddings=True, show_progress_bar=False))
            parts.append(np.load(cp))
            print(f"library {min(c + CH, len(texts))}/{len(texts)}", flush=True)
        np.save(lp, np.concatenate(parts))
        lib[["book", "grade", "k"]].to_csv(os.path.join(CACHE, "ctx_library_meta.csv"), index=False)
    L = np.load(lp)
    print("library", L.shape, flush=True)
    for task in ("binary", "ternary"):
        for split in ("train", "val", "test"):
            if os.path.exists(os.path.join(CACHE, f"context_{task}_{split}.csv")):
                continue
            df = load_split(task, split)
            rows = train_subset(df) if split == "train" else list(range(len(df)))
            q = df.raw_text.astype(str).iloc[rows]
            E = m.encode(q.tolist(), batch_size=64, normalize_embeddings=True, show_progress_bar=False)
            F = features(E, q.map(key).tolist(), L, lib.k.tolist())
            out = pd.DataFrame(np.nan, index=range(len(df)), columns=["ctx_max", "ctx_top5"])
            out.iloc[rows] = F
            out.to_csv(os.path.join(CACHE, f"context_{task}_{split}.csv"), index=False)
            print(task, split, "encoded", len(rows), flush=True)


if __name__ == "__main__":
    main()
