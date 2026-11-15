"""Sanad branch: narrator linking, reliability lookup and chain features.

Narrators are linked to AR-Sanad 280K narrator records (Mahmoud et al., 2022) by matching each
name mention against all recorded namings, then choosing among ambiguous candidates with a
Viterbi pass that maximises teacher-student edges between neighbours. Each linked narrator gets
an ordinal reliability level from his Ibn Hajar grade (Taqrib al-Tahdhib) as recorded in the
dataset. Features describe only the narrators and their links, never the raw isnad text, its
length or the linking rate, which differ by source corpus and would act as shortcuts.
"""
import ast, os, re, sys
import numpy as np
import pandas as pd
from common import DATA, norm

NAR = os.path.join(DATA, "AR-Sanad-280K", "Narrators data.csv")
SEP = r"(?:حدثنا|حدثني|حدثه|حدثهم|اخبرنا|اخبرني|اخبره|انبانا|انباني|سمعت|سمعنا|عن|قال|قالا|قالوا|يقول|ح)"
PROPHET = re.compile(r"(رسول الله|النبي|صلي الله عليه وسلم|رضي الله عنهما?|رضي الله عنها)")

# Ibn Hajar grade -> ordinal level (0 = Companion, 8 = accused of lying). Checked in order.
LEVELS = [("صحابي", 0), ("صحابيه", 0), ("كذاب", 8), ("متهم", 8), ("وضاع", 8), ("متروك", 7),
          ("منكر الحديث", 7), ("ضعيف", 6), ("مجهول", 6), ("لين", 5), ("مستور", 5), ("مقبول", 4),
          ("يخطئ", 3), ("يهم", 3), ("اوهام", 3), ("تغير", 3), ("صدوق", 2), ("لا باس", 2),
          ("ثقه", 1), ("امام", 1), ("حافظ", 1)]


def level(grade):
    g = norm(grade)
    if g in ("", "-", "nan"):
        return None
    for w, v in LEVELS:
        if w in g:
            return v
    return None


class Narrators:
    def __init__(self, path=NAR):
        d = pd.read_csv(path)
        self.name2ids, self.level, self.death, self.edges = {}, {}, {}, set()
        for r in d.itertuples(index=False):
            i = int(r.id)
            forms = [r.name] + (ast.literal_eval(r.namings) if str(r.namings).startswith("[") else [])
            for f in forms:
                self.name2ids.setdefault(norm(f), set()).add(i)
            self.level[i] = level(r.Ibnhajar_rank)
            m = re.search(r"\d+", str(r.death_year))
            self.death[i] = int(m.group()) if m else None
            if str(r.narrated_from).startswith("["):
                for t in ast.literal_eval(r.narrated_from):
                    self.edges.add((i, int(t)))

    def linked(self, a, b):
        return (a, b) in self.edges or (b, a) in self.edges

    def candidates(self, mention, cap=200):
        return sorted(self.name2ids.get(norm(mention), ()))[:cap]

    def resolve(self, mentions):
        """Returns one id (or None) per mention; ambiguous mentions are resolved by graph Viterbi."""
        cands = [self.candidates(m) for m in mentions]
        idx = [k for k, c in enumerate(cands) if c]
        out = [None] * len(mentions)
        if not idx:
            return out
        best = {c: (0, [c]) for c in cands[idx[0]]}
        for k in idx[1:]:
            cur = {}
            for c in cands[k]:
                sc, path = max(((s + self.linked(c, p[-1]), p) for s, p in best.values()), key=lambda t: t[0])
                cur[c] = (sc, path + [c])
            best = cur
        path = max(best.values(), key=lambda t: t[0])[1]
        for k, i in zip(idx, path):
            # an ambiguous mention counts as linked only when the chosen id is edge-supported
            nb = [path[j] for j in (idx.index(k) - 1, idx.index(k) + 1) if 0 <= j < len(path)]
            if len(cands[k]) == 1 or any(self.linked(i, n) for n in nb):
                out[k] = i
        return out


def mentions(isnad):
    s = PROPHET.sub(" ", norm(isnad))
    parts = [p.strip(" ,.:،؛\"'") for p in re.split(r"\b" + SEP + r"\b", s)]
    return [p for p in parts if len(p) >= 3]


FEATURES = ["worst_level", "mean_level", "n_weak", "n_unknown", "n_trusted", "frac_weak",
            "broken_links", "frac_broken", "date_gap"]


def chain_features(nar, isnad):
    """Feature vector for one isnad, or None when fewer than two narrators are linked."""
    ids = [i for i in nar.resolve(mentions(isnad)) if i is not None]
    if len(ids) < 2:
        return None
    lv = [nar.level[i] for i in ids if nar.level[i] is not None]
    lv = lv or [4]
    broken = sum(not nar.linked(a, b) for a, b in zip(ids, ids[1:]))
    dates = [nar.death[i] for i in ids]
    # a student who died more than 120 years after his teacher cannot plausibly have heard from him
    gap = sum(1 for a, b in zip(dates, dates[1:]) if a and b and a - b > 120)
    return dict(worst_level=max(lv), mean_level=float(np.mean(lv)), n_weak=sum(v >= 6 for v in lv),
                n_unknown=sum(nar.level[i] is None for i in ids) + sum(v in (5, 6) for v in lv),
                n_trusted=sum(v <= 1 for v in lv), frac_weak=sum(v >= 5 for v in lv) / len(lv),
                broken_links=broken, frac_broken=broken / (len(ids) - 1), date_gap=gap)


def rule_score(f):
    """Classical weakest-link rule: 1 = chain judged weak, 0 = sound."""
    return float(f["worst_level"] >= 5 or f["broken_links"] > 0 or f["date_gap"] > 0)


def evaluate_linker(n=3000, seed=0):
    """Narrator-level linking accuracy on AR-Sanad real test sanads (gold segmentation and ids)."""
    base = os.path.join(DATA, "AR-Sanad-280K", "Large dataset")
    S = open(os.path.join(base, "sanads.test"), encoding="utf-8").read().splitlines()
    L = open(os.path.join(base, "labels.test"), encoding="utf-8").read().splitlines()
    rng = np.random.default_rng(seed)
    nar = Narrators()
    tot = linked = correct = 0
    for j in rng.choice(len(S), n, replace=False):
        names = [x.strip() for x in S[j].split("[فاصل]")]
        gold = [int(x) for x in L[j].split(",")]
        if len(names) != len(gold):
            continue
        pred = nar.resolve(names)
        for p, g in zip(pred, gold):
            tot += 1
            linked += p is not None
            correct += p == g
    return dict(mentions=tot, linked_rate=linked / tot, precision=correct / max(linked, 1), recall=correct / tot)


if __name__ == "__main__":
    print(evaluate_linker(int(sys.argv[1]) if len(sys.argv) > 1 else 3000))
