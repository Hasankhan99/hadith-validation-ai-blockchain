"""Computes chain features for every split row; rows without an isnad or with <2 linked narrators get NaN.
Output: experiments/v3_cache/sanad_{task}_{split}.csv (row-aligned with the split file)."""
import os
import numpy as np
import pandas as pd
from common import CACHE
from sanad_branch import Narrators, chain_features, FEATURES

nar = Narrators()
memo = {}
for task in ("binary", "ternary"):
    for split in ("train", "val", "test"):
        isn = pd.read_csv(os.path.join(CACHE, f"isnad_{task}_{split}.csv")).isnad.fillna("")
        rows = []
        for s in isn:
            if s not in memo:
                memo[s] = chain_features(nar, s) if s else None
            f = memo[s]
            rows.append(f if f else {k: np.nan for k in FEATURES})
        out = pd.DataFrame(rows, columns=FEATURES)
        out.to_csv(os.path.join(CACHE, f"sanad_{task}_{split}.csv"), index=False)
        print(task, split, "rows with chain features:", round(out.worst_level.notna().mean(), 3))
