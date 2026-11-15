# src_v3 Changelog

`src/` and `src_v2/` are untouched. `src_v3/` is a post-freeze extension (FREEZE_PROTOCOL, 2026-11-06 entry); it does not change any frozen number.

## 2026-11-06
- **common.py (new).** Paths, Arabic normalisation, letters-only join key, LK corpus loader with grade classes.
- **align_isnad.py (new).** Recovers the real isnad of each split row from MAHADDAT.csv, NAH_plus.csv and LK by exact matn key. Coverage: 100% Sahih, about 52% Mawdu, about 92% Da'if.
- **sanad_branch.py (new).** Narrator linking to AR-Sanad 280K (naming match + teacher-student graph Viterbi), Ibn Hajar grade levels, chain features, weakest-link rule, and linker evaluation on the AR-Sanad real test sanads.
- **build_sanad_features.py (new).** Chain features for every split row.
- **context_branch.py (new).** QulBERT corroboration features against a Sahih/Hasan LK library with self-exclusion.
- **experiments/supplement_v3.py (new).** Validation-only selection of branch weights and tau; single locked test run.
