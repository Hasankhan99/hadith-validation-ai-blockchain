# Freeze Protocol (written 2026-10-25, before any test score exists)

Test splits (`data/processed_v2/processed_test.csv`, `data/processed_3class/processed_test.csv`) have not been scored by any experiment of the thesis. Only the legacy R0 audit read the old v1 test split to reproduce the defense paper.

## What is frozen
- Data: v2 binary and three-class splits (SHA-256 digests in `experiments/FREEZE_MANIFEST.json`).
- Members: LR(word, balanced), RF(word, balanced_subsample), calibrated LinearSVC(char 2-5, balanced), ComplementNB(word); seeds 0-4.
- Consensus: uniform soft-probability average with prior correction; alpha from R6 (training out-of-fold): binary 0.0, three-class 0.25.
- Abstention: tau by the R7 rule (largest tau with mean validation coverage >= 0.90): binary 0.75, three-class 0.45 for the classical consensus. For a consensus that includes transformers, tau is re-derived by the same rule on validation (stage `val`).
- Transformers: AraBERT v02 and CAMeLBERT-CA, max_len 256, 4 epochs, lr 2e-5, batch 32, early stopping 2 on validation, seeds 0-4 (`colab/train_transformer_v2.py`). Nothing is retuned after results arrive.
- Candidates scored on test (all of them, once): M-LR, M-RF, M-SVM, M-CNB, M-B6, each transformer, CONS-classical, CONS-all (if transformers are present).
- Headline system: the candidate with the highest mean validation primary metric (binary: Mawdu F1; three-class: macro-F1). The headline is named in the manifest before the test is scored. All other candidates are reported next to it, whatever the outcome.

## Procedure
1. Transformer outputs placed in `colab/outputs/`.
2. `python experiments/final_test.py --stage val` writes the manifest.
3. `python experiments/final_test.py --stage test --confirm-freeze` verifies the manifest hashes, creates `final_test.lock`, scores the test once.
4. After step 3 only bug fixes are allowed. A changed script or data file makes the stage refuse to run.

## Deviations recorded so far
- R7 grid extended from 0.50-0.90 to 0.35-0.90 because the three-class task had no admissible tau on the first grid. The binary selection (0.75) is unchanged by this.
- The uncalibrated char-SVM (B6) was added as a standalone candidate after the first manifest, because it is the best single model of R1/R3 and would otherwise be absent from the comparison.
- Binary headline on current validation evidence: classical consensus (0.9006 ± 0.0043) vs calibrated SVM (0.8991) differ by less than one standard deviation; the headline rule still selects the higher mean, and the difference is reported as within noise.
- 2026-10-30 (written before any test score exists): the transformer runs were never completed (no GPU was available locally, and no Colab outputs arrived by the submission deadline). The freeze therefore proceeds with the classical candidates only (M-LR, M-RF, M-SVM, M-CNB, M-B6, CONS-classical). CONS-all is not evaluated. Transformer members are reported as planned-but-not-run in the thesis (limitation and future work). If transformer outputs appear after this date, they may only be reported as a labelled post-freeze supplement; they cannot change the headline system.
- 2026-10-31 (written before any transformer has been trained): the Colab runs are being attempted as the post-freeze supplement allowed above. The supplement is defined now, before outputs exist:
  - Models, seeds and hyperparameters are exactly those frozen above. `colab/train_transformer_v2.py` changed only in output paths, resume, checkpoint cleanup, a row limit for smoke tests, disabled progress bars, and removal of the test labels on Colab.
  - Candidates: each transformer alone and CONS-all (the classical members plus both transformers, frozen alpha, tau by the R7 rule on validation). All are scored on test once, by a separate script. `final_test.py` is not re-run, and its manifest, lock and results stay unchanged.
  - The headline systems and all reported classical numbers stay as they are. Any seed that does not finish is reported as missing, and a candidate without all five seeds gets no test score.
  - The supplement is post hoc: the classical test results were already known when these runs started, and the thesis must say so.
- 2026-11-06 (written before any validation or test score of the extension exists): post-freeze supplement V3 adds two optional, low-weight branches to the frozen consensus (code in `src_v3/`, runner `experiments/supplement_v3.py`):
  - Sanad branch: real isnads are joined to split rows by letters-only matn key (MAHADDAT.csv, NAH_plus.csv, LK). Narrators are linked to AR-Sanad 280K records by naming match plus a teacher-student-graph Viterbi; chain features use only Ibn Hajar grade levels, link breaks and death-date gaps. A class-balanced logistic regression on train rows with features gives the branch distribution. The branch is silent when an item has no isnad or fewer than two linked narrators.
  - Context branch: QulBERT similarity between the matn and a Sahih/Hasan LK library with self-exclusion (same key or cosine >= 0.995); a class-balanced logistic regression on a stratified train subset (all minority rows, 3,000 majority rows, seed 0) gives the branch distribution.
  - Fusion: q = (1 - ws*gs - wc) p_cons + ws*gs p_sanad + wc p_context with ws, wc in {0, 0.1, 0.2, 0.3}. Selection by mean validation primary over the five seeds, ties to the smaller total weight; tau by the R7 rule (largest tau with validation coverage >= 0.90). The frozen CONS-classical validation score must be reproduced exactly before selection.
  - Test is scored once, by `--stage test`, with its own lock (`experiments/supplement_v3.lock`), and with the frozen test distributions from `final_test_probs_{task}.npz`. Results on all rows and on isnad rows are reported whether or not they improve on the frozen system. The headline systems do not change.
  - Known asymmetry declared in advance: an isnad is recovered for every Sahih row but only about 52% of Mawdu rows, so isnad presence is never used as a feature and results are also reported on isnad rows only.
