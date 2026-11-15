# src_v2 Changelog

The original code in `src/` is untouched. Every change below is reflected in the thesis Methodology and Implementation chapters.

## 2026-10-24 (Day 1)
- **dataset_loader.py (rewritten)**
  - Uses the official MAHADDAT train, dev and test files instead of pooling and re-splitting.
  - Label fix: `Degree = 1` means Mawdu, so it maps to label 0; `Degree = 0` means Sahih and maps to label 1. Before the fix, 2,368 of 2,370 known-fabricated texts were labelled Sahih.
  - Removes texts under 15 characters, duplicates within a split, and dev or test texts already present in an earlier split (51 dev, 48 test).
  - Drops the separate LK-Hadith loader. v1 read the `Chapter_Arabic` column (chapter titles), and the MAHADDAT authentic class already is the LK corpus.
  - Output goes to `data/processed_v2/`; v1 data in `data/processed/` is kept.
- No other module changed yet. Planned for Day 2:
  - Chaincode: use `GetTxTimestamp`, check the full registered ensemble, handle unmarshal errors.
  - `main_pipeline.py`: weights from validation, new label semantics, calibrated SVM.
  - Ledger benchmark.

## 2026-10-25 (Day 2)
- **dataset_loader_3class.py (new).** Builds the three-class benchmark (0 = Mawdu, 1 = Da'if, 2 = Sahih) on top of the v2 splits.
  - Da'if sources: LK-Hadith Corpus `English_Grade == "Daif - Weak"` (Arabic_Matn) and NAH Plus `Degree == "Daif"` (Matan).
  - 3,888 raw; 43 duplicates and 46 overlaps with the v2 benchmark removed; 3,799 kept (2,858 LK, 941 NAH).
  - Split 80/10/10 per source (seed 42). Output to `data/processed_3class/`.
- **blockchain_v2/chaincode/hadith_validation.go (new, replaces blockchain/chaincode for the experiments).**
  - Timestamps from `GetTxTimestamp` (deterministic across endorsers).
  - Submission must contain exactly the registered ensemble with matching digests; duplicates, bad labels and confidences are rejected.
  - Model registration restricted to Org1MSP.
  - Three-class labels (0 Mawdu, 1 Daif, 2 Sahih) and consensus status are stored.
  - `go.mod` pins `fabric-contract-api-go/v2 v2.1.0` (v2.2.4 needs Go 1.26); dependencies vendored (`go mod vendor`); `go vet` and `go build` pass on Go 1.22.
  - Deployed on Fabric 2.5.16 (test-network, channel `hadithchannel`, chaincode `hadith`, hand-built package because the host has no Go toolchain).
- **colab/train_transformer_v2.py (new).** Binary and three-class fine-tuning, seeds 0-4, selection on validation, sealed test outputs. Exact commands in `colab/COLAB_RUN_INSTRUCTIONS.md`.
- No change to `src/`; `src_v2/` modules other than the two loaders are unchanged copies.

## 2026-10-25 (Day 3)
- **consensus_soft.py (new).** The frozen consensus: uniform soft-probability average, prior correction with exponent alpha, abstention at tau, ties to Mawdu, and `to_ledger_label` (binary Sahih = 1 must be sent to the chaincode as code 2).
- **train_baselines.py.** Refuses to score the test split unless `--legacy-v1-reproduction` is passed.
- **experiments/final_test.py (new).** Two-stage frozen evaluation (validation manifest, then one locked test run).

## 2026-10-30 (freeze)
- **experiments/final_test.py.** The test stage also saves the frozen test distributions (`final_test_probs_{task}.npz`), so confidence intervals and confusion matrices need no second test pass. The models, candidates, selection rule and metrics are unchanged. This change was made before the manifest was regenerated and before the test was scored.
- **experiments/final_test_analysis.py (new).** Bootstrap confidence intervals, a paired headline-vs-runner-up difference, confusion matrices, a PR curve and a coverage curve, all from the saved distributions (no refit, no selection).
- **Freeze without transformers.** The transformer runs were never executed (no GPU). The freeze proceeds with the classical candidates; recorded in `thesis/FREEZE_PROTOCOL.md`.
- **main_pipeline.py is not integrated** with `consensus_soft.py` or Fabric; stated as a limitation in the thesis.

## 2026-10-31 (post-freeze supplement preparation)
- **colab/train_transformer_v2.py.** No change to models, seeds, hyperparameters or selection.
  - Added `--out_dir` (results on Drive) and `--work_dir` (local checkpoints, deleted after each seed). Previously checkpoints went to the working folder on Drive and would have exceeded the 15 GB free quota.
  - Rerunning now skips finished seeds and merges results. Previously, a rerun with only the missing seeds overwrote the results CSV and lost the earlier rows.
  - Test labels are removed before prediction, so no test metric can be computed on Colab. Previously the labels were still present and `trainer.predict` computed a test metric that was discarded.
  - Added `--limit` for the smoke test and turned off progress bars.
- **colab/hadith_transformers_v2.ipynb and hadith_v2_colab_pack.zip (new).** Run-all notebook with data digest check against the freeze manifest, pinned libraries, smoke test, full run and packaging. Libraries are pinned to the versions tested end to end on 2026-10-31 (transformers 4.51.3, accelerate 1.15.0, huggingface_hub 0.36.2), replacing the planned, untested transformers 4.44.2. `datasets` is left unpinned because 2.21.0 failed dependency resolution; the exact versions used are recorded in the returned `pip_freeze.txt`.
