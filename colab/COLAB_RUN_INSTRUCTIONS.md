# Colab transformer runs (post-freeze supplement)

The test set was scored on 2026-10-30 without transformers, so the headline systems are fixed. These runs are reported as a labelled post-freeze supplement and cannot change the headline (see `thesis/FREEZE_PROTOCOL.md`).

**Do not run `experiments/final_test.py --stage val` again.** It overwrites the frozen manifest and validation results. The returned outputs are scored by a separate supplement script.

## 1. Set up (5 minutes)
1. In Google Drive, create the folder `hadith_v2` directly in My Drive.
2. Upload `colab/hadith_v2_colab_pack.zip` into it. Do not unzip it.
3. Open https://colab.research.google.com > File > Upload notebook > `colab/hadith_transformers_v2.ipynb`.
4. Runtime > Change runtime type > T4 GPU > Save.

## 2. Run
1. Runtime > Run all, and approve the Drive prompt.
2. Check section 3 of the notebook prints `SMOKE TEST PASSED`. If a cell stops with an error, send the last 30 lines of its output.
3. Keep the tab open until section 5 finishes. Estimated time for all 20 trainings on a T4 is 2 to 3 hours; the first finished seed gives the real pace through `train_sec` in its printed result.

Hyperparameters are fixed: max_len 256, 4 epochs, lr 2e-5, batch 32, early stopping patience 2 on validation. Do not change them.

## 3. If the session drops or GPU quota runs out
Reconnect and use Runtime > Run all again. Finished seeds are skipped, and every result is already on Drive. If the quota is exhausted, wait until it resets or use a second attempt later, then Run all with the same Drive folder.

## 4. Send back
- `hadith_v2_outputs.zip` from My Drive/hadith_v2/ (validation results, sealed test probabilities, logs, library versions, GPU details).
- A copy of the table printed at the end of section 5, and the `MISSING:` line.

Put the zip in `colab/` on this machine. Do not open or score anything in `sealed_test_preds/`.
