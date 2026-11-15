# Digital Hadith Validation Using AI and Blockchain

Code, data splits, experiment logs and frozen results for the MS (Software Engineering) thesis
**"Digital Hadith Validation Using Artificial Intelligence and Blockchain Technology: A Decentralized Ensemble
Consensus Architecture with Immutable Audit Trail"** (Muhammad Hasan Khan, FAST-NUCES Karachi).

The system classifies Arabic hadith texts (binary: Mawdu' / Sahih; three-class: Mawdu' / Da'if / Sahih) with an
ensemble of classical text classifiers combined by a soft-probability consensus with abstention. Every verdict, with
the SHA-256 digests of the models that produced it, is written to a Hyperledger Fabric chaincode that rejects
tampered, missing or unregistered models.

---

## Quick start: check the results in under a minute

```bash
git clone https://github.com/Hasankhan99/hadith-validation-ai-blockchain.git
cd hadith-validation-ai-blockchain
python -m venv .venv
# Windows: .venv\Scripts\activate      Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python reproduce/verify_results.py
```

Expected last line: `RESULT: ALL CHECKS PASSED`. The script:

1. Checks the SHA-256 digests of the train/val/test splits and the frozen scripts against
   `experiments/FREEZE_MANIFEST.json` (the manifest written **before** the test set was scored).
2. Recomputes every test metric from the saved frozen test distributions
   (`experiments/final_test_probs_{binary,ternary}.npz`) and compares it with `experiments/final_test_results.csv`.

To retrain every model from scratch and re-score the test set (CPU only, roughly 10 to 30 minutes):

```bash
python reproduce/verify_results.py --refit
```

This writes to `reproduce/out/` only. It never touches the frozen lock (`experiments/final_test.lock`), the manifest,
or the published results.

Tested with Python 3.12 on Windows 10. No GPU is needed for anything in the quick start.

---

## Frozen test results

Test split scored **once** on 2026-10-30 (see `thesis/FREEZE_PROTOCOL.md`). Values are mean ± std over seeds 0 to 4.
The headline system for each task was chosen on validation and named in the manifest before the test was opened.

**Binary task (Mawdu' vs Sahih, n = 2,601).** Primary metric: F1 of the Mawdu' class.

| System | Mawdu' F1 | Macro-F1 | Accuracy |
|---|---|---|---|
| **CONS-classical (headline, chosen on validation)** | 0.8605 ± 0.0010 | 0.9236 ± 0.0005 | 0.9755 ± 0.0002 |
| Calibrated char-SVM (M-SVM) | 0.8767 | 0.9327 | 0.9792 |
| Char-SVM, uncalibrated (M-B6) | 0.8761 | 0.9319 | 0.9777 |
| Random Forest (M-RF) | 0.8170 ± 0.0049 | 0.9000 ± 0.0027 | 0.9689 ± 0.0008 |
| Logistic Regression (M-LR) | 0.7684 | 0.8707 | 0.9516 |
| Complement NB (M-CNB) | 0.7160 | 0.8429 | 0.9454 |
| CONS-classical with abstention (tau = 0.75) | 0.9403 ± 0.0028 on 92.66% coverage | | |

**Three-class task (Mawdu' / Da'if / Sahih, n = 2,982).** Primary metric: macro-F1.

| System | Macro-F1 | Accuracy |
|---|---|---|
| **M-B6 char-SVM (headline, chosen on validation)** | 0.6971 | 0.8635 |
| CONS-classical | 0.6846 ± 0.0028 | 0.8555 ± 0.0015 |
| M-SVM | 0.6591 | 0.8669 |
| M-LR | 0.6586 | 0.7944 |
| M-RF | 0.5974 ± 0.0055 | 0.8506 ± 0.0018 |
| M-CNB | 0.5772 | 0.7897 |
| CONS-classical with abstention (tau = 0.45) | 0.7156 ± 0.0029 on 93.80% coverage | |

Results are reported as they came out. On the binary test set, the consensus headline is **below** the single
calibrated SVM: the paired bootstrap difference is −0.0166, 95% CI [−0.0422, 0.0069]
(`experiments/final_test_analysis.json`). The thesis discusses this as a negative result. Its value is
selective prediction: with abstention, the covered Mawdu' F1 rises to 0.94.

**Blockchain (Hyperledger Fabric 2.5.16 test network, 2 orgs, Raft, synthetic payloads):**
9 of 9 security tests rejected or accepted as expected (tampered digest, missing model, unregistered model, and so on);
concurrent submit throughput 43.4 TPS at concurrency 10 and 57.2 TPS at concurrency 50 (0 failed); query latency
p50 8.1 ms. Raw numbers are in `experiments/b1_fabric/b1_fabric_results.json`.

Every number in the thesis traces to a row in `thesis/RESULTS_LOG.csv` (run id, change, config, seed, split, metric, value, artifact).

---

## Repository layout

| Path | Contents |
|---|---|
| `src/` | Original v1 pipeline, kept unchanged for the audit (contains the label inversion found on Day 1; do not use for results). |
| `src_v2/` | Corrected pipeline used for all thesis results: loaders, preprocessing, members, `consensus_soft.py` (frozen consensus). See `src_v2/CHANGELOG.md`. |
| `src_v3/` | Pre-registered post-freeze sanad/context supplement (optional branches). See `src_v3/CHANGELOG.md`. |
| `experiments/` | One script per run (A0 audit, R0 to R7, final test, analysis), their CSV/JSON outputs and logs. |
| `experiments/FREEZE_MANIFEST.json`, `final_test.lock` | Frozen selection and the one-time test lock. |
| `experiments/b1_fabric/` | Fabric benchmark script and results. |
| `blockchain_v2/chaincode/` | Go chaincode used in the experiments (dependencies vendored). `blockchain/` is the v1 prototype. |
| `colab/` | Transformer fine-tuning script and notebook (AraBERT, CAMeLBERT); planned, not part of the frozen results. |
| `data/processed_v2/`, `data/processed_3class/` | The exact binary and three-class splits used (digests in the manifest). |
| `data/processed/` | v1 splits, only for the R0 reproduction of the preliminary paper. |
| `thesis/RESULTS_LOG.csv`, `thesis/FREEZE_PROTOCOL.md`, `thesis/figures/` | Results log, evaluation protocol, figures. |
| `reproduce/` | `verify_results.py` (result check) and `fetch_raw_data.py` (raw datasets at pinned commits). |

The thesis manuscript itself is not in this repository.

---

## Full pipeline (optional)

The processed splits are already committed, so none of this is needed to check the results.

```bash
# 1. Raw datasets at the commits used (MAHADDAT, LK-Hadith-Corpus, AR-Sanad 280K)
python reproduce/fetch_raw_data.py

# 2. Validation experiments (each writes its own CSV in experiments/)
python experiments/r1_baselines_val.py              # binary baselines, 5 seeds
python experiments/r3_baselines_3class_val.py       # three-class baselines
python experiments/r4_consensus_ablation_val.py     # consensus ablation + error analysis
python experiments/r5_view_ablation_val.py          # input-view / diacritics ablation
python experiments/r6_balanced_consensus_val.py     # prior-correction alpha (training OOF)
python experiments/r7_tau_selection_val.py          # abstention threshold

# 3. Bootstrap CIs, confusion matrices, PR and coverage curves from the saved test distributions
python experiments/final_test_analysis.py
```

Notes:

- `src_v2/dataset_loader.py` and `src_v2/dataset_loader_3class.py` rebuild the splits from the raw data and
  **overwrite** `data/processed_v2` and `data/processed_3class`. Run `python reproduce/verify_results.py` afterwards:
  the digest check tells you whether the rebuilt splits are byte-identical to the frozen ones.
- `experiments/final_test.py --stage test` refuses to run because the lock exists. That is by design: the test set
  was scored once. Use `reproduce/verify_results.py --refit` to re-score.
- Do not run `experiments/final_test.py --stage val`. It overwrites the frozen manifest.
- The R0 script reproduces the preliminary paper on the v1 splits, label inversion included, for the audit only.

### Chaincode

```bash
cd blockchain_v2/chaincode
go vet ./... && go build ./...       # Go 1.22; dependencies are vendored
```

Deployment used the Hyperledger Fabric 2.5.16 `test-network` (channel `hadithchannel`, chaincode name `hadith`).
The benchmark is run with
`node experiments/b1_fabric/b1_fabric_benchmark.js <fabric-samples/test-network> <out_dir> [N_SEQ] [N_CONC]`
(needs `@grpc/grpc-js` and `@hyperledger/fabric-gateway`).

---

## Data sources

| Dataset | Source | Used for |
|---|---|---|
| MAHADDAT (+ NAH Plus) | https://github.com/kamelgaanoun/mhdetection | Binary benchmark (official train/dev/test files), Da'if from NAH Plus |
| LK-Hadith-Corpus | https://github.com/ShathaTm/LK-Hadith-Corpus | Da'if class (three-class task) |
| AR-Sanad 280K | https://github.com/somaia02/Narrator-Disambiguation | Narrator records for the V3 sanad supplement |

All credit for the datasets belongs to their authors; check each repository for its license and cite the
original papers when you use them.

## Evaluation protocol in one paragraph

All model and hyperparameter selection used the validation split only. The candidates, the consensus (uniform
soft-probability average with prior correction alpha = 0 binary / 0.25 three-class), the abstention thresholds and
the headline rule were frozen in `experiments/FREEZE_MANIFEST.json` with SHA-256 digests of every input. The test split
was then scored once, all candidates together, and a lock file prevents a second run. Transformer members were
planned but not trained (no GPU before the freeze); this is stated as a limitation.

## Citation

```
Muhammad Hasan Khan. Digital Hadith Validation Using Artificial Intelligence and Blockchain Technology:
A Decentralized Ensemble Consensus Architecture with Immutable Audit Trail. MS thesis,
FAST National University of Computer and Emerging Sciences, Karachi, 2026.
```
