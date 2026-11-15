"""
Transformer fine-tuning on the corrected v2 benchmark (Google Colab, GPU).

Protocol (thesis Section 3.7):
  - Data: data/processed_v2/processed_{train,val,test}.csv  (label 1 = Sahih, 0 = Mawdu)
  - Model selection / early stopping on VALIDATION Mawdu-class F1 only.
  - Seeds 0-4. Validation metrics are printed and saved.
  - Test probabilities are written to sealed_test_preds/ with the test labels removed, so no test metric
    can be computed here. The classical freeze is closed (thesis/FREEZE_PROTOCOL.md): these outputs are
    scored once on the local machine as a labelled post-freeze supplement.

Tasks:
  --task binary   data/processed_v2      labels 0 = Mawdu, 1 = Sahih            selection: Mawdu F1
  --task ternary  data/processed_3class  labels 0 = Mawdu, 1 = Da'if, 2 = Sahih  selection: macro-F1

Run it through colab/hadith_transformers_v2.ipynb (steps in colab/COLAB_RUN_INSTRUCTIONS.md).
--out_dir holds everything that must survive a dropped session; --work_dir holds checkpoints and is cleaned
after each seed. A seed already complete in --out_dir is skipped, so a rerun continues where it stopped.
"""

import os
import json
import time
import shutil
import hashlib
import argparse
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import (AutoTokenizer, AutoModelForSequenceClassification, Trainer,
                          TrainingArguments, DataCollatorWithPadding, EarlyStoppingCallback, set_seed)
from datasets import Dataset

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

NUM_LABELS = 2

def metrics(labels, preds):
    labs = list(range(NUM_LABELS))
    f = f1_score(labels, preds, labels=labs, average=None, zero_division=0)
    out = {
        "mawdu_f1": f[0],
        "mawdu_p": precision_score(labels, preds, labels=[0], average="micro", zero_division=0),
        "mawdu_r": recall_score(labels, preds, labels=[0], average="micro", zero_division=0),
        "macro_f1": f.mean(),
        "accuracy": accuracy_score(labels, preds),
    }
    if NUM_LABELS == 3:
        out["daif_f1"], out["sahih_f1"] = f[1], f[2]
    return out

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    return metrics(labels, np.argmax(logits, axis=1))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["binary", "ternary"], default="binary")
    ap.add_argument("--model", default="aubmindlab/bert-base-arabertv02")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--data_dir", default=".")
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--out_dir", default=".")
    ap.add_argument("--work_dir", default="work")
    ap.add_argument("--limit", type=int, default=0, help="smoke test only: keep the first N rows of each split")
    args = ap.parse_args()
    global NUM_LABELS
    NUM_LABELS = 2 if args.task == "binary" else 3
    select = "mawdu_f1" if args.task == "binary" else "macro_f1"

    print("CUDA:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
    rd = lambda s: pd.read_csv(os.path.join(args.data_dir, f"processed_{s}.csv"))
    tr, va, te = rd("train"), rd("val"), rd("test")
    if args.limit:
        tr, va, te = tr.head(args.limit), va.head(args.limit), te.head(args.limit)
    tok = AutoTokenizer.from_pretrained(args.model)
    enc = lambda df: Dataset.from_dict({"text": df["normalized_text"].fillna("").astype(str).tolist(),
                                        "label": df["label"].astype(int).tolist()}).map(
        lambda b: tok(b["text"], truncation=True, max_length=args.max_len), batched=True)
    ds_tr, ds_va, ds_te = enc(tr), enc(va), enc(te).remove_columns("label")
    lens = [len(x) for x in ds_tr["input_ids"]]
    print(f"train token length: median={np.median(lens):.0f} p95={np.percentile(lens,95):.0f} "
          f"truncated={np.mean(np.array(lens) >= args.max_len):.3%}")

    short = f'{args.task}_{args.model.split("/")[-1]}'
    os.makedirs(os.path.join(args.out_dir, "sealed_test_preds"), exist_ok=True)
    os.makedirs(args.work_dir, exist_ok=True)
    res_csv = os.path.join(args.out_dir, f"results_transformer_v2_{short}.csv")
    rows = pd.read_csv(res_csv).to_dict("records") if os.path.exists(res_csv) else []
    for seed in args.seeds:
        val_npy = os.path.join(args.out_dir, f"val_probs_{short}_s{seed}.npy")
        test_npy = os.path.join(args.out_dir, "sealed_test_preds", f"test_probs_{short}_s{seed}.npy")
        if any(r["seed"] == seed for r in rows) and os.path.exists(val_npy) and os.path.exists(test_npy):
            print(f"seed {seed}: already complete, skipped", flush=True)
            continue
        rows = [r for r in rows if r["seed"] != seed]
        set_seed(seed)
        model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=NUM_LABELS, ignore_mismatched_sizes=True)
        out = os.path.join(args.work_dir, f"ckpt_{short}_s{seed}")
        targs = TrainingArguments(
            output_dir=out, eval_strategy="epoch", save_strategy="epoch", save_total_limit=1,
            learning_rate=args.lr, per_device_train_batch_size=args.bs, per_device_eval_batch_size=64,
            num_train_epochs=args.epochs, weight_decay=0.01, warmup_ratio=0.06,
            fp16=torch.cuda.is_available(), load_best_model_at_end=True,
            metric_for_best_model=select, greater_is_better=True,
            logging_steps=100, report_to="none", seed=seed, disable_tqdm=True)
        trainer = Trainer(model=model, args=targs, train_dataset=ds_tr, eval_dataset=ds_va,
                          tokenizer=tok, data_collator=DataCollatorWithPadding(tok),
                          compute_metrics=compute_metrics, callbacks=[EarlyStoppingCallback(2)])
        t0 = time.time()
        trainer.train()
        train_sec = time.time() - t0

        va_logits = trainer.predict(ds_va).predictions
        va_prob = torch.softmax(torch.tensor(va_logits), -1).numpy()
        m = metrics(va["label"].values, va_prob.argmax(1))
        np.save(val_npy, va_prob)

        # Sealed: test probabilities only; ds_te carries no labels, so no metric can be computed.
        te_prob = torch.softmax(torch.tensor(trainer.predict(ds_te).predictions), -1).numpy()
        np.save(test_npy, te_prob)

        save_dir = os.path.join(args.work_dir, f"model_{short}_s{seed}")
        trainer.save_model(save_dir)
        wfile = os.path.join(save_dir, "model.safetensors")
        digest = sha256(wfile) if os.path.exists(wfile) else ""
        row = {"task": args.task, "model": args.model, "seed": seed, "split": "val", **{k: round(v, 4) for k, v in m.items()},
               "best_epoch_ckpt": trainer.state.best_model_checkpoint, "train_sec": round(train_sec),
               "max_len": args.max_len, "lr": args.lr, "bs": args.bs, "epochs": args.epochs, "sha256": digest}
        print(json.dumps(row, indent=1))
        rows.append(row)
        pd.DataFrame(rows).sort_values("seed").to_csv(res_csv, index=False)
        # Weights and optimizer states run to GBs across 20 trainings; only the digest is kept.
        shutil.rmtree(out, ignore_errors=True)
        shutil.rmtree(save_dir, ignore_errors=True)

    df = pd.DataFrame(rows)
    print("\nVALIDATION mean ± std over seeds:")
    for k in [c for c in ["macro_f1", "mawdu_f1", "daif_f1", "sahih_f1", "mawdu_p", "mawdu_r", "accuracy"] if c in df]:
        print(f"  {k}: {df[k].mean():.4f} ± {df[k].std(ddof=1) if len(df) > 1 else 0:.4f}")

if __name__ == "__main__":
    main()
