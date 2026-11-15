"""
Complete Google Colab AraBERT Fine-Tuning & Evaluation Script.
Run this script in Google Colab with GPU (T4/V100/A100) enabled.
"""

import os
import torch
import hashlib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
from datasets import Dataset

MODEL_NAME = "aubmindlab/bert-base-arabertv02"
TRAIN_FILE = "processed_train.csv"
VAL_FILE = "processed_val.csv"
OUTPUT_DIR = "./arabert_hadith_model"

def compute_sha256(filepath):
    hasher = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def detect_text_column(df: pd.DataFrame) -> str:
    """Automatically detects text column in DataFrame."""
    candidates = ['normalized_text', 'raw_text', 'matn', 'text', 'hadith']
    for col in candidates:
        if col in df.columns:
            return col
    # Fallback to first string column
    for col in df.columns:
        if col.lower() != 'label':
            return col
    return df.columns[0]

def run_training():
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU Device: {torch.cuda.get_device_name(0)}")
    else:
        print("WARNING: GPU is NOT enabled! Please go to Runtime -> Change runtime type -> Select T4 GPU in Colab.")

    # 1. Load Data
    if not os.path.exists(TRAIN_FILE) or not os.path.exists(VAL_FILE):
        print(f"ERROR: Could not find {TRAIN_FILE} or {VAL_FILE}. Please upload them to Colab files sidebar!")
        return

    train_df = pd.read_csv(TRAIN_FILE)
    val_df = pd.read_csv(VAL_FILE)

    train_text_col = detect_text_column(train_df)
    val_text_col = detect_text_column(val_df)

    print(f"Loaded {len(train_df)} training samples and {len(val_df)} validation samples.")
    print(f"Using text column: '{train_text_col}'")

    # Extract text & labels
    train_texts = train_df[train_text_col].fillna("").astype(str).tolist()
    train_labels = train_df["label"].astype(int).tolist()

    val_texts = val_df[val_text_col].fillna("").astype(str).tolist()
    val_labels = val_df["label"].astype(int).tolist()

    train_ds = Dataset.from_dict({"text": train_texts, "label": train_labels})
    val_ds = Dataset.from_dict({"text": val_texts, "label": val_labels})

    # 2. Tokenization
    print(f"Loading Tokenizer & Model: {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize_fn(examples):
        return tokenizer(examples["text"], truncation=True, max_length=128)

    train_ds = train_ds.map(tokenize_fn, batched=True)
    val_ds = val_ds.map(tokenize_fn, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    # 3. Training Arguments
    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=3,
        weight_decay=0.01,
        fp16=torch.cuda.is_available(),
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=50,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics
    )

    # 4. Fine-Tune AraBERT
    print("\n==========================================")
    print("STARTING ARABERT FINE-TUNING...")
    print("==========================================")
    trainer.train()

    # 5. Final Evaluation
    print("\n==========================================")
    print("FINAL EVALUATION RESULTS ON VALIDATION SET:")
    print("==========================================")
    eval_results = trainer.evaluate()
    for metric_name, val in eval_results.items():
        if isinstance(val, float):
            print(f"  {metric_name}: {val:.4f}")

    # 6. Save Model Artifact & Compute SHA-256
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    bin_path = os.path.join(OUTPUT_DIR, "model.safetensors")
    if not os.path.exists(bin_path):
        bin_path = os.path.join(OUTPUT_DIR, "pytorch_model.bin")

    if os.path.exists(bin_path):
        sha256_hash = compute_sha256(bin_path)
        print("\n==========================================")
        print("ARABERT MODEL WEIGHT SHA-256 HASH:")
        print("==========================================")
        print(f"File: {bin_path}")
        print(f"SHA-256: {sha256_hash}")
        print("Copy this SHA-256 hash to register AraBERT on your Hyperledger Fabric ledger!")

if __name__ == "__main__":
    run_training()
