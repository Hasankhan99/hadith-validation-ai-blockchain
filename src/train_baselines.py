"""
Train & Evaluate Baseline Models (Logistic Regression & Random Forest).
Saves model weight artifacts and SHA-256 hashes for blockchain registration.
"""

import os
import sys
import pandas as pd
from baseline_models import BaselineHadithClassifier

def run_baseline_experiments(data_dir: str, models_dir: str):
    train_path = os.path.join(data_dir, "processed_train.csv")
    test_path = os.path.join(data_dir, "processed_test.csv")
    
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print("Dataset files not found. Run dataset_loader.py first.")
        return

    print("Loading processed train and test datasets...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train = train_df["normalized_text"].fillna("").astype(str)
    y_train = train_df["label"].values

    X_test = test_df["normalized_text"].fillna("").astype(str)
    y_test = test_df["label"].values

    results = {}

    # 1. Logistic Regression
    print("\n==========================================")
    print("Training Model 1: Logistic Regression (TF-IDF)")
    print("==========================================")
    lr_classifier = BaselineHadithClassifier(model_type='logistic_regression')
    lr_classifier.train(X_train, y_train)
    lr_metrics, _, _ = lr_classifier.evaluate(X_test, y_test)
    lr_hash = lr_classifier.save_model(os.path.join(models_dir, "model_logistic_regression.joblib"))
    results["logistic_regression"] = {"metrics": lr_metrics, "sha256": lr_hash}

    # 2. Random Forest
    print("\n==========================================")
    print("Training Model 2: Random Forest (TF-IDF)")
    print("==========================================")
    rf_classifier = BaselineHadithClassifier(model_type='random_forest')
    rf_classifier.train(X_train, y_train)
    rf_metrics, _, _ = rf_classifier.evaluate(X_test, y_test)
    rf_hash = rf_classifier.save_model(os.path.join(models_dir, "model_random_forest.joblib"))
    results["random_forest"] = {"metrics": rf_metrics, "sha256": rf_hash}

    print("\n==========================================")
    print("BASELINE EXPERIMENTAL RESULTS SUMMARY:")
    print("==========================================")
    for model_name, info in results.items():
        print(f"Model: {model_name.upper()}")
        print(f"  Accuracy:  {info['metrics']['accuracy']:.4f}")
        print(f"  Precision: {info['metrics']['precision']:.4f}")
        print(f"  Recall:    {info['metrics']['recall']:.4f}")
        print(f"  F1-Score:  {info['metrics']['f1_score']:.4f}")
        print(f"  SHA-256:   {info['sha256']}")
        print("------------------------------------------")

if __name__ == "__main__":
    data_dir = r"d:\research paper\data\processed"
    models_dir = r"d:\research paper\models"
    run_baseline_experiments(data_dir, models_dir)
