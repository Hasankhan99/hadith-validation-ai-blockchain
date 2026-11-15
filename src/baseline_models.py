"""
Baseline Machine Learning Classifiers:
1. Logistic Regression (TF-IDF)
2. Random Forest Classifier (TF-IDF)
Computes SHA-256 weight hash for blockchain registration.
"""

import os
import hashlib
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support

def compute_file_sha256(filepath: str) -> str:
    """Computes SHA-256 hash of a file (model weights artifact)."""
    hasher = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()

class BaselineHadithClassifier:
    def __init__(self, model_type: str = 'logistic_regression'):
        self.model_type = model_type
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=10000)
        if model_type == 'logistic_regression':
            self.model = LogisticRegression(max_iter=1000, random_state=42)
        elif model_type == 'random_forest':
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

    def train(self, X_train, y_train):
        print(f"Extracting TF-IDF features for {len(X_train)} samples...")
        X_vec = self.vectorizer.fit_transform(X_train)
        print(f"Training {self.model_type}...")
        self.model.fit(X_vec, y_train)
        print("Training complete.")

    def evaluate(self, X_test, y_test):
        X_vec = self.vectorizer.transform(X_test)
        preds = self.model.predict(X_vec)
        probs = self.model.predict_proba(X_vec) if hasattr(self.model, "predict_proba") else None
        
        acc = accuracy_score(y_test, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, preds, average='weighted')
        
        metrics = {
            "accuracy": float(acc),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1)
        }
        return metrics, preds, probs

    def save_model(self, output_path: str) -> str:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        joblib.dump({"vectorizer": self.vectorizer, "model": self.model, "type": self.model_type}, output_path)
        sha256_hash = compute_file_sha256(output_path)
        print(f"Saved {self.model_type} to {output_path} | SHA-256: {sha256_hash}")
        return sha256_hash
