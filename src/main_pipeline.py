"""
End-to-End Hadith Validation Pipeline.
Integrates Arabic NLP Preprocessing, Baseline & Ensemble AI Classifiers,
and Hyperledger Fabric Blockchain Audit Registration.
"""

import os
import joblib
import pandas as pd
from typing import Dict, Any

from preprocessing import normalize_arabic
from sanad_segmenter import segment_isnad_matn
from baseline_models import compute_file_sha256
from ensemble_consensus import EnsembleConsensusEngine
from blockchain_simulator import HyperledgerFabricSimulator

class HadithValidationPipeline:
    def __init__(self, models_dir: str = r"d:\research paper\models", ledger_path: str = r"d:\research paper\blockchain\ledger.json"):
        self.models_dir = models_dir
        self.blockchain = HyperledgerFabricSimulator(ledger_path)
        self.consensus_engine = EnsembleConsensusEngine()
        self.loaded_models = {}
        self.model_hashes = {}

        self._load_and_register_models()

    def _load_and_register_models(self):
        model_files = {
            "model_lr": ("Logistic Regression", "model_logistic_regression.joblib", 0.9719),
            "model_rf": ("Random Forest", "model_random_forest.joblib", 0.9780)
        }

        for model_id, (name, fname, weight) in model_files.items():
            fpath = os.path.join(self.models_dir, fname)
            if os.path.exists(fpath):
                data = joblib.load(fpath)
                sha256_hash = compute_file_sha256(fpath)
                self.loaded_models[model_id] = data
                self.model_hashes[model_id] = sha256_hash
                # Register on blockchain ledger
                self.blockchain.register_model(model_id, name, sha256_hash, weight)
            else:
                print(f"Warning: Model file {fpath} not found.")

    def validate_hadith(self, input_text: str) -> Dict[str, Any]:
        """
        Validates an input Hadith narration end-to-end.
        """
        # 1. Preprocessing & Segmentation
        norm_text = normalize_arabic(input_text)
        seg = segment_isnad_matn(input_text)

        # 2. Individual AI Model Inferences
        verdicts = {}
        confidences = {}

        for model_id, mdata in self.loaded_models.items():
            vec = mdata["vectorizer"]
            model = mdata["model"]
            X_vec = vec.transform([norm_text])
            pred = int(model.predict(X_vec)[0])
            probs = model.predict_proba(X_vec)[0] if hasattr(model, "predict_proba") else [0.5, 0.5]
            conf = float(probs[pred])

            verdicts[model_id] = pred
            confidences[model_id] = conf

        # 3. Ensemble Consensus Aggregation
        consensus, consensus_conf, status = self.consensus_engine.confidence_threshold_vote(verdicts, confidences)

        # 4. Hyperledger Fabric Blockchain Verification & Audit Trail Recording
        success, tx_id, tx_record = self.blockchain.record_validation_event(
            hadith_text=norm_text,
            consensus_result=consensus,
            confidence_score=consensus_conf,
            model_verdicts=verdicts,
            submitted_model_hashes=self.model_hashes
        )

        return {
            "status": "SUCCESS" if success else "SECURITY_ERROR",
            "hadith_segmentation": seg,
            "consensus_result": consensus,
            "consensus_label": "Sahih (Authentic)" if consensus == 1 else "Mawdu (Fabricated)",
            "confidence_score": round(consensus_conf, 4),
            "verification_status": status,
            "blockchain_audit": {
                "tx_id": tx_id,
                "block_index": tx_record.get("block_index", 0),
                "timestamp": tx_record.get("timestamp", ""),
                "model_integrity_verified": success
            },
            "individual_models": {
                m_id: {
                    "verdict": verdicts[m_id],
                    "label": "Sahih" if verdicts[m_id] == 1 else "Mawdu",
                    "confidence": round(confidences[m_id], 4),
                    "sha256": self.model_hashes[m_id]
                } for m_id in verdicts
            }
        }

if __name__ == "__main__":
    pipeline = HadithValidationPipeline()
    test_hadith = "حَدَّثَنَا مُحَمَّدُ بْنُ بَشَّارٍ، قَالَ: حَدَّثَنَا غُنْدَرٌ، قَالَ: حَدَّثَنَا شُعْبَةُ، عَنْ قَتَادَةَ، عَنْ أَنَسِ بْنِ مَالِكٍ، عَنِ النَّبِيِّ صَلَّى اللَّهُ عَلَيْهِ وَسَلَّمَ قَالَ: لاَ يُؤْمِنُ أَحَدُكُمْ حَتَّى يُحِبَّ لأَخِيهِ مَا يُحِبُّ لِنَفْسِهِ"
    result = pipeline.validate_hadith(test_hadith)
    print("\n==========================================")
    print("END-TO-END VALIDATION RESULT:")
    print("==========================================")
    import json
    print(json.dumps(result, indent=2, ensure_ascii=False))
