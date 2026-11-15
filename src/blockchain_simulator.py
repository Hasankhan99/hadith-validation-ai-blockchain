"""
Hyperledger Fabric Blockchain Ledger Simulator & Audit Trail Client.
Implements model registration, SHA-256 weight hash validation, and tamper-resistant transaction logging.
"""

import os
import time
import json
import hashlib
from typing import Dict, Any, Tuple

class HyperledgerFabricSimulator:
    def __init__(self, ledger_file: str = r"d:\research paper\blockchain\ledger.json"):
        self.ledger_file = ledger_file
        os.makedirs(os.path.dirname(self.ledger_file), exist_ok=True)
        if not os.path.exists(self.ledger_file):
            self._save_state({"models": {}, "transactions": []})

    def _load_state(self) -> Dict[str, Any]:
        with open(self.ledger_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _save_state(self, state: Dict[str, Any]):
        with open(self.ledger_file, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=2, ensure_ascii=False)

    def register_model(self, model_id: str, model_name: str, sha256_hash: str, weight: float) -> str:
        state = self._load_state()
        state["models"][model_id] = {
            "model_id": model_id,
            "model_name": model_name,
            "sha256_hash": sha256_hash,
            "accuracy_weight": weight,
            "registered_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        }
        self._save_state(state)
        print(f"[BLOCKCHAIN LEDGER] Registered Model {model_id} | SHA-256: {sha256_hash[:16]}...")
        return sha256_hash

    def record_validation_event(
        self,
        hadith_text: str,
        consensus_result: int,
        confidence_score: float,
        model_verdicts: Dict[str, int],
        submitted_model_hashes: Dict[str, str]
    ) -> Tuple[bool, str, Dict[str, Any]]:
        
        state = self._load_state()

        # 1. Cryptographic Security Check: Verify SHA-256 model hashes against registered ledger
        for model_id, submitted_hash in submitted_model_hashes.items():
            if model_id not in state["models"]:
                return False, f"SECURITY ERROR: Model {model_id} is not registered on the ledger!", {}
            registered_hash = state["models"][model_id]["sha256_hash"]
            if registered_hash != submitted_hash:
                return False, f"SECURITY ALERT: Tampering detected for model {model_id}! Hash mismatch.", {}

        # 2. Compute Hadith Hash & Transaction Hash
        hadith_hash = hashlib.sha256(hadith_text.encode('utf-8')).hexdigest()
        timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        tx_payload = f"{hadith_hash}:{consensus_result}:{confidence_score}:{timestamp}"
        tx_id = "0x" + hashlib.sha256(tx_payload.encode('utf-8')).hexdigest()

        tx_record = {
            "tx_id": tx_id,
            "hadith_hash": hadith_hash,
            "consensus_result": consensus_result,
            "consensus_label": "Sahih (Authentic)" if consensus_result == 1 else "Mawdu (Fabricated)",
            "confidence_score": round(confidence_score, 4),
            "model_verdicts": model_verdicts,
            "model_hashes": submitted_model_hashes,
            "timestamp": timestamp,
            "block_index": len(state["transactions"]) + 1
        }

        state["transactions"].append(tx_record)
        self._save_state(state)
        return True, tx_id, tx_record

if __name__ == "__main__":
    bc = HyperledgerFabricSimulator()
    bc.register_model("model_lr", "Logistic Regression Baseline", "a1b2c3d4e5f60000000000000000000000000000000000000000000000000000", 0.85)
    
    verdicts = {"model_lr": 1}
    hashes = {"model_lr": "a1b2c3d4e5f60000000000000000000000000000000000000000000000000000"}
    
    success, tx_id, rec = bc.record_validation_event("حدثنا مسدد قال", 1, 0.85, verdicts, hashes)
    print("Tx Success:", success)
    print("Tx ID:", tx_id)
