"""
Ensemble Consensus Aggregation Module.
Implements:
1. Weighted Majority Voting (weights = validation set accuracy/F1)
2. Simple Majority Voting
3. Confidence-Threshold Voting
"""

import numpy as np
from typing import Dict, List, Tuple

class EnsembleConsensusEngine:
    def __init__(self, model_weights: Dict[str, float] = None):
        """
        model_weights: Dictionary mapping model_id to its validation score weight.
        Example: {'model_lr': 0.85, 'model_rf': 0.89, 'model_bilstm': 0.91, 'model_arabert': 0.96, 'model_deberta': 0.94}
        """
        self.model_weights = model_weights or {
            'model_lr': 0.85,
            'model_rf': 0.89,
            'model_bilstm': 0.91,
            'model_arabert': 0.96,
            'model_deberta': 0.94
        }

    def simple_majority_vote(self, verdicts: Dict[str, int]) -> Tuple[int, float]:
        """Simple majority voting across models."""
        votes = list(verdicts.values())
        ones = sum(1 for v in votes if v == 1)
        zeros = sum(1 for v in votes if v == 0)
        consensus = 1 if ones >= zeros else 0
        confidence = max(ones, zeros) / len(votes)
        return consensus, float(confidence)

    def weighted_majority_vote(self, verdicts: Dict[str, int], confidences: Dict[str, float] = None) -> Tuple[int, float]:
        """
        Weighted majority voting where each model's vote is scaled by its accuracy weight
        and optional prediction confidence score.
        """
        score_1 = 0.0
        score_0 = 0.0
        total_weight = 0.0

        for model_id, verdict in verdicts.items():
            w = self.model_weights.get(model_id, 1.0)
            conf = confidences.get(model_id, 1.0) if confidences else 1.0
            effective_w = w * conf
            total_weight += effective_w
            
            if verdict == 1:
                score_1 += effective_w
            else:
                score_0 += effective_w

        consensus = 1 if score_1 >= score_0 else 0
        consensus_confidence = max(score_1, score_0) / (total_weight if total_weight > 0 else 1.0)
        return consensus, float(consensus_confidence)

    def confidence_threshold_vote(self, verdicts: Dict[str, int], confidences: Dict[str, float], threshold: float = 0.70) -> Tuple[int, float, str]:
        """
        Confidence threshold voting. Returns status 'FLAGGED_FOR_HUMAN_REVIEW' if confidence < threshold.
        """
        consensus, conf = self.weighted_majority_vote(verdicts, confidences)
        if conf < threshold:
            return consensus, conf, "FLAGGED_FOR_HUMAN_REVIEW"
        return consensus, conf, "VERIFIED"

if __name__ == "__main__":
    engine = EnsembleConsensusEngine()
    test_verdicts = {
        'model_lr': 1,
        'model_rf': 1,
        'model_bilstm': 1,
        'model_arabert': 0,
        'model_deberta': 1
    }
    test_conf = {
        'model_lr': 0.78,
        'model_rf': 0.82,
        'model_bilstm': 0.85,
        'model_arabert': 0.95,
        'model_deberta': 0.91
    }

    c_simple, conf_simple = engine.simple_majority_vote(test_verdicts)
    c_weighted, conf_weighted = engine.weighted_majority_vote(test_verdicts, test_conf)
    c_thresh, conf_thresh, status = engine.confidence_threshold_vote(test_verdicts, test_conf, threshold=0.75)

    print("Simple Majority Consensus:", c_simple, "Conf:", conf_simple)
    print("Weighted Consensus:", c_weighted, "Conf:", round(conf_weighted, 4))
    print("Threshold Voting:", c_thresh, "Conf:", round(conf_thresh, 4), "Status:", status)
