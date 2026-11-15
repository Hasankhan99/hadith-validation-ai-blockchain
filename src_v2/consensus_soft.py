"""
Soft-probability consensus with prior correction and abstention (frozen design, thesis Section 4.5).

This module implements exactly the rule evaluated in experiments/r4 and r6:
    q(k)  = sum_i w_i * p_i(k) / sum_i w_i            (members' class-probability vectors p_i)
    q'(k) = q(k) / prior(k)**alpha, renormalised        (alpha = 0 disables prior correction)
    verdict = argmax q', confidence = max q', status = VERIFIED if confidence >= tau else FLAGGED_FOR_HUMAN_REVIEW
Ties in argmax go to the lowest label index, i.e. to Mawdu (label 0), the conservative choice.
"""
import numpy as np

VERIFIED = "VERIFIED"
FLAGGED = "FLAGGED_FOR_HUMAN_REVIEW"


def consensus_distribution(member_probs, prior, alpha=0.0, weights=None):
    """member_probs: list of (n, K) arrays; prior: (K,) class frequencies of the training split."""
    w = np.ones(len(member_probs)) if weights is None else np.asarray(weights, float)
    q = sum(wi * p for wi, p in zip(w, member_probs)) / w.sum()
    q = q / (np.asarray(prior, float) ** alpha)
    return q / q.sum(axis=1, keepdims=True)


def decide(member_probs, prior, alpha=0.0, tau=0.0, weights=None):
    """Returns (verdict, confidence, status) arrays."""
    q = consensus_distribution(member_probs, prior, alpha, weights)
    conf = q.max(axis=1)
    verdict = q.argmax(axis=1)
    status = np.where(conf >= tau, VERIFIED, FLAGGED)
    return verdict, conf, status


def to_ledger_label(task, k):
    """Chaincode codes are 0 = Mawdu, 1 = Daif, 2 = Sahih. The binary task uses 0 = Mawdu, 1 = Sahih."""
    k = int(k)
    return {0: 0, 1: 2}[k] if task == "binary" else k
