"""
utils/metrics.py

Evaluation metrics: accuracy, precision, recall, F1.
"""

import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def compute_metrics(preds, targets):
    """
    preds, targets: 1D tensors or lists of integer labels.
    Returns dict with accuracy, precision, recall, f1.
    """
    if isinstance(preds, torch.Tensor):
        preds   = preds.cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()

    preds   = np.array(preds)
    targets = np.array(targets)

    acc  = accuracy_score(targets, preds)
    prec = precision_score(targets, preds, zero_division=0)
    rec  = recall_score(targets, preds, zero_division=0)
    f1   = f1_score(targets, preds, zero_division=0)

    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


def format_metrics(metrics: dict, prefix="") -> str:
    p = f"{prefix} " if prefix else ""
    return (f"{p}Accuracy={metrics['accuracy']:.4f}  "
            f"Recall={metrics['recall']:.4f}  "
            f"Precision={metrics['precision']:.4f}  "
            f"F1={metrics['f1']:.4f}")
