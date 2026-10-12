from __future__ import annotations

import numpy as np


def binary_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    target_bool = target.astype(bool)
    prediction_bool = prediction.astype(bool)
    tp = np.logical_and(target_bool, prediction_bool).sum()
    fp = np.logical_and(~target_bool, prediction_bool).sum()
    fn = np.logical_and(target_bool, ~prediction_bool).sum()
    tn = np.logical_and(~target_bool, ~prediction_bool).sum()
    union = tp + fp + fn
    return {
        "iou": float(tp / union) if union else 1.0,
        "dice": float(2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else 1.0,
        "precision": float(tp / (tp + fp)) if (tp + fp) else 1.0,
        "recall": float(tp / (tp + fn)) if (tp + fn) else 1.0,
        "pixel_accuracy": float((tp + tn) / (tp + fp + fn + tn)),
        "fp_pixels": int(fp),
        "fn_pixels": int(fn),
    }
