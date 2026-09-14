"""Donor-level immune-age evaluation after predictions are sealed."""
from __future__ import annotations

import numpy as np


def age_metrics(age: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    age = np.asarray(age, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    if age.shape != prediction.shape or age.ndim != 1 or not len(age):
        raise ValueError("Age and prediction must be aligned non-empty vectors")
    if not np.isfinite(age).all() or not np.isfinite(prediction).all():
        raise ValueError("Age metrics require finite values")
    residual = prediction - age
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((age - age.mean()) ** 2))
    pearson = (
        float(np.corrcoef(age, prediction)[0, 1])
        if np.std(age) > 1e-12 and np.std(prediction) > 1e-12
        else float("nan")
    )
    return {
        "donors": int(len(age)),
        "mae": float(np.mean(np.abs(residual))),
        "median_ae": float(np.median(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "pearson_r": pearson,
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else float("nan"),
        "prediction_sd": float(np.std(prediction)),
    }
