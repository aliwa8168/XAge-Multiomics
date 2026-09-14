"""Predict-only use of frozen 500-feature RNA and Fusion immune-age clocks."""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd


def feature_order(path: str | Path) -> list[str]:
    features = pd.read_csv(path)["feature"].astype(str).tolist()
    if len(features) != 500 or len(set(features)) != 500:
        raise ValueError("The frozen immune-age feature contract must contain 500 unique rows")
    return features


def prepare_clock_input(
    frame: pd.DataFrame, feature_table: str | Path, medians: str | Path
) -> pd.DataFrame:
    features = feature_order(feature_table)
    stored = pd.read_parquet(medians)["development_median"].astype(float)
    stored.index = stored.index.astype(str)
    if stored.index.tolist() != features:
        raise ValueError("Stored median order differs from the feature contract")
    output = frame.reindex(columns=features).astype(float).fillna(stored).fillna(0.0)
    if not np.isfinite(output.to_numpy()).all():
        raise ValueError("Non-finite immune-age input after frozen-median imputation")
    return output


def predict_immune_age(
    frame: pd.DataFrame,
    *,
    model: str | Path,
    feature_table: str | Path,
    medians: str | Path,
) -> pd.Series:
    """Run a frozen clock without fitting or feature selection."""
    matrix = prepare_clock_input(frame, feature_table, medians)
    prediction = np.asarray(joblib.load(model).predict(matrix.to_numpy(dtype=float)), dtype=float)
    if not np.isfinite(prediction).all():
        raise RuntimeError("Non-finite immune-age prediction")
    return pd.Series(prediction, index=frame.index, name="predicted_immune_age")


def make_fusion_frame(rna: pd.DataFrame, predicted_atac: pd.DataFrame) -> pd.DataFrame:
    """Join RNA and DATP features; missing-indicator columns remain caller controlled."""
    if not rna.index.equals(predicted_atac.index):
        raise ValueError("RNA and predicted ATAC donor indices must be identical")
    duplicated = set(rna.columns).intersection(predicted_atac.columns)
    if duplicated:
        raise ValueError(f"Duplicated RNA/ATAC feature names: {sorted(duplicated)[:5]}")
    return pd.concat([rna, predicted_atac], axis=1)
