"""Fold-fitted donor-global and cell-type-local source-modality context."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge


def fit_scaler(values: np.ndarray, variance_floor: float) -> dict[str, np.ndarray]:
    values = np.asarray(values, dtype=np.float64)
    mean = values.mean(axis=0)
    sd = values.std(axis=0)
    keep = np.isfinite(sd) & (sd >= variance_floor)
    if not keep.any():
        raise RuntimeError("No variable context features remain")
    sd = sd.copy()
    sd[~keep] = 1.0
    return {"mean": mean, "sd": sd, "keep": keep}


def apply_scaler(values: np.ndarray, state: dict[str, np.ndarray], clip_z: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    result = (values[:, state["keep"]] - state["mean"][state["keep"]]) / state["sd"][
        state["keep"]
    ]
    result = np.clip(result, -clip_z, clip_z)
    if not np.isfinite(result).all():
        raise RuntimeError("Non-finite context values")
    return result


def fit_design_state(
    frame: pd.DataFrame,
    cell_type: str,
    *,
    source_modality: str = "rna",
    global_components: int = 16,
    local_components: int = 8,
    ridge_alpha: float = 10.0,
    variance_floor: float = 1e-4,
    clip_z: float = 8.0,
    seed: int = 42,
) -> tuple[dict[str, Any], np.ndarray]:
    """Fit donor-global plus cell-type-local residual source context."""
    source_modality = str(source_modality).lower()
    if source_modality not in {"rna", "atac"}:
        raise ValueError(f"Unsupported source modality: {source_modality!r}")
    feature_order = frame.columns.astype(str).tolist()
    local_prefix = (
        f"{cell_type}__" if source_modality == "rna" else f"ATAC__{cell_type}__"
    )
    local_columns = [x for x in feature_order if x.startswith(local_prefix)]
    if not local_columns:
        raise ValueError(
            f"No {source_modality.upper()} columns start with {local_prefix}"
        )
    global_scaler = fit_scaler(frame.to_numpy(float), variance_floor)
    global_scaled = apply_scaler(frame.to_numpy(float), global_scaler, clip_z)
    n_global = min(global_components, len(frame) - 1, global_scaled.shape[1])
    global_pca = PCA(n_components=n_global, svd_solver="randomized", random_state=seed)
    global_scores = global_pca.fit_transform(global_scaled)

    local_scaler = fit_scaler(frame[local_columns].to_numpy(float), variance_floor)
    local_scaled = apply_scaler(frame[local_columns].to_numpy(float), local_scaler, clip_z)
    local_regression = Ridge(alpha=ridge_alpha).fit(global_scores, local_scaled)
    residual = local_scaled - local_regression.predict(global_scores)
    residual_scaler = fit_scaler(residual, variance_floor)
    residual_scaled = apply_scaler(residual, residual_scaler, clip_z)
    n_local = min(local_components, len(frame) - 1, residual_scaled.shape[1])
    local_pca = PCA(n_components=n_local, svd_solver="randomized", random_state=seed + 1000)
    local_scores = local_pca.fit_transform(residual_scaled)

    combined = np.column_stack([global_scores, local_scores])
    final_scaler = fit_scaler(combined, variance_floor)
    state = {
        "feature_order": feature_order,
        "source_modality": source_modality,
        "local_prefix": local_prefix,
        "local_columns": local_columns,
        "global_scaler": global_scaler,
        "global_pca": global_pca,
        "local_scaler": local_scaler,
        "local_regression": local_regression,
        "residual_scaler": residual_scaler,
        "local_pca": local_pca,
        "final_scaler": final_scaler,
        "clip_z": clip_z,
    }
    return state, apply_scaler(combined, final_scaler, clip_z)


def transform_design(frame: pd.DataFrame, state: dict[str, Any]) -> np.ndarray:
    if frame.columns.astype(str).tolist() != list(state["feature_order"]):
        raise ValueError("Source feature names or order differ from the fitted DATP contract")
    clip_z = float(state["clip_z"])
    global_scaled = apply_scaler(frame.to_numpy(float), state["global_scaler"], clip_z)
    global_scores = state["global_pca"].transform(global_scaled)
    local_scaled = apply_scaler(
        frame[state["local_columns"]].to_numpy(float), state["local_scaler"], clip_z
    )
    residual = local_scaled - state["local_regression"].predict(global_scores)
    local_scores = state["local_pca"].transform(
        apply_scaler(residual, state["residual_scaler"], clip_z)
    )
    return apply_scaler(
        np.column_stack([global_scores, local_scores]), state["final_scaler"], clip_z
    )
