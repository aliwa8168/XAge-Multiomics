"""Inference with trained sex- and direction-specific DATP bundles."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .context import transform_design
from .directions import ATAC_TO_RNA, RNA_TO_ATAC, DirectionSpec, get_direction
from .features import transform_features
from .reverse import predict_reverse_target


def load_datp(path: str | Path) -> dict[str, Any]:
    bundle = joblib.load(Path(path))
    required = {"states", "target_order", "architecture", "sex"}
    missing = required.difference(bundle)
    if missing:
        raise ValueError(f"Invalid DATP bundle; missing {sorted(missing)}")
    bundle.setdefault("direction", "rna_to_atac")
    get_direction(bundle["direction"])
    return bundle


def load_flow_hidden(path: str | Path) -> tuple[np.ndarray, dict[tuple[str, int], int]]:
    payload = np.load(Path(path), allow_pickle=False)
    hidden = np.asarray(payload["hidden"], dtype=np.float64)
    donors = payload["sample_id"].astype(str)
    cell_types = np.asarray(payload["ct_id"], dtype=np.int64)
    lookup = {(donor, int(ct)): i for i, (donor, ct) in enumerate(zip(donors, cell_types))}
    if hidden.ndim != 2 or len(lookup) != len(hidden) or not np.isfinite(hidden).all():
        raise ValueError("Invalid or duplicated donor-by-cell-type Flow-hidden rows")
    return hidden, lookup


def load_flow_hidden_with_context(
    path: str | Path,
) -> tuple[np.ndarray, np.ndarray, dict[tuple[str, int], int]]:
    """Load the original reverse DATP donor-cell-type hidden/context contract."""
    payload = np.load(Path(path), allow_pickle=False)
    required = {"hidden", "source_context", "sample_id", "ct_id"}
    if missing := required.difference(payload.files):
        raise ValueError(f"Reverse Flow payload is missing {sorted(missing)}")
    hidden = np.asarray(payload["hidden"], dtype=np.float64)
    context = np.asarray(payload["source_context"], dtype=np.float64)
    donors = payload["sample_id"].astype(str)
    cell_types = np.asarray(payload["ct_id"], dtype=np.int64)
    lookup = {
        (donor, int(cell_type)): index
        for index, (donor, cell_type) in enumerate(zip(donors, cell_types))
    }
    if (
        hidden.ndim != 2
        or context.ndim != 2
        or len(hidden) != len(context)
        or len(lookup) != len(hidden)
        or not np.isfinite(hidden).all()
        or not np.isfinite(context).all()
    ):
        raise ValueError("Invalid reverse donor-by-cell-type Flow-hidden/context rows")
    return hidden, context, lookup


def predict_targeted_modality(
    bundle: dict[str, Any],
    source_features: pd.DataFrame,
    flow_hidden: np.ndarray,
    hidden_lookup: dict[tuple[str, int], int],
    *,
    direction: str | DirectionSpec | None = None,
    source_context: np.ndarray | None = None,
) -> pd.DataFrame:
    """Predict the contracted target without reading query targets or ages."""
    requested = get_direction(direction or bundle.get("direction", "rna_to_atac"))
    stored = get_direction(bundle.get("direction", "rna_to_atac"))
    if requested.name != stored.name:
        raise ValueError(
            f"DATP direction mismatch: bundle={stored.name}, requested={requested.name}"
        )
    source_features = source_features.copy()
    source_features.index = source_features.index.astype(str)
    donors = source_features.index.tolist()
    if stored.name == ATAC_TO_RNA.name:
        if source_context is None:
            raise ValueError("ATAC-to-RNA DATP requires donor-cell-type source_context")
        return predict_reverse_target(
            bundle, donors, flow_hidden, source_context, hidden_lookup
        )
    output = pd.DataFrame(
        index=source_features.index, columns=bundle["target_order"], dtype=float
    )
    donor_position = {donor: i for i, donor in enumerate(donors)}
    for state in bundle["states"]:
        cell_type_id = int(state["ct_id"])
        available = [donor for donor in donors if (donor, cell_type_id) in hidden_lookup]
        if not available:
            continue
        context = transform_design(source_features, state["design_state"])
        flow_rows = np.asarray([hidden_lookup[(donor, cell_type_id)] for donor in available])
        context_rows = np.asarray([donor_position[donor] for donor in available])
        features = transform_features(
            flow_hidden[flow_rows], context[context_rows], state["feature_state"]
        )
        pred_z = np.asarray(state["model"].predict(features), dtype=np.float64)
        pred = state["target_scaler"]["mean"] + pred_z * state["target_scaler"]["sd"]
        output.loc[available, state["columns"]] = pred
    if np.isinf(output.to_numpy(float)).any():
        raise RuntimeError("Infinite DATP predictions")
    return output


def predict_targeted_atac(
    bundle: dict[str, Any],
    rna_features: pd.DataFrame,
    flow_hidden: np.ndarray,
    hidden_lookup: dict[tuple[str, int], int],
) -> pd.DataFrame:
    return predict_targeted_modality(
        bundle,
        rna_features,
        flow_hidden,
        hidden_lookup,
        direction=RNA_TO_ATAC,
    )


def predict_targeted_rna(
    bundle: dict[str, Any],
    atac_features: pd.DataFrame,
    flow_hidden: np.ndarray,
    hidden_lookup: dict[tuple[str, int], int],
    source_context: np.ndarray,
) -> pd.DataFrame:
    return predict_targeted_modality(
        bundle,
        atac_features,
        flow_hidden,
        hidden_lookup,
        direction=ATAC_TO_RNA,
        source_context=source_context,
    )
