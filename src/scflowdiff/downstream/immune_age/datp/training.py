"""Donor-level five-fold OOF selection and final bidirectional DATP fitting."""
from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import KFold

from .context import fit_design_state, transform_design
from .directions import RNA_TO_ATAC, DirectionSpec, get_direction
from .features import fit_feature_state, transform_features


def _rows(donors: list[str], cell_type_id: int, lookup: dict[tuple[str, int], int]):
    return np.asarray([lookup[(donor, cell_type_id)] for donor in donors], dtype=np.int64)


def _available(
    donors: list[str],
    cell_type: str,
    cell_type_id: int,
    columns: list[str],
    target: pd.DataFrame,
    lookup: dict[tuple[str, int], int],
    target_modality: str,
) -> list[str]:
    missing_columns = (
        f"Missing__{target_modality.upper()}__{cell_type}",
        f"Missing__{cell_type}",
    )
    return [
        donor
        for donor in donors
        if (donor, cell_type_id) in lookup
        and np.isfinite(target.loc[donor, columns].to_numpy(float)).all()
        and all(
            column not in target or float(target.loc[donor, column]) == 0.0
            for column in missing_columns
        )
    ]


def _target_scaler(values: np.ndarray) -> dict[str, np.ndarray]:
    mean = np.asarray(values, dtype=np.float64).mean(axis=0)
    sd = np.asarray(values, dtype=np.float64).std(axis=0)
    sd[~np.isfinite(sd) | (sd < 1e-6)] = 1.0
    return {"mean": mean, "sd": sd}


def _fit_pls(x, y, requested, *, scale, max_iter, tolerance):
    effective = max(1, min(int(requested), x.shape[1], len(x) - 1, y.shape[1]))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model = PLSRegression(
            n_components=effective, scale=scale, max_iter=max_iter, tol=tolerance
        ).fit(x, y)
    convergence = any(issubclass(item.category, ConvergenceWarning) for item in caught)
    return model, effective, convergence


def _score(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    correlations = []
    for column in range(truth.shape[1]):
        left, right = truth[:, column], prediction[:, column]
        correlations.append(
            np.corrcoef(left, right)[0, 1]
            if np.std(left) > 1e-12 and np.std(right) > 1e-12
            else np.nan
        )
    correlations = np.asarray(correlations)
    true_sd, pred_sd = np.std(truth, axis=0), np.std(prediction, axis=0)
    evaluable = true_sd > 1e-12
    return {
        "median_feature_wise_donor_pcc": float(np.nanmedian(correlations)),
        "median_between_donor_sd_ratio": float(np.nanmedian(pred_sd[evaluable] / true_sd[evaluable])),
        "mae": float(np.mean(np.abs(prediction - truth))),
    }


def fit_datp_bundle(
    source: pd.DataFrame,
    target: pd.DataFrame,
    flow_hidden: np.ndarray,
    hidden_lookup: dict[tuple[str, int], int],
    target_table: pd.DataFrame,
    cell_types: list[str],
    *,
    sex: str,
    direction: str | DirectionSpec = RNA_TO_ATAC,
    seed: int = 42,
    folds: int = 5,
    pls_components: tuple[int, ...] = (1, 2, 4, 8, 12),
    flow_pca_components: int = 32,
    flow_pca_whiten: bool = True,
    global_components: int = 16,
    local_components: int = 8,
    local_residual_ridge_alpha: float = 10.0,
    variance_floor: float = 1e-4,
    clip_z: float = 8.0,
    pls_scale: bool = False,
    pls_max_iter: int = 5000,
    pls_tolerance: float = 1e-5,
) -> dict[str, Any]:
    """Fit one DATP direction with every transform learned inside its OOF fold."""
    spec = get_direction(direction)
    source, target = source.copy(), target.copy()
    source.index, target.index = source.index.astype(str), target.index.astype(str)
    donors = source.index.tolist()
    if target.index.tolist() != donors:
        raise ValueError("Source and target development donors/order must match")
    required_columns = {"cell_type", "feature"}
    if not required_columns.issubset(target_table.columns):
        raise ValueError(f"target_table requires {sorted(required_columns)}")
    split = list(KFold(n_splits=folds, shuffle=True, random_state=seed).split(donors))
    states, selection = [], []

    def fit_context(frame, cell_type, context_seed):
        return fit_design_state(
            frame,
            cell_type,
            source_modality=spec.source_modality,
            global_components=global_components,
            local_components=local_components,
            ridge_alpha=local_residual_ridge_alpha,
            variance_floor=variance_floor,
            clip_z=clip_z,
            seed=context_seed,
        )

    for ct_index, (cell_type, table) in enumerate(target_table.groupby("cell_type", sort=False)):
        cell_type = str(cell_type)
        cell_type_id = cell_types.index(cell_type)
        columns = table["feature"].astype(str).tolist()
        available = _available(
            donors,
            cell_type,
            cell_type_id,
            columns,
            target,
            hidden_lookup,
            spec.target_modality,
        )
        position = {donor: index for index, donor in enumerate(available)}
        truth = target.loc[available, columns].to_numpy(np.float64)
        predictions = {value: np.full_like(truth, np.nan) for value in pls_components}
        effective = {value: [] for value in pls_components}
        convergence = {value: False for value in pls_components}

        for fold_index, (train_index, valid_index) in enumerate(split):
            fold_train = [donors[index] for index in train_index]
            fold_valid = [donors[index] for index in valid_index]
            design_state, _ = fit_context(
                source.loc[fold_train], cell_type, seed + ct_index * 100 + fold_index
            )
            train_design = transform_design(source.loc[fold_train], design_state)
            valid_design = transform_design(source.loc[fold_valid], design_state)
            train_design_lookup = dict(zip(fold_train, train_design))
            valid_design_lookup = dict(zip(fold_valid, valid_design))
            train_kept = _available(
                fold_train,
                cell_type,
                cell_type_id,
                columns,
                target,
                hidden_lookup,
                spec.target_modality,
            )
            valid_kept = _available(
                fold_valid,
                cell_type,
                cell_type_id,
                columns,
                target,
                hidden_lookup,
                spec.target_modality,
            )
            feature_state = fit_feature_state(
                flow_hidden[_rows(train_kept, cell_type_id, hidden_lookup)],
                np.stack([train_design_lookup[x] for x in train_kept]),
                pca_components=flow_pca_components,
                whiten=flow_pca_whiten,
                seed=seed + 2000 + ct_index * 100 + fold_index,
            )
            x_train = transform_features(
                flow_hidden[_rows(train_kept, cell_type_id, hidden_lookup)],
                np.stack([train_design_lookup[x] for x in train_kept]),
                feature_state,
            )
            x_valid = transform_features(
                flow_hidden[_rows(valid_kept, cell_type_id, hidden_lookup)],
                np.stack([valid_design_lookup[x] for x in valid_kept]),
                feature_state,
            )
            scaler = _target_scaler(target.loc[train_kept, columns].to_numpy(float))
            y_train = np.clip(
                (target.loc[train_kept, columns].to_numpy(float) - scaler["mean"])
                / scaler["sd"],
                -clip_z,
                clip_z,
            )
            for requested in pls_components:
                model, used, warned = _fit_pls(
                    x_train,
                    y_train,
                    requested,
                    scale=pls_scale,
                    max_iter=pls_max_iter,
                    tolerance=pls_tolerance,
                )
                pred = scaler["mean"] + np.asarray(model.predict(x_valid)) * scaler["sd"]
                for donor, row in zip(valid_kept, pred):
                    predictions[requested][position[donor]] = row
                effective[requested].append(used)
                convergence[requested] |= warned

        candidate_rows = []
        for requested, prediction in predictions.items():
            if not np.isfinite(prediction).all():
                raise RuntimeError(f"Incomplete OOF prediction for {cell_type}/{requested}")
            candidate_rows.append(
                {
                    "requested_components": requested,
                    "effective_components_by_fold": effective[requested],
                    "convergence_warning": convergence[requested],
                    **_score(truth, prediction),
                }
            )
        selected = min(
            candidate_rows,
            key=lambda row: (
                -row["median_feature_wise_donor_pcc"],
                abs(row["median_between_donor_sd_ratio"] - 1.0),
                row["mae"],
                row["requested_components"],
            ),
        )
        selection.append({"cell_type": cell_type, "candidates": candidate_rows, "selected": selected})

        design_state, _ = fit_context(source, cell_type, seed + ct_index * 100)
        design = transform_design(source, design_state)
        kept = _available(
            donors,
            cell_type,
            cell_type_id,
            columns,
            target,
            hidden_lookup,
            spec.target_modality,
        )
        design_lookup = dict(zip(donors, design))
        feature_state = fit_feature_state(
            flow_hidden[_rows(kept, cell_type_id, hidden_lookup)],
            np.stack([design_lookup[x] for x in kept]),
            pca_components=flow_pca_components,
            whiten=flow_pca_whiten,
            seed=seed + 2000 + ct_index * 100,
        )
        x = transform_features(
            flow_hidden[_rows(kept, cell_type_id, hidden_lookup)],
            np.stack([design_lookup[x] for x in kept]),
            feature_state,
        )
        target_values = target.loc[kept, columns].to_numpy(float)
        scaler = _target_scaler(target_values)
        y = np.clip((target_values - scaler["mean"]) / scaler["sd"], -clip_z, clip_z)
        model, used, warned = _fit_pls(
            x,
            y,
            selected["requested_components"],
            scale=pls_scale,
            max_iter=pls_max_iter,
            tolerance=pls_tolerance,
        )
        states.append(
            {
                "cell_type": cell_type,
                "ct_id": cell_type_id,
                "columns": columns,
                "design_state": design_state,
                "feature_state": feature_state,
                "target_scaler": scaler,
                "model": model,
                "requested_components": selected["requested_components"],
                "effective_components": used,
                "convergence_warning": warned,
                "n_development": len(kept),
            }
        )
    return {
        "run_id": f"datp_{spec.name}_seed{seed}",
        "status": "PASS",
        "sex": sex,
        "direction": spec.name,
        "source_modality": spec.source_modality,
        "target_modality": spec.target_modality,
        "flow_mode": spec.flow_mode,
        "hidden_stream": spec.hidden_stream,
        "architecture": (
            f"FlowPCA{flow_pca_components}_whiten__"
            f"Global{spec.source_modality.upper()}PCA{global_components}__"
            f"Local{spec.source_modality.upper()}Ridge{local_residual_ridge_alpha:g}"
            f"ResidualPCA{local_components}__PLS2"
        ),
        "selection_boundary": "development_donor_5fold_all_transforms_fold_fitted",
        "selection_rule": "median_featurewise_PCC__then_SD_ratio__then_MAE__then_components",
        "states": states,
        "selection": selection,
        "target_order": target_table["feature"].astype(str).tolist(),
    }
