"""Fold-safe ATAC-context backend for the ATAC-to-RNA DATP direction.

This preserves the original reverse DATP contract: Flow hidden features and
observed donor-by-cell-type ATAC context are kept at the same row granularity.
Missing donor/cell-type combinations in the global context are filled only
from the corresponding training fold.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

EPS = 1e-8


@dataclass
class ArrayScaler:
    mean: np.ndarray
    sd: np.ndarray
    keep: np.ndarray
    clip: float = 8.0

    @classmethod
    def fit(cls, values: np.ndarray, clip: float = 8.0) -> ArrayScaler:
        values = np.asarray(values, dtype=np.float64)
        mean = np.nanmean(values, axis=0)
        sd = np.nanstd(values, axis=0)
        keep = np.isfinite(mean) & np.isfinite(sd) & (sd >= EPS)
        if not keep.any():
            raise RuntimeError("All reverse DATP context features failed variance filtering")
        return cls(mean=mean, sd=sd, keep=keep, clip=float(clip))

    def transform(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        result = (values[:, self.keep] - self.mean[self.keep]) / self.sd[self.keep]
        result = np.clip(result, -self.clip, self.clip)
        if not np.isfinite(result).all():
            raise RuntimeError("Non-finite reverse DATP context")
        return result


def _fit_pca(
    values: np.ndarray, components: int, seed: int, *, whiten: bool = False
) -> PCA:
    effective = min(int(components), len(values) - 1, values.shape[1])
    if effective < 1:
        raise RuntimeError("Insufficient reverse DATP rows/features for PCA")
    return PCA(
        n_components=effective,
        svd_solver="randomized",
        whiten=whiten,
        random_state=int(seed),
    ).fit(np.asarray(values, dtype=np.float64))


def _fit_cell_type_fill(
    donors: list[str],
    cell_type_ids: list[int],
    lookup: dict[tuple[str, int], int],
    context: np.ndarray,
) -> np.ndarray:
    fills = []
    for cell_type_id in cell_type_ids:
        rows = [lookup[(donor, cell_type_id)] for donor in donors if (donor, cell_type_id) in lookup]
        if not rows:
            raise RuntimeError(f"No training ATAC context for cell-type id {cell_type_id}")
        fills.append(np.median(context[rows], axis=0))
    return np.stack(fills)


def _donor_tensor(
    donors: list[str],
    cell_type_ids: list[int],
    lookup: dict[tuple[str, int], int],
    context: np.ndarray,
    cell_type_fill: np.ndarray,
) -> np.ndarray:
    result = np.empty(
        (len(donors), len(cell_type_ids), context.shape[1]), dtype=np.float64
    )
    for donor_index, donor in enumerate(donors):
        for type_index, cell_type_id in enumerate(cell_type_ids):
            row = lookup.get((donor, cell_type_id))
            result[donor_index, type_index] = (
                context[row] if row is not None else cell_type_fill[type_index]
            )
    return result


def fit_global_state(
    donors: list[str],
    cell_type_ids: list[int],
    lookup: dict[tuple[str, int], int],
    context: np.ndarray,
    *,
    components: int,
    seed: int,
) -> tuple[dict[str, Any], np.ndarray]:
    fill = _fit_cell_type_fill(donors, cell_type_ids, lookup, context)
    flat = _donor_tensor(donors, cell_type_ids, lookup, context, fill).reshape(
        len(donors), -1
    )
    scaler = ArrayScaler.fit(flat)
    scaled = scaler.transform(flat)
    pca = _fit_pca(scaled, components, seed)
    state = {
        "cell_type_ids": cell_type_ids,
        "cell_type_fill": fill,
        "scaler": scaler,
        "pca": pca,
    }
    return state, pca.transform(scaled)


def transform_global(
    state: dict[str, Any],
    donors: list[str],
    lookup: dict[tuple[str, int], int],
    context: np.ndarray,
) -> np.ndarray:
    flat = _donor_tensor(
        donors,
        list(state["cell_type_ids"]),
        lookup,
        context,
        state["cell_type_fill"],
    ).reshape(len(donors), -1)
    return state["pca"].transform(state["scaler"].transform(flat))


def _available_rows(
    donors: list[str], cell_type_id: int, lookup: dict[tuple[str, int], int]
) -> tuple[list[str], np.ndarray]:
    kept = [donor for donor in donors if (donor, cell_type_id) in lookup]
    rows = np.asarray([lookup[(donor, cell_type_id)] for donor in kept], dtype=np.int64)
    return kept, rows


def fit_cell_type_state(
    donors: list[str],
    cell_type: str,
    cell_type_id: int,
    global_scores: np.ndarray,
    lookup: dict[tuple[str, int], int],
    flow_hidden: np.ndarray,
    context: np.ndarray,
    *,
    flow_components: int,
    local_components: int,
    ridge_alpha: float,
    seed: int,
) -> tuple[dict[str, Any], list[str], np.ndarray]:
    kept, rows = _available_rows(donors, cell_type_id, lookup)
    if len(kept) < 3:
        raise RuntimeError(f"Insufficient reverse DATP training donors for {cell_type}")
    donor_position = {donor: index for index, donor in enumerate(donors)}
    kept_global = global_scores[[donor_position[donor] for donor in kept]]

    local_scaler = ArrayScaler.fit(context[rows])
    local_scaled = local_scaler.transform(context[rows])
    ridge = Ridge(alpha=float(ridge_alpha)).fit(kept_global, local_scaled)
    residual = local_scaled - ridge.predict(kept_global)
    residual_scaler = ArrayScaler.fit(residual)
    residual_scaled = residual_scaler.transform(residual)
    local_pca = _fit_pca(residual_scaled, local_components, seed + 1000)
    local_scores = local_pca.transform(residual_scaled)

    hidden_scaler = ArrayScaler.fit(flow_hidden[rows])
    hidden_scaled = hidden_scaler.transform(flow_hidden[rows])
    hidden_pca = _fit_pca(hidden_scaled, flow_components, seed + 2000, whiten=True)
    hidden_scores = hidden_pca.transform(hidden_scaled)

    raw = np.column_stack([hidden_scores, kept_global, local_scores])
    design_scaler = ArrayScaler.fit(raw)
    state = {
        "cell_type": cell_type,
        "ct_id": int(cell_type_id),
        "local_scaler": local_scaler,
        "ridge": ridge,
        "residual_scaler": residual_scaler,
        "local_pca": local_pca,
        "hidden_scaler": hidden_scaler,
        "hidden_pca": hidden_pca,
        "design_scaler": design_scaler,
    }
    return state, kept, design_scaler.transform(raw)


def transform_cell_type(
    state: dict[str, Any],
    donors: list[str],
    global_scores: np.ndarray,
    lookup: dict[tuple[str, int], int],
    flow_hidden: np.ndarray,
    context: np.ndarray,
) -> tuple[list[str], np.ndarray]:
    kept, rows = _available_rows(donors, int(state["ct_id"]), lookup)
    if not kept:
        return [], np.empty((0, 0), dtype=np.float64)
    donor_position = {donor: index for index, donor in enumerate(donors)}
    kept_global = global_scores[[donor_position[donor] for donor in kept]]
    local_scaled = state["local_scaler"].transform(context[rows])
    residual = local_scaled - state["ridge"].predict(kept_global)
    local_scores = state["local_pca"].transform(
        state["residual_scaler"].transform(residual)
    )
    hidden_scores = state["hidden_pca"].transform(
        state["hidden_scaler"].transform(flow_hidden[rows])
    )
    raw = np.column_stack([hidden_scores, kept_global, local_scores])
    return kept, state["design_scaler"].transform(raw)


def _score(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    correlations = []
    for column in range(truth.shape[1]):
        left, right = truth[:, column], prediction[:, column]
        correlations.append(
            np.corrcoef(left, right)[0, 1]
            if np.std(left) > 1e-12 and np.std(right) > 1e-12
            else np.nan
        )
    true_sd = np.std(truth, axis=0)
    pred_sd = np.std(prediction, axis=0)
    evaluable = true_sd > 1e-12
    return {
        "median_feature_wise_donor_pcc": float(np.nanmedian(correlations)),
        "median_between_donor_sd_ratio": float(
            np.nanmedian(pred_sd[evaluable] / true_sd[evaluable])
        ),
        "mae": float(np.mean(np.abs(prediction - truth))),
    }


def _fit_pls(
    design: np.ndarray, target: np.ndarray, requested: int
) -> tuple[PLSRegression, int, bool]:
    effective = max(
        1, min(int(requested), design.shape[1], len(design) - 1, target.shape[1])
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model = PLSRegression(
            n_components=effective, scale=True, max_iter=2000
        ).fit(design, target)
    warned = any(issubclass(item.category, ConvergenceWarning) for item in caught)
    return model, effective, warned


def fit_reverse_datp_bundle(
    source: pd.DataFrame,
    target: pd.DataFrame,
    flow_hidden: np.ndarray,
    source_context: np.ndarray,
    lookup: dict[tuple[str, int], int],
    target_table: pd.DataFrame,
    cell_types: list[str],
    *,
    sex: str,
    seed: int = 42,
    folds: int = 5,
    pls_components: tuple[int, ...] = (1, 2, 4, 8, 12),
    flow_pca_components: int = 32,
    global_components: int = 16,
    local_components: int = 8,
    ridge_alpha: float = 10.0,
) -> dict[str, Any]:
    """Fit the original fold-safe ATAC-to-RNA global/local DATP."""
    source, target = source.copy(), target.copy()
    source.index, target.index = source.index.astype(str), target.index.astype(str)
    donors = source.index.tolist()
    if target.index.tolist() != donors:
        raise ValueError("Reverse DATP source and target donor order must match")
    flow_hidden = np.asarray(flow_hidden, dtype=np.float64)
    source_context = np.asarray(source_context, dtype=np.float64)
    if (
        len(flow_hidden) != len(source_context)
        or not np.isfinite(flow_hidden).all()
        or not np.isfinite(source_context).all()
    ):
        raise ValueError("Invalid reverse donor-by-cell-type hidden/context rows")
    cell_type_ids = list(range(len(cell_types)))
    split = list(KFold(n_splits=folds, shuffle=True, random_state=seed).split(donors))
    fold_global = []
    for fold_index, (train_index, valid_index) in enumerate(split):
        train_donors = [donors[index] for index in train_index]
        valid_donors = [donors[index] for index in valid_index]
        state, train_scores = fit_global_state(
            train_donors,
            cell_type_ids,
            lookup,
            source_context,
            components=global_components,
            seed=seed + fold_index,
        )
        fold_global.append(
            {
                "train": train_donors,
                "valid": valid_donors,
                "train_scores": train_scores,
                "valid_scores": transform_global(
                    state, valid_donors, lookup, source_context
                ),
            }
        )

    selection: list[dict[str, Any]] = []
    states: list[dict[str, Any]] = []
    for type_index, (cell_type, table) in enumerate(
        target_table.groupby("cell_type", sort=False)
    ):
        cell_type = str(cell_type)
        cell_type_id = cell_types.index(cell_type)
        columns = table["feature"].astype(str).tolist()
        available, _ = _available_rows(donors, cell_type_id, lookup)
        truth = target.loc[available, columns].to_numpy(np.float64)
        position = {donor: index for index, donor in enumerate(available)}
        predictions = {value: np.full_like(truth, np.nan) for value in pls_components}
        effective = {value: [] for value in pls_components}
        convergence = {value: False for value in pls_components}
        for fold_index, payload in enumerate(fold_global):
            local_state, train_kept, train_design = fit_cell_type_state(
                payload["train"],
                cell_type,
                cell_type_id,
                payload["train_scores"],
                lookup,
                flow_hidden,
                source_context,
                flow_components=flow_pca_components,
                local_components=local_components,
                ridge_alpha=ridge_alpha,
                seed=seed + fold_index,
            )
            valid_kept, valid_design = transform_cell_type(
                local_state,
                payload["valid"],
                payload["valid_scores"],
                lookup,
                flow_hidden,
                source_context,
            )
            train_target = target.loc[train_kept, columns].to_numpy(np.float64)
            for requested in pls_components:
                model, used, warned = _fit_pls(
                    train_design, train_target, requested
                )
                for donor, row in zip(valid_kept, model.predict(valid_design)):
                    predictions[requested][position[donor]] = row
                effective[requested].append(used)
                convergence[requested] |= warned
        candidates = []
        for requested, prediction in predictions.items():
            if not np.isfinite(prediction).all():
                raise RuntimeError(
                    f"Incomplete reverse OOF prediction for {cell_type}/{requested}"
                )
            candidates.append(
                {
                    "requested_components": requested,
                    "effective_components_by_fold": effective[requested],
                    "convergence_warning": convergence[requested],
                    **_score(truth, prediction),
                }
            )
        selected = min(
            candidates,
            key=lambda row: (
                -row["median_feature_wise_donor_pcc"],
                abs(row["median_between_donor_sd_ratio"] - 1.0),
                row["mae"],
                row["requested_components"],
            ),
        )
        selection.append(
            {"cell_type": cell_type, "candidates": candidates, "selected": selected}
        )

    global_state, global_scores = fit_global_state(
        donors,
        cell_type_ids,
        lookup,
        source_context,
        components=global_components,
        seed=seed,
    )
    selected_by_type = {row["cell_type"]: row["selected"] for row in selection}
    for type_index, (cell_type, table) in enumerate(
        target_table.groupby("cell_type", sort=False)
    ):
        cell_type = str(cell_type)
        cell_type_id = cell_types.index(cell_type)
        columns = table["feature"].astype(str).tolist()
        state, kept, design = fit_cell_type_state(
            donors,
            cell_type,
            cell_type_id,
            global_scores,
            lookup,
            flow_hidden,
            source_context,
            flow_components=flow_pca_components,
            local_components=local_components,
            ridge_alpha=ridge_alpha,
            seed=seed + type_index * 100,
        )
        model, used, warned = _fit_pls(
            design,
            target.loc[kept, columns].to_numpy(np.float64),
            selected_by_type[cell_type]["requested_components"],
        )
        states.append(
            {
                **state,
                "columns": columns,
                "model": model,
                "requested_components": selected_by_type[cell_type][
                    "requested_components"
                ],
                "effective_components": used,
                "convergence_warning": warned,
                "n_development": len(kept),
            }
        )

    target_order = target_table["feature"].astype(str).tolist()
    return {
        "run_id": f"datp_atac_to_rna_seed{seed}",
        "status": "PASS",
        "sex": sex,
        "direction": "atac_to_rna",
        "source_modality": "atac",
        "target_modality": "rna",
        "context_backend": "donor_cell_type_atac_context",
        "architecture": (
            f"FlowPCA{flow_pca_components}_whiten__GlobalATACPCA{global_components}__"
            f"LocalATACRidge{ridge_alpha:g}ResidualPCA{local_components}__PLS2"
        ),
        "selection_boundary": "development_donor_5fold_all_transforms_fold_fitted",
        "selection_rule": "median_featurewise_PCC__then_SD_ratio__then_MAE__then_components",
        "cell_types": cell_types,
        "global_state": global_state,
        "states": states,
        "selection": selection,
        "target_order": target_order,
        "target_medians": target[target_order].median(axis=0).fillna(0.0).to_numpy(float),
    }


def predict_reverse_target(
    bundle: dict[str, Any],
    donors: list[str],
    flow_hidden: np.ndarray,
    source_context: np.ndarray,
    lookup: dict[tuple[str, int], int],
) -> pd.DataFrame:
    """Apply a locked reverse DATP without reading query RNA or age."""
    global_scores = transform_global(
        bundle["global_state"], donors, lookup, source_context
    )
    output = pd.DataFrame(index=donors, columns=bundle["target_order"], dtype=float)
    medians = dict(zip(bundle["target_order"], bundle["target_medians"]))
    for state in bundle["states"]:
        kept, design = transform_cell_type(
            state, donors, global_scores, lookup, flow_hidden, source_context
        )
        if kept:
            output.loc[kept, state["columns"]] = state["model"].predict(design)
        for donor in set(donors).difference(kept):
            output.loc[donor, state["columns"]] = [
                medians[column] for column in state["columns"]
            ]
    if not np.isfinite(output.to_numpy(float)).all():
        raise RuntimeError("Non-finite reverse DATP predictions")
    return output
