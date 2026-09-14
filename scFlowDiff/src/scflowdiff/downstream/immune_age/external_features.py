"""Source-only candidate construction and Flow-hidden extraction."""
from __future__ import annotations

import json
from itertools import pairwise
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

from .pipeline import _resolve
from .workflow import _sha256, output_root


def _decode(values) -> np.ndarray:
    return np.asarray([x.decode() if isinstance(x, bytes) else str(x) for x in values])


def _column(group: h5py.Group, key: str) -> np.ndarray:
    node = group[key]
    if isinstance(node, h5py.Group):
        categories = _decode(node["categories"][:])
        codes = np.asarray(node["codes"][:], dtype=np.int64)
        return np.asarray([categories[x] if x >= 0 else "" for x in codes])
    return _decode(node[:]) if node.dtype.kind in {"S", "O", "U"} else np.asarray(node[:])


def _feature_names(group: h5py.Group) -> list[str]:
    key = group["var"].attrs.get("_index", "_index")
    if isinstance(key, bytes):
        key = key.decode()
    return _decode(group[f"var/{key}"][:]).astype(str).tolist()


def _selected_rows(
    group: h5py.Group, rows: np.ndarray, columns: np.ndarray
) -> sp.csr_matrix:
    """Read selected CSR rows/columns without loading the complete H5AD matrix."""
    rows = np.asarray(rows, dtype=np.int64)
    columns = np.asarray(columns, dtype=np.int64)
    if not len(rows):
        return sp.csr_matrix((0, len(columns)), dtype=np.float32)
    order = np.argsort(rows)
    sorted_rows = rows[order]
    x = group["X"]
    width = int(x.attrs["shape"][1])
    boundaries = np.r_[0, np.flatnonzero(np.diff(sorted_rows) != 1) + 1, len(sorted_rows)]
    blocks = []
    for left, right in pairwise(boundaries):
        first, last = int(sorted_rows[left]), int(sorted_rows[right - 1])
        pointers = np.asarray(x["indptr"][first:last + 2], dtype=np.int64)
        lower, upper = int(pointers[0]), int(pointers[-1])
        values = np.asarray(x["data"][lower:upper], dtype=np.float32)
        indices = np.asarray(x["indices"][lower:upper], dtype=np.int64)
        block = sp.csr_matrix(
            (values, indices, pointers - lower), shape=(last - first + 1, width)
        )
        blocks.append(block[:, columns])
    matrix = sp.vstack(blocks, format="csr")
    inverse = np.empty_like(order)
    inverse[order] = np.arange(len(order))
    return matrix[inverse]


def _full_library(group: h5py.Group, rows: np.ndarray) -> float:
    x = group["X"]
    total = 0.0
    for row in np.asarray(rows, dtype=np.int64):
        lower, upper = int(x["indptr"][row]), int(x["indptr"][row + 1])
        total += float(np.asarray(x["data"][lower:upper], dtype=np.float64).sum())
    return total


def _canon_cell_type(values: np.ndarray) -> np.ndarray:
    return np.char.replace(values.astype(str), " ", "_")


def _canon_sex(values: np.ndarray) -> np.ndarray:
    values = np.char.lower(values.astype(str))
    return np.where(np.isin(values, ["female", "f"]), "female", "male")


def _candidate_cell_types(columns: list[str], modality: str) -> set[str]:
    """Return cell types required by the source-candidate feature contract."""
    if modality == "rna":
        return {
            name.split("__", 1)[0]
            for name in columns
            if "__" in name and not name.startswith("Missing__")
        }
    if modality == "atac":
        return {
            name.split("__", 2)[1]
            for name in columns
            if name.startswith("ATAC__") and name.count("__") >= 2
        }
    raise ValueError(f"Unsupported source modality: {modality!r}")


def _source_spec(config: dict[str, Any], root: Path, *, cohort: str | None, modality: str):
    if cohort is None:
        path = _resolve(root, config["dataset_path"])
        group_path = f"mod/{modality}"
        sex_key = "sex_standardized" if modality == "rna" else "sex"
        return path, group_path, "sample_id", "cell_type", sex_key
    spec = config["external_cohorts"][cohort]
    path = _resolve(root, spec["path"])
    return path, "", spec["donor_key"], spec["cell_type_key"], spec["sex_key"]


def build_external_source_candidates(
    config: dict[str, Any], root: Path, *, sex: str, direction: str
) -> Path:
    """Aggregate the observed external modality without reading age or target modality."""
    direction_spec = config["directions"][direction]
    cohort, modality = direction_spec["external_cohort"], direction_spec["source_modality"]
    path, group_path, donor_key, cell_type_key, sex_key = _source_spec(
        config, root, cohort=cohort, modality=modality
    )
    data_root = _resolve(root, config["data_root"]) / sex
    reference = pd.read_parquet(
        data_root / f"candidates/{modality}_candidates_development.parquet"
    )
    columns = reference.columns.astype(str).tolist()
    candidate_cell_types = _candidate_cell_types(columns, modality)
    with h5py.File(path, "r") as handle:
        group = handle[group_path] if group_path else handle
        donors = _column(group["obs"], donor_key).astype(str)
        cell_types = _canon_cell_type(_column(group["obs"], cell_type_key))
        sexes = _canon_sex(_column(group["obs"], sex_key))
        features = _feature_names(group)
        keep = np.flatnonzero(
            (sexes == sex) & np.isin(cell_types, list(candidate_cell_types))
        )
        donor_order = pd.unique(donors[keep]).astype(str).tolist()
        result = pd.DataFrame(
            np.nan, index=pd.Index(donor_order, name="sample_id"), columns=columns
        )
        feature_lookup = {name: index for index, name in enumerate(features)}
        present: set[tuple[str, str]] = set()
        grouped = pd.Series(
            keep, index=pd.MultiIndex.from_arrays([donors[keep], cell_types[keep]])
        ).groupby(level=[0, 1], sort=False)
        for (donor, cell_type), rows in grouped:
            row_index = rows.to_numpy(dtype=np.int64)
            if cohort is not None and modality == "rna" and len(row_index) < int(
                config["flow_hidden"]["min_cells_per_donor_cell_type"]
            ):
                continue
            present.add((str(donor), str(cell_type)))
            prefix = f"{cell_type}__" if modality == "rna" else f"ATAC__{cell_type}__"
            selected_columns = [name for name in columns if name.startswith(prefix)]
            physical = [name[len(prefix):] for name in selected_columns]
            physical_index = np.asarray(
                [feature_lookup[name] for name in physical], dtype=np.int64
            )
            group_matrix = _selected_rows(group, row_index, physical_index)
            sums = np.asarray(group_matrix.sum(axis=0)).ravel().astype(float)
            if modality == "rna":
                values = sums / len(row_index)
            else:
                library = _full_library(group, row_index)
                if library <= 0:
                    raise RuntimeError(f"Non-positive ATAC library: {donor}/{cell_type}")
                values = np.log1p(sums / library * 1_000_000.0)
            result.loc[str(donor), selected_columns] = values
    missing_prefix = "Missing__" if modality == "rna" else "Missing__ATAC__"
    for name in [x for x in columns if x.startswith(missing_prefix)]:
        cell_type = name[len(missing_prefix):]
        result[name] = [0.0 if (donor, cell_type) in present else 1.0 for donor in donor_order]
    continuous = [name for name in columns if not name.startswith("Missing__")]
    if modality == "rna":
        transform_path = _resolve(
            root, config["external_cohorts"][cohort]["transform_contract"]
        )
        transform = pd.read_csv(transform_path).set_index("feature").reindex(continuous)
        if transform.isna().any().any() or transform.index.tolist() != continuous:
            raise RuntimeError("External RNA transform does not match candidate feature order")
        if sex == "female":
            result[continuous] = result[continuous].subtract(
                transform["female_effect"], axis=1
            )
        result[continuous] = result[continuous].fillna(
            transform["missing_imputation"]
        )
    else:
        development_medians = reference[continuous].median(axis=0).fillna(0.0)
        result[continuous] = result[continuous].fillna(development_medians)
    destination = (
        output_root(config, root) / direction / sex / "external" / cohort
        / f"real_{modality}_candidates.parquet"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(destination)
    audit = {"status": "PASS", "cohort": cohort, "sex": sex, "modality": modality,
             "donors": len(result), "features": result.shape[1], "age_reads": 0,
             "target_modality_reads": 0}
    destination.with_suffix(".json").write_text(json.dumps(audit, indent=2) + "\n")
    return destination


def extract_flow_hidden(
    config: dict[str, Any], root: Path, *, sex: str, direction: str,
    cohort: str | None, paired_split: str = "development",
) -> Path:
    """Extract donor-cell-type target-stream hidden features from source modality only."""
    from scflowdiff.inference.flow_hidden import integrate_and_extract
    from scflowdiff.inference.translation import (
        load_latent_stats,
        load_stage1,
        load_stage2,
    )
    modality = config["directions"][direction]["source_modality"]
    if paired_split not in {"development", "sealed_test"}:
        raise ValueError("paired_split must be development or sealed_test")
    if cohort is not None and paired_split != "development":
        raise ValueError("paired_split applies only to the paired CIMA dataset")
    path, group_path, donor_key, cell_type_key, sex_key = _source_spec(
        config, root, cohort=cohort, modality=modality
    )
    contract = json.loads(_resolve(root, config["sexes"][sex]["feature_contract"]).read_text())
    allowed = [str(x) for x in contract["cell_types"]]
    with h5py.File(path, "r") as handle:
        group = handle[group_path] if group_path else handle
        donors = _column(group["obs"], donor_key).astype(str)
        cell_types = _canon_cell_type(_column(group["obs"], cell_type_key))
        sexes = _canon_sex(_column(group["obs"], sex_key))
        if cohort is None:
            split = pd.read_csv(
                _resolve(root, config["sexes"][sex]["split_csv"]),
                dtype={"sample_id": str},
            )
            if paired_split == "development":
                paired_donors = set(
                    split.loc[
                        ~split["benchmark_split"].eq("sealed_test"), "sample_id"
                    ]
                )
            else:
                paired_donors = set(
                    split.loc[split["benchmark_split"].eq("sealed_test"), "sample_id"]
                )
            eligible = (sexes == sex) & np.isin(donors, list(paired_donors))
        else:
            eligible = sexes == sex
        eligible &= np.isin(cell_types, allowed)
        max_cells = int(config["flow_hidden"]["max_cells_per_donor_cell_type"])
        selected: list[int] = []
        grouped = pd.Series(
            np.flatnonzero(eligible),
            index=pd.MultiIndex.from_arrays([donors[eligible], cell_types[eligible]]),
        ).groupby(level=[0, 1], sort=False)
        for _, rows in grouped:
            row_values = rows.to_numpy(dtype=np.int64)
            if cohort is not None and modality == "rna" and len(row_values) < int(
                config["flow_hidden"]["min_cells_per_donor_cell_type"]
            ):
                continue
            selected.extend(row_values[:max_cells].tolist())
        selected = sorted(selected)
        feature_indices = np.asarray(
            contract[f"{modality}_feature_indices"], dtype=np.int64
        )
        source = _selected_rows(group, np.asarray(selected), feature_indices)
    selected_donors, selected_cell_types = donors[selected], cell_types[selected]

    output_backbone = output_root(config, root) / "shared/scflowdiff" / sex
    pretrained_backbone = (
        root / "pretrained/downstream/immune_age/seed42/scflowdiff" / sex
    )
    required = (
        Path("stage1/best.ckpt"),
        Path("stage2/best_ema.ckpt"),
        Path("stage2/latent_stats.pt"),
    )
    backbone = (
        output_backbone
        if all((output_backbone / path).is_file() for path in required)
        else pretrained_backbone
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ae = load_stage1(
        backbone / "stage1/best.ckpt", len(contract["rna_feature_names"]),
        len(contract["atac_feature_names"]), device
    )
    stats = load_latent_stats(backbone / "stage2/latent_stats.pt")
    flow = load_stage2(backbone / "stage2/best_ema.ckpt", stats, device)
    ct_to_id = {name: index for index, name in enumerate(allowed)}
    generator = torch.Generator(device=device).manual_seed(config["seed"])
    batch_size = int(config["flow_hidden"]["batch_size"])
    hidden_batches = []
    for start in range(0, len(selected), batch_size):
        block = source[start:start + batch_size]
        size = block.shape[0]
        if modality == "rna":
            values = torch.as_tensor(block.toarray(), dtype=torch.float32, device=device)
            tokens = torch.arange(1, block.shape[1] + 1, device=device).repeat(size, 1)
            latent = ae.ae_model.rna_encoder(
                ae.ae_model.rna_input_layer(values, tokens, modality="rna")
            )
        else:
            token_array = np.zeros((size, 16), dtype=np.int64)
            value_array = np.zeros((size, 16), dtype=np.float32)
            for row in range(size):
                nonzero = block.getrow(row).indices
                row_rng = np.random.default_rng(
                    int(config["seed"]) + int(selected[start + row]) + 100_000
                )
                chosen = (
                    row_rng.choice(nonzero, size=16, replace=False)
                    if len(nonzero) > 16 else nonzero
                )
                token_array[row, :len(chosen)] = chosen + 1
                value_array[row, :len(chosen)] = 1.0
            tokens = torch.as_tensor(token_array, device=device)
            values = torch.as_tensor(value_array, device=device)
            latent = ae.ae_model.atac_encoder(
                ae.ae_model.atac_input_layer(values, tokens, modality="atac")
            )
        mean, sd = stats[f"{modality}_mean"].to(device), stats[f"{modality}_std"].to(device)
        normalized = (latent - mean) / sd
        initial = torch.randn(normalized.shape, generator=generator, device=device)
        labels = torch.as_tensor(
            [ct_to_id[x] for x in selected_cell_types[start:start + size]], device=device
        )
        _, hidden = integrate_and_extract(
            flow, normalized, initial, direction=direction,
            steps=int(config["flow_hidden"]["euler_steps"]), condition={"cell_type": labels}
        )
        hidden_batches.append(hidden.mean(dim=1).float().cpu().numpy())
    cell_hidden = np.concatenate(hidden_batches)
    keys = pd.MultiIndex.from_arrays([selected_donors, selected_cell_types])
    records, pooled, counts, source_context = [], [], [], []
    for key in keys.unique():
        index = np.flatnonzero(keys == key)
        records.append(key)
        pooled.append(cell_hidden[index].mean(axis=0))
        counts.append(len(index))
        if modality == "atac":
            binary_context = source[index].copy()
            binary_context.data = np.ones_like(binary_context.data, dtype=np.float32)
            source_context.append(
                np.asarray(binary_context.mean(axis=0), dtype=np.float32).ravel()
            )
    if cohort is not None:
        relative = f"external/{cohort}/flow_hidden.npz"
    elif paired_split == "sealed_test":
        relative = "paired/cima78/flow_hidden.npz"
    else:
        relative = "features/development_flow_hidden.npz"
    destination = output_root(config, root) / direction / sex / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "hidden": np.asarray(pooled, dtype=np.float32),
        "sample_id": np.asarray([x[0] for x in records]),
        "ct_id": np.asarray([ct_to_id[x[1]] for x in records], dtype=np.int64),
        "n_cells": np.asarray(counts, dtype=np.int64),
    }
    if modality == "atac":
        payload["source_context"] = np.asarray(source_context, dtype=np.float32)
    np.savez_compressed(destination, **payload)
    manifest = {
        "status": "PASS",
        "cohort": cohort or ("cima78" if paired_split == "sealed_test" else "cima316"),
        "split": "external" if cohort is not None else paired_split,
        "direction": direction,
        "sex": sex,
        "source_modality": modality,
        "integration_method": "explicit_euler",
        "euler_steps": int(config["flow_hidden"]["euler_steps"]),
        "min_cells_per_donor_cell_type": int(
            config["flow_hidden"]["min_cells_per_donor_cell_type"]
        ),
        "max_cells_per_donor_cell_type": max_cells,
        "donors": len(set(payload["sample_id"].astype(str))),
        "donor_cell_type_groups": len(payload["sample_id"]),
        "selected_source_cells": len(selected),
        "target_modality_reads": 0,
        "age_reads": 0,
        "sha256": _sha256(destination),
    }
    destination.with_suffix(".json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return destination
