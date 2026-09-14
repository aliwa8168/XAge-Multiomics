import csv
import hashlib
import json
import os
from collections.abc import Callable, Sequence
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Literal, cast

import anndata as ad
import numpy as np
import torch
from anndata import AnnData
from pytorch_lightning import LightningDataModule

try:
    from cellarium.ml.data import (
        DistributedAnnDataCollection,
        IterableDistributedAnnDataCollectionDataset,
    )
    from cellarium.ml.utilities.data import AnnDataField, convert_to_tensor
except ImportError as e:
    raise ImportError(
        "cellarium-ml is required for DataModule functionality. "
        "If `cellarium-ml>0.0.7` is available on PyPI, install with: pip install cellarium-ml . "
        "Otherwise: pip install 'cellarium-ml @ git+https://github.com/cellarium-ai/cellarium-ml.git'"
    ) from e
from torch.utils._pytree import tree_map
from torch.utils.data import DataLoader, Dataset

from scflowdiff.core.data_utils import get_tissue_adata_files, sort_h5ad_files
from scflowdiff.core.constants import ModelEnum
from scflowdiff.core.encoder import VocabularyEncoderSimplified
from scflowdiff.core.logger import logger


class H5CSRRowMatrix:
    """Read one CSR row at a time from an H5MU modality without materializing it."""

    def __init__(self, path: Path, group_path: str):
        import h5py

        self.path = Path(path)
        self.group_path = group_path.strip("/")
        with h5py.File(self.path, "r") as handle:
            group = handle[self.group_path]
            self.shape = tuple(int(x) for x in group.attrs["shape"])
        self._handle: Any | None = None

    def _group(self) -> Any:
        import h5py

        if self._handle is None:
            self._handle = h5py.File(self.path, "r")
        return self._handle[self.group_path]

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_handle"] = None
        return state

    def getrow(self, idx: int) -> Any:
        from scipy import sparse

        if not 0 <= int(idx) < self.shape[0]:
            raise IndexError(f"row index {idx} is outside {self.shape}")
        group = self._group()
        start = int(group["indptr"][idx])
        stop = int(group["indptr"][idx + 1])
        return sparse.csr_matrix(
            (
                np.asarray(group["data"][start:stop], dtype=np.float32),
                np.asarray(group["indices"][start:stop], dtype=np.int64),
                np.asarray([0, stop - start], dtype=np.int64),
            ),
            shape=(1, self.shape[1]),
        )

    def row_sum(self, idx: int) -> float:
        group = self._group()
        start = int(group["indptr"][idx])
        stop = int(group["indptr"][idx + 1])
        return float(np.asarray(group["data"][start:stop], dtype=np.float64).sum())


class MatrixView:
    """Row/column view over an in-memory or streamed sparse matrix."""

    def __init__(
        self,
        matrix: Any,
        *,
        rows: np.ndarray | None = None,
        columns: np.ndarray | None = None,
    ):
        self.matrix = matrix
        self.rows = (
            np.arange(matrix.shape[0], dtype=np.int64)
            if rows is None
            else np.asarray(rows, dtype=np.int64)
        )
        self.columns = (
            np.arange(matrix.shape[1], dtype=np.int64)
            if columns is None
            else np.asarray(columns, dtype=np.int64)
        )
        if len(np.unique(self.rows)) != len(self.rows):
            raise ValueError("MatrixView row indices must be unique")
        if len(np.unique(self.columns)) != len(self.columns):
            raise ValueError("MatrixView column indices must be unique")
        if len(self.rows) and (self.rows.min() < 0 or self.rows.max() >= matrix.shape[0]):
            raise IndexError("MatrixView row index is out of bounds")
        if len(self.columns) and (
            self.columns.min() < 0 or self.columns.max() >= matrix.shape[1]
        ):
            raise IndexError("MatrixView column index is out of bounds")
        self.shape = (len(self.rows), len(self.columns))

    def getrow(self, idx: int) -> Any:
        if not 0 <= int(idx) < self.shape[0]:
            raise IndexError(f"row index {idx} is outside {self.shape}")
        physical = int(self.rows[int(idx)])
        if hasattr(self.matrix, "getrow"):
            return self.matrix.getrow(physical)[:, self.columns]
        from scipy import sparse

        return sparse.csr_matrix(np.asarray(self.matrix[physical, self.columns]).reshape(1, -1))

    def row_sum(self, idx: int) -> float:
        return float(self.getrow(idx).sum())


def _to_dense_float32(x: Any) -> np.ndarray:
    """Convert dense or sparse single-cell matrices to float32 NumPy arrays."""
    if hasattr(x, "toarray"):
        x = x.toarray()
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D matrix, got shape {arr.shape}")
    return arr


def _tokenize_binary_or_count_matrix(
    matrix: np.ndarray,
    token_ids: np.ndarray,
    seq_len: int | None,
    pad_token_idx: int,
    sample: Literal["expressed", "random", "none", "balanced", "top"] = "expressed",
    seed: int = 42,
    binarize: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Tokenize each row into fixed-length feature/value arrays.

    For sparse modalities, non-zero features are emitted first and padded with
    ``pad_token_idx``. If a row has more non-zero features than ``seq_len``, a
    reproducible subset is sampled. This keeps the transformer sequence length
    bounded while preserving paired feature tokens and observed values.
    """
    values = (matrix > 0).astype(np.float32) if binarize else matrix.astype(np.float32)
    n_cells, n_features = values.shape
    if seq_len is None or seq_len <= 0:
        seq_len = n_features
    if token_ids.shape[0] != n_features:
        raise ValueError("token_ids length must match the matrix feature dimension")

    rng = np.random.default_rng(seed)
    out_tokens = np.full((n_cells, seq_len), pad_token_idx, dtype=np.int64)
    out_values = np.zeros((n_cells, seq_len), dtype=np.float32)

    for i in range(n_cells):
        if sample == "none":
            idx = np.arange(n_features)
        elif sample == "random":
            idx = rng.choice(n_features, size=min(seq_len, n_features), replace=False)
        elif sample == "expressed":
            idx = np.flatnonzero(values[i] > 0)
            if len(idx) == 0:
                idx = np.arange(min(seq_len, n_features))
            elif len(idx) > seq_len:
                idx = rng.choice(idx, size=seq_len, replace=False)
        elif sample == "top":
            idx = np.flatnonzero(values[i] > 0)
            if len(idx) == 0:
                idx = np.arange(min(seq_len, n_features))
            elif len(idx) > seq_len:
                top_order = np.argpartition(values[i, idx], -seq_len)[-seq_len:]
                idx = idx[top_order]
        elif sample == "balanced":
            positive_idx = np.flatnonzero(values[i] > 0)
            negative_idx = np.flatnonzero(values[i] <= 0)
            n_pos = min(len(positive_idx), max(1, seq_len // 2))
            n_neg = min(len(negative_idx), seq_len - n_pos)
            pos = rng.choice(positive_idx, size=n_pos, replace=False) if n_pos > 0 else np.empty(0, dtype=np.int64)
            neg = rng.choice(negative_idx, size=n_neg, replace=False) if n_neg > 0 else np.empty(0, dtype=np.int64)
            idx = np.concatenate([pos, neg])
            rng.shuffle(idx)
        else:
            raise ValueError(f"Unsupported multimodal sampling mode: {sample}")

        idx = np.asarray(idx[:seq_len], dtype=np.int64)
        length = len(idx)
        out_tokens[i, :length] = token_ids[idx]
        out_values[i, :length] = values[i, idx]

    return out_tokens, out_values


class MultimodalMuDataDataset(Dataset):
    """Paired RNA+ATAC token dataset backed by in-memory or row-streamed matrices."""

    def __init__(
        self,
        rna_matrix: Any,
        atac_matrix: Any,
        rna_seq_len: int | None,
        atac_seq_len: int | None,
        sample_features: Literal["expressed", "random", "none", "balanced", "top"] = "expressed",
        rna_sample_features: Literal["expressed", "random", "none", "balanced", "top"] | None = None,
        atac_sample_features: Literal["expressed", "random", "none", "balanced", "top"] | None = None,
        rna_encoder_seq_len: int | None = None,
        rna_encoder_sample_features: Literal["expressed", "random", "none", "balanced", "top"] | None = None,
        atac_encoder_seq_len: int | None = None,
        atac_encoder_sample_features: Literal["expressed", "random", "none", "balanced", "top"] | None = None,
        seed: int = 42,
        pad_token_idx: int = 0,
        cell_type: np.ndarray | None = None,
    ):
        if rna_matrix.shape[0] != atac_matrix.shape[0]:
            raise ValueError("RNA and ATAC matrices must contain the same number of cells")
        if len(rna_matrix.shape) != 2 or len(atac_matrix.shape) != 2:
            raise ValueError("RNA and ATAC matrices must both be 2D")
        self.rna_matrix = rna_matrix
        self.atac_matrix = atac_matrix
        self.rna_token_ids = np.arange(1, self.rna_matrix.shape[1] + 1, dtype=np.int64)
        self.atac_token_ids = np.arange(1, self.atac_matrix.shape[1] + 1, dtype=np.int64)
        self.pad_token_idx = pad_token_idx
        self.rna_seq_len = self.rna_matrix.shape[1] if rna_seq_len is None or rna_seq_len <= 0 else rna_seq_len
        self.atac_seq_len = self.atac_matrix.shape[1] if atac_seq_len is None or atac_seq_len <= 0 else atac_seq_len
        self.rna_sample_features = rna_sample_features or sample_features
        self.atac_sample_features = atac_sample_features or sample_features
        self.rna_encoder_seq_len = rna_encoder_seq_len
        self.rna_encoder_sample_features = rna_encoder_sample_features
        self.atac_encoder_seq_len = atac_encoder_seq_len
        self.atac_encoder_sample_features = atac_encoder_sample_features
        self.seed = seed
        self.rna_library_size = None
        if not hasattr(self.rna_matrix, "row_sum"):
            self.rna_library_size = np.asarray(self.rna_matrix.sum(axis=1), dtype=np.float32).reshape(-1, 1)
            self.rna_library_size = np.maximum(self.rna_library_size, 1.0)
        self.cell_type = None if cell_type is None else np.asarray(cell_type, dtype=np.int64)
        if self.cell_type is not None and self.cell_type.shape[0] != self.rna_matrix.shape[0]:
            raise ValueError("cell_type labels must have the same number of rows as RNA/ATAC matrices")

    @property
    def n_genes(self) -> int:
        return int(self.rna_matrix.shape[1])

    @property
    def n_peaks(self) -> int:
        return int(self.atac_matrix.shape[1])

    def __len__(self) -> int:
        return int(self.rna_matrix.shape[0])

    def _tokenize_row(
        self,
        matrix: Any,
        idx: int,
        seq_len: int | None,
        token_ids: np.ndarray,
        binarize: bool,
        seed_offset: int,
        sample_features: Literal["expressed", "random", "none", "balanced", "top"],
    ) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(self.seed + seed_offset + idx)
        n_features = matrix.shape[1]
        if seq_len is None or seq_len <= 0:
            seq_len = n_features
        if hasattr(matrix, "getrow"):
            row = matrix.getrow(idx)
            nonzero_idx = row.indices.astype(np.int64, copy=False)
            row_values = row.data.astype(np.float32, copy=False)
            value_lookup = dict(zip(nonzero_idx.tolist(), row_values.tolist(), strict=False))
        else:
            dense_row = np.asarray(matrix[idx], dtype=np.float32).reshape(-1)
            nonzero_idx = np.flatnonzero(dense_row > 0)
            value_lookup = None

        if sample_features == "none":
            selected_idx = np.arange(min(seq_len, n_features), dtype=np.int64)
        elif sample_features == "random":
            selected_idx = rng.choice(n_features, size=min(seq_len, n_features), replace=False).astype(np.int64)
        elif sample_features == "expressed":
            if len(nonzero_idx) == 0:
                selected_idx = np.arange(min(seq_len, n_features), dtype=np.int64)
            elif len(nonzero_idx) > seq_len:
                selected_idx = rng.choice(nonzero_idx, size=seq_len, replace=False).astype(np.int64)
            else:
                selected_idx = nonzero_idx[:seq_len].astype(np.int64, copy=False)
        elif sample_features == "top":
            if len(nonzero_idx) == 0:
                selected_idx = np.arange(min(seq_len, n_features), dtype=np.int64)
            elif len(nonzero_idx) > seq_len:
                if value_lookup is None:
                    values_for_rank = dense_row[nonzero_idx]
                else:
                    values_for_rank = row_values
                top_local = np.argpartition(values_for_rank, -seq_len)[-seq_len:]
                selected_idx = nonzero_idx[top_local].astype(np.int64, copy=False)
            else:
                selected_idx = nonzero_idx[:seq_len].astype(np.int64, copy=False)
        elif sample_features == "balanced":
            all_idx = np.arange(n_features, dtype=np.int64)
            if len(nonzero_idx) == 0:
                selected_idx = rng.choice(all_idx, size=min(seq_len, n_features), replace=False).astype(np.int64)
            else:
                zero_mask = np.ones(n_features, dtype=bool)
                zero_mask[nonzero_idx] = False
                zero_idx = np.flatnonzero(zero_mask).astype(np.int64, copy=False)
                n_pos = min(len(nonzero_idx), max(1, seq_len // 2))
                n_neg = min(len(zero_idx), seq_len - n_pos)
                pos = rng.choice(nonzero_idx, size=n_pos, replace=False).astype(np.int64)
                neg = rng.choice(zero_idx, size=n_neg, replace=False).astype(np.int64)
                selected_idx = np.concatenate([pos, neg])
                if len(selected_idx) < min(seq_len, n_features):
                    remaining = np.setdiff1d(all_idx, selected_idx, assume_unique=False)
                    extra = rng.choice(
                        remaining,
                        size=min(seq_len, n_features) - len(selected_idx),
                        replace=False,
                    ).astype(np.int64)
                    selected_idx = np.concatenate([selected_idx, extra])
                rng.shuffle(selected_idx)
        else:
            raise ValueError(f"Unsupported multimodal sampling mode: {sample_features}")

        out_tokens = np.full(seq_len, self.pad_token_idx, dtype=np.int64)
        out_values = np.zeros(seq_len, dtype=np.float32)
        length = len(selected_idx)
        out_tokens[:length] = token_ids[selected_idx]

        if sample_features == "none" and value_lookup is not None:
            valid = nonzero_idx[nonzero_idx < length]
            if len(valid) > 0:
                values = row_values[nonzero_idx < length]
                out_values[valid] = values
            if binarize:
                out_values = (out_values > 0).astype(np.float32)
        elif value_lookup is None:
            values = dense_row[selected_idx]
            if binarize:
                values = (values > 0).astype(np.float32)
            out_values[:length] = values
        else:
            values = np.asarray([value_lookup.get(int(j), 0.0) for j in selected_idx], dtype=np.float32)
            if binarize:
                values = (values > 0).astype(np.float32)
            out_values[:length] = values
        return out_tokens, out_values

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        rna_genes, rna_counts = self._tokenize_row(
            self.rna_matrix,
            idx,
            self.rna_seq_len,
            self.rna_token_ids,
            binarize=False,
            seed_offset=0,
            sample_features=self.rna_sample_features,
        )
        atac_peaks, atac_values = self._tokenize_row(
            self.atac_matrix,
            idx,
            self.atac_seq_len,
            self.atac_token_ids,
            binarize=True,
            seed_offset=100_000,
            sample_features=self.atac_sample_features,
        )
        if self.rna_library_size is None:
            rna_library_size = np.asarray(
                [max(self.rna_matrix.row_sum(idx), 1.0)], dtype=np.float32
            )
        else:
            rna_library_size = self.rna_library_size[idx].astype(np.float32, copy=True)
        output = {
            ModelEnum.RNA_COUNTS.value: torch.from_numpy(rna_counts),
            ModelEnum.RNA_GENES.value: torch.from_numpy(rna_genes),
            ModelEnum.RNA_LIBRARY_SIZE.value: torch.from_numpy(rna_library_size),
            ModelEnum.ATAC_VALUES.value: torch.from_numpy(atac_values),
            ModelEnum.ATAC_PEAKS.value: torch.from_numpy(atac_peaks),
        }
        if self.rna_encoder_seq_len is not None and self.rna_encoder_sample_features is not None:
            rna_genes_subset, rna_counts_subset = self._tokenize_row(
                self.rna_matrix,
                idx,
                self.rna_encoder_seq_len,
                self.rna_token_ids,
                binarize=False,
                seed_offset=200_000,
                sample_features=self.rna_encoder_sample_features,
            )
            output[ModelEnum.RNA_COUNTS_SUBSET.value] = torch.from_numpy(rna_counts_subset)
            output[ModelEnum.RNA_GENES_SUBSET.value] = torch.from_numpy(rna_genes_subset)
        if self.atac_encoder_seq_len is not None and self.atac_encoder_sample_features is not None:
            atac_peaks_subset, atac_values_subset = self._tokenize_row(
                self.atac_matrix,
                idx,
                self.atac_encoder_seq_len,
                self.atac_token_ids,
                binarize=True,
                seed_offset=300_000,
                sample_features=self.atac_encoder_sample_features,
            )
            output[ModelEnum.ATAC_VALUES_SUBSET.value] = torch.from_numpy(atac_values_subset)
            output[ModelEnum.ATAC_PEAKS_SUBSET.value] = torch.from_numpy(atac_peaks_subset)
        if self.cell_type is not None:
            output[ModelEnum.CELL_TYPE.value] = torch.tensor(int(self.cell_type[idx]), dtype=torch.long)
        return output


class MultimodalMuDataModule(LightningDataModule):
    """Minimal MuData loader for paired RNA and binary ATAC transformer batches."""

    def __init__(
        self,
        h5mu_path: Path,
        batch_size: int = 32,
        test_batch_size: int | None = None,
        rna_seq_len: int | None = 512,
        atac_seq_len: int | None = 1024,
        sample_features: Literal["expressed", "random", "none", "balanced", "top"] = "expressed",
        rna_sample_features: Literal["expressed", "random", "none", "balanced", "top"] | None = None,
        atac_sample_features: Literal["expressed", "random", "none", "balanced", "top"] | None = "balanced",
        rna_encoder_seq_len: int | None = None,
        rna_encoder_sample_features: Literal["expressed", "random", "none", "balanced", "top"] | None = None,
        atac_encoder_seq_len: int | None = None,
        atac_encoder_sample_features: Literal["expressed", "random", "none", "balanced", "top"] | None = None,
        train_fraction: float = 0.8,
        val_fraction: float = 0.1,
        test_fraction: float = 0.1,
        split_manifest_path: Path | None = None,
        donor_split_csv: Path | None = None,
        donor_key: str = "sample_id",
        donor_split_column: str | None = None,
        donor_train_label: str = "train",
        donor_validation_label: str = "validation",
        donor_test_label: str = "sealed_test",
        outer_train_label: str = "train",
        inner_val_fraction: float = 0.1,
        feature_contract_path: Path | None = None,
        seed: int = 42,
        num_workers: int = 0,
    ):
        super().__init__()
        self.h5mu_path = Path(h5mu_path)
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size or batch_size
        self.rna_seq_len = rna_seq_len
        self.atac_seq_len = atac_seq_len
        self.sample_features = sample_features
        self.rna_sample_features = rna_sample_features
        self.atac_sample_features = atac_sample_features
        self.rna_encoder_seq_len = rna_encoder_seq_len
        self.rna_encoder_sample_features = rna_encoder_sample_features
        self.atac_encoder_seq_len = atac_encoder_seq_len
        self.atac_encoder_sample_features = atac_encoder_sample_features
        self.train_fraction = float(train_fraction)
        self.val_fraction = val_fraction
        self.test_fraction = float(test_fraction)
        self.split_manifest_path = (
            None if split_manifest_path is None else Path(split_manifest_path)
        )
        self.donor_split_csv = None if donor_split_csv is None else Path(donor_split_csv)
        self.donor_key = donor_key
        self.donor_split_column = donor_split_column
        self.donor_train_label = donor_train_label
        self.donor_validation_label = donor_validation_label
        self.donor_test_label = donor_test_label
        self.outer_train_label = outer_train_label
        self.inner_val_fraction = float(inner_val_fraction)
        self.feature_contract_path = (
            None if feature_contract_path is None else Path(feature_contract_path)
        )
        self.feature_contract: dict[str, Any] | None = None
        self.seed = seed
        self.num_workers = num_workers
        self.n_genes: int | None = None
        self.n_peaks: int | None = None
        self.num_cell_types: int | None = None
        self.cell_type_mapping: dict[str, int] = {}
        self.split_info: dict[str, Any] = {}

        if self.donor_split_csv is None:
            fractions = np.asarray(
                [self.train_fraction, self.val_fraction, self.test_fraction],
                dtype=np.float64,
            )
            if np.any(fractions <= 0) or not np.isclose(fractions.sum(), 1.0):
                raise ValueError(
                    "train_fraction, val_fraction, and test_fraction must be "
                    f"positive and sum to 1.0, got {fractions.tolist()}"
                )
        elif self.donor_split_column is None and not 0.0 < self.inner_val_fraction < 1.0:
            raise ValueError("inner_val_fraction must be between 0 and 1")

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _create_donor_split(
        self,
        donor_ids: np.ndarray,
        n_genes: int,
        n_peaks: int,
    ) -> tuple[list[int], list[int], list[int], dict[str, Any]]:
        split_csv = self.donor_split_csv
        if split_csv is None or not split_csv.exists():
            raise FileNotFoundError(f"Donor split CSV not found: {split_csv}")
        with split_csv.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        split_column = self.donor_split_column or "split"
        required = {self.donor_key, split_column}
        if not rows or not required.issubset(rows[0]):
            raise ValueError(f"Donor split CSV must contain columns {sorted(required)}")
        if len({row[self.donor_key] for row in rows}) != len(rows):
            raise ValueError(f"Donor split CSV contains duplicate {self.donor_key} values")

        if self.donor_split_column is not None:
            label_sets = {
                "train": {
                    row[self.donor_key]
                    for row in rows
                    if row[split_column] == self.donor_train_label
                },
                "validation": {
                    row[self.donor_key]
                    for row in rows
                    if row[split_column] == self.donor_validation_label
                },
                "test": {
                    row[self.donor_key]
                    for row in rows
                    if row[split_column] == self.donor_test_label
                },
            }
            if any(not values for values in label_sets.values()):
                raise ValueError(
                    f"Frozen donor split has an empty group: "
                    f"{ {key: len(value) for key, value in label_sets.items()} }"
                )
            if (
                label_sets["train"] & label_sets["validation"]
                or label_sets["train"] & label_sets["test"]
                or label_sets["validation"] & label_sets["test"]
            ):
                raise ValueError("Frozen donor groups overlap")
            observed_donors = set(map(str, donor_ids.tolist()))
            required_donors = set().union(*label_sets.values())
            missing = required_donors - observed_donors
            if missing:
                raise ValueError(
                    f"H5MU is missing {len(missing)} frozen-split donors; "
                    f"examples={sorted(missing)[:5]}"
                )
            donor_values = np.asarray(donor_ids, dtype=str)
            train_indices = np.flatnonzero(
                np.isin(donor_values, list(label_sets["train"]))
            ).astype(int).tolist()
            val_indices = np.flatnonzero(
                np.isin(donor_values, list(label_sets["validation"]))
            ).astype(int).tolist()
            test_indices = np.flatnonzero(
                np.isin(donor_values, list(label_sets["test"]))
            ).astype(int).tolist()
            excluded_indices = np.flatnonzero(
                ~np.isin(donor_values, list(required_donors))
            ).astype(int).tolist()
            manifest = {
                "version": 3,
                "dataset": str(self.h5mu_path.resolve()),
                "seed": self.seed,
                "split_strategy": "frozen_donor_train_validation_sealed_test",
                "donor_key": self.donor_key,
                "split_column": split_column,
                "train_label": self.donor_train_label,
                "validation_label": self.donor_validation_label,
                "test_label": self.donor_test_label,
                "outer_split_csv": str(split_csv.resolve()),
                "outer_split_sha256": self._sha256(split_csv),
                "fractions": {"train": None, "validation": None, "test": None},
                "n_total": len(donor_ids),
                "n_train": len(train_indices),
                "n_val": len(val_indices),
                "n_test": len(test_indices),
                "n_excluded": len(excluded_indices),
                "n_rna_features": n_genes,
                "n_atac_features": n_peaks,
                "n_train_donors": len(label_sets["train"]),
                "n_validation_donors": len(label_sets["validation"]),
                "n_test_donors": len(label_sets["test"]),
                "train_donors": sorted(label_sets["train"]),
                "validation_donors": sorted(label_sets["validation"]),
                "test_donors": sorted(label_sets["test"]),
                "cell_sampling_applied": False,
                "feature_sampling_applied": self.feature_contract_path is not None,
                "feature_contract": (
                    None
                    if self.feature_contract_path is None
                    else str(self.feature_contract_path.resolve())
                ),
                "feature_contract_sha256": (
                    None
                    if self.feature_contract_path is None
                    else self._sha256(self.feature_contract_path)
                ),
                "test_is_validation": False,
                "train_indices": train_indices,
                "val_indices": val_indices,
                "test_indices": test_indices,
                "excluded_indices": excluded_indices,
            }
            return train_indices, val_indices, test_indices, manifest

        outer_train_rows = [row for row in rows if row["split"] == self.outer_train_label]
        outer_train_donors = {row[self.donor_key] for row in outer_train_rows}
        outer_excluded_donors = {row[self.donor_key] for row in rows} - outer_train_donors
        observed_donors = set(map(str, donor_ids.tolist()))
        missing_train = outer_train_donors - observed_donors
        if missing_train:
            raise ValueError(
                f"Capped training H5MU is missing {len(missing_train)} outer-train donors; "
                f"examples={sorted(missing_train)[:5]}"
            )

        strata: dict[str, list[str]] = {}
        for row in outer_train_rows:
            stratum = row.get("age_bin") or "all"
            strata.setdefault(stratum, []).append(row[self.donor_key])
        rng = np.random.default_rng(self.seed)
        inner_val_donors: set[str] = set()
        for stratum in sorted(strata):
            values = np.asarray(sorted(strata[stratum]), dtype=object)
            rng.shuffle(values)
            n_val = max(1, int(round(len(values) * self.inner_val_fraction)))
            if n_val >= len(values) and len(values) > 1:
                n_val = len(values) - 1
            inner_val_donors.update(map(str, values[:n_val].tolist()))
        inner_train_donors = outer_train_donors - inner_val_donors
        if not inner_train_donors or not inner_val_donors:
            raise ValueError("Donor-level inner train/validation split is empty")

        donor_ids_str = np.asarray(donor_ids, dtype=str)
        train_indices = np.flatnonzero(np.isin(donor_ids_str, list(inner_train_donors))).astype(int).tolist()
        val_indices = np.flatnonzero(np.isin(donor_ids_str, list(inner_val_donors))).astype(int).tolist()
        excluded_indices = np.flatnonzero(~np.isin(donor_ids_str, list(outer_train_donors))).astype(int).tolist()
        if not train_indices or not val_indices:
            raise ValueError("No cells were assigned to donor-level train or validation")
        manifest = {
            "version": 2,
            "dataset": str(self.h5mu_path.resolve()),
            "seed": self.seed,
            "split_strategy": "outer_train_only_donor_stratified_inner_validation",
            "donor_key": self.donor_key,
            "outer_train_label": self.outer_train_label,
            "outer_split_csv": str(split_csv.resolve()),
            "outer_split_sha256": self._sha256(split_csv),
            "inner_val_fraction": self.inner_val_fraction,
            "fractions": {
                "inner_train": 1.0 - self.inner_val_fraction,
                "inner_validation": self.inner_val_fraction,
                "test": "validation_alias",
            },
            "n_total": len(donor_ids),
            "n_train": len(train_indices),
            "n_val": len(val_indices),
            "n_test": len(val_indices),
            "n_excluded": len(excluded_indices),
            "n_rna_features": n_genes,
            "n_atac_features": n_peaks,
            "n_outer_train_donors": len(outer_train_donors),
            "n_outer_excluded_donors": len(outer_excluded_donors),
            "n_inner_train_donors": len(inner_train_donors),
            "n_inner_val_donors": len(inner_val_donors),
            "outer_train_donors": sorted(outer_train_donors),
            "outer_excluded_donors": sorted(outer_excluded_donors),
            "inner_train_donors": sorted(inner_train_donors),
            "inner_val_donors": sorted(inner_val_donors),
            "cell_sampling_applied": False,
            "feature_sampling_applied": False,
            "test_is_validation": True,
            "train_indices": train_indices,
            "val_indices": val_indices,
            "test_indices": val_indices,
            "excluded_indices": excluded_indices,
        }
        return train_indices, val_indices, val_indices, manifest

    def _load_or_create_split(
        self,
        *,
        n_cells: int,
        n_genes: int,
        n_peaks: int,
        donor_ids: np.ndarray | None = None,
    ) -> tuple[list[int], list[int], list[int]]:
        manifest_path = self.split_manifest_path
        if manifest_path is not None and manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            expected = {
                "n_total": n_cells,
                "n_rna_features": n_genes,
                "n_atac_features": n_peaks,
                "seed": self.seed,
            }
            observed = {key: manifest.get(key) for key in expected}
            if observed != expected:
                raise ValueError(
                    f"Split manifest does not match the current dataset: "
                    f"{observed} != {expected}"
                )
            if self.donor_split_csv is not None:
                if self.donor_split_column is None:
                    donor_expected = {
                        "split_strategy": "outer_train_only_donor_stratified_inner_validation",
                        "outer_split_sha256": self._sha256(self.donor_split_csv),
                        "donor_key": self.donor_key,
                        "outer_train_label": self.outer_train_label,
                        "inner_val_fraction": self.inner_val_fraction,
                    }
                else:
                    donor_expected = {
                        "split_strategy": "frozen_donor_train_validation_sealed_test",
                        "outer_split_sha256": self._sha256(self.donor_split_csv),
                        "donor_key": self.donor_key,
                        "split_column": self.donor_split_column,
                        "train_label": self.donor_train_label,
                        "validation_label": self.donor_validation_label,
                        "test_label": self.donor_test_label,
                    }
                donor_observed = {key: manifest.get(key) for key in donor_expected}
                if donor_observed != donor_expected:
                    raise ValueError(
                        "Existing split manifest violates the donor-isolation contract: "
                        f"{donor_observed} != {donor_expected}"
                    )
            train_indices = [int(x) for x in manifest["train_indices"]]
            val_indices = [int(x) for x in manifest["val_indices"]]
            test_indices = [int(x) for x in manifest["test_indices"]]
            if self.donor_split_csv is not None:
                if donor_ids is None:
                    raise ValueError("donor_ids are required to validate donor-level indices")
                donor_values = np.asarray(donor_ids, dtype=str)
                observed_train_donors = set(donor_values[train_indices].tolist())
                observed_val_donors = set(donor_values[val_indices].tolist())
                if self.donor_split_column is None:
                    if observed_train_donors != set(manifest["inner_train_donors"]):
                        raise ValueError(
                            "Stored training indices no longer match inner_train_donors"
                        )
                    if observed_val_donors != set(manifest["inner_val_donors"]):
                        raise ValueError(
                            "Stored validation indices no longer match inner_val_donors"
                        )
                else:
                    observed_test_donors = set(donor_values[test_indices].tolist())
                    if observed_train_donors != set(manifest["train_donors"]):
                        raise ValueError("Stored training indices no longer match train_donors")
                    if observed_val_donors != set(manifest["validation_donors"]):
                        raise ValueError(
                            "Stored validation indices no longer match validation_donors"
                        )
                    if observed_test_donors != set(manifest["test_donors"]):
                        raise ValueError("Stored test indices no longer match test_donors")
        else:
            if self.donor_split_csv is not None:
                if donor_ids is None:
                    raise ValueError("donor_ids are required for donor-level splitting")
                train_indices, val_indices, test_indices, manifest = self._create_donor_split(
                    donor_ids, n_genes, n_peaks
                )
                if manifest_path is not None:
                    manifest_path.parent.mkdir(parents=True, exist_ok=True)
                    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            else:
                rng = np.random.default_rng(self.seed)
                permutation = rng.permutation(n_cells)
                n_train = int(round(n_cells * self.train_fraction))
                n_val = int(round(n_cells * self.val_fraction))
                n_test = n_cells - n_train - n_val
                if min(n_train, n_val, n_test) <= 0:
                    raise ValueError(
                        f"Dataset with {n_cells} cells is too small for "
                        f"{self.train_fraction}/{self.val_fraction}/{self.test_fraction}"
                    )
                train_indices = permutation[:n_train].astype(int).tolist()
                val_indices = permutation[n_train : n_train + n_val].astype(int).tolist()
                test_indices = permutation[n_train + n_val :].astype(int).tolist()

                manifest = {
                    "version": 1,
                    "dataset": str(self.h5mu_path.resolve()),
                    "seed": self.seed,
                    "fractions": {
                        "train": self.train_fraction,
                        "validation": self.val_fraction,
                        "test": self.test_fraction,
                    },
                    "n_total": n_cells,
                    "n_train": len(train_indices),
                    "n_val": len(val_indices),
                    "n_test": len(test_indices),
                    "n_rna_features": n_genes,
                    "n_atac_features": n_peaks,
                    "cell_sampling_applied": False,
                    "feature_sampling_applied": False,
                    "test_is_validation": False,
                    "train_indices": train_indices,
                    "val_indices": val_indices,
                    "test_indices": test_indices,
                }
                if manifest_path is not None:
                    manifest_path.parent.mkdir(parents=True, exist_ok=True)
                    manifest_path.write_text(
                        json.dumps(manifest, indent=2),
                        encoding="utf-8",
                    )

        index_sets = [
            set(train_indices),
            set(val_indices),
            set(test_indices),
        ]
        test_is_validation = bool(manifest.get("test_is_validation", False))
        if index_sets[0] & index_sets[1] or index_sets[0] & index_sets[2]:
            raise ValueError("Train, validation, and test split indices overlap")
        if not test_is_validation and index_sets[1] & index_sets[2]:
            raise ValueError("Validation and test split indices overlap")
        if test_is_validation and index_sets[1] != index_sets[2]:
            raise ValueError("test_is_validation requires identical validation and test indices")
        combined = index_sets[0] | index_sets[1] | index_sets[2]
        expected_covered = set(range(n_cells)) - set(map(int, manifest.get("excluded_indices", [])))
        if combined != expected_covered:
            raise ValueError("Split indices do not cover the intended non-excluded cells")

        self.split_info = {
            "dataset": str(self.h5mu_path.resolve()),
            "split_manifest": None if manifest_path is None else str(manifest_path),
            "seed": self.seed,
            "fractions": manifest.get(
                "fractions",
                {
                    "train": self.train_fraction,
                    "validation": self.val_fraction,
                    "test": self.test_fraction,
                },
            ),
            "n_total": n_cells,
            "n_train": len(train_indices),
            "n_val": len(val_indices),
            "n_test": len(test_indices),
            "n_rna_features": n_genes,
            "n_atac_features": n_peaks,
            "cell_sampling_applied": False,
            "feature_sampling_applied": self.feature_contract_path is not None,
            "feature_contract": (
                None
                if self.feature_contract_path is None
                else str(self.feature_contract_path.resolve())
            ),
            "feature_contract_sha256": (
                None
                if self.feature_contract_path is None
                else self._sha256(self.feature_contract_path)
            ),
            "test_is_validation": test_is_validation,
            "split_strategy": manifest.get("split_strategy", "random_cell"),
            "n_excluded": int(manifest.get("n_excluded", 0)),
            "n_outer_train_donors": manifest.get("n_outer_train_donors"),
            "n_outer_excluded_donors": manifest.get("n_outer_excluded_donors"),
            "n_inner_train_donors": manifest.get("n_inner_train_donors"),
            "n_inner_val_donors": manifest.get("n_inner_val_donors"),
            "outer_split_csv": manifest.get("outer_split_csv"),
            "outer_split_sha256": manifest.get("outer_split_sha256"),
        }
        return train_indices, val_indices, test_indices

    def setup(self, stage: str | None = None) -> None:
        if not self.h5mu_path.exists():
            raise FileNotFoundError(f"MuData file not found: {self.h5mu_path}")
        feature_contract = None
        if self.feature_contract_path is not None:
            if not self.feature_contract_path.is_file():
                raise FileNotFoundError(
                    f"Feature contract not found: {self.feature_contract_path}"
                )
            feature_contract = json.loads(
                self.feature_contract_path.read_text(encoding="utf-8")
            )
            required = {
                "rna_feature_names",
                "rna_feature_indices",
                "atac_feature_names",
                "atac_feature_indices",
                "cell_types",
            }
            if not required.issubset(feature_contract):
                raise ValueError(
                    f"Feature contract is missing {sorted(required - set(feature_contract))}"
                )
            self.feature_contract = feature_contract
        lazy_setting = os.environ.get("SCFLOW_H5MU_LAZY", "auto").strip().lower()
        if lazy_setting not in {"auto", "0", "1"}:
            raise ValueError("SCFLOW_H5MU_LAZY must be one of auto, 0, or 1")
        use_lazy = lazy_setting == "1" or (
            lazy_setting == "auto" and self.h5mu_path.stat().st_size >= 32 * 1024**3
        )

        if use_lazy:
            import h5py

            rna = H5CSRRowMatrix(self.h5mu_path, "mod/rna/X")
            atac = H5CSRRowMatrix(self.h5mu_path, "mod/atac/X")
            if rna.shape[0] != atac.shape[0]:
                raise ValueError("RNA and ATAC H5MU modalities have unequal cell counts")
            with h5py.File(self.h5mu_path, "r") as handle:
                cell_type_group = handle.get("mod/rna/obs/cell_type")
                if cell_type_group is not None and "codes" in cell_type_group:
                    categories = [
                        x.decode("utf-8") if isinstance(x, bytes) else str(x)
                        for x in cell_type_group["categories"][:]
                    ]
                    cell_type_codes = np.asarray(cell_type_group["codes"][:], dtype=np.int64)
                    cell_type_values = np.asarray(categories, dtype=object)[cell_type_codes]
                    cell_type_encoded = cell_type_codes
                    self.cell_type_mapping = {
                        label: int(index) for index, label in enumerate(categories)
                    }
                    self.num_cell_types = int(len(categories))
                else:
                    cell_type_encoded = None
                    cell_type_values = None
                    self.num_cell_types = None
                    self.cell_type_mapping = {}
                donor_group = handle.get(f"mod/rna/obs/{self.donor_key}")
                if donor_group is not None and hasattr(donor_group, "keys") and "codes" in donor_group:
                    donor_categories = np.asarray([
                        x.decode("utf-8") if isinstance(x, bytes) else str(x)
                        for x in donor_group["categories"][:]
                    ])
                    donor_ids = donor_categories[np.asarray(donor_group["codes"][:], dtype=np.int64)]
                elif donor_group is not None:
                    donor_ids = np.asarray([
                        x.decode("utf-8") if isinstance(x, bytes) else str(x)
                        for x in donor_group[:]
                    ])
                else:
                    donor_ids = None
                rna_index_name = handle["mod/rna/var"].attrs["_index"]
                atac_index_name = handle["mod/atac/var"].attrs["_index"]
                rna_feature_names = np.asarray(
                    [
                        x.decode("utf-8") if isinstance(x, bytes) else str(x)
                        for x in handle[f"mod/rna/var/{rna_index_name}"][:]
                    ],
                    dtype=object,
                )
                atac_feature_names = np.asarray(
                    [
                        x.decode("utf-8") if isinstance(x, bytes) else str(x)
                        for x in handle[f"mod/atac/var/{atac_index_name}"][:]
                    ],
                    dtype=object,
                )
        else:
            import muon as mu

            mdata = mu.read(self.h5mu_path)
            if "rna" not in mdata.mod or "atac" not in mdata.mod:
                raise ValueError("Expected MuData modalities named 'rna' and 'atac'")
            rna = mdata.mod["rna"].X
            atac = mdata.mod["atac"].X
            cell_type_encoded = None
            if "cell_type" in mdata.mod["rna"].obs:
                cell_type_values = np.asarray(mdata.mod["rna"].obs["cell_type"].astype(str).values)
                classes, inverse = np.unique(cell_type_values, return_inverse=True)
                self.cell_type_mapping = {str(label): int(i) for i, label in enumerate(classes)}
                self.num_cell_types = int(len(classes))
                cell_type_encoded = inverse.astype(np.int64)
            else:
                self.num_cell_types = None
                self.cell_type_mapping = {}
                cell_type_values = None
            donor_ids = (
                np.asarray(mdata.mod["rna"].obs[self.donor_key].astype(str).values)
                if self.donor_key in mdata.mod["rna"].obs
                else None
            )
            rna_feature_names = np.asarray(mdata.mod["rna"].var_names.astype(str), dtype=object)
            atac_feature_names = np.asarray(mdata.mod["atac"].var_names.astype(str), dtype=object)

        if feature_contract is not None:
            rna_columns = np.asarray(feature_contract["rna_feature_indices"], dtype=np.int64)
            atac_columns = np.asarray(feature_contract["atac_feature_indices"], dtype=np.int64)
            expected_rna = np.asarray(feature_contract["rna_feature_names"], dtype=str)
            expected_atac = np.asarray(feature_contract["atac_feature_names"], dtype=str)
            if rna_feature_names[rna_columns].astype(str).tolist() != expected_rna.tolist():
                raise ValueError("RNA physical feature order violates the frozen contract")
            if atac_feature_names[atac_columns].astype(str).tolist() != expected_atac.tolist():
                raise ValueError("ATAC physical feature order violates the frozen contract")
            if donor_ids is None or cell_type_values is None:
                raise ValueError(
                    "Feature-contracted training requires donor and cell-type metadata"
                )
            allowed_cell_types = [str(value) for value in feature_contract["cell_types"]]
            allowed_donors = None
            if self.donor_split_csv is not None:
                with self.donor_split_csv.open(newline="", encoding="utf-8") as handle:
                    split_rows = list(csv.DictReader(handle))
                allowed_donors = {str(row[self.donor_key]) for row in split_rows}
            keep = np.isin(np.asarray(cell_type_values, dtype=str), allowed_cell_types)
            if allowed_donors is not None:
                keep &= np.isin(np.asarray(donor_ids, dtype=str), list(allowed_donors))
            selected_rows = np.flatnonzero(keep).astype(np.int64)
            if not len(selected_rows):
                raise ValueError("No cells remain after applying the frozen feature contract")
            rna = MatrixView(rna, rows=selected_rows, columns=rna_columns)
            atac = MatrixView(atac, rows=selected_rows, columns=atac_columns)
            donor_ids = np.asarray(donor_ids, dtype=str)[selected_rows]
            cell_type_values = np.asarray(cell_type_values, dtype=str)[selected_rows]
            self.cell_type_mapping = {
                label: index for index, label in enumerate(allowed_cell_types)
            }
            self.num_cell_types = len(allowed_cell_types)
            cell_type_encoded = np.asarray(
                [self.cell_type_mapping[value] for value in cell_type_values],
                dtype=np.int64,
            )
        dataset = MultimodalMuDataDataset(
            rna,
            atac,
            rna_seq_len=self.rna_seq_len,
            atac_seq_len=self.atac_seq_len,
            sample_features=self.sample_features,
            rna_sample_features=self.rna_sample_features,
            atac_sample_features=self.atac_sample_features,
            rna_encoder_seq_len=self.rna_encoder_seq_len,
            rna_encoder_sample_features=self.rna_encoder_sample_features,
            atac_encoder_seq_len=self.atac_encoder_seq_len,
            atac_encoder_sample_features=self.atac_encoder_sample_features,
            seed=self.seed,
            cell_type=cell_type_encoded,
        )
        self.n_genes = dataset.n_genes
        self.n_peaks = dataset.n_peaks

        train_indices, val_indices, test_indices = self._load_or_create_split(
            n_cells=len(dataset),
            n_genes=dataset.n_genes,
            n_peaks=dataset.n_peaks,
            donor_ids=donor_ids,
        )
        self.train_dataset = torch.utils.data.Subset(dataset, train_indices)
        self.val_dataset = torch.utils.data.Subset(dataset, val_indices)
        self.test_dataset = torch.utils.data.Subset(dataset, test_indices)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.test_batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.test_batch_size, shuffle=False, num_workers=self.num_workers)


class DataModule(LightningDataModule):
    def __init__(
        self,
        train_adata_path: Path,
        test_adata_path: Path,
        adata_attr: str,
        adata_key: str | None,
        vocabulary_encoder: VocabularyEncoderSimplified,
        val_as_test: bool = True,
        data_path: Path | None = None,
        allow_missing_train: bool = False,
        batch_size: int = 256,
        test_batch_size: int = 256,
        num_workers: int = 4,
        seed: int = 42,
        prefetch_factor: int = 4,
        persistent_workers: bool = True,
        drop_last_indices: bool = False,
        drop_incomplete_batch: bool = True,
        sample_genes: Literal["random", "weighted", "expressed", "expressed_zero", "none"] = "none",
        genes_seq_len: int = 100,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)

        assert isinstance(vocabulary_encoder, VocabularyEncoderSimplified)

        self.vocabulary_encoder = vocabulary_encoder
        self.adata_attr = adata_attr
        self.adata_key = adata_key
        self.train_adata_path = Path(train_adata_path) if train_adata_path is not None else None
        self.test_adata_path = Path(test_adata_path) if test_adata_path is not None else None
        self.val_as_test = val_as_test
        self.allow_missing_train = allow_missing_train
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.prefetch_factor = prefetch_factor
        self.persistent_workers = persistent_workers
        self.drop_last_indices = drop_last_indices
        self.drop_incomplete_batch = drop_incomplete_batch
        self.sample_genes = sample_genes
        self.genes_seq_len = genes_seq_len
        self.data_path = data_path

        # this should be done in `setup`, but we need to read the metadata to get the number of cells
        # this won't work in distributed training
        if "adata_0.h5ad" in str(self.train_adata_path):
            # Read metadata from folder
            metadata_path = os.path.join(self.train_adata_path.parent, "metadata.json")
            with open(metadata_path) as f:
                self.train_metadata = json.load(f)
            self.n_cells = self.train_metadata["n_cells"]
        elif self.data_path is not None:
            _, self.n_cells, _ = get_tissue_adata_files(self.data_path, "train")
            self.train_metadata = None
        elif self.train_adata_path is not None:
            if self.train_adata_path.exists():
                train_adata = ad.read_h5ad(self.train_adata_path)
                self.n_cells = train_adata.n_obs
                self.train_metadata = None
            elif self.allow_missing_train:
                logger.info("Train adata path missing; continuing in predict-only mode")
                self.n_cells = 0
                self.train_metadata = None
            else:
                raise FileNotFoundError(f"Train adata path not found: {self.train_adata_path}")
        else:
            logger.info("No train adata path provided, make sure to set up datamodule for inference")
            self.n_cells = 0

        self._adata_inference = None

    @property
    def adata_inference(self):
        return self._adata_inference

    @adata_inference.setter
    def adata_inference(self, adata: AnnData):
        if hasattr(self.vocabulary_encoder, "genes") and self.vocabulary_encoder.genes is not None:
            available_genes = set(adata.var_names)
            required_genes = set(self.vocabulary_encoder.genes)

            missing_genes = required_genes - available_genes
            kept = available_genes & required_genes
            logger.info(f"Filtering genes to encoder vocabulary: kept={len(kept)}, missing={len(missing_genes)}")

            genes = [g for g in adata.var_names if g in required_genes]
            adata = adata[:, genes].copy()
        self._adata_inference = adata

    def get_library_size(self, x: np.ndarray) -> np.ndarray:
        """Compute library size (total counts per cell)."""
        return x.sum(axis=1, keepdims=True)

    def _setup_prediction_only(self):
        """Set up prediction dataset using adata_inference, skipping all training/validation setup."""
        # Recompute labels for predict based on columns actually present in adata_inference
        predict_labels = {}
        if self.vocabulary_encoder.labels is not None and isinstance(self.vocabulary_encoder.labels, dict):
            present_label_keys = [
                label for label in self.vocabulary_encoder.labels.keys() if label in self.adata_inference.obs
            ]
            missing_label_keys = [
                label for label in self.vocabulary_encoder.labels.keys() if label not in self.adata_inference.obs
            ]
            if missing_label_keys:
                logger.info(f"[predict] Skipping missing label columns in adata_inference: {missing_label_keys}")
            predict_labels = {
                label: AnnDataField(
                    attr="obs",
                    key=label,
                    convert_fn=lambda x, label=label: self.vocabulary_encoder.encode_metadata(x, label=label),
                )
                for label in present_label_keys
            }

        gene_tokens_transform = partial(
            tokenize_cells,
            encoder=self.vocabulary_encoder,
        )
        predict_batch_keys = {
            ModelEnum.COUNTS.value: AnnDataField(
                attr=self.adata_attr, key=self.adata_key, convert_fn=lambda x: x.toarray()
            ),
            ModelEnum.GENES.value: AnnDataFieldWithVarNames(
                attr=self.adata_attr,
                convert_fn=cast(
                    Any,
                    lambda data_dict: gene_tokens_transform(
                        cell=data_dict["data"],
                        var_names=data_dict["var_names"],
                        genes_seq_len=self.genes_seq_len,
                        sample_genes=self.sample_genes,
                    ),
                ),
            ),
            ModelEnum.LIBRARY_SIZE.value: AnnDataField(
                attr=self.adata_attr,
                key=self.adata_key,
                convert_fn=lambda x: self.get_library_size(x.toarray()),
            ),
            **predict_labels,
        }

        dataset = partial(
            IterableDistributedAnnDataCollectionDataset,
            shuffle_seed=self.seed,
            worker_seed=None,
        )

        self.predict_dataset = dataset(
            batch_keys=predict_batch_keys,  # type: ignore
            dadc=self.adata_inference,
            shuffle=False,
            shuffle_seed=False,
            batch_size=self.test_batch_size,
            drop_last_indices=False,
            drop_incomplete_batch=False,
        )

    def _setup_prediction_from_test(self):
        """Set up prediction dataset using test data, skipping all training/validation setup."""
        labels = {}
        if self.vocabulary_encoder.labels is not None and isinstance(self.vocabulary_encoder.labels, dict):
            labels = {
                label: AnnDataField(
                    attr="obs",
                    key=label,
                    convert_fn=lambda x, label=label: self.vocabulary_encoder.encode_metadata(x, label=label),
                )
                for label in self.vocabulary_encoder.labels.keys()
            }

        gene_tokens_transform = partial(
            tokenize_cells,
            encoder=self.vocabulary_encoder,
        )
        test_batch_keys = {
            ModelEnum.COUNTS.value: AnnDataField(
                attr=self.adata_attr, key=self.adata_key, convert_fn=lambda x: x.toarray()
            ),
            ModelEnum.GENES.value: AnnDataFieldWithVarNames(
                attr=self.adata_attr,
                convert_fn=cast(
                    Any,
                    lambda data_dict: gene_tokens_transform(
                        cell=data_dict["data"],
                        var_names=data_dict["var_names"],
                        genes_seq_len=self.genes_seq_len,
                        sample_genes=self.sample_genes,
                    ),
                ),
            ),
            ModelEnum.LIBRARY_SIZE.value: AnnDataField(
                attr=self.adata_attr,
                key=self.adata_key,
                convert_fn=lambda x: self.get_library_size(x.toarray()),
            ),
            **labels,
        }

        dataset = partial(
            IterableDistributedAnnDataCollectionDataset,
            shuffle_seed=self.seed,
            worker_seed=None,
        )

        self.predict_dataset = dataset(
            batch_keys=test_batch_keys,  # type: ignore
            dadc=self.test_adata,
            shuffle=False,
            shuffle_seed=False,
            batch_size=self.test_batch_size,
            drop_last_indices=False,
            drop_incomplete_batch=False,
        )

    def setup(self, stage: str | None = None):
        if stage == "predict":
            if self.adata_inference is not None:
                if not isinstance(self.adata_inference, AnnData):
                    raise TypeError("adata_inference must be an AnnData object")
                self._setup_prediction_only()
                return
            else:
                if self.test_adata_path is None:
                    raise ValueError("test_adata_path must be set for predict when adata_inference is not provided")
                if not hasattr(self, "test_adata") or self.test_adata is None:
                    self.test_adata = ad.read_h5ad(self.test_adata_path)
                self._setup_prediction_from_test()
                return

        if "adata_0.h5ad" in str(self.train_adata_path):
            logger.info("Using train_val_split_list from sharded train files")
            # Read metadata from folder
            train_metadata_path = os.path.join(self.train_adata_path.parent, "metadata.json")
            with open(train_metadata_path) as f:
                self.train_metadata = json.load(f)
            test_metadata_path = os.path.join(self.test_adata_path.parent, "metadata.json")
            with open(test_metadata_path) as f:
                self.test_metadata = json.load(f)
            self.train_files = sort_h5ad_files(self.train_adata_path.parent)
            self.test_files = sort_h5ad_files(self.test_adata_path.parent)
        elif self.data_path is not None:
            self.train_files, n_cells_train, shard_size_train = get_tissue_adata_files(self.data_path, "train")
            self.test_files, n_cells_val, shard_size_val = get_tissue_adata_files(self.data_path, "test")
            self.train_metadata = {
                "n_cells": n_cells_train,
                "shard_size": shard_size_train,
                "last_shard_size": shard_size_train,
            }
            self.test_metadata = {
                "n_cells": n_cells_val,
                "shard_size": shard_size_val,
                "last_shard_size": shard_size_val,
            }
        else:
            self.train_adata = ad.read_h5ad(self.train_adata_path)
            self.train_metadata = None
            self.test_metadata = None
            self.test_adata = ad.read_h5ad(self.test_adata_path)

        if self.val_as_test:
            if self.train_metadata is None:
                self.val_adata = self.test_adata
                self.val_ann_collection = self.val_adata
                self.train_ann_collection = self.train_adata
                self.test_ann_collection = self.test_adata

            else:
                self.train_ann_collection = DistributedAnnDataCollection(
                    self.train_files,
                    shard_size=self.train_metadata["shard_size"],
                    last_shard_size=self.train_metadata["last_shard_size"],
                    indices_strict=False,
                    max_cache_size=10,
                )
                self.val_ann_collection = DistributedAnnDataCollection(
                    self.test_files,
                    shard_size=self.test_metadata["shard_size"],
                    last_shard_size=self.test_metadata["last_shard_size"],
                    indices_strict=False,
                    max_cache_size=10,
                )
                self.test_ann_collection = DistributedAnnDataCollection(
                    self.test_files,
                    shard_size=self.test_metadata["shard_size"],
                    last_shard_size=self.test_metadata["last_shard_size"],
                    indices_strict=False,
                    max_cache_size=10,
                )
        else:
            # Split train files into train and validation sets
            if self.train_metadata is None:
                rng = np.random.RandomState(self.seed)
                n_cells = self.train_adata.n_obs
                n_val_cells = int(0.1 * n_cells)
                indices = np.arange(n_cells)
                resample_indices = rng.permutation(indices)

                train_indices = resample_indices[:-n_val_cells]
                val_indices = resample_indices[-n_val_cells:]

                self.val_adata = self.train_adata[val_indices]
                self.train_adata = self.train_adata[train_indices]
                self.train_ann_collection = self.train_adata
                self.val_ann_collection = self.val_adata
                self.test_ann_collection = self.test_adata
            else:
                logger.info("Using train_val_split_list from sharded train files")
                train_indices, val_indices = train_val_split_list(self.train_files, self.seed)
                self.val_files = [self.train_files[i] for i in val_indices]
                self.train_files = [self.train_files[i] for i in train_indices]
                self.train_ann_collection = DistributedAnnDataCollection(
                    self.train_files,
                    shard_size=self.train_metadata["shard_size"],
                    last_shard_size=self.train_metadata["last_shard_size"],
                    indices_strict=False,
                    max_cache_size=10,
                )
                self.val_ann_collection = DistributedAnnDataCollection(
                    self.val_files,
                    shard_size=self.train_metadata["shard_size"],
                    last_shard_size=self.test_metadata["last_shard_size"]
                    if self.val_as_test
                    else self.train_metadata["shard_size"],
                    indices_strict=False,
                    max_cache_size=10,
                )
                self.test_ann_collection = DistributedAnnDataCollection(
                    self.test_files,
                    shard_size=self.test_metadata["shard_size"],
                    last_shard_size=self.test_metadata["last_shard_size"],
                    indices_strict=False,
                    max_cache_size=10,
                )

        labels = {}
        if self.vocabulary_encoder.labels is not None and isinstance(self.vocabulary_encoder.labels, dict):
            labels = {
                label: AnnDataField(
                    attr="obs",
                    key=label,
                    convert_fn=lambda x, label=label: self.vocabulary_encoder.encode_metadata(x, label=label),
                )
                for label in self.vocabulary_encoder.labels.keys()
            }

        gene_tokens_transform = partial(
            tokenize_cells,
            encoder=self.vocabulary_encoder,
        )

        train_batch_keys = {
            ModelEnum.GENES.value: AnnDataFieldWithVarNames(
                attr=self.adata_attr,
                key=self.adata_key,
                convert_fn=cast(
                    Any,
                    lambda data_dict: gene_tokens_transform(
                        cell=data_dict["data"],
                        genes_seq_len=self.genes_seq_len,
                        var_names=data_dict["var_names"],
                        sample_genes=self.sample_genes,
                    ),
                ),
            ),
            **labels,
        }
        val_batch_keys = {
            ModelEnum.GENES.value: AnnDataFieldWithVarNames(
                attr=self.adata_attr,
                key=self.adata_key,
                convert_fn=cast(
                    Any,
                    lambda data_dict: gene_tokens_transform(
                        cell=data_dict["data"],
                        genes_seq_len=self.genes_seq_len,
                        var_names=data_dict["var_names"],
                        sample_genes=self.sample_genes,
                    ),
                ),
            ),
            **labels,
        }
        test_batch_keys = {
            ModelEnum.GENES.value: AnnDataFieldWithVarNames(
                attr=self.adata_attr,
                key=self.adata_key,
                convert_fn=cast(
                    Any,
                    lambda data_dict: gene_tokens_transform(
                        cell=data_dict["data"],
                        genes_seq_len=self.genes_seq_len,
                        var_names=data_dict["var_names"],
                        sample_genes=self.sample_genes,
                    ),
                ),
            ),
            **labels,
        }
        logger.info("Using IterableDistributedAnnDataCollectionDataset", stacklevel=2)

        dataset = partial(
            IterableDistributedAnnDataCollectionDataset,
            shuffle_seed=self.seed,
            worker_seed=None,
        )

        self.train_dataset = dataset(
            batch_keys=train_batch_keys,  # type: ignore
            dadc=self.train_ann_collection,
            shuffle=False,
            batch_size=self.batch_size,
            drop_last_indices=True,
            drop_incomplete_batch=True,
        )
        self.val_dataset = dataset(
            batch_keys=val_batch_keys,  # type: ignore
            dadc=self.val_ann_collection,
            shuffle=False,
            shuffle_seed=False,
            batch_size=self.test_batch_size,
            drop_last_indices=True,
            drop_incomplete_batch=True,
        )
        self.test_dataset = dataset(
            batch_keys=test_batch_keys,  # type: ignore
            dadc=self.test_ann_collection,
            shuffle=False,
            shuffle_seed=False,
            batch_size=self.test_batch_size,
            drop_last_indices=False,
            drop_incomplete_batch=False,
        )
        if stage == "predict":
            if self.adata_inference is not None:
                if not isinstance(self.adata_inference, AnnData):
                    raise TypeError("adata_inference must be an AnnData object")
                # Recompute labels for predict based on columns actually present in adata_inference
                predict_labels = {}
                if self.vocabulary_encoder.labels is not None and isinstance(self.vocabulary_encoder.labels, dict):
                    present_label_keys = [
                        label for label in self.vocabulary_encoder.labels.keys() if label in self.adata_inference.obs
                    ]
                    missing_label_keys = [
                        label
                        for label in self.vocabulary_encoder.labels.keys()
                        if label not in self.adata_inference.obs
                    ]
                    if missing_label_keys:
                        logger.info(
                            f"[predict] Skipping missing label columns in adata_inference: {missing_label_keys}"
                        )
                    predict_labels = {
                        label: AnnDataField(
                            attr="obs",
                            key=label,
                            convert_fn=lambda x, label=label: self.vocabulary_encoder.encode_metadata(x, label=label),
                        )
                        for label in present_label_keys
                    }

                predict_batch_keys = {
                    ModelEnum.GENES.value: AnnDataFieldWithVarNames(
                        attr="X",
                        convert_fn=cast(
                            Any,
                            lambda data_dict: gene_tokens_transform(
                                cell=data_dict["data"],
                                genes_seq_len=self.genes_seq_len,
                                var_names=data_dict["var_names"],
                                sample_genes="none",
                            ),
                        ),
                    ),
                    # propagate original obs_names for alignment downstream
                    "obs_names": AnnDataField(
                        attr="obs_names",
                        convert_fn=lambda x: np.asarray(x),
                    ),
                    **predict_labels,
                }
                self.predict_dataset = dataset(
                    batch_keys=predict_batch_keys,  # type: ignore
                    dadc=self.adata_inference,
                    shuffle=False,
                    shuffle_seed=False,
                    batch_size=self.batch_size,
                    drop_last_indices=False,
                    drop_incomplete_batch=False,
                )
            else:
                predict_batch_keys = deepcopy(test_batch_keys)
                self.predict_dataset = dataset(
                    batch_keys=predict_batch_keys,  # type: ignore
                    dadc=self.test_ann_collection,
                    shuffle=False,
                    shuffle_seed=False,
                    batch_size=self.batch_size,
                    drop_last_indices=False,
                    drop_incomplete_batch=False,
                )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            collate_fn=collate_fn,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            collate_fn=collate_fn,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            collate_fn=collate_fn,
            num_workers=self.num_workers,
            drop_last=False,
            pin_memory=True,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.predict_dataset,
            collate_fn=collate_fn,
            num_workers=self.num_workers,
            drop_last=False,
            pin_memory=True,
        )

    def collate_fn_annloader(
        self,
        batch,
        sample_genes: Literal["random", "weighted", "expressed", "expressed_zero", "none"],
        genes_seq_len: int,
    ):
        output = tokenize_cells(batch.X, batch.var_names, self.vocabulary_encoder, genes_seq_len, sample_genes)
        output.update(
            {
                k: self.vocabulary_encoder.encode_metadata(batch.obs[k].values, label=k)
                for k in self.vocabulary_encoder.labels
            }
        )
        output = tree_map(lambda x: x.detach().clone() if torch.is_tensor(x) else torch.tensor(x), output)
        return output


def collate_fn(
    batch: list[dict[str, dict[str, np.ndarray] | np.ndarray]],
) -> dict[str, dict[str, np.ndarray | torch.Tensor] | np.ndarray | torch.Tensor]:
    keys = batch[0].keys()
    collated_batch: dict[str, dict[str, np.ndarray | torch.Tensor] | np.ndarray | torch.Tensor] = {}
    if len(batch) > 1 and not all(keys == data.keys() for data in batch[1:]):
        raise ValueError("All dictionaries in the batch must have the same keys.")
    for key in keys:
        if key == ModelEnum.GENES.value:
            collated_batch[ModelEnum.COUNTS.value] = np.concatenate(
                [data[key][ModelEnum.COUNTS.value] for data in batch],
                axis=0,  # type: ignore
            )
            collated_batch[ModelEnum.GENES.value] = np.concatenate(
                [data[key][ModelEnum.GENES.value] for data in batch],
                axis=0,  # type: ignore
            )
            collated_batch[ModelEnum.LIBRARY_SIZE.value] = np.concatenate(
                [data[key][ModelEnum.LIBRARY_SIZE.value] for data in batch],
                axis=0,  # type: ignore
            )
            # Optional extras if provided by tokenizer
            if ModelEnum.GENES_SUBSET.value in batch[0][key]:  # type: ignore
                collated_batch[ModelEnum.GENES_SUBSET.value] = np.concatenate(
                    [data[key][ModelEnum.GENES_SUBSET.value] for data in batch],
                    axis=0,  # type: ignore
                )
            if ModelEnum.COUNTS_SUBSET.value in batch[0][key]:  # type: ignore
                collated_batch[ModelEnum.COUNTS_SUBSET.value] = np.concatenate(
                    [data[key][ModelEnum.COUNTS_SUBSET.value] for data in batch],
                    axis=0,  # type: ignore
                )
            continue
        if isinstance(batch[0][key], dict):
            subkeys = batch[0][key].keys()  # type: ignore
            if len(batch) > 1 and not all(subkeys == data[key].keys() for data in batch[1:]):  # type: ignore
                raise ValueError(f"All '{key}' sub-dictionaries in the batch must have the same subkeys.")
            # Concatenate all subkeys regardless of their suffix
            value = {
                subkey: np.concatenate([data[key][subkey] for data in batch], axis=0)
                for subkey in subkeys  # type: ignore
            }
        elif key.endswith("_g") or key.endswith("_categories"):
            # Check that all values are the same
            if len(batch) > 1:
                if not all(np.array_equal(batch[0][key], data[key]) for data in batch[1:]):
                    raise ValueError(f"All dictionaries in the batch must have the same {key}.")
            value = batch[0][key]
        else:
            value = np.concatenate([data[key] for data in batch], axis=0)  # type: ignore

        collated_batch[key] = value
    return tree_map(convert_to_tensor, collated_batch)


def tokenize_cells(
    cell: np.ndarray,
    var_names: Sequence[str],
    encoder: VocabularyEncoderSimplified,
    genes_seq_len: int,
    sample_genes: Literal["random", "weighted", "expressed", "expressed_zero", "none"],
    gene_tokens_key: str = ModelEnum.GENES.value,
    counts_key: str = ModelEnum.COUNTS.value,
    seed: int | None = None,
) -> dict[str, np.ndarray]:
    """Tokenize cell counts into gene tokens.

    Parameters
    ----------
    cell
        Count matrix of shape (N, G) where N is number of cells, G is number of genes
    var_names
        Gene names corresponding to columns of cell
    encoder
        Vocabulary encoder to map gene names to token indices
    genes_seq_len
        Maximum sequence length for sampled genes
    sample_genes
        Sampling strategy for genes
    gene_tokens_key
        Key for gene tokens in output dict
    counts_key
        Key for counts in output dict
    seed
        Random seed for reproducibility

    Returns
    -------
    dict[str, np.ndarray]
        Dictionary with tokenized genes, counts, and library sizes
    """
    counts = cell
    gene_idx = np.tile(encoder.encode_genes(var_names), (len(counts), 1))
    library_size = counts.sum(1, keepdims=True)

    rng = np.random.default_rng(seed=seed)
    N, G = counts.shape

    if sample_genes == "weighted":
        if encoder.metadata_genes is None:
            raise ValueError("encoder.metadata_genes must be set for weighted sampling")

        scaled_counts = (counts + 1) / encoder.metadata_genes["means"].values
        scaled_counts = scaled_counts / scaled_counts.sum(1, keepdims=True)
        sampled_idx = np.stack([rng.choice(G, size=genes_seq_len, replace=False, p=p) for p in scaled_counts])
        return {
            gene_tokens_key: np.take_along_axis(gene_idx, sampled_idx, axis=1),
            counts_key: np.take_along_axis(counts, sampled_idx, axis=1),
            "library_size": library_size,
        }

    elif sample_genes == "expressed":
        mask_idx = encoder.mask_token_idx
        expressed = counts > 0
        num_expressed = expressed.sum(axis=1)

        if (num_expressed > genes_seq_len).any():
            raise ValueError("genes_seq_len is smaller than number of expressed genes")

        pos_order = expressed.cumsum(axis=1) - 1
        genes_out = np.full((N, genes_seq_len), mask_idx, dtype=gene_idx.dtype)
        counts_out = np.zeros((N, genes_seq_len), dtype=counts.dtype)

        ii, jj = np.where(expressed)
        pp = pos_order[expressed]
        genes_out[ii, pp] = gene_idx[ii, jj]
        counts_out[ii, pp] = counts[ii, jj]

        return {
            gene_tokens_key: gene_idx,
            counts_key: counts,
            ModelEnum.GENES_SUBSET.value: genes_out,
            ModelEnum.COUNTS_SUBSET.value: counts_out,
            "library_size": library_size,
        }

    elif sample_genes == "expressed_zero":
        expressed = counts > 0
        permuted_indices = np.stack([rng.permutation(G) for _ in range(N)])

        shuffled_gene_idx = np.take_along_axis(gene_idx, permuted_indices, axis=1)
        shuffled_counts = np.take_along_axis(counts, permuted_indices, axis=1)
        shuffled_expressed = np.take_along_axis(expressed, permuted_indices, axis=1)

        priority = shuffled_expressed.astype(int)
        sort_indices = np.argsort(priority, axis=1, kind="stable")

        final_gene_idx = np.take_along_axis(shuffled_gene_idx, sort_indices, axis=1)
        final_counts = np.take_along_axis(shuffled_counts, sort_indices, axis=1)

        return {
            gene_tokens_key: gene_idx,
            counts_key: counts,
            ModelEnum.GENES_SUBSET.value: final_gene_idx[:, :genes_seq_len],
            ModelEnum.COUNTS_SUBSET.value: final_counts[:, :genes_seq_len],
            "library_size": library_size,
        }

    elif sample_genes == "random_expressed":
        mask_idx = encoder.mask_token_idx
        nonzero_mask = counts > 0

        sampled_idx = np.stack(
            [
                np.pad(
                    rng.choice(
                        np.nonzero(nonzero_mask[i])[0],
                        size=min(genes_seq_len, nonzero_mask[i].sum()),
                        replace=False,
                    ),
                    (0, max(0, genes_seq_len - nonzero_mask[i].sum())),
                    constant_values=-1,
                )
                for i in range(N)
            ]
        )

        padded_mask = sampled_idx == -1
        safe_sampled_idx = np.where(padded_mask, 0, sampled_idx)

        sampled_gene_idx = np.take_along_axis(gene_idx, safe_sampled_idx, axis=1)
        subset_counts = np.take_along_axis(counts, safe_sampled_idx, axis=1)

        sampled_gene_idx[padded_mask] = mask_idx
        subset_counts[padded_mask] = 0

        return {
            gene_tokens_key: sampled_gene_idx,
            counts_key: subset_counts,
            "library_size": library_size,
        }

    elif sample_genes == "random":
        sampled_idx = np.stack([rng.choice(G, size=genes_seq_len, replace=False) for _ in range(N)])
        return {
            gene_tokens_key: np.take_along_axis(gene_idx, sampled_idx, axis=1),
            counts_key: np.take_along_axis(counts, sampled_idx, axis=1),
            "library_size": library_size,
        }

    elif sample_genes == "none":
        return {
            gene_tokens_key: gene_idx,
            counts_key: counts,
            "library_size": library_size,
        }

    else:
        raise ValueError(f"Invalid sample_genes value: {sample_genes}")


@dataclass
class AnnDataFieldWithVarNames:
    """
    Custom AnnDataField that returns both the data and var_names.

    This is useful when you need both the X data and the corresponding var_names
    for processing, such as for tokenization where you need the actual gene names.
    """

    attr: str
    key: list[str] | str | None = None
    convert_fn: Callable[[dict[str, Any]], np.ndarray] | None = None

    def __call__(self, adata: AnnData) -> np.ndarray:
        from operator import attrgetter

        value = attrgetter(self.attr)(adata)
        if self.key is not None:
            value = value[self.key]

        # Create a dictionary with both the data and var_names
        data_dict = {"data": value.toarray(), "var_names": adata.var_names}

        if self.convert_fn is not None:
            return self.convert_fn(data_dict)
        else:
            return np.asarray(data_dict["data"])


def train_val_split_list(files: list[str], seed: int) -> tuple[list[int], list[int]]:
    rng = np.random.RandomState(seed)
    n_files = len(files)
    n_val_files = int(0.1 * n_files)
    # Only resample from first 50% of files to avoid last file with different cell count
    n_resample = n_files // 2
    indices = np.arange(n_files)
    resample_indices = rng.permutation(n_resample)
    train_indices_arr = np.concatenate([resample_indices[:-n_val_files], indices[n_resample:]])
    val_indices_arr = resample_indices[-n_val_files:]
    return train_indices_arr.tolist(), val_indices_arr.tolist()
