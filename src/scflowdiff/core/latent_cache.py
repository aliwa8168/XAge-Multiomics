from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from scflowdiff.core.constants import ModelEnum


RNA_LATENT_KEY = "z_rna"
ATAC_LATENT_KEY = "z_atac"
CACHE_VERSION = 1


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def latent_cache_provenance(
    *,
    h5mu_path: Path,
    ae_checkpoint: Path,
    split_manifest: Path,
    n_genes: int,
    n_peaks: int,
    source_representation: str,
    conditioning_contract: str,
) -> dict[str, Any]:
    """Build the exact identity required for safe latent-cache reuse."""
    h5mu_path = h5mu_path.resolve()
    ae_checkpoint = ae_checkpoint.resolve()
    split_manifest = split_manifest.resolve()
    for path in (h5mu_path, ae_checkpoint, split_manifest):
        if not path.is_file():
            raise FileNotFoundError(path)
    data_stat = h5mu_path.stat()
    return {
        "version": CACHE_VERSION,
        "dataset": str(h5mu_path),
        "dataset_size": int(data_stat.st_size),
        "dataset_mtime_ns": int(data_stat.st_mtime_ns),
        "ae_checkpoint": str(ae_checkpoint),
        "ae_checkpoint_sha256": _sha256(ae_checkpoint),
        "split_manifest": str(split_manifest),
        "split_manifest_sha256": _sha256(split_manifest),
        "n_rna_features": int(n_genes),
        "n_atac_features": int(n_peaks),
        "source_representation": source_representation,
        "conditioning_contract": conditioning_contract,
        "dtype": "float32",
    }


class LatentCacheDataset(Dataset):
    """Random-access, memory-mapped RNA/ATAC latent split."""

    def __init__(self, cache_dir: Path, split: str):
        self.cache_dir = Path(cache_dir)
        manifest_path = self.cache_dir / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Latent cache manifest is missing: {manifest_path}")
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not self.manifest.get("complete", False):
            raise RuntimeError(f"Latent cache is incomplete: {self.cache_dir}")
        if split not in self.manifest["splits"]:
            raise KeyError(f"Latent cache has no {split!r} split")
        self.split = split
        split_dir = self.cache_dir / split
        self.z_rna = np.load(split_dir / "z_rna.npy", mmap_mode="r")
        self.z_atac = np.load(split_dir / "z_atac.npy", mmap_mode="r")
        self.cell_type = np.load(split_dir / "cell_type.npy", mmap_mode="r")
        self.cell_index = np.load(split_dir / "cell_index.npy", mmap_mode="r")
        expected = int(self.manifest["splits"][split]["cells"])
        lengths = {len(self.z_rna), len(self.z_atac), len(self.cell_type), len(self.cell_index)}
        if lengths != {expected}:
            raise ValueError(
                f"Corrupt latent cache split {split!r}: lengths={sorted(lengths)}, expected={expected}"
            )
        if self.z_rna.shape != self.z_atac.shape:
            raise ValueError(
                f"RNA/ATAC cached latent shapes differ: {self.z_rna.shape} != {self.z_atac.shape}"
            )

    def __len__(self) -> int:
        return int(self.z_rna.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        # Copy one small latent row out of the read-only mmap before tensor conversion.
        return {
            RNA_LATENT_KEY: torch.from_numpy(np.array(self.z_rna[index], copy=True)),
            ATAC_LATENT_KEY: torch.from_numpy(np.array(self.z_atac[index], copy=True)),
            ModelEnum.CELL_TYPE.value: torch.tensor(int(self.cell_type[index]), dtype=torch.long),
            "cell_index": torch.tensor(int(self.cell_index[index]), dtype=torch.long),
        }


def validate_latent_cache(cache_dir: Path, provenance: dict[str, Any]) -> dict[str, Any] | None:
    manifest_path = Path(cache_dir) / "manifest.json"
    if not manifest_path.is_file():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not manifest.get("complete", False) or manifest.get("provenance") != provenance:
        return None
    for split in ("train", "validation"):
        LatentCacheDataset(cache_dir, split)
    return manifest


@torch.inference_mode()
def build_latent_cache(
    *,
    cache_dir: Path,
    split_datasets: dict[str, Subset],
    encode_fn: Callable[[dict[str, torch.Tensor]], tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    batch_size: int,
    provenance: dict[str, Any],
    rebuild: bool = False,
) -> dict[str, Any]:
    """Encode train/validation once and atomically publish float32 mmap files."""
    cache_dir = Path(cache_dir)
    existing = validate_latent_cache(cache_dir, provenance)
    if existing is not None and not rebuild:
        return existing
    if cache_dir.exists() and not rebuild:
        raise RuntimeError(
            f"Latent cache exists but is incomplete or incompatible: {cache_dir}. "
            "Set MMFLOW_REBUILD_LATENT_CACHE=1 to preserve it as stale and rebuild."
        )

    temp_dir = cache_dir.with_name(
        f".{cache_dir.name}.building-{os.getpid()}-{time.time_ns()}"
    )
    temp_dir.mkdir(parents=True, exist_ok=False)
    split_records: dict[str, Any] = {}

    for split_name, dataset in split_datasets.items():
        n_cells = len(dataset)
        if n_cells == 0:
            raise RuntimeError(f"Cannot cache empty split {split_name!r}")
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        iterator = iter(loader)
        first_batch = next(iterator)
        first_gpu = {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in first_batch.items()
        }
        first_rna, first_atac = encode_fn(first_gpu)
        if first_rna.shape != first_atac.shape:
            raise ValueError(
                f"RNA/ATAC latent shapes differ in {split_name}: "
                f"{first_rna.shape} != {first_atac.shape}"
            )
        latent_shape = tuple(int(x) for x in first_rna.shape[1:])
        split_dir = temp_dir / split_name
        split_dir.mkdir()
        z_rna_mm = np.lib.format.open_memmap(
            split_dir / "z_rna.npy", mode="w+", dtype=np.float32, shape=(n_cells, *latent_shape)
        )
        z_atac_mm = np.lib.format.open_memmap(
            split_dir / "z_atac.npy", mode="w+", dtype=np.float32, shape=(n_cells, *latent_shape)
        )
        cell_type_mm = np.lib.format.open_memmap(
            split_dir / "cell_type.npy", mode="w+", dtype=np.int64, shape=(n_cells,)
        )
        cell_index_mm = np.lib.format.open_memmap(
            split_dir / "cell_index.npy", mode="w+", dtype=np.int64, shape=(n_cells,)
        )
        original_indices = np.asarray(dataset.indices, dtype=np.int64)

        offset = 0

        def write_batch(
            batch: dict[str, torch.Tensor], z_rna: torch.Tensor, z_atac: torch.Tensor
        ) -> None:
            nonlocal offset
            size = int(z_rna.shape[0])
            stop = offset + size
            z_rna_mm[offset:stop] = z_rna.detach().float().cpu().numpy()
            z_atac_mm[offset:stop] = z_atac.detach().float().cpu().numpy()
            cell_type_mm[offset:stop] = batch[ModelEnum.CELL_TYPE.value].detach().cpu().numpy()
            cell_index_mm[offset:stop] = original_indices[offset:stop]
            offset = stop

        write_batch(first_gpu, first_rna, first_atac)
        print(
            json.dumps({"event": "latent_cache_progress", "split": split_name, "cells": offset, "total": n_cells}),
            flush=True,
        )
        for batch_number, batch in enumerate(iterator, start=2):
            batch_gpu = {
                key: value.to(device) if isinstance(value, torch.Tensor) else value
                for key, value in batch.items()
            }
            z_rna, z_atac = encode_fn(batch_gpu)
            write_batch(batch_gpu, z_rna, z_atac)
            if batch_number % 100 == 0 or offset == n_cells:
                print(
                    json.dumps(
                        {"event": "latent_cache_progress", "split": split_name, "cells": offset, "total": n_cells}
                    ),
                    flush=True,
                )
        if offset != n_cells:
            raise RuntimeError(f"Cached {offset} cells for {split_name}, expected {n_cells}")
        for array in (z_rna_mm, z_atac_mm, cell_type_mm, cell_index_mm):
            array.flush()
        split_records[split_name] = {
            "cells": n_cells,
            "latent_shape": list(latent_shape),
            "dtype": "float32",
        }

    manifest = {
        "version": CACHE_VERSION,
        "complete": True,
        "created_at_unix": time.time(),
        "provenance": provenance,
        "splits": split_records,
    }
    (temp_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    if cache_dir.exists():
        stale_dir = cache_dir.with_name(
            f"{cache_dir.name}.stale-{time.strftime('%Y%m%d-%H%M%S')}"
        )
        cache_dir.rename(stale_dir)
        print(f"Preserved incompatible latent cache at {stale_dir}", flush=True)
    temp_dir.rename(cache_dir)
    return manifest
