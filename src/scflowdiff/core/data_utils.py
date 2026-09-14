"""Small filesystem helpers used by the multimodal data module."""
from __future__ import annotations

import json
from pathlib import Path


def sort_h5ad_files(path: Path) -> list[str]:
    return sorted(
        [file.as_posix() for file in path.glob("*.h5ad")],
        key=lambda value: int(value.replace(".h5ad", "").split("_")[-1]),
    )


def get_tissue_adata_files(
    base_path: Path, split: str = "train"
) -> tuple[list[str], int, int]:
    files: list[str] = []
    shard_sizes: list[int] = []
    total_cells = 0
    for tissue_dir in Path(base_path).iterdir():
        split_dir = tissue_dir / split
        if not tissue_dir.is_dir() or "genes" in str(tissue_dir) or not split_dir.exists():
            continue
        metadata_file = split_dir / "metadata.json"
        if metadata_file.exists():
            metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
            total_cells += metadata["n_cells"] - metadata["last_shard_size"]
            shard_sizes.append(metadata["shard_size"])
        shards = sort_h5ad_files(split_dir)
        files.extend(shards[:-1])
    unique_sizes = set(shard_sizes)
    if len(unique_sizes) != 1:
        raise ValueError(f"Expected one shard size, got {sorted(unique_sizes)}")
    return sorted(files), total_cells, unique_sizes.pop()

