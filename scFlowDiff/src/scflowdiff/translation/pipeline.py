"""Orchestration for 30k+30k paired-cell translation training."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .contracts import load_translation_config, validate_translation_config


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def resolve_environment(config: dict[str, Any], repository_root: str | Path) -> dict[str, str]:
    config = validate_translation_config(config)
    root = Path(repository_root).resolve()
    data = _resolve(root, str(config["dataset_path"])).resolve()
    output = _resolve(root, str(config["output_root"])).resolve()
    split = config["split"]
    stage1 = config["stage1"]
    stage2 = config["stage2"]
    task_probabilities = stage2["task_probabilities"]
    selection = stage2["checkpoint_selection"]
    shared = {
        "SCFLOW_SEED": str(config["seed"]),
        "SCFLOW_TRAIN_FRACTION": str(split["train"]),
        "SCFLOW_VAL_FRACTION": str(split["validation"]),
        "SCFLOW_TEST_FRACTION": str(split["test"]),
        "SCFLOW_SPLIT_MANIFEST": str(output / "contracts" / "split_manifest.json"),
    }
    return {
        **shared,
        "MMAE_H5MU": str(data),
        "MMAE_OUT_DIR": str(output / "stage1"),
        "MMAE_METRICS_PATH": str(output / "stage1" / "training_metrics.json"),
        "MMAE_MAX_STEPS": str(stage1["max_steps"]),
        "MMAE_SEED": str(config["seed"]),
        "MMAE_BATCH_SIZE": str(stage1["batch_size"]),
        "MMAE_PRECISION": str(stage1["precision"]),
        "MMAE_EVAL_EVERY": str(stage1["validation_every"]),
        "MMAE_EVAL_BATCHES_DURING_TRAIN": str(stage1["validation_max_batches"]),
        "MMAE_FINAL_EVAL_BATCHES": "0",
        "MMFLOW_H5MU": str(data),
        "MMFLOW_AE_CKPT": str(output / "stage1" / "best.ckpt"),
        "MMFLOW_OUT_DIR": str(output / "stage2"),
        "MMFLOW_LOG_JSON": str(output / "stage2" / "training_metrics.json"),
        "MMFLOW_MAX_STEPS": str(stage2["max_steps"]),
        "MMFLOW_SEED": str(config["seed"]),
        "MMFLOW_BATCH_SIZE": str(stage2["batch_size"]),
        "MMFLOW_VAL_EVERY": str(stage2["validation_every"]),
        "MMFLOW_VAL_MAX_BATCHES": str(stage2["validation_max_batches"]),
        "MMFLOW_RNA_TO_ATAC_PROB": str(task_probabilities["rna_to_atac"]),
        "MMFLOW_ATAC_TO_RNA_PROB": str(task_probabilities["atac_to_rna"]),
        "MMFLOW_JOINT_PROB": str(task_probabilities["joint"]),
        "MMFLOW_SELECTION_RNA_TO_ATAC_WEIGHT": str(selection["rna_to_atac_weight"]),
        "MMFLOW_SELECTION_ATAC_TO_RNA_WEIGHT": str(selection["atac_to_rna_weight"]),
        "MMFLOW_SELECTION_MMD_CELLS": str(selection["mmd_cells"]),
        "MMFLOW_SELECTION_MMD_STEPS": str(selection["mmd_steps"]),
        "MMFLOW_SELECTION_MMD_WEIGHT": str(selection["mmd_weight"]),
        "MMFLOW_STATS_PATH": str(output / "stage2" / "latent_stats.pt"),
        "MMFLOW_EVAL_SEED": str(config["seed"]),
        "MMFLOW_EVAL_BATCH_SIZE": str(stage2["batch_size"]),
        "MMFLOW_TRANSLATION_STEPS": "50",
        "MMFLOW_ALIGNED_PCA_COMPONENTS": str(
            config["evaluation"]["distribution_projection"]["pca_components"]
        ),
        "MMFLOW_ALIGNED_REPORT_PATH": str(
            output / "evaluation" / "translation_full_feature_report.json"
        ),
        "MMFLOW_ALIGNED_SUMMARY_PATH": str(
            output / "evaluation" / "translation_full_feature_summary.md"
        ),
    }


def preflight_translation(
    config_path: str | Path, *, repository_root: str | Path
) -> dict[str, object]:
    """Validate the locked config and model-input file without reading its matrices."""
    config = load_translation_config(config_path)
    root = Path(repository_root).resolve()
    data = _resolve(root, str(config["dataset_path"])).resolve()
    output = _resolve(root, str(config["output_root"])).resolve()
    if not data.is_file():
        raise FileNotFoundError(f"Translation dataset is missing: {data}")
    import h5py

    with h5py.File(data, "r") as handle:
        shapes = {
            modality: tuple(int(value) for value in handle[f"mod/{modality}/X"].attrs.get("shape", ()))
            for modality in ("rna", "atac")
        }
    expected = config["feature_contract"]
    if shapes["rna"][-1] != int(expected["rna_features"]):
        raise ValueError(f"RNA feature mismatch: {shapes['rna']} vs {expected['rna_features']}")
    if shapes["atac"][-1] != int(expected["atac_features"]):
        raise ValueError(f"ATAC feature mismatch: {shapes['atac']} vs {expected['atac_features']}")
    if shapes["rna"][0] != shapes["atac"][0]:
        raise ValueError(f"Paired modality row mismatch: {shapes}")
    payload = {
        "status": "PASS",
        "dataset": config["dataset"],
        "dataset_path": str(data),
        "shapes": {key: list(value) for key, value in shapes.items()},
        "seed": 42,
        "split": config["split"],
        "stage1_steps": 30000,
        "stage2_steps": 30000,
        "formal_training_started": False,
    }
    contract_dir = output / "contracts"
    contract_dir.mkdir(parents=True, exist_ok=True)
    (contract_dir / "preflight.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    return payload


def run_translation(
    config_path: str | Path,
    *,
    repository_root: str | Path,
    stage: str = "both",
) -> None:
    """Run selected training stages; evaluation remains an explicit later command."""
    if stage not in {"stage1", "stage2", "evaluate", "both", "all"}:
        raise ValueError("stage must be stage1, stage2, evaluate, both, or all")
    config = load_translation_config(config_path)
    root = Path(repository_root).resolve()
    environment = resolve_environment(config, repository_root)
    old = {key: os.environ.get(key) for key in environment}
    os.environ.update(environment)
    try:
        if stage in {"stage1", "both", "all"}:
            from scflowdiff.training.stage1 import main as stage1_main

            stage1_main()
        if stage in {"stage2", "both", "all"}:
            from scflowdiff.training.stage2 import main as stage2_main

            stage2_main()
        if stage in {"evaluate", "all"}:
            from scflowdiff.translation.workflow import evaluate_translation

            evaluate_translation(config, repository_root=root)
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
