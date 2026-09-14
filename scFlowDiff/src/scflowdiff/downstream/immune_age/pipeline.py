"""Configuration and Stage-1/Stage-2 orchestration for immune-age runs."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


def load_immune_age_config(path: str | Path) -> dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("task") != "immune_age":
        raise ValueError("Immune-age config requires task: immune_age")
    if int(payload.get("seed", -1)) != 42:
        raise ValueError("The release immune-age workflow is locked to seed 42")
    model = payload.get("scflowdiff", {})
    if int(model.get("stage1_steps", -1)) != 30000:
        raise ValueError("scflowdiff.stage1_steps must equal 30000")
    if int(model.get("stage2_steps", -1)) != 30000:
        raise ValueError("scflowdiff.stage2_steps must equal 30000")
    expected_model = {
        "batch_size": 32,
        "precision": "bf16_mixed",
        "stage1_validation_every": 5000,
        "stage1_validation_max_batches": 0,
        "stage2_validation_every": 1000,
        "stage2_validation_max_batches": 0,
    }
    for key, expected in expected_model.items():
        if model.get(key) != expected:
            raise ValueError(f"scflowdiff.{key} must equal {expected!r}")
    probabilities = model.get("task_probabilities", {})
    if probabilities != {"rna_to_atac": 0.4, "atac_to_rna": 0.4, "joint": 0.2}:
        raise ValueError("Immune-age Stage 2 task probabilities must be 0.4/0.4/0.2")
    selection = model.get("checkpoint_selection", {})
    expected_selection = {
        "rna_to_atac_weight": 0.5,
        "atac_to_rna_weight": 0.5,
        "mmd_cells": 256,
        "mmd_steps": 5,
        "mmd_weight": 1.0,
        "selected_weight_kind": "best_ema",
    }
    if selection != expected_selection:
        raise ValueError("Immune-age checkpoint selection contract is not locked")
    split = payload.get("split", {})
    if split.get("strategy") != "frozen_donor":
        raise ValueError("Immune-age requires the frozen_donor split")
    if int(split.get("development", -1)) != 316 or int(split.get("sealed_test", -1)) != 78:
        raise ValueError("Immune-age split must contain 316 development and 78 sealed-test donors")
    if set(payload.get("directions", {})) != {"rna_to_atac", "atac_to_rna"}:
        raise ValueError("Both downstream directions must be configured")
    flow_hidden = payload.get("flow_hidden", {})
    if int(flow_hidden.get("euler_steps", -1)) != 50:
        raise ValueError("flow_hidden.euler_steps must equal the locked 50-step Euler contract")
    return payload


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def resolve_scflowdiff_environment(
    config: dict[str, Any], repository_root: str | Path, sex: str
) -> dict[str, str]:
    root = Path(repository_root).resolve()
    sex_config = config["sexes"][sex]
    data = _resolve(root, config["dataset_path"]).resolve()
    split = _resolve(root, sex_config["split_csv"]).resolve()
    feature = _resolve(root, sex_config["feature_contract"]).resolve()
    output = _resolve(root, config["output_root"]).resolve() / "shared" / "scflowdiff" / sex
    shared = {
        "SCFLOW_SEED": "42",
        "SCFLOW_DONOR_SPLIT_CSV": str(split),
        "SCFLOW_DONOR_SPLIT_COLUMN": "benchmark_split",
        "SCFLOW_DONOR_TRAIN_LABEL": "train",
        "SCFLOW_DONOR_VALIDATION_LABEL": "validation",
        "SCFLOW_DONOR_TEST_LABEL": "sealed_test",
        "SCFLOW_FEATURE_CONTRACT": str(feature),
        "SCFLOW_SPLIT_MANIFEST": str(output / "contracts" / "split_manifest.json"),
    }
    return {
        **shared,
        "MMAE_H5MU": str(data),
        "MMAE_OUT_DIR": str(output / "stage1"),
        "MMAE_METRICS_PATH": str(output / "stage1" / "training_metrics.json"),
        "MMAE_MAX_STEPS": str(config["scflowdiff"]["stage1_steps"]),
        "MMAE_SEED": str(config["seed"]),
        "MMAE_BATCH_SIZE": str(config["scflowdiff"]["batch_size"]),
        "MMAE_PRECISION": str(config["scflowdiff"]["precision"]),
        "MMAE_EVAL_EVERY": str(config["scflowdiff"]["stage1_validation_every"]),
        "MMAE_EVAL_BATCHES_DURING_TRAIN": str(
            config["scflowdiff"]["stage1_validation_max_batches"]
        ),
        "MMAE_FINAL_EVAL_BATCHES": "0",
        "MMFLOW_H5MU": str(data),
        "MMFLOW_AE_CKPT": str(output / "stage1" / "best.ckpt"),
        "MMFLOW_OUT_DIR": str(output / "stage2"),
        "MMFLOW_LOG_JSON": str(output / "stage2" / "training_metrics.json"),
        "MMFLOW_MAX_STEPS": str(config["scflowdiff"]["stage2_steps"]),
        "MMFLOW_SEED": str(config["seed"]),
        "MMFLOW_BATCH_SIZE": str(config["scflowdiff"]["batch_size"]),
        "MMFLOW_VAL_EVERY": str(config["scflowdiff"]["stage2_validation_every"]),
        "MMFLOW_VAL_MAX_BATCHES": str(
            config["scflowdiff"]["stage2_validation_max_batches"]
        ),
        "MMFLOW_RNA_TO_ATAC_PROB": str(
            config["scflowdiff"]["task_probabilities"]["rna_to_atac"]
        ),
        "MMFLOW_ATAC_TO_RNA_PROB": str(
            config["scflowdiff"]["task_probabilities"]["atac_to_rna"]
        ),
        "MMFLOW_JOINT_PROB": str(config["scflowdiff"]["task_probabilities"]["joint"]),
        "MMFLOW_SELECTION_RNA_TO_ATAC_WEIGHT": str(
            config["scflowdiff"]["checkpoint_selection"]["rna_to_atac_weight"]
        ),
        "MMFLOW_SELECTION_ATAC_TO_RNA_WEIGHT": str(
            config["scflowdiff"]["checkpoint_selection"]["atac_to_rna_weight"]
        ),
        "MMFLOW_SELECTION_MMD_CELLS": str(
            config["scflowdiff"]["checkpoint_selection"]["mmd_cells"]
        ),
        "MMFLOW_SELECTION_MMD_STEPS": str(
            config["scflowdiff"]["checkpoint_selection"]["mmd_steps"]
        ),
        "MMFLOW_SELECTION_MMD_WEIGHT": str(
            config["scflowdiff"]["checkpoint_selection"]["mmd_weight"]
        ),
    }


def run_scflowdiff(
    config_path: str | Path,
    *,
    repository_root: str | Path,
    sex: str,
    stage: str = "both",
) -> None:
    if sex not in {"female", "male"}:
        raise ValueError("sex must be female or male")
    if stage not in {"stage1", "stage2", "both"}:
        raise ValueError("stage must be stage1, stage2, or both")
    config = load_immune_age_config(config_path)
    environment = resolve_scflowdiff_environment(config, repository_root, sex)
    old = {key: os.environ.get(key) for key in environment}
    os.environ.update(environment)
    try:
        if stage in {"stage1", "both"}:
            from scflowdiff.training.stage1 import main as stage1_main

            stage1_main()
        if stage in {"stage2", "both"}:
            from scflowdiff.training.stage2 import main as stage2_main

            stage2_main()
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
