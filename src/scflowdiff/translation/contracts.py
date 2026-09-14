"""Configuration contract for paired-cell translation runs."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def validate_translation_config(config: dict[str, Any]) -> dict[str, Any]:
    if config.get("task") != "translation":
        raise ValueError("Translation config requires task: translation")
    if int(config.get("seed", -1)) != 42:
        raise ValueError("The release workflow is locked to seed 42")
    for stage in ("stage1", "stage2"):
        if int(config.get(stage, {}).get("max_steps", -1)) != 30000:
            raise ValueError(f"{stage}.max_steps must equal 30000")
    stage1 = config["stage1"]
    if int(stage1.get("batch_size", -1)) != 32:
        raise ValueError("stage1.batch_size must equal 32")
    if stage1.get("precision") != "bf16_mixed":
        raise ValueError("stage1.precision must equal bf16_mixed")
    if int(stage1.get("validation_every", -1)) != 5000:
        raise ValueError("stage1.validation_every must equal 5000")
    if int(stage1.get("validation_max_batches", -1)) != 0:
        raise ValueError("Stage 1 checkpoint selection must use the full validation set")
    stage2 = config["stage2"]
    if int(stage2.get("batch_size", -1)) != 32:
        raise ValueError("stage2.batch_size must equal 32")
    if int(stage2.get("validation_every", -1)) != 1000:
        raise ValueError("stage2.validation_every must equal 1000")
    if int(stage2.get("validation_max_batches", -1)) != 0:
        raise ValueError("Stage 2 checkpoint selection must use the full validation set")
    probabilities = stage2.get("task_probabilities", {})
    expected_probabilities = {"rna_to_atac": 0.4, "atac_to_rna": 0.4, "joint": 0.2}
    if any(
        abs(float(probabilities.get(key, -1)) - value) > 1e-9
        for key, value in expected_probabilities.items()
    ):
        raise ValueError(f"stage2.task_probabilities must equal {expected_probabilities}")
    selection = stage2.get("checkpoint_selection", {})
    expected_selection = {
        "rna_to_atac_weight": 0.5,
        "atac_to_rna_weight": 0.5,
        "mmd_cells": 256,
        "mmd_steps": 5,
        "mmd_weight": 1.0,
    }
    if any(
        abs(float(selection.get(key, -1)) - value) > 1e-9
        for key, value in expected_selection.items()
    ):
        raise ValueError(f"stage2.checkpoint_selection must include {expected_selection}")
    if selection.get("compare_raw_and_ema") is not True:
        raise ValueError("Stage 2 selection must compare raw and EMA weights")
    if selection.get("selected_weight_kind") != "best_ema":
        raise ValueError("Stage 2 release checkpoint must be best_ema")
    split = config.get("split", {})
    if split.get("strategy") != "random_cell":
        raise ValueError("Translation requires a random_cell split")
    fractions = [float(split.get(key, -1)) for key in ("train", "validation", "test")]
    if any(value <= 0 for value in fractions) or abs(sum(fractions) - 1.0) > 1e-9:
        raise ValueError(f"Invalid translation split fractions: {fractions}")
    directions = set(config.get("directions", []))
    if directions != {"rna_to_atac", "atac_to_rna"}:
        raise ValueError("Translation must enable both RNA-to-ATAC and ATAC-to-RNA")
    if not config.get("dataset_path") or not config.get("output_root"):
        raise ValueError("dataset_path and output_root are required")
    evaluation = config.get("evaluation", {})
    projection = evaluation.get("distribution_projection", {})
    expected_evaluation = {
        "split": "test",
        "cell_sampling": False,
        "feature_sampling": False,
        "atac_accuracy": "bernoulli_sample",
        "cell_type_min_cells": 30,
        "summary_schema": "cima_translation_full_feature_v1",
    }
    for key, expected in expected_evaluation.items():
        if evaluation.get(key) != expected:
            raise ValueError(f"evaluation.{key} must equal {expected!r}")
    if projection != {
        "method": "countsketch_joint_pca",
        "sketch_components": 512,
        "pca_components": 50,
        "seed": 123,
    }:
        raise ValueError("evaluation.distribution_projection does not match CIMA v1")
    return config


def load_translation_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return validate_translation_config(payload)
