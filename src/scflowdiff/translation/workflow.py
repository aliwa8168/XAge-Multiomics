"""Full-test bidirectional translation evaluation for a completed run."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch

from scflowdiff.core.constants import ModelEnum
from scflowdiff.core.datamodule import MultimodalMuDataModule
from scflowdiff.inference.translation import (
    denormalize_latent,
    load_latent_stats,
    load_stage1,
    load_stage2,
    normalize_latent,
    translate_latent,
)
from scflowdiff.training.stage2 import encode_latents
from scflowdiff.translation.evaluation import (
    FullTestDistributionAccumulator,
    StreamingAtacMetrics,
    StreamingCellTypeProfiles,
    StreamingFeaturePearson,
    StreamingRnaMetrics,
    write_full_feature_report,
)


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _strings(values: np.ndarray) -> list[str]:
    return [value.decode() if isinstance(value, bytes) else str(value) for value in values]


def _feature_names(path: Path, modality: str, count: int) -> list[str]:
    with h5py.File(path, "r") as handle:
        var = handle[f"mod/{modality}/var"]
        index_key = var.attrs.get("_index")
        if isinstance(index_key, bytes):
            index_key = index_key.decode()
        for key in (index_key, "_index", "index", "gene_symbol", "canonical_gene_key"):
            if key and key in var and len(var[key]) == count:
                return _strings(var[key][:])
    return [f"{modality}_{index}" for index in range(count)]


def _sample_rna(distribution: Any) -> torch.Tensor:
    try:
        return distribution.sample()
    except NotImplementedError:
        mean = getattr(distribution, "mu", distribution.mean)
        return torch.poisson(mean.clamp_min(0))


def _numpy(value: torch.Tensor, dtype=np.float32) -> np.ndarray:
    return value.detach().cpu().numpy().astype(dtype, copy=False)


def _selected_checkpoint(stage2: Path) -> Path:
    selection = stage2 / "selection.json"
    if selection.is_file():
        payload = json.loads(selection.read_text(encoding="utf-8"))
        selected = Path(payload["checkpoint"])
        return selected if selected.is_absolute() else (stage2 / selected).resolve()
    return stage2 / "best_ema.ckpt"


def evaluate_translation(config: dict[str, Any], *, repository_root: str | Path) -> dict[str, str]:
    """Evaluate all test cells and all features using the locked CIMA-v1 metrics."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for translation evaluation")
    root = Path(repository_root).resolve()
    data = _resolve(root, str(config["dataset_path"])).resolve()
    output = _resolve(root, str(config["output_root"])).resolve()
    stage1_checkpoint = output / "stage1" / "best.ckpt"
    stage2_dir = output / "stage2"
    stage2_checkpoint = _selected_checkpoint(stage2_dir)
    stats_path = stage2_dir / "latent_stats.pt"
    for path in (stage1_checkpoint, stage2_checkpoint, stats_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required evaluation artifact is missing: {path}")

    torch.manual_seed(int(config["seed"]))
    np.random.seed(int(config["seed"]))
    device = torch.device("cuda")
    batch_size = int(config["stage2"]["batch_size"])
    split = config["split"]
    datamodule = MultimodalMuDataModule(
        h5mu_path=data,
        batch_size=batch_size,
        test_batch_size=batch_size,
        rna_seq_len=None,
        atac_seq_len=None,
        num_workers=0,
        train_fraction=float(split["train"]),
        val_fraction=float(split["validation"]),
        test_fraction=float(split["test"]),
        split_manifest_path=output / "contracts" / "split_manifest.json",
        seed=int(config["seed"]),
        rna_sample_features="none",
        atac_sample_features="none",
    )
    datamodule.setup()
    n_cells = len(datamodule.test_dataset)
    ae = load_stage1(stage1_checkpoint, datamodule.n_genes, datamodule.n_peaks, device)
    stats = load_latent_stats(stats_path)
    flow = load_stage2(stage2_checkpoint, stats, device)
    if getattr(flow, "conditioning_contract", None) != "source_only_private":
        raise RuntimeError("Stage-2 checkpoint does not use source_only_private conditioning")

    rna_metrics = StreamingRnaMetrics(datamodule.n_genes)
    rna_sample_metrics = StreamingRnaMetrics(datamodule.n_genes)
    atac_prob_metrics = StreamingAtacMetrics(datamodule.n_peaks)
    atac_sample_metrics = StreamingAtacMetrics(datamodule.n_peaks)
    gene_metrics = StreamingFeaturePearson(datamodule.n_genes)
    peak_metrics = StreamingFeaturePearson(datamodule.n_peaks)
    atac_sample_features = StreamingFeaturePearson(datamodule.n_peaks)
    rna_celltypes = StreamingCellTypeProfiles(datamodule.n_genes, datamodule.cell_type_mapping)
    atac_celltypes = StreamingCellTypeProfiles(datamodule.n_peaks, datamodule.cell_type_mapping)
    projection = config["evaluation"]["distribution_projection"]
    rna_distribution = FullTestDistributionAccumulator(
        n_cells, datamodule.n_genes, "rna",
        pca_components=int(projection["pca_components"]),
        sketch_components=int(projection["sketch_components"]),
        seed=int(projection["seed"]),
    )
    atac_distribution = FullTestDistributionAccumulator(
        n_cells, datamodule.n_peaks, "atac",
        pca_components=int(projection["pca_components"]),
        sketch_components=int(projection["sketch_components"]),
        seed=int(projection["seed"]),
    )

    for batch in datamodule.test_dataloader():
        batch = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
        with torch.no_grad():
            z_rna, z_atac = encode_latents(ae, batch)
            rna_source = normalize_latent(z_rna, stats, "rna")
            atac_source = normalize_latent(z_atac, stats, "atac")
            condition = {"cell_type": batch[ModelEnum.CELL_TYPE.value].long()}
            z_atac_pred = denormalize_latent(
                translate_latent(flow, rna_source, "rna_to_atac", steps=50, condition=condition),
                stats, "atac",
            )
            z_rna_pred = denormalize_latent(
                translate_latent(flow, atac_source, "atac_to_rna", steps=50, condition=condition),
                stats, "rna",
            )
            atac_prob = ae.ae_model.decode_atac(
                z_atac_pred, batch[ModelEnum.ATAC_PEAKS.value], z_rna_source=z_rna
            )
            atac_sample = torch.bernoulli(atac_prob).float()
            rna_dist = ae.ae_model.decode_rna(
                z_rna_pred,
                batch[ModelEnum.RNA_GENES.value],
                batch[ModelEnum.RNA_LIBRARY_SIZE.value],
                z_atac_source=z_atac,
            )
            rna_mu = getattr(rna_dist, "mu", rna_dist.mean)
            rna_sample = _sample_rna(rna_dist)

        true_rna = _numpy(batch[ModelEnum.RNA_COUNTS.value])
        true_atac = _numpy(batch[ModelEnum.ATAC_VALUES.value])
        pred_rna = _numpy(rna_mu)
        sampled_rna = _numpy(rna_sample)
        pred_atac = _numpy(atac_prob)
        sampled_atac = _numpy(atac_sample)
        celltypes = _numpy(batch[ModelEnum.CELL_TYPE.value], np.int64).reshape(-1)
        true_log, pred_log = rna_metrics.update(true_rna, pred_rna)
        rna_sample_metrics.update(true_rna, sampled_rna)
        rna_distribution.update(true_rna, sampled_rna)
        gene_metrics.update(true_log, pred_log)
        rna_celltypes.update(true_log, pred_log, celltypes)
        atac_prob_metrics.update(pred_atac, true_atac)
        atac_sample_metrics.update(sampled_atac, true_atac)
        atac_distribution.update(true_atac, sampled_atac)
        peak_metrics.update(true_atac, pred_atac)
        atac_sample_features.update(true_atac, sampled_atac)
        atac_celltypes.update(true_atac, pred_atac, celltypes)

    rna_names = _feature_names(data, "rna", datamodule.n_genes)
    atac_names = _feature_names(data, "atac", datamodule.n_peaks)
    gene_summary, gene_rows = gene_metrics.compute(rna_names)
    peak_summary, peak_rows = peak_metrics.compute(atac_names)
    rna_ct_summary, rna_ct_rows = rna_celltypes.compute(
        min_cells=int(config["evaluation"]["cell_type_min_cells"])
    )
    atac_ct_summary, atac_ct_rows = atac_celltypes.compute(
        min_cells=int(config["evaluation"]["cell_type_min_cells"])
    )
    rna_dist_summary = rna_distribution.compute(device="cuda")
    rna_dist_summary.update(rna_sample_metrics.distribution_summary())
    atac_dist_summary = atac_distribution.compute(device="cuda")
    atac_dist_summary.update(atac_sample_features.distribution_summary())
    report = {
        "dataset": str(data),
        "checkpoint": str(stage2_checkpoint),
        "ae_checkpoint": str(stage1_checkpoint),
        "latent_stats": str(stats_path),
        "translation_steps": 50,
        "evaluation_split": "test",
        "eval_cells": n_cells,
        "n_rna_features": datamodule.n_genes,
        "n_atac_features": datamodule.n_peaks,
        "cell_sampling_applied": False,
        "feature_sampling_applied": False,
        "split": datamodule.split_info,
        "atac_to_rna": {
            "distribution": rna_dist_summary,
            "token_metrics": rna_metrics.compute(),
            "sampled_token_metrics": rna_sample_metrics.compute(),
            "featurewise_metrics": {"gene_wise_pearson": gene_summary},
            "cell_type_metrics": {"pcc": rna_ct_summary},
        },
        "rna_to_atac": {
            "distribution": atac_dist_summary,
            "prob_metrics": atac_prob_metrics.compute(),
            "sample_metrics": atac_sample_metrics.compute(),
            "featurewise_metrics": {"peak_wise_pearson": peak_summary},
            "cell_type_metrics": {"pcc": atac_ct_summary},
        },
    }
    paths = write_full_feature_report(
        output / "evaluation", report,
        {
            "atac_to_rna_gene_wise_pearson": gene_rows,
            "atac_to_rna_cell_type_pcc": rna_ct_rows,
            "rna_to_atac_peak_wise_pearson": peak_rows,
            "rna_to_atac_cell_type_pcc": atac_ct_rows,
        },
    )
    complete = output / "evaluation" / "COMPLETE.json"
    complete.write_text(json.dumps({"status": "PASS", "artifacts": paths}, indent=2) + "\n")
    paths["complete"] = str(complete)
    return paths
