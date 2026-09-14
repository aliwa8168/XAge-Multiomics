from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr

from scflowdiff.core.constants import ModelEnum
from scflowdiff.core.datamodule import MultimodalMuDataModule
from scflowdiff.core.layers import InputTransformerAE
from scflowdiff.core.multimodal_ae import MultimodalAE, normalize_stage1_state_dict
from scflowdiff.core.nnets import Decoder, Encoder
from scflowdiff.core.streaming_reconstruction_metrics import (
    StreamingAtacReconstructionMetrics,
    StreamingRnaReconstructionMetrics,
)
from scflowdiff.core.stochastic_layers import (
    BernoulliTransformerLayer,
    NegativeBinomialTransformerLayer,
)
from scflowdiff.core.ae import MultimodalTransformerAE

def env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, str(default)))


def env_precision(name: str = "MMAE_PRECISION") -> str:
    precision = os.environ.get(name, "bf16_mixed").strip().lower()
    if precision not in {"fp32", "bf16_mixed"}:
        raise ValueError(f"{name} must be fp32 or bf16_mixed, got {precision!r}")
    return precision


def env_seq_len(name: str, default: int | None) -> int | None:
    value = os.environ.get(name)
    if value is None:
        return default
    if value.strip().lower() in {"all", "full", "none", "0", "-1"}:
        return None
    parsed = int(value)
    return None if parsed <= 0 else parsed


def env_sample_features(name: str, default: str) -> str:
    value = os.environ.get(name, default).strip().lower()
    allowed = {"expressed", "random", "none", "balanced", "top"}
    if value not in allowed:
        raise ValueError(f"{name} must be one of {sorted(allowed)}, got {value!r}")
    return value


def env_agg_func(name: str, default: str) -> str:
    value = os.environ.get(name, default).strip().lower()
    allowed = {"log1p", "anscombe", "sqrt", "proj", "projlog1p", "projconcat", "softbin", "log1pzero"}
    if value not in allowed:
        raise ValueError(f"{name} must be one of {sorted(allowed)}, got {value!r}")
    return value


def make_encoder(n_embed: int, n_latent: int, n_inducing: int, n_layer: int, n_head: int) -> Encoder:
    return Encoder(
        n_layer=n_layer,
        n_inducing_points=n_inducing,
        n_embed=n_embed,
        n_embed_latent=n_latent,
        n_head=n_head,
        n_head_cross=n_head,
        dropout=0.0,
        bias=False,
        multiple_of=4,
        layernorm_eps=1e-6,
        norm_layer="layernorm",
        positional_encoding=True,
    )


def make_decoder(n_tokens: int, n_embed: int, n_latent: int, n_inducing: int, n_layer: int, n_head: int) -> Decoder:
    return Decoder(
        n_genes=n_tokens,
        n_embed=n_embed,
        n_embed_latent=n_latent,
        n_head=n_head,
        n_head_cross=n_head,
        n_layer=n_layer,
        n_inducing_points=n_inducing,
        dropout=0.0,
        bias=False,
        multiple_of=4,
        layernorm_eps=1e-6,
        norm_layer="layernorm",
        shared_embedding=True,
        use_adaln=False,
    )


def build_model(n_genes: int, n_peaks: int) -> MultimodalAE:
    n_embed = env_int("MMAE_N_EMBED", 64)
    n_latent = env_int("MMAE_N_LATENT", 32)
    n_inducing = env_int("MMAE_N_INDUCING", 16)
    n_layer = env_int("MMAE_N_LAYER", 2)
    n_head = env_int("MMAE_N_HEAD", 4)
    joint_layers = env_int("MMAE_JOINT_LAYERS", 2)
    rna_likelihood = os.environ.get("MMAE_RNA_LIKELIHOOD", "nb").strip().lower()
    if rna_likelihood != "nb":
        raise ValueError("scFlowDiff RNA likelihood is fixed to 'nb'")
    rna_agg_func = env_agg_func("MMAE_RNA_AGG_FUNC", "log1p")
    atac_agg_func = env_agg_func("MMAE_ATAC_AGG_FUNC", "proj")

    rna_input = InputTransformerAE(n_genes=n_genes, n_embed=n_embed, agg_func=rna_agg_func)
    atac_input = InputTransformerAE(
        n_genes=n_genes,
        n_atac=n_peaks,
        n_embed=n_embed,
        agg_func=rna_agg_func,
        atac_agg_func=atac_agg_func,
    )

    rna_decoder_head = NegativeBinomialTransformerLayer(
        n_genes=n_genes,
        shared_theta=True,
        n_embed=n_embed,
        layernorm_eps=1e-6,
    )

    ae = MultimodalTransformerAE(
        rna_encoder=make_encoder(n_embed, n_latent, n_inducing, n_layer, n_head),
        rna_decoder=make_decoder(n_genes, n_embed, n_latent, n_inducing, n_layer, n_head),
        rna_decoder_head=rna_decoder_head,
        rna_input_layer=rna_input,
        atac_encoder=make_encoder(n_embed, n_latent, n_inducing, n_layer, n_head),
        atac_decoder=make_decoder(n_peaks, n_embed, n_latent, n_inducing, n_layer, n_head),
        atac_decoder_head=BernoulliTransformerLayer(n_embed=n_embed, layernorm_eps=1e-6),
        atac_input_layer=atac_input,
        multimodal="multimodal",
        encoder_multimodal_joint_layers=joint_layers if joint_layers > 0 else None,
        n_head=n_head,
        layernorm_eps=1e-6,
    )
    optimizer_factory = lambda params: torch.optim.AdamW(
        params,
        lr=env_float("MMAE_LR", 2e-4),
        weight_decay=env_float("MMAE_WEIGHT_DECAY", 1e-4),
        betas=(0.9, 0.95),
    )
    return MultimodalAE(ae_model=ae, ae_optimizer=optimizer_factory)


def pearson_or_nan(x: np.ndarray, y: np.ndarray) -> float:
    if x.size == 0 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(pearsonr(x, y)[0])


def spearman_or_nan(x: np.ndarray, y: np.ndarray) -> float:
    if x.size == 0 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(spearmanr(x, y)[0])


def binary_auc(probs: np.ndarray, target: np.ndarray) -> float:
    probs = probs.reshape(-1).astype(np.float64)
    target = (target.reshape(-1) > 0.5).astype(np.int64)
    positives = target.sum()
    negatives = len(target) - positives
    if positives == 0 or negatives == 0:
        return float("nan")
    order = np.argsort(probs, kind="mergesort")
    ranks = np.empty_like(probs, dtype=np.float64)
    ranks[order] = np.arange(1, len(probs) + 1, dtype=np.float64)
    rank_sum_pos = ranks[target == 1].sum()
    return float((rank_sum_pos - positives * (positives + 1) / 2) / (positives * negatives))


def average_precision(probs: np.ndarray, target: np.ndarray) -> float:
    probs = probs.reshape(-1)
    target = (target.reshape(-1) > 0.5).astype(np.int64)
    positives = target.sum()
    if positives == 0:
        return float("nan")
    order = np.argsort(-probs, kind="mergesort")
    sorted_target = target[order]
    precision_at_k = np.cumsum(sorted_target) / (np.arange(len(sorted_target)) + 1)
    return float((precision_at_k * sorted_target).sum() / positives)


def forward_and_loss(
    model: MultimodalAE,
    batch: dict[str, torch.Tensor],
    precision: str,
) -> tuple[dict[str, dict[str, torch.Tensor]], dict[str, torch.Tensor]]:
    """Run tensor-core BF16 forward while keeping likelihood arithmetic FP32."""
    if precision == "fp32":
        params, _ = model.forward(batch)
        return params, model.loss(batch, params)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        params, _ = model.forward(batch)
    params_fp32 = {
        modality: {name: value.float() for name, value in values.items()}
        for modality, values in params.items()
    }
    with torch.autocast(device_type="cuda", enabled=False):
        losses = model.loss(batch, params_fp32)
    return params_fp32, losses


@torch.no_grad()
def evaluate(
    model: MultimodalAE,
    dataloader,
    device: torch.device,
    precision: str,
    max_batches: int = 0,
) -> dict[str, object]:
    model.eval()
    rna_accumulator = None
    atac_accumulator = None
    losses = []

    for batch_idx, batch in enumerate(dataloader):
        if max_batches > 0 and batch_idx >= max_batches:
            break
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        params, loss_output = forward_and_loss(model, batch, precision)
        losses.append({k: float(v.detach().cpu()) for k, v in loss_output.items()})
        rna_true = batch[ModelEnum.RNA_COUNTS.value].detach().cpu().numpy()
        rna_pred = params["rna"]["mu"].detach().cpu().numpy()
        atac_true = batch[ModelEnum.ATAC_VALUES.value].detach().cpu().numpy()
        atac_prob = params["atac"]["probs"].detach().cpu().numpy()
        if rna_accumulator is None:
            rna_accumulator = StreamingRnaReconstructionMetrics(rna_true.shape[1])
            atac_accumulator = StreamingAtacReconstructionMetrics(atac_true.shape[1])
        rna_accumulator.update(rna_true, rna_pred)
        atac_accumulator.update(atac_prob, atac_true)

    if rna_accumulator is None or atac_accumulator is None:
        raise RuntimeError("Evaluation dataloader produced no batches")

    mean_losses = {
        key: float(np.mean([entry[key] for entry in losses if key in entry]))
        for key in sorted({key for entry in losses for key in entry})
    }
    return {
        "losses": mean_losses,
        "rna_metrics": rna_accumulator.compute(),
        "atac_metrics": atac_accumulator.compute(),
    }


@torch.no_grad()
def evaluate_reconstruction_loss(
    model: MultimodalAE,
    dataloader,
    device: torch.device,
    precision: str,
    max_batches: int = 0,
) -> dict[str, float | int]:
    """Compute cell-weighted validation losses without retaining full matrices."""
    was_training = model.training
    model.eval()
    totals = {"rna_loss": 0.0, "atac_loss": 0.0, "llh": 0.0}
    cells = 0
    batches = 0
    for batch_idx, batch in enumerate(dataloader):
        if max_batches > 0 and batch_idx >= max_batches:
            break
        batch = {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        }
        _, losses = forward_and_loss(model, batch, precision)
        batch_cells = int(batch[ModelEnum.RNA_COUNTS.value].shape[0])
        for key in totals:
            totals[key] += float(losses[key].detach().cpu()) * batch_cells
        cells += batch_cells
        batches += 1
    if was_training:
        model.train()
    if cells == 0:
        raise RuntimeError("No validation cells were available")
    return {
        **{key: value / cells for key, value in totals.items()},
        "cells": cells,
        "batches": batches,
    }


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required because this repository uses flex_attention.")

    repo = Path(__file__).resolve().parents[3]
    default_h5mu = repo / "data" / "openproblem" / "openproblem_filtered.h5mu"
    h5mu_path = Path(os.environ.get("MMAE_H5MU", str(default_h5mu))).resolve()
    default_run_dir = repo / "outputs" / h5mu_path.stem
    out_dir = Path(
        os.environ.get(
            "MMAE_OUT_DIR",
            str(default_run_dir / "stage1_ae"),
        )
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = Path(
        os.environ.get(
            "MMAE_METRICS_PATH",
            str(default_run_dir / "stage1_ae" / "training_metrics.json"),
        )
    )
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    seed = env_int("MMAE_SEED", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")

    batch_size = env_int("MMAE_BATCH_SIZE", 32)
    rna_seq_len = env_seq_len("MMAE_RNA_SEQ_LEN", None)
    atac_seq_len = env_seq_len("MMAE_ATAC_SEQ_LEN", None)
    rna_encoder_seq_len = env_seq_len("MMAE_RNA_ENCODER_SEQ_LEN", None)
    atac_encoder_seq_len = env_seq_len("MMAE_ATAC_ENCODER_SEQ_LEN", None)
    max_steps = env_int("MMAE_MAX_STEPS", 30000)
    eval_every = env_int("MMAE_EVAL_EVERY", 5000)
    eval_batches = env_int("MMAE_EVAL_BATCHES_DURING_TRAIN", 0)
    final_eval_batches = env_int("MMAE_FINAL_EVAL_BATCHES", 0)
    grad_clip = env_float("MMAE_GRAD_CLIP", 5.0)
    precision = env_precision()
    rna_sample_features = env_sample_features("MMAE_RNA_SAMPLE_FEATURES", "none")
    atac_sample_features = env_sample_features("MMAE_ATAC_SAMPLE_FEATURES", "none")
    rna_encoder_sample_features = os.environ.get("MMAE_RNA_ENCODER_SAMPLE_FEATURES")
    rna_encoder_sample_features = (
        None if rna_encoder_sample_features is None else env_sample_features("MMAE_RNA_ENCODER_SAMPLE_FEATURES", "top")
    )
    atac_encoder_sample_features = os.environ.get("MMAE_ATAC_ENCODER_SAMPLE_FEATURES")
    atac_encoder_sample_features = (
        None if atac_encoder_sample_features is None else env_sample_features("MMAE_ATAC_ENCODER_SAMPLE_FEATURES", "top")
    )

    datamodule = MultimodalMuDataModule(
        h5mu_path=h5mu_path,
        batch_size=batch_size,
        test_batch_size=batch_size,
        rna_seq_len=rna_seq_len,
        atac_seq_len=atac_seq_len,
        num_workers=0,
        train_fraction=env_float("SCFLOW_TRAIN_FRACTION", 0.8),
        val_fraction=env_float("SCFLOW_VAL_FRACTION", 0.1),
        test_fraction=env_float("SCFLOW_TEST_FRACTION", 0.1),
        split_manifest_path=Path(
            os.environ.get(
                "SCFLOW_SPLIT_MANIFEST",
                str(default_run_dir / "split_manifest.json"),
            )
        ),
        donor_split_csv=(
            Path(os.environ["SCFLOW_DONOR_SPLIT_CSV"])
            if os.environ.get("SCFLOW_DONOR_SPLIT_CSV")
            else None
        ),
        donor_key=os.environ.get("SCFLOW_DONOR_KEY", "sample_id"),
        donor_split_column=os.environ.get("SCFLOW_DONOR_SPLIT_COLUMN"),
        donor_train_label=os.environ.get("SCFLOW_DONOR_TRAIN_LABEL", "train"),
        donor_validation_label=os.environ.get(
            "SCFLOW_DONOR_VALIDATION_LABEL", "validation"
        ),
        donor_test_label=os.environ.get("SCFLOW_DONOR_TEST_LABEL", "sealed_test"),
        outer_train_label=os.environ.get("SCFLOW_OUTER_TRAIN_LABEL", "train"),
        inner_val_fraction=env_float("SCFLOW_INNER_VAL_FRACTION", 0.1),
        feature_contract_path=(
            Path(os.environ["SCFLOW_FEATURE_CONTRACT"])
            if os.environ.get("SCFLOW_FEATURE_CONTRACT")
            else None
        ),
        seed=seed,
        rna_sample_features=rna_sample_features,
        atac_sample_features=atac_sample_features,
        rna_encoder_seq_len=rna_encoder_seq_len,
        rna_encoder_sample_features=rna_encoder_sample_features,
        atac_encoder_seq_len=atac_encoder_seq_len,
        atac_encoder_sample_features=atac_encoder_sample_features,
    )
    datamodule.setup()

    model = build_model(datamodule.n_genes, datamodule.n_peaks).to(device)
    optimizer = model.configure_optimizers()["optimizer"]
    ckpt_path = out_dir / "last.ckpt"
    best_ckpt_path = out_dir / "best.ckpt"
    selection_path = out_dir / "selection.json"
    step = 0
    epoch = 0
    history: list[dict[str, object]] = []
    best_val_loss = float("inf")
    if os.environ.get("MMAE_RESUME", "0") == "1" and ckpt_path.exists():
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if (checkpoint.get("config") or {}).get("decoder_latent_contract") != "joint_attention_then_modality_split_v1":
            raise RuntimeError("Refusing to resume a Stage-1 checkpoint from the pre-split decoder contract")
        state_dict, legacy_keys_remapped = normalize_stage1_state_dict(
            checkpoint["state_dict"]
        )
        model.load_state_dict(state_dict, strict=True)
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        step = int(checkpoint.get("step", 0))
        epoch = int(checkpoint.get("epoch", 0))
        history = list(checkpoint.get("history", []))
        best_val_loss = float(checkpoint.get("best_val_loss", float("inf")))
        print(
            f"Resumed multimodal AE checkpoint from step={step}, epoch={epoch}; "
            f"legacy_keys_remapped={legacy_keys_remapped}",
            flush=True,
        )
    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()

    print(
        json.dumps(
            {
                "device": torch.cuda.get_device_name(0),
                "n_genes": datamodule.n_genes,
                "n_peaks": datamodule.n_peaks,
                "batch_size": batch_size,
                "rna_seq_len": rna_seq_len,
                "atac_seq_len": atac_seq_len,
                "rna_encoder_seq_len": rna_encoder_seq_len,
                "atac_encoder_seq_len": atac_encoder_seq_len,
                "rna_sample_features": rna_sample_features,
                "atac_sample_features": atac_sample_features,
                "rna_encoder_sample_features": rna_encoder_sample_features,
                "atac_encoder_sample_features": atac_encoder_sample_features,
                "rna_agg_func": os.environ.get("MMAE_RNA_AGG_FUNC", "log1p"),
                "atac_agg_func": os.environ.get("MMAE_ATAC_AGG_FUNC", "proj"),
                "joint_layers": env_int("MMAE_JOINT_LAYERS", 2),
                "decoder_latent_contract": "joint_attention_then_modality_split_v1",
                "rna_likelihood": "nb",
                "precision": precision,
                "max_steps": max_steps,
                "train_batches_per_epoch": len(train_loader),
                "val_batches": len(val_loader),
                "test_cells": len(datamodule.test_dataset),
                "split": datamodule.split_info,
            },
            indent=2,
        ),
        flush=True,
    )

    checkpoint_config = {
        "batch_size": batch_size,
        "rna_seq_len": rna_seq_len,
        "atac_seq_len": atac_seq_len,
        "rna_encoder_seq_len": rna_encoder_seq_len,
        "atac_encoder_seq_len": atac_encoder_seq_len,
        "rna_sample_features": rna_sample_features,
        "atac_sample_features": atac_sample_features,
        "rna_encoder_sample_features": rna_encoder_sample_features,
        "atac_encoder_sample_features": atac_encoder_sample_features,
        "rna_agg_func": os.environ.get("MMAE_RNA_AGG_FUNC", "log1p"),
        "atac_agg_func": os.environ.get("MMAE_ATAC_AGG_FUNC", "proj"),
        "max_steps": max_steps,
        "joint_layers": env_int("MMAE_JOINT_LAYERS", 2),
        "decoder_latent_contract": "joint_attention_then_modality_split_v1",
        "rna_likelihood": "nb",
        "stage1_model": "deterministic_multimodal_ae",
        "latent_sampling": False,
        "kl_loss": False,
        "dataset": str(h5mu_path),
        "split": datamodule.split_info,
        "n_rna_features": int(datamodule.n_genes),
        "n_atac_features": int(datamodule.n_peaks),
        "selection_metric": "full_validation_reconstruction_llh",
        "precision": precision,
    }
    start = time.time()
    model.train()

    while step < max_steps:
        epoch += 1
        for batch in train_loader:
            step += 1
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            _, loss_output = forward_and_loss(model, batch, precision)
            loss = loss_output["llh"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            if step == 1 or step % 25 == 0:
                elapsed = time.time() - start
                print(
                    f"step={step} epoch={epoch} loss={float(loss.detach().cpu()):.4f} "
                    f"rna={float(loss_output['rna_loss'].detach().cpu()):.4f} "
                    f"atac={float(loss_output['atac_loss'].detach().cpu()):.4f} elapsed={elapsed:.1f}s",
                    flush=True,
                )

            if step % eval_every == 0 or step == max_steps:
                eval_metrics = evaluate_reconstruction_loss(
                    model, val_loader, device, precision, max_batches=eval_batches
                )
                entry = {"step": step, "epoch": epoch, "validation": eval_metrics}
                history.append(entry)
                val_loss = float(eval_metrics["llh"])
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(
                        {
                            "state_dict": model.state_dict(),
                            "step": step,
                            "epoch": epoch,
                            "validation": eval_metrics,
                            "config": {**checkpoint_config, "weight_kind": "best"},
                        },
                        best_ckpt_path,
                    )
                selection_path.write_text(
                    json.dumps(
                        {
                            "checkpoint": str(best_ckpt_path),
                            "weight_kind": "best",
                            "selection_loss": best_val_loss,
                            "selection_split": "validation",
                            "validation_cells": int(eval_metrics["cells"]),
                            "feature_sampling_applied": False,
                            "cell_sampling_applied": False,
                        },
                        indent=2,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                torch.save(
                    {
                        "state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "step": step,
                        "epoch": epoch,
                        "history": history,
                        "best_val_loss": best_val_loss,
                        "config": {**checkpoint_config, "weight_kind": "last"},
                    },
                    ckpt_path,
                )
                metrics_path.write_text(
                    json.dumps(
                        {
                            "history": history,
                            "best_checkpoint": str(best_ckpt_path),
                            "best_val_loss": best_val_loss,
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                print("--- validation snapshot ---", flush=True)
                print(json.dumps(entry, indent=2), flush=True)
                model.train()

            if step >= max_steps:
                break

    if not best_ckpt_path.exists():
        raise RuntimeError("Stage-1 best checkpoint was not created")
    best_checkpoint = torch.load(best_ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(best_checkpoint["state_dict"], strict=True)
    final_metrics = evaluate(
        model, val_loader, device, precision, max_batches=final_eval_batches
    )
    result = {
        "checkpoint": str(best_ckpt_path),
        "last_checkpoint": str(ckpt_path),
        "selection_path": str(selection_path),
        "best_val_loss": best_val_loss,
        "metrics_path": str(metrics_path),
        "final_step": step,
        "final_epoch": epoch,
        "final_metrics": final_metrics,
        "history": history,
    }
    metrics_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("\n--- final multimodal reconstruction metrics ---", flush=True)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
