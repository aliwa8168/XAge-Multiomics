from __future__ import annotations

import copy
import gc
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from scflowdiff.core.constants import ModelEnum
from scflowdiff.core.datamodule import MultimodalMuDataModule
from scflowdiff.core.latent_cache import (
    ATAC_LATENT_KEY,
    RNA_LATENT_KEY,
    LatentCacheDataset,
    build_latent_cache,
    latent_cache_provenance,
    validate_latent_cache,
)
from scflowdiff.core.layers import InputTransformerAE
from scflowdiff.core.multimodal_ae import MultimodalAE, normalize_stage1_state_dict
from scflowdiff.core.nnets import Decoder, Encoder, MultimodalDiT
from scflowdiff.core.stochastic_layers import (
    BernoulliTransformerLayer,
    NegativeBinomialTransformerLayer,
)
from scflowdiff.core.ae import MultimodalTransformerAE


def env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, str(default)))


def env_str(name: str, default: str) -> str:
    return os.environ.get(name, default)


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
    allowed = {"expressed", "random", "none", "balanced"}
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


def build_ae(n_genes: int, n_peaks: int) -> MultimodalAE:
    n_embed = env_int("MMAE_N_EMBED", 64)
    n_latent = env_int("MMAE_N_LATENT", 32)
    n_inducing = env_int("MMAE_N_INDUCING", 16)
    n_layer = env_int("MMAE_N_LAYER", 2)
    n_head = env_int("MMAE_N_HEAD", 4)
    joint_layers = env_int("MMAE_JOINT_LAYERS", 2)
    rna_likelihood = env_str("MMAE_RNA_LIKELIHOOD", "nb").strip().lower()
    if rna_likelihood != "nb":
        raise ValueError("scFlowDiff RNA likelihood is fixed to 'nb'")

    rna_input = InputTransformerAE(n_genes=n_genes, n_embed=n_embed, agg_func="log1p")
    atac_input = InputTransformerAE(
        n_genes=n_genes,
        n_atac=n_peaks,
        n_embed=n_embed,
        agg_func="log1p",
        atac_agg_func="proj",
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
    return MultimodalAE(ae_model=ae, ae_optimizer=lambda params: torch.optim.AdamW(params, lr=1e-4))


def load_frozen_ae(n_genes: int, n_peaks: int, ckpt_path: Path, device: torch.device) -> MultimodalAE:
    model = build_ae(n_genes, n_peaks).to(device)
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    checkpoint_likelihood = (checkpoint.get("config") or {}).get("rna_likelihood")
    if checkpoint_likelihood not in {None, "nb"}:
        raise ValueError(
            f"Stage-2 requires an NB autoencoder checkpoint, got rna_likelihood={checkpoint_likelihood!r}"
        )
    decoder_contract = (checkpoint.get("config") or {}).get("decoder_latent_contract")
    if decoder_contract != "joint_attention_then_modality_split_v1":
        raise ValueError(
            "Stage-2 requires a Stage-1 checkpoint trained with the modality-split "
            f"decoder contract, got {decoder_contract!r}"
        )
    state_dict, legacy_keys_remapped = normalize_stage1_state_dict(
        checkpoint["state_dict"]
    )
    model.load_state_dict(state_dict, strict=True)
    print(
        json.dumps(
            {
                "loaded_ae_checkpoint": str(ckpt_path),
                "rna_likelihood": "nb",
                "strict_load": True,
                "legacy_keys_remapped": legacy_keys_remapped,
            },
            indent=2,
        ),
        flush=True,
    )
    model.eval()
    model.requires_grad_(False)
    return model


@torch.no_grad()
def encode_latents(
    ae: MultimodalAE,
    batch: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode independent source-only RNA and ATAC representations."""
    ae_model = ae.ae_model
    z_rna = ae_model.rna_encoder(
        ae_model.rna_input_layer(
            batch[ModelEnum.RNA_COUNTS.value],
            batch[ModelEnum.RNA_GENES.value],
            modality="rna",
        )
    )
    z_atac = ae_model.atac_encoder(
        ae_model.atac_input_layer(
            batch[ModelEnum.ATAC_VALUES.value],
            batch[ModelEnum.ATAC_PEAKS.value],
            modality="atac",
        )
    )
    return z_rna, z_atac


@torch.no_grad()
def latents_from_batch(
    batch: dict[str, torch.Tensor],
    ae: MultimodalAE | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if RNA_LATENT_KEY in batch and ATAC_LATENT_KEY in batch:
        return batch[RNA_LATENT_KEY], batch[ATAC_LATENT_KEY]
    if ae is None:
        raise RuntimeError("Raw feature batch requires the frozen multimodal AE encoder")
    return encode_latents(ae, batch)


@torch.no_grad()
def compute_latent_stats(
    ae: MultimodalAE | None,
    dataloader,
    device: torch.device,
    max_batches: int = 0,
) -> dict[str, torch.Tensor]:
    """计算 RNA/ATAC latent 的均值和标准差，用于稳定 Flow-DiT 训练。"""
    count = 0
    sum_rna = None
    sum_atac = None
    sumsq_rna = None
    sumsq_atac = None
    for batch_idx, batch in enumerate(dataloader):
        if max_batches > 0 and batch_idx >= max_batches:
            break
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        z_rna, z_atac = latents_from_batch(batch, ae)
        bsz = z_rna.shape[0]
        if sum_rna is None:
            sum_rna = torch.zeros_like(z_rna[:1])
            sum_atac = torch.zeros_like(z_atac[:1])
            sumsq_rna = torch.zeros_like(z_rna[:1])
            sumsq_atac = torch.zeros_like(z_atac[:1])
        sum_rna += z_rna.sum(dim=0, keepdim=True)
        sum_atac += z_atac.sum(dim=0, keepdim=True)
        sumsq_rna += (z_rna**2).sum(dim=0, keepdim=True)
        sumsq_atac += (z_atac**2).sum(dim=0, keepdim=True)
        count += bsz
    if count == 0 or sum_rna is None or sum_atac is None or sumsq_rna is None or sumsq_atac is None:
        raise RuntimeError("No batches were available for latent statistics")
    mean_rna = sum_rna / count
    mean_atac = sum_atac / count
    std_rna = torch.sqrt((sumsq_rna / count - mean_rna**2).clamp_min(1e-6))
    std_atac = torch.sqrt((sumsq_atac / count - mean_atac**2).clamp_min(1e-6))
    return {
        "rna_mean": mean_rna.detach().cpu(),
        "rna_std": std_rna.detach().cpu(),
        "atac_mean": mean_atac.detach().cpu(),
        "atac_std": std_atac.detach().cpu(),
    }


def normalize_latents(z_rna: torch.Tensor, z_atac: torch.Tensor, stats: dict[str, torch.Tensor], device: torch.device):
    rna_mean = stats["rna_mean"].to(device)
    rna_std = stats["rna_std"].to(device)
    atac_mean = stats["atac_mean"].to(device)
    atac_std = stats["atac_std"].to(device)
    return (z_rna - rna_mean) / rna_std, (z_atac - atac_mean) / atac_std


def flow_loss(
    model: MultimodalDiT,
    z_rna: torch.Tensor,
    z_atac: torch.Tensor,
    rna_to_atac_prob: float,
    atac_to_rna_prob: float,
    condition: dict[str, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """线性 Rectified Flow 训练目标，同时混入双向翻译条件任务。"""
    bsz = z_rna.shape[0]
    t = torch.rand((bsz,), device=z_rna.device)
    t_view = t.view(bsz, *([1] * (z_rna.ndim - 1)))
    noise_rna = torch.randn_like(z_rna)
    noise_atac = torch.randn_like(z_atac)
    x_rna_t = (1.0 - t_view) * noise_rna + t_view * z_rna
    x_atac_t = (1.0 - t_view) * noise_atac + t_view * z_atac
    target_rna = z_rna - noise_rna
    target_atac = z_atac - noise_atac

    mode_rand = torch.rand((), device=z_rna.device).item()
    if mode_rand < atac_to_rna_prob:
        # ATAC -> RNA：ATAC latent 作为已知条件，RNA latent 从噪声流向真实数据。
        pred_rna, _ = model(x_rna_t, z_atac, t, condition=None, mode="atac_to_rna")
        loss = (pred_rna - target_rna).pow(2).mean()
        mode = "atac_to_rna"
    elif mode_rand < atac_to_rna_prob + rna_to_atac_prob:
        # RNA -> ATAC：RNA latent 作为已知条件，ATAC latent 从噪声流向真实数据。
        _, pred_atac = model(z_rna, x_atac_t, t, condition=None, mode="rna_to_atac")
        loss = (pred_atac - target_atac).pow(2).mean()
        mode = "rna_to_atac"
    else:
        pred_rna, pred_atac = model(x_rna_t, x_atac_t, t, condition=None, mode="joint")
        loss_rna = (pred_rna - target_rna).pow(2).mean()
        loss_atac = (pred_atac - target_atac).pow(2).mean()
        loss = 0.5 * (loss_rna + loss_atac)
        mode = "joint"
    return loss, {"mode": mode, "t_mean": float(t.detach().mean().cpu())}


def flow_loss_conditional(
    model: MultimodalDiT,
    z_rna: torch.Tensor,
    z_atac: torch.Tensor,
    rna_to_atac_prob: float,
    atac_to_rna_prob: float,
    condition: dict[str, torch.Tensor] | None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Rectified Flow loss that forwards cell-type conditioning to Flow-DiT."""
    bsz = z_rna.shape[0]
    t = torch.rand((bsz,), device=z_rna.device)
    t_view = t.view(bsz, *([1] * (z_rna.ndim - 1)))
    noise_rna = torch.randn_like(z_rna)
    noise_atac = torch.randn_like(z_atac)
    x_rna_t = (1.0 - t_view) * noise_rna + t_view * z_rna
    x_atac_t = (1.0 - t_view) * noise_atac + t_view * z_atac
    target_rna = z_rna - noise_rna
    target_atac = z_atac - noise_atac

    mode_rand = torch.rand((), device=z_rna.device).item()
    if mode_rand < atac_to_rna_prob:
        pred_rna, _ = model(x_rna_t, z_atac, t, condition=condition, mode="atac_to_rna")
        loss = (pred_rna - target_rna).pow(2).mean()
        mode = "atac_to_rna"
    elif mode_rand < atac_to_rna_prob + rna_to_atac_prob:
        _, pred_atac = model(z_rna, x_atac_t, t, condition=condition, mode="rna_to_atac")
        loss = (pred_atac - target_atac).pow(2).mean()
        mode = "rna_to_atac"
    else:
        pred_rna, pred_atac = model(x_rna_t, x_atac_t, t, condition=condition, mode="joint")
        loss_rna = (pred_rna - target_rna).pow(2).mean()
        loss_atac = (pred_atac - target_atac).pow(2).mean()
        loss = 0.5 * (loss_rna + loss_atac)
        mode = "joint"
    return loss, {"mode": mode, "t_mean": float(t.detach().mean().cpu())}


@torch.no_grad()
def update_ema_model(
    ema_model: MultimodalDiT,
    model: MultimodalDiT,
    decay: float,
) -> None:
    for ema_parameter, parameter in zip(
        ema_model.parameters(), model.parameters(), strict=True
    ):
        ema_parameter.mul_(decay).add_(parameter.detach(), alpha=1.0 - decay)
    for ema_buffer, buffer in zip(
        ema_model.buffers(), model.buffers(), strict=True
    ):
        ema_buffer.copy_(buffer)


@torch.no_grad()
def evaluate_directional_validation(
    model: MultimodalDiT,
    ae: MultimodalAE | None,
    dataloader,
    stats: dict[str, torch.Tensor],
    device: torch.device,
    seed: int,
    max_batches: int = 0,
    rna_to_atac_weight: float = 0.5,
    atac_to_rna_weight: float = 0.5,
) -> dict[str, float | int]:
    """Select Flow checkpoints using deterministic full-validation directional MSE."""
    was_training = model.training
    model.eval()
    atac_to_rna_sum = 0.0
    rna_to_atac_sum = 0.0
    cells = 0
    batches = 0
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    for batch_idx, batch in enumerate(dataloader):
        if max_batches > 0 and batch_idx >= max_batches:
            break
        batch = {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        }
        z_rna, z_atac = latents_from_batch(batch, ae)
        z_rna, z_atac = normalize_latents(z_rna, z_atac, stats, device)
        batch_size = int(z_rna.shape[0])
        t = torch.rand((batch_size,), generator=generator, device=device)
        t_view = t.view(batch_size, *([1] * (z_rna.ndim - 1)))
        noise_rna = torch.randn(
            z_rna.shape, generator=generator, device=device, dtype=z_rna.dtype
        )
        noise_atac = torch.randn(
            z_atac.shape, generator=generator, device=device, dtype=z_atac.dtype
        )
        x_rna_t = (1.0 - t_view) * noise_rna + t_view * z_rna
        x_atac_t = (1.0 - t_view) * noise_atac + t_view * z_atac
        target_rna = z_rna - noise_rna
        target_atac = z_atac - noise_atac
        condition = {"cell_type": batch[ModelEnum.CELL_TYPE.value].long()}
        pred_rna, _ = model(
            x_rna_t,
            z_atac,
            t,
            condition=condition,
            force_drop_ids=False,
            mode="atac_to_rna",
        )
        _, pred_atac = model(
            z_rna,
            x_atac_t,
            t,
            condition=condition,
            force_drop_ids=False,
            mode="rna_to_atac",
        )
        atac_to_rna = (
            (pred_rna - target_rna).pow(2).reshape(batch_size, -1).mean(dim=1)
        )
        rna_to_atac = (
            (pred_atac - target_atac).pow(2).reshape(batch_size, -1).mean(dim=1)
        )
        atac_to_rna_sum += float(atac_to_rna.sum().cpu())
        rna_to_atac_sum += float(rna_to_atac.sum().cpu())
        cells += batch_size
        batches += 1
    if was_training:
        model.train()
    if cells == 0:
        raise RuntimeError("No validation cells were available")
    atac_to_rna_loss = atac_to_rna_sum / cells
    rna_to_atac_loss = rna_to_atac_sum / cells
    return {
        "atac_to_rna_loss": atac_to_rna_loss,
        "rna_to_atac_loss": rna_to_atac_loss,
        "selection_loss": (
            atac_to_rna_weight * atac_to_rna_loss
            + rna_to_atac_weight * rna_to_atac_loss
        ),
        "cells": cells,
        "batches": batches,
    }


def deterministic_subset_indices(
    n_total: int,
    n_requested: int,
    seed: int,
) -> tuple[list[int], str]:
    """Choose and fingerprint a fixed validation subset by positional index."""
    if n_total <= 0:
        raise ValueError("Validation dataset is empty")
    n_selected = min(max(int(n_requested), 2), n_total)
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(n_total, size=n_selected, replace=False)).astype(np.int64)
    fingerprint = hashlib.sha256(indices.tobytes()).hexdigest()
    return indices.tolist(), fingerprint


@torch.no_grad()
def multiscale_rbf_mmd2(
    real: torch.Tensor,
    generated: torch.Tensor,
) -> tuple[torch.Tensor, float]:
    """Biased, non-negative multi-scale RBF MMD-squared on flattened latents."""
    real = real.float().reshape(real.shape[0], -1)
    generated = generated.float().reshape(generated.shape[0], -1)
    if real.shape != generated.shape or real.shape[0] < 2:
        raise ValueError(
            "Latent MMD requires matching tensors with at least two cells; "
            f"got {tuple(real.shape)} and {tuple(generated.shape)}"
        )
    combined = torch.cat([real, generated], dim=0)
    squared_distance = torch.cdist(combined, combined, p=2).square()
    off_diagonal = ~torch.eye(
        squared_distance.shape[0], dtype=torch.bool, device=squared_distance.device
    )
    bandwidth = torch.median(squared_distance[off_diagonal]).clamp_min(1e-6)
    kernel = torch.zeros_like(squared_distance)
    for scale in (0.25, 0.5, 1.0, 2.0, 4.0):
        kernel.add_(torch.exp(-squared_distance / (2.0 * bandwidth * scale)))
    kernel.div_(5.0)
    n_cells = real.shape[0]
    xx = kernel[:n_cells, :n_cells].mean()
    yy = kernel[n_cells:, n_cells:].mean()
    xy = kernel[:n_cells, n_cells:].mean()
    return (xx + yy - 2.0 * xy).clamp_min(0.0), float(bandwidth.cpu())


@torch.no_grad()
def translate_latent_for_selection(
    model: MultimodalDiT,
    source: torch.Tensor,
    initial_noise: torch.Tensor,
    direction: Literal["rna_to_atac", "atac_to_rna"],
    steps: int,
    condition: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Short deterministic Euler rollout used only by checkpoint selection."""
    target = initial_noise
    for step_idx in range(steps):
        t_value = float(step_idx) / float(steps)
        t = torch.full(
            (source.shape[0],), t_value, device=source.device, dtype=source.dtype
        )
        if direction == "rna_to_atac":
            _, velocity = model(
                source,
                target,
                t,
                condition=condition,
                force_drop_ids=False,
                mode="rna_to_atac",
            )
        else:
            velocity, _ = model(
                target,
                source,
                t,
                condition=condition,
                force_drop_ids=False,
                mode="atac_to_rna",
            )
        target = target + velocity / float(steps)
    return target


@torch.no_grad()
def evaluate_latent_distribution_validation(
    model: MultimodalDiT,
    ae: MultimodalAE | None,
    dataloader,
    stats: dict[str, torch.Tensor],
    device: torch.device,
    seed: int,
    translation_steps: int,
    rna_to_atac_weight: float = 0.5,
    atac_to_rna_weight: float = 0.5,
) -> dict[str, float | int | str]:
    """Evaluate bidirectional generated-vs-real latent MMD on a fixed subset."""
    if translation_steps <= 0:
        raise ValueError("MMFLOW_SELECTION_MMD_STEPS must be positive")
    was_training = model.training
    model.eval()
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    real_rna: list[torch.Tensor] = []
    real_atac: list[torch.Tensor] = []
    generated_rna: list[torch.Tensor] = []
    generated_atac: list[torch.Tensor] = []
    batches = 0
    for batch in dataloader:
        batch = {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        }
        z_rna, z_atac = latents_from_batch(batch, ae)
        z_rna, z_atac = normalize_latents(z_rna, z_atac, stats, device)
        condition = {"cell_type": batch[ModelEnum.CELL_TYPE.value].long()}
        noise_rna = torch.randn(
            z_rna.shape, generator=generator, device=device, dtype=z_rna.dtype
        )
        noise_atac = torch.randn(
            z_atac.shape, generator=generator, device=device, dtype=z_atac.dtype
        )
        translated_rna = translate_latent_for_selection(
            model, z_atac, noise_rna, "atac_to_rna", translation_steps, condition
        )
        translated_atac = translate_latent_for_selection(
            model, z_rna, noise_atac, "rna_to_atac", translation_steps, condition
        )
        real_rna.append(z_rna)
        real_atac.append(z_atac)
        generated_rna.append(translated_rna)
        generated_atac.append(translated_atac)
        batches += 1
    if was_training:
        model.train()
    if not real_rna:
        raise RuntimeError("No fixed-subset cells were available for latent MMD")
    true_rna = torch.cat(real_rna, dim=0)
    true_atac = torch.cat(real_atac, dim=0)
    pred_rna = torch.cat(generated_rna, dim=0)
    pred_atac = torch.cat(generated_atac, dim=0)
    atac_to_rna_mmd, rna_bandwidth = multiscale_rbf_mmd2(true_rna, pred_rna)
    rna_to_atac_mmd, atac_bandwidth = multiscale_rbf_mmd2(true_atac, pred_atac)
    weighted_directional_mmd = (
        atac_to_rna_weight * atac_to_rna_mmd
        + rna_to_atac_weight * rna_to_atac_mmd
    )
    return {
        "atac_to_rna_mmd": float(atac_to_rna_mmd.cpu()),
        "rna_to_atac_mmd": float(rna_to_atac_mmd.cpu()),
        "weighted_directional_mmd": float(weighted_directional_mmd.cpu()),
        "rna_kernel_bandwidth": rna_bandwidth,
        "atac_kernel_bandwidth": atac_bandwidth,
        "cells": int(true_rna.shape[0]),
        "batches": batches,
        "translation_steps": translation_steps,
        "metric": "biased_multiscale_rbf_mmd2_on_normalized_latents",
    }


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the multimodal Flow-DiT training script")

    repo = Path(__file__).resolve().parents[3]
    default_h5mu = repo / "data" / "openproblem" / "openproblem_filtered.h5mu"
    h5mu_path = Path(env_str("MMFLOW_H5MU", str(default_h5mu))).resolve()
    default_run_dir = repo / "outputs" / h5mu_path.stem
    ae_ckpt = Path(
        env_str(
            "MMFLOW_AE_CKPT",
            str(default_run_dir / "stage1_ae" / "last.ckpt"),
        )
    )
    out_dir = Path(
        env_str(
            "MMFLOW_OUT_DIR",
            str(default_run_dir / "stage2_flow"),
        )
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    log_json = Path(
        env_str(
            "MMFLOW_LOG_JSON",
            str(default_run_dir / "stage2_flow" / "training_metrics.json"),
        )
    )
    log_json.parent.mkdir(parents=True, exist_ok=True)

    seed = env_int("MMFLOW_SEED", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")

    datamodule = MultimodalMuDataModule(
        h5mu_path=h5mu_path,
        batch_size=env_int("MMFLOW_BATCH_SIZE", 32),
        test_batch_size=env_int("MMFLOW_BATCH_SIZE", 32),
        rna_seq_len=env_seq_len("MMFLOW_RNA_SEQ_LEN", None),
        atac_seq_len=env_seq_len("MMFLOW_ATAC_SEQ_LEN", None),
        num_workers=0,
        train_fraction=env_float("SCFLOW_TRAIN_FRACTION", 0.8),
        val_fraction=env_float("SCFLOW_VAL_FRACTION", 0.1),
        test_fraction=env_float("SCFLOW_TEST_FRACTION", 0.1),
        split_manifest_path=Path(
            env_str(
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
        rna_sample_features=env_sample_features("MMFLOW_RNA_SAMPLE_FEATURES", "none"),
        atac_sample_features=env_sample_features("MMFLOW_ATAC_SAMPLE_FEATURES", "none"),
    )
    datamodule.setup()
    if datamodule.num_cell_types is None:
        raise RuntimeError("Flow-DiT cell type conditioning requires mdata.mod['rna'].obs['cell_type']")
    source_representation = "private"
    conditioning_contract = "source_only_private"

    smoke_mode = env_int("MMFLOW_SMOKE", 0) == 1
    use_latent_cache = env_int("MMFLOW_USE_LATENT_CACHE", 1) == 1
    rebuild_latent_cache = env_int("MMFLOW_REBUILD_LATENT_CACHE", 0) == 1
    cache_max_cells = env_int("MMFLOW_LATENT_CACHE_MAX_CELLS_PER_SPLIT", 0)
    if cache_max_cells > 0 and not smoke_mode:
        raise ValueError(
            "MMFLOW_LATENT_CACHE_MAX_CELLS_PER_SPLIT is permitted only when "
            "MMFLOW_SMOKE=1; formal caching must cover complete train/validation splits"
        )
    cache_dir = Path(
        env_str("MMFLOW_LATENT_CACHE_DIR", str(out_dir / "latent_cache"))
    ).resolve()
    cache_manifest: dict[str, object] | None = None
    ae: MultimodalAE | None
    if use_latent_cache:
        split_manifest = Path(datamodule.split_info["split_manifest"])
        provenance = latent_cache_provenance(
            h5mu_path=h5mu_path,
            ae_checkpoint=ae_ckpt,
            split_manifest=split_manifest,
            n_genes=int(datamodule.n_genes),
            n_peaks=int(datamodule.n_peaks),
            source_representation=source_representation,
            conditioning_contract=conditioning_contract,
        )
        provenance["cache_scope"] = (
            {"kind": "smoke_limited", "max_cells_per_split": cache_max_cells}
            if cache_max_cells > 0
            else {"kind": "complete_train_validation"}
        )
        cache_manifest = validate_latent_cache(cache_dir, provenance)
        if cache_manifest is None or rebuild_latent_cache:
            ae = load_frozen_ae(
                int(datamodule.n_genes), int(datamodule.n_peaks), ae_ckpt, device
            )
            cache_split_datasets = {
                "train": datamodule.train_dataset,
                "validation": datamodule.val_dataset,
            }
            if cache_max_cells > 0:
                cache_split_datasets = {
                    split_name: torch.utils.data.Subset(
                        split_dataset.dataset,
                        list(split_dataset.indices[:cache_max_cells]),
                    )
                    for split_name, split_dataset in cache_split_datasets.items()
                }
            cache_manifest = build_latent_cache(
                cache_dir=cache_dir,
                split_datasets=cache_split_datasets,
                encode_fn=lambda batch: encode_latents(ae, batch),
                device=device,
                batch_size=env_int(
                    "MMFLOW_LATENT_CACHE_BATCH_SIZE",
                    env_int("MMFLOW_BATCH_SIZE", 32),
                ),
                provenance=provenance,
                rebuild=rebuild_latent_cache,
            )
            del ae
            gc.collect()
            torch.cuda.empty_cache()
        ae = None
        train_cache_dataset = LatentCacheDataset(cache_dir, "train")
        validation_dataset_for_selection = LatentCacheDataset(cache_dir, "validation")
        train_loader = DataLoader(
            train_cache_dataset,
            batch_size=env_int("MMFLOW_BATCH_SIZE", 32),
            shuffle=True,
            num_workers=0,
        )
        val_loader = DataLoader(
            validation_dataset_for_selection,
            batch_size=env_int("MMFLOW_BATCH_SIZE", 32),
            shuffle=False,
            num_workers=0,
        )
        print(
            json.dumps(
                {
                    "event": "latent_cache_ready",
                    "path": str(cache_dir),
                    "splits": cache_manifest["splits"],
                },
                indent=2,
            ),
            flush=True,
        )
    else:
        ae = load_frozen_ae(
            int(datamodule.n_genes), int(datamodule.n_peaks), ae_ckpt, device
        )
        train_loader = datamodule.train_dataloader()
        val_loader = datamodule.val_dataloader()
        validation_dataset_for_selection = datamodule.val_dataset

    selection_mmd_cells = env_int("MMFLOW_SELECTION_MMD_CELLS", 256)
    selection_mmd_steps = env_int("MMFLOW_SELECTION_MMD_STEPS", 5)
    selection_mmd_weight = env_float("MMFLOW_SELECTION_MMD_WEIGHT", 1.0)
    selection_rna_to_atac_weight = env_float("MMFLOW_SELECTION_RNA_TO_ATAC_WEIGHT", 0.5)
    selection_atac_to_rna_weight = env_float("MMFLOW_SELECTION_ATAC_TO_RNA_WEIGHT", 0.5)
    if selection_mmd_cells < 2:
        raise ValueError("MMFLOW_SELECTION_MMD_CELLS must be at least 2")
    if selection_mmd_steps <= 0:
        raise ValueError("MMFLOW_SELECTION_MMD_STEPS must be positive")
    if selection_mmd_weight < 0.0:
        raise ValueError("MMFLOW_SELECTION_MMD_WEIGHT must be non-negative")
    if not np.isclose(selection_rna_to_atac_weight + selection_atac_to_rna_weight, 1.0):
        raise ValueError("Directional checkpoint-selection weights must sum to 1")
    selection_subset_indices, selection_subset_sha256 = deterministic_subset_indices(
        len(validation_dataset_for_selection),
        selection_mmd_cells,
        seed=seed + 200_000,
    )
    selection_mmd_loader = DataLoader(
        Subset(validation_dataset_for_selection, selection_subset_indices),
        batch_size=env_int("MMFLOW_BATCH_SIZE", 32),
        shuffle=False,
        num_workers=0,
    )

    # Cache creation and cache reuse must begin Flow-DiT from the same RNG state.
    torch.manual_seed(seed)
    np.random.seed(seed)
    first_batch = next(iter(train_loader))
    first_batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in first_batch.items()}
    with torch.no_grad():
        z0_rna, z0_atac = latents_from_batch(first_batch, ae)
    if z0_rna.shape != z0_atac.shape:
        raise ValueError(f"RNA and ATAC latent shapes must match for MultimodalDiT, got {z0_rna.shape} and {z0_atac.shape}")

    stats_path = Path(env_str("MMFLOW_STATS_PATH", str(out_dir / "latent_stats.pt")))
    recompute_stats = env_int("MMFLOW_RECOMPUTE_STATS", 0) == 1
    if stats_path.exists() and not recompute_stats:
        stats = torch.load(stats_path, map_location="cpu", weights_only=False)
        stored_contract = stats.get("conditioning_contract")
        if stored_contract != conditioning_contract:
            print(
                f"Recomputing latent statistics: stored contract={stored_contract!r}, "
                f"required contract={conditioning_contract!r}",
                flush=True,
            )
            recompute_stats = True
    if not stats_path.exists() or recompute_stats:
        stats = compute_latent_stats(
            ae,
            train_loader,
            device,
            max_batches=env_int("MMFLOW_STATS_MAX_BATCHES", 0),
        )
        stats["source_representation"] = source_representation
        stats["conditioning_contract"] = conditioning_contract
        torch.save(stats, stats_path)

    seq_len = z0_rna.shape[1]
    latent_dim = z0_rna.shape[2]
    dit_embed = env_int("MMFLOW_DIT_EMBED", 128)
    dit_layers = env_int("MMFLOW_DIT_LAYERS", 4)
    dit_heads = env_int("MMFLOW_DIT_HEADS", 4)
    cross_feature_dim = env_int("MMFLOW_CROSS_FEATURE_DIM", 64)
    dropout = env_float("MMFLOW_DROPOUT", 0.0)
    backbone = env_str("MMFLOW_BACKBONE", "flat").strip().lower()
    if backbone not in {"flat", "uvit"}:
        raise ValueError("MMFLOW_BACKBONE must be either 'flat' or 'uvit'")
    model = MultimodalDiT(
        n_embed=dit_embed,
        n_embed_input=latent_dim,
        n_layer=dit_layers,
        n_head=dit_heads,
        seq_len=seq_len,
        dropout=dropout,
        bias=False,
        norm_layer="layernorm",
        multiple_of=4,
        layernorm_eps=1e-6,
        class_vocab_sizes={"cell_type": int(datamodule.num_cell_types)},
        cfg_dropout_prob=env_float("MMFLOW_CFG_DROPOUT", 0.1),
        condition_strategy="joint",
        feature_dim=cross_feature_dim,
        backbone=backbone,  # "uvit" 开启 U-shaped skip connections；默认 "flat" 保持旧路径。
    ).to(device)
    ema_rate = env_float("MMFLOW_EMA_RATE", 0.9999)
    if not 0.0 < ema_rate < 1.0:
        raise ValueError("MMFLOW_EMA_RATE must be between 0 and 1")
    ema_model = copy.deepcopy(model).eval()
    ema_model.requires_grad_(False)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=env_float("MMFLOW_LR", 2e-4),
        weight_decay=env_float("MMFLOW_WEIGHT_DECAY", 1e-4),
        betas=(0.9, 0.95),
    )
    ckpt_path = out_dir / "last.ckpt"
    ema_ckpt_path = out_dir / "ema_last.ckpt"
    best_ckpt_path = out_dir / "best.ckpt"
    best_ema_ckpt_path = out_dir / "best_ema.ckpt"
    selection_path = out_dir / "selection.json"
    step = 0
    epoch = 0
    history: list[dict[str, object]] = []
    validation_history: list[dict[str, object]] = []
    best_raw_score = float("inf")
    best_ema_score = float("inf")
    best_raw_validation: dict[str, object] | None = None
    best_ema_validation: dict[str, object] | None = None
    resume_config: dict[str, object] | None = None
    if ckpt_path.exists() and env_int("MMFLOW_RESUME", 0) == 1:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        resume_config = dict(ckpt.get("config") or {})
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        step = int(ckpt.get("step", 0))
        epoch = int(ckpt.get("epoch", 0))
        history = list(ckpt.get("history", []))
        validation_history = list(ckpt.get("validation_history", []))
        if "best_raw_score" not in ckpt or "best_ema_score" not in ckpt:
            raise RuntimeError(
                "Cannot resume a legacy Flow checkpoint without composite "
                "Flow+MMD selection state; start this selection experiment in a new output directory"
            )
        best_raw_score = float(ckpt["best_raw_score"])
        best_ema_score = float(ckpt["best_ema_score"])
        best_raw_validation = ckpt.get("best_raw_validation")
        best_ema_validation = ckpt.get("best_ema_validation")
        if ema_ckpt_path.exists():
            ema_checkpoint = torch.load(
                ema_ckpt_path, map_location="cpu", weights_only=False
            )
            ema_model.load_state_dict(
                ema_checkpoint["model_state_dict"], strict=True
            )
        else:
            ema_model.load_state_dict(model.state_dict(), strict=True)
        print(f"Resumed Flow-DiT from step={step}, epoch={epoch}", flush=True)

    max_steps = env_int("MMFLOW_MAX_STEPS", 30000)
    save_every = env_int("MMFLOW_SAVE_EVERY", 1000)
    val_every = env_int("MMFLOW_VAL_EVERY", 1000)
    val_max_batches = env_int("MMFLOW_VAL_MAX_BATCHES", 0)
    if min(save_every, val_every) <= 0:
        raise ValueError("MMFLOW_SAVE_EVERY and MMFLOW_VAL_EVERY must be positive")
    if val_max_batches > 0 and not smoke_mode:
        raise ValueError(
            "MMFLOW_VAL_MAX_BATCHES is permitted only when MMFLOW_SMOKE=1; "
            "formal validation must use every validation cell"
        )
    log_every = env_int("MMFLOW_LOG_EVERY", 50)
    grad_clip = env_float("MMFLOW_GRAD_CLIP", 1.0)
    rna_to_atac_prob = env_float("MMFLOW_RNA_TO_ATAC_PROB", 0.4)
    atac_to_rna_prob = env_float("MMFLOW_ATAC_TO_RNA_PROB", 0.4)
    joint_prob = env_float("MMFLOW_JOINT_PROB", 0.2)
    task_probabilities = np.asarray([rna_to_atac_prob, atac_to_rna_prob, joint_prob])
    if np.any(task_probabilities < 0.0) or not np.isclose(task_probabilities.sum(), 1.0):
        raise ValueError(
            "MMFLOW_RNA_TO_ATAC_PROB, MMFLOW_ATAC_TO_RNA_PROB and MMFLOW_JOINT_PROB "
            f"must be non-negative and sum to 1, got {task_probabilities.tolist()}"
        )
    if resume_config is not None:
        required_resume_contract = {
            "task_probabilities": {
                "rna_to_atac": rna_to_atac_prob,
                "atac_to_rna": atac_to_rna_prob,
                "joint": joint_prob,
            },
            "selection_rna_to_atac_weight": selection_rna_to_atac_weight,
            "selection_atac_to_rna_weight": selection_atac_to_rna_weight,
        }
        observed_resume_contract = {
            key: resume_config.get(key) for key in required_resume_contract
        }
        if observed_resume_contract != required_resume_contract:
            raise RuntimeError(
                "Refusing to resume a Flow checkpoint with a different task/selection contract: "
                f"{observed_resume_contract} != {required_resume_contract}"
            )
    start_time = time.time()

    print(
        json.dumps(
            {
                "device": torch.cuda.get_device_name(0),
                "h5mu_path": str(h5mu_path),
                "ae_ckpt": str(ae_ckpt),
                "latent_shape": list(z0_rna.shape),
                "source_representation": source_representation,
                "conditioning_contract": conditioning_contract,
                "decoder_latent_contract": "joint_attention_then_modality_split_v1",
                "max_steps": max_steps,
                "train_cells": len(datamodule.train_dataset),
                "val_cells": len(datamodule.val_dataset),
                "test_cells": len(datamodule.test_dataset),
                "split": datamodule.split_info,
                "backbone": backbone,
                "ema_rate": ema_rate,
                "task_probabilities": {
                    "rna_to_atac": rna_to_atac_prob,
                    "atac_to_rna": atac_to_rna_prob,
                    "joint": joint_prob,
                },
                "validation_interval": val_every,
                "latent_cache_enabled": use_latent_cache,
                "latent_cache_path": str(cache_dir) if use_latent_cache else None,
                "validation_scope": (
                    "smoke_limited"
                    if val_max_batches > 0
                    else "full_validation_split"
                ),
                "checkpoint_selection": {
                    "formula": "weighted_directional_flow_mse + mmd_weight * weighted_directional_latent_mmd",
                    "mmd_weight": selection_mmd_weight,
                    "rna_to_atac_weight": selection_rna_to_atac_weight,
                    "atac_to_rna_weight": selection_atac_to_rna_weight,
                    "fixed_subset_cells": len(selection_subset_indices),
                    "fixed_subset_seed": seed + 200_000,
                    "fixed_subset_indices_sha256": selection_subset_sha256,
                    "translation_steps": selection_mmd_steps,
                    "mmd_metric": "biased_multiscale_rbf_mmd2_on_normalized_latents",
                },
            },
            indent=2,
        ),
        flush=True,
    )

    model.train()
    while step < max_steps:
        epoch += 1
        for batch in train_loader:
            step += 1
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            with torch.no_grad():
                z_rna, z_atac = latents_from_batch(batch, ae)
                z_rna, z_atac = normalize_latents(z_rna, z_atac, stats, device)
            condition = {"cell_type": batch[ModelEnum.CELL_TYPE.value].long()}
            optimizer.zero_grad(set_to_none=True)
            loss, meta = flow_loss_conditional(
                model,
                z_rna,
                z_atac,
                rna_to_atac_prob=rna_to_atac_prob,
                atac_to_rna_prob=atac_to_rna_prob,
                condition=condition,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            update_ema_model(ema_model, model, ema_rate)

            if step == 1 or step % log_every == 0:
                entry = {
                    "step": step,
                    "epoch": epoch,
                    "loss": float(loss.detach().cpu()),
                    "mode": meta["mode"],
                    "t_mean": meta["t_mean"],
                    "elapsed_seconds": time.time() - start_time,
                }
                history.append(entry)
                print(json.dumps(entry, ensure_ascii=False), flush=True)

            checkpoint_config = {
                "seq_len": seq_len,
                "latent_dim": latent_dim,
                "source_representation": source_representation,
                "conditioning_contract": conditioning_contract,
                "ae_checkpoint": str(ae_ckpt),
                "task_probabilities": {
                    "rna_to_atac": rna_to_atac_prob,
                    "atac_to_rna": atac_to_rna_prob,
                    "joint": joint_prob,
                },
                "dit_embed": dit_embed,
                "dit_layers": dit_layers,
                "dit_heads": dit_heads,
                "cross_feature_dim": cross_feature_dim,
                "dropout": dropout,
                "backbone": backbone,
                "class_vocab_sizes": {"cell_type": int(datamodule.num_cell_types)},
                "cfg_dropout_prob": env_float("MMFLOW_CFG_DROPOUT", 0.1),
                "ema_rate": ema_rate,
                "dataset": str(h5mu_path),
                "split": datamodule.split_info,
                "n_rna_features": int(datamodule.n_genes),
                "n_atac_features": int(datamodule.n_peaks),
                "selection_metric": "weighted_directional_flow_mse + mmd_weight * weighted_directional_latent_mmd",
                "selection_mmd_weight": selection_mmd_weight,
                "selection_rna_to_atac_weight": selection_rna_to_atac_weight,
                "selection_atac_to_rna_weight": selection_atac_to_rna_weight,
                "selection_mmd_cells": len(selection_subset_indices),
                "selection_mmd_steps": selection_mmd_steps,
                "selection_subset_seed": seed + 200_000,
                "selection_subset_indices_sha256": selection_subset_sha256,
                "selection_mmd_metric": "biased_multiscale_rbf_mmd2_on_normalized_latents",
                "latent_cache_enabled": use_latent_cache,
                "latent_cache_path": str(cache_dir) if use_latent_cache else None,
            }

            if step % val_every == 0 or step == max_steps:
                raw_metrics = evaluate_directional_validation(
                    model,
                    ae,
                    val_loader,
                    stats,
                    device,
                    seed=seed + 100_000,
                    max_batches=val_max_batches,
                    rna_to_atac_weight=selection_rna_to_atac_weight,
                    atac_to_rna_weight=selection_atac_to_rna_weight,
                )
                ema_metrics = evaluate_directional_validation(
                    ema_model,
                    ae,
                    val_loader,
                    stats,
                    device,
                    seed=seed + 100_000,
                    max_batches=val_max_batches,
                    rna_to_atac_weight=selection_rna_to_atac_weight,
                    atac_to_rna_weight=selection_atac_to_rna_weight,
                )
                raw_distribution = evaluate_latent_distribution_validation(
                    model,
                    ae,
                    selection_mmd_loader,
                    stats,
                    device,
                    seed=seed + 300_000,
                    translation_steps=selection_mmd_steps,
                    rna_to_atac_weight=selection_rna_to_atac_weight,
                    atac_to_rna_weight=selection_atac_to_rna_weight,
                )
                ema_distribution = evaluate_latent_distribution_validation(
                    ema_model,
                    ae,
                    selection_mmd_loader,
                    stats,
                    device,
                    seed=seed + 300_000,
                    translation_steps=selection_mmd_steps,
                    rna_to_atac_weight=selection_rna_to_atac_weight,
                    atac_to_rna_weight=selection_atac_to_rna_weight,
                )
                raw_score = float(raw_metrics["selection_loss"]) + selection_mmd_weight * float(
                    raw_distribution["weighted_directional_mmd"]
                )
                ema_score = float(ema_metrics["selection_loss"]) + selection_mmd_weight * float(
                    ema_distribution["weighted_directional_mmd"]
                )
                raw_metrics = {
                    **raw_metrics,
                    "latent_distribution": raw_distribution,
                    "selection_score": raw_score,
                }
                ema_metrics = {
                    **ema_metrics,
                    "latent_distribution": ema_distribution,
                    "selection_score": ema_score,
                }
                validation_entry = {
                    "step": step,
                    "epoch": epoch,
                    "raw": raw_metrics,
                    "ema": ema_metrics,
                }
                validation_history.append(validation_entry)
                print(
                    json.dumps(
                        {"event": "validation", **validation_entry},
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
                if raw_score < best_raw_score:
                    best_raw_score = raw_score
                    best_raw_validation = copy.deepcopy(raw_metrics)
                    torch.save(
                        {
                            "model_state_dict": model.state_dict(),
                            "step": step,
                            "epoch": epoch,
                            "validation": raw_metrics,
                            "latent_stats_path": str(stats_path),
                            "config": {**checkpoint_config, "weight_kind": "best_raw"},
                        },
                        best_ckpt_path,
                    )
                if ema_score < best_ema_score:
                    best_ema_score = ema_score
                    best_ema_validation = copy.deepcopy(ema_metrics)
                    torch.save(
                        {
                            "model_state_dict": ema_model.state_dict(),
                            "step": step,
                            "epoch": epoch,
                            "validation": ema_metrics,
                            "latent_stats_path": str(stats_path),
                            "config": {**checkpoint_config, "weight_kind": "best_ema"},
                        },
                        best_ema_ckpt_path,
                    )
                if best_ema_validation is None:
                    raise RuntimeError("EMA checkpoint-selection record was not initialized")
                selected_checkpoint = best_ema_ckpt_path
                selected_kind = "best_ema"
                selected_score = best_ema_score
                selected_validation = best_ema_validation
                selection_path.write_text(
                    json.dumps(
                        {
                            "checkpoint": str(selected_checkpoint),
                            "weight_kind": selected_kind,
                            "selection_score": selected_score,
                            "selection_loss": float(selected_validation["selection_loss"]),
                            "selection_latent_mmd": float(
                                selected_validation["latent_distribution"]["weighted_directional_mmd"]
                            ),
                            "best_raw_score": best_raw_score,
                            "best_ema_score": best_ema_score,
                            "selection_formula": "weighted_directional_flow_mse + mmd_weight * weighted_directional_latent_mmd",
                            "selection_mmd_weight": selection_mmd_weight,
                            "selection_rna_to_atac_weight": selection_rna_to_atac_weight,
                            "selection_atac_to_rna_weight": selection_atac_to_rna_weight,
                            "selection_split": "validation",
                            "validation_cells": int(raw_metrics["cells"]),
                            "distribution_validation_cells": len(selection_subset_indices),
                            "distribution_validation_steps": selection_mmd_steps,
                            "distribution_subset_indices_sha256": selection_subset_sha256,
                            "feature_sampling_applied": False,
                            "cell_sampling_applied": False,
                        },
                        indent=2,
                    )
                    + "\n",
                    encoding="utf-8",
                )

            if step % save_every == 0 or step == max_steps:
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "step": step,
                        "epoch": epoch,
                        "history": history,
                        "validation_history": validation_history,
                        "best_raw_score": best_raw_score,
                        "best_ema_score": best_ema_score,
                        "best_raw_validation": best_raw_validation,
                        "best_ema_validation": best_ema_validation,
                        "latent_stats_path": str(stats_path),
                        "config": {**checkpoint_config, "weight_kind": "last_raw"},
                    },
                    ckpt_path,
                )
                torch.save(
                    {
                        "model_state_dict": ema_model.state_dict(),
                        "step": step,
                        "epoch": epoch,
                        "validation_history": validation_history,
                        "latent_stats_path": str(stats_path),
                        "config": {**checkpoint_config, "weight_kind": "last_ema"},
                    },
                    ema_ckpt_path,
                )
                log_json.write_text(
                    json.dumps(
                        {
                            "checkpoint": str(ckpt_path),
                            "step": step,
                            "epoch": epoch,
                            "history": history,
                            "validation_history": validation_history,
                            "selection": (
                                json.loads(selection_path.read_text(encoding="utf-8"))
                                if selection_path.exists()
                                else None
                            ),
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                print(f"Saved Flow-DiT checkpoint to {ckpt_path}", flush=True)

            if step >= max_steps:
                break

    if not selection_path.exists():
        raise RuntimeError("Stage-2 best-checkpoint selection was not created")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    print("--- final multimodal Flow-DiT training summary ---", flush=True)
    print(
        json.dumps(
            {
                "checkpoint": selection["checkpoint"],
                "weight_kind": selection["weight_kind"],
                "selection_score": selection["selection_score"],
                "selection_loss": selection["selection_loss"],
                "selection_latent_mmd": selection["selection_latent_mmd"],
                "last_checkpoint": str(ckpt_path),
                "step": step,
                "epoch": epoch,
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
