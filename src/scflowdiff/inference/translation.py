"""Frozen Stage-1/Stage-2 model loading and latent translation."""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import torch

from scflowdiff.core.nnets import MultimodalDiT


def load_stage1(
    checkpoint: str | Path,
    n_rna_features: int,
    n_atac_features: int,
    device: str | torch.device = "cpu",
):
    """Load a frozen deterministic multimodal AE with strict validation."""
    from scflowdiff.training.stage2 import load_frozen_ae

    return load_frozen_ae(
        int(n_rna_features), int(n_atac_features), Path(checkpoint), torch.device(device)
    )


def load_latent_stats(path: str | Path) -> dict[str, torch.Tensor]:
    stats = torch.load(Path(path), map_location="cpu", weights_only=False)
    required = {"rna_mean", "rna_std", "atac_mean", "atac_std"}
    missing = required.difference(stats)
    if missing:
        raise ValueError(f"Latent statistics are missing keys: {sorted(missing)}")
    return stats


def load_stage2(
    checkpoint: str | Path,
    latent_stats: dict[str, torch.Tensor],
    device: str | torch.device = "cpu",
) -> MultimodalDiT:
    """Build and strictly load a Stage-2 EMA flow checkpoint."""
    device = torch.device(device)
    payload = torch.load(Path(checkpoint), map_location="cpu", weights_only=False)
    config = payload.get("config", {})
    seq_len = int(config.get("seq_len", latent_stats["rna_mean"].shape[1]))
    latent_dim = int(config.get("latent_dim", latent_stats["rna_mean"].shape[2]))
    model = MultimodalDiT(
        n_embed=int(config.get("dit_embed", 128)),
        n_embed_input=latent_dim,
        n_layer=int(config.get("dit_layers", 4)),
        n_head=int(config.get("dit_heads", 4)),
        seq_len=seq_len,
        dropout=float(config.get("dropout", 0.0)),
        bias=False,
        norm_layer="layernorm",
        multiple_of=4,
        layernorm_eps=1e-6,
        class_vocab_sizes=dict(config.get("class_vocab_sizes", {"cell_type": 1})),
        cfg_dropout_prob=float(config.get("cfg_dropout_prob", 0.1)),
        condition_strategy="joint",
        feature_dim=int(config.get("cross_feature_dim", 64)),
        backbone=str(config.get("backbone", "flat")),
    ).to(device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.conditioning_contract = config.get("conditioning_contract")
    model.eval().requires_grad_(False)
    return model


def normalize_latent(
    latent: torch.Tensor,
    stats: dict[str, torch.Tensor],
    modality: Literal["rna", "atac"],
) -> torch.Tensor:
    mean = stats[f"{modality}_mean"].to(latent.device)
    std = stats[f"{modality}_std"].to(latent.device)
    return (latent - mean) / std


def denormalize_latent(
    latent: torch.Tensor,
    stats: dict[str, torch.Tensor],
    modality: Literal["rna", "atac"],
) -> torch.Tensor:
    return latent * stats[f"{modality}_std"].to(latent.device) + stats[
        f"{modality}_mean"
    ].to(latent.device)


@torch.no_grad()
def translate_latent(
    model: MultimodalDiT,
    source: torch.Tensor,
    direction: Literal["rna_to_atac", "atac_to_rna"],
    *,
    steps: int = 50,
    condition: dict[str, torch.Tensor] | None = None,
    initial_target: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Translate a normalized source latent using explicit Euler integration."""
    if steps < 1:
        raise ValueError("steps must be positive")
    target = (
        torch.randn(source.shape, device=source.device, dtype=source.dtype, generator=generator)
        if initial_target is None
        else initial_target.to(source.device)
    )
    if target.shape != source.shape:
        raise ValueError(f"source and initial_target shapes differ: {source.shape} != {target.shape}")
    t_grid = torch.linspace(0.0, 1.0, steps + 1, device=source.device)
    for index in range(steps):
        time = torch.full((source.shape[0],), t_grid[index], device=source.device)
        if direction == "rna_to_atac":
            _, velocity = model(
                source, target, time, condition=condition, force_drop_ids=False, mode=direction
            )
        elif direction == "atac_to_rna":
            velocity, _ = model(
                target, source, time, condition=condition, force_drop_ids=False, mode=direction
            )
        else:
            raise ValueError(f"Unsupported direction: {direction}")
        target = target + (t_grid[index + 1] - t_grid[index]) * velocity
    return target
