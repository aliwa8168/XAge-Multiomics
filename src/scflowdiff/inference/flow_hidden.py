"""Expose direction-specific late Flow-DiT streams for downstream DATP."""
from __future__ import annotations

import torch


def forward_with_hidden(model, x_rna, x_atac, time, condition=None, mode=None):
    time_embedding = model.t_embedder(time).unsqueeze(1)
    condition_embedding = (
        model._get_condition_embedding(condition, False)
        if condition
        else model._empty_condition_embedding(x_rna.shape[0], x_rna.device)
    )
    time_embedding = time_embedding + condition_embedding
    if model.backbone == "uvit":
        time_embedding = time_embedding + model._get_direction_embedding(
            mode, x_rna.shape[0], x_rna.device
        )
    h_rna = model.input_proj_rna(x_rna) + model.pos_embed
    h_atac = model.input_proj_atac(x_atac) + model.pos_embed
    if model.backbone == "flat":
        for block in model.blocks:
            h_rna, h_atac = model._apply_dual_block(h_rna, h_atac, time_embedding, block)
    else:
        rna_skips, atac_skips = [], []
        for block in model.encoder_blocks:
            h_rna, h_atac = model._apply_dual_block(h_rna, h_atac, time_embedding, block)
            rna_skips.append(h_rna)
            atac_skips.append(h_atac)
        for index, block in enumerate(model.decoder_blocks):
            h_rna = model.rna_skip_projs[index](
                torch.cat([h_rna, rna_skips[-index - 1]], dim=-1)
            )
            h_atac = model.atac_skip_projs[index](
                torch.cat([h_atac, atac_skips[-index - 1]], dim=-1)
            )
            h_rna, h_atac = model._apply_dual_block(h_rna, h_atac, time_embedding, block)
    return (
        model.final_layer_rna(h_rna, time_embedding),
        model.final_layer_atac(h_atac, time_embedding),
        h_rna,
        h_atac,
    )


def _ordered_modalities(source, target, direction: str):
    if direction == "rna_to_atac":
        return source, target, 1, 3
    if direction == "atac_to_rna":
        return target, source, 0, 2
    raise ValueError(f"Unsupported translation direction: {direction!r}")


@torch.no_grad()
def integrate_and_extract(
    model,
    source,
    initial_target,
    *,
    direction: str = "rna_to_atac",
    steps: int,
    condition,
):
    """Integrate one direction and return the generated latent and target stream.

    ``source`` is always expressed in the named source modality. The helper
    reorders RNA and ATAC tensors only at the fixed model call boundary.
    """
    if steps <= 0:
        raise ValueError("steps must be positive")
    target = initial_target
    grid = torch.linspace(0.0, 1.0, steps + 1, device=source.device)
    for index in range(steps):
        time = torch.full((source.shape[0],), grid[index], device=source.device)
        x_rna, x_atac, velocity_index, _ = _ordered_modalities(
            source, target, direction
        )
        velocities = model(
            x_rna,
            x_atac,
            time,
            condition=condition,
            force_drop_ids=False,
            mode=direction,
        )
        target = target + (grid[index + 1] - grid[index]) * velocities[velocity_index]
    final_time = torch.ones(source.shape[0], device=source.device)
    x_rna, x_atac, _, hidden_index = _ordered_modalities(source, target, direction)
    hidden = forward_with_hidden(
        model, x_rna, x_atac, final_time, condition=condition, mode=direction
    )[hidden_index]
    return target, hidden


@torch.no_grad()
def verify_forward_equivalence(
    model,
    device: str | torch.device = "cpu",
    direction: str = "rna_to_atac",
) -> float:
    """Return the maximum difference from the model's ordinary forward path."""
    device = torch.device(device)
    generator = torch.Generator(device=device).manual_seed(42001)
    shape = (2, model.seq_len, model.input_proj_rna.in_features)
    rna = torch.randn(shape, generator=generator, device=device)
    atac = torch.randn(shape, generator=generator, device=device)
    time = torch.tensor([0.25, 0.75], device=device)
    n_types = model.class_embeddings["cell_type"].num_embeddings - int(
        model.cfg_dropout_prob > 0
    )
    condition = {"cell_type": torch.tensor([0, min(1, n_types - 1)], device=device)}
    expected = model(
        rna, atac, time, condition=condition, force_drop_ids=False, mode=direction
    )
    observed = forward_with_hidden(
        model, rna, atac, time, condition=condition, mode=direction
    )[:2]
    return max(float((a - b).abs().max()) for a, b in zip(expected, observed))
