"""Trainable wrapper around the deterministic Stage-1 multimodal AE."""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
import torch.nn as nn

from .constants import ModelEnum
from .distributions import log_nb_positive
from .ae import MultimodalTransformerAE


def normalize_stage1_state_dict(
    state_dict: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], bool]:
    """Map historical ``vae_model.*`` keys to the deterministic AE namespace."""
    legacy_prefix = "vae_model."
    current_prefix = "ae_model."
    has_legacy = any(key.startswith(legacy_prefix) for key in state_dict)
    has_current = any(key.startswith(current_prefix) for key in state_dict)
    if has_legacy and has_current:
        raise ValueError("Stage-1 checkpoint mixes legacy and current model namespaces")
    if not has_legacy:
        return state_dict, False
    remapped = {
        current_prefix + key[len(legacy_prefix) :]
        if key.startswith(legacy_prefix)
        else key: value
        for key, value in state_dict.items()
    }
    return remapped, True


class MultimodalAE(nn.Module):
    """Stage-1 wrapper retained by both training and frozen inference.

    The loss contains RNA and ATAC reconstruction terms only. There is no
    variational posterior, stochastic reparameterization or KL penalty.
    """

    def __init__(
        self,
        ae_model: MultimodalTransformerAE,
        ae_optimizer: Callable[[Any], torch.optim.Optimizer],
        ae_scheduler: Callable[[int], float] | None = None,
    ):
        super().__init__()
        self.ae_model = ae_model
        self.ae_optimizer = ae_optimizer
        self.ae_scheduler = ae_scheduler

    def configure_optimizers(self) -> dict[str, Any]:
        parameters = [parameter for parameter in self.ae_model.parameters() if parameter.requires_grad]
        if not parameters:
            return {}
        result: dict[str, Any] = {"optimizer": self.ae_optimizer(parameters)}
        if self.ae_scheduler is not None:
            result["lr_scheduler"] = {
                "scheduler": torch.optim.lr_scheduler.LambdaLR(
                    result["optimizer"], self.ae_scheduler
                ),
                "interval": "step",
            }
        return result

    def forward(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[dict[str, dict[str, torch.Tensor]], dict[str, torch.Tensor]]:
        return self.ae_model(
            rna_counts=batch[ModelEnum.RNA_COUNTS.value],
            rna_genes=batch[ModelEnum.RNA_GENES.value],
            rna_library_size=batch[ModelEnum.RNA_LIBRARY_SIZE.value],
            atac_values=batch.get(ModelEnum.ATAC_VALUES.value),
            atac_peaks=batch.get(ModelEnum.ATAC_PEAKS.value),
            rna_counts_subset=batch.get(ModelEnum.RNA_COUNTS_SUBSET.value),
            rna_genes_subset=batch.get(ModelEnum.RNA_GENES_SUBSET.value),
            atac_values_subset=batch.get(ModelEnum.ATAC_VALUES_SUBSET.value),
            atac_peaks_subset=batch.get(ModelEnum.ATAC_PEAKS_SUBSET.value),
        )

    def loss(
        self,
        batch: dict[str, torch.Tensor],
        params: dict[str, dict[str, torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        rna_counts = batch[ModelEnum.RNA_COUNTS.value].float()
        rna_params = params["rna"]
        rna_reconstruction = -log_nb_positive(
            rna_counts, rna_params["mu"], rna_params["theta"]
        )
        output = {"rna_loss": rna_reconstruction.sum(dim=1).mean()}
        if "atac" in params:
            atac_values = batch[ModelEnum.ATAC_VALUES.value].float().clamp(0.0, 1.0)
            atac_log_prob = self.ae_model.atac_decoder_head.log_prob(
                atac_values, params["atac"]["probs"]
            )
            output["atac_loss"] = (-atac_log_prob).sum(dim=1).mean()
        output["llh"] = output["rna_loss"] + output.get(
            "atac_loss", torch.zeros_like(output["rna_loss"])
        )
        for name, value in output.items():
            if not torch.isfinite(value).all():
                raise ValueError(f"Invalid multimodal loss detected in {name}: {value}")
        return output

    @torch.no_grad()
    def inference(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        params, latents = self.forward(batch)
        output = {
            "rna_mu": params["rna"]["mu"].detach().cpu(),
            "z_rna": latents["rna"].detach().cpu(),
        }
        if "atac" in params:
            output["atac_probs"] = params["atac"]["probs"].detach().cpu()
            output["z_atac"] = latents["atac"].detach().cpu()
        return output
