"""Direction contracts for donor-aware targeted projection (DATP)."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DirectionSpec:
    name: str
    source_modality: str
    target_modality: str
    flow_mode: str
    hidden_stream: str
    external_cohort: str
    baseline_clock: str
    final_clock: str = "Fusion"


RNA_TO_ATAC = DirectionSpec(
    name="rna_to_atac",
    source_modality="rna",
    target_modality="atac",
    flow_mode="rna_to_atac",
    hidden_stream="atac",
    external_cohort="rna_only27",
    baseline_clock="RNA",
)

ATAC_TO_RNA = DirectionSpec(
    name="atac_to_rna",
    source_modality="atac",
    target_modality="rna",
    flow_mode="atac_to_rna",
    hidden_stream="rna",
    external_cohort="atac_only7",
    baseline_clock="ATAC",
)

DIRECTIONS = {item.name: item for item in (RNA_TO_ATAC, ATAC_TO_RNA)}


def get_direction(value: str | DirectionSpec) -> DirectionSpec:
    if isinstance(value, DirectionSpec):
        return value
    try:
        return DIRECTIONS[str(value)]
    except KeyError as error:
        raise ValueError(
            f"Unknown DATP direction {value!r}; expected {sorted(DIRECTIONS)}"
        ) from error

