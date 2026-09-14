"""Bidirectional donor-aware targeted projection (DATP)."""

from .directions import ATAC_TO_RNA, RNA_TO_ATAC, DirectionSpec, get_direction
from .inference import (
    load_datp,
    load_flow_hidden,
    load_flow_hidden_with_context,
    predict_targeted_atac,
    predict_targeted_modality,
    predict_targeted_rna,
)
from .reverse import fit_reverse_datp_bundle
from .training import fit_datp_bundle

__all__ = [
    "ATAC_TO_RNA",
    "RNA_TO_ATAC",
    "DirectionSpec",
    "fit_datp_bundle",
    "fit_reverse_datp_bundle",
    "get_direction",
    "load_datp",
    "load_flow_hidden",
    "load_flow_hidden_with_context",
    "predict_targeted_atac",
    "predict_targeted_modality",
    "predict_targeted_rna",
]
