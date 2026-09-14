"""Bidirectional DATP and frozen immune-age downstream workflows."""

from .clocks import make_fusion_frame, predict_immune_age, prepare_clock_input
from .datp import (
    ATAC_TO_RNA,
    RNA_TO_ATAC,
    fit_datp_bundle,
    load_datp,
    predict_targeted_atac,
    predict_targeted_modality,
    predict_targeted_rna,
)
from .workflow import (
    evaluate_external_age,
    evaluate_paired_age,
    paired_test_donor_metadata,
    predict_paired_target,
    preflight,
)

__all__ = [
    "ATAC_TO_RNA",
    "RNA_TO_ATAC",
    "evaluate_external_age",
    "evaluate_paired_age",
    "fit_datp_bundle",
    "load_datp",
    "make_fusion_frame",
    "paired_test_donor_metadata",
    "predict_immune_age",
    "predict_paired_target",
    "predict_targeted_atac",
    "predict_targeted_modality",
    "predict_targeted_rna",
    "preflight",
    "prepare_clock_input",
]
