try:
    from enum import StrEnum
except ImportError:
    from enum import Enum

    class StrEnum(str, Enum):
        """Python 3.10-compatible subset of enum.StrEnum."""

        def __str__(self) -> str:
            return self.value


class DatasetEnum(StrEnum):
    """Enum for dataset keys."""

    TISSUE = "tissue"
    TISSUE_GENERAL = "tissue_general"
    DONOR_ID = "donor_id"
    ASSAY = "assay"
    SUSPENSION_TYPE = "suspension_type"
    DATASET_ID = "dataset_id"
    NNZ = "nnz"
    RAW_SUM = "raw_sum"
    N_MEASURED_VARS = "n_measured_vars"
    SEX = "sex"
    DISEASE = "disease"
    DEVELOPMENT_STAGE = "development_stage"
    CELL_TYPE = "cell_type"


class ModelEnum(StrEnum):
    """Enum for model keys."""

    COUNTS = "counts"
    GENES = "genes"
    LIBRARY_SIZE = "library_size"
    GENES_SUBSET = "genes_subset"
    COUNTS_SUBSET = "counts_subset"
    RNA_COUNTS = "rna_counts"
    RNA_GENES = "rna_genes"
    RNA_LIBRARY_SIZE = "rna_library_size"
    RNA_COUNTS_SUBSET = "rna_counts_subset"
    RNA_GENES_SUBSET = "rna_genes_subset"
    ATAC_VALUES = "atac_values"
    ATAC_PEAKS = "atac_peaks"
    ATAC_VALUES_SUBSET = "atac_values_subset"
    ATAC_PEAKS_SUBSET = "atac_peaks_subset"
    CELL_TYPE = "cell_type"


class LossEnum(StrEnum):
    """Enum for model keys."""

    LLH_LOSS = "llh"
    DIFF_LOSS = "diff"
    CR_LOSS = "cr"
