from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.stats import pearsonr, spearmanr


@dataclass
class StreamingRnaReconstructionMetrics:
    """Exact full-axis RNA metrics from bounded-memory sufficient statistics."""

    n_features: int
    n_rows: int = 0
    n_values: int = 0
    zero_values: int = 0
    squared_error_sum: float = 0.0
    true_sum: float = 0.0
    pred_sum: float = 0.0
    true_square_sum: float = 0.0
    pred_square_sum: float = 0.0
    cross_sum: float = 0.0
    cell_pcc_sum: float = 0.0
    valid_cell_pcc: int = 0
    true_gene_sum: np.ndarray = field(init=False, repr=False)
    pred_gene_sum: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.true_gene_sum = np.zeros(self.n_features, dtype=np.float64)
        self.pred_gene_sum = np.zeros(self.n_features, dtype=np.float64)

    def update(self, true_counts: np.ndarray, pred_counts: np.ndarray) -> None:
        true_counts = np.asarray(true_counts, dtype=np.float32)
        pred_counts = np.asarray(pred_counts, dtype=np.float32)
        if true_counts.shape != pred_counts.shape or true_counts.shape[1] != self.n_features:
            raise ValueError(
                f"RNA shapes must both be (*, {self.n_features}), got "
                f"{true_counts.shape} and {pred_counts.shape}"
            )
        true_library = np.maximum(true_counts.sum(axis=1, keepdims=True), 1.0)
        pred_library = np.maximum(pred_counts.sum(axis=1, keepdims=True), 1e-8)
        true_log = np.log1p((true_counts / true_library) * 10_000)
        pred_log = np.log1p((pred_counts / pred_library) * 10_000)
        difference = true_log - pred_log

        self.n_rows += int(true_counts.shape[0])
        self.n_values += int(true_counts.size)
        self.zero_values += int(np.count_nonzero(true_counts == 0))
        self.squared_error_sum += float(
            np.einsum("ij,ij->", difference, difference, dtype=np.float64)
        )
        self.true_sum += float(true_log.sum(dtype=np.float64))
        self.pred_sum += float(pred_log.sum(dtype=np.float64))
        self.true_square_sum += float(
            np.einsum("ij,ij->", true_log, true_log, dtype=np.float64)
        )
        self.pred_square_sum += float(
            np.einsum("ij,ij->", pred_log, pred_log, dtype=np.float64)
        )
        self.cross_sum += float(
            np.einsum("ij,ij->", true_log, pred_log, dtype=np.float64)
        )
        self.true_gene_sum += true_counts.sum(axis=0, dtype=np.float64)
        self.pred_gene_sum += pred_counts.sum(axis=0, dtype=np.float64)

        true_centered = true_log - true_log.mean(axis=1, keepdims=True)
        pred_centered = pred_log - pred_log.mean(axis=1, keepdims=True)
        numerator = np.einsum(
            "ij,ij->i", true_centered, pred_centered, dtype=np.float64
        )
        denominator = np.sqrt(
            np.einsum("ij,ij->i", true_centered, true_centered, dtype=np.float64)
            * np.einsum("ij,ij->i", pred_centered, pred_centered, dtype=np.float64)
        )
        valid = denominator > 0
        self.cell_pcc_sum += float(np.sum(numerator[valid] / denominator[valid]))
        self.valid_cell_pcc += int(valid.sum())

    def compute(self) -> dict[str, object]:
        if self.n_values == 0:
            raise RuntimeError("No RNA values were accumulated")
        covariance = self.cross_sum - self.true_sum * self.pred_sum / self.n_values
        true_ss = self.true_square_sum - self.true_sum**2 / self.n_values
        pred_ss = self.pred_square_sum - self.pred_sum**2 / self.n_values
        denominator = np.sqrt(max(true_ss, 0.0) * max(pred_ss, 0.0))
        true_mean = self.true_gene_sum / self.n_rows
        pred_mean = self.pred_gene_sum / self.n_rows
        valid_gene_profile = np.std(true_mean) > 0 and np.std(pred_mean) > 0
        return {
            "eval_shape": [self.n_rows, self.n_features],
            "log1p_cpm_mse": float(self.squared_error_sum / self.n_values),
            "log1p_cpm_flat_pearson": (
                float(covariance / denominator) if denominator > 0 else float("nan")
            ),
            "mean_cell_pcc": (
                float(self.cell_pcc_sum / self.valid_cell_pcc)
                if self.valid_cell_pcc
                else float("nan")
            ),
            "gene_mean_pearson_on_sampled_tokens": (
                float(pearsonr(true_mean, pred_mean)[0])
                if valid_gene_profile
                else float("nan")
            ),
            "gene_mean_spearman_on_sampled_tokens": (
                float(spearmanr(true_mean, pred_mean)[0])
                if valid_gene_profile
                else float("nan")
            ),
            "true_zero_rate_sampled_tokens": float(self.zero_values / self.n_values),
            "calculation": "exact streaming sufficient statistics",
        }


@dataclass
class StreamingAtacReconstructionMetrics:
    """Bounded-memory ATAC metrics; threshold metrics exact, AUC/AP histogram-based."""

    n_features: int
    histogram_bins: int = 16_384
    n_rows: int = 0
    n_values: int = 0
    positives: int = 0
    predicted_positives: int = 0
    true_positives: int = 0
    probability_sum: float = 0.0
    open_probability_sum: float = 0.0
    closed_probability_sum: float = 0.0
    positive_histogram: np.ndarray = field(init=False, repr=False)
    negative_histogram: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.positive_histogram = np.zeros(self.histogram_bins, dtype=np.int64)
        self.negative_histogram = np.zeros(self.histogram_bins, dtype=np.int64)

    def update(self, probabilities: np.ndarray, targets: np.ndarray) -> None:
        probabilities = np.asarray(probabilities, dtype=np.float32)
        targets = np.asarray(targets) >= 0.5
        if probabilities.shape != targets.shape or probabilities.shape[1] != self.n_features:
            raise ValueError(
                f"ATAC shapes must both be (*, {self.n_features}), got "
                f"{probabilities.shape} and {targets.shape}"
            )
        prediction = probabilities >= 0.5
        positives = int(np.count_nonzero(targets))
        predicted_positives = int(np.count_nonzero(prediction))
        true_positives = int(np.count_nonzero(prediction & targets))
        probability_sum = float(probabilities.sum(dtype=np.float64))
        bins = np.minimum(
            (probabilities * self.histogram_bins).astype(np.int32),
            self.histogram_bins - 1,
        )
        self.positive_histogram += np.bincount(
            bins[targets], minlength=self.histogram_bins
        )
        self.negative_histogram += np.bincount(
            bins[~targets], minlength=self.histogram_bins
        )
        self.n_rows += int(probabilities.shape[0])
        self.n_values += int(probabilities.size)
        self.positives += positives
        self.predicted_positives += predicted_positives
        self.true_positives += true_positives
        self.probability_sum += probability_sum
        self.open_probability_sum += float(probabilities[targets].sum(dtype=np.float64))
        self.closed_probability_sum += probability_sum - float(
            probabilities[targets].sum(dtype=np.float64)
        )

    def _auc(self) -> float:
        positives = int(self.positive_histogram.sum())
        negatives = int(self.negative_histogram.sum())
        negatives_below = np.cumsum(self.negative_histogram) - self.negative_histogram
        concordant = np.sum(
            self.positive_histogram
            * (negatives_below + 0.5 * self.negative_histogram),
            dtype=np.float64,
        )
        return float(concordant / (positives * negatives))

    def _average_precision(self) -> float:
        pos = self.positive_histogram[::-1]
        neg = self.negative_histogram[::-1]
        cumulative_pos = np.cumsum(pos, dtype=np.int64)
        cumulative_total = np.cumsum(pos + neg, dtype=np.int64)
        precision = np.divide(
            cumulative_pos,
            cumulative_total,
            out=np.zeros_like(cumulative_pos, dtype=np.float64),
            where=cumulative_total > 0,
        )
        return float(np.sum(precision * pos, dtype=np.float64) / self.positives)

    def compute(self) -> dict[str, object]:
        if self.n_values == 0:
            raise RuntimeError("No ATAC values were accumulated")
        tp = self.true_positives
        fp = self.predicted_positives - tp
        fn = self.positives - tp
        negatives = self.n_values - self.positives
        tn = negatives - fp
        sensitivity = tp / max(tp + fn, 1)
        specificity = tn / max(tn + fp, 1)
        precision = tp / max(tp + fp, 1)
        return {
            "eval_shape": [self.n_rows, self.n_features],
            "true_open_rate_balanced_eval": float(self.positives / self.n_values),
            "pred_open_rate_threshold_0_5": float(self.predicted_positives / self.n_values),
            "mean_pred_probability": float(self.probability_sum / self.n_values),
            "mean_pred_probability_open_peaks": float(
                self.open_probability_sum / max(self.positives, 1)
            ),
            "mean_pred_probability_closed_peaks": float(
                self.closed_probability_sum / max(negatives, 1)
            ),
            "accuracy": float((tp + tn) / self.n_values),
            "balanced_accuracy": float((sensitivity + specificity) / 2),
            "auc_roc": self._auc(),
            "average_precision": self._average_precision(),
            "precision": float(precision),
            "recall_sensitivity": float(sensitivity),
            "specificity": float(specificity),
            "f1": float(2 * precision * sensitivity / max(precision + sensitivity, 1e-12)),
            "auc_ap_method": f"streaming histogram ({self.histogram_bins} bins)",
        }

