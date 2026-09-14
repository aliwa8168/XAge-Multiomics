"""Memory-bounded paired metrics for dense full-feature RNA/ATAC outputs."""
from __future__ import annotations

from dataclasses import dataclass, field
import csv
import json
import os
from pathlib import Path

import h5py
import numpy as np
from scipy import sparse
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA


SUMMARY_SCHEMA = "cima_translation_full_feature_v1"


def _rna_log1p_cpm(values: np.ndarray) -> np.ndarray:
    library = np.maximum(values.sum(axis=1, keepdims=True), 1.0)
    return np.log1p(values / library * 10_000.0).astype(np.float32, copy=False)


def pearson_or_nan(left: np.ndarray, right: np.ndarray) -> float:
    if left.size == 0 or np.std(left) == 0 or np.std(right) == 0:
        return float("nan")
    return float(pearsonr(left, right)[0])


def spearman_or_nan(left: np.ndarray, right: np.ndarray) -> float:
    if left.size == 0 or np.std(left) == 0 or np.std(right) == 0:
        return float("nan")
    return float(spearmanr(left, right)[0])


def _fit_pca_space(
    real: np.ndarray, generated: np.ndarray, n_components: int
) -> tuple[np.ndarray, np.ndarray]:
    joined = np.concatenate([real, generated]).astype(np.float32, copy=False)
    n_components = min(n_components, joined.shape[0] - 1, joined.shape[1])
    embedded = PCA(n_components=n_components, random_state=0).fit_transform(joined)
    return embedded[: len(real)], embedded[len(real) :]


def _symmetric_matrix_sqrt(matrix: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    matrix = (matrix + matrix.T) * 0.5
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    eigenvalues = np.clip(eigenvalues, a_min=eps, a_max=None)
    return (eigenvectors * np.sqrt(eigenvalues)) @ eigenvectors.T


def frechet_distance(
    real: np.ndarray, generated: np.ndarray, eps: float = 1e-6
) -> float:
    real64 = np.asarray(real, dtype=np.float64)
    generated64 = np.asarray(generated, dtype=np.float64)
    if real64.ndim != 2 or generated64.ndim != 2:
        raise ValueError("Frechet distance requires 2D matrices")
    if real64.shape[1] != generated64.shape[1]:
        raise ValueError("Frechet distance feature dimensions do not match")
    mean_delta = real64.mean(axis=0) - generated64.mean(axis=0)
    covariance_real = np.atleast_2d(np.cov(real64, rowvar=False))
    covariance_generated = np.atleast_2d(np.cov(generated64, rowvar=False))
    identity = np.eye(covariance_real.shape[0], dtype=np.float64)
    covariance_real += identity * eps
    covariance_generated += identity * eps
    root_real = _symmetric_matrix_sqrt(covariance_real, eps)
    covariance_mean = _symmetric_matrix_sqrt(
        root_real @ covariance_generated @ root_real, eps
    )
    value = mean_delta @ mean_delta + np.trace(
        covariance_real + covariance_generated - 2.0 * covariance_mean
    )
    return float(max(value, 0.0))


def mmd_rbf_scdiffusion(real: np.ndarray, generated: np.ndarray) -> float:
    return _exact_blockwise_multiscale_mmd(real, generated)


def mmd_rbf(real: np.ndarray, generated: np.ndarray) -> float:
    return mmd_rbf_scdiffusion(real, generated)


class _CanonicalMetricNamespace:
    _rna_log1p_cpm = staticmethod(_rna_log1p_cpm)
    _fit_pca_space = staticmethod(_fit_pca_space)
    mmd_rbf_scdiffusion = staticmethod(mmd_rbf_scdiffusion)
    mmd_rbf = staticmethod(mmd_rbf)
    frechet_distance = staticmethod(frechet_distance)
    pearson_or_nan = staticmethod(pearson_or_nan)
    spearman_or_nan = staticmethod(spearman_or_nan)


CANONICAL_METRICS = _CanonicalMetricNamespace()


def summarize_distribution_without_w2_lisi(
    true_matrix: np.ndarray,
    generated_matrix: np.ndarray,
    modality: str,
    pca_components: int,
    canonical_metrics,
) -> dict[str, float]:
    """Canonical distribution summary excluding W2 and all LISI variants."""
    if modality == "rna":
        true_features = canonical_metrics._rna_log1p_cpm(true_matrix)
        generated_features = canonical_metrics._rna_log1p_cpm(generated_matrix)
    elif modality == "atac":
        true_features = true_matrix.astype(np.float32, copy=False)
        generated_features = generated_matrix.astype(np.float32, copy=False)
    else:
        raise ValueError(f"Unknown modality: {modality}")

    true_pca, generated_pca = canonical_metrics._fit_pca_space(
        true_features, generated_features, pca_components
    )
    true_bulk = true_features.mean(axis=0)
    generated_bulk = generated_features.mean(axis=0)
    return {
        "mmd_rbf": canonical_metrics.mmd_rbf_scdiffusion(
            true_pca, generated_pca
        ),
        "frechet_distance": canonical_metrics.frechet_distance(
            true_pca, generated_pca
        ),
        "mmd_rbf_legacy_median_single_kernel": canonical_metrics.mmd_rbf(
            true_pca, generated_pca
        ),
        "bulk_pearson": canonical_metrics.pearson_or_nan(
            true_bulk, generated_bulk
        ),
        "bulk_spearman": canonical_metrics.spearman_or_nan(
            true_bulk, generated_bulk
        ),
        "feature_mse": float(
            np.mean((true_features - generated_features) ** 2)
        ),
    }


def _exact_blockwise_multiscale_mmd(
    real: np.ndarray,
    generated: np.ndarray,
    block_size: int = 2048,
    device: str = "cuda",
    kernel_mul: float = 2.0,
    kernel_num: int = 5,
) -> float:
    """Exact biased multi-scale RBF MMD without materializing the Gram matrix."""
    import torch

    if real.shape != generated.shape or real.ndim != 2:
        raise ValueError(f"MMD inputs must align, got {real.shape} and {generated.shape}")
    n = int(real.shape[0])
    if n == 0:
        raise ValueError("MMD requires at least one row")
    target_device = torch.device(device if torch.cuda.is_available() else "cpu")
    if target_device.type == "cpu":
        block_size = min(block_size, 512)

    combined = np.concatenate([real, generated], axis=0).astype(np.float64, copy=False)
    m = int(combined.shape[0])
    combined_sum = combined.sum(axis=0, dtype=np.float64)
    combined_square_sum = float(np.einsum("ij,ij->", combined, combined, dtype=np.float64))
    pairwise_square_sum = 2.0 * m * combined_square_sum - 2.0 * float(
        np.dot(combined_sum, combined_sum)
    )
    bandwidth = max(pairwise_square_sum / max(m * m - m, 1), 1e-12)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidths = [bandwidth * (kernel_mul**index) for index in range(kernel_num)]

    def kernel_sum(left: np.ndarray, right: np.ndarray, symmetric: bool) -> float:
        total = 0.0
        phase = "symmetric" if symmetric else "cross"
        n_left_blocks = (len(left) + block_size - 1) // block_size
        for left_start in range(0, len(left), block_size):
            left_np = left[left_start : left_start + block_size]
            left_tensor = torch.as_tensor(left_np, dtype=torch.float32, device=target_device)
            left_norm = left_tensor.square().sum(dim=1, keepdim=True)
            right_limit = left_start + 1 if symmetric else len(right)
            for right_start in range(0, right_limit, block_size):
                right_np = right[right_start : right_start + block_size]
                right_tensor = torch.as_tensor(right_np, dtype=torch.float32, device=target_device)
                distances = (
                    left_norm
                    + right_tensor.square().sum(dim=1).unsqueeze(0)
                    - 2.0 * left_tensor @ right_tensor.T
                ).clamp_min_(0.0)
                block_total = 0.0
                for current_bandwidth in bandwidths:
                    block_total += float(
                        torch.exp(-distances / current_bandwidth).sum(dtype=torch.float64).item()
                    )
                if symmetric and right_start != left_start:
                    block_total *= 2.0
                total += block_total
            left_block = left_start // block_size + 1
            if left_block == 1 or left_block == n_left_blocks or left_block % 5 == 0:
                print(
                    f"mmd_{phase}_left_blocks={left_block}/{n_left_blocks}",
                    flush=True,
                )
        return total

    xx = kernel_sum(real, real, symmetric=True)
    yy = kernel_sum(generated, generated, symmetric=True)
    xy = kernel_sum(real, generated, symmetric=False)
    denominator = float(n * n)
    return float(xx / denominator + yy / denominator - 2.0 * xy / denominator)


class FullTestDistributionAccumulator:
    """Collect bounded full-test sketches and compute PCA-50 FD/exact MMD."""

    def __init__(
        self,
        n_rows: int,
        n_features: int,
        modality: str,
        pca_components: int = 50,
        sketch_components: int = 512,
        seed: int = 123,
        mmd_block_size: int = 2048,
        feature_order: np.ndarray | None = None,
    ) -> None:
        if modality not in {"rna", "atac"}:
            raise ValueError(f"Unsupported modality: {modality}")
        if n_rows <= 1 or n_features <= 0:
            raise ValueError("Distribution dimensions must be positive")
        if pca_components <= 0 or sketch_components < pca_components:
            raise ValueError("Sketch width must be at least the PCA width")
        self.n_rows = int(n_rows)
        self.n_features = int(n_features)
        self.modality = modality
        self.pca_components = int(pca_components)
        self.sketch_components = int(sketch_components)
        self.seed = int(seed)
        self.mmd_block_size = int(mmd_block_size)
        self.rows_seen = 0
        self.true_sketch = np.empty((n_rows, sketch_components), dtype=np.float32)
        self.generated_sketch = np.empty((n_rows, sketch_components), dtype=np.float32)

        rng = np.random.default_rng(seed + (0 if modality == "rna" else 1_000_003))
        columns = rng.integers(0, sketch_components, size=n_features, dtype=np.int32)
        signs = rng.choice(np.asarray([-1.0, 1.0], dtype=np.float32), size=n_features)
        if feature_order is not None:
            feature_order = np.asarray(feature_order, dtype=np.int64)
            if feature_order.shape != (n_features,) or not np.array_equal(
                np.sort(feature_order), np.arange(n_features, dtype=np.int64)
            ):
                raise ValueError("feature_order must be a complete feature permutation")
            columns = columns[feature_order]
            signs = signs[feature_order]
        rows = np.arange(n_features, dtype=np.int64)
        self.projection = sparse.csr_matrix(
            (signs, (rows, columns)), shape=(n_features, sketch_components)
        )

    def update(self, true_values: np.ndarray, generated_values: np.ndarray) -> None:
        true_values = np.asarray(true_values, dtype=np.float32)
        generated_values = np.asarray(generated_values, dtype=np.float32)
        if true_values.shape != generated_values.shape:
            raise ValueError("Distribution matrices do not align")
        if true_values.ndim != 2 or true_values.shape[1] != self.n_features:
            raise ValueError(f"Unexpected distribution batch shape: {true_values.shape}")
        end = self.rows_seen + int(true_values.shape[0])
        if end > self.n_rows:
            raise ValueError("Distribution accumulator received too many rows")
        if self.modality == "rna":
            true_values, generated_values = StreamingRnaMetrics.log1p_cpm(
                true_values, generated_values
            )
        true_projected = sparse.csr_matrix(true_values) @ self.projection
        generated_projected = sparse.csr_matrix(generated_values) @ self.projection
        if sparse.issparse(true_projected):
            true_projected = true_projected.toarray()
        if sparse.issparse(generated_projected):
            generated_projected = generated_projected.toarray()
        self.true_sketch[self.rows_seen : end] = np.asarray(true_projected, dtype=np.float32)
        self.generated_sketch[self.rows_seen : end] = np.asarray(
            generated_projected, dtype=np.float32
        )
        self.rows_seen = end

    def compute(
        self, canonical_metrics=CANONICAL_METRICS, device: str = "cuda"
    ) -> dict[str, object]:
        if self.rows_seen != self.n_rows:
            raise RuntimeError(
                f"Distribution rows incomplete: {self.rows_seen}/{self.n_rows}"
            )
        print(
            f"distribution_pca modality={self.modality} cells={self.n_rows} "
            f"sketch={self.sketch_components} pcs={self.pca_components}",
            flush=True,
        )
        combined = np.concatenate([self.true_sketch, self.generated_sketch], axis=0)
        embedding = PCA(
            n_components=self.pca_components,
            svd_solver="randomized",
            random_state=self.seed,
        ).fit_transform(combined)
        true_embedding = embedding[: self.n_rows].astype(np.float32, copy=False)
        generated_embedding = embedding[self.n_rows :].astype(np.float32, copy=False)
        result = {
            "mmd_rbf": _exact_blockwise_multiscale_mmd(
                true_embedding,
                generated_embedding,
                block_size=self.mmd_block_size,
                device=device,
            ),
            "frechet_distance": canonical_metrics.frechet_distance(
                true_embedding, generated_embedding
            ),
            "distribution_eval_cells": self.n_rows,
            "projection": (
                f"all-feature deterministic CountSketch-{self.sketch_components} + "
                f"joint PCA-{self.pca_components}"
            ),
            "mmd_calculation": "exact full-test blockwise biased multi-scale RBF MMD",
        }
        return result


def export_full_test_umap_sketches(
    output_path: Path,
    *,
    source_row_index: np.ndarray,
    rna_accumulator: FullTestDistributionAccumulator,
    atac_accumulator: FullTestDistributionAccumulator,
    model_name: str,
    checkpoint: str,
) -> dict[str, object]:
    """Save sampled full-test sketches for a shared cross-model UMAP."""
    source_row_index = np.asarray(source_row_index, dtype=np.int64)
    expected_rows = len(source_row_index)
    for accumulator in (rna_accumulator, atac_accumulator):
        if accumulator.rows_seen != expected_rows:
            raise RuntimeError(
                f"Incomplete UMAP sketch: {accumulator.rows_seen}/{expected_rows}"
            )
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with h5py.File(temporary, "w") as handle:
        handle.attrs["schema_version"] = "full_test_sampled_countsketch_v1"
        handle.attrs["model_name"] = model_name
        handle.attrs["checkpoint"] = str(checkpoint)
        handle.attrs["n_cells"] = expected_rows
        handle.attrs["sampling_representation"] = (
            "RNA negative-binomial sample; ATAC Bernoulli sample"
        )
        handle.create_dataset("obs/source_row_index", data=source_row_index)
        for modality, accumulator in (
            ("rna", rna_accumulator),
            ("atac", atac_accumulator),
        ):
            group = handle.create_group(modality)
            group.attrs["n_input_features"] = accumulator.n_features
            group.attrs["sketch_components"] = accumulator.sketch_components
            group.attrs["seed"] = accumulator.seed
            group.attrs["preprocessing"] = (
                "log1p-CPM before CountSketch" if modality == "rna"
                else "binary sampled accessibility before CountSketch"
            )
            chunks = (min(256, expected_rows), accumulator.sketch_components)
            group.create_dataset(
                "real", data=accumulator.true_sketch, chunks=chunks, compression="lzf"
            )
            group.create_dataset(
                "predicted",
                data=accumulator.generated_sketch,
                chunks=chunks,
                compression="lzf",
            )
        handle.attrs["contract_json"] = json.dumps(
            {
                "cell_sampling_applied": False,
                "feature_sampling_applied": False,
                "n_cells": expected_rows,
                "rna_input_features": rna_accumulator.n_features,
                "atac_input_features": atac_accumulator.n_features,
            }
        )
    os.replace(temporary, output_path)
    return {
        "path": str(output_path),
        "n_cells": expected_rows,
        "sketch_components": rna_accumulator.sketch_components,
        "cell_sampling_applied": False,
        "feature_sampling_applied": False,
    }


def markdown_table_without_w2_lisi(report: dict[str, object]) -> str:
    """Render the common report without removed W2/LISI fields."""
    rna = report["atac_to_rna"]
    atac = report["rna_to_atac"]
    rows = [
        ("ATAC -> RNA", "MMD-RBF", rna["distribution"]["mmd_rbf"]),
        ("ATAC -> RNA", "Frechet Distance", rna["distribution"]["frechet_distance"]),
        ("ATAC -> RNA", "Bulk Pearson", rna["distribution"]["bulk_pearson"]),
        ("ATAC -> RNA", "Gene Mean Pearson", rna["token_metrics"]["gene_mean_pearson"]),
        ("ATAC -> RNA", "Mean Cell PCC", rna["token_metrics"]["mean_cell_pcc"]),
        ("ATAC -> RNA", "Gene-wise Pearson (mean)", rna["featurewise_metrics"]["gene_wise_pearson"]["mean"]),
        ("ATAC -> RNA", "Cell-type PCC (macro)", rna["cell_type_metrics"]["pcc"]["macro_mean"]),
        ("RNA -> ATAC", "MMD-RBF", atac["distribution"]["mmd_rbf"]),
        ("RNA -> ATAC", "Frechet Distance", atac["distribution"]["frechet_distance"]),
        ("RNA -> ATAC", "Bulk Pearson", atac["distribution"]["bulk_pearson"]),
        ("RNA -> ATAC", "AUC-ROC", atac["prob_metrics"]["auc_roc"]),
        ("RNA -> ATAC", "Accuracy", atac["sample_metrics"]["accuracy"]),
        ("RNA -> ATAC", "Peak-wise Pearson (mean)", atac["featurewise_metrics"]["peak_wise_pearson"]["mean"]),
        ("RNA -> ATAC", "Cell-type PCC (macro)", atac["cell_type_metrics"]["pcc"]["macro_mean"]),
    ]
    lines = ["| Translation | Metric | Result |", "|---|---|---:|"]
    lines.extend(
        f"| {direction} | {metric} | {float(value):.4f} |"
        for direction, metric, value in rows
    )
    return "\n".join(lines)


def write_full_feature_report(
    output_dir: str | Path,
    report: dict[str, object],
    detail_rows: dict[str, list[dict[str, object]]] | None = None,
) -> dict[str, str]:
    """Write the locked JSON, 14-row Markdown summary, and optional detail TSVs."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    report["summary_schema"] = SUMMARY_SCHEMA
    report_path = output / "translation_full_feature_report.json"
    summary_path = output / "translation_full_feature_summary.md"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    summary_path.write_text(
        markdown_table_without_w2_lisi(report) + "\n", encoding="utf-8"
    )
    paths = {"report": str(report_path), "summary": str(summary_path)}
    for name, rows in (detail_rows or {}).items():
        if not rows:
            continue
        path = output / f"{name}.tsv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        paths[name] = str(path)
    return paths


@dataclass
class StreamingRnaMetrics:
    """Exact bounded-memory equivalent of the canonical RNA token metrics."""

    n_features: int
    n_rows: int = 0
    n_values: int = 0
    squared_error_sum: float = 0.0
    true_log_sum: float = 0.0
    pred_log_sum: float = 0.0
    true_log_square_sum: float = 0.0
    pred_log_square_sum: float = 0.0
    log_cross_sum: float = 0.0
    cell_pcc_sum: float = 0.0
    valid_cell_pcc: int = 0
    true_raw_sum: np.ndarray = field(init=False, repr=False)
    pred_raw_sum: np.ndarray = field(init=False, repr=False)
    true_log_feature_sum: np.ndarray = field(init=False, repr=False)
    pred_log_feature_sum: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.n_features <= 0:
            raise ValueError("n_features must be positive")
        self.true_raw_sum = np.zeros(self.n_features, dtype=np.float64)
        self.pred_raw_sum = np.zeros(self.n_features, dtype=np.float64)
        self.true_log_feature_sum = np.zeros(self.n_features, dtype=np.float64)
        self.pred_log_feature_sum = np.zeros(self.n_features, dtype=np.float64)

    @staticmethod
    def log1p_cpm(
        true_counts: np.ndarray, predicted_counts: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        true_counts = np.asarray(true_counts, dtype=np.float32)
        predicted_counts = np.asarray(predicted_counts, dtype=np.float32)
        true_library = np.maximum(true_counts.sum(axis=1, keepdims=True), 1.0)
        pred_library = np.maximum(predicted_counts.sum(axis=1, keepdims=True), 1e-8)
        true_log = np.log1p((true_counts / true_library) * 10_000)
        pred_log = np.log1p((predicted_counts / pred_library) * 10_000)
        return true_log, pred_log

    def update(
        self, true_counts: np.ndarray, predicted_counts: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        true_counts = np.asarray(true_counts, dtype=np.float32)
        predicted_counts = np.asarray(predicted_counts, dtype=np.float32)
        expected = (true_counts.shape[0], self.n_features)
        if true_counts.shape != expected or predicted_counts.shape != expected:
            raise ValueError(
                f"RNA input mismatch: {true_counts.shape}, {predicted_counts.shape}, "
                f"expected {expected}"
            )
        if not np.isfinite(predicted_counts).all():
            raise ValueError("RNA predictions contain NaN or Inf")

        true_log, pred_log = self.log1p_cpm(true_counts, predicted_counts)
        difference = true_log - pred_log
        self.n_rows += int(true_counts.shape[0])
        self.n_values += int(true_counts.size)
        self.squared_error_sum += float(
            np.einsum("ij,ij->", difference, difference, dtype=np.float64)
        )
        self.true_log_sum += float(true_log.sum(dtype=np.float64))
        self.pred_log_sum += float(pred_log.sum(dtype=np.float64))
        self.true_log_square_sum += float(
            np.einsum("ij,ij->", true_log, true_log, dtype=np.float64)
        )
        self.pred_log_square_sum += float(
            np.einsum("ij,ij->", pred_log, pred_log, dtype=np.float64)
        )
        self.log_cross_sum += float(
            np.einsum("ij,ij->", true_log, pred_log, dtype=np.float64)
        )
        self.true_raw_sum += true_counts.sum(axis=0, dtype=np.float64)
        self.pred_raw_sum += predicted_counts.sum(axis=0, dtype=np.float64)
        self.true_log_feature_sum += true_log.sum(axis=0, dtype=np.float64)
        self.pred_log_feature_sum += pred_log.sum(axis=0, dtype=np.float64)

        true_centered = true_log - true_log.mean(axis=1, keepdims=True)
        pred_centered = pred_log - pred_log.mean(axis=1, keepdims=True)
        denominator = np.sqrt(
            np.einsum("ij,ij->i", true_centered, true_centered, dtype=np.float64)
            * np.einsum("ij,ij->i", pred_centered, pred_centered, dtype=np.float64)
        )
        valid = denominator > 0
        if valid.any():
            numerator = np.einsum(
                "ij,ij->i", true_centered, pred_centered, dtype=np.float64
            )
            self.cell_pcc_sum += float(np.sum(numerator[valid] / denominator[valid]))
            self.valid_cell_pcc += int(valid.sum())
        return true_log, pred_log

    def compute(self) -> dict[str, object]:
        if self.n_values == 0:
            raise RuntimeError("No RNA values were accumulated")
        flat_numerator = (
            self.log_cross_sum
            - self.true_log_sum * self.pred_log_sum / self.n_values
        )
        true_ss = (
            self.true_log_square_sum - self.true_log_sum**2 / self.n_values
        )
        pred_ss = (
            self.pred_log_square_sum - self.pred_log_sum**2 / self.n_values
        )
        flat_denominator = np.sqrt(max(true_ss, 0.0) * max(pred_ss, 0.0))
        true_gene_mean = self.true_raw_sum / self.n_rows
        pred_gene_mean = self.pred_raw_sum / self.n_rows
        gene_valid = np.std(true_gene_mean) > 0 and np.std(pred_gene_mean) > 0
        return {
            "eval_shape": [self.n_rows, self.n_features],
            "mse": float(self.squared_error_sum / self.n_values),
            "flat_pearson": (
                float(flat_numerator / flat_denominator)
                if flat_denominator > 0
                else float("nan")
            ),
            "mean_cell_pcc": (
                float(self.cell_pcc_sum / self.valid_cell_pcc)
                if self.valid_cell_pcc
                else float("nan")
            ),
            "gene_mean_pearson": (
                float(pearsonr(true_gene_mean, pred_gene_mean)[0])
                if gene_valid
                else float("nan")
            ),
            "gene_mean_spearman": (
                float(spearmanr(true_gene_mean, pred_gene_mean)[0])
                if gene_valid
                else float("nan")
            ),
            "calculation": "exact streaming sufficient statistics",
        }

    def distribution_summary(self) -> dict[str, float]:
        if self.n_rows == 0:
            raise RuntimeError("No RNA values were accumulated")
        true_bulk = self.true_log_feature_sum / self.n_rows
        pred_bulk = self.pred_log_feature_sum / self.n_rows
        return {
            "bulk_pearson": float(pearsonr(true_bulk, pred_bulk)[0]),
            "bulk_spearman": float(spearmanr(true_bulk, pred_bulk)[0]),
            "feature_mse": float(self.squared_error_sum / self.n_values),
        }


@dataclass
class StreamingAtacMetrics:
    n_features: int
    histogram_bins: int = 16_384
    threshold: float = 0.5
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
        if self.n_features <= 0:
            raise ValueError("n_features must be positive")
        if self.histogram_bins < 2:
            raise ValueError("histogram_bins must be at least 2")
        self.positive_histogram = np.zeros(self.histogram_bins, dtype=np.int64)
        self.negative_histogram = np.zeros(self.histogram_bins, dtype=np.int64)

    def update(self, probabilities: np.ndarray, targets: np.ndarray) -> None:
        probabilities = np.asarray(probabilities, dtype=np.float32)
        targets = np.asarray(targets)
        if probabilities.ndim != 2 or probabilities.shape[1] != self.n_features:
            raise ValueError(
                f"probability shape {probabilities.shape} does not match "
                f"(*, {self.n_features})"
            )
        if targets.shape != probabilities.shape:
            raise ValueError(
                f"target shape {targets.shape} != probability shape {probabilities.shape}"
            )
        if not np.isfinite(probabilities).all():
            raise ValueError("probabilities contain NaN or Inf")
        if probabilities.min(initial=0.0) < 0.0 or probabilities.max(initial=1.0) > 1.0:
            raise ValueError("probabilities must be in [0, 1]")

        target = targets >= 0.5
        prediction = probabilities >= self.threshold
        batch_values = int(probabilities.size)
        batch_positives = int(np.count_nonzero(target))
        batch_predicted_positives = int(np.count_nonzero(prediction))
        batch_true_positives = int(np.count_nonzero(prediction & target))

        probability_sum = float(probabilities.sum(dtype=np.float64))
        open_probability_sum = float(probabilities[target].sum(dtype=np.float64))
        bins = np.minimum(
            (probabilities * self.histogram_bins).astype(np.int32),
            self.histogram_bins - 1,
        )
        self.positive_histogram += np.bincount(
            bins[target], minlength=self.histogram_bins
        ).astype(np.int64, copy=False)
        self.negative_histogram += np.bincount(
            bins[~target], minlength=self.histogram_bins
        ).astype(np.int64, copy=False)

        self.n_rows += int(probabilities.shape[0])
        self.n_values += batch_values
        self.positives += batch_positives
        self.predicted_positives += batch_predicted_positives
        self.true_positives += batch_true_positives
        self.probability_sum += probability_sum
        self.open_probability_sum += open_probability_sum
        self.closed_probability_sum += probability_sum - open_probability_sum

    def _auc_roc(self) -> float:
        positives = int(self.positive_histogram.sum())
        negatives = int(self.negative_histogram.sum())
        if positives == 0 or negatives == 0:
            return float("nan")
        negatives_below = np.cumsum(self.negative_histogram) - self.negative_histogram
        concordant = np.sum(
            self.positive_histogram
            * (negatives_below + 0.5 * self.negative_histogram),
            dtype=np.float64,
        )
        return float(concordant / (positives * negatives))

    def _average_precision(self) -> float:
        positives = int(self.positive_histogram.sum())
        if positives == 0:
            return float("nan")
        pos_desc = self.positive_histogram[::-1]
        neg_desc = self.negative_histogram[::-1]
        cumulative_pos = np.cumsum(pos_desc, dtype=np.int64)
        cumulative_total = np.cumsum(pos_desc + neg_desc, dtype=np.int64)
        precision = np.divide(
            cumulative_pos,
            cumulative_total,
            out=np.zeros_like(cumulative_pos, dtype=np.float64),
            where=cumulative_total > 0,
        )
        return float(np.sum(precision * pos_desc, dtype=np.float64) / positives)

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
            "auc_roc": self._auc_roc(),
            "average_precision": self._average_precision(),
            "accuracy": float((tp + tn) / self.n_values),
            "balanced_accuracy": float((sensitivity + specificity) / 2),
            "precision": float(precision),
            "recall_sensitivity": float(sensitivity),
            "specificity": float(specificity),
            "f1": float(
                2 * precision * sensitivity / max(precision + sensitivity, 1e-12)
            ),
            "true_open_rate": float(self.positives / self.n_values),
            "pred_open_rate": float(self.predicted_positives / self.n_values),
            "mean_pred_probability": float(self.probability_sum / self.n_values),
            "mean_pred_probability_open_peaks": float(
                self.open_probability_sum / max(self.positives, 1)
            ),
            "mean_pred_probability_closed_peaks": float(
                self.closed_probability_sum / max(negatives, 1)
            ),
            "auc_ap_method": (
                "full-population streaming probability histogram; "
                f"bins={self.histogram_bins}; bin_width={1 / self.histogram_bins:.12g}"
            ),
            "auc_ap_probability_resolution": float(1 / self.histogram_bins),
            "threshold": float(self.threshold),
        }


@dataclass
class StreamingFeaturePearson:
    n_features: int
    n_rows: int = 0
    true_sum: np.ndarray = field(init=False, repr=False)
    pred_sum: np.ndarray = field(init=False, repr=False)
    true_square_sum: np.ndarray = field(init=False, repr=False)
    pred_square_sum: np.ndarray = field(init=False, repr=False)
    cross_sum: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        for name in (
            "true_sum",
            "pred_sum",
            "true_square_sum",
            "pred_square_sum",
            "cross_sum",
        ):
            setattr(self, name, np.zeros(self.n_features, dtype=np.float64))

    def update(self, true_values: np.ndarray, predicted_values: np.ndarray) -> None:
        true_values = np.asarray(true_values, dtype=np.float32)
        predicted_values = np.asarray(predicted_values, dtype=np.float32)
        expected = (true_values.shape[0], self.n_features)
        if true_values.shape != expected or predicted_values.shape != expected:
            raise ValueError(
                f"Feature-wise input mismatch: {true_values.shape}, "
                f"{predicted_values.shape}, expected {expected}"
            )
        self.n_rows += true_values.shape[0]
        self.true_sum += true_values.sum(axis=0, dtype=np.float64)
        self.pred_sum += predicted_values.sum(axis=0, dtype=np.float64)
        self.true_square_sum += np.einsum(
            "ij,ij->j", true_values, true_values, dtype=np.float64
        )
        self.pred_square_sum += np.einsum(
            "ij,ij->j", predicted_values, predicted_values, dtype=np.float64
        )
        self.cross_sum += np.einsum(
            "ij,ij->j", true_values, predicted_values, dtype=np.float64
        )

    def correlations(self) -> np.ndarray:
        if self.n_rows == 0:
            raise RuntimeError("No feature rows were accumulated")
        numerator = self.cross_sum - self.true_sum * self.pred_sum / self.n_rows
        true_ss = self.true_square_sum - self.true_sum**2 / self.n_rows
        pred_ss = self.pred_square_sum - self.pred_sum**2 / self.n_rows
        denominator = np.sqrt(np.maximum(true_ss, 0) * np.maximum(pred_ss, 0))
        result = np.full(self.n_features, np.nan, dtype=np.float64)
        valid = denominator > 0
        result[valid] = numerator[valid] / denominator[valid]
        return np.clip(result, -1.0, 1.0)

    def compute(
        self, feature_names: list[str]
    ) -> tuple[dict[str, object], list[dict[str, object]]]:
        if len(feature_names) != self.n_features:
            raise ValueError("Feature-name count does not match accumulator width")
        correlations = self.correlations()
        valid = np.isfinite(correlations)
        summary = {
            "mean": float(np.nanmean(correlations)) if valid.any() else float("nan"),
            "median": float(np.nanmedian(correlations)) if valid.any() else float("nan"),
            "n_features": int(self.n_features),
            "n_valid_features": int(valid.sum()),
            "n_excluded_zero_variance": int((~valid).sum()),
            "unit": "Pearson correlation across test cells for each feature",
            "calculation": "exact streaming sufficient statistics",
        }
        rows = [
            {
                "feature_index": int(index),
                "feature_name": feature_names[index],
                "pearson": None if not valid[index] else float(correlations[index]),
                "valid": bool(valid[index]),
            }
            for index in range(self.n_features)
        ]
        return summary, rows

    def distribution_summary(self) -> dict[str, float]:
        if self.n_rows == 0:
            raise RuntimeError("No feature rows were accumulated")
        true_bulk = self.true_sum / self.n_rows
        pred_bulk = self.pred_sum / self.n_rows
        squared_error_sum = (
            self.true_square_sum.sum(dtype=np.float64)
            + self.pred_square_sum.sum(dtype=np.float64)
            - 2.0 * self.cross_sum.sum(dtype=np.float64)
        )
        return {
            "bulk_pearson": float(pearsonr(true_bulk, pred_bulk)[0]),
            "bulk_spearman": float(spearmanr(true_bulk, pred_bulk)[0]),
            "feature_mse": float(squared_error_sum / (self.n_rows * self.n_features)),
        }


@dataclass
class StreamingCellTypeProfiles:
    n_features: int
    cell_type_mapping: dict[str, int]
    counts: np.ndarray = field(init=False, repr=False)
    true_sum: np.ndarray = field(init=False, repr=False)
    pred_sum: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        n_types = max(self.cell_type_mapping.values(), default=-1) + 1
        self.counts = np.zeros(n_types, dtype=np.int64)
        self.true_sum = np.zeros((n_types, self.n_features), dtype=np.float64)
        self.pred_sum = np.zeros((n_types, self.n_features), dtype=np.float64)

    def update(
        self,
        true_values: np.ndarray,
        predicted_values: np.ndarray,
        cell_type_codes: np.ndarray,
    ) -> None:
        true_values = np.asarray(true_values, dtype=np.float32)
        predicted_values = np.asarray(predicted_values, dtype=np.float32)
        codes = np.asarray(cell_type_codes, dtype=np.int64)
        if true_values.shape != predicted_values.shape:
            raise ValueError("Cell-type profile matrices do not align")
        if true_values.shape != (len(codes), self.n_features):
            raise ValueError("Cell-type codes do not align with profile rows")
        for code in np.unique(codes):
            if code < 0 or code >= len(self.counts):
                raise ValueError(f"Unknown cell-type code: {code}")
            mask = codes == code
            self.counts[code] += int(mask.sum())
            self.true_sum[code] += true_values[mask].sum(axis=0, dtype=np.float64)
            self.pred_sum[code] += predicted_values[mask].sum(axis=0, dtype=np.float64)

    def compute(
        self, min_cells: int = 30
    ) -> tuple[dict[str, object], list[dict[str, object]]]:
        code_to_name = {code: name for name, code in self.cell_type_mapping.items()}
        rows = []
        eligible_scores = []
        eligible_weights = []
        for code in np.flatnonzero(self.counts):
            n_cells = int(self.counts[code])
            true_mean = self.true_sum[code] / n_cells
            pred_mean = self.pred_sum[code] / n_cells
            if np.std(true_mean) == 0 or np.std(pred_mean) == 0:
                score = float("nan")
            else:
                score = float(np.corrcoef(true_mean, pred_mean)[0, 1])
            eligible = n_cells >= min_cells and np.isfinite(score)
            if eligible:
                eligible_scores.append(score)
                eligible_weights.append(n_cells)
            rows.append(
                {
                    "cell_type_code": int(code),
                    "cell_type": code_to_name.get(int(code), f"unknown_{code}"),
                    "n_cells": n_cells,
                    "pearson": None if not np.isfinite(score) else score,
                    "eligible_for_summary": bool(eligible),
                }
            )
        summary = {
            "macro_mean": float(np.mean(eligible_scores)) if eligible_scores else float("nan"),
            "weighted_mean": (
                float(np.average(eligible_scores, weights=eligible_weights))
                if eligible_scores
                else float("nan")
            ),
            "n_cell_types": len(rows),
            "n_eligible_cell_types": len(eligible_scores),
            "min_cells_per_type": min_cells,
            "unit": "Pearson correlation of paired cell-type mean profiles across features",
            "calculation": "exact streaming cell-type sums",
        }
        return summary, rows


__all__ = [
    "CANONICAL_METRICS",
    "FullTestDistributionAccumulator",
    "SUMMARY_SCHEMA",
    "StreamingAtacMetrics",
    "StreamingCellTypeProfiles",
    "StreamingFeaturePearson",
    "StreamingRnaMetrics",
    "markdown_table_without_w2_lisi",
    "summarize_distribution_without_w2_lisi",
    "write_full_feature_report",
]
