"""End-to-end orchestration after the sex-specific scFlowDiff backbones."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import h5py
import joblib
import numpy as np
import pandas as pd

from .clocks import predict_immune_age
from .datp import (
    fit_datp_bundle,
    fit_reverse_datp_bundle,
    load_datp,
    load_flow_hidden,
    load_flow_hidden_with_context,
    predict_targeted_modality,
)
from .pipeline import _resolve, load_immune_age_config


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_h5ad_column(path: Path, key: str) -> np.ndarray:
    with h5py.File(path, "r") as handle:
        node = handle[f"obs/{key}"]
        if isinstance(node, h5py.Group):
            categories = np.asarray(
                [x.decode() if isinstance(x, bytes) else str(x) for x in node["categories"][:]],
                dtype=object,
            )
            codes = np.asarray(node["codes"][:], dtype=np.int64)
            return np.asarray([categories[x] if x >= 0 else "" for x in codes])
        values = node[:]
    if values.dtype.kind in {"S", "O", "U"}:
        return np.asarray([x.decode() if isinstance(x, bytes) else str(x) for x in values])
    return np.asarray(values)


def external_donor_metadata(config: dict[str, Any], root: Path, cohort: str) -> pd.DataFrame:
    spec = config["external_cohorts"][cohort]
    path = _resolve(root, spec["path"]).resolve()
    donor = _read_h5ad_column(path, spec["donor_key"]).astype(str)
    sex = np.char.lower(_read_h5ad_column(path, spec["sex_key"]).astype(str))
    sex = np.where(np.isin(sex, ["f", "female"]), "female", "male")
    age = _read_h5ad_column(path, spec["age_key"]).astype(float)
    frame = pd.DataFrame({"sample_id": donor, "sex": sex, "chronological_age": age})
    grouped = frame.groupby("sample_id", sort=False).agg(
        sex=("sex", "first"), chronological_age=("chronological_age", "first")
    )
    inconsistent = frame.groupby("sample_id").agg({"sex": "nunique", "chronological_age": "nunique"})
    if (inconsistent > 1).any().any():
        raise ValueError(f"{cohort} donor metadata are inconsistent across cells")
    if len(grouped) != int(spec["expected_donors"]):
        raise ValueError(f"{cohort} expected {spec['expected_donors']} donors, found {len(grouped)}")
    return grouped


def paired_test_donor_metadata(config: dict[str, Any], root: Path) -> pd.DataFrame:
    """Return the frozen CIMA78 sealed-test donor contract without reading the H5MU."""
    frames = []
    for sex in ("female", "male"):
        split = pd.read_csv(
            _resolve(root, config["sexes"][sex]["split_csv"]),
            dtype={"sample_id": str},
        )
        required = {"sample_id", "age_numeric", "sex", "benchmark_split"}
        if missing := required.difference(split.columns):
            raise ValueError(f"{sex} split is missing {sorted(missing)}")
        sealed = split.loc[
            split["benchmark_split"].eq("sealed_test"),
            ["sample_id", "sex", "age_numeric"],
        ].copy()
        expected = int(config["sexes"][sex]["sealed_test"])
        if len(sealed) != expected or sealed["sample_id"].duplicated().any():
            raise ValueError(
                f"Expected {expected} unique sealed-test {sex} donors, found {len(sealed)}"
            )
        if not sealed["sex"].astype(str).str.lower().eq(sex).all():
            raise ValueError(f"{sex} sealed-test split contains inconsistent sex labels")
        sealed = sealed.rename(columns={"age_numeric": "chronological_age"})
        sealed["sex"] = sex
        frames.append(sealed)
    result = pd.concat(frames, ignore_index=True).set_index("sample_id")
    if len(result) != int(config["split"]["sealed_test"]) or not result.index.is_unique:
        raise ValueError("Combined CIMA78 sealed-test donor contract is invalid")
    return result


def output_root(config: dict[str, Any], root: Path) -> Path:
    return _resolve(root, config["output_root"]).resolve()


def preflight(config_path: str | Path, repository_root: str | Path) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    config = load_immune_age_config(config_path)
    required = [
        _resolve(root, config["dataset_path"]),
        _resolve(root, config["data_root"]) / "splits/split_manifest.json",
    ]
    for sex, sex_spec in config["sexes"].items():
        data = _resolve(root, config["data_root"]) / sex
        required.extend(
            [
                _resolve(root, sex_spec["split_csv"]),
                _resolve(root, sex_spec["feature_contract"]),
                data / "candidates/rna_candidates_development.parquet",
                data / "candidates/atac_candidates_development.parquet",
                data / "candidates/rna_candidates_locked_test.parquet",
                data / "candidates/atac_candidates_locked_test.parquet",
                data / "targets/target_rna_feature_order.csv",
                data / "targets/target_atac_feature_order.csv",
            ]
        )
        for clock in ("RNA", "ATAC", "Fusion"):
            folder = _resolve(root, config["clock_root"]) / sex / clock
            required.extend(
                [folder / "model.joblib", folder / "selected_features_500.csv",
                 folder / "selected_feature_medians.parquet"]
            )
    for cohort in config["external_cohorts"].values():
        required.append(_resolve(root, cohort["path"]))
        if cohort.get("transform_contract"):
            required.append(_resolve(root, cohort["transform_contract"]))
    missing = [str(path) for path in required if not path.is_file()]
    paired_columns: dict[str, list[str]] = {}
    paired_path = _resolve(root, config["dataset_path"])
    if paired_path.is_file():
        paired_contract = {
            "rna": ["sample_id", "cell_type", "sex_standardized"],
            "atac": ["sample_id", "cell_type", "sex"],
        }
        with h5py.File(paired_path, "r") as handle:
            for modality, expected in paired_contract.items():
                obs = handle[f"mod/{modality}/obs"]
                absent = [key for key in expected if key not in obs]
                if absent:
                    raise KeyError(
                        f"Paired {modality} obs is missing source fields: {absent}"
                    )
                paired_columns[modality] = expected
    metadata = {
        name: {"donors": len(external_donor_metadata(config, root, name))}
        for name in config["external_cohorts"]
    }
    result = {
        "status": "PASS" if not missing else "FAIL",
        "missing": missing,
        "required_files": len(required),
        "external_cohorts": metadata,
        "paired_source_columns": paired_columns,
        "formal_training_started": False,
    }
    if missing:
        raise FileNotFoundError(json.dumps(result, indent=2))
    target = output_root(config, root) / "contracts/preflight.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def train_datp(
    config: dict[str, Any], root: Path, *, sex: str, direction: str
) -> Path:
    data = _resolve(root, config["data_root"]) / sex
    out = output_root(config, root) / direction / sex / "datp"
    hidden_path = out.parent / "features/development_flow_hidden.npz"
    source_modality = config["directions"][direction]["source_modality"]
    target_modality = config["directions"][direction]["target_modality"]
    source = pd.read_parquet(data / f"candidates/{source_modality}_candidates_development.parquet")
    target = pd.read_parquet(data / f"candidates/{target_modality}_candidates_development.parquet")
    target_table = pd.read_csv(data / f"targets/target_{target_modality}_feature_order.csv")
    if "required_from_model" in target_table:
        keep = target_table["required_from_model"].astype(str).str.lower().eq("true")
        target_table = target_table.loc[keep]
    feature = json.loads((_resolve(root, config["sexes"][sex]["feature_contract"])).read_text())
    parameters = config["datp"]
    common = {
        "sex": sex,
        "seed": config["seed"],
        "folds": parameters["folds"],
        "pls_components": tuple(parameters["pls_components"]),
        "flow_pca_components": parameters["flow_pca_components"],
        "global_components": parameters["global_pca_components"],
        "local_components": parameters["local_pca_components"],
    }
    if direction == "atac_to_rna":
        hidden, source_context, lookup = load_flow_hidden_with_context(hidden_path)
        bundle = fit_reverse_datp_bundle(
            source,
            target,
            hidden,
            source_context,
            lookup,
            target_table,
            [str(x) for x in feature["cell_types"]],
            ridge_alpha=parameters["ridge_alpha"],
            **common,
        )
    else:
        hidden, lookup = load_flow_hidden(hidden_path)
        bundle = fit_datp_bundle(
            source,
            target,
            hidden,
            lookup,
            target_table,
            [str(x) for x in feature["cell_types"]],
            direction=direction,
            local_residual_ridge_alpha=parameters["ridge_alpha"],
            **common,
        )
    out.mkdir(parents=True, exist_ok=True)
    model_path = out / "datp.joblib"
    joblib.dump(bundle, model_path, compress=3)
    (out / "selection.json").write_text(
        json.dumps({"status": "PASS", "direction": direction, "sex": sex,
                    "sha256": _sha256(model_path), "selection": bundle["selection"]}, indent=2)
        + "\n", encoding="utf-8"
    )
    return model_path


def predict_external_target(
    config: dict[str, Any], root: Path, *, sex: str, direction: str
) -> Path:
    cohort = config["directions"][direction]["external_cohort"]
    modality = config["directions"][direction]["source_modality"]
    base = output_root(config, root) / direction / sex
    source_path = base / f"external/{cohort}/real_{modality}_candidates.parquet"
    hidden_path = base / f"external/{cohort}/flow_hidden.npz"
    output_model = base / "datp/datp.joblib"
    pretrained_model = (
        root
        / "pretrained/downstream/immune_age/seed42/datp"
        / direction
        / sex
        / "datp.joblib"
    )
    model_path = output_model if output_model.is_file() else pretrained_model
    source = pd.read_parquet(source_path)
    if direction == "atac_to_rna":
        hidden, source_context, lookup = load_flow_hidden_with_context(hidden_path)
    else:
        hidden, lookup = load_flow_hidden(hidden_path)
        source_context = None
    prediction = predict_targeted_modality(
        load_datp(model_path), source, hidden, lookup, direction=direction,
        source_context=source_context,
    )
    target = base / f"external/{cohort}/predicted_targeted_{config['directions'][direction]['target_modality']}.parquet"
    target.parent.mkdir(parents=True, exist_ok=True)
    prediction.to_parquet(target)
    lock = {"status": "PASS", "path": str(target), "sha256": _sha256(target),
            "direction": direction, "sex": sex, "age_reads": 0}
    target.with_suffix(".lock.json").write_text(json.dumps(lock, indent=2) + "\n")
    return target


def _paired_candidate_path(
    config: dict[str, Any], root: Path, *, sex: str, modality: str
) -> Path:
    return (
        _resolve(root, config["data_root"])
        / sex
        / f"candidates/{modality}_candidates_locked_test.parquet"
    )


def predict_paired_target(
    config: dict[str, Any], root: Path, *, sex: str, direction: str
) -> Path:
    """Predict the CIMA78 target from the sealed source only, before age access."""
    direction_spec = config["directions"][direction]
    source_modality = direction_spec["source_modality"]
    target_modality = direction_spec["target_modality"]
    base = output_root(config, root) / direction / sex
    paired = base / "paired/cima78"
    source_path = _paired_candidate_path(
        config, root, sex=sex, modality=source_modality
    )
    hidden_path = paired / "flow_hidden.npz"
    output_model = base / "datp/datp.joblib"
    pretrained_model = (
        root
        / "pretrained/downstream/immune_age/seed42/datp"
        / direction
        / sex
        / "datp.joblib"
    )
    model_path = output_model if output_model.is_file() else pretrained_model
    source = pd.read_parquet(source_path)
    expected = paired_test_donor_metadata(config, root)
    expected_donors = expected.index[expected["sex"].eq(sex)].astype(str).tolist()
    source.index = source.index.astype(str)
    if set(source.index) != set(expected_donors) or len(source) != len(expected_donors):
        raise ValueError(f"{direction}/{sex} source candidates do not match CIMA78")
    source = source.loc[expected_donors]
    if direction == "atac_to_rna":
        hidden, source_context, lookup = load_flow_hidden_with_context(hidden_path)
    else:
        hidden, lookup = load_flow_hidden(hidden_path)
        source_context = None
    prediction = predict_targeted_modality(
        load_datp(model_path),
        source,
        hidden,
        lookup,
        direction=direction,
        source_context=source_context,
    )
    target = paired / f"predicted_targeted_{target_modality}.parquet"
    target.parent.mkdir(parents=True, exist_ok=True)
    prediction.to_parquet(target)
    lock = {
        "status": "PASS",
        "cohort": "cima78",
        "path": str(target),
        "sha256": _sha256(target),
        "direction": direction,
        "sex": sex,
        "donors": len(prediction),
        "source_candidates_sha256": _sha256(source_path),
        "flow_hidden_sha256": _sha256(hidden_path),
        "datp_sha256": _sha256(model_path),
        "age_reads": 0,
        "target_modality_reads": 0,
    }
    target.with_suffix(".lock.json").write_text(
        json.dumps(lock, indent=2) + "\n", encoding="utf-8"
    )
    return target


def _metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float | int | None]:
    error = prediction - truth
    denominator = np.sum((truth - truth.mean()) ** 2)
    correlation = (
        float(np.corrcoef(truth, prediction)[0, 1])
        if len(truth) >= 3 and np.std(prediction) > 1e-12 else None
    )
    return {"n": len(truth), "MAE": float(np.mean(np.abs(error))),
            "median_AE": float(np.median(np.abs(error))),
            "RMSE": float(np.sqrt(np.mean(error**2))), "Pearson_R": correlation,
            "R2": float(1 - np.sum(error**2) / denominator) if denominator > 0 else None,
            "prediction_SD": float(np.std(prediction))}


def evaluate_external_age(
    config: dict[str, Any], root: Path, *, direction: str
) -> dict[str, Any]:
    cohort = config["directions"][direction]["external_cohort"]
    metadata = external_donor_metadata(config, root, cohort)
    frames, metric_rows = [], []
    for sex in ("female", "male"):
        donors = metadata.index[metadata["sex"].eq(sex)].astype(str)
        base = output_root(config, root) / direction / sex / f"external/{cohort}"
        target_modality = config["directions"][direction]["target_modality"]
        prediction_path = base / f"predicted_targeted_{target_modality}.parquet"
        lock_path = prediction_path.with_suffix(".lock.json")
        if not lock_path.is_file() or json.loads(lock_path.read_text()).get("sha256") != _sha256(prediction_path):
            raise RuntimeError("Age access denied until target predictions are locked")
        source_modality = config["directions"][direction]["source_modality"]
        source = pd.read_parquet(base / f"real_{source_modality}_candidates.parquet").loc[donors]
        predicted = pd.read_parquet(prediction_path).loc[donors]
        fusion = pd.concat(
            [source, predicted] if direction == "rna_to_atac" else [predicted, source], axis=1
        )
        if direction == "rna_to_atac" and "Missing__CD8_Naïve" in source:
            fusion["Missing__ATAC__CD8_Naïve"] = source["Missing__CD8_Naïve"]
        clock_root = _resolve(root, config["clock_root"]) / sex
        baseline_clock = config["directions"][direction]["baseline_clock"]
        clock_paths = {
            "baseline": clock_root / baseline_clock / "model.joblib",
            "fusion": clock_root / "Fusion/model.joblib",
        }
        hashes_before = {name: _sha256(path) for name, path in clock_paths.items()}
        baseline = predict_immune_age(
            source, model=clock_paths["baseline"],
            feature_table=clock_root / baseline_clock / "selected_features_500.csv",
            medians=clock_root / baseline_clock / "selected_feature_medians.parquet",
        )
        fused = predict_immune_age(
            fusion, model=clock_paths["fusion"],
            feature_table=clock_root / "Fusion/selected_features_500.csv",
            medians=clock_root / "Fusion/selected_feature_medians.parquet",
        )
        if hashes_before != {name: _sha256(path) for name, path in clock_paths.items()}:
            raise RuntimeError(f"Frozen clock changed during {direction}/{sex} prediction")
        truth = metadata.loc[donors, "chronological_age"].to_numpy(float)
        frame = pd.DataFrame({"sample_id": donors, "sex": sex,
            "chronological_age": truth, "source_only_prediction": baseline.to_numpy(),
            "fusion_prediction": fused.to_numpy()})
        frame["source_only_AE"] = np.abs(frame.chronological_age - frame.source_only_prediction)
        frame["fusion_AE"] = np.abs(frame.chronological_age - frame.fusion_prediction)
        frame["delta_AE_fusion_minus_source"] = frame.fusion_AE - frame.source_only_AE
        frames.append(frame)
        for method, values in (("source_only", baseline), ("fusion_predicted_target", fused)):
            metric_rows.append({"direction": direction, "cohort": cohort, "sex": sex,
                                "method": method, **_metrics(truth, values.to_numpy())})
    donor_frame = pd.concat(frames, ignore_index=True)
    for method, column in (("source_only", "source_only_prediction"),
                           ("fusion_predicted_target", "fusion_prediction")):
        metric_rows.append({"direction": direction, "cohort": cohort, "sex": "combined",
            "method": method, **_metrics(donor_frame.chronological_age.to_numpy(),
                                          donor_frame[column].to_numpy())})
    destination = output_root(config, root) / direction / "evaluation"
    destination.mkdir(parents=True, exist_ok=True)
    donor_frame.to_csv(destination / f"{cohort}_age_predictions.tsv", sep="\t", index=False)
    pd.DataFrame(metric_rows).to_csv(destination / f"{cohort}_age_metrics.tsv", sep="\t", index=False)
    result = {"status": "PASS", "direction": direction, "cohort": cohort,
              "donors": len(donor_frame), "fit_calls": 0, "feature_selection_calls": 0,
              "age_access_gate": "prediction_hash_locked",
              "clock_hashes_unchanged": True, "metrics": metric_rows}
    (destination / f"{cohort}_COMPLETE.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def _paired_effect(
    truth: np.ndarray,
    source_prediction: np.ndarray,
    comparison_prediction: np.ndarray,
    *,
    seed: int,
    draws: int = 10_000,
) -> dict[str, float | int]:
    source_error = np.abs(source_prediction - truth)
    comparison_error = np.abs(comparison_prediction - truth)
    delta = comparison_error - source_error
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, len(delta), size=(draws, len(delta)))
    bootstrap = delta[samples].mean(axis=1)
    return {
        "n": len(delta),
        "source_MAE": float(source_error.mean()),
        "comparison_MAE": float(comparison_error.mean()),
        "delta_MAE_comparison_minus_source": float(delta.mean()),
        "median_delta_AE": float(np.median(delta)),
        "improved_donors": int(np.sum(delta < 0)),
        "improved_fraction": float(np.mean(delta < 0)),
        "paired_bootstrap_CI95_low": float(np.quantile(bootstrap, 0.025)),
        "paired_bootstrap_CI95_high": float(np.quantile(bootstrap, 0.975)),
        "bootstrap_draws": int(draws),
    }


def evaluate_paired_age(
    config: dict[str, Any], root: Path, *, direction: str
) -> dict[str, Any]:
    """Evaluate locked CIMA78 predictions against source-only and real-Fusion clocks."""
    cohort = "cima78"
    metadata = paired_test_donor_metadata(config, root)
    frames: list[pd.DataFrame] = []
    metric_rows: list[dict[str, Any]] = []
    effect_rows: list[dict[str, Any]] = []
    clock_hashes: dict[str, dict[str, str]] = {}
    direction_spec = config["directions"][direction]
    source_modality = direction_spec["source_modality"]
    target_modality = direction_spec["target_modality"]
    for sex_index, sex in enumerate(("female", "male")):
        donors = metadata.index[metadata["sex"].eq(sex)].astype(str).tolist()
        base = output_root(config, root) / direction / sex / "paired/cima78"
        prediction_path = base / f"predicted_targeted_{target_modality}.parquet"
        lock_path = prediction_path.with_suffix(".lock.json")
        if not lock_path.is_file():
            raise RuntimeError("Age access denied until paired target predictions are locked")
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        if lock.get("sha256") != _sha256(prediction_path) or lock.get("age_reads") != 0:
            raise RuntimeError("CIMA78 target-prediction lock is invalid")
        source_path = _paired_candidate_path(
            config, root, sex=sex, modality=source_modality
        )
        real_target_path = _paired_candidate_path(
            config, root, sex=sex, modality=target_modality
        )
        source = pd.read_parquet(source_path)
        predicted = pd.read_parquet(prediction_path)
        real_target = pd.read_parquet(real_target_path)
        for name, frame in (
            ("source", source),
            ("predicted_target", predicted),
            ("real_target", real_target),
        ):
            frame.index = frame.index.astype(str)
            if set(frame.index) != set(donors) or len(frame) != len(donors):
                raise ValueError(f"{direction}/{sex} {name} donor set differs from CIMA78")
        source = source.loc[donors]
        predicted = predicted.loc[donors]
        real_target = real_target.loc[donors]
        if direction == "rna_to_atac":
            predicted_fusion = pd.concat([source, predicted], axis=1)
            real_fusion = pd.concat([source, real_target], axis=1)
            if "Missing__CD8_Naïve" in source:
                predicted_fusion["Missing__ATAC__CD8_Naïve"] = source[
                    "Missing__CD8_Naïve"
                ]
        else:
            predicted_fusion = pd.concat([predicted, source], axis=1)
            real_fusion = pd.concat([real_target, source], axis=1)
        clock_root = _resolve(root, config["clock_root"]) / sex
        baseline_clock = direction_spec["baseline_clock"]
        clock_paths = {
            "baseline_model": clock_root / baseline_clock / "model.joblib",
            "fusion_model": clock_root / "Fusion/model.joblib",
        }
        before = {name: _sha256(path) for name, path in clock_paths.items()}
        source_prediction = predict_immune_age(
            source,
            model=clock_paths["baseline_model"],
            feature_table=clock_root / baseline_clock / "selected_features_500.csv",
            medians=clock_root / baseline_clock / "selected_feature_medians.parquet",
        )
        predicted_fusion_prediction = predict_immune_age(
            predicted_fusion,
            model=clock_paths["fusion_model"],
            feature_table=clock_root / "Fusion/selected_features_500.csv",
            medians=clock_root / "Fusion/selected_feature_medians.parquet",
        )
        real_fusion_prediction = predict_immune_age(
            real_fusion,
            model=clock_paths["fusion_model"],
            feature_table=clock_root / "Fusion/selected_features_500.csv",
            medians=clock_root / "Fusion/selected_feature_medians.parquet",
        )
        after = {name: _sha256(path) for name, path in clock_paths.items()}
        if before != after:
            raise RuntimeError(f"Frozen clock changed during CIMA78 {direction}/{sex}")
        clock_hashes[sex] = before
        truth = metadata.loc[donors, "chronological_age"].to_numpy(float)
        frame = pd.DataFrame(
            {
                "sample_id": donors,
                "sex": sex,
                "chronological_age": truth,
                "source_only_prediction": source_prediction.to_numpy(),
                "fusion_predicted_target_prediction": predicted_fusion_prediction.to_numpy(),
                "fusion_real_target_prediction": real_fusion_prediction.to_numpy(),
            }
        )
        for method, column in (
            ("source_only", "source_only_prediction"),
            ("fusion_predicted_target", "fusion_predicted_target_prediction"),
            ("fusion_real_target", "fusion_real_target_prediction"),
        ):
            frame[f"{method}_AE"] = np.abs(frame[column] - frame["chronological_age"])
            metric_rows.append(
                {
                    "direction": direction,
                    "cohort": cohort,
                    "sex": sex,
                    "method": method,
                    **_metrics(truth, frame[column].to_numpy(float)),
                }
            )
        frame["delta_AE_predicted_minus_source"] = (
            frame["fusion_predicted_target_AE"] - frame["source_only_AE"]
        )
        frame["delta_AE_real_minus_source"] = (
            frame["fusion_real_target_AE"] - frame["source_only_AE"]
        )
        for comparison_index, (comparison, column) in enumerate(
            (
                ("fusion_predicted_target", "fusion_predicted_target_prediction"),
                ("fusion_real_target", "fusion_real_target_prediction"),
            )
        ):
            effect_rows.append(
                {
                    "direction": direction,
                    "cohort": cohort,
                    "sex": sex,
                    "comparison": comparison,
                    **_paired_effect(
                        truth,
                        frame["source_only_prediction"].to_numpy(float),
                        frame[column].to_numpy(float),
                        seed=int(config["seed"]) + 100 * sex_index + comparison_index,
                    ),
                }
            )
        frames.append(frame)
    donor_frame = pd.concat(frames, ignore_index=True)
    truth = donor_frame["chronological_age"].to_numpy(float)
    for method, column in (
        ("source_only", "source_only_prediction"),
        ("fusion_predicted_target", "fusion_predicted_target_prediction"),
        ("fusion_real_target", "fusion_real_target_prediction"),
    ):
        metric_rows.append(
            {
                "direction": direction,
                "cohort": cohort,
                "sex": "combined",
                "method": method,
                **_metrics(truth, donor_frame[column].to_numpy(float)),
            }
        )
    for comparison_index, (comparison, column) in enumerate(
        (
            ("fusion_predicted_target", "fusion_predicted_target_prediction"),
            ("fusion_real_target", "fusion_real_target_prediction"),
        )
    ):
        effect_rows.append(
            {
                "direction": direction,
                "cohort": cohort,
                "sex": "combined",
                "comparison": comparison,
                **_paired_effect(
                    truth,
                    donor_frame["source_only_prediction"].to_numpy(float),
                    donor_frame[column].to_numpy(float),
                    seed=int(config["seed"]) + 1_000 + comparison_index,
                ),
            }
        )
    destination = output_root(config, root) / direction / "evaluation"
    destination.mkdir(parents=True, exist_ok=True)
    predictions_path = destination / "cima78_age_predictions.tsv"
    metrics_path = destination / "cima78_age_metrics.tsv"
    effects_path = destination / "cima78_effect_summary.tsv"
    donor_frame.to_csv(predictions_path, sep="\t", index=False)
    pd.DataFrame(metric_rows).to_csv(metrics_path, sep="\t", index=False)
    pd.DataFrame(effect_rows).to_csv(effects_path, sep="\t", index=False)
    result = {
        "status": "PASS",
        "direction": direction,
        "cohort": cohort,
        "donors": len(donor_frame),
        "female_donors": int((donor_frame["sex"] == "female").sum()),
        "male_donors": int((donor_frame["sex"] == "male").sum()),
        "fit_calls": 0,
        "feature_selection_calls": 0,
        "age_access_gate": "paired_prediction_hash_locked",
        "clock_hashes_unchanged": True,
        "clock_hashes": clock_hashes,
        "metrics": metric_rows,
        "effects": effect_rows,
        "artifacts": {
            "predictions": str(predictions_path),
            "metrics": str(metrics_path),
            "effects": str(effects_path),
        },
    }
    (destination / "cima78_COMPLETE.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result
