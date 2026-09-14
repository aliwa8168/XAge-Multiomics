from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from scflowdiff.downstream.immune_age.external_features import (
    build_external_source_candidates,
    extract_flow_hidden,
)
from scflowdiff.downstream.immune_age.pipeline import load_immune_age_config, run_scflowdiff
from scflowdiff.downstream.immune_age.workflow import (
    evaluate_external_age,
    evaluate_paired_age,
    predict_external_target,
    predict_paired_target,
    preflight,
    train_datp,
)

EXTERNAL_STAGES = ("preflight", "backbone", "features", "datp", "predict", "evaluate")
PAIRED_STAGES = ("paired_features", "paired_predict", "paired_evaluate")
DOWNSTREAM_STAGES = EXTERNAL_STAGES[2:] + PAIRED_STAGES
FULL_PIPELINE_STAGES = EXTERNAL_STAGES + PAIRED_STAGES


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the complete seed42 immune-age workflow")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--stage", required=True,
        choices=(
            "preflight",
            "backbone",
            "features",
            "datp",
            "predict",
            "evaluate",
            "paired_features",
            "paired_predict",
            "paired_evaluate",
            "paired_all",
            "downstream",
            "all",
        ),
    )
    parser.add_argument("--sex", choices=("female", "male", "both"), default="both")
    parser.add_argument(
        "--direction", choices=("rna_to_atac", "atac_to_rna", "both"), default="both"
    )
    parser.add_argument(
        "--parallel", action="store_true",
        help="Parallelize sex-specific backbones and independent DATP/prediction/evaluation jobs",
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    root = args.repository_root.resolve()
    config = load_immune_age_config(args.config)
    sexes = ("female", "male") if args.sex == "both" else (args.sex,)
    directions = (
        ("rna_to_atac", "atac_to_rna") if args.direction == "both" else (args.direction,)
    )
    if args.stage == "all":
        stages = FULL_PIPELINE_STAGES
    elif args.stage == "downstream":
        stages = DOWNSTREAM_STAGES
    elif args.stage == "paired_all":
        stages = PAIRED_STAGES
    else:
        stages = (args.stage,)
    results: dict[str, object] = {}

    def parallel_backbones() -> dict[str, object]:
        output_root = Path(config["output_root"])
        output_root = output_root if output_root.is_absolute() else root / output_root
        log_dir = output_root / "logs" / "parallel_workers"
        log_dir.mkdir(parents=True, exist_ok=True)
        workers = []
        for sex in sexes:
            log_path = log_dir / f"backbone_{sex}.log"
            handle = log_path.open("a", encoding="utf-8")
            command = [
                sys.executable, "-m", "scflowdiff.cli.run_immune_age_pipeline",
                "--repository-root", str(root), "--config", str(args.config.resolve()),
                "--stage", "backbone", "--sex", sex, "--direction", "both", "--worker",
            ]
            process = subprocess.Popen(
                command, cwd=root, env=os.environ.copy(), stdout=handle,
                stderr=subprocess.STDOUT, start_new_session=True,
            )
            workers.append((sex, process, handle, log_path))
        failures = []
        worker_results = {}
        for sex, process, handle, log_path in workers:
            return_code = process.wait()
            handle.close()
            worker_results[sex] = {
                "pid": process.pid, "return_code": return_code, "log": str(log_path)
            }
            if return_code != 0:
                failures.append(f"{sex}={return_code}")
        if failures:
            raise RuntimeError(f"Parallel backbone worker failure: {', '.join(failures)}")
        return {"status": "PASS", "workers": worker_results}

    def parallel_map(function, jobs):
        with ThreadPoolExecutor(max_workers=len(jobs)) as executor:
            futures = {key: executor.submit(function, *values) for key, values in jobs.items()}
            return {key: future.result() for key, future in futures.items()}

    for stage in stages:
        if stage == "preflight":
            results[stage] = preflight(args.config, root)
        elif stage == "backbone":
            if args.parallel and len(sexes) > 1:
                results[stage] = parallel_backbones()
            else:
                for sex in sexes:
                    run_scflowdiff(args.config, repository_root=root, sex=sex, stage="both")
                results[stage] = {"status": "PASS", "sexes": list(sexes)}
        elif stage == "features":
            paths = []
            for direction in directions:
                cohort = config["directions"][direction]["external_cohort"]
                for sex in sexes:
                    paths.append(str(extract_flow_hidden(
                        config, root, sex=sex, direction=direction, cohort=None
                    )))
                    paths.append(str(build_external_source_candidates(
                        config, root, sex=sex, direction=direction
                    )))
                    paths.append(str(extract_flow_hidden(
                        config, root, sex=sex, direction=direction, cohort=cohort
                    )))
            results[stage] = {"status": "PASS", "paths": paths}
        elif stage == "datp":
            jobs = {
                f"{direction}/{sex}": (sex, direction)
                for direction in directions for sex in sexes
            }
            if args.parallel and len(jobs) > 1:
                flat = parallel_map(
                    lambda sex, direction: str(
                        train_datp(config, root, sex=sex, direction=direction)
                    ),
                    jobs,
                )
                results[stage] = {"status": "PASS", "jobs": flat}
            else:
                results[stage] = {direction: {
                    sex: str(train_datp(config, root, sex=sex, direction=direction))
                    for sex in sexes} for direction in directions}
        elif stage == "predict":
            jobs = {
                f"{direction}/{sex}": (sex, direction)
                for direction in directions for sex in sexes
            }
            if args.parallel and len(jobs) > 1:
                flat = parallel_map(
                    lambda sex, direction: str(
                        predict_external_target(config, root, sex=sex, direction=direction)
                    ),
                    jobs,
                )
                results[stage] = {"status": "PASS", "jobs": flat}
            else:
                results[stage] = {direction: {
                    sex: str(predict_external_target(config, root, sex=sex, direction=direction))
                    for sex in sexes} for direction in directions}
        elif stage == "evaluate":
            if set(sexes) != {"female", "male"}:
                raise ValueError("Evaluation requires --sex both for combined MAE")
            if args.parallel and len(directions) > 1:
                results[stage] = parallel_map(
                    lambda direction: evaluate_external_age(config, root, direction=direction),
                    {direction: (direction,) for direction in directions},
                )
            else:
                results[stage] = {
                    direction: evaluate_external_age(config, root, direction=direction)
                    for direction in directions
                }
        elif stage == "paired_features":
            paths = []
            for direction in directions:
                for sex in sexes:
                    paths.append(
                        str(
                            extract_flow_hidden(
                                config,
                                root,
                                sex=sex,
                                direction=direction,
                                cohort=None,
                                paired_split="sealed_test",
                            )
                        )
                    )
            results[stage] = {"status": "PASS", "paths": paths}
        elif stage == "paired_predict":
            jobs = {
                f"{direction}/{sex}": (sex, direction)
                for direction in directions
                for sex in sexes
            }
            if args.parallel and len(jobs) > 1:
                flat = parallel_map(
                    lambda sex, direction: str(
                        predict_paired_target(
                            config, root, sex=sex, direction=direction
                        )
                    ),
                    jobs,
                )
                results[stage] = {"status": "PASS", "jobs": flat}
            else:
                results[stage] = {
                    direction: {
                        sex: str(
                            predict_paired_target(
                                config, root, sex=sex, direction=direction
                            )
                        )
                        for sex in sexes
                    }
                    for direction in directions
                }
        elif stage == "paired_evaluate":
            if set(sexes) != {"female", "male"}:
                raise ValueError("Paired evaluation requires --sex both")
            if args.parallel and len(directions) > 1:
                results[stage] = parallel_map(
                    lambda direction: evaluate_paired_age(
                        config, root, direction=direction
                    ),
                    {direction: (direction,) for direction in directions},
                )
            else:
                results[stage] = {
                    direction: evaluate_paired_age(config, root, direction=direction)
                    for direction in directions
                }
        marker = Path(config["output_root"])
        marker = marker if marker.is_absolute() else root / marker
        marker_name = (
            f"{stage}_{args.sex}_{args.direction}_WORKER_COMPLETE.json"
            if args.worker
            else (
                f"{stage}_COMPLETE.json"
                if args.direction == "both"
                else f"{stage}_{args.direction}_COMPLETE.json"
            )
        )
        marker = marker / "pipeline" / marker_name
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(
            json.dumps(
                {"status": "PASS", "stage": stage, "sexes": list(sexes),
                 "directions": list(directions), "result": results[stage]},
                indent=2, ensure_ascii=False,
            ) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
