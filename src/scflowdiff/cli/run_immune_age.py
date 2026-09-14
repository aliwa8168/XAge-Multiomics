from __future__ import annotations

import argparse
from pathlib import Path

from scflowdiff.downstream.immune_age.pipeline import run_scflowdiff


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the sex-specific 30k+30k scFlowDiff backbone for immune age"
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--sex", choices=("female", "male", "both"), default="both")
    parser.add_argument("--stage", choices=("stage1", "stage2", "both"), default="both")
    args = parser.parse_args()
    sexes = ("female", "male") if args.sex == "both" else (args.sex,)
    for sex in sexes:
        run_scflowdiff(
            args.config,
            repository_root=args.repository_root,
            sex=sex,
            stage=args.stage,
        )


if __name__ == "__main__":
    main()
