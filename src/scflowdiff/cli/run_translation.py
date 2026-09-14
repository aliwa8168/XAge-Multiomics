from __future__ import annotations

import argparse
from pathlib import Path

from scflowdiff.translation import preflight_translation, run_translation


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a 30k+30k translation workflow")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--stage",
        choices=("preflight", "stage1", "stage2", "evaluate", "both", "all"),
        default="all",
    )
    args = parser.parse_args()
    preflight_translation(args.config, repository_root=args.repository_root)
    if args.stage != "preflight":
        run_translation(args.config, repository_root=args.repository_root, stage=args.stage)


if __name__ == "__main__":
    main()
