from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from scflowdiff.downstream.immune_age.datp import (
    load_datp,
    load_flow_hidden,
    predict_targeted_modality,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply a frozen bidirectional DATP model")
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--direction", choices=("rna_to_atac", "atac_to_rna"))
    parser.add_argument("--source-features", required=True, type=Path)
    parser.add_argument("--flow-hidden", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    source = pd.read_parquet(args.source_features)
    hidden, lookup = load_flow_hidden(args.flow_hidden)
    prediction = predict_targeted_modality(
        load_datp(args.model), source, hidden, lookup, direction=args.direction
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.suffix == ".parquet":
        prediction.to_parquet(args.output)
    else:
        prediction.to_csv(args.output)


if __name__ == "__main__":
    main()
