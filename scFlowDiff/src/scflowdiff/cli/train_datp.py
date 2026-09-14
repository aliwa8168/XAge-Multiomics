from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import pandas as pd

from scflowdiff.downstream.immune_age.datp import fit_datp_bundle, load_flow_hidden


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one sex/direction DATP bundle")
    parser.add_argument("--direction", required=True, choices=("rna_to_atac", "atac_to_rna"))
    parser.add_argument("--sex", required=True, choices=("female", "male"))
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--target", required=True, type=Path)
    parser.add_argument("--flow-hidden", required=True, type=Path)
    parser.add_argument("--target-contract", required=True, type=Path)
    parser.add_argument("--feature-contract", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    source = pd.read_parquet(args.source)
    target = pd.read_parquet(args.target)
    target_table = pd.read_csv(args.target_contract)
    if "required_from_model" in target_table:
        keep = target_table["required_from_model"].astype(str).str.lower().eq("true")
        target_table = target_table.loc[keep].copy()
    feature = json.loads(args.feature_contract.read_text(encoding="utf-8"))
    hidden, lookup = load_flow_hidden(args.flow_hidden)
    bundle = fit_datp_bundle(
        source,
        target,
        hidden,
        lookup,
        target_table,
        [str(value) for value in feature["cell_types"]],
        sex=args.sex,
        direction=args.direction,
        seed=42,
        folds=5,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, args.output, compress=3)
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    report = {
        "status": "PASS",
        "direction": args.direction,
        "sex": args.sex,
        "model": str(args.output),
        "sha256": digest,
        "target_features": len(bundle["target_order"]),
        "selection": bundle["selection"],
    }
    args.output.with_suffix(".selection.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
