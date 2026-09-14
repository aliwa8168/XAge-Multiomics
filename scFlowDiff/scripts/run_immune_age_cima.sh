#!/usr/bin/env bash
set -euo pipefail

REPOSITORY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${SCFLOW_PYTHON:-python}"
STAGE="${1:-all}"
RUN_ROOT="$REPOSITORY_ROOT/outputs/downstream/immune_age/cima/seed42_donor_split_316_78"
LOG_DIR="$RUN_ROOT/logs"
mkdir -p "$LOG_DIR"

export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export MMAE_RESUME="${MMAE_RESUME:-1}"
export MMFLOW_RESUME="${MMFLOW_RESUME:-1}"

cd "$REPOSITORY_ROOT"
"$PYTHON_BIN" -m scflowdiff.cli.run_immune_age_pipeline \
  --repository-root "$REPOSITORY_ROOT" \
  --config configs/downstream/immune_age/cima_seed42.yaml \
  --stage "$STAGE" --sex both --direction both --parallel \
  2>&1 | tee -a "$LOG_DIR/${STAGE}.log"
