#!/usr/bin/env bash
set -euo pipefail

REPOSITORY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${SCFLOW_PYTHON:-python}"
STAGE="${1:-all}"
LOG_DIR="$REPOSITORY_ROOT/outputs/translation/openproblem/seed42_cell_split_80_10_10/logs"
mkdir -p "$LOG_DIR"

export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export MMAE_RESUME="${MMAE_RESUME:-1}"
export MMFLOW_RESUME="${MMFLOW_RESUME:-1}"

cd "$REPOSITORY_ROOT"
"$PYTHON_BIN" -m scflowdiff.cli.run_translation \
  --repository-root "$REPOSITORY_ROOT" \
  --config configs/translation/openproblem_seed42.yaml \
  --stage "$STAGE" \
  2>&1 | tee -a "$LOG_DIR/${STAGE}.log"
