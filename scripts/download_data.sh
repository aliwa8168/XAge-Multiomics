#!/usr/bin/env bash
set -euo pipefail

REPOSITORY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE_URL="https://huggingface.co/datasets/aliwa8168/XAge-Multiomics/resolve/main"

download() {
  local filename="$1"
  local folder="$2"
  local destination="$REPOSITORY_ROOT/data/$folder/$filename"
  mkdir -p "$(dirname "$destination")"
  if [[ -f "$destination" ]]; then
    echo "Already present: data/$folder/$filename"
    return
  fi
  echo "Downloading: $filename"
  curl --fail --location --retry 3 --continue-at - \
    --output "$destination.part" "$BASE_URL/$filename"
  mv "$destination.part" "$destination"
}

download cima_paired_max2000_per_donor_seed42.h5mu cima
download openproblem_filtered.h5mu openproblem
download cima_rna_only_27_donors.h5ad rna_only
download cima_atac_only_7_donors.h5ad atac_only
