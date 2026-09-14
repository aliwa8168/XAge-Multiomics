# scFlowDiff

scFlowDiff is a two-stage framework for bidirectional single-cell RNA–ATAC
translation and downstream multiomic immune-age prediction.

- **Stage 1 — multimodal autoencoder:** learns RNA and ATAC latent
  representations with modality-specific reconstruction.
- **Stage 2 — flow matching:** learns cell-type-conditioned transport between
  the RNA and ATAC latent spaces, supporting both RNA→ATAC and ATAC→RNA.
- **Immune-age application — DATP:** projects direction-specific Flow hidden
  features and observed-modality context onto the target features required by
  frozen immune-age clocks. Observed RNA plus predicted ATAC, or predicted RNA
  plus observed ATAC, are then used by the Fusion clock.



## Install

```bash
conda env create -f environment.yml
conda activate scflowdiff
pip install -e ".[train]"
```

A CUDA-capable GPU is required for Stage 1/2 training.

## Data

The four datasets are hosted on
[Hugging Face](https://huggingface.co/datasets/aliwa8168/XAge-Multiomics):

- [CIMA paired RNA/ATAC](https://huggingface.co/datasets/aliwa8168/XAge-Multiomics/resolve/main/cima_paired_max2000_per_donor_seed42.h5mu)
- [OpenProblem](https://huggingface.co/datasets/aliwa8168/XAge-Multiomics/resolve/main/openproblem_filtered.h5mu)
- [CIMA RNA-only, 27 donors](https://huggingface.co/datasets/aliwa8168/XAge-Multiomics/resolve/main/cima_rna_only_27_donors.h5ad)
- [CIMA ATAC-only, 7 donors](https://huggingface.co/datasets/aliwa8168/XAge-Multiomics/resolve/main/cima_atac_only_7_donors.h5ad)

Download them to the paths expected by the workflows (approximately 22 GB):

```bash
bash scripts/download_data.sh
```

Donor splits and downstream feature tables are provided under `data/`.

## Run

```bash
# Complete workflows, from Stage 1/2 training to evaluation
CUDA_VISIBLE_DEVICES=0 bash scripts/run_translation_cima.sh
CUDA_VISIBLE_DEVICES=0 bash scripts/run_translation_openproblem.sh
CUDA_VISIBLE_DEVICES=0 bash scripts/run_immune_age_cima.sh
```

Run each command separately, or choose different GPUs for concurrent jobs.
The immune-age workflow trains female/male scFlowDiff models and both DATP
directions, then evaluates the frozen clocks. Both directions use 50-step
Euler integration for Flow-hidden extraction.

To skip Stage 1/2 training and use the supplied immune-age backbone weights:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_immune_age_cima.sh downstream
```

This still extracts features and trains DATP before prediction and evaluation.
Pretrained scFlowDiff, DATP and immune-age clocks are stored under
`pretrained/downstream/immune_age/seed42/`. New checkpoints, features,
predictions, metrics and logs are written to `outputs/`.
