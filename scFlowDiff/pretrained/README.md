# Pretrained models

`downstream/immune_age/seed42/` contains the models used by the CIMA immune-age
workflow:

- `scflowdiff/`: female- and male-specific Stage 1 multimodal AE and Stage 2
  flow-matching weights.
- `datp/`: female- and male-specific DATP models for RNA-to-ATAC and
  ATAC-to-RNA prediction.
- `clocks/`: frozen RNA, ATAC and Fusion immune-age models.

Run `bash scripts/run_immune_age_cima.sh` to reproduce the complete workflow.
When the selected Stage 1/2 or DATP weights are absent from a new output run,
the downstream inference code uses the copies stored here.
