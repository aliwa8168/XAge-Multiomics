"""Internal neural-network and data-loading components for scFlowDiff.

Modules are intentionally not imported eagerly so Stage-2-only inference does
not pull Stage-1 data-loading dependencies into the process.
"""
