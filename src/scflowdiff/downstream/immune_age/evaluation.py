"""Assemble direction-specific frozen-clock inputs."""
from __future__ import annotations

import pandas as pd

from .clocks import make_fusion_frame
from .datp import get_direction


def make_directional_fusion(
    source: pd.DataFrame, predicted_target: pd.DataFrame, direction: str
) -> pd.DataFrame:
    spec = get_direction(direction)
    if spec.name == "rna_to_atac":
        return make_fusion_frame(source, predicted_target)
    return make_fusion_frame(predicted_target, source)
