"""Shared load/filter for num_experts vs intermediate_size fits."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

_EXPL = Path(__file__).resolve().parent
_ROOT = _EXPL.parent
for p in (_ROOT, _EXPL):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from survey_load import load_survey


def dimension_outlier_indices(df: pd.DataFrame) -> list[object]:
    """Max intermediate_size, then max num_experts on the remainder (2nd axis outlier)."""
    idx_inter = df["intermediate_size"].idxmax()
    remaining = df.drop(idx_inter)
    idx_ne = remaining["num_experts"].idxmax()
    return [idx_inter, idx_ne]


def load_num_experts_fit_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Return (fit_df, outlier_df) using full survey data.

    Removes 2 models: highest intermediate_size and highest num_experts (2nd on
    remainder when the same model wins both axes).
    """
    full = load_survey()
    full = full.copy()
    full["intermediate_size"] = full["num_experts"] * full["expert_size"]
    ok = (full["num_experts"] > 0) & (full["intermediate_size"] > 0)
    full = full.loc[ok].copy()

    drop_idx = dimension_outlier_indices(full)
    outlier_df = full.loc[drop_idx].copy()
    fit_df = full.drop(drop_idx).copy()
    return fit_df, outlier_df
