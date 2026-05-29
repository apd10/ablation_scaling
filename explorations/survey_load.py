"""Load and normalize the architecture survey CSV."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CSV_PATH = Path(__file__).resolve().parents[1] / "Model Architecture Experiments - Survey.csv"

_PARAM_RE = re.compile(r"^\s*([\d.]+)\s*([BMKT])?\s*$", re.IGNORECASE)

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from CONFIG import FIT_OUTLIER_COORDS  # noqa: E402


def parse_parameters(value: str | float | int) -> float:
    """Parse ``358B``, ``1.1T``, or numeric billions."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    text = str(value).strip().upper()
    m = _PARAM_RE.match(text)
    if not m:
        raise ValueError(f"Cannot parse parameters: {value!r}")
    num = float(m.group(1))
    suffix = (m.group(2) or "B").upper()
    if suffix in ("B", ""):
        return num
    if suffix == "M":
        return num / 1e3
    if suffix == "K":
        return num / 1e6
    if suffix == "T":
        return num * 1e3
    raise ValueError(f"Unknown suffix in parameters: {value!r}")


def ensure_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "param_b" not in df.columns and "parameters" in df.columns:
        df["param_b"] = df["parameters"].map(parse_parameters)
    if "layer_size_m" not in df.columns and "param_b" in df.columns and "num_layers" in df.columns:
        df["layer_size_m"] = df["param_b"] * 1e9 / df["num_layers"] / 1e6
    if (
        "total_expert_hidden" not in df.columns
        and "num_experts" in df.columns
        and "expert_size" in df.columns
    ):
        df["total_expert_hidden"] = df["num_experts"] * df["expert_size"]
    return df


def fit_outlier_mask(df: pd.DataFrame) -> np.ndarray:
    """True for rows in the three high (layer_size_m, total_expert_hidden) outlier clusters."""
    df = ensure_derived_columns(df)
    mask = np.zeros(len(df), dtype=bool)
    for layer_size_m, total_expert_hidden in FIT_OUTLIER_COORDS:
        mask |= np.isclose(df["layer_size_m"], layer_size_m, rtol=1e-4, atol=1e-3) & (
            df["total_expert_hidden"] == total_expert_hidden
        )
    return mask


def exclude_fit_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Drop the three high (layer_size_m, total_expert_hidden) outlier clusters."""
    mask = fit_outlier_mask(df)
    return ensure_derived_columns(df).loc[~mask].copy()


def split_fit_outliers(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (inliers, outliers) for the three configured outlier clusters."""
    df = ensure_derived_columns(df)
    mask = fit_outlier_mask(df)
    return df.loc[~mask].copy(), df.loc[mask].copy()


def load_survey(*, dedupe: bool = True, exclude_outliers: bool = False) -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH)
    if dedupe:
        df = df.drop_duplicates(subset=["url"])
    df = df.copy()
    df["param_b"] = df["parameters"].map(parse_parameters)
    df["layer_size_m"] = df["param_b"] * 1e9 / df["num_layers"] / 1e6
    df = df[(df["param_b"] > 0) & (df["num_layers"] > 0)].copy()
    if exclude_outliers:
        df = exclude_fit_outliers(df)
    return df
