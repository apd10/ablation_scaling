"""Load survey data with derived numeric columns for plotting."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from CONFIG import FIT_OUTLIER_COORDS

CSV_PATH = _ROOT / "Model Architecture Experiments - Survey.csv"
_PARAM_RE = re.compile(r"^\s*([\d.]+)\s*([BMKT])?\s*$", re.IGNORECASE)

NUMERIC_COLUMNS: tuple[str, ...] = (
    "param_b",
    "hidden_size",
    "num_experts",
    "num_layers",
    "num_attn_heads",
    "num_kv_heads",
    "num_expert_per_tok",
    "expert_size",
    "layer_size_m",
    "intermediate_size",
    "intermediate_size_m",
    "total_expert_hidden",
    "moe_start_layer",
)


def parse_parameters(value: str | float | int) -> float:
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


def _attach_moe_start_layer(df: pd.DataFrame) -> pd.DataFrame:
    try:
        from explorations.config_load import attach_moe_start_layer
    except ImportError:
        df["moe_start_layer"] = np.nan
        return df
    return attach_moe_start_layer(df)


def load_plot_data() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH).drop_duplicates(subset=["url"]).copy()
    df["param_b"] = df["parameters"].map(parse_parameters)
    df["layer_size_m"] = df["param_b"] * 1e9 / df["num_layers"] / 1e6
    df["intermediate_size"] = df["num_experts"] * df["expert_size"]
    df["intermediate_size_m"] = df["intermediate_size"] / 1e6
    df["total_expert_hidden"] = df["intermediate_size"]
    df = _attach_moe_start_layer(df)
    return df


def get_numeric_series(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        raise KeyError(f"unknown column {column!r}")
    return pd.to_numeric(df[column], errors="coerce")


def survey_cluster_outlier_mask(df: pd.DataFrame) -> np.ndarray:
    """True for rows in the three configured Nemotron / DeepSeek-V4-Pro clusters."""
    mask = np.zeros(len(df), dtype=bool)
    if "layer_size_m" not in df.columns or "total_expert_hidden" not in df.columns:
        return mask
    for layer_size_m, total_expert_hidden in FIT_OUTLIER_COORDS:
        mask |= np.isclose(df["layer_size_m"], layer_size_m, rtol=1e-4, atol=1e-3) & (
            df["total_expert_hidden"] == total_expert_hidden
        )
    return mask


def compute_plot_outlier_indices(
    frame: pd.DataFrame,
    x_col: str,
    y_col: str,
    *,
    remove_survey_clusters: bool,
    n_remove_x: int = 0,
    n_remove_y: int = 0,
) -> tuple[int, ...]:
    from final_setup.outliers import axis_outlier_indices, merge_drop_indices

    positional = frame.reset_index(drop=True)
    drop: list[int] = []
    if remove_survey_clusters and "_survey_outlier" in positional.columns:
        drop.extend(int(i) for i in np.flatnonzero(positional["_survey_outlier"].to_numpy()))

    x = positional[x_col].to_numpy(dtype=float)
    y = positional[y_col].to_numpy(dtype=float)
    if n_remove_x > 0 or n_remove_y > 0:
        drop.extend(
            axis_outlier_indices(
                x,
                y,
                n_remove_x=max(0, n_remove_x),
                n_remove_y=max(0, n_remove_y),
            )
        )
    return merge_drop_indices(drop)


def plot_frame(df: pd.DataFrame, x_col: str, y_col: str) -> pd.DataFrame:
    out = df[["url", x_col, y_col]].copy()
    out["_survey_outlier"] = survey_cluster_outlier_mask(df).reindex(out.index, fill_value=False).to_numpy()
    out[x_col] = get_numeric_series(out, x_col)
    out[y_col] = get_numeric_series(out, y_col)
    out = out.replace([np.inf, -np.inf], np.nan).dropna()
    out = out[(out[x_col] > 0) & (out[y_col] > 0)]
    return out
