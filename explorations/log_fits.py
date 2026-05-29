"""Shared log-linear fits: num_layers = slope * log10(param_b) + intercept."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_EXPL = Path(__file__).resolve().parent
if str(_EXPL) not in sys.path:
    sys.path.insert(0, str(_EXPL))

from survey_load import load_survey

# Anchor: (100M parameters, 1 layer) → param_b = 0.1, log10(0.1) = -1
ANCHOR_PARAM_B = 0.1
ANCHOR_LAYERS = 1
ANCHOR_LOG10 = np.log10(ANCHOR_PARAM_B)

__all__ = [
    "ANCHOR_PARAM_B",
    "ANCHOR_LAYERS",
    "fit_log",
    "predict",
    "load_survey",
    "compute_below_above_fits",
]


def fit_log(
    param_b: np.ndarray,
    num_layers: np.ndarray,
    *,
    anchor_param_b: float = ANCHOR_PARAM_B,
    anchor_layers: float = ANCHOR_LAYERS,
) -> tuple[float, float]:
    """OLS fit constrained to pass through (anchor_param_b, anchor_layers)."""
    log_p = np.log10(np.asarray(param_b, dtype=float))
    y = np.asarray(num_layers, dtype=float)
    log_anchor = float(np.log10(anchor_param_b))
    # y = slope * (log_p - log_anchor) + anchor_layers
    u = log_p - log_anchor
    denom = float(np.dot(u, u))
    if denom == 0.0:
        return 0.0, float(anchor_layers)
    slope = float(np.dot(u, y - anchor_layers) / denom)
    intercept = anchor_layers - slope * log_anchor
    return slope, intercept


def predict(slope: float, intercept: float, param_b: np.ndarray | float) -> np.ndarray | float:
    out = slope * np.log10(np.asarray(param_b, dtype=float)) + intercept
    return float(out) if np.ndim(param_b) == 0 else out


def compute_below_above_fits(
    df,
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float], int, int]:
    x = df["param_b"].to_numpy()
    y = df["num_layers"].to_numpy()
    split = fit_log(x, y)
    m0, b0 = split
    y_hat = predict(m0, b0, x)
    below = y < y_hat
    above = y > y_hat
    return (
        split,
        fit_log(x[below], y[below]),
        fit_log(x[above], y[above]),
        int(below.sum()),
        int(above.sum()),
    )
