"""Metrics for log-linear line fits."""

from __future__ import annotations

import numpy as np

from log_fits import predict


def piecewise_mse(
    param_b: np.ndarray,
    num_layers: np.ndarray,
    lines: tuple[tuple[float, float], ...],
    assignments: np.ndarray,
) -> float:
    """Mean squared error using each point's assigned line."""
    y_pred = np.zeros_like(num_layers, dtype=float)
    for k, coef in enumerate(lines):
        mask = assignments == k
        if mask.any():
            y_pred[mask] = predict(*coef, param_b[mask])
    return float(np.mean((num_layers - y_pred) ** 2))


def assign_closest_line(
    param_b: np.ndarray,
    num_layers: np.ndarray,
    lines: tuple[tuple[float, float], ...],
) -> np.ndarray:
    errors = np.stack(
        [np.abs(num_layers - predict(m, b, param_b)) for m, b in lines],
        axis=1,
    )
    return np.argmin(errors, axis=1)
