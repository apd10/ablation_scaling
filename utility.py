"""
Fitting utilities: k-means piecewise lines with linear / log / sqrt x-transform.

Default PHASE1 use: num_layers vs param_b through anchor (100M, 1 layer).
Supports arbitrary x/y columns; anchor optional (unconstrained OLS if omitted).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from CONFIG import KMEANS_NUM_FITS

_EXPL = Path(__file__).resolve().parent / "explorations"
if str(_EXPL) not in sys.path:
    sys.path.insert(0, str(_EXPL))
from survey_load import exclude_fit_outliers, parse_parameters  # noqa: E402

XAxisType = Literal["linear", "log", "sqrt"]

ANCHOR_PARAM_B = 0.1  # 100M
ANCHOR_LAYERS = 1.0
MAX_ITERS = 100


@dataclass
class FitResult:
    lines: tuple[tuple[float, float], ...]
    assignments: np.ndarray
    counts: tuple[int, ...]
    n_iters: int
    converged: bool
    mse: float
    k: int
    x_axis_type: XAxisType
    x_col: str
    y_col: str

    def predict(self, x: np.ndarray | float, line_idx: int) -> np.ndarray | float:
        return predict(self.lines[line_idx], x, self.x_axis_type)

    def predict_piecewise(self, x: np.ndarray) -> np.ndarray:
        y = np.zeros(len(x), dtype=float)
        for k, coef in enumerate(self.lines):
            mask = self.assignments == k
            if mask.any():
                y[mask] = predict(coef, x[mask], self.x_axis_type)
        return y


def _transform(x: np.ndarray | float, x_axis_type: XAxisType) -> np.ndarray | float:
    arr = np.asarray(x, dtype=float)
    if x_axis_type == "log":
        out = np.log10(arr)
    elif x_axis_type == "linear":
        out = arr
    elif x_axis_type == "sqrt":
        out = np.sqrt(arr)
    else:
        raise ValueError(f"x_axis_type must be linear, log, or sqrt; got {x_axis_type!r}")
    return float(out) if np.ndim(x) == 0 else out


def predict(
    line: tuple[float, float],
    x: np.ndarray | float,
    x_axis_type: XAxisType,
) -> np.ndarray | float:
    slope, intercept = line
    t = _transform(x, x_axis_type)
    out = slope * t + intercept
    return float(out) if np.ndim(x) == 0 else out


def fit_line(
    x: np.ndarray,
    y: np.ndarray,
    x_axis_type: XAxisType,
    *,
    anchor_x: float | None = None,
    anchor_y: float | None = None,
) -> tuple[float, float]:
    """Fit y = slope * transform(x) + intercept; anchor optional."""
    t = np.asarray(_transform(x, x_axis_type), dtype=float)
    y = np.asarray(y, dtype=float)
    if anchor_x is None or anchor_y is None:
        slope, intercept = np.polyfit(t, y, 1)
        return float(slope), float(intercept)
    if x_axis_type == "log" and anchor_x == 0.0 and anchor_y == 0.0:
        denom = float(np.dot(t, t))
        if denom == 0.0:
            return 0.0, 0.0
        slope = float(np.dot(t, y) / denom)
        return slope, 0.0
    t_anchor = float(_transform(anchor_x, x_axis_type))
    u = t - t_anchor
    denom = float(np.dot(u, u))
    if denom == 0.0:
        return 0.0, float(anchor_y)
    slope = float(np.dot(u, y - anchor_y) / denom)
    intercept = anchor_y - slope * t_anchor
    return slope, intercept


def line_from_slope(
    slope: float,
    x_axis_type: XAxisType,
    *,
    anchor_x: float,
    anchor_y: float,
) -> tuple[float, float]:
    if x_axis_type == "log" and anchor_x == 0.0 and anchor_y == 0.0:
        return slope, 0.0
    t_anchor = float(_transform(anchor_x, x_axis_type))
    return slope, anchor_y - slope * t_anchor


def _normalize_data(
    data: Any,
    *,
    x_col: str,
    y_col: str,
    exclude_outliers: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(data, pd.DataFrame):
        df = data
    elif isinstance(data, dict):
        df = pd.DataFrame(data)
    else:
        raise TypeError("data must be a DataFrame or dict")

    df = df.copy()
    if exclude_outliers:
        df = exclude_fit_outliers(df)
    if x_col not in df.columns and x_col == "param_b" and "parameters" in df.columns:
        df["param_b"] = df["parameters"].map(parse_parameters)

    if x_col not in df.columns or y_col not in df.columns:
        raise KeyError(f"data must contain {x_col!r} and {y_col!r}")

    x = df[x_col].astype(float).to_numpy()
    y = df[y_col].astype(float).to_numpy()
    ok = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
    return x[ok], y[ok]


def _assign_clusters(
    x: np.ndarray,
    y: np.ndarray,
    lines: tuple[tuple[float, float], ...],
    x_axis_type: XAxisType,
) -> np.ndarray:
    errors = np.stack(
        [np.abs(y - predict(line, x, x_axis_type)) for line in lines],
        axis=1,
    )
    return np.argmin(errors, axis=1)


def _piecewise_mse(
    x: np.ndarray,
    y: np.ndarray,
    lines: tuple[tuple[float, float], ...],
    assignments: np.ndarray,
    x_axis_type: XAxisType,
) -> float:
    y_pred = np.zeros_like(y, dtype=float)
    for k, line in enumerate(lines):
        mask = assignments == k
        if mask.any():
            y_pred[mask] = predict(line, x[mask], x_axis_type)
    return float(np.mean((y - y_pred) ** 2))


def _seed_below_above(
    x: np.ndarray,
    y: np.ndarray,
    x_axis_type: XAxisType,
    *,
    anchor_x: float | None,
    anchor_y: float | None,
) -> tuple[tuple[float, float], tuple[float, float]]:
    split = fit_line(x, y, x_axis_type, anchor_x=anchor_x, anchor_y=anchor_y)
    y_split = predict(split, x, x_axis_type)
    below = y < y_split
    above = y > y_split
    below_line = fit_line(x[below], y[below], x_axis_type, anchor_x=anchor_x, anchor_y=anchor_y)
    above_line = fit_line(x[above], y[above], x_axis_type, anchor_x=anchor_x, anchor_y=anchor_y)
    return below_line, above_line


def _seed_lines(
    x: np.ndarray,
    y: np.ndarray,
    k: int,
    x_axis_type: XAxisType,
    *,
    anchor_x: float | None,
    anchor_y: float | None,
) -> list[tuple[float, float]]:
    if k == 1:
        return [fit_line(x, y, x_axis_type, anchor_x=anchor_x, anchor_y=anchor_y)]

    below_line, above_line = _seed_below_above(
        x, y, x_axis_type, anchor_x=anchor_x, anchor_y=anchor_y
    )
    m_below, _ = below_line
    m_above, _ = above_line

    if k == 2:
        return [below_line, above_line]

    span = m_above - m_below
    pad = span / (k - 1)
    seed_slopes = np.linspace(m_below - pad, m_above + pad, k)
    if anchor_x is not None and anchor_y is not None:
        return [
            line_from_slope(float(s), x_axis_type, anchor_x=anchor_x, anchor_y=anchor_y)
            for s in seed_slopes
        ]
    # Unconstrained: rebuild lines from slopes at mean t
    t_mean = float(np.mean(_transform(x, x_axis_type)))
    y_mean = float(np.mean(y))
    return [(float(s), y_mean - float(s) * t_mean) for s in seed_slopes]


def _run_kmeans(
    x: np.ndarray,
    y: np.ndarray,
    k: int,
    x_axis_type: XAxisType,
    *,
    anchor_x: float | None,
    anchor_y: float | None,
    max_iters: int = MAX_ITERS,
) -> tuple[list[tuple[float, float]], np.ndarray, int, bool]:
    lines = _seed_lines(x, y, k, x_axis_type, anchor_x=anchor_x, anchor_y=anchor_y)
    assignments = _assign_clusters(x, y, tuple(lines), x_axis_type)

    converged = False
    n_iters = 0
    for _ in range(max_iters):
        n_iters += 1
        new_assignments = _assign_clusters(x, y, tuple(lines), x_axis_type)
        new_lines: list[tuple[float, float]] = []
        for j in range(k):
            mask = new_assignments == j
            if mask.any():
                new_lines.append(
                    fit_line(x[mask], y[mask], x_axis_type, anchor_x=anchor_x, anchor_y=anchor_y)
                )
            else:
                new_lines.append(lines[j])
        lines = new_lines
        if np.array_equal(new_assignments, assignments):
            assignments = new_assignments
            converged = True
            break
        assignments = new_assignments

    return lines, assignments, n_iters, converged


def return_fits(
    data: Any,
    *,
    kmeans_num_fit: int | None = None,
    x_axis_type: XAxisType = "log",
    x_col: str = "param_b",
    y_col: str = "num_layers",
    anchor_x: float | None = ANCHOR_PARAM_B,
    anchor_y: float | None = ANCHOR_LAYERS,
    exclude_outliers: bool = True,
) -> FitResult:
    """
    Piecewise line fit with k-means refinement.

    x_axis_type sets the internal transform: linear, log10, or sqrt.
    anchor_x / anchor_y constrain lines to pass through a point; pass None for
    unconstrained OLS on the transform.
    """
    k = KMEANS_NUM_FITS if kmeans_num_fit is None else kmeans_num_fit
    if k < 1:
        raise ValueError(f"kmeans_num_fit must be >= 1, got {k}")

    x, y = _normalize_data(
        data, x_col=x_col, y_col=y_col, exclude_outliers=exclude_outliers
    )

    if k == 1:
        line = fit_line(x, y, x_axis_type, anchor_x=anchor_x, anchor_y=anchor_y)
        lines = [line]
        assignments = np.zeros(len(x), dtype=int)
        n_iters = 0
        converged = True
    else:
        lines, assignments, n_iters, converged = _run_kmeans(
            x, y, k, x_axis_type, anchor_x=anchor_x, anchor_y=anchor_y
        )

    lines_t = tuple(lines)
    mse = _piecewise_mse(x, y, lines_t, assignments, x_axis_type)
    counts = tuple(int((assignments == j).sum()) for j in range(k))

    return FitResult(
        lines=lines_t,
        assignments=assignments,
        counts=counts,
        n_iters=n_iters,
        converged=converged,
        mse=mse,
        k=k,
        x_axis_type=x_axis_type,
        x_col=x_col,
        y_col=y_col,
    )
