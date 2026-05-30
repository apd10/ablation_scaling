"""K-means piecewise line fits with linear / log10 / sqrt x-transform."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

FunctionFamily = Literal["linear", "log", "sqrt"]
MAX_ITERS = 100


@dataclass
class PiecewiseFitResult:
    lines: tuple[tuple[float, float], ...]
    assignments: np.ndarray
    counts: tuple[int, ...]
    n_iters: int
    converged: bool
    mse: float
    k: int
    function_family: FunctionFamily


def _as_arrays(x, y) -> tuple[np.ndarray, np.ndarray]:
    x_arr = np.asarray(x, dtype=float).ravel()
    y_arr = np.asarray(y, dtype=float).ravel()
    if x_arr.shape != y_arr.shape:
        raise ValueError(f"x and y must have the same length; got {len(x_arr)} and {len(y_arr)}")
    ok = np.isfinite(x_arr) & np.isfinite(y_arr) & (x_arr > 0) & (y_arr > 0)
    return x_arr[ok], y_arr[ok]


def transform(x: np.ndarray | float, function_family: FunctionFamily) -> np.ndarray | float:
    arr = np.asarray(x, dtype=float)
    if function_family == "log":
        out = np.log10(arr)
    elif function_family == "linear":
        out = arr
    elif function_family == "sqrt":
        out = np.sqrt(arr)
    else:
        raise ValueError(f"function_family must be linear, log, or sqrt; got {function_family!r}")
    return float(out) if np.ndim(x) == 0 else out


def predict_line(
    line: tuple[float, float],
    x: np.ndarray | float,
    function_family: FunctionFamily,
) -> np.ndarray | float:
    slope, intercept = line
    t = transform(x, function_family)
    out = slope * t + intercept
    return float(out) if np.ndim(x) == 0 else out


def fit_line(
    x: np.ndarray,
    y: np.ndarray,
    function_family: FunctionFamily,
    *,
    anchor_x: float | None = None,
    anchor_y: float | None = None,
) -> tuple[float, float]:
    t = np.asarray(transform(x, function_family), dtype=float)
    y = np.asarray(y, dtype=float)
    if anchor_x is None or anchor_y is None:
        slope, intercept = np.polyfit(t, y, 1)
        return float(slope), float(intercept)
    if function_family == "log" and anchor_x == 0.0 and anchor_y == 0.0:
        denom = float(np.dot(t, t))
        if denom == 0.0:
            return 0.0, 0.0
        slope = float(np.dot(t, y) / denom)
        return slope, 0.0
    t_anchor = float(transform(anchor_x, function_family))
    u = t - t_anchor
    denom = float(np.dot(u, u))
    if denom == 0.0:
        return 0.0, float(anchor_y)
    slope = float(np.dot(u, y - anchor_y) / denom)
    intercept = anchor_y - slope * t_anchor
    return slope, intercept


def line_from_slope(
    slope: float,
    function_family: FunctionFamily,
    *,
    anchor_x: float,
    anchor_y: float,
) -> tuple[float, float]:
    if function_family == "log" and anchor_x == 0.0 and anchor_y == 0.0:
        return slope, 0.0
    t_anchor = float(transform(anchor_x, function_family))
    return slope, anchor_y - slope * t_anchor


def _assign_clusters(
    x: np.ndarray,
    y: np.ndarray,
    lines: tuple[tuple[float, float], ...],
    function_family: FunctionFamily,
) -> np.ndarray:
    errors = np.stack(
        [np.abs(y - predict_line(line, x, function_family)) for line in lines],
        axis=1,
    )
    return np.argmin(errors, axis=1)


def piecewise_mse(
    x: np.ndarray,
    y: np.ndarray,
    lines: tuple[tuple[float, float], ...],
    assignments: np.ndarray,
    function_family: FunctionFamily,
) -> float:
    y_pred = np.zeros_like(y, dtype=float)
    for j, line in enumerate(lines):
        mask = assignments == j
        if mask.any():
            y_pred[mask] = predict_line(line, x[mask], function_family)
    return float(np.mean((y - y_pred) ** 2))


def _seed_below_above(
    x: np.ndarray,
    y: np.ndarray,
    function_family: FunctionFamily,
    *,
    anchor_x: float | None,
    anchor_y: float | None,
) -> tuple[tuple[float, float], tuple[float, float]]:
    split = fit_line(x, y, function_family, anchor_x=anchor_x, anchor_y=anchor_y)
    y_split = predict_line(split, x, function_family)
    below = y < y_split
    above = y > y_split
    below_line = fit_line(x[below], y[below], function_family, anchor_x=anchor_x, anchor_y=anchor_y)
    above_line = fit_line(x[above], y[above], function_family, anchor_x=anchor_x, anchor_y=anchor_y)
    return below_line, above_line


def _seed_lines(
    x: np.ndarray,
    y: np.ndarray,
    k: int,
    function_family: FunctionFamily,
    *,
    anchor_x: float | None,
    anchor_y: float | None,
) -> list[tuple[float, float]]:
    if k == 1:
        return [fit_line(x, y, function_family, anchor_x=anchor_x, anchor_y=anchor_y)]

    below_line, above_line = _seed_below_above(
        x, y, function_family, anchor_x=anchor_x, anchor_y=anchor_y
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
            line_from_slope(float(s), function_family, anchor_x=anchor_x, anchor_y=anchor_y)
            for s in seed_slopes
        ]
    t_mean = float(np.mean(transform(x, function_family)))
    y_mean = float(np.mean(y))
    return [(float(s), y_mean - float(s) * t_mean) for s in seed_slopes]


def _run_kmeans(
    x: np.ndarray,
    y: np.ndarray,
    k: int,
    function_family: FunctionFamily,
    *,
    anchor_x: float | None,
    anchor_y: float | None,
    max_iters: int = MAX_ITERS,
) -> tuple[list[tuple[float, float]], np.ndarray, int, bool]:
    lines = _seed_lines(x, y, k, function_family, anchor_x=anchor_x, anchor_y=anchor_y)
    assignments = _assign_clusters(x, y, tuple(lines), function_family)

    converged = False
    n_iters = 0
    for _ in range(max_iters):
        n_iters += 1
        new_assignments = _assign_clusters(x, y, tuple(lines), function_family)
        new_lines: list[tuple[float, float]] = []
        for j in range(k):
            mask = new_assignments == j
            if mask.any():
                new_lines.append(
                    fit_line(x[mask], y[mask], function_family, anchor_x=anchor_x, anchor_y=anchor_y)
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


def fit_piecewise(
    x,
    y,
    *,
    k: int,
    function_family: FunctionFamily = "log",
    anchor_x: float | None = None,
    anchor_y: float | None = None,
) -> PiecewiseFitResult:
    """Fit k piecewise lines: y = slope * transform(x) + intercept."""
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")

    x_arr, y_arr = _as_arrays(x, y)
    if len(x_arr) < k:
        raise ValueError(f"need at least {k} points to fit {k} lines, got {len(x_arr)}")

    if k == 1:
        line = fit_line(x_arr, y_arr, function_family, anchor_x=anchor_x, anchor_y=anchor_y)
        lines = [line]
        assignments = np.zeros(len(x_arr), dtype=int)
        n_iters = 0
        converged = True
    else:
        lines, assignments, n_iters, converged = _run_kmeans(
            x_arr, y_arr, k, function_family, anchor_x=anchor_x, anchor_y=anchor_y
        )

    lines_t = tuple(lines)
    mse = piecewise_mse(x_arr, y_arr, lines_t, assignments, function_family)
    counts = tuple(int((assignments == j).sum()) for j in range(k))

    return PiecewiseFitResult(
        lines=lines_t,
        assignments=assignments,
        counts=counts,
        n_iters=n_iters,
        converged=converged,
        mse=mse,
        k=k,
        function_family=function_family,
    )
