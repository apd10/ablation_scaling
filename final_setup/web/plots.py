"""Build scatter (+ optional piecewise fit) plots for the web UI."""

from __future__ import annotations

import base64
import io
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from final_setup.outliers import apply_drop_indices
from final_setup.piecewise import PiecewiseFitResult, fit_piecewise, predict_line

COLORS = ("#444444", "#2563eb", "#ea580c", "#16a34a", "#9333ea")


def _sorted_lines(result: PiecewiseFitResult) -> tuple[tuple[float, float], ...]:
    return tuple(sorted(result.lines, key=lambda line: line[0]))


def render_plot_png(
    x_in: np.ndarray,
    y_in: np.ndarray,
    *,
    x_label: str,
    y_label: str,
    log_x: bool,
    fit_result: PiecewiseFitResult | None = None,
    x_out: np.ndarray | None = None,
    y_out: np.ndarray | None = None,
) -> bytes:
    fig, ax = plt.subplots(figsize=(9, 6))

    if x_out is not None and len(x_out) > 0:
        ax.scatter(
            x_out,
            y_out,
            c="#aaaaaa",
            s=40,
            alpha=0.75,
            marker="x",
            linewidths=1.2,
            label=f"outliers excluded (n={len(x_out)})",
            zorder=0,
        )

    if fit_result is not None:
        assignments = fit_result.assignments
        for j in range(fit_result.k):
            mask = assignments == j
            color = COLORS[j % len(COLORS)]
            ax.scatter(
                x_in[mask],
                y_in[mask],
                c=color,
                label=f"Cluster {j + 1} (n={fit_result.counts[j]})",
                s=48,
                alpha=0.85,
                edgecolors="white",
                linewidths=0.4,
            )
        x_curve = np.linspace(float(x_in.min()), float(x_in.max()), 400)
        for j, line in enumerate(_sorted_lines(fit_result)):
            color = COLORS[j % len(COLORS)]
            ax.plot(
                x_curve,
                predict_line(line, x_curve, fit_result.function_family),
                color=color,
                lw=2,
                label=f"Line {j + 1}",
            )
        mse_text = f"MSE: {fit_result.mse:.4g}"
        if fit_result.mse >= 100:
            mse_text = f"MSE: {fit_result.mse:.2f}"
        ax.text(
            0.02,
            0.98,
            f"{mse_text}\nk={fit_result.k}, {fit_result.function_family}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
        )
        ax.legend(loc="best", fontsize=8, framealpha=0.9)
    else:
        ax.scatter(
            x_in,
            y_in,
            alpha=0.75,
            s=48,
            c="#2563eb",
            edgecolors="white",
            linewidths=0.4,
        )

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(f"{y_label} vs {x_label}")
    if log_x:
        ax.set_xscale("log")
    ax.grid(True, alpha=0.3, which="both")

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    return buf.getvalue()


def png_to_data_uri(png_bytes: bytes) -> str:
    encoded = base64.b64encode(png_bytes).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def exponential_x_grid(x: np.ndarray, *, n_points: int = 12) -> np.ndarray:
    """Evenly spaced points on a log scale from min(x) to max(x)."""
    x_min = float(np.min(x))
    x_max = float(np.max(x))
    if x_min <= 0:
        raise ValueError("x must be positive for an exponential grid")
    if x_min == x_max:
        return np.array([x_min], dtype=float)
    n = max(2, int(n_points))
    return np.geomspace(x_min, x_max, n, dtype=float)


def build_fit_table(
    x_in: np.ndarray,
    fit_result: PiecewiseFitResult,
    *,
    x_label: str,
    y_label: str,
    n_points: int = 12,
) -> dict[str, Any]:
    """Table of x (geomspace over data range) and y from each fitted line."""
    x_grid = exponential_x_grid(x_in, n_points=n_points)
    lines = _sorted_lines(fit_result)
    y_columns = [f"{y_label}_line_{j}" for j in range(1, len(lines) + 1)]
    rows: list[dict[str, float]] = []
    for x_val in x_grid:
        row: dict[str, float] = {"x": float(x_val)}
        for j, line in enumerate(lines, start=1):
            row[y_columns[j - 1]] = float(
                predict_line(line, x_val, fit_result.function_family)
            )
        rows.append(row)
    return {
        "x_col": x_label,
        "y_col": y_label,
        "x_min": float(x_in.min()),
        "x_max": float(x_in.max()),
        "n_points": int(len(x_grid)),
        "y_columns": y_columns,
        "rows": rows,
    }


def fit_for_plot(
    x: np.ndarray,
    y: np.ndarray,
    *,
    kmeans_num_fits_space: list[int],
    function_family_space: list[str],
    anchor_x: float | None,
    anchor_y: float | None,
) -> PiecewiseFitResult:
    best: PiecewiseFitResult | None = None
    for k in kmeans_num_fits_space:
        for family in function_family_space:
            result = fit_piecewise(
                x,
                y,
                k=k,
                function_family=family,  # type: ignore[arg-type]
                anchor_x=anchor_x,
                anchor_y=anchor_y,
            )
            if best is None or result.mse < best.mse:
                best = result
    if best is None:
        raise ValueError("no fit configuration to evaluate")
    return best


def build_plot_response(
    x: np.ndarray,
    y: np.ndarray,
    *,
    x_label: str,
    y_label: str,
    log_x: bool,
    enable_fit: bool,
    kmeans_num_fits_space: list[int],
    function_family_space: list[str],
    anchor_x: float | None,
    anchor_y: float | None,
    outlier_drop_indices: Sequence[int] = (),
) -> dict[str, Any]:
    x_in, y_in, x_out, y_out, dropped = apply_drop_indices(x, y, outlier_drop_indices)

    fit_result = None
    meta: dict[str, Any] = {
        "n_points": int(len(x)),
        "n_fit_points": int(len(x_in)),
        "n_outliers_removed": int(len(dropped)),
        "outlier_indices": list(dropped),
    }

    if enable_fit and kmeans_num_fits_space and function_family_space:
        if len(x_in) < min(kmeans_num_fits_space):
            raise ValueError(
                f"not enough points after outlier removal ({len(x_in)} left; "
                f"need at least {min(kmeans_num_fits_space)} to fit)"
            )
        fit_result = fit_for_plot(
            x_in,
            y_in,
            kmeans_num_fits_space=kmeans_num_fits_space,
            function_family_space=function_family_space,
            anchor_x=anchor_x,
            anchor_y=anchor_y,
        )
        meta.update(
            {
                "k": fit_result.k,
                "function_family": fit_result.function_family,
                "mse": fit_result.mse,
                "counts": list(fit_result.counts),
                "lines": [
                    {"slope": line[0], "intercept": line[1]}
                    for line in _sorted_lines(fit_result)
                ],
            }
        )

    table = None
    if fit_result is not None:
        table = build_fit_table(
            x_in,
            fit_result,
            x_label=x_label,
            y_label=y_label,
        )

    png = render_plot_png(
        x_in,
        y_in,
        x_label=x_label,
        y_label=y_label,
        log_x=log_x,
        fit_result=fit_result,
        x_out=x_out if len(x_out) else None,
        y_out=y_out if len(y_out) else None,
    )
    return {"image": png_to_data_uri(png), "meta": meta, "table": table}
