"""Scatter: num_layers vs parameters. Log-linear fits; log and linear x-axis plots."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_EXPL = Path(__file__).resolve().parent
if str(_EXPL) not in sys.path:
    sys.path.insert(0, str(_EXPL))

from fit_metrics import piecewise_mse
from log_fits import compute_below_above_fits, predict
from survey_load import exclude_fit_outliers, load_survey

ROOT = Path(__file__).resolve().parents[1]
OUT_LOG = ROOT / "figures" / "layers_vs_params.png"
OUT_LINEAR = ROOT / "figures" / "layers_vs_params_linear.png"


def _split_assignments(y: np.ndarray, y_split: np.ndarray) -> np.ndarray:
    """0 = below split (below line), 1 = above split (above line)."""
    assignments = np.zeros(len(y), dtype=int)
    assignments[y > y_split] = 1
    on_line = y == y_split
    if on_line.any():
        assignments[on_line] = 0
    return assignments


def save_plot(df, *, log_x: bool, out: Path) -> float:
    x = df["param_b"].to_numpy()
    y = df["num_layers"].to_numpy()

    split_fit, below_fit, above_fit, n_below, n_above = compute_below_above_fits(df)
    m0, b0 = split_fit
    m_a, b_a = below_fit
    m_b, b_b = above_fit
    y_split = predict(m0, b0, x)
    below = y < y_split
    above = y > y_split

    assignments = _split_assignments(y, y_split)
    piecewise_lines = (below_fit, above_fit)
    mse = piecewise_mse(x, y, piecewise_lines, assignments)
    mse_split = float(np.mean((y - y_split) ** 2))

    if log_x:
        x_curve = np.logspace(np.log10(x.min()), np.log10(x.max()), 200)
        x_label = "Parameter count (billions, log scale)"
    else:
        x_curve = np.linspace(x.min(), x.max(), 200)
        x_label = "Parameter count (billions)"

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(x[below], y[below], c="#2563eb", label=f"Below split (n={n_below})", s=48, alpha=0.8)
    ax.scatter(x[above], y[above], c="#ea580c", label=f"Above split (n={n_above})", s=48, alpha=0.8)
    ax.plot(x_curve, predict(m0, b0, x_curve), "--", color="#444444", lw=1.5, label="Split fit")
    ax.plot(x_curve, predict(m_a, b_a, x_curve), color="#1d4ed8", lw=2, label="Below fit")
    ax.plot(x_curve, predict(m_b, b_b, x_curve), color="#c2410c", lw=2, label="Above fit")

    if log_x:
        ax.set_xscale("log")
    ax.set_xlabel(x_label)
    ax.set_ylabel("num_layers")
    ax.set_title("Split / below / above fit (through 100M, 1 layer)")
    ax.text(
        0.02,
        0.98,
        f"MSE (below/above piecewise): {mse:.2f}\nMSE (split line): {mse_split:.2f}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
    )
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
    ax.grid(True, alpha=0.3, which="both")

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return mse


def main() -> None:
    df = exclude_fit_outliers(load_survey())
    mse_log = save_plot(df, log_x=True, out=OUT_LOG)
    save_plot(df, log_x=False, out=OUT_LINEAR)
    print(f"Wrote {OUT_LOG}")
    print(f"Wrote {OUT_LINEAR}")
    print(f"  MSE (below/above piecewise): {mse_log:.4f}")


if __name__ == "__main__":
    main()
