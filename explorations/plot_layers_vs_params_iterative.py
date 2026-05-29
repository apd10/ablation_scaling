"""Scatter: iterative 3-line k-means refinement (separate figures from plot_layers_vs_params.py)."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_EXPL = Path(__file__).resolve().parent
if str(_EXPL) not in sys.path:
    sys.path.insert(0, str(_EXPL))

from fit_metrics import piecewise_mse
from iterative_log_fits import LINE_NAMES, fit_iterative_from_survey
from log_fits import predict
from survey_load import exclude_fit_outliers, load_survey

ROOT = Path(__file__).resolve().parents[1]
OUT_LOG = ROOT / "figures" / "layers_vs_params_iterative.png"
OUT_LINEAR = ROOT / "figures" / "layers_vs_params_linear_iterative.png"

COLORS = ("#444444", "#2563eb", "#ea580c")
LABELS = ("Split cluster", "Below cluster", "Above cluster")


def save_plot(df, result, *, log_x: bool, out: Path) -> float:
    x = df["param_b"].to_numpy()
    y = df["num_layers"].to_numpy()
    mse = piecewise_mse(x, y, result.lines, result.assignments)

    if log_x:
        x_curve = np.logspace(np.log10(x.min()), np.log10(x.max()), 200)
        x_label = "Parameter count (billions, log scale)"
    else:
        x_curve = np.linspace(x.min(), x.max(), 200)
        x_label = "Parameter count (billions)"

    fig, ax = plt.subplots(figsize=(9, 6))

    for k in range(3):
        mask = result.assignments == k
        m, b = result.lines[k]
        ax.scatter(
            x[mask],
            y[mask],
            c=COLORS[k],
            label=f"{LABELS[k]} (n={result.counts[k]})",
            s=48,
            alpha=0.8,
            edgecolors="white",
            linewidths=0.4,
        )
        ax.plot(
            x_curve,
            predict(m, b, x_curve),
            color=COLORS[k],
            lw=2,
            label=f"{LINE_NAMES[k]}: {m:.2f}·log10(P)+{b:.1f}",
        )

    if log_x:
        ax.set_xscale("log")
    ax.set_xlabel(x_label)
    ax.set_ylabel("num_layers")
    title = "Iterative 3-line fit (k-means on split/below/above), through (100M, 1)"
    if result.converged:
        title += f" — {result.n_iters} iters"
    ax.set_title(title)
    ax.text(
        0.02,
        0.98,
        f"MSE (3-line piecewise): {mse:.2f}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
    )
    ax.legend(loc="lower right", fontsize=7, framealpha=0.9)
    ax.grid(True, alpha=0.3, which="both")

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return mse


def main() -> None:
    df = exclude_fit_outliers(load_survey())
    result = fit_iterative_from_survey(df)
    mse_log = save_plot(df, result, log_x=True, out=OUT_LOG)
    save_plot(df, result, log_x=False, out=OUT_LINEAR)

    print(f"Wrote {OUT_LOG}")
    print(f"Wrote {OUT_LINEAR}")
    print(f"  MSE (3-line piecewise): {mse_log:.4f}")
    print(f"  converged={result.converged} in {result.n_iters} iterations")
    print(f"  counts: split={result.counts[0]}, below={result.counts[1]}, above={result.counts[2]}")
    for k, name in enumerate(LINE_NAMES):
        m, b = result.lines[k]
        print(f"  {name}: num_layers = {m:.4f}·log10(param_b) + {b:.4f}")


if __name__ == "__main__":
    main()
