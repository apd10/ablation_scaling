"""
Plot k=3 return_fits for each fit type (y vs x, y vs log x, y vs sqrt x).

Display is always num_layers (y) vs param_b (x) on a linear parameter axis.
The fitted curves use the corresponding transform internally.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
_EXPL = Path(__file__).resolve().parent
for p in (_ROOT, _EXPL):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from CONFIG import KMEANS_NUM_FITS
from survey_load import exclude_fit_outliers, load_survey
from utility import FitResult, predict, return_fits, XAxisType

FIG_DIR = _ROOT / "figures"
AXIS_TYPES: tuple[XAxisType, ...] = ("log", "linear", "sqrt")
COLORS = ("#444444", "#2563eb", "#ea580c", "#16a34a", "#9333ea")

FIT_LABEL = {
    "linear": "fit: y vs x",
    "log": "fit: y vs log₁₀(x)",
    "sqrt": "fit: y vs √x",
}


def save_plot(
    param_b: np.ndarray,
    num_layers: np.ndarray,
    result: FitResult,
    out: Path,
) -> None:
    x_curve = np.linspace(param_b.min(), param_b.max(), 400)

    fig, ax = plt.subplots(figsize=(9, 6))
    k = result.k

    for j in range(k):
        mask = result.assignments == j
        line = result.lines[j]
        color = COLORS[j % len(COLORS)]
        ax.scatter(
            param_b[mask],
            num_layers[mask],
            c=color,
            label=f"Cluster {j} (n={result.counts[j]})",
            s=48,
            alpha=0.8,
            edgecolors="white",
            linewidths=0.4,
        )
        ax.plot(
            x_curve,
            predict(line, x_curve, result.x_axis_type),
            color=color,
            lw=2,
            label=f"Line {j}",
        )

    ax.set_xlabel("Parameter count (billions)")
    ax.set_ylabel("num_layers")
    title = (
        f"k={k} k-means ({FIT_LABEL[result.x_axis_type]}), "
        f"plot: y vs x, anchor (100M, 1 layer)"
    )
    if result.converged:
        title += f" — {result.n_iters} iters"
    ax.set_title(title)
    ax.text(
        0.02,
        0.98,
        f"MSE: {result.mse:.2f}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
    )
    ax.legend(loc="lower right", fontsize=7, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> None:
    df = exclude_fit_outliers(load_survey())
    param_b = df["param_b"].to_numpy()
    num_layers = df["num_layers"].to_numpy()

    for axis_type in AXIS_TYPES:
        result = return_fits(df, kmeans_num_fit=KMEANS_NUM_FITS, x_axis_type=axis_type)
        out = FIG_DIR / f"layers_vs_params_k{KMEANS_NUM_FITS}_{axis_type}.png"
        save_plot(param_b, num_layers, result, out=out)
        print(f"{axis_type}: MSE={result.mse:.4f} counts={result.counts} -> {out}")


if __name__ == "__main__":
    main()
