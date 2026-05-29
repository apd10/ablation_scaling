"""k-means fits: hidden_size (y) vs layer_size (x), layer_size = total params / num_layers."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
_EXPL = Path(__file__).resolve().parent
for p in (_ROOT, _EXPL):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from CONFIG import KMEANS_NUM_FIT_HIDDEN_SIZE_VS_LAYER_SIZE
from survey_load import exclude_fit_outliers, load_survey
from utility import FitResult, predict, return_fits, XAxisType

FIG_DIR = _ROOT / "figures"
AXIS_TYPES: tuple[XAxisType, ...] = ("sqrt", "linear", "log")
COLORS = ("#444444", "#2563eb", "#ea580c", "#16a34a", "#9333ea")
FIT_LABEL = {
    "linear": "y vs x",
    "log": "y vs log₁₀(x)",
    "sqrt": "y vs √x",
}


def load_data():
    df = exclude_fit_outliers(load_survey())
    return df[(df["hidden_size"] > 0) & (df["layer_size_m"] > 0)].copy()


def save_plot(x: np.ndarray, y: np.ndarray, result: FitResult, out: Path) -> None:
    x_curve = np.linspace(x.min(), x.max(), 400)
    fig, ax = plt.subplots(figsize=(9, 6))

    for j in range(result.k):
        mask = result.assignments == j
        color = COLORS[j % len(COLORS)]
        ax.scatter(
            x[mask], y[mask], c=color,
            label=f"Cluster {j} (n={result.counts[j]})",
            s=48, alpha=0.8, edgecolors="white", linewidths=0.4,
        )
        ax.plot(
            x_curve,
            predict(result.lines[j], x_curve, result.x_axis_type),
            color=color, lw=2, label=f"Line {j}",
        )

    ax.set_xlabel("layer_size (millions of parameters per layer)")
    ax.set_ylabel("hidden_size")
    ax.set_title(
        f"k={result.k} hidden_size vs layer_size ({FIT_LABEL[result.x_axis_type]})"
        + (f" — {result.n_iters} iters" if result.converged else "")
    )
    ax.text(
        0.02, 0.98, f"MSE: {result.mse:.2f}",
        transform=ax.transAxes, va="top", ha="left", fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
    )
    ax.legend(loc="best", fontsize=7, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> None:
    df = load_data()
    x = df["layer_size_m"].to_numpy()
    y = df["hidden_size"].to_numpy()
    print(f"Models for fit: {len(df)}")

    for axis_type in AXIS_TYPES:
        result = return_fits(
            df,
            kmeans_num_fit=KMEANS_NUM_FIT_HIDDEN_SIZE_VS_LAYER_SIZE,
            x_axis_type=axis_type,
            x_col="layer_size_m",
            y_col="hidden_size",
            anchor_x=None,
            anchor_y=None,
        )
        out = FIG_DIR / f"hidden_vs_layer_size_k{KMEANS_NUM_FIT_HIDDEN_SIZE_VS_LAYER_SIZE}_{axis_type}.png"
        save_plot(x, y, result, out)
        print(
            f"  {axis_type}: MSE={result.mse:.2f} counts={result.counts} "
            f"iters={result.n_iters} -> {out.name}"
        )
        for j, (m, b) in enumerate(result.lines):
            print(f"    line {j}: slope={m:.6f} intercept={b:.4f}")


if __name__ == "__main__":
    main()
