"""k-means fits: num_experts (y) vs intermediate_size (x), intermediate_size = num_experts * expert_size."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
_EXPL = Path(__file__).resolve().parent
for p in (_ROOT, _EXPL):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from CONFIG import (
    KMEANS_NUM_FITS,
    KMEANS_NUM_FIT_HIDDEN_SIZE_VS_LAYER_SIZE,
    NUM_EXPERTS_FIT_ANCHOR_INTERMEDIATE,
    NUM_EXPERTS_FIT_ANCHOR_NUM_EXPERTS,
)
from num_experts_fit_data import load_num_experts_fit_data
from utility import FitResult, predict, return_fits, XAxisType

FIG_DIR = _ROOT / "figures"
K_VALUES = (KMEANS_NUM_FIT_HIDDEN_SIZE_VS_LAYER_SIZE, KMEANS_NUM_FITS)
AXIS_TYPES: tuple[XAxisType, ...] = ("sqrt", "linear", "log")
COLORS = ("#444444", "#2563eb", "#ea580c", "#16a34a", "#9333ea")
FIT_LABEL = {
    "linear": "y vs x",
    "log": "y vs log10(x)",
    "sqrt": "y vs sqrt(x)",
}


def save_plot(
    x: np.ndarray,
    y: np.ndarray,
    result: FitResult,
    out: Path,
    *,
    x_out: np.ndarray | None = None,
    y_out: np.ndarray | None = None,
) -> None:
    x_curve = np.linspace(x.min(), x.max(), 400)
    fig, ax = plt.subplots(figsize=(9, 6))

    if x_out is not None and len(x_out) > 0:
        ax.scatter(
            x_out,
            y_out,
            c="#aaaaaa",
            s=40,
            alpha=0.7,
            marker="x",
            linewidths=1.2,
            label=f"outliers excluded (n={len(x_out)})",
            zorder=0,
        )

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

    ax.set_xlabel("intermediate_size (num_experts × expert_size)")
    ax.set_ylabel("num_experts")
    ax.set_title(
        f"k={result.k} num_experts vs intermediate_size ({FIT_LABEL[result.x_axis_type]})\n"
        "2 axis outliers excluded (max intermediate, max num_experts)"
        + (f" — {result.n_iters} iters" if result.converged else "")
    )
    mse_text = f"{result.mse:.4g}" if result.mse < 100 else f"{result.mse:.2f}"
    ax.text(
        0.02, 0.98, f"MSE: {mse_text}",
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
    df, outliers = load_num_experts_fit_data()
    x = df["intermediate_size"].to_numpy()
    y = df["num_experts"].to_numpy()
    x_out = outliers["intermediate_size"].to_numpy()
    y_out = outliers["num_experts"].to_numpy()
    print(
        f"Models for fit: {len(df)} (excluded {len(outliers)}: "
        "max intermediate_size + max num_experts)"
    )

    for k in K_VALUES:
        print(f"\nk={k}")
        for axis_type in AXIS_TYPES:
            result = return_fits(
                df,
                kmeans_num_fit=k,
                x_axis_type=axis_type,
                x_col="intermediate_size",
                y_col="num_experts",
                anchor_x=NUM_EXPERTS_FIT_ANCHOR_INTERMEDIATE,
                anchor_y=NUM_EXPERTS_FIT_ANCHOR_NUM_EXPERTS,
                exclude_outliers=False,
            )
            out = FIG_DIR / f"num_experts_vs_intermediate_size_k{k}_{axis_type}.png"
            save_plot(x, y, result, out, x_out=x_out, y_out=y_out)
            print(
                f"  {axis_type}: MSE={result.mse:.2f} counts={result.counts} "
                f"iters={result.n_iters} -> {out.name}"
            )
            for j, (m, b) in enumerate(result.lines):
                print(f"    line {j}: slope={m:.6f} intercept={b:.4f}")


if __name__ == "__main__":
    main()
