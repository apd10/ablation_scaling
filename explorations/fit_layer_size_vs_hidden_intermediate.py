"""
Fit and plot layer_size (y) vs hidden_size * intermediate_size (x).

  layer_size_m = total_parameters / num_layers  (millions per layer)
  intermediate_size = num_experts * expert_size  (expert_size = per-expert intermediate dim)
  x_m = hidden_size * intermediate_size / 1e6  (millions)
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
_EXPL = Path(__file__).resolve().parent
for p in (_ROOT, _EXPL):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from survey_load import exclude_fit_outliers, load_survey
from utility import FitResult, predict, return_fits, XAxisType

FIG_DIR = _ROOT / "figures"
K = 1
AXIS_TYPES: tuple[XAxisType, ...] = ("linear", "log", "sqrt")
FIT_LABEL = {
    "linear": "y vs x",
    "log": "y vs log10(x)",
    "sqrt": "y vs sqrt(x)",
}


def load_data():
    df = exclude_fit_outliers(load_survey())
    df = df.copy()
    df["intermediate_size"] = df["num_experts"] * df["expert_size"]
    df["hidden_x_intermediate_m"] = df["hidden_size"] * df["intermediate_size"] / 1e6
    return df[
        (df["layer_size_m"] > 0)
        & (df["hidden_size"] > 0)
        & (df["num_experts"] > 0)
        & (df["expert_size"] > 0)
    ].copy()


def save_plot(x: np.ndarray, y: np.ndarray, result: FitResult, out: Path) -> None:
    x_curve = np.linspace(x.min(), x.max(), 400)
    fig, ax = plt.subplots(figsize=(9, 6))

    ax.scatter(x, y, alpha=0.75, s=48, edgecolors="white", linewidths=0.5, c="#2563eb")
    slope, intercept = result.lines[0]
    ax.plot(
        x_curve,
        predict((slope, intercept), x_curve, result.x_axis_type),
        color="#ea580c",
        lw=2,
        label=f"fit: {slope:.4g}·t + {intercept:.2g}",
    )

    ax.set_xlabel("hidden_size × (num_experts × expert_size) (millions)")
    ax.set_ylabel("layer_size (millions of parameters per layer)")
    ax.set_title(f"k={K} layer_size vs hidden×intermediate ({FIT_LABEL[result.x_axis_type]})")
    ax.text(
        0.02, 0.98, f"MSE: {result.mse:.2f}",
        transform=ax.transAxes, va="top", ha="left", fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
    )
    ax.legend(loc="best", fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> None:
    df = load_data()
    x = df["hidden_x_intermediate_m"].to_numpy()
    y = df["layer_size_m"].to_numpy()
    print(f"Models: {len(df)}")
    print("y = layer_size_m (total params / num_layers, millions)")
    print("x = hidden_size * (num_experts * expert_size) / 1e6 (millions)")
    print()

    for axis_type in AXIS_TYPES:
        result = return_fits(
            df,
            kmeans_num_fit=K,
            x_axis_type=axis_type,
            x_col="hidden_x_intermediate_m",
            y_col="layer_size_m",
            anchor_x=None,
            anchor_y=None,
        )
        out = FIG_DIR / f"layer_size_vs_hidden_intermediate_k{K}_{axis_type}.png"
        save_plot(x, y, result, out)
        slope, intercept = result.lines[0]
        print(f"{axis_type} ({FIT_LABEL[axis_type]})")
        print(f"  layer_size_m = {slope:.8f} * t + {intercept:.4f}")
        print(f"  MSE: {result.mse:.2f} -> {out.name}")
        print()


if __name__ == "__main__":
    main()
