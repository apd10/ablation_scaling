"""
Theoretical expert_size vs intermediate_size from k=2 num_experts linear fit.

  intermediate_size = num_experts * expert_size
  expert_size = intermediate_size / num_experts(fit)

num_experts from k=2 linear fit: num_experts = f(intermediate_size).
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

from CONFIG import (
    KMEANS_NUM_FIT_HIDDEN_SIZE_VS_LAYER_SIZE,
    NUM_EXPERTS_FIT_ANCHOR_INTERMEDIATE,
    NUM_EXPERTS_FIT_ANCHOR_NUM_EXPERTS,
)
from num_experts_fit_data import load_num_experts_fit_data
from utility import predict, return_fits

FIG_DIR = _ROOT / "figures"
OUT = FIG_DIR / "theoretical_expert_size_vs_intermediate_size.png"
K = KMEANS_NUM_FIT_HIDDEN_SIZE_VS_LAYER_SIZE
X_AXIS_TYPE = "linear"
COLORS = ("#444444", "#2563eb")


def load_data():
    df, _ = load_num_experts_fit_data()
    return df[(df["expert_size"] > 0)].copy()


def theoretical_expert_size(intermediate: np.ndarray, num_experts: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return intermediate / num_experts


def main() -> None:
    df = load_data()
    result = return_fits(
        df,
        kmeans_num_fit=K,
        x_axis_type=X_AXIS_TYPE,
        x_col="intermediate_size",
        y_col="num_experts",
        anchor_x=NUM_EXPERTS_FIT_ANCHOR_INTERMEDIATE,
        anchor_y=NUM_EXPERTS_FIT_ANCHOR_NUM_EXPERTS,
        exclude_outliers=False,
    )

    x_curve = np.linspace(df["intermediate_size"].min(), df["intermediate_size"].max(), 400)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(
        df["intermediate_size"],
        df["expert_size"],
        c="#cccccc",
        s=36,
        alpha=0.6,
        edgecolors="white",
        linewidths=0.3,
        label=f"survey (n={len(df)})",
        zorder=1,
    )

    for j, line in enumerate(result.lines):
        n_exp = predict(line, x_curve, X_AXIS_TYPE)
        expert = theoretical_expert_size(x_curve, n_exp)
        valid = (x_curve > 0) & (n_exp > 0) & np.isfinite(expert) & (expert > 0)
        slope, intercept = line
        ax.plot(
            x_curve[valid],
            expert[valid],
            color=COLORS[j % len(COLORS)],
            lw=2,
            label=f"line {j + 1}: {slope:.4g}·x + {intercept:.2g}",
            zorder=2,
        )
        print(
            f"line {j + 1}: num_experts = {slope:.8f} * intermediate_size + {intercept:.4f}"
        )

    ax.set_xlabel("intermediate_size (num_experts × expert_size)")
    ax.set_ylabel("expert_size")
    ax.set_title(
        f"Theoretical expert_size vs intermediate_size (k={K} num_experts {X_AXIS_TYPE} fit)\n"
        "expert_size = intermediate_size / num_experts(fit)"
    )
    ax.legend(loc="best", fontsize=8, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT, dpi=150)
    plt.close(fig)
    print(f"Wrote {OUT}")
    print(f"MSE (num_experts fit): {result.mse:.2f} counts={result.counts}")


if __name__ == "__main__":
    main()
