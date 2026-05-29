"""
Theoretical hidden_size vs layer_size from k=3 intermediate_size sqrt fit through (0, 0).

  hidden_size = layer_size_m / 3 / intermediate_size_m(fit)

intermediate_size_m from k=3 sqrt fits (num_experts * expert_size / 1e6), anchored at origin.
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

from CONFIG import KMEANS_NUM_FITS
from survey_load import exclude_fit_outliers, load_survey
from utility import predict, return_fits

FIG_DIR = _ROOT / "figures"
OUT = FIG_DIR / "theoretical_hidden_vs_layer_size.png"
K = KMEANS_NUM_FITS
X_AXIS_TYPE = "sqrt"
ANCHOR_X = 0.0
ANCHOR_Y = 0.0
COLORS = ("#444444", "#2563eb", "#ea580c", "#16a34a", "#9333ea")


def load_data():
    df = exclude_fit_outliers(load_survey()).copy()
    df["intermediate_size_m"] = df["num_experts"] * df["expert_size"] / 1e6
    return df[
        (df["layer_size_m"] > 0)
        & (df["intermediate_size_m"] > 0)
        & (df["hidden_size"] > 0)
        & (df["num_experts"] > 0)
        & (df["expert_size"] > 0)
    ].copy()


def theoretical_hidden(layer_size_m: np.ndarray, intermediate_size_m: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return layer_size_m / (3.0 * intermediate_size_m)


def main() -> None:
    df = load_data()
    result = return_fits(
        df,
        kmeans_num_fit=K,
        x_axis_type=X_AXIS_TYPE,
        x_col="layer_size_m",
        y_col="intermediate_size_m",
        anchor_x=ANCHOR_X,
        anchor_y=ANCHOR_Y,
    )

    x_curve = np.linspace(0, df["layer_size_m"].max(), 400)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(
        df["layer_size_m"],
        df["hidden_size"],
        c="#cccccc",
        s=36,
        alpha=0.6,
        edgecolors="white",
        linewidths=0.3,
        label=f"survey (n={len(df)})",
        zorder=1,
    )

    for j, line in enumerate(result.lines):
        inter_m = predict(line, x_curve, X_AXIS_TYPE)
        hidden = theoretical_hidden(x_curve, inter_m)
        valid = (x_curve > 0) & (inter_m > 0) & np.isfinite(hidden)
        slope, intercept = line
        ax.plot(
            x_curve[valid],
            hidden[valid],
            color=COLORS[j % len(COLORS)],
            lw=2,
            label=f"line {j + 1}: {slope:.4g}·√x",
            zorder=2,
        )
        print(
            f"line {j + 1}: intermediate_size_m = {slope:.8f} * sqrt(layer_size_m) "
            f"(intercept={intercept:.6f})"
        )

    ax.set_xlabel("layer_size (millions of parameters per layer)")
    ax.set_ylabel("hidden_size")
    ax.set_title(
        f"Theoretical hidden_size vs layer_size (k={K} intermediate_size {X_AXIS_TYPE}, through (0, 0))\n"
        "hidden_size = layer_size_m / 3 / intermediate_size_m(fit)"
    )
    ax.legend(loc="best", fontsize=8, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT, dpi=150)
    plt.close(fig)
    print(f"Wrote {OUT}")
    print(f"MSE (intermediate fit): {result.mse:.4g} counts={result.counts}")


if __name__ == "__main__":
    main()
