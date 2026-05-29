"""
Theoretical intermediate_size vs layer_size from Phase 2 hidden_size fits.

  intermediate_size_m = layer_size_m / 3 / hidden_size

hidden_size from k=2 sqrt fits (line 1 lower cluster, line 2 upper cluster).
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
from utility import predict

FIG_DIR = _ROOT / "figures"
OUT = FIG_DIR / "theoretical_intermediate_vs_layer_size.png"

# k=2 sqrt fits: hidden_size = slope * sqrt(layer_size_m) + intercept
HIDDEN_LINE_1 = (49.972214742677494, 737.3517642383326)
HIDDEN_LINE_2 = (45.34000089028387, 2291.3421595597288)


def hidden_size(line: tuple[float, float], layer_size_m: np.ndarray) -> np.ndarray:
    return predict(line, layer_size_m, "sqrt")


def theoretical_intermediate_m(layer_size_m: np.ndarray, hidden: np.ndarray) -> np.ndarray:
    """intermediate_size_m = layer_size_m / 3 / hidden_size (both in millions)."""
    return layer_size_m / (3.0 * hidden)


def main() -> None:
    df = exclude_fit_outliers(load_survey()).copy()
    df["intermediate_size_m"] = df["num_experts"] * df["expert_size"] / 1e6
    df = df[(df["layer_size_m"] > 0) & (df["intermediate_size_m"] > 0) & (df["hidden_size"] > 0)]

    x_min = max(25.0, df["layer_size_m"].min())
    x_max = df["layer_size_m"].max()
    x_curve = np.linspace(x_min, x_max, 400)

    h1 = hidden_size(HIDDEN_LINE_1, x_curve)
    h2 = hidden_size(HIDDEN_LINE_2, x_curve)
    y1 = theoretical_intermediate_m(x_curve, h1)
    y2 = theoretical_intermediate_m(x_curve, h2)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(
        df["layer_size_m"],
        df["intermediate_size_m"],
        c="#cccccc",
        s=36,
        alpha=0.6,
        edgecolors="white",
        linewidths=0.3,
        label=f"survey (n={len(df)})",
        zorder=1,
    )
    ax.plot(x_curve, y1, color="#444444", lw=2, label="line 1 (lower hidden_size)", zorder=2)
    ax.plot(x_curve, y2, color="#2563eb", lw=2, label="line 2 (upper hidden_size)", zorder=2)

    ax.set_xlabel("layer_size (millions of parameters per layer)")
    ax.set_ylabel("intermediate_size (num_experts × expert_size, millions)")
    ax.set_title(
        "Theoretical intermediate_size vs layer_size\n"
        "intermediate_size_m = layer_size_m / 3 / hidden_size(fit)"
    )
    ax.legend(loc="best", fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT, dpi=150)
    plt.close(fig)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
