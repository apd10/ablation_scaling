"""Scatter: moe_start_layer (y) vs parameters (x) from survey + HF configs."""

from pathlib import Path

import matplotlib.pyplot as plt
import sys

_EXPL = Path(__file__).resolve().parent
if str(_EXPL) not in sys.path:
    sys.path.insert(0, str(_EXPL))

from config_load import attach_moe_start_layer
from survey_load import load_survey

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures" / "moe_start_layer_vs_parameters.png"


def main() -> None:
    df = attach_moe_start_layer(load_survey())
    plot_df = df[df["moe_start_layer"].notna()].copy()
    n_missing = len(df) - len(plot_df)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(
        plot_df["param_b"],
        plot_df["moe_start_layer"],
        alpha=0.75,
        s=48,
        edgecolors="white",
        linewidths=0.5,
    )
    ax.set_xscale("log")
    ax.set_xlabel("parameters (billions)")
    ax.set_ylabel("moe_start_layer")
    ax.set_title(
        "Survey models: moe_start_layer vs parameters\n"
        f"({len(plot_df)} MoE configs plotted; {n_missing} without moe_start_layer in config)"
    )
    ax.grid(True, alpha=0.3, which="both")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT, dpi=150)
    plt.close(fig)
    print(f"Wrote {OUT} ({len(plot_df)} models, {n_missing} skipped)")


if __name__ == "__main__":
    main()
