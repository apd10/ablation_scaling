"""Scatter: layer_size (x) vs expert_size / hidden_size (y) from survey CSV."""

from pathlib import Path

import matplotlib.pyplot as plt
import sys

_EXPL = Path(__file__).resolve().parent
if str(_EXPL) not in sys.path:
    sys.path.insert(0, str(_EXPL))

from survey_load import load_survey

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures" / "expert_ratio_vs_layer_size.png"


def main() -> None:
    df = load_survey().copy()
    df["expert_over_hidden"] = df["expert_size"] / df["hidden_size"]
    df = df[
        (df["layer_size_m"] > 0)
        & (df["hidden_size"] > 0)
        & (df["expert_size"] > 0)
    ]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(
        df["layer_size_m"],
        df["expert_over_hidden"],
        alpha=0.75,
        s=48,
        edgecolors="white",
        linewidths=0.5,
    )
    ax.set_xlabel("layer_size (millions of parameters per layer)")
    ax.set_ylabel("expert_size / hidden_size")
    ax.set_title("Survey models: expert_size / hidden_size vs layer_size")
    ax.grid(True, alpha=0.3)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT, dpi=150)
    plt.close(fig)
    print(f"Wrote {OUT} ({len(df)} models)")


if __name__ == "__main__":
    main()
