"""Scatter: intermediate_size (x) vs hidden_size (y) from survey CSV."""

from pathlib import Path

import matplotlib.pyplot as plt
import sys

_EXPL = Path(__file__).resolve().parent
if str(_EXPL) not in sys.path:
    sys.path.insert(0, str(_EXPL))

from survey_load import load_survey

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures" / "intermediate_vs_hidden_size.png"


def main() -> None:
    df = load_survey().copy()
    df["intermediate_size"] = df["num_experts"] * df["expert_size"]
    df = df[(df["intermediate_size"] > 0) & (df["hidden_size"] > 0)]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(
        df["intermediate_size"],
        df["hidden_size"],
        alpha=0.75,
        s=48,
        edgecolors="white",
        linewidths=0.5,
    )
    ax.set_xlabel("intermediate_size (num_experts × expert_size)")
    ax.set_ylabel("hidden_size")
    ax.set_title("Survey models: intermediate_size vs hidden_size")
    ax.grid(True, alpha=0.3)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT, dpi=150)
    plt.close(fig)
    print(f"Wrote {OUT} ({len(df)} models)")


if __name__ == "__main__":
    main()
