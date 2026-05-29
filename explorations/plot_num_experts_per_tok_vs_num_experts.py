"""Scatter: num_experts_per_tok (y) vs num_experts (x) from survey CSV."""

from pathlib import Path

import matplotlib.pyplot as plt
import sys

_EXPL = Path(__file__).resolve().parent
if str(_EXPL) not in sys.path:
    sys.path.insert(0, str(_EXPL))

from survey_load import load_survey

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures" / "num_experts_per_tok_vs_num_experts.png"


def main() -> None:
    df = load_survey().copy()
    df = df[(df["num_experts"] > 0) & (df["num_expert_per_tok"] > 0)]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(
        df["num_experts"],
        df["num_expert_per_tok"],
        alpha=0.75,
        s=48,
        edgecolors="white",
        linewidths=0.5,
    )
    ax.set_xlabel("num_experts")
    ax.set_ylabel("num_experts_per_tok")
    ax.set_title("Survey models: num_experts_per_tok vs num_experts")
    ax.grid(True, alpha=0.3)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT, dpi=150)
    plt.close(fig)
    print(f"Wrote {OUT} ({len(df)} models)")


if __name__ == "__main__":
    main()
