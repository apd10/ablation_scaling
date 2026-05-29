"""
Plot parameters per layer (search_space.get_parameters) vs hidden_size.

One curve per LayerTypeConfig recipe.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from search_space import (
    AttentionConfig,
    DenseFFNConfig,
    FFNConfig,
    FullAttentionConfig,
    LayerTypeConfig,
    MoEFFNConfig,
    StandardMoEConfig,
    StandardShapeConfig,
    TopKRoutingConfig,
)

OUT = ROOT / "figures" / "layer_params_vs_hidden.png"


def shape_for_hidden(
    hidden: int,
    *,
    num_kv_heads: int | None = None,
    intermediate_ratio: float = 4.0,
) -> StandardShapeConfig:
    """Llama-style: head_dim ≈ 128, intermediate = ratio × hidden."""
    num_heads = max(1, hidden // 128)
    while hidden % num_heads != 0 and num_heads > 1:
        num_heads -= 1
    if num_kv_heads is None:
        num_kv_heads = num_heads
    else:
        num_kv_heads = min(num_kv_heads, num_heads)
    return StandardShapeConfig(
        hidden_size=hidden,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        intermediate_size=int(intermediate_ratio * hidden),
    )


def _layer(
    name: str,
    shape: StandardShapeConfig,
    *,
    moe_experts: int | None = None,
    expert_ratio: float = 3.5,
) -> LayerTypeConfig:
    if moe_experts is None:
        ffn = FFNConfig(mode=DenseFFNConfig(shape=shape))
    else:
        expert_h = int(expert_ratio * shape.hidden_size)
        ffn = FFNConfig(
            mode=MoEFFNConfig(
                shape=shape,
                moe=StandardMoEConfig(
                    num_experts=moe_experts,
                    expert_hidden_size=expert_h,
                    routing=TopKRoutingConfig(num_experts_per_tok=min(8, moe_experts)),
                ),
            )
        )
    return LayerTypeConfig(
        name=name,
        attention=AttentionConfig(
            shape=shape,
            mechanism=FullAttentionConfig(),
        ),
        ffn=ffn,
    )


def layer_config_builders() -> list[tuple[str, Callable[[int], LayerTypeConfig]]]:
    def dense_mha(h: int) -> LayerTypeConfig:
        s = shape_for_hidden(h)
        return _layer("dense_mha", s)

    def dense_gqa(h: int) -> LayerTypeConfig:
        s = shape_for_hidden(h)
        s = StandardShapeConfig(
            hidden_size=s.hidden_size,
            num_heads=s.num_heads,
            num_kv_heads=max(1, s.num_heads // 4),
            intermediate_size=s.intermediate_size,
        )
        return _layer("dense_gqa", s)

    def moe_8(h: int) -> LayerTypeConfig:
        return _layer("moe_8", shape_for_hidden(h), moe_experts=8)

    def moe_64(h: int) -> LayerTypeConfig:
        return _layer("moe_64", shape_for_hidden(h), moe_experts=64)

    return [
        ("Dense MHA (4× FFN)", dense_mha),
        ("Dense GQA (4× FFN)", dense_gqa),
        ("MoE 8 experts", moe_8),
        ("MoE 64 experts", moe_64),
    ]


def hidden_sizes() -> np.ndarray:
    return np.array(
        [256, 512, 768, 1024, 1536, 2048, 2560, 3072, 4096, 5120, 6144],
        dtype=int,
    )


def main() -> None:
    h_grid = hidden_sizes()
    fig, ax = plt.subplots(figsize=(9, 6))

    for label, builder in layer_config_builders():
        params_m = []
        for h in h_grid:
            layer = builder(int(h))
            n = layer.get_parameters()
            params_m.append(n / 1e6)
        ax.plot(params_m, h_grid, marker="o", markersize=4, linewidth=2, label=label)

    ax.set_xlabel("Parameters per layer (millions)")
    ax.set_ylabel("hidden_size")
    ax.set_title("hidden_size vs layer parameters (search_space.get_parameters)")
    ax.legend(loc="best", fontsize=8, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT, dpi=150)
    plt.close(fig)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
