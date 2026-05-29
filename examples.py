"""Examples: computing parameter counts at model, layer, and component level."""

from __future__ import annotations

from search_space import (
    AttentionConfig,
    DenseFFNConfig,
    EmbeddingConfig,
    FFNConfig,
    FullAttentionConfig,
    LayerStack,
    LayerTypeConfig,
    ModelConfig,
    MoEFFNConfig,
    NormConfig,
    StandardMoEConfig,
    StandardShapeConfig,
    TopKRoutingConfig,
    default_model_config,
)


def model_parameter_count() -> int:
    """Total parameters for a full model config."""
    model = default_model_config()
    return model.get_parameters()


def model_parameter_breakdown() -> dict[str, int]:
    """Parameters by top-level model section."""
    model = default_model_config()
    ctx = {
        "hidden_size": model.embedding.hidden_size,
        "vocab_size": model.embedding.vocab_size,
    }
    return {
        "embedding": model.embedding.get_parameters(**ctx),
        "position": model.position.get_parameters(**ctx),
        "layer_stack": model.layer_stack.get_parameters(**ctx),
        "final_norm": model.final_norm.get_parameters(**ctx),
        "lm_head": model.lm_head.get_parameters(**ctx),
        "total": model.get_parameters(**ctx),
    }


def layer_parameter_count() -> dict[str, int]:
    """Parameters for one layer template and a heterogeneous stack."""
    dense_layer = LayerTypeConfig(
        name="dense",
        attention=AttentionConfig(
            shape=StandardShapeConfig(hidden_size=2048, num_heads=16),
            mechanism=FullAttentionConfig(),
        ),
        ffn=FFNConfig(
            mode=DenseFFNConfig(
                shape=StandardShapeConfig(
                    hidden_size=2048,
                    intermediate_size=5504,
                ),
            )
        ),
    )
    moe_layer = LayerTypeConfig(
        name="moe",
        ffn=FFNConfig(
            mode=MoEFFNConfig(
                shape=StandardShapeConfig(hidden_size=2048),
                moe=StandardMoEConfig(
                    num_experts=128,
                    expert_hidden_size=768,
                    routing=TopKRoutingConfig(num_experts_per_tok=8),
                ),
            )
        ),
    )
    stack = LayerStack(
        num_layers=48,
        layer_types=[dense_layer, moe_layer],
        stack=[0] * 40 + [1] * 8,
    )
    return {
        "dense_layer": dense_layer.get_parameters(),
        "moe_layer": moe_layer.get_parameters(),
        "stack": stack.get_parameters(),
    }


def component_parameter_count() -> dict[str, int]:
    """Parameters for attention, FFN, norm, and mechanism inside one block."""
    shape = StandardShapeConfig(
        hidden_size=2048,
        num_heads=16,
        num_kv_heads=4,
        intermediate_size=5504,
    )
    ctx = {"hidden_size": shape.hidden_size, "shape": shape}

    attention = AttentionConfig(
        shape=shape,
        mechanism=FullAttentionConfig(),
        qk_norm=True,
    )
    ffn = FFNConfig(mode=DenseFFNConfig(shape=shape, activation="swiglu"))
    norm = NormConfig()
    moe_ffn = MoEFFNConfig(
        shape=shape,
        moe=StandardMoEConfig(num_experts=64, expert_hidden_size=1408),
    )

    return {
        "mechanism": attention.mechanism.get_parameters(
            **ctx, attention_bias=attention.attention_bias
        ),
        "attention": attention.get_parameters(**ctx),
        "ffn_dense": ffn.get_parameters(**ctx),
        "norm": norm.get_parameters(**ctx),
        "ffn_moe": moe_ffn.get_parameters(**ctx),
    }


def custom_model_counts() -> dict[str, int]:
    """Model, layer, and submodule counts for a hand-built config."""
    layer = LayerTypeConfig(
        ffn=FFNConfig(
            mode=MoEFFNConfig(
                moe=StandardMoEConfig(num_experts=64, expert_hidden_size=1408),
            )
        ),
    )
    model = ModelConfig(
        embedding=EmbeddingConfig(vocab_size=32000, hidden_size=2048),
        layer_stack=LayerStack(num_layers=24, layer_types=[layer]),
    )
    hidden = model.embedding.hidden_size
    return {
        "model": model.get_parameters(),
        "layer": layer.get_parameters(),
        "attention": layer.attention.get_parameters(),
        "ffn": layer.ffn.get_parameters(),
        "norm": layer.norm.get_parameters(hidden_size=hidden),
    }
