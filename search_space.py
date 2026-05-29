"""
LLM architecture search space as a config tree.

Each interior node is an abstract base class with a ``choices`` tuple listing
registered concrete variants. Subclasses set ``kind: ClassVar[str]``; registration
happens automatically via :class:`ChoiceConfig`.
"""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field
from typing import Any, ClassVar, TypeVar

T = TypeVar("T", bound="ChoiceConfig")


# ---------------------------------------------------------------------------
# Parameter counting helpers
# ---------------------------------------------------------------------------


def _require_shape(context: dict[str, Any], owner: str) -> "StandardShapeConfig":
    shape = context.get("shape")
    if not isinstance(shape, StandardShapeConfig):
        raise TypeError(f"{owner}.get_parameters() requires shape=StandardShapeConfig")
    return shape


def _resolve_shape(shape: StandardShapeConfig) -> tuple[int, int, int, int, int]:
    """Return ``hidden``, ``num_heads``, ``head_dim``, ``num_kv_heads``, ``intermediate``."""
    hidden = shape.hidden_size
    num_heads = shape.num_heads
    head_dim = shape.head_dim or (hidden // num_heads)
    num_kv_heads = shape.num_kv_heads or num_heads
    intermediate = shape.intermediate_size or (4 * hidden)
    return hidden, num_heads, head_dim, num_kv_heads, intermediate


def _linear(hidden: int, out: int, *, bias: bool = False) -> int:
    return hidden * out + (out if bias else 0)


def _mha_projections(
    hidden: int,
    num_heads: int,
    head_dim: int,
    num_kv_heads: int,
    *,
    bias: bool = False,
) -> int:
    q = _linear(hidden, num_heads * head_dim, bias=bias)
    k = _linear(hidden, num_kv_heads * head_dim, bias=bias)
    v = _linear(hidden, num_kv_heads * head_dim, bias=bias)
    o = _linear(num_heads * head_dim, hidden, bias=bias)
    return q + k + v + o


def _swiglu_ffn(hidden: int, intermediate: int, *, bias: bool = False) -> int:
    # gate, up, down
    return (
        _linear(hidden, intermediate, bias=bias)
        + _linear(hidden, intermediate, bias=bias)
        + _linear(intermediate, hidden, bias=bias)
    )


def _geglu_ffn(hidden: int, intermediate: int, *, bias: bool = False) -> int:
    return _swiglu_ffn(hidden, intermediate, bias=bias)


def _gelu_ffn(hidden: int, intermediate: int, *, bias: bool = False) -> int:
    return _linear(hidden, intermediate, bias=bias) + _linear(
        intermediate, hidden, bias=bias
    )


def _ffn_from_activation(
    activation: str, hidden: int, intermediate: int, *, bias: bool = False
) -> int:
    if activation in ("swiglu", "geglu"):
        fn = _geglu_ffn if activation == "geglu" else _swiglu_ffn
        return fn(hidden, intermediate, bias=bias)
    if activation in ("gelu", "relu", "squared_relu"):
        return _gelu_ffn(hidden, intermediate, bias=bias)
    raise ValueError(f"Unknown activation {activation!r}")


class ChoiceConfig(ABC):
    """Base for every node in the config tree."""

    choices: ClassVar[tuple[str, ...]] = ()
    _registry: ClassVar[dict[str, type[ChoiceConfig]]]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        kind = getattr(cls, "kind", None)
        if not kind or not isinstance(kind, str):
            return
        # Walk ABC bases to register on the immediate ChoiceConfig parent(s).
        for base in cls.__mro__[1:]:
            if base is ChoiceConfig or base is object:
                continue
            if not issubclass(base, ChoiceConfig):
                continue
            reg: dict[str, type[ChoiceConfig]] = getattr(base, "_registry", None) or {}
            reg = dict(reg)
            reg[kind] = cls  # type: ignore[assignment]
            base._registry = reg  # type: ignore[attr-defined]
            base.choices = tuple(reg.keys())

    @property
    def kind(self) -> str:
        """Discriminator string; must match an entry in the parent's ``choices``."""
        k = getattr(type(self), "kind", None)
        if not isinstance(k, str):
            raise AttributeError(f"{type(self).__name__} has no class-level kind")
        return k

    @classmethod
    def get(cls: type[T], kind: str) -> type[T]:
        registry: dict[str, type[T]] = getattr(cls, "_registry", {})
        try:
            return registry[kind]
        except KeyError as exc:
            raise ValueError(
                f"Unknown {cls.__name__} kind {kind!r}; "
                f"expected one of {cls.choices}"
            ) from exc

    def get_parameters(self, **context: Any) -> int:
        """Learnable parameter count for this node (see ``**context`` on subclasses)."""
        return 0


# ---------------------------------------------------------------------------
# Shape (shared by attention / FFN)
# ---------------------------------------------------------------------------


class ShapeConfig(ChoiceConfig):
    """Width/head geometry for a submodule."""


@dataclass
class StandardShapeConfig(ShapeConfig):
    kind: ClassVar[str] = "standard"
    hidden_size: int = 2048
    num_heads: int = 16
    head_dim: int | None = None  # default: hidden_size // num_heads
    num_kv_heads: int | None = None  # GQA/MQA; default: num_heads
    intermediate_size: int | None = None  # FFN hidden; default: ~4x hidden

    def get_parameters(self, **context: Any) -> int:
        return 0


# ---------------------------------------------------------------------------
# Attention mechanism
# ---------------------------------------------------------------------------


class MechanismConfig(ChoiceConfig):
    """How tokens mix (attention family or SSM)."""


@dataclass
class FullAttentionConfig(MechanismConfig):
    kind: ClassVar[str] = "full_attention"
    dropout: float = 0.0
    use_flash: bool = True

    def get_parameters(self, **context: Any) -> int:
        shape = _require_shape(context, type(self).__name__)
        hidden, num_heads, head_dim, num_kv_heads, _ = _resolve_shape(shape)
        bias = bool(context.get("attention_bias", False))
        return _mha_projections(
            hidden, num_heads, head_dim, num_kv_heads, bias=bias
        )


@dataclass
class SlidingWindowAttentionConfig(MechanismConfig):
    kind: ClassVar[str] = "sliding_window_attention"
    window_size: int = 4096
    dropout: float = 0.0

    def get_parameters(self, **context: Any) -> int:
        return FullAttentionConfig().get_parameters(**context)


@dataclass
class LinearAttentionConfig(MechanismConfig):
    kind: ClassVar[str] = "linear_attention"
    feature_map: str = "elu"  # kernel feature map name
    chunk_size: int = 64

    def get_parameters(self, **context: Any) -> int:
        # Q/K/V/O plus a modest feature-map projection budget.
        base = FullAttentionConfig().get_parameters(**context)
        shape = _require_shape(context, type(self).__name__)
        hidden, _, _, _, _ = _resolve_shape(shape)
        return base + hidden * hidden


@dataclass
class MLAConfig(MechanismConfig):
    """Multi-latent attention (low-rank KV)."""

    kind: ClassVar[str] = "mla"
    kv_lora_rank: int = 512
    q_lora_rank: int | None = None
    rope_head_dim: int | None = None

    def get_parameters(self, **context: Any) -> int:
        shape = _require_shape(context, type(self).__name__)
        hidden, num_heads, head_dim, _, _ = _resolve_shape(shape)
        q_rank = self.q_lora_rank or self.kv_lora_rank
        rope_dim = self.rope_head_dim or (head_dim // 2)
        nope_dim = head_dim - rope_dim
        # Simplified DeepSeek-style MLA projection budget.
        q = _linear(hidden, q_rank) + _linear(
            q_rank, num_heads * (nope_dim + rope_dim)
        )
        kv_down = _linear(hidden, self.kv_lora_rank)
        k_up = _linear(self.kv_lora_rank, num_heads * nope_dim)
        v_up = _linear(self.kv_lora_rank, num_heads * head_dim)
        o = _linear(num_heads * head_dim, hidden)
        return q + kv_down + k_up + v_up + o


@dataclass
class MambaConfig(MechanismConfig):
    kind: ClassVar[str] = "mamba"
    state_dim: int = 16
    conv_kernel: int = 4
    expand: int = 2

    def get_parameters(self, **context: Any) -> int:
        shape = _require_shape(context, type(self).__name__)
        hidden, _, _, _, _ = _resolve_shape(shape)
        inner = self.expand * hidden
        dt_rank = max(1, hidden // 16)
        in_proj = _linear(hidden, 2 * inner)
        conv1d = inner * self.conv_kernel
        x_proj = _linear(inner, dt_rank + 2 * self.state_dim)
        dt_proj = _linear(dt_rank, inner)
        out_proj = _linear(inner, hidden)
        return in_proj + conv1d + x_proj + dt_proj + out_proj


@dataclass
class CrossAttentionConfig(MechanismConfig):
    kind: ClassVar[str] = "cross_attention"
    dropout: float = 0.0

    def get_parameters(self, **context: Any) -> int:
        return FullAttentionConfig().get_parameters(**context)


# ---------------------------------------------------------------------------
# Position encoding inside attention
# ---------------------------------------------------------------------------


class PositionInAttnConfig(ChoiceConfig):
    """Positional treatment at the attention block."""


@dataclass
class RoPEConfig(PositionInAttnConfig):
    kind: ClassVar[str] = "rope"
    base: float = 10000.0
    scaling: str | None = None  # e.g. "yarn", "ntk"
    max_position_embeddings: int = 8192


@dataclass
class ALiBiConfig(PositionInAttnConfig):
    kind: ClassVar[str] = "alibi"
    max_position_embeddings: int = 8192


@dataclass
class NoPositionInAttnConfig(PositionInAttnConfig):
    """Absolute positions handled outside the block (e.g. learned embeddings)."""

    kind: ClassVar[str] = "none"


# RoPE / ALiBi / none: no learned weights in standard formulations.


# ---------------------------------------------------------------------------
# Attention block
# ---------------------------------------------------------------------------


@dataclass
class AttentionConfig:
    shape: ShapeConfig = field(default_factory=StandardShapeConfig)
    mechanism: MechanismConfig = field(default_factory=FullAttentionConfig)
    position: PositionInAttnConfig = field(default_factory=RoPEConfig)
    qk_norm: bool = False
    attention_bias: bool = False

    def get_parameters(self, **context: Any) -> int:
        ctx = {**context, "shape": self.shape, "attention_bias": self.attention_bias}
        shape = _require_shape(ctx, type(self).__name__)
        _, num_heads, head_dim, _, _ = _resolve_shape(shape)
        params = self.shape.get_parameters(**ctx)
        params += self.mechanism.get_parameters(**ctx)
        params += self.position.get_parameters(**ctx)
        if self.qk_norm:
            params += 2 * num_heads * head_dim
        return params


# ---------------------------------------------------------------------------
# MoE / FFN
# ---------------------------------------------------------------------------


class MoERoutingConfig(ChoiceConfig):
    """Token-to-expert assignment strategy."""


@dataclass
class TopKRoutingConfig(MoERoutingConfig):
    kind: ClassVar[str] = "top_k"
    num_experts_per_tok: int = 8
    normalize_router_weights: bool = True


@dataclass
class GroupLimitedRoutingConfig(MoERoutingConfig):
    kind: ClassVar[str] = "group_limited"
    num_experts_per_tok: int = 8
    num_groups: int = 8
    topk_groups: int = 4


class MoEConfig(ChoiceConfig):
    """Mixture-of-experts FFN."""

    @property
    def num_experts(self) -> int:
        raise NotImplementedError

    def get_parameters(self, **context: Any) -> int:
        raise NotImplementedError


@dataclass
class StandardMoEConfig(MoEConfig):
    kind: ClassVar[str] = "standard_moe"
    num_experts: int = 64
    expert_hidden_size: int = 1408
    routing: MoERoutingConfig = field(default_factory=TopKRoutingConfig)
    num_shared_experts: int = 0
    load_balancing_loss_coef: float = 0.01
    router_z_loss_coef: float = 0.0

    def get_parameters(self, **context: Any) -> int:
        shape = _require_shape(context, type(self).__name__)
        hidden, _, _, _, _ = _resolve_shape(shape)
        activation = str(context.get("activation", "swiglu"))
        bias = bool(context.get("bias", False))
        router = _linear(hidden, self.num_experts, bias=bias)
        experts = self.num_experts * _ffn_from_activation(
            activation, hidden, self.expert_hidden_size, bias=bias
        )
        shared = self.num_shared_experts * _ffn_from_activation(
            activation, hidden, self.expert_hidden_size, bias=bias
        )
        return router + experts + shared + self.routing.get_parameters(**context)


@dataclass
class DenseMoEConfig(MoEConfig):
    """Single expert (dense FFN expressed as 1-of-1 MoE)."""

    kind: ClassVar[str] = "dense_as_moe"
    expert_hidden_size: int = 5504

    @property
    def num_experts(self) -> int:
        return 1

    def get_parameters(self, **context: Any) -> int:
        shape = _require_shape(context, type(self).__name__)
        hidden, _, _, _, _ = _resolve_shape(shape)
        activation = str(context.get("activation", "swiglu"))
        bias = bool(context.get("bias", False))
        router = _linear(hidden, 1, bias=bias)
        expert = _ffn_from_activation(
            activation, hidden, self.expert_hidden_size, bias=bias
        )
        return router + expert


class FFNModeConfig(ChoiceConfig):
    """Dense vs mixture feed-forward."""


@dataclass
class DenseFFNConfig(FFNModeConfig):
    kind: ClassVar[str] = "dense"
    shape: ShapeConfig = field(default_factory=StandardShapeConfig)
    activation: str = "swiglu"  # swiglu | geglu | gelu

    def get_parameters(self, **context: Any) -> int:
        shape = _require_shape({**context, "shape": self.shape}, type(self).__name__)
        hidden, _, _, _, intermediate = _resolve_shape(shape)
        bias = bool(context.get("bias", False))
        return _ffn_from_activation(self.activation, hidden, intermediate, bias=bias)


@dataclass
class MoEFFNConfig(FFNModeConfig):
    kind: ClassVar[str] = "moe"
    shape: ShapeConfig = field(default_factory=StandardShapeConfig)
    moe: MoEConfig = field(default_factory=StandardMoEConfig)
    activation: str = "swiglu"

    def get_parameters(self, **context: Any) -> int:
        ctx = {**context, "shape": self.shape, "activation": self.activation}
        return self.moe.get_parameters(**ctx)


@dataclass
class FFNConfig:
    mode: FFNModeConfig = field(default_factory=DenseFFNConfig)

    def get_parameters(self, **context: Any) -> int:
        ctx = {**context}
        if isinstance(self.mode, DenseFFNConfig):
            ctx.setdefault("shape", self.mode.shape)
        elif isinstance(self.mode, MoEFFNConfig):
            ctx.setdefault("shape", self.mode.shape)
            ctx.setdefault("activation", self.mode.activation)
        return self.mode.get_parameters(**ctx)


# ---------------------------------------------------------------------------
# Norm
# ---------------------------------------------------------------------------


class NormTypeConfig(ChoiceConfig):
    """Normalization operator."""


@dataclass
class RMSNormConfig(NormTypeConfig):
    kind: ClassVar[str] = "rmsnorm"
    eps: float = 1e-6
    affine: bool = True

    def get_parameters(self, **context: Any) -> int:
        hidden = int(context["hidden_size"])
        return hidden if self.affine else 0


@dataclass
class LayerNormConfig(NormTypeConfig):
    kind: ClassVar[str] = "layernorm"
    eps: float = 1e-5
    affine: bool = True

    def get_parameters(self, **context: Any) -> int:
        hidden = int(context["hidden_size"])
        return (2 * hidden) if self.affine else 0


class NormPlacementConfig(ChoiceConfig):
    """Where norms sit relative to sub-blocks."""


@dataclass
class PreNormConfig(NormPlacementConfig):
    kind: ClassVar[str] = "pre_norm"


@dataclass
class PostNormConfig(NormPlacementConfig):
    kind: ClassVar[str] = "post_norm"


@dataclass
class SandwichNormConfig(NormPlacementConfig):
    kind: ClassVar[str] = "sandwich"


@dataclass
class NormConfig:
    norm_type: NormTypeConfig = field(default_factory=RMSNormConfig)
    placement: NormPlacementConfig = field(default_factory=PreNormConfig)
    qk_norm: bool = False

    def get_parameters(self, **context: Any) -> int:
        hidden = int(context["hidden_size"])
        ctx = {**context, "hidden_size": hidden}
        per_norm = self.norm_type.get_parameters(**ctx)
        if isinstance(self.placement, SandwichNormConfig):
            count = 4
        elif isinstance(self.placement, PostNormConfig):
            count = 2
        else:
            count = 2
        return per_norm * count


# ---------------------------------------------------------------------------
# Block topology
# ---------------------------------------------------------------------------


class BlockTopologyConfig(ChoiceConfig):
    """Residual wiring and sub-block layout."""


@dataclass
class SerialBlockConfig(BlockTopologyConfig):
    """Classic: attn then FFN (or reversed via ``subblock_order``)."""

    kind: ClassVar[str] = "serial"
    subblock_order: tuple[str, ...] = ("attention", "ffn")


@dataclass
class ParallelBlockConfig(BlockTopologyConfig):
    """GPT-J style: attention and FFN branches in parallel."""

    kind: ClassVar[str] = "parallel"


@dataclass
class ScaledResidualBlockConfig(BlockTopologyConfig):
    kind: ClassVar[str] = "scaled_residual"
    residual_scale: float = 1.0
    subblock_order: tuple[str, ...] = ("attention", "ffn")

    def get_parameters(self, **context: Any) -> int:
        return 0


# ---------------------------------------------------------------------------
# Residual / dropout / init / recompute (optional layer extras)
# ---------------------------------------------------------------------------


class ResidualConfig(ChoiceConfig):
    pass


@dataclass
class StandardResidualConfig(ResidualConfig):
    kind: ClassVar[str] = "standard"
    dropout: float = 0.0


@dataclass
class LayerScaleResidualConfig(ResidualConfig):
    kind: ClassVar[str] = "layer_scale"
    attn_scale: float = 1.0
    ffn_scale: float = 1.0

    def get_parameters(self, **context: Any) -> int:
        hidden = int(context["hidden_size"])
        return 2 * hidden


@dataclass
class DropoutConfig:
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0
    residual_dropout: float = 0.0

    def get_parameters(self, **context: Any) -> int:
        return 0


class InitConfig(ChoiceConfig):
    pass


@dataclass
class DefaultInitConfig(InitConfig):
    kind: ClassVar[str] = "default"
    std: float = 0.02


@dataclass
class DepthScaledInitConfig(InitConfig):
    kind: ClassVar[str] = "depth_scaled"
    base_std: float = 0.02


@dataclass
class RecomputeConfig:
    """Activation checkpointing for this layer type."""

    enabled: bool = False
    granularity: str = "full_block"  # full_block | attention | ffn

    def get_parameters(self, **context: Any) -> int:
        return 0


# ---------------------------------------------------------------------------
# Layer type & stack
# ---------------------------------------------------------------------------


@dataclass
class LayerTypeConfig:
    """Full recipe for one block variant (indexed by stack ids)."""

    name: str = "default"
    topology: BlockTopologyConfig = field(default_factory=SerialBlockConfig)
    attention: AttentionConfig = field(default_factory=AttentionConfig)
    ffn: FFNConfig = field(default_factory=FFNConfig)
    norm: NormConfig = field(default_factory=NormConfig)
    residual: ResidualConfig = field(default_factory=StandardResidualConfig)
    dropout: DropoutConfig = field(default_factory=DropoutConfig)
    init: InitConfig = field(default_factory=DefaultInitConfig)
    recompute: RecomputeConfig = field(default_factory=RecomputeConfig)

    def get_parameters(self, **context: Any) -> int:
        if not isinstance(self.attention.shape, StandardShapeConfig):
            raise TypeError("LayerTypeConfig requires attention.shape=StandardShapeConfig")
        hidden, _, _, _, _ = _resolve_shape(self.attention.shape)
        ctx = {**context, "hidden_size": hidden}
        return (
            self.topology.get_parameters(**ctx)
            + self.attention.get_parameters(**ctx)
            + self.ffn.get_parameters(**ctx)
            + self.norm.get_parameters(**ctx)
            + self.residual.get_parameters(**ctx)
            + self.dropout.get_parameters(**ctx)
            + self.init.get_parameters(**ctx)
            + self.recompute.get_parameters(**ctx)
        )


@dataclass
class LayerStack:
    num_layers: int = 32
    layer_types: list[LayerTypeConfig] = field(
        default_factory=lambda: [LayerTypeConfig()]
    )
    stack: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.stack:
            self.stack = [0] * self.num_layers
        if len(self.stack) != self.num_layers:
            raise ValueError(
                f"stack length {len(self.stack)} != num_layers {self.num_layers}"
            )
        max_id = max(self.stack, default=-1)
        if max_id >= len(self.layer_types):
            raise ValueError(
                f"stack references type {max_id} but only "
                f"{len(self.layer_types)} layer_types defined"
            )

    @property
    def num_types(self) -> int:
        return len(self.layer_types)

    def get_parameters(self, **context: Any) -> int:
        return sum(
            self.layer_types[i].get_parameters(**context) for i in self.stack
        )


# ---------------------------------------------------------------------------
# Model-level
# ---------------------------------------------------------------------------


class PositionConfig(ChoiceConfig):
    """Global positional encoding (model scope)."""


@dataclass
class GlobalRoPEConfig(PositionConfig):
    kind: ClassVar[str] = "rope"
    base: float = 10000.0
    scaling: str | None = None
    max_position_embeddings: int = 8192


@dataclass
class LearnedPositionConfig(PositionConfig):
    kind: ClassVar[str] = "learned_absolute"
    max_position_embeddings: int = 8192

    def get_parameters(self, **context: Any) -> int:
        hidden = int(context["hidden_size"])
        return self.max_position_embeddings * hidden


@dataclass
class NoGlobalPositionConfig(PositionConfig):
    kind: ClassVar[str] = "none"


@dataclass
class EmbeddingConfig:
    vocab_size: int = 128256
    hidden_size: int = 2048
    embedding_dropout: float = 0.0
    tie_word_embeddings: bool = True

    def get_parameters(self, **context: Any) -> int:
        return self.vocab_size * self.hidden_size


@dataclass
class LMHeadConfig:
    tie_word_embeddings: bool = True
    bias: bool = False
    logit_softcap: float | None = None

    def get_parameters(self, **context: Any) -> int:
        vocab = int(context["vocab_size"])
        if self.tie_word_embeddings:
            return vocab if self.bias else 0
        hidden = int(context["hidden_size"])
        params = vocab * hidden
        if self.bias:
            params += vocab
        return params


@dataclass
class FinalNormConfig:
    norm: NormTypeConfig = field(default_factory=RMSNormConfig)

    def get_parameters(self, **context: Any) -> int:
        return self.norm.get_parameters(**context)


@dataclass
class ModelConfig:
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    position: PositionConfig = field(default_factory=GlobalRoPEConfig)
    layer_stack: LayerStack = field(default_factory=LayerStack)
    final_norm: FinalNormConfig = field(default_factory=FinalNormConfig)
    lm_head: LMHeadConfig = field(default_factory=LMHeadConfig)
    architecture: str = "decoder_only_causal_lm"

    def get_parameters(self, **context: Any) -> int:
        hidden = self.embedding.hidden_size
        vocab = self.embedding.vocab_size
        tie = self.embedding.tie_word_embeddings and self.lm_head.tie_word_embeddings
        ctx = {**context, "hidden_size": hidden, "vocab_size": vocab}
        total = self.embedding.get_parameters(**ctx)
        total += self.position.get_parameters(**ctx)
        total += self.layer_stack.get_parameters(**ctx)
        total += self.final_norm.get_parameters(**ctx)
        if not tie:
            total += self.lm_head.get_parameters(**ctx)
        elif self.lm_head.bias:
            total += vocab
        return total


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------


def default_model_config() -> ModelConfig:
    """Homogeneous decoder stack (all layers share type 0)."""
    layer_type = LayerTypeConfig(
        name="dense_decoder",
        ffn=FFNConfig(mode=DenseFFNConfig()),
        attention=AttentionConfig(
            mechanism=FullAttentionConfig(),
            position=RoPEConfig(),
        ),
    )
    return ModelConfig(
        layer_stack=LayerStack(
            num_layers=32,
            layer_types=[layer_type],
            stack=[0] * 32,
        ),
    )


__all__ = [
    "ChoiceConfig",
    "ShapeConfig",
    "StandardShapeConfig",
    "MechanismConfig",
    "FullAttentionConfig",
    "SlidingWindowAttentionConfig",
    "LinearAttentionConfig",
    "MLAConfig",
    "MambaConfig",
    "CrossAttentionConfig",
    "PositionInAttnConfig",
    "RoPEConfig",
    "ALiBiConfig",
    "NoPositionInAttnConfig",
    "AttentionConfig",
    "MoERoutingConfig",
    "TopKRoutingConfig",
    "GroupLimitedRoutingConfig",
    "MoEConfig",
    "StandardMoEConfig",
    "DenseMoEConfig",
    "FFNModeConfig",
    "DenseFFNConfig",
    "MoEFFNConfig",
    "FFNConfig",
    "NormTypeConfig",
    "RMSNormConfig",
    "LayerNormConfig",
    "NormPlacementConfig",
    "PreNormConfig",
    "PostNormConfig",
    "SandwichNormConfig",
    "NormConfig",
    "BlockTopologyConfig",
    "SerialBlockConfig",
    "ParallelBlockConfig",
    "ScaledResidualBlockConfig",
    "ResidualConfig",
    "StandardResidualConfig",
    "LayerScaleResidualConfig",
    "DropoutConfig",
    "InitConfig",
    "DefaultInitConfig",
    "DepthScaledInitConfig",
    "RecomputeConfig",
    "LayerTypeConfig",
    "LayerStack",
    "PositionConfig",
    "GlobalRoPEConfig",
    "LearnedPositionConfig",
    "NoGlobalPositionConfig",
    "EmbeddingConfig",
    "LMHeadConfig",
    "FinalNormConfig",
    "ModelConfig",
    "default_model_config",
]
