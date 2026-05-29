# LLM architecture search space

[`search_space.py`](search_space.py) defines a **typed config tree** for describing decoder LLM architectures: widths, attention mechanisms, FFN/MoE, norms, block topology, and heterogeneous depth patterns (e.g. MoE only on some layers).

Use it to:

- Specify architectures for ablation studies
- Enumerate valid design choices (`MechanismConfig.choices`, etc.)
- Build heterogeneous stacks (Mixtral-style, hybrid attn/SSM, MoE on last *k* layers)

**Requirements:** Python 3.10+ (uses `|` union syntax and `dataclasses`).

---

## Config tree overview

```
ModelConfig
├── EmbeddingConfig
├── PositionConfig              # ChoiceConfig (model-wide)
├── LayerStack
│   ├── num_layers
│   ├── layer_types[]           # LayerTypeConfig (templates)
│   └── stack[]                 # int per layer → index into layer_types
├── FinalNormConfig
└── LMHeadConfig

LayerTypeConfig                 # one block variant
├── BlockTopologyConfig         # ChoiceConfig
├── AttentionConfig
│   ├── ShapeConfig             # ChoiceConfig
│   ├── MechanismConfig         # ChoiceConfig
│   └── PositionInAttnConfig    # ChoiceConfig
├── FFNConfig
│   └── FFNModeConfig           # ChoiceConfig → dense | moe
│       └── MoEConfig           # ChoiceConfig (if moe)
│           └── MoERoutingConfig
├── NormConfig
├── ResidualConfig              # ChoiceConfig
├── DropoutConfig
├── InitConfig                  # ChoiceConfig
└── RecomputeConfig
```

**Two node kinds:**

| Kind | Base class | Role |
|------|------------|------|
| **Choice node** | `ChoiceConfig` | ABC with multiple concrete variants; exposes `.choices` and `.get(kind)` |
| **Container node** | plain `@dataclass` | Holds child configs (no `.choices`) |

---

## Parameter counting (`get_parameters`)

Every config node implements `get_parameters(**context) -> int`, returning the **number of learnable scalar weights** in that subtree.

- **Containers** sum their children and pass context down (`hidden_size`, `vocab_size`, `shape`, …).
- **Leaf choice nodes** use formulas for linear layers, SwiGLU FFNs, MoE routers/experts, norms, etc.
- **Shape-only nodes** (`StandardShapeConfig`, RoPE, routing metadata, dropout/init/recompute) return `0`.

```python
from search_space import default_model_config

cfg = default_model_config()
print(cfg.get_parameters())                    # full model
print(cfg.layer_stack.get_parameters())        # all transformer layers
print(cfg.layer_stack.layer_types[0].get_parameters())  # one block template
```

**Context keys** (passed automatically by parents when possible):

| Key | Used by |
|-----|---------|
| `shape` | `MechanismConfig`, `FFNModeConfig`, `MoEConfig` |
| `hidden_size` | `NormConfig`, `PositionConfig`, `ResidualConfig`, model-level |
| `vocab_size` | `LMHeadConfig`, `ModelConfig` |
| `attention_bias` | attention mechanisms |
| `activation` | FFN / MoE (`swiglu`, `geglu`, `gelu`, …) |

Mechanism configs require `shape=StandardShapeConfig` in context; `AttentionConfig.get_parameters()` supplies it.

**Tied embeddings:** when `embedding.tie_word_embeddings` and `lm_head.tie_word_embeddings` are both true, the LM projection is not double-counted (only optional output bias is added).

Counts are **approximate** for exotic blocks (MLA, linear attention, Mamba) and assume LLaMA-style layouts; use them for ablation budgeting, not exact checkpoint parity.

### Fields that affect parameter count

Only fields read by `get_parameters()` are listed. Everything else (dropout rates, RoPE `base`/`scaling`, loss coefficients, `name`, checkpointing flags, etc.) does **not** change the count.

#### Model (`ModelConfig`)

| Field | Config | Effect |
|-------|--------|--------|
| `vocab_size` | `EmbeddingConfig` | Embedding matrix rows |
| `hidden_size` | `EmbeddingConfig` | Embedding columns; drives layer width via `attention.shape` |
| `tie_word_embeddings` | `EmbeddingConfig` + `LMHeadConfig` | If both true, LM projection weights are not counted again |
| `bias` | `LMHeadConfig` | Untied head: `+vocab`; tied head: `+vocab` if bias |
| `position` (choice) | `PositionConfig` | `learned_absolute` adds `max_position_embeddings × hidden_size`; RoPE/none → 0 |
| `max_position_embeddings` | `LearnedPositionConfig` | Learned position table size |
| `norm` → `affine` | `FinalNormConfig` | Final RMS/LayerNorm weights |
| `layer_stack` | `LayerStack` | See below |

#### Stack (`LayerStack`)

| Field | Effect |
|-------|--------|
| `num_layers` | Number of layers summed (must match `len(stack)`) |
| `stack` | Per-layer index into `layer_types` (heterogeneous depth pattern) |
| `layer_types` | Full `LayerTypeConfig` per template (all fields below apply per occurrence) |

#### Layer (`LayerTypeConfig`)

| Field | Effect |
|-------|--------|
| `topology` (choice) | `serial` / `parallel` → 0 extra; use `residual` for layer-scale |
| `attention` | See **Attention** |
| `ffn` | See **FFN / MoE** |
| `norm` | See **Norm** |
| `residual` (choice) | `layer_scale` → `+2 × hidden_size`; `standard` → 0 |

`dropout`, `init`, `recompute`, and `LayerTypeConfig.name` do not affect the count.

#### Shape (`StandardShapeConfig`)

Used by attention and FFN; shared geometry for the block.

| Field | Effect |
|-------|--------|
| `hidden_size` | Width for projections, norms, embeddings |
| `num_heads` | Q/O head count |
| `head_dim` | Per-head width (default `hidden_size // num_heads`) |
| `num_kv_heads` | K/V width for GQA/MQA (default `num_heads`) |
| `intermediate_size` | Dense FFN inner dim (default `4 × hidden_size`) |

#### Attention (`AttentionConfig` + `MechanismConfig`)

| Field | Effect |
|-------|--------|
| `mechanism` (choice) | Which counting formula (full, linear, MLA, Mamba, …) |
| `attention_bias` | `AttentionConfig` → adds biases on Q/K/V/O linears |
| `qk_norm` | `AttentionConfig` → `+2 × num_heads × head_dim` |
| `shape` | All `StandardShapeConfig` fields above |

Mechanism-specific fields:

| Mechanism | Fields that affect count |
|-----------|--------------------------|
| `full_attention`, `sliding_window_attention`, `cross_attention` | `shape` only (sliding `window_size` ignored) |
| `linear_attention` | `shape` (+ fixed `hidden²` feature-map budget) |
| `mla` | `kv_lora_rank`, `q_lora_rank`, `rope_head_dim`, `shape` |
| `mamba` | `state_dim`, `conv_kernel`, `expand`, `shape` |

`PositionInAttnConfig` (RoPE / ALiBi / none), `dropout`, and `use_flash` do not affect the count.

#### FFN / MoE (`FFNConfig` → `FFNModeConfig`)

| Field | Effect |
|-------|--------|
| `mode` (choice) | `dense` vs `moe` |
| `activation` | `swiglu` / `geglu` (3 matrices) vs `gelu` / `relu` (2 matrices) |
| `shape` | `hidden_size`, `intermediate_size` for dense FFN |

**`DenseFFNConfig`:** `3 × hidden × intermediate` (SwiGLU) or `2 × hidden × intermediate` (GELU).

**`MoEFFNConfig` / `StandardMoEConfig`:**

| Field | Effect |
|-------|--------|
| `num_experts` | Router `hidden × E` and `E` expert FFNs |
| `expert_hidden_size` | Per-expert SwiGLU size |
| `num_shared_experts` | Extra dense experts always in the stack |
| `activation` | Same as dense FFN, per expert |

`MoERoutingConfig` (`top_k`, `group_limited`, `num_experts_per_tok`, …), `load_balancing_loss_coef`, and `router_z_loss_coef` do **not** affect stored parameter count (routing is not weighted).

#### Norm (`NormConfig`)

| Field | Effect |
|-------|--------|
| `norm_type` (choice) | `rmsnorm` + `affine` → `hidden`; `layernorm` + `affine` → `2 × hidden` |
| `affine` | `RMSNormConfig` / `LayerNormConfig` — if false, 0 |
| `placement` (choice) | `pre_norm` / `post_norm` → ×2 norms; `sandwich` → ×4 |

`NormConfig.qk_norm` is not used by `get_parameters()` (use `AttentionConfig.qk_norm`).

#### Impact tiers (largest → smallest)

**Tier 1 — usually dominates**

| Field(s) | Scales |
|----------|--------|
| `LayerStack.num_layers` | Linear on all layer weights |
| `StandardShapeConfig.hidden_size` | ~quadratic in block (attn + FFN); linear in embedding/router |
| `EmbeddingConfig.vocab_size`, `tie_word_embeddings`, `LMHeadConfig.bias` | `vocab × hidden` (×2 if untied head) |
| `StandardMoEConfig.num_experts`, `expert_hidden_size` | `num_experts × expert FFN` + router |
| `intermediate_size` (dense), `FFNModeConfig.activation` | `3·d·I` (SwiGLU) vs `2·d·I` (GELU) |

**Tier 2 — meaningful, smaller than tier 1**

| Field(s) | Scales |
|----------|--------|
| `LayerStack.stack`, `layer_types` | Per-layer sum; MoE vs dense templates |
| `num_kv_heads` | Reduces K/V only (GQA) |
| `num_heads`, `head_dim` | Usually coupled to `hidden_size` |
| `NormPlacementConfig`, `NormTypeConfig.affine` | `O(d)` per norm; ×2 or ×4 placement |
| `AttentionConfig.attention_bias`, `qk_norm` | Small add-ons on attention |

**Tier 3 — step change or negligible**

| Field(s) | Notes |
|----------|--------|
| `FFNModeConfig` dense vs `moe` | MoE often much larger |
| `MechanismConfig` choice | Different formulas (MLA, Mamba, linear, …) |
| `ResidualConfig.layer_scale` | `+2·hidden` |
| `LearnedPositionConfig.max_position_embeddings` | `max_pos × hidden`; RoPE/none → 0 |
| `MoERoutingConfig.*`, dropout, RoPE `base`, loss coefs | 0 in `get_parameters()` |

---

## `ChoiceConfig` pattern

Every architectural *family* (attention mechanism, FFN mode, norm type, …) subclasses `ChoiceConfig`. Each concrete variant is a `@dataclass` with a class-level discriminator:

```python
@dataclass
class FullAttentionConfig(MechanismConfig):
    kind: ClassVar[str] = "full_attention"
    dropout: float = 0.0
    use_flash: bool = True
```

When the subclass is defined, it is registered on the parent:

```python
>>> MechanismConfig.choices
('full_attention', 'sliding_window_attention', 'linear_attention', 'mla', 'mamba', 'cross_attention')

>>> MechanismConfig.get("mla")
<class 'MLAConfig'>

>>> FullAttentionConfig().kind
'full_attention'
```

**Adding a new variant:** subclass the ABC, set `kind`, add fields. No manual registry update—the new `kind` appears in `Parent.choices` automatically.

---

## Quick start

### Default homogeneous model

```python
from search_space import default_model_config

cfg = default_model_config()
# 32 identical layers, dense FFN, full attention, RoPE
```

Example functions for model / layer / component counting: [`examples.py`](examples.py).

### Heterogeneous stack (e.g. MoE on last 8 layers)

```python
from search_space import (
    ModelConfig,
    LayerStack,
    LayerTypeConfig,
    FFNConfig,
    MoEFFNConfig,
    StandardMoEConfig,
    TopKRoutingConfig,
)

dense = LayerTypeConfig(name="dense")
moe = LayerTypeConfig(
    name="moe",
    ffn=FFNConfig(
        mode=MoEFFNConfig(
            moe=StandardMoEConfig(
                num_experts=128,
                expert_hidden_size=768,
                routing=TopKRoutingConfig(num_experts_per_tok=8),
            )
        )
    ),
)

model = ModelConfig(
    layer_stack=LayerStack(
        num_layers=48,
        layer_types=[dense, moe],
        stack=[0] * 40 + [1] * 8,
    )
)
```

`LayerStack` validates that `len(stack) == num_layers` and that every index in `stack` is `< len(layer_types)`. If `stack` is omitted, it defaults to `[0] * num_layers`.

### Inspecting choices

```python
from search_space import MechanismConfig, FFNModeConfig, MoEConfig

print(MechanismConfig.choices)
print(FFNModeConfig.choices)   # ('dense', 'moe')
print(MoEConfig.choices)       # ('standard_moe', 'dense_as_moe')
```

---

## Registered choices (reference)

### `ShapeConfig`

| `kind` | Class |
|--------|--------|
| `standard` | `StandardShapeConfig` |

Fields: `hidden_size`, `num_heads`, `head_dim`, `num_kv_heads` (GQA/MQA), `intermediate_size`.

### `MechanismConfig`

| `kind` | Class |
|--------|--------|
| `full_attention` | `FullAttentionConfig` |
| `sliding_window_attention` | `SlidingWindowAttentionConfig` |
| `linear_attention` | `LinearAttentionConfig` |
| `mla` | `MLAConfig` |
| `mamba` | `MambaConfig` |
| `cross_attention` | `CrossAttentionConfig` |

### `PositionInAttnConfig`

| `kind` | Class |
|--------|--------|
| `rope` | `RoPEConfig` |
| `alibi` | `ALiBiConfig` |
| `none` | `NoPositionInAttnConfig` |

### `FFNModeConfig` / `MoEConfig` / `MoERoutingConfig`

| Parent | `kind` | Class |
|--------|--------|--------|
| `FFNModeConfig` | `dense` | `DenseFFNConfig` |
| `FFNModeConfig` | `moe` | `MoEFFNConfig` |
| `MoEConfig` | `standard_moe` | `StandardMoEConfig` |
| `MoEConfig` | `dense_as_moe` | `DenseMoEConfig` |
| `MoERoutingConfig` | `top_k` | `TopKRoutingConfig` |
| `MoERoutingConfig` | `group_limited` | `GroupLimitedRoutingConfig` |

### `NormTypeConfig` / `NormPlacementConfig`

| Parent | `kind` | Class |
|--------|--------|--------|
| `NormTypeConfig` | `rmsnorm` | `RMSNormConfig` |
| `NormTypeConfig` | `layernorm` | `LayerNormConfig` |
| `NormPlacementConfig` | `pre_norm` | `PreNormConfig` |
| `NormPlacementConfig` | `post_norm` | `PostNormConfig` |
| `NormPlacementConfig` | `sandwich` | `SandwichNormConfig` |

### `BlockTopologyConfig`

| `kind` | Class |
|--------|--------|
| `serial` | `SerialBlockConfig` |
| `parallel` | `ParallelBlockConfig` |
| `scaled_residual` | `ScaledResidualBlockConfig` |

### `ResidualConfig` / `InitConfig` / `PositionConfig`

| Parent | `kind` | Class |
|--------|--------|--------|
| `ResidualConfig` | `standard` | `StandardResidualConfig` |
| `ResidualConfig` | `layer_scale` | `LayerScaleResidualConfig` |
| `InitConfig` | `default` | `DefaultInitConfig` |
| `InitConfig` | `depth_scaled` | `DepthScaledInitConfig` |
| `PositionConfig` | `rope` | `GlobalRoPEConfig` |
| `PositionConfig` | `learned_absolute` | `LearnedPositionConfig` |
| `PositionConfig` | `none` | `NoGlobalPositionConfig` |

---

## Mapping to survey / Hugging Face fields

The repo includes [`Model Architecture Experiments - Survey.csv`](Model%20Architecture%20Experiments%20-%20Survey.csv) with common HF config fields. Rough mapping:

| CSV / HF field | Config location |
|----------------|-----------------|
| `hidden_size` | `EmbeddingConfig.hidden_size`, `StandardShapeConfig.hidden_size` |
| `num_layers` | `LayerStack.num_layers` |
| `num_attn_heads` | `StandardShapeConfig.num_heads` |
| `num_kv_heads` | `StandardShapeConfig.num_kv_heads` |
| `num_experts` | `StandardMoEConfig.num_experts` |
| `num_expert_per_tok` | `TopKRoutingConfig.num_experts_per_tok` |
| `expert_size` | `StandardMoEConfig.expert_hidden_size` |

Heterogeneous models (e.g. Nemotron Cascade, Qwen3-Next) map to **multiple `LayerTypeConfig` entries** plus a non-uniform `stack`, not a single global MoE flag.

---

## Design conventions

1. **Layer-to-layer variation** → different `LayerTypeConfig` templates + `LayerStack.stack`.
2. **Global settings** → `ModelConfig` (`embedding`, `position`, `lm_head`).
3. **Width / head counts** → `ShapeConfig` under attention or FFN.
4. **Different block families** (full attn vs linear vs Mamba) → different `MechanismConfig` variants, often as separate layer types in `stack`.

`search_space.py` describes architecture only—it does not build PyTorch modules, load checkpoints, or run training. A separate builder can walk `ModelConfig` and instantiate layers from each node's `.kind`.

---

## Public API

Import from `search_space` directly. Main entry points:

- `ModelConfig`, `LayerStack`, `LayerTypeConfig`
- `default_model_config()`
- All `ChoiceConfig` ABCs and their concrete `@dataclass` variants (see `__all__` in `search_space.py`)

---

## Extending the tree

Example: add sparse attention as a mechanism.

```python
from dataclasses import dataclass
from typing import ClassVar

from search_space import MechanismConfig


@dataclass
class SparseAttentionConfig(MechanismConfig):
    kind: ClassVar[str] = "sparse_attention"
    block_size: int = 128
    num_random_blocks: int = 3


# MechanismConfig.choices now includes "sparse_attention"
```

Use the new class in `LayerTypeConfig.attention.mechanism` or register it in your own module and re-export if you keep extensions outside `search_space.py`.

For a new **choice family** (not just another mechanism), add a new `ChoiceConfig` ABC and container field on `LayerTypeConfig` or `ModelConfig` as needed.

# fields to decide based on #parameter

num_layers
hidden_size
moe_start_layer
num_experts
expert_hidden_size
num_experts_per_tok
vocab_size
intermediate_size
