"""Read architecture fields from cached Hugging Face config.json files."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def repo_dir_name(repo_id: str) -> str:
    return repo_id.replace("/", "__")


def config_path(repo_id: str, config_dir: Path = CONFIG_DIR) -> Path:
    return config_dir / repo_dir_name(repo_id) / "config.json"


def moe_start_layer_from_config(config: dict) -> int | None:
    """
    First MoE layer index from HF config.

    Uses ``first_k_dense_replace`` when present (DeepSeek-style: MoE starts after
    that many dense layers). Otherwise 0 for MoE model types, or after ``mlp_only_layers``.
    """
    value = config.get("first_k_dense_replace")
    if isinstance(value, (int, float)) and value == value:
        return int(value)

    mlp_only = config.get("mlp_only_layers")
    if isinstance(mlp_only, list) and len(mlp_only) > 0:
        return int(max(mlp_only)) + 1

    model_type = str(config.get("model_type", "")).lower()
    if config.get("decoder_sparse_step") is not None:
        return 0
    if "moe" in model_type or config.get("num_local_experts"):
        return 0
    return None


def load_config(repo_id: str, config_dir: Path = CONFIG_DIR) -> dict | None:
    path = config_path(repo_id, config_dir)
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def attach_moe_start_layer(
    df: pd.DataFrame,
    *,
    config_dir: Path = CONFIG_DIR,
    url_col: str = "url",
) -> pd.DataFrame:
    """Add ``moe_start_layer`` column (nullable) from ``configs/<repo>/config.json``."""
    df = df.copy()
    starts: list[int | None] = []
    for repo_id in df[url_col].astype(str):
        config = load_config(repo_id, config_dir)
        starts.append(moe_start_layer_from_config(config) if config else None)
    df["moe_start_layer"] = starts
    return df
