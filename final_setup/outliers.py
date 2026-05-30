"""Outlier removal helpers for piecewise fitting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np

OutlierMode = Literal["none", "axis", "survey_clusters", "both"]


@dataclass(frozen=True)
class OutlierRemoval:
    """Configuration for excluding points before fitting."""

    mode: OutlierMode = "none"
    n_remove_x: int = 1
    n_remove_y: int = 1

    def enabled(self) -> bool:
        return self.mode != "none"


def axis_outlier_indices(
    x: np.ndarray,
    y: np.ndarray,
    *,
    n_remove_x: int = 1,
    n_remove_y: int = 1,
) -> set[int]:
    """
    Drop the highest-x point(s), then highest-y on the remainder.

    Matches the num_experts fit convention in this repo.
    """
    n = len(x)
    drop: set[int] = set()
    if n == 0:
        return drop

    if n_remove_x > 0:
        for idx in np.argsort(x)[::-1][:n_remove_x]:
            drop.add(int(idx))

    remaining = [i for i in range(n) if i not in drop]
    if n_remove_y > 0 and remaining:
        y_remaining = y[remaining]
        for rank in np.argsort(y_remaining)[::-1][:n_remove_y]:
            drop.add(int(remaining[int(rank)]))

    return drop


def merge_drop_indices(*groups: Sequence[int]) -> tuple[int, ...]:
    merged: set[int] = set()
    for group in groups:
        merged.update(int(i) for i in group)
    return tuple(sorted(merged))


def apply_drop_indices(
    x: np.ndarray,
    y: np.ndarray,
    drop_indices: Sequence[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[int, ...]]:
    """
    Split (x, y) into inliers used for fitting and outliers for display.

    Returns ``x_in, y_in, x_out, y_out, drop_indices``.
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    drop = merge_drop_indices(drop_indices)
    if drop and (max(drop) >= len(x) or min(drop) < 0):
        raise ValueError("drop_indices out of range for x/y length")

    keep = np.ones(len(x), dtype=bool)
    if drop:
        keep[list(drop)] = False

    x_in, y_in = x[keep], y[keep]
    x_out, y_out = x[~keep], y[~keep]
    return x_in, y_in, x_out, y_out, drop


def prepare_xy_for_fit(
    x,
    y,
    *,
    outlier: OutlierRemoval | None = None,
    extra_drop_indices: Sequence[int] = (),
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[int, ...]]:
    """Apply configured outlier removal before fitting."""
    x_arr = np.asarray(x, dtype=float).ravel()
    y_arr = np.asarray(y, dtype=float).ravel()
    drop: list[int] = list(extra_drop_indices)

    if outlier is not None and outlier.enabled():
        if outlier.mode in ("axis", "both"):
            drop.extend(
                axis_outlier_indices(
                    x_arr,
                    y_arr,
                    n_remove_x=outlier.n_remove_x,
                    n_remove_y=outlier.n_remove_y,
                )
            )
        # survey_clusters handled via extra_drop_indices from caller

    return apply_drop_indices(x_arr, y_arr, drop)
