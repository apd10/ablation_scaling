"""Thin wrapper around utility.return_fits for backward compatibility."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_EXPL = Path(__file__).resolve().parent
_ROOT = _EXPL.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from utility import FitResult, return_fits  # noqa: E402

LINE_NAMES = ("split", "below", "above")


@dataclass
class IterativeFitResult:
    lines: tuple[tuple[float, float], tuple[float, float], tuple[float, float]]
    assignments: np.ndarray
    counts: tuple[int, int, int]
    n_iters: int
    converged: bool
    mse: float

    def line(self, idx: int) -> tuple[float, float]:
        return self.lines[idx]

    def equation(self, idx: int) -> str:
        m, b = self.lines[idx]
        return f"num_layers = {m:.3f}·log10(param_b) + {b:.2f}"


def _to_iterative(result: FitResult) -> IterativeFitResult:
    if result.k != 3:
        raise ValueError(f"expected k=3, got {result.k}")
    return IterativeFitResult(
        lines=(result.lines[0], result.lines[1], result.lines[2]),
        assignments=result.assignments,
        counts=(result.counts[0], result.counts[1], result.counts[2]),
        n_iters=result.n_iters,
        converged=result.converged,
        mse=result.mse,
    )


def fit_iterative_from_survey(df) -> IterativeFitResult:
    return _to_iterative(return_fits(df, kmeans_num_fit=3, x_axis_type="log"))
