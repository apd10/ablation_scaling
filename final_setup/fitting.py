"""High-level fitting API: search k and function family, return callable line fits."""

from __future__ import annotations

from typing import Callable, Mapping, Sequence

from .outliers import OutlierRemoval, prepare_xy_for_fit
from .piecewise import FunctionFamily, PiecewiseFitResult, fit_piecewise, predict_line

PredictFn = Callable[[float | int], float]


def _sanitize_name(name: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in name.strip())
    if not cleaned:
        raise ValueError("name must contain at least one alphanumeric character")
    return cleaned


def _make_predict_fn(line: tuple[float, float], function_family: FunctionFamily) -> PredictFn:
    def predict(x: float | int) -> float:
        return float(predict_line(line, x, function_family))

    return predict


def _sorted_lines(result: PiecewiseFitResult) -> tuple[tuple[float, float], ...]:
    return tuple(sorted(result.lines, key=lambda line: line[0]))


def _validate_families(function_family_space: list[str]) -> list[FunctionFamily]:
    families: list[FunctionFamily] = []
    for family in function_family_space:
        if family not in ("linear", "log", "sqrt"):
            raise ValueError(f"unsupported function_family {family!r}; use linear, log, or sqrt")
        families.append(family)  # type: ignore[arg-type]
    return families


def _search_best_fit(
    x,
    y,
    kmeans_num_fits_space: list[int],
    function_family_space: list[str],
    *,
    anchor_x: float | None,
    anchor_y: float | None,
    outlier: OutlierRemoval | None = None,
    outlier_drop_indices: Sequence[int] = (),
) -> tuple[PiecewiseFitResult, tuple[int, ...]]:
    if not kmeans_num_fits_space:
        raise ValueError("kmeans_num_fits_space must be non-empty")
    if not function_family_space:
        raise ValueError("function_family_space must be non-empty")

    x_in, y_in, _, _, dropped = prepare_xy_for_fit(
        x,
        y,
        outlier=outlier,
        extra_drop_indices=outlier_drop_indices,
    )
    if len(x_in) < min(kmeans_num_fits_space):
        raise ValueError(
            f"not enough points after outlier removal ({len(x_in)} left; "
            f"need at least {min(kmeans_num_fits_space)})"
        )

    families = _validate_families(function_family_space)
    best: PiecewiseFitResult | None = None

    for k in kmeans_num_fits_space:
        if k < 1:
            raise ValueError(f"kmeans_num_fit must be >= 1, got {k}")
        for function_family in families:
            result = fit_piecewise(
                x_in,
                y_in,
                k=k,
                function_family=function_family,
                anchor_x=anchor_x,
                anchor_y=anchor_y,
            )
            if best is None or result.mse < best.mse:
                best = result

    assert best is not None
    return best, dropped


def fit_y_from_x(
    name: str,
    x,
    y,
    kmeans_num_fits_space: list[int],
    function_family_space: list[str],
    num_fits_to_return: int,
    *,
    anchor_x: float | None = None,
    anchor_y: float | None = None,
    remove_outliers: bool = False,
    outlier_n_remove_x: int = 1,
    outlier_n_remove_y: int = 1,
    outlier_drop_indices: Sequence[int] = (),
) -> Mapping[str, PredictFn]:
    """
    Search over k and function family, pick lowest-MSE fit, return line predictors.

    Parameters
    ----------
    name:
        Prefix for returned keys, e.g. ``num_layers`` -> ``num_layers_line_1``.
    x, y:
        Primary input (x-axis) and target arrays. Must be same length; only finite
        points with x > 0 and y > 0 are used.
    kmeans_num_fits_space:
        Candidate line counts, e.g. ``[2, 3]``.
    function_family_space:
        Candidate transforms on x: ``linear``, ``log`` (log10), ``sqrt``.
    num_fits_to_return:
        Number of line lambdas to return from the winning fit (capped at winning k).
    anchor_x, anchor_y:
        Optional point all lines must pass through.
    remove_outliers:
        If True, drop ``outlier_n_remove_x`` max-x and ``outlier_n_remove_y`` max-y
        points before fitting (after any ``outlier_drop_indices``).
    outlier_drop_indices:
        Additional positional indices to exclude (e.g. survey cluster rows).

    Returns
    -------
    dict[str, callable]
        ``{name}_line_1``, ``{name}_line_2``, ... each ``callable(x) -> y``.
    """
    if num_fits_to_return < 1:
        raise ValueError("num_fits_to_return must be >= 1")

    outlier = (
        OutlierRemoval(
            mode="axis",
            n_remove_x=outlier_n_remove_x,
            n_remove_y=outlier_n_remove_y,
        )
        if remove_outliers
        else None
    )

    best, _ = _search_best_fit(
        x,
        y,
        kmeans_num_fits_space,
        function_family_space,
        anchor_x=anchor_x,
        anchor_y=anchor_y,
        outlier=outlier,
        outlier_drop_indices=outlier_drop_indices,
    )
    return _lines_to_functions(name, best, num_fits_to_return)


def fit_y_from_x_with_meta(
    name: str,
    x,
    y,
    kmeans_num_fits_space: list[int],
    function_family_space: list[str],
    num_fits_to_return: int,
    *,
    anchor_x: float | None = None,
    anchor_y: float | None = None,
    remove_outliers: bool = False,
    outlier_n_remove_x: int = 1,
    outlier_n_remove_y: int = 1,
    outlier_drop_indices: Sequence[int] = (),
) -> tuple[Mapping[str, PredictFn], PiecewiseFitResult, tuple[int, ...]]:
    """Same as ``fit_y_from_x`` but also returns the winning fit and dropped indices."""
    if num_fits_to_return < 1:
        raise ValueError("num_fits_to_return must be >= 1")

    outlier = None
    if remove_outliers:
        outlier = OutlierRemoval(
            mode="axis",
            n_remove_x=outlier_n_remove_x,
            n_remove_y=outlier_n_remove_y,
        )

    best, dropped = _search_best_fit(
        x,
        y,
        kmeans_num_fits_space,
        function_family_space,
        anchor_x=anchor_x,
        anchor_y=anchor_y,
        outlier=outlier,
        outlier_drop_indices=outlier_drop_indices,
    )
    return _lines_to_functions(name, best, num_fits_to_return), best, dropped


def _lines_to_functions(
    name: str,
    result: PiecewiseFitResult,
    num_fits_to_return: int,
) -> Mapping[str, PredictFn]:
    prefix = _sanitize_name(name)
    n_return = min(num_fits_to_return, result.k)
    ordered_lines = _sorted_lines(result)[:n_return]
    return {
        f"{prefix}_line_{j + 1}": _make_predict_fn(line, result.function_family)
        for j, line in enumerate(ordered_lines)
    }
