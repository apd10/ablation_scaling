"""Clean piecewise fitting utilities extracted from the ablations survey work."""

from .fitting import PredictFn, fit_y_from_x, fit_y_from_x_with_meta
from .outliers import OutlierRemoval, prepare_xy_for_fit
from .piecewise import PiecewiseFitResult, fit_piecewise, predict_line

__all__ = [
    "OutlierRemoval",
    "PiecewiseFitResult",
    "PredictFn",
    "fit_piecewise",
    "fit_y_from_x",
    "fit_y_from_x_with_meta",
    "prepare_xy_for_fit",
    "predict_line",
]
