"""Local web UI for survey scatter / piecewise-fit plots."""

from __future__ import annotations

import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from final_setup.outliers import apply_drop_indices
from final_setup.web.data import NUMERIC_COLUMNS, compute_plot_outlier_indices, load_plot_data, plot_frame
from final_setup.web.export import generate_fit_module
from final_setup.web.plots import build_plot_response, fit_for_plot

STATIC_DIR = Path(__file__).resolve().parent / "static"
app = FastAPI(title="Ablations Plot Explorer")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

_DF = load_plot_data()


class PlotRequest(BaseModel):
    x_col: str
    y_col: str
    log_x: bool = False
    enable_fit: bool = False
    kmeans_num_fits_space: list[int] = Field(default_factory=lambda: [2, 3])
    function_family_space: list[str] = Field(default_factory=lambda: ["log", "linear", "sqrt"])
    anchor_x: float | None = None
    anchor_y: float | None = None
    remove_survey_clusters: bool = False
    outlier_n_remove_x: int = 0
    outlier_n_remove_y: int = 0


def _validate_columns(body: PlotRequest) -> None:
    if body.x_col not in _DF.columns or body.y_col not in _DF.columns:
        raise HTTPException(status_code=400, detail="invalid x or y column")
    if body.x_col == body.y_col:
        raise HTTPException(status_code=400, detail="x and y must differ")


def _validate_fit_options(body: PlotRequest) -> None:
    if not body.enable_fit:
        return
    if not body.kmeans_num_fits_space:
        raise HTTPException(status_code=400, detail="select at least one k in kmeans_num_fits_space")
    if not body.function_family_space:
        raise HTTPException(status_code=400, detail="select at least one function family")


def _load_xy(body: PlotRequest):
    frame = plot_frame(_DF, body.x_col, body.y_col)
    if frame.empty:
        raise HTTPException(status_code=400, detail="no valid points for selected columns")

    drop_indices = compute_plot_outlier_indices(
        frame,
        body.x_col,
        body.y_col,
        remove_survey_clusters=body.remove_survey_clusters,
        n_remove_x=body.outlier_n_remove_x,
        n_remove_y=body.outlier_n_remove_y,
    )
    x = frame[body.x_col].to_numpy()
    y = frame[body.y_col].to_numpy()
    return x, y, drop_indices


def _run_fit(body: PlotRequest, x, y, drop_indices):
    x_in, y_in, _, _, dropped = apply_drop_indices(x, y, drop_indices)
    if len(x_in) < min(body.kmeans_num_fits_space):
        raise HTTPException(
            status_code=400,
            detail=(
                f"not enough points after outlier removal ({len(x_in)} left; "
                f"need at least {min(body.kmeans_num_fits_space)} to fit)"
            ),
        )
    result = fit_for_plot(
        x_in,
        y_in,
        kmeans_num_fits_space=body.kmeans_num_fits_space,
        function_family_space=body.function_family_space,
        anchor_x=body.anchor_x,
        anchor_y=body.anchor_y,
    )
    meta = {
        "x_col": body.x_col,
        "y_col": body.y_col,
        "n_points": int(len(x)),
        "n_fit_points": int(len(x_in)),
        "n_outliers_removed": int(len(dropped)),
        "outlier_indices": list(dropped),
        "anchor_x": body.anchor_x,
        "anchor_y": body.anchor_y,
        "k": result.k,
        "function_family": result.function_family,
        "mse": result.mse,
        "counts": list(result.counts),
    }
    return result, meta


@app.get("/")
def index() -> FileResponse:
    return FileResponse(
        STATIC_DIR / "index.html",
        headers={"Cache-Control": "no-cache"},
    )


@app.get("/api/columns")
def columns() -> dict:
    available = [c for c in NUMERIC_COLUMNS if c in _DF.columns]
    return {"columns": available}


@app.post("/api/plot")
def plot(body: PlotRequest) -> dict:
    _validate_columns(body)
    _validate_fit_options(body)
    x, y, drop_indices = _load_xy(body)

    try:
        return build_plot_response(
            x,
            y,
            x_label=body.x_col,
            y_label=body.y_col,
            log_x=body.log_x,
            enable_fit=body.enable_fit,
            kmeans_num_fits_space=body.kmeans_num_fits_space,
            function_family_space=body.function_family_space,
            anchor_x=body.anchor_x,
            anchor_y=body.anchor_y,
            outlier_drop_indices=drop_indices,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/export")
def export_fit(body: PlotRequest) -> Response:
    _validate_columns(body)
    if not body.enable_fit:
        raise HTTPException(status_code=400, detail="enable piecewise fit before exporting")
    _validate_fit_options(body)

    x, y, drop_indices = _load_xy(body)
    try:
        result, meta = _run_fit(body, x, y, drop_indices)
        source = generate_fit_module(
            x_col=body.x_col,
            y_col=body.y_col,
            result=result,
            meta=meta,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    filename = f"fit_{body.y_col}_from_{body.x_col}.py"
    return Response(
        content=source,
        media_type="text/x-python",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


def main() -> None:
    import uvicorn

    uvicorn.run(
        "final_setup.web.app:app",
        host="127.0.0.1",
        port=8765,
        reload=True,
    )


if __name__ == "__main__":
    main()
