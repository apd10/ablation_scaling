"""Regenerate Choice 1 table in search_space.md (k=3 log₁₀ num_layers fits)."""

from __future__ import annotations

import re
import sys
from pathlib import Path

_EXPL = Path(__file__).resolve().parent
if str(_EXPL) not in sys.path:
    sys.path.insert(0, str(_EXPL))

_ROOT = _EXPL.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from CONFIG import KMEANS_NUM_FITS
from log_fits import ANCHOR_LAYERS, ANCHOR_PARAM_B
from survey_load import exclude_fit_outliers, load_survey
from utility import FitResult, predict, return_fits

OUT_PATH = _ROOT / "search_space.md"
PHASE1_MARKER = "# Choice 1:"
PHASE2_MARKER = "# Choice2:"

PARAM_LABELS: list[tuple[str, float]] = [
    ("100M", 0.1),
    ("250M", 0.25),
    ("500M", 0.5),
    ("1B", 1.0),
    ("2B", 2.0),
    ("4B", 4.0),
    ("8B", 8.0),
    ("16B", 16.0),
    ("32B", 32.0),
    ("64B", 64.0),
    ("100B", 100.0),
    ("128B", 128.0),
    ("500B", 500.0),
    ("1T", 1000.0),
]

X_AXIS_TYPE = "log"


def layer_size_params(param_b: float) -> int:
    return int(round(param_b * 1e9))


def fmt_params(n: int) -> str:
    if n >= 1_000_000_000_000:
        v, suffix = n / 1e12, "T"
    elif n >= 1_000_000_000:
        v, suffix = n / 1e9, "B"
    elif n >= 1_000_000:
        v, suffix = n / 1e6, "M"
    else:
        return str(n)
    text = f"{v:.2f}".rstrip("0").rstrip(".")
    return f"{text}{suffix}"


def sorted_lines(result: FitResult) -> tuple[tuple[tuple[float, float], int], ...]:
    """Order lines by slope (ascending) for stable table columns 1–3."""
    indexed = [(result.lines[j], result.counts[j]) for j in range(result.k)]
    return tuple(sorted(indexed, key=lambda item: item[0][0]))


def fmt_layers_eq(line: tuple[float, float]) -> str:
    slope, intercept = line
    sign = "+" if intercept >= 0 else "−"
    return f"`num_layers = {slope:.3f}·log₁₀(param_b) {sign} {abs(intercept):.2f}`"


def build_table(sorted_line_data: tuple[tuple[tuple[float, float], int], ...]) -> str:
    lines_only = [line for line, _ in sorted_line_data]
    header = (
        "| Parameters | num_layers 1 | num_layers 2 | num_layers 3 | "
        "layer_size 1 | layer_size 2 | layer_size 3 |"
    )
    sep = (
        "|------------|-------------:|-------------:|-------------:|"
        "-------------:|-------------:|-------------:|"
    )
    rows = [header, sep]
    for label, pb in PARAM_LABELS:
        total = layer_size_params(pb)
        layer_cols: list[str] = []
        size_cols: list[str] = []
        for line in lines_only:
            n_layers = int(round(predict(line, pb, X_AXIS_TYPE)))
            layer_cols.append(str(n_layers))
            size_cols.append(
                fmt_params(round(total / n_layers)) if n_layers > 0 else "—"
            )
        rows.append(
            "| "
            + " | ".join([label, *layer_cols, *size_cols])
            + " |"
        )
    return "\n".join(rows)


def build_phase1_section(result: FitResult) -> str:
    ordered = sorted_lines(result)
    table = build_table(ordered)
    eq_rows = "\n".join(
        f"| **{j + 1}** | {count} | {fmt_layers_eq(line)} |"
        for j, (line, count) in enumerate(ordered)
    )

    return f"""{PHASE1_MARKER} Layer depth vs parameter budget (survey fits)

Fits from [Model Architecture Experiments - Survey.csv](Model%20Architecture%20Experiments%20-%20Survey.csv), same method as [layers vs params (k=3, log)](figures/layers_vs_params_k3_log.png):

1. **k=3** k-means on survey models: **`num_layers = a·log₁₀(param_b) + b`** (`param_b` = billions), **through (100M, 1 layer)**.
2. Three Nemotron / DeepSeek-V4-Pro outlier clusters excluded (**150** models).
3. Table columns **1–3** are the three fit lines ordered by slope (lowest → highest).

### Equations

All fits pass through **`(param_b = 0.1, num_layers = 1)`** i.e. **100M params → 1 layer**.

| Line | Points | Equation |
|------|--------|----------|
{eq_rows}

MSE (piecewise fit): **{result.mse:.2f}**.

`layer_size` = total parameters ÷ predicted `num_layers` for that line, shown as **M** / **B**. `—` when layers ≤ 0.

## 1. num_layers

{table}

```bash
python explorations/generate_layer_table.py
python explorations/plot_layers_vs_params_by_axis.py
```
"""


def replace_phase1(markdown: str, phase1: str) -> str:
    if PHASE2_MARKER in markdown:
        rest = markdown[markdown.index(PHASE2_MARKER) :]
        return phase1.rstrip() + "\n\n---\n\n" + rest.lstrip()
    if PHASE1_MARKER in markdown:
        markdown = re.sub(
            rf"{re.escape(PHASE1_MARKER)}.*?(?=\n---\n\n# |\Z)",
            "",
            markdown,
            count=1,
            flags=re.DOTALL,
        )
    return markdown.rstrip() + "\n\n---\n\n" + phase1.lstrip()


def main() -> None:
    df = exclude_fit_outliers(load_survey())
    result = return_fits(
        df,
        kmeans_num_fit=KMEANS_NUM_FITS,
        x_axis_type=X_AXIS_TYPE,
        x_col="param_b",
        y_col="num_layers",
        anchor_x=ANCHOR_PARAM_B,
        anchor_y=ANCHOR_LAYERS,
        exclude_outliers=False,
    )
    phase1 = build_phase1_section(result)
    text = OUT_PATH.read_text(encoding="utf-8") if OUT_PATH.is_file() else ""
    OUT_PATH.write_text(replace_phase1(text, phase1), encoding="utf-8")
    print(f"Updated Choice 1 in {OUT_PATH} (k={result.k} {X_AXIS_TYPE}, MSE={result.mse:.2f})")


if __name__ == "__main__":
    main()
