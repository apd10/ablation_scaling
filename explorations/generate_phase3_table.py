"""Append Phase 3 table to search_space.md (num_experts / expert_size from fits)."""

from __future__ import annotations

import re
import sys
from pathlib import Path

_EXPL = Path(__file__).resolve().parent
_ROOT = _EXPL.parent
for p in (_ROOT, _EXPL):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from CONFIG import (
    KMEANS_NUM_FIT_HIDDEN_SIZE_VS_LAYER_SIZE,
    KMEANS_NUM_FITS,
    NUM_EXPERTS_FIT_ANCHOR_INTERMEDIATE,
    NUM_EXPERTS_FIT_ANCHOR_NUM_EXPERTS,
)
from num_experts_fit_data import load_num_experts_fit_data
from survey_load import exclude_fit_outliers, load_survey
from utility import predict, return_fits

OUT_PATH = _ROOT / "search_space.md"
PHASE3_MARKER = "# Phase 3:"

# Distinct layer_size_m values from Phase 2 table (M per layer from Phase 1 grid).
LAYER_SIZES_M: tuple[float, ...] = (
    25,
    29.41,
    35.71,
    41.67,
    45.45,
    62.5,
    64.52,
    100,
    105.26,
    166.67,
    177.78,
    275.86,
    307.69,
    484.85,
    542.37,
    842.11,
    969.7,
    1410,
    1520,
    1750,
    2220,
    2780,
    5750,
    9090,
    10640,
    16950,
)


def load_phase2_ref_df():
    """Phase 2 intermediate vs layer_size reference (3 cluster outliers excluded)."""
    df = exclude_fit_outliers(load_survey()).copy()
    df["intermediate_size_m"] = df["num_experts"] * df["expert_size"] / 1e6
    return df[
        (df["layer_size_m"] > 0)
        & (df["intermediate_size_m"] > 0)
        & (df["num_experts"] > 0)
        & (df["expert_size"] > 0)
    ].copy()


def load_num_experts_df():
    """Full survey minus 2 axis outliers (for num_experts fits only)."""
    df, _ = load_num_experts_fit_data()
    return df[(df["expert_size"] > 0)].copy()


def fmt_k(n: float) -> str:
    v = n / 1e3
    text = f"{v:.2f}".rstrip("0").rstrip(".")
    return f"{text}K"


def fmt_num_experts_eq(line: tuple[float, float]) -> str:
    slope, intercept = line
    sign = "+" if intercept >= 0 else "−"
    return f"`num_experts = {slope:.3e}·intermediate_size {sign} {abs(intercept):.2f}`"


def phase2_intermediate_sizes(inter_lines: tuple[tuple[float, float], ...]) -> list[float]:
    values: set[int] = set()
    for layer_m in LAYER_SIZES_M:
        for line in inter_lines:
            inter = predict(line, layer_m, "sqrt") * 1e6
            values.add(int(round(inter)))
    return sorted(values)


def row_cells(inter: float, ne_lines: tuple[tuple[float, float], ...]) -> tuple[str, str, str, str, str]:
    cols: list[str] = [fmt_k(inter)]
    for line in ne_lines:
        n_exp = predict(line, inter, "linear")
        if n_exp > 0:
            cols.append(str(int(round(n_exp))))
            cols.append(str(int(round(inter / n_exp))))
        else:
            cols.extend(("—", "—"))
    return tuple(cols)  # type: ignore[return-value]


def build_table_rows(
    intermediate_sizes: list[float],
    ne_lines: tuple[tuple[float, float], ...],
) -> str:
    header = (
        "| intermediate_size | num_experts fit 1 | expert_size 1 | "
        "num_experts fit 2 | expert_size 2 |"
    )
    sep = "|------------------:|-------------------:|--------------:|-------------------:|--------------:|"
    lines = [header, sep]
    for inter in intermediate_sizes:
        cells = row_cells(float(inter), ne_lines)
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def format_intermediate_eq(slope_m: float) -> str:
    coeff = slope_m * 1e6
    return f"`intermediate_size = {coeff:.0f}·√(layer_size / 10⁶)`"


def build_phase3_section(df_phase2, df_ne) -> str:
    inter_result = return_fits(
        df_phase2,
        kmeans_num_fit=KMEANS_NUM_FITS,
        x_axis_type="sqrt",
        x_col="layer_size_m",
        y_col="intermediate_size_m",
        anchor_x=0.0,
        anchor_y=0.0,
        exclude_outliers=False,
    )
    ne_result = return_fits(
        df_ne,
        kmeans_num_fit=KMEANS_NUM_FIT_HIDDEN_SIZE_VS_LAYER_SIZE,
        x_axis_type="linear",
        x_col="intermediate_size",
        y_col="num_experts",
        anchor_x=NUM_EXPERTS_FIT_ANCHOR_INTERMEDIATE,
        anchor_y=NUM_EXPERTS_FIT_ANCHOR_NUM_EXPERTS,
        exclude_outliers=False,
    )

    intermediate_sizes = phase2_intermediate_sizes(inter_result.lines)
    table = build_table_rows(intermediate_sizes, ne_result.lines)

    inter_eq_rows = "\n".join(
        f"| **{j + 1}** | {inter_result.counts[j]} | {format_intermediate_eq(line[0])} |"
        for j, line in enumerate(inter_result.lines)
    )
    ne_eq_rows = "\n".join(
        f"| **{j + 1}** | {ne_result.counts[j]} | {fmt_num_experts_eq(line)} |"
        for j, line in enumerate(ne_result.lines)
    )

    return f"""---

{PHASE3_MARKER} selecting num_experts and expert_size

Fits from [Model Architecture Experiments - Survey.csv](Model%20Architecture%20Experiments%20-%20Survey.csv), same method as [num_experts vs intermediate_size (k=2, linear)](figures/num_experts_vs_intermediate_size_k2_linear.png) and [theoretical expert_size vs intermediate_size](figures/theoretical_expert_size_vs_intermediate_size.png):

1. **k=2** k-means on survey models: **`num_experts = a·intermediate_size + b`** through **`(intermediate_size = 1024, num_experts = 1)`** (`intermediate_size` = `num_experts × expert_size`).
2. **Theoretical expert_size** per fit line: **`expert_size = intermediate_size / num_experts(fit)`**.
3. **Two axis outliers** excluded before num_experts fitting (full **158**-model survey otherwise): highest **`intermediate_size`**, then highest **`num_experts`** on the remainder — **{len(df_ne)}** models used.
4. Row **`intermediate_size`** values are the distinct Phase 2 predictions (`k=3` sqrt intermediate vs `layer_size`, through origin) at each Phase 1 `layer_size`, shown as **K**. `—` when a fit predicts `num_experts ≤ 0`.

### Equations

**Intermediate_size** (Phase 2 reference; `layer_size_m` = layer_size / 10⁶):

| Line | Points | Equation |
|------|--------|----------|
{inter_eq_rows}

**num_experts** (k=2 linear vs `intermediate_size`):

| Line | Points | Equation |
|------|--------|----------|
{ne_eq_rows}

MSE (num_experts fit): **{ne_result.mse:.2f}**. All lines pass through **`(1024, 1)`**. `—` when a line predicts `num_experts ≤ 0`.

## 3. num_experts and expert_size (from num_experts fits)

{table}

```bash
python explorations/generate_phase3_table.py
python explorations/plot_num_experts_vs_intermediate_size_fits.py
python explorations/plot_theoretical_expert_size_vs_intermediate_size.py
```
"""


def append_phase3(markdown: str, phase3: str) -> str:
    if PHASE3_MARKER in markdown:
        markdown = re.sub(
            r"\n---\n\n# Phase 3:.*\Z",
            "",
            markdown,
            flags=re.DOTALL,
        )
    return markdown.rstrip() + "\n" + phase3


def main() -> None:
    df_phase2 = load_phase2_ref_df()
    df_ne = load_num_experts_df()
    inter_result = return_fits(
        df_phase2,
        kmeans_num_fit=KMEANS_NUM_FITS,
        x_axis_type="sqrt",
        x_col="layer_size_m",
        y_col="intermediate_size_m",
        anchor_x=0.0,
        anchor_y=0.0,
        exclude_outliers=False,
    )
    n_rows = len(phase2_intermediate_sizes(inter_result.lines))
    phase3 = build_phase3_section(df_phase2, df_ne)
    text = OUT_PATH.read_text(encoding="utf-8")
    OUT_PATH.write_text(append_phase3(text, phase3), encoding="utf-8")
    print(f"Appended Phase 3 to {OUT_PATH} ({n_rows} rows)")


if __name__ == "__main__":
    main()
