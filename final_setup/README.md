# final_setup

Piecewise fitting library + local plot explorer for the architecture survey.

## Run the plot explorer

From the **repo root** (`Ablations/`):

```powershell
cd path\to\Ablations
pip install -r final_setup/requirements.txt
python run_plot_explorer.py
```

Open **http://127.0.0.1:8765**. Stop with **Ctrl+C**. Hard-refresh (**Ctrl+F5**) if the UI looks stale.

## How to use

1. Pick **X** and **Y** columns. Toggle **Log scale on X** if needed.
2. Set **Outliers** (optional).
3. Check which **k** values and **function families** to try.
4. Enable **piecewise fit** (and anchor if needed).
5. Click **Generate plot**.
6. With fit enabled, a **table** below the plot lists exponentially spaced X over the data range and Y per line.
7. Click **Export Python** to copy code into your project.

Fit metadata (chosen k, family, MSE) appears under the buttons after a plot with fit enabled.

---

### Outliers

Two independent filters; both apply before plotting and fitting.

| Control | What it does |
|--------|----------------|
| **Survey clusters** | Drops 8 fixed rows (3 Nemotron / DeepSeek clusters from `CONFIG.FIT_OUTLIER_COORDS`). |
| **Remove highest X** | Drops the top *N* points by X value. |
| **Remove highest Y after X** | On the remaining points, drops the top *N* by Y. |

Excluded points are not used for the scatter fit lines. Outlier settings also apply to export.

---

### Fit search space

Defines which models are **tried** when piecewise fit is on. The UI picks the **lowest-MSE** `(k, family)` pair from your selections.

**k-means num fits** — number of piecewise lines (1–5). Check every k you want considered (e.g. 2 and 3).

**Function family** — x-transform for each line:

| Family | Model shape |
|--------|-------------|
| `linear` | `y = slope * x + intercept` |
| `log` | `y = slope * log10(x) + intercept` |
| `sqrt` | `y = slope * sqrt(x) + intercept` |

Check at least one k and one family. More options = wider search, slightly slower.

---

### Piecewise fit

- **Enable k-means line fits** — draw fitted lines on the plot and unlock export.
- **Anchor X / Y** (optional) — force one line through `(anchor_x, anchor_y)` (e.g. `1024, 1` for num_experts fits).

Without enable fit: scatter only, no export.

---

### Export Python

Requires **Enable k-means line fits**.

1. Click **Export Python**.
2. Copy from the dialog (**Copy to clipboard** or select all).
3. Paste into a `.py` file.

Generated code includes:

- `{y_col}_line_1`, `{y_col}_line_2`, … — callables `f(x) -> y`
- `FUNCTIONS` — dict mapping names to those functions
- `FIT_META` — k, family, MSE, slopes/intercepts, outlier counts

Example:

```python
from my_fit_module import FUNCTIONS

y = FUNCTIONS["num_layers_line_1"](1.0)  # x in same units as the plot axis
```

Export uses the **same** axes, outliers, search space, and anchor as the current sidebar settings (not necessarily the last plot if you changed settings without replotting).

---

## Python API (no web UI)

```python
from final_setup.fitting import fit_y_from_x_with_meta

result, meta = fit_y_from_x_with_meta(
    x, y,
    kmeans_num_fits_space=[2, 3],
    function_family_space=["log", "linear", "sqrt"],
)
```

See `fitting.py`, `piecewise.py`, and `outliers.py`.
