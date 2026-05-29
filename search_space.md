# Choice 1: Layer depth vs parameter budget (survey fits)

Fits from [Model Architecture Experiments - Survey.csv](Model%20Architecture%20Experiments%20-%20Survey.csv), same method as [layers vs params (k=3, log)](figures/layers_vs_params_k3_log.png):

1. **k=3** k-means on survey models: **`num_layers = a·log₁₀(param_b) + b`** (`param_b` = billions), **through (100M, 1 layer)**.
2. Three Nemotron / DeepSeek-V4-Pro outlier clusters excluded (**150** models).
3. Table columns **1–3** are the three fit lines ordered by slope (lowest → highest).

### Equations

All fits pass through **`(param_b = 0.1, num_layers = 1)`** i.e. **100M params → 1 layer**.

| Line | Points | Equation |
|------|--------|----------|
| **1** | 44 | `num_layers = 11.033·log₁₀(param_b) + 12.03` |
| **2** | 83 | `num_layers = 17.191·log₁₀(param_b) + 18.19` |
| **3** | 23 | `num_layers = 26.447·log₁₀(param_b) + 27.45` |

MSE (piecewise fit): **25.89**.

`layer_size` = total parameters ÷ predicted `num_layers` for that line, shown as **M** / **B**. `—` when layers ≤ 0.

## 1. num_layers

| Parameters | num_layers 1 | num_layers 2 | num_layers 3 | layer_size 1 | layer_size 2 | layer_size 3 |
|------------|-------------:|-------------:|-------------:|-------------:|-------------:|-------------:|
| 100M | 1 | 1 | 1 | 100M | 100M | 100M |
| 250M | 5 | 8 | 12 | 50M | 31.25M | 20.83M |
| 500M | 9 | 13 | 19 | 55.56M | 38.46M | 26.32M |
| 1B | 12 | 18 | 27 | 83.33M | 55.56M | 37.04M |
| 2B | 15 | 23 | 35 | 133.33M | 86.96M | 57.14M |
| 4B | 19 | 29 | 43 | 210.53M | 137.93M | 93.02M |
| 8B | 22 | 34 | 51 | 363.64M | 235.29M | 156.86M |
| 16B | 25 | 39 | 59 | 640M | 410.26M | 271.19M |
| 32B | 29 | 44 | 67 | 1.1B | 727.27M | 477.61M |
| 64B | 32 | 49 | 75 | 2B | 1.31B | 853.33M |
| 100B | 34 | 53 | 80 | 2.94B | 1.89B | 1.25B |
| 128B | 35 | 54 | 83 | 3.66B | 2.37B | 1.54B |
| 500B | 42 | 65 | 99 | 11.9B | 7.69B | 5.05B |
| 1T | 45 | 70 | 107 | 22.22B | 14.29B | 9.35B |

```bash
python explorations/generate_layer_table.py
python explorations/plot_layers_vs_params_by_axis.py
```

---

# Choice2: Selecting hidden_size

Fits from [Model Architecture Experiments - Survey.csv](Model%20Architecture%20Experiments%20-%20Survey.csv), same method as [intermediate vs layer_size (k=3, sqrt)](figures/intermediate_vs_layer_size_k3_sqrt.png) and [theoretical hidden vs layer_size](figures/theoretical_hidden_vs_layer_size.png):

1. **k=3** k-means on survey models: **`intermediate_size_m = a·√(layer_size_m)`** through **(0, 0)** (`intermediate_size_m` = `num_experts × expert_size / 1e6`).
2. **Theoretical hidden_size** per line: **`hidden_size = layer_size / 3 / intermediate_size`** (`layer_size` = parameters per layer, `intermediate_size` = `num_experts × expert_size`).
3. Three Nemotron / DeepSeek-V4-Pro outlier clusters excluded (same as other fits).

### Equations (fits use `layer_size_m` = layer_size / 10⁶)

| Line | Points | Equation |
|------|--------|----------|
| **1** | 92 | `intermediate_size = 3689·√(layer_size / 10⁶)` |
| **2** | 27 | `intermediate_size = 5111·√(layer_size / 10⁶)` |
| **3** | 31 | `intermediate_size = 6856·√(layer_size / 10⁶)` |

MSE (piecewise intermediate fit): **0.0015**. Higher line → larger `intermediate_size` → smaller theoretical `hidden_size` at the same `layer_size`.

`layer_size` values are the distinct per-layer budgets from Phase 1 (below / above branches), shown as **M** / **B**. `intermediate_size` = `num_experts × expert_size`, shown as **K**.

## 2. hidden_size (from intermediate fits)

| layer_size | intermediate_size 1 | theoretical hidden_size 1 | intermediate_size 2 | theoretical hidden_size 2 | intermediate_size 3 | theoretical hidden_size 3 |
|------------|--------------------:|--------------------------:|--------------------:|--------------------------:|--------------------:|--------------------------:|
| 25M | 18.44K | 452 | 25.55K | 326 | 34.28K | 243 |
| 29.41M | 20.01K | 490 | 27.72K | 354 | 37.18K | 264 |
| 35.71M | 22.04K | 540 | 30.54K | 390 | 40.97K | 291 |
| 41.67M | 23.81K | 583 | 32.99K | 421 | 44.26K | 314 |
| 45.45M | 24.87K | 609 | 34.46K | 440 | 46.22K | 328 |
| 62.5M | 29.16K | 714 | 40.4K | 516 | 54.21K | 384 |
| 64.52M | 29.63K | 726 | 41.05K | 524 | 55.07K | 391 |
| 100M | 36.89K | 904 | 51.11K | 652 | 68.56K | 486 |
| 105.26M | 37.85K | 927 | 52.44K | 669 | 70.34K | 499 |
| 166.67M | 47.62K | 1167 | 65.98K | 842 | 88.52K | 628 |
| 177.78M | 49.19K | 1205 | 68.15K | 870 | 91.42K | 648 |
| 275.86M | 61.27K | 1501 | 84.89K | 1083 | 113.88K | 807 |
| 307.69M | 64.71K | 1585 | 89.65K | 1144 | 120.27K | 853 |
| 484.85M | 81.23K | 1990 | 112.54K | 1436 | 150.97K | 1070 |
| 542.37M | 85.91K | 2104 | 119.03K | 1519 | 159.68K | 1132 |
| 842.11M | 107.05K | 2622 | 148.31K | 1893 | 198.97K | 1411 |
| 969.7M | 114.87K | 2814 | 159.15K | 2031 | 213.51K | 1514 |
| 1.41B | 138.52K | 3393 | 191.91K | 2449 | 257.46K | 1826 |
| 1.52B | 143.82K | 3523 | 199.26K | 2543 | 267.31K | 1895 |
| 1.75B | 154.32K | 3780 | 213.8K | 2728 | 286.83K | 2034 |
| 2.22B | 173.81K | 4257 | 240.81K | 3073 | 323.06K | 2291 |
| 2.78B | 194.5K | 4764 | 269.47K | 3439 | 361.51K | 2563 |
| 5.75B | 279.73K | 6852 | 387.55K | 4946 | 519.92K | 3686 |
| 9.09B | 351.71K | 8615 | 487.28K | 6218 | 653.71K | 4635 |
| 10.64B | 380.52K | 9321 | 527.19K | 6728 | 707.25K | 5015 |
| 16.95B | 480.27K | 11764 | 665.39K | 8491 | 892.66K | 6329 |

```bash
python explorations/plot_intermediate_vs_layer_size_fits.py
python explorations/plot_theoretical_hidden_vs_layer_size.py
```
---

# Choice 3: selecting num_experts and expert_size

Fits from [Model Architecture Experiments - Survey.csv](Model%20Architecture%20Experiments%20-%20Survey.csv), same method as [num_experts vs intermediate_size (k=2, linear)](figures/num_experts_vs_intermediate_size_k2_linear.png) and [theoretical expert_size vs intermediate_size](figures/theoretical_expert_size_vs_intermediate_size.png):

1. **k=2** k-means on survey models: **`num_experts = a·intermediate_size + b`** through **`(intermediate_size = 1024, num_experts = 1)`** (`intermediate_size` = `num_experts × expert_size`).
2. **Theoretical expert_size** per fit line: **`expert_size = intermediate_size / num_experts(fit)`**.
3. **Two axis outliers** excluded before num_experts fitting (full **158**-model survey otherwise): highest **`intermediate_size`**, then highest **`num_experts`** on the remainder — **156** models used.
4. Row **`intermediate_size`** values are the distinct Phase 2 predictions (`k=3` sqrt intermediate vs `layer_size`, through origin) at each Phase 1 `layer_size`, shown as **K**. `—` when a fit predicts `num_experts ≤ 0`.

### Equations

**Intermediate_size** (Phase 2 reference; `layer_size_m` = layer_size / 10⁶):

| Line | Points | Equation |
|------|--------|----------|
| **1** | 92 | `intermediate_size = 3689·√(layer_size / 10⁶)` |
| **2** | 27 | `intermediate_size = 5111·√(layer_size / 10⁶)` |
| **3** | 31 | `intermediate_size = 6856·√(layer_size / 10⁶)` |

**num_experts** (k=2 linear vs `intermediate_size`):

| Line | Points | Equation |
|------|--------|----------|
| **1** | 29 | `num_experts = 3.642e-04·intermediate_size + 0.63` |
| **2** | 127 | `num_experts = 5.921e-04·intermediate_size + 0.39` |

MSE (num_experts fit): **7133.13**. All lines pass through **`(1024, 1)`**. `—` when a line predicts `num_experts ≤ 0`.

##  and expert_size (from num_experts fits)

| intermediate_size | num_experts fit 1 | expert_size 1 | num_experts fit 2 | expert_size 2 |
|------------------:|-------------------:|--------------:|-------------------:|--------------:|
| 18.45K | 7 | 2511 | 11 | 1630 |
| 20.01K | 8 | 2528 | 12 | 1635 |
| 22.04K | 9 | 2547 | 13 | 1639 |
| 23.81K | 9 | 2561 | 14 | 1643 |
| 24.87K | 10 | 2568 | 15 | 1645 |
| 25.55K | 10 | 2572 | 16 | 1646 |
| 27.72K | 11 | 2585 | 17 | 1649 |
| 29.16K | 11 | 2593 | 18 | 1651 |
| 29.63K | 11 | 2595 | 18 | 1652 |
| 30.54K | 12 | 2599 | 18 | 1653 |
| 32.99K | 13 | 2610 | 20 | 1655 |
| 34.28K | 13 | 2614 | 21 | 1657 |
| 34.46K | 13 | 2615 | 21 | 1657 |
| 36.89K | 14 | 2623 | 22 | 1659 |
| 37.18K | 14 | 2624 | 22 | 1659 |
| 37.85K | 14 | 2626 | 23 | 1660 |
| 40.41K | 15 | 2634 | 24 | 1661 |
| 40.97K | 16 | 2635 | 25 | 1662 |
| 41.05K | 16 | 2635 | 25 | 1662 |
| 44.26K | 17 | 2643 | 27 | 1664 |
| 46.22K | 17 | 2647 | 28 | 1665 |
| 47.62K | 18 | 2650 | 29 | 1666 |
| 49.19K | 19 | 2653 | 30 | 1666 |
| 51.11K | 19 | 2656 | 31 | 1667 |
| 52.44K | 20 | 2658 | 31 | 1668 |
| 54.2K | 20 | 2661 | 32 | 1668 |
| 55.07K | 21 | 2662 | 33 | 1669 |
| 61.27K | 23 | 2671 | 37 | 1671 |
| 64.71K | 24 | 2675 | 39 | 1672 |
| 65.98K | 25 | 2676 | 39 | 1672 |
| 68.14K | 25 | 2678 | 41 | 1673 |
| 68.56K | 26 | 2678 | 41 | 1673 |
| 70.34K | 26 | 2680 | 42 | 1673 |
| 81.23K | 30 | 2689 | 48 | 1675 |
| 84.89K | 32 | 2691 | 51 | 1676 |
| 85.91K | 32 | 2692 | 51 | 1676 |
| 88.52K | 33 | 2693 | 53 | 1676 |
| 89.65K | 33 | 2694 | 53 | 1676 |
| 91.42K | 34 | 2695 | 55 | 1677 |
| 107.05K | 40 | 2702 | 64 | 1678 |
| 112.54K | 42 | 2704 | 67 | 1679 |
| 113.88K | 42 | 2705 | 68 | 1679 |
| 114.87K | 42 | 2705 | 68 | 1679 |
| 119.03K | 44 | 2707 | 71 | 1679 |
| 120.27K | 44 | 2707 | 72 | 1680 |
| 138.52K | 51 | 2712 | 82 | 1681 |
| 143.82K | 53 | 2713 | 86 | 1681 |
| 148.31K | 55 | 2714 | 88 | 1681 |
| 150.97K | 56 | 2715 | 90 | 1681 |
| 154.32K | 57 | 2715 | 92 | 1682 |
| 159.15K | 59 | 2716 | 95 | 1682 |
| 159.68K | 59 | 2716 | 95 | 1682 |
| 173.81K | 64 | 2719 | 103 | 1682 |
| 191.91K | 71 | 2721 | 114 | 1683 |
| 194.5K | 71 | 2722 | 116 | 1683 |
| 198.97K | 73 | 2722 | 118 | 1683 |
| 199.26K | 73 | 2722 | 118 | 1683 |
| 213.51K | 78 | 2724 | 127 | 1684 |
| 213.8K | 78 | 2724 | 127 | 1684 |
| 240.81K | 88 | 2726 | 143 | 1684 |
| 257.46K | 94 | 2727 | 153 | 1684 |
| 267.31K | 98 | 2728 | 159 | 1685 |
| 269.47K | 99 | 2728 | 160 | 1685 |
| 279.73K | 103 | 2729 | 166 | 1685 |
| 286.83K | 105 | 2729 | 170 | 1685 |
| 323.06K | 118 | 2731 | 192 | 1685 |
| 351.71K | 129 | 2732 | 209 | 1686 |
| 361.51K | 132 | 2733 | 214 | 1686 |
| 380.52K | 139 | 2733 | 226 | 1686 |
| 387.55K | 142 | 2734 | 230 | 1686 |
| 480.27K | 176 | 2736 | 285 | 1686 |
| 487.28K | 178 | 2736 | 289 | 1687 |
| 519.92K | 190 | 2737 | 308 | 1687 |
| 527.19K | 193 | 2737 | 313 | 1687 |
| 653.71K | 239 | 2739 | 387 | 1687 |
| 665.39K | 243 | 2739 | 394 | 1687 |
| 707.25K | 258 | 2739 | 419 | 1687 |
| 892.66K | 326 | 2740 | 529 | 1688 |

```bash
python explorations/generate_phase3_table.py
python explorations/plot_num_experts_vs_intermediate_size_fits.py
python explorations/plot_theoretical_expert_size_vs_intermediate_size.py
```
