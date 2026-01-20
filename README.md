# LearningLab Stats

[![MIT License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Python 3.13+](https://img.shields.io/badge/python-3.13%2B-blue.svg)](#)
[![Built with NumPy | Pandas | SciPy | Statsmodels | Seaborn | Matplotlib](https://img.shields.io/badge/built%20with-NumPy%20%7C%20Pandas%20%7C%20SciPy%20%7C%20Statsmodels%20%7C%20Seaborn%20%7C%20Matplotlib-orange.svg)](#)

A friendly, teaching-oriented toolkit and notebooks for common statistical analyses in Python.

## Contents

- Code: lightweight helpers in [src](src) for:
	- Descriptives: compact summaries for series/dataframes
	- Inference: t-tests, ANOVA, chi-square, nonparametric tests, proportions
	- Effect sizes: Cohen's d, Hedges' g, eta/omega squared, Cramér's V, Cliff's delta
	- Regression: quick OLS/Logistic fits with tidy outputs
	- Power: power and sample size for t-tests, proportions, ANOVA
	- Visualization: QQ plots, histograms/KDE, box/violin
- Notebooks: see [notebooks/BasicsOfStatistics.ipynb](notebooks/BasicsOfStatistics.ipynb) and [notebooks/StatisticalInference.ipynb](notebooks/StatisticalInference.ipynb)
- Demo: a quick script in [examples/demo.py](examples/demo.py)

## Getting Started

### Environment

This repo includes a conda-style environment file at [requirements.txt](requirements.txt).

Create and activate the environment:

```bash
conda env create -f requirements.txt
conda activate statsenv
```

Alternatively, with pip (for minimal usage):

```bash
python -m pip install numpy pandas scipy statsmodels matplotlib seaborn
```

### Run the Demo

```bash
python examples/demo.py
```

### Usage in Code

Import helpers directly from the local `src` package:

```python
from src import describe_series, independent_ttest, ols_formula

# Example data
import numpy as np, pandas as pd
rng = np.random.default_rng(0)
a = rng.normal(0, 1, 100)
b = rng.normal(0.3, 1.1, 120)

# Descriptives
print(describe_series(a))

# Tests
print(independent_ttest(a, b, equal_var=False))

# Regression
df = pd.DataFrame({"y": b, "x": np.linspace(0, 10, len(b))})
res = ols_formula("y ~ x", df)
print(res["r2"])  # tidy params in res["params"]
```

## Module Overview

- [src/descriptives.py](src/descriptives.py): `describe_series()`, `describe_dataframe()`
- [src/inference.py](src/inference.py): `one_sample_ttest()`, `independent_ttest()`, `paired_ttest()`, `one_way_anova()`, `chi2_independence()`, `mann_whitney_u()`, `wilcoxon_signed_rank()`, `proportion_ztest()`, `proportion_confint()`
- [src/effect_sizes.py](src/effect_sizes.py): `cohen_d_independent()`, `cohen_d_paired()`, `hedges_g()`, `eta_squared_anova()`, `omega_squared_anova()`, `cramer_v()`, `cliff_delta()`
- [src/regression.py](src/regression.py): `ols_formula()`, `logistic_formula()`
- [src/power.py](src/power.py): `ttest_power()`, `two_sample_ttest_power()`, `proportion_diff_power()`, `anova_power()`
- [src/visualization.py](src/visualization.py): `qqplot()`, `hist_kde()`, `box_violin()`

## Notebooks

Open the notebooks with JupyterLab:

```bash
jupyter lab notebooks/
```

## Contributing

Pull requests are welcome for fixes and improvements. For larger changes, please open an issue first to discuss what you’d like to add.

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE).

