"""Quick demo of the stats helpers on synthetic data.

What this file is for:
- Show minimal usage of descriptive stats, hypothesis tests, ANOVA,
  effect sizes, simple regression, and quick plots.

Run directly from the repo root:
    python examples/demo.py

Note: This demo imports from the `src` package in this repo.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from utils import (
    describe_series,
    one_sample_ttest,
    independent_ttest,
    paired_ttest,
    one_way_anova,
    cohen_d_independent,
    ols_formula,
    qqplot,
    hist_kde,
)


def main():
    # Reproducible random generator
    rng = np.random.default_rng(42)
    # Three normal samples with different means/SDs
    a = rng.normal(0.0, 1.0, size=100)
    b = rng.normal(0.3, 1.1, size=120)
    c = rng.normal(-0.2, 0.9, size=90)

    print("Descriptives for A:\n", describe_series(a), "\n", sep="")

    # A few common tests and an effect size
    print("One-sample t-test (mu=0):", one_sample_ttest(a, mu=0.0))
    print("Two-sample t-test (A vs B):", independent_ttest(a, b, equal_var=False))
    print("Paired t-test (A vs A+noise):", paired_ttest(a, a + rng.normal(0, 0.2, size=a.size)))
    print("One-way ANOVA (A, B, C):", one_way_anova(a, b, c))
    print("Cohen's d (A vs B):", cohen_d_independent(a, b))

    # Simple linear regression example (y ~ x)
    df = pd.DataFrame({"y": b, "x": np.linspace(0, 10, num=b.size) + rng.normal(0, 0.5, size=b.size)})
    ols_res = ols_formula("y ~ x", df)
    # R^2 and tidy parameters available in the result dict
    print("OLS R^2:", ols_res["r2"])  # tidy params in ols_res["params"]

    try:
        import matplotlib.pyplot as plt
        # Quick diagnostic plots
        qqplot(a)
        hist_kde(a)
        plt.show()
    except Exception as e:
        print("Plotting skipped:", e)


if __name__ == "__main__":
    main()
