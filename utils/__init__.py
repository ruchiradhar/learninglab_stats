"""Lightweight utilities for common statistical analyses.

Modules:
- descriptives: Summary statistics for series and dataframes
- inference: Hypothesis tests and confidence intervals
- effect_sizes: Effect size computations for common tests
- regression: Simple wrappers around statsmodels OLS/Logit
- power: Power and sample size calculations
- visualization: Quick helper plots for distributions and QQ
"""

from .descriptives import describe_series, describe_dataframe
from .inference import (
    one_sample_ttest,
    independent_ttest,
    paired_ttest,
    one_way_anova,
    chi2_independence,
    mann_whitney_u,
    wilcoxon_signed_rank,
    proportion_ztest,
    proportion_confint,
)
from .effect_sizes import (
    cohen_d_independent,
    cohen_d_paired,
    hedges_g,
    eta_squared_anova,
    omega_squared_anova,
    cramer_v,
    cliff_delta,
)
from .regression import ols_formula, logistic_formula
from .power import (
    ttest_power,
    two_sample_ttest_power,
    proportion_diff_power,
    anova_power,
)
from .visualization import qqplot, hist_kde, box_violin

__all__ = [
    "describe_series",
    "describe_dataframe",
    "one_sample_ttest",
    "independent_ttest",
    "paired_ttest",
    "one_way_anova",
    "chi2_independence",
    "mann_whitney_u",
    "wilcoxon_signed_rank",
    "proportion_ztest",
    "proportion_confint",
    "cohen_d_independent",
    "cohen_d_paired",
    "hedges_g",
    "eta_squared_anova",
    "omega_squared_anova",
    "cramer_v",
    "cliff_delta",
    "ols_formula",
    "logistic_formula",
    "ttest_power",
    "two_sample_ttest_power",
    "proportion_diff_power",
    "anova_power",
    "qqplot",
    "hist_kde",
    "box_violin",
]
