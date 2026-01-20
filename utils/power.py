"""Statistical power and sample size helpers (via statsmodels).

Functions compute either achieved power given sample size, or the
required sample size given desired power. Results are rounded up for n.
"""

from __future__ import annotations

from typing import Optional

from math import ceil

try:
    from statsmodels.stats.power import (
        TTestPower,
        TTestIndPower,
        NormalIndPower,
        FTestAnovaPower,
    )
except Exception:  # Defer hard failure to function calls with a clear message
    TTestPower = None  # type: ignore
    TTestIndPower = None  # type: ignore
    NormalIndPower = None  # type: ignore
    FTestAnovaPower = None  # type: ignore


def ttest_power(effect_size: float, nobs: Optional[int] = None, alpha: float = 0.05, power: Optional[float] = None, alternative: str = "two-sided") -> float:
    """One-sample or paired t-test power or required n.

    Provide either nobs to compute power, or power to compute required nobs.
    """
    if TTestPower is None:
        raise ImportError("statsmodels is required for power analysis (pip/conda install statsmodels)")
    tt = TTestPower()
    if nobs is not None:
        return float(tt.power(effect_size=effect_size, nobs=nobs, alpha=alpha, alternative=alternative))
    if power is not None:
        n = tt.solve_power(effect_size=effect_size, power=power, alpha=alpha, alternative=alternative)
        return float(ceil(n))
    raise ValueError("Provide either nobs to compute power or power to compute required nobs.")


def two_sample_ttest_power(effect_size: float, nobs1: Optional[int] = None, alpha: float = 0.05, power: Optional[float] = None, ratio: float = 1.0, alternative: str = "two-sided") -> float:
    """Two-sample t-test power or required n per group (rounded up)."""
    if TTestIndPower is None:
        raise ImportError("statsmodels is required for power analysis (pip/conda install statsmodels)")
    tt = TTestIndPower()
    if nobs1 is not None:
        return float(tt.power(effect_size=effect_size, nobs1=nobs1, alpha=alpha, ratio=ratio, alternative=alternative))
    if power is not None:
        n1 = tt.solve_power(effect_size=effect_size, power=power, alpha=alpha, ratio=ratio, alternative=alternative)
        return float(ceil(n1))
    raise ValueError("Provide either nobs1 to compute power or power to compute required n per group.")


def proportion_diff_power(effect_size: float, nobs1: Optional[int] = None, alpha: float = 0.05, power: Optional[float] = None, ratio: float = 1.0, alternative: str = "two-sided") -> float:
    """Power or required n for difference in two proportions (normal approximation)."""
    if NormalIndPower is None:
        raise ImportError("statsmodels is required for power analysis (pip/conda install statsmodels)")
    pw = NormalIndPower()
    if nobs1 is not None:
        return float(pw.power(effect_size=effect_size, nobs1=nobs1, alpha=alpha, ratio=ratio, alternative=alternative))
    if power is not None:
        n1 = pw.solve_power(effect_size=effect_size, power=power, alpha=alpha, ratio=ratio, alternative=alternative)
        return float(ceil(n1))
    raise ValueError("Provide either nobs1 to compute power or power to compute required n per group.")


def anova_power(effect_size: float, k_groups: int, nobs: Optional[int] = None, alpha: float = 0.05, power: Optional[float] = None) -> float:
    """One-way ANOVA (fixed effects) power or required total n.

    effect_size is Cohen's f.
    """
    if FTestAnovaPower is None:
        raise ImportError("statsmodels is required for power analysis (pip/conda install statsmodels)")
    ft = FTestAnovaPower()
    if nobs is not None:
        return float(ft.power(effect_size=effect_size, k_groups=k_groups, nobs=nobs, alpha=alpha))
    if power is not None:
        n = ft.solve_power(effect_size=effect_size, k_groups=k_groups, power=power, alpha=alpha)
        return float(ceil(n))
    raise ValueError("Provide either nobs to compute power or power to compute required total n.")
