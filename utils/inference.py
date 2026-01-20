"""Classical statistical inference helpers.

Includes convenient wrappers for:
- t-tests (one-sample, independent/pooled or Welch, and paired)
- one-way ANOVA
- chi-square test of independence
- nonparametric tests (Mann–Whitney U, Wilcoxon signed-rank)
- proportion tests and confidence intervals

Designed for readability in teaching and quick analyses.
"""

from __future__ import annotations

from typing import Iterable, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats
try:
    from statsmodels.stats import proportion as sm_prop
except Exception:
    sm_prop = None  # type: ignore


Number = Union[int, float, np.number]


def _to_array(x: Iterable[Number]) -> np.ndarray:
    """Convert an iterable to float array and drop NaNs."""
    a = np.asarray(list(x), dtype=float)
    return a[~np.isnan(a)]


def _ci_from_stat(se: float, estimate: float, df: Optional[int], conf_level: float, dist: str = "t") -> Tuple[float, float]:
    """Compute CI from standard error and estimate using t or normal crit."""
    alpha = 1 - conf_level
    if dist == "t":
        crit = stats.t.ppf(1 - alpha / 2, df) if df is not None else np.nan
    else:
        crit = stats.norm.ppf(1 - alpha / 2)
    me = crit * se
    return estimate - me, estimate + me


def one_sample_ttest(
    sample: Iterable[Number],
    mu: float = 0.0,
    alternative: str = "two-sided",
    conf_level: float = 0.95,
) -> dict:
    x = _to_array(sample)
    res = stats.ttest_1samp(x, popmean=mu, alternative=alternative)
    n = x.size
    df = n - 1
    mean_diff = x.mean() - mu
    se = x.std(ddof=1) / np.sqrt(n)
    ci = _ci_from_stat(se, x.mean(), df, conf_level, dist="t")
    return {"t": float(res.statistic), "p": float(res.pvalue), "df": int(df), "mean": float(x.mean()), "ci": ci, "n": int(n), "mean_diff": float(mean_diff)}


def independent_ttest(
    a: Iterable[Number],
    b: Iterable[Number],
    equal_var: bool = False,
    alternative: str = "two-sided",
    conf_level: float = 0.95,
) -> dict:
    x, y = _to_array(a), _to_array(b)
    res = stats.ttest_ind(x, y, equal_var=equal_var, alternative=alternative)
    na, nb = x.size, y.size
    if equal_var:
        # Student's t-test: pooled variance estimate
        df = na + nb - 2
        sp2 = (((na - 1) * x.var(ddof=1)) + ((nb - 1) * y.var(ddof=1))) / df
        se = np.sqrt(sp2 * (1 / na + 1 / nb))
    else:
        # Welch's t-test: no equal variance assumption, Satterthwaite df
        vx, vy = x.var(ddof=1) / na, y.var(ddof=1) / nb
        se = np.sqrt(vx + vy)
        df = (vx + vy) ** 2 / (vx**2 / (na - 1) + vy**2 / (nb - 1))
    diff = x.mean() - y.mean()
    ci = _ci_from_stat(se, diff, int(df), conf_level, dist="t")
    return {"t": float(res.statistic), "p": float(res.pvalue), "df": float(df), "mean_diff": float(diff), "ci": ci, "n1": int(na), "n2": int(nb)}


def paired_ttest(
    a: Iterable[Number],
    b: Iterable[Number],
    alternative: str = "two-sided",
    conf_level: float = 0.95,
) -> dict:
    x, y = _to_array(a), _to_array(b)
    n = min(x.size, y.size)
    x, y = x[:n], y[:n]
    d = x - y
    res = stats.ttest_rel(x, y, alternative=alternative)
    df = n - 1
    se = d.std(ddof=1) / np.sqrt(n)
    ci = _ci_from_stat(se, d.mean(), df, conf_level, dist="t")
    return {"t": float(res.statistic), "p": float(res.pvalue), "df": int(df), "mean_diff": float(d.mean()), "ci": ci, "n": int(n)}


def one_way_anova(*groups: Iterable[Number]) -> dict:
    arrays = [ _to_array(g) for g in groups ]
    res = stats.f_oneway(*arrays)
    k = len(arrays)
    n = sum(a.size for a in arrays)
    df_between = k - 1
    df_within = n - k
    return {"F": float(res.statistic), "p": float(res.pvalue), "df_between": int(df_between), "df_within": int(df_within)}


def chi2_independence(table: Union[pd.DataFrame, np.ndarray]) -> dict:
    if isinstance(table, pd.DataFrame):
        chi2, p, dof, expected = stats.chi2_contingency(table.values)
        index = table.index
        columns = table.columns
    else:
        chi2, p, dof, expected = stats.chi2_contingency(np.asarray(table))
        index = None
        columns = None
    return {"chi2": float(chi2), "p": float(p), "df": int(dof), "expected": pd.DataFrame(expected, index=index, columns=columns)}


def mann_whitney_u(a: Iterable[Number], b: Iterable[Number], alternative: str = "two-sided") -> dict:
    x, y = _to_array(a), _to_array(b)
    res = stats.mannwhitneyu(x, y, alternative=alternative)
    return {"U": float(res.statistic), "p": float(res.pvalue), "n1": int(x.size), "n2": int(y.size)}


def wilcoxon_signed_rank(a: Iterable[Number], b: Iterable[Number], alternative: str = "two-sided") -> dict:
    x, y = _to_array(a), _to_array(b)
    n = min(x.size, y.size)
    res = stats.wilcoxon(x[:n], y[:n], alternative=alternative)
    return {"W": float(res.statistic), "p": float(res.pvalue), "n": int(n)}


def proportion_ztest(
    count: Union[int, Iterable[int]],
    nobs: Union[int, Iterable[int]],
    value: Optional[float] = None,
    alternative: str = "two-sided",
) -> dict:
    if sm_prop is None:
        raise ImportError("statsmodels is required for proportion z-tests (pip/conda install statsmodels)")
    stat, p = sm_prop.proportions_ztest(count, nobs, value=value, alternative=alternative)
    return {"z": float(stat), "p": float(p)}


def proportion_confint(
    count: int,
    nobs: int,
    alpha: float = 0.05,
    method: str = "wilson",
) -> Tuple[float, float]:
    if sm_prop is None:
        raise ImportError("statsmodels is required for proportion confidence intervals (pip/conda install statsmodels)")
    return sm_prop.proportion_confint(count, nobs, alpha=alpha, method=method)
