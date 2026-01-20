"""Effect size computations for common tests.

Includes Cohen's d (independent/paired), Hedges' g, ANOVA eta/omega squared,
Cramer's V, and Cliff's delta.
"""

from __future__ import annotations

from typing import Iterable, Tuple, Union

import numpy as np
import pandas as pd


Number = Union[int, float, np.number]


def _to_array(x: Iterable[Number]) -> np.ndarray:
    """Convert input to float array and drop NaNs."""
    a = np.asarray(list(x), dtype=float)
    return a[~np.isnan(a)]


def cohen_d_independent(a: Iterable[Number], b: Iterable[Number], equal_var: bool = False) -> float:
    x, y = _to_array(a), _to_array(b)
    if equal_var:
        # Pooled SD (Student's t-test)
        nx, ny = x.size, y.size
        df = nx + ny - 2
        sp2 = (((nx - 1) * x.var(ddof=1)) + ((ny - 1) * y.var(ddof=1))) / df
        s = np.sqrt(sp2)
    else:
        # Average variance (Welch-like)
        s = np.sqrt((x.var(ddof=1) + y.var(ddof=1)) / 2)
    return float((x.mean() - y.mean()) / s)


def cohen_d_paired(a: Iterable[Number], b: Iterable[Number]) -> float:
    x, y = _to_array(a), _to_array(b)
    n = min(x.size, y.size)
    d = x[:n] - y[:n]
    # Standardized mean difference of paired deltas
    return float(d.mean() / d.std(ddof=1))


def hedges_g(d: float, n1: int, n2: int) -> float:
    # Small-sample correction factor J
    j = 1 - 3 / (4 * (n1 + n2) - 9)
    return float(j * d)


def eta_squared_anova(F: float, df_between: int, df_within: int) -> float:
    return float((F * df_between) / (F * df_between + df_within))


def omega_squared_anova(F: float, df_between: int, df_within: int) -> float:
    return float((F * df_between - 1) / (F * df_between + df_within + 1))


def cramer_v(table: Union[pd.DataFrame, np.ndarray]) -> float:
    if isinstance(table, pd.DataFrame):
        observed = table.values
    else:
        observed = np.asarray(table)
    n = observed.sum()
        # Compute chi-square from observed and expected counts without external call
        chi2 = (((observed - observed.sum(axis=1, keepdims=True) * observed.sum(axis=0, keepdims=True) / n) ** 2)
            / (observed.sum(axis=1, keepdims=True) * observed.sum(axis=0, keepdims=True) / n)).sum()
    r, k = observed.shape
    return float(np.sqrt(chi2 / (n * (min(r, k) - 1))))


def cliff_delta(a: Iterable[Number], b: Iterable[Number]) -> Tuple[float, float, float]:
    x, y = _to_array(a), _to_array(b)
    m, n = x.size, y.size
    # Efficient pairwise comparison using broadcasting for modest sizes
    if m * n <= 2_000_000:
        comp = np.subtract.outer(x, y)
        n_greater = np.sum(comp > 0)
        n_less = np.sum(comp < 0)
    else:
        n_greater = 0
        n_less = 0
        y_sorted = np.sort(y)
        for xi in x:
            # number of y less than xi
            less = np.searchsorted(y_sorted, xi, side="left")
            # number of y greater than xi
            greater = n - np.searchsorted(y_sorted, xi, side="right")
            n_less += less
            n_greater += greater
    delta = (n_greater - n_less) / (m * n)
    return float(delta), float(n_greater), float(n_less)
