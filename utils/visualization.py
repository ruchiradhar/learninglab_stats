"""Quick visualization helpers for distributions and normality checks.

Includes QQ plots (via statsmodels), histogram with optional KDE (seaborn),
and box/violin plots for grouped comparisons.
"""

from __future__ import annotations

from typing import Iterable, Optional, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
try:
    import statsmodels.api as sm
except Exception:
    sm = None  # type: ignore


def _to_series(x: Union[pd.Series, Iterable[float]]) -> pd.Series:
    """Normalize input to a float Series."""
    return x if isinstance(x, pd.Series) else pd.Series(list(x), dtype=float)


def qqplot(x: Union[pd.Series, Iterable[float]], dist: str = "norm", ax=None):
    s = _to_series(x).dropna()
    if ax is None:
        fig, ax = plt.subplots()
    if sm is None:
        raise ImportError("statsmodels is required for QQ plots (pip/conda install statsmodels)")
    sm.ProbPlot(s, dist=dist).qqplot(line="45", ax=ax)
    ax.set_title("QQ Plot")
    return ax


def hist_kde(x: Union[pd.Series, Iterable[float]], bins: int = 30, kde: bool = True, ax=None):
    s = _to_series(x).dropna()
    if ax is None:
        fig, ax = plt.subplots()
    sns.histplot(s, bins=bins, kde=kde, ax=ax)
    ax.set_title("Distribution")
    return ax


def box_violin(df: pd.DataFrame, x: Optional[str] = None, y: Optional[str] = None, kind: str = "box", ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    if kind == "box":
        sns.boxplot(data=df, x=x, y=y, ax=ax)
    else:
        sns.violinplot(data=df, x=x, y=y, inner="quartile", ax=ax)
    ax.set_title("Box/Violin Plot")
    return ax
