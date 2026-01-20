"""Visualization utilities for quick EDA.

Overview:
- Purpose: simplify common statistical visualizations .
- Dependencies: seaborn, matplotlib, numpy, pandas; statsmodels.

Exports:
- `set_theme(style, palette, context, font_scale)`: Set seaborn/matplotlib theme.
- `qqplot(x, dist='norm', ax=None)`: Normal Q–Q plot via statsmodels.
- `hist_kde(x, bins=30, kde=True, ax=None)`: Histogram with optional KDE.
- `distplot(data, x, hue=None, bins=None, kde=True, rug=False, fill=True, alpha=0.85, ax=None)`: Flexible distribution plot.
- `box_violin(df, x=None, y=None, kind='box', ax=None)`: Box or violin plot.
- `ecdf(data, x, hue=None, complementary=False, ax=None)`: Empirical CDF (or 1−CDF).
- `corr_heatmap(df, method='pearson', annot=False, cmap='coolwarm', mask_upper=True, ax=None)`: Correlation heatmap.
- `pairgrid(data, vars=None, hue=None, corner=True, diag_kind='kde', kind='scatter')`: Pairwise relationships.
- `scatter_fit(data, x, y, hue=None, ci=95, ax=None)`: Scatter with fitted line and CI.
- `bar_ci(data, x, y, estimator=np.mean, ci=95, capsize=0.1, ax=None)`: Bar chart with CIs.
- `line_ci(data, x, y, hue=None, estimator=np.mean, ci=95, marker='o', ax=None)`: Line plot with CIs.
- `ridgeline(data, x, row, fill=True, height=1.2, aspect=3.5, palette='muted')`: Faceted ridge KDEs.
"""

from __future__ import annotations

from typing import Iterable, Optional, Union, Literal, Any

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


def set_theme(
    style: Literal["white", "dark", "whitegrid", "darkgrid", "ticks"] = "whitegrid",
    palette: str = "deep",
    context: Literal["paper", "notebook", "talk", "poster"] = "notebook",
    font_scale: float = 1.1,
):
    """Set a consistent seaborn/matplotlib theme for polished visuals."""
    sns.set_theme(style=style, palette=palette, context=context, font_scale=font_scale)
    plt.rcParams.update({
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.autolayout": True,
    })


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


def distplot(
    data: pd.DataFrame,
    x: str,
    hue: Optional[str] = None,
    bins: Optional[int] = None,
    kde: bool = True,
    rug: bool = False,
    fill: bool = True,
    alpha: float = 0.85,
    ax=None,
):
    """Beautiful histogram/KDE with optional hue and rug."""
    if ax is None:
        fig, ax = plt.subplots()
    sns.histplot(data=data, x=x, hue=hue, bins=bins, stat="density", kde=False, alpha=alpha, ax=ax)
    if kde:
        sns.kdeplot(data=data, x=x, hue=hue, fill=fill, common_norm=False, alpha=0.25 if fill else 1.0, ax=ax)
    if rug:
        sns.rugplot(data=data, x=x, hue=hue, height=0.03, ax=ax)
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


def ecdf(data: pd.DataFrame, x: str, hue: Optional[str] = None, complementary: bool = False, ax=None):
    """Empirical cumulative distribution function (optionally complementary)."""
    if ax is None:
        fig, ax = plt.subplots()
    sns.ecdfplot(data=data, x=x, hue=hue, complementary=complementary, ax=ax)
    ax.set_title("ECDF" + (" (1-CDF)" if complementary else ""))
    return ax


def corr_heatmap(
    df: pd.DataFrame,
    method: Literal["pearson", "kendall", "spearman"] = "pearson",
    annot: bool = False,
    cmap: str = "coolwarm",
    mask_upper: bool = True,
    ax=None,
):
    """Correlation heatmap with optional upper-triangle masking."""
    corr = df.corr(numeric_only=True, method=method)
    mask = None
    if mask_upper:
        mask = np.triu(np.ones_like(corr, dtype=bool))
    if ax is None:
        fig, ax = plt.subplots()
    sns.heatmap(corr, mask=mask, cmap=cmap, annot=annot, fmt=".2f", center=0, square=True, cbar_kws={"shrink": 0.8}, linewidths=0.5, ax=ax)
    ax.set_title(f"Correlation ({method.title()})")
    return ax


def pairgrid(
    data: pd.DataFrame,
    vars: Optional[list[str]] = None,
    hue: Optional[str] = None,
    corner: bool = True,
    diag_kind: Literal["auto", "hist", "kde"] = "kde",
    kind: Literal["scatter", "kde", "hist", "reg"] = "scatter",
):
    """Pairwise relationships with Seaborn PairGrid."""
    g = sns.pairplot(data=data, vars=vars, hue=hue, corner=corner, diag_kind=diag_kind, kind=kind, plot_kws={"alpha": 0.7})
    return g


def scatter_fit(
    data: pd.DataFrame,
    x: str,
    y: str,
    hue: Optional[str] = None,
    ci: Optional[int] = 95,
    ax=None,
):
    """Scatter with fitted line and confidence interval.

    Uses seaborn.regplot for single series; falls back to lmplot when hue is provided.
    """
    if hue is None:
        if ax is None:
            fig, ax = plt.subplots()
        sns.regplot(data=data, x=x, y=y, ci=ci, scatter_kws={"alpha": 0.7}, line_kws={"lw": 2}, ax=ax)
        ax.set_title("Scatter with Fit")
        return ax
    else:
        g = sns.lmplot(data=data, x=x, y=y, hue=hue, ci=ci, scatter_kws={"alpha": 0.6}, line_kws={"lw": 2})
        g.fig.suptitle("Scatter with Fit", y=1.02)
        return g.ax


def bar_ci(data: pd.DataFrame, x: str, y: str, estimator = np.mean, ci: Optional[int] = 95, capsize: float = 0.1, ax=None):
    """Bar chart with confidence intervals for an estimator (default mean)."""
    if ax is None:
        fig, ax = plt.subplots()
    sns.barplot(data=data, x=x, y=y, estimator=estimator, ci=ci, capsize=capsize, errcolor="0.2", ax=ax)
    ax.set_title("Mean with CI")
    return ax


def line_ci(data: pd.DataFrame, x: str, y: str, hue: Optional[str] = None, estimator = np.mean, ci: Optional[int] = 95, marker: str = "o", ax=None):
    """Line plot of an estimator over x with confidence interval shading."""
    if ax is None:
        fig, ax = plt.subplots()
    sns.lineplot(data=data, x=x, y=y, hue=hue, estimator=estimator, ci=ci, marker=marker, ax=ax)
    ax.set_title("Trend with CI")
    return ax


def ridgeline(data: pd.DataFrame, x: str, row: str, fill: bool = True, height: float = 1.2, aspect: float = 3.5, palette: str = "muted"):
    """Ridgeline-style faceted KDEs per category using FacetGrid rows."""
    g = sns.FacetGrid(data, row=row, hue=row, aspect=aspect, height=height, palette=palette, sharex=True, sharey=False)
    g.map(sns.kdeplot, x, fill=fill, alpha=0.7 if fill else 1.0, linewidth=1)
    g.map(plt.axhline, y=0, lw=1, clip_on=False)
    g.set_titles("")
    g.set(yticks=[], ylabel="")
    g.despine(bottom=True, left=True)
    return g
