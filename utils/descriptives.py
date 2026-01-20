"""Descriptive statistics helpers for quick exploratory summaries.

Provides:
- `describe_series`: compact stats for a 1D numeric sequence/Series
- `describe_dataframe`: per-column summaries, optionally by group(s)

Intended for teaching and quick EDA, not a full replacement for pandas.describe.
"""

from __future__ import annotations

from typing import Iterable, Optional, Union

import numpy as np
import pandas as pd


Number = Union[int, float, np.number]


def _to_array(x: Iterable[Number]) -> np.ndarray:
    """Convert an iterable to a float numpy array and drop NaNs."""
    a = np.asarray(list(x), dtype=float)
    return a[~np.isnan(a)]


def describe_series(s: Union[pd.Series, Iterable[Number]]) -> pd.Series:
    """Return a compact descriptive summary for a numeric series.

    Includes: count, missing, mean, std, var, min, 25%, 50%, 75%, max,
    IQR, skew, kurtosis, and MAD.
    """
    # Normalize to pandas Series and remove missing values
    if not isinstance(s, pd.Series):
        s = pd.Series(list(s), dtype=float)
    clean = s.dropna()
    # Quartiles and interquartile range
    q = clean.quantile([0.25, 0.5, 0.75])
    iqr = q.loc[0.75] - q.loc[0.25] if not q.empty else np.nan
    return pd.Series(
        {
            "count": int(clean.size),
            "missing": int(s.size - clean.size),
            "mean": clean.mean(),
            "std": clean.std(ddof=1),
            "var": clean.var(ddof=1),
            "min": clean.min(),
            "25%": q.loc[0.25] if 0.25 in q.index else np.nan,
            "50%": q.loc[0.5] if 0.5 in q.index else np.nan,
            "75%": q.loc[0.75] if 0.75 in q.index else np.nan,
            "max": clean.max(),
            "iqr": iqr,
            "skew": clean.skew(),
            "kurtosis": clean.kurt(),
            "mad": clean.mad(),
        }
    )


def describe_dataframe(
    df: pd.DataFrame,
    include: Optional[Iterable[str]] = None,
    groupby: Optional[Union[str, Iterable[str]]] = None,
) -> pd.DataFrame:
    """Describe numeric columns of a dataframe, optionally by group.

    - include: list of column names to summarize (defaults to numeric dtypes)
    - groupby: column(s) to group by before summarizing
    """
    # Select numeric columns by default
    if include is None:
        cols = df.select_dtypes(include=["number"]).columns
    else:
        cols = list(include)

    def _agg(g: pd.DataFrame) -> pd.DataFrame:
        # Compute basic stats for the selected columns within a group
        out = {}
        for c in cols:
            s = g[c]
            clean = s.dropna()
            q = clean.quantile([0.25, 0.5, 0.75]) if not clean.empty else pd.Series()
            out[c] = {
                "count": int(clean.size),
                "missing": int(s.size - clean.size),
                "mean": clean.mean(),
                "std": clean.std(ddof=1),
                "min": clean.min(),
                "25%": q.loc[0.25] if 0.25 in q.index else np.nan,
                "50%": q.loc[0.5] if 0.5 in q.index else np.nan,
                "75%": q.loc[0.75] if 0.75 in q.index else np.nan,
                "max": clean.max(),
            }
        return pd.DataFrame(out).T

    if groupby is None:
        return _agg(df)

    # Group by one or more keys, summarizing each group separately
    gb = df.groupby(groupby, dropna=False)
    parts = []
    for keys, grp in gb:
        part = _agg(grp)
        if not isinstance(keys, tuple):
            keys = (keys,)
        for i, k in enumerate(gb.keys.names or [groupby]):
            part.insert(0, f"{k}", keys[i])
        parts.append(part.reset_index().rename(columns={"index": "variable"}))
    return pd.concat(parts, axis=0, ignore_index=True)
