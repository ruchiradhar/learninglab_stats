"""Simple regression wrappers using statsmodels.

Exports OLS and Logistic formula helpers returning:
- model stats (n, R^2, AIC/BIC)
- tidy params with SE, test stats, p-values, and CIs
- full text summary (string)
"""

from __future__ import annotations

from typing import Dict

import pandas as pd
import numpy as np
try:
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
except Exception:
    sm = None  # type: ignore
    smf = None  # type: ignore


def _summarize_fit(model_fit) -> Dict:
    """Return core model diagnostics and a tidy parameter table."""
    ci = model_fit.conf_int()
    ci.columns = ["ci_low", "ci_high"]
    # Tidy parameter table with coef, SE, t/z, p, and CI
    params = pd.concat([model_fit.params.rename("coef"), model_fit.bse.rename("se"), model_fit.tvalues.rename("t"), model_fit.pvalues.rename("p"), ci], axis=1)
    out = {
        "n": int(model_fit.nobs),
        "r2": float(getattr(model_fit, "rsquared", float("nan"))),
        "r2_adj": float(getattr(model_fit, "rsquared_adj", float("nan"))),
        "aic": float(model_fit.aic),
        "bic": float(model_fit.bic),
        "params": params.reset_index().rename(columns={"index": "term"}),
        "summary": str(model_fit.summary()),
    }
    return out


def ols_formula(formula: str, data: pd.DataFrame) -> Dict:
    if smf is None:
        raise ImportError("statsmodels is required for OLS regression (pip/conda install statsmodels)")
    model = smf.ols(formula, data=data, missing="drop")
    fit = model.fit()
    return _summarize_fit(fit)


def logistic_formula(formula: str, data: pd.DataFrame) -> Dict:
    if smf is None:
        raise ImportError("statsmodels is required for logistic regression (pip/conda install statsmodels)")
    model = smf.logit(formula, data=data, missing="drop")
    fit = model.fit(disp=False)
    # For logistic, use conf_int already on log-odds; provide odds ratios too
    res = _summarize_fit(fit)
    params = res["params"].copy()
    params["odds_ratio"] = params["coef"].apply(lambda x: float(np.exp(x)))
    res["params"] = params
    return res
