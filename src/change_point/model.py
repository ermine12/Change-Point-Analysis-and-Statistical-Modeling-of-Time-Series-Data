"""Change point modeling helpers.

Provides a baseline single change-point model fit using PyMC and a simple
recursive binary-segmentation wrapper to find multiple change points.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
from typing import List, Tuple


def fit_single_change_point(returns: pd.Series, draws=2000, tune=1000, chains=2, random_seed=None) -> Tuple[az.InferenceData, pm.Model]:
    """Fit a single-change-point model on a 1D returns series.

    The model assumes two regimes with different means and stddevs and a
    discrete change-point `tau`. Because `tau` is discrete we use a
    Metropolis step for it and NUTS for continuous variables.

    Returns an ArviZ InferenceData object and the PyMC model context.
    """
    if not isinstance(returns, pd.Series):
        raise TypeError('returns must be a pandas Series')

    y = returns.values
    T = len(y)
    if T < 10:
        raise ValueError('Series too short for change-point modeling')

    model = pm.Model()
    with model:
        mu1 = pm.Normal('mu1', mu=0.0, sigma=0.05)
        mu2 = pm.Normal('mu2', mu=0.0, sigma=0.05)
        sigma1 = pm.HalfNormal('sigma1', sigma=0.05)
        sigma2 = pm.HalfNormal('sigma2', sigma=0.05)
        tau = pm.DiscreteUniform('tau', lower=0, upper=T-1)

        idx = np.arange(T)
        mu = pm.math.switch(idx < tau, mu1, mu2)
        sigma = pm.math.switch(idx < tau, sigma1, sigma2)

        obs = pm.Normal('obs', mu=mu, sigma=sigma, observed=y)

        # Mixed samplers: NUTS for continuous, Metropolis for discrete tau
        step1 = pm.NUTS()  # continuous
        step2 = pm.Metropolis(vars=[tau])

        idata = pm.sample(draws=draws, tune=tune, chains=chains, cores=1, step=[step1, step2], random_seed=random_seed, progressbar=True)

    return idata, model


def detect_change_points_recursive(returns: pd.Series, max_depth: int = 3, min_size: int = 20, draws=1500, tune=800) -> List[pd.Timestamp]:
    """Recursive binary segmentation using the single change-point model.

    Returns list of pandas.Timestamp objects for detected change points (sorted).
    """
    cps: List[pd.Timestamp] = []

    def _segment_search(series: pd.Series, depth: int):
        if depth <= 0 or len(series) < min_size:
            return

        try:
            idata, _ = fit_single_change_point(series, draws=draws, tune=tune, chains=2)
        except Exception:
            return

        # extract posterior samples of tau
        if 'tau' not in idata.posterior:
            return
        tau_samples = idata.posterior['tau'].values.flatten()
        tau_med = int(np.median(tau_samples))

        # enforce minimum segment sizes
        if tau_med < min_size or (len(series) - tau_med) < min_size:
            return

        cp_date = series.index[tau_med]
        cps.append(cp_date)

        # recurse on left and right segments
        left = series.iloc[:tau_med]
        right = series.iloc[tau_med:]
        _segment_search(left, depth - 1)
        _segment_search(right, depth - 1)

    _segment_search(returns, max_depth)
    cps = sorted(list(set(cps)))
    return cps
