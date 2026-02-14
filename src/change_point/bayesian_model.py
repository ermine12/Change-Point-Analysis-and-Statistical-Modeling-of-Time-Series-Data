"""Bayesian Change Point Detection Model - Refactored and Enhanced

This module provides a class-based API for Bayesian change point detection
with support for both single and multiple change points. It uses log-return
transformation for improved stationarity and includes automated convergence
diagnostics.
"""
from __future__ import annotations

import os
# Configure PyTensor before importing PyMC to avoid C++ compiler issues
os.environ.setdefault('PYTENSOR_FLAGS', 'cxx=')

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
from typing import Optional, Dict, List, Tuple, Any
import warnings



class BayesianChangePointModel:
    """Bayesian change point detection for time series data.
    
    This class implements both single and multiple change point detection
    using PyMC probabilistic programming. It operates on log-returns for
    better stationarity properties.
    
    Attributes:
        data: Original price data (pandas Series with DatetimeIndex)
        returns: Log-returns computed from price data
        model: PyMC model object
        trace: ArviZ InferenceData containing MCMC samples
        change_points: Detected change point dates
        convergence_diagnostics: Dictionary of R-hat and ESS values
    """
    
    def __init__(self, price_series: pd.Series, use_log_returns: bool = True):
        """Initialize the change point model.
        
        Args:
            price_series: Time series of prices with DatetimeIndex
            use_log_returns: If True, model log-returns; if False, model prices directly
        """
        if not isinstance(price_series, pd.Series):
            raise TypeError("price_series must be a pandas Series")
        
        if not isinstance(price_series.index, pd.DatetimeIndex):
            raise TypeError("price_series must have a DatetimeIndex")
        
        self.data = price_series.sort_index()
        self.use_log_returns = use_log_returns
        
        if use_log_returns:
            # Compute log returns: r_t = log(P_t / P_{t-1})
            self.returns = np.log(self.data / self.data.shift(1)).dropna()
        else:
            self.returns = self.data
        
        self.model: Optional[pm.Model] = None
        self.trace: Optional[az.InferenceData] = None
        self.change_points: List[pd.Timestamp] = []
        self.convergence_diagnostics: Dict[str, Any] = {}
    
    def fit_single_change_point(
        self,
        draws: int = 2000,
        tune: int = 1000,
        chains: int = 2,
        random_seed: Optional[int] = None,
        heteroskedastic: bool = True
    ) -> 'BayesianChangePointModel':
        """Fit a single change point model.
        
        Args:
            draws: Number of MCMC samples to draw
            tune: Number of tuning steps
            chains: Number of MCMC chains
            random_seed: Random seed for reproducibility
            heteroskedastic: If True, allow different volatilities before/after change
            
        Returns:
            self for method chaining
        """
        y = self.returns.values
        T = len(y)
        
        if T < 10:
            raise ValueError("Series too short for change-point modeling (min 10 obs)")
        
        with pm.Model() as model:
            # Priors for change point location
            tau = pm.DiscreteUniform('tau', lower=1, upper=T-2)
            
            # Priors for regime means (weakly informative)
            mu_prior_center = float(np.median(y))
            mu_prior_scale = float(np.std(y))
            mu_1 = pm.Normal('mu_1', mu=mu_prior_center, sigma=mu_prior_scale)
            mu_2 = pm.Normal('mu_2', mu=mu_prior_center, sigma=mu_prior_scale)
            
            # Priors for regime volatilities
            sigma_prior_scale = float(np.std(y))
            if heteroskedastic:
                sigma_1 = pm.HalfNormal('sigma_1', sigma=sigma_prior_scale)
                sigma_2 = pm.HalfNormal('sigma_2', sigma=sigma_prior_scale)
            else:
                sigma_shared = pm.HalfNormal('sigma', sigma=sigma_prior_scale)
                sigma_1 = sigma_2 = sigma_shared
            
            # Build likelihood with switch
            idx = np.arange(T)
            mu = pm.math.switch(idx < tau, mu_1, mu_2)
            sigma = pm.math.switch(idx < tau, sigma_1, sigma_2)
            
            obs = pm.Normal('obs', mu=mu, sigma=sigma, observed=y)
            
            # Mixed sampling strategy
            step1 = pm.NUTS([mu_1, mu_2] + ([sigma_1, sigma_2] if heteroskedastic else [sigma_shared]))
            step2 = pm.Metropolis([tau])
            
            trace = pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                cores=1,
                step=[step1, step2],
                random_seed=random_seed,
                progressbar=True,
                return_inferencedata=True
            )
        
        self.model = model
        self.trace = trace
        self._check_convergence()
        self._extract_change_points_single()
        
        return self
    
    def fit_multi_change_point(
        self,
        max_change_points: int = 3,
        min_segment_size: int = 20,
        draws: int = 1500,
        tune: int = 800,
        chains: int = 2,
        random_seed: Optional[int] = None
    ) -> 'BayesianChangePointModel':
        """Fit a multiple change point model using recursive binary segmentation.
        
        This method recursively searches for change points by:
        1. Finding the strongest change point in the full series
        2. Recursively searching in the left and right segments
        3. Combining all detected change points
        
        Args:
            max_change_points: Maximum number of change points to detect
            min_segment_size: Minimum observations per segment
            draws: MCMC draws per iteration
            tune: Tuning steps per iteration
            chains: Number of chains
            random_seed: Random seed for reproducibility
            
        Returns:
            self for method chaining
        """
        self.change_points = self._recursive_segmentation(
            self.returns,
            max_depth=max_change_points,
            min_size=min_segment_size,
            draws=draws,
            tune=tune,
            chains=chains,
            random_seed=random_seed
        )
        
        return self
    
    def _recursive_segmentation(
        self,
        series: pd.Series,
        max_depth: int,
        min_size: int,
        draws: int,
        tune: int,
        chains: int,
        random_seed: Optional[int]
    ) -> List[pd.Timestamp]:
        """Recursive binary segmentation for multiple change points."""
        change_points_found: List[pd.Timestamp] = []
        
        def _search_segment(segment: pd.Series, depth: int):
            if depth <= 0 or len(segment) < min_size * 2:
                return
            
            # Fit single change point to this segment
            y = segment.values
            T = len(y)
            
            try:
                with pm.Model() as temp_model:
                    tau = pm.DiscreteUniform('tau', lower=1, upper=T-2)
                    mu_1 = pm.Normal('mu_1', mu=0.0, sigma=0.05)
                    mu_2 = pm.Normal('mu_2', mu=0.0, sigma=0.05)
                    sigma_1 = pm.HalfNormal('sigma_1', sigma=0.05)
                    sigma_2 = pm.HalfNormal('sigma_2', sigma=0.05)
                    
                    idx = np.arange(T)
                    mu = pm.math.switch(idx < tau, mu_1, mu_2)
                    sigma = pm.math.switch(idx < tau, sigma_1, sigma_2)
                    
                    obs = pm.Normal('obs', mu=mu, sigma=sigma, observed=y)
                    
                    step1 = pm.NUTS([mu_1, mu_2, sigma_1, sigma_2])
                    step2 = pm.Metropolis([tau])
                    
                    idata = pm.sample(
                        draws=draws,
                        tune=tune,
                        chains=chains,
                        cores=1,
                        step=[step1, step2],
                        random_seed=random_seed,
                        progressbar=False,
                        return_inferencedata=True
                    )
                
                # Extract tau
                tau_samples = idata.posterior['tau'].values.flatten()
                tau_median = int(np.median(tau_samples))
                
                # Validate segment sizes
                if tau_median < min_size or (T - tau_median) < min_size:
                    return
                
                # Record change point
                cp_date = segment.index[tau_median]
                change_points_found.append(cp_date)
                
                # Recurse on left and right
                left = segment.iloc[:tau_median]
                right = segment.iloc[tau_median:]
                _search_segment(left, depth - 1)
                _search_segment(right, depth - 1)
                
            except Exception as e:
                warnings.warn(f"Segmentation failed: {e}")
                return
        
        _search_segment(series, max_depth)
        return sorted(list(set(change_points_found)))
    
    def _check_convergence(self):
        """Check MCMC convergence using R-hat and ESS."""
        if self.trace is None:
            return
        
        summary = az.summary(self.trace, var_names=['mu_1', 'mu_2', 'tau'])
        
        # Extract R-hat values
        rhat_values = summary['r_hat'].to_dict()
        ess_bulk = summary['ess_bulk'].to_dict()
        ess_tail = summary['ess_tail'].to_dict()
        
        self.convergence_diagnostics = {
            'r_hat': rhat_values,
            'ess_bulk': ess_bulk,
            'ess_tail': ess_tail,
            'converged': all(v < 1.05 for v in rhat_values.values()),
            'sufficient_ess': all(v > 400 for v in ess_bulk.values())
        }
        
        if not self.convergence_diagnostics['converged']:
            warnings.warn("Model may not have converged (R-hat >= 1.05 detected)")
        
        if not self.convergence_diagnostics['sufficient_ess']:
            warnings.warn("Low effective sample size (ESS < 400 detected)")
    
    def _extract_change_points_single(self):
        """Extract change point from single-point model."""
        if self.trace is None:
            return
        
        tau_samples = self.trace.posterior['tau'].values.flatten()
        tau_mode = int(pd.Series(tau_samples).mode()[0])
        
        # Map back to date
        cp_date = self.returns.index[tau_mode]
        self.change_points = [cp_date]
    
    def summary(self) -> Dict[str, Any]:
        """Generate a summary of the fitted model.
        
        Returns:
            Dictionary containing:
            - change_points: List of detected change point dates
            - regime_parameters: Mean and std for each regime
            - convergence: Convergence diagnostics
            - impact: Quantified impact of change points
        """
        if self.trace is None and not self.change_points:
            raise ValueError("Model has not been fitted yet")
        
        summary_dict = {
            'change_points': [str(cp) for cp in self.change_points],
            'convergence': self.convergence_diagnostics
        }
        
        # Extract regime parameters if single-point model
        if self.trace is not None:
            mu1_samples = self.trace.posterior['mu_1'].values.flatten()
            mu2_samples = self.trace.posterior['mu_2'].values.flatten()
            
            summary_dict['regime_parameters'] = {
                'mu_1': {
                    'mean': float(np.mean(mu1_samples)),
                    'std': float(np.std(mu1_samples)),
                    'credible_interval_95': [
                        float(np.percentile(mu1_samples, 2.5)),
                        float(np.percentile(mu1_samples, 97.5))
                    ]
                },
                'mu_2': {
                    'mean': float(np.mean(mu2_samples)),
                    'std': float(np.std(mu2_samples)),
                    'credible_interval_95': [
                        float(np.percentile(mu2_samples, 2.5)),
                        float(np.percentile(mu2_samples, 97.5))
                    ]
                }
            }
            
            # Compute impact
            mu1_mean = np.mean(mu1_samples)
            mu2_mean = np.mean(mu2_samples)
            
            if self.use_log_returns:
                # For log returns, the difference is already a percentage change
                impact_pct = (mu2_mean - mu1_mean) * 100
            else:
                impact_pct = (mu2_mean - mu1_mean) / mu1_mean * 100
            
            summary_dict['impact'] = {
                'mean_shift_percentage': float(impact_pct),
                'direction': 'increase' if mu2_mean > mu1_mean else 'decrease'
            }
        
        return summary_dict
    
    def predict(self) -> pd.DataFrame:
        """Generate posterior predictive samples.
        
        Returns:
            DataFrame with columns: date, observed, predicted_mean, predicted_lower, predicted_upper
        """
        if self.trace is None:
            raise ValueError("Single change point model has not been fitted")
        
        # Get posterior predictive
        with self.model:
            ppc = pm.sample_posterior_predictive(self.trace, progressbar=False)
        
        obs_samples = ppc.posterior_predictive['obs'].values
        # Shape: (chains, draws, T)
        obs_samples_flat = obs_samples.reshape(-1, obs_samples.shape[-1])
        
        predicted_mean = np.mean(obs_samples_flat, axis=0)
        predicted_lower = np.percentile(obs_samples_flat, 2.5, axis=0)
        predicted_upper = np.percentile(obs_samples_flat, 97.5, axis=0)
        
        return pd.DataFrame({
            'date': self.returns.index,
            'observed': self.returns.values,
            'predicted_mean': predicted_mean,
            'predicted_lower': predicted_lower,
            'predicted_upper': predicted_upper
        })
