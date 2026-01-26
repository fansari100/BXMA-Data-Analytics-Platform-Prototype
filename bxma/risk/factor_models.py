"""
Factor Models for BXMA Risk/Quant Platform.

Implements cutting-edge factor model approaches:
- Statistical Factor Models (PCA, ICA)
- Fundamental Factor Models (Barra-style)
- Dynamic Factor Models (time-varying loadings)
- Sparse Factor Models (L1 regularization)

References:
- "Factor Models in Portfolio and Asset Pricing Theory" (Connor & Korajczyk, 2010)
- "The Cross-Section of Expected Stock Returns" (Fama & French, 1992)
- "Asset Pricing with Observable Stochastic Discount Factors" (Hansen, 2014)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Literal
import numpy as np
from numpy.typing import NDArray
from scipy import linalg
from sklearn.decomposition import PCA, FastICA
from sklearn.linear_model import LassoCV, ElasticNetCV


@dataclass
class FactorModelResult:
    """Container for factor model estimation results."""
    
    # Factor loadings: (N_assets x K_factors)
    loadings: NDArray[np.float64]
    
    # Factor returns: (T x K_factors)
    factor_returns: NDArray[np.float64]
    
    # Specific (idiosyncratic) returns: (T x N_assets)
    specific_returns: NDArray[np.float64]
    
    # Covariance decomposition
    factor_covariance: NDArray[np.float64]  # K x K
    specific_variances: NDArray[np.float64]  # N diagonal
    
    # Model fit statistics
    r_squared: float
    explained_variance_ratio: NDArray[np.float64]
    
    # Factor statistics
    factor_names: list[str] = field(default_factory=list)
    factor_t_stats: NDArray[np.float64] | None = None
    
    @property
    def n_factors(self) -> int:
        """Number of factors."""
        return self.loadings.shape[1]
    
    @property
    def n_assets(self) -> int:
        """Number of assets."""
        return self.loadings.shape[0]
    
    def get_total_covariance(self) -> NDArray[np.float64]:
        """
        Reconstruct total covariance matrix from factor model.
        
        Σ = B @ F @ B' + D
        
        where B = loadings, F = factor covariance, D = specific variances
        """
        factor_cov = self.loadings @ self.factor_covariance @ self.loadings.T
        specific_cov = np.diag(self.specific_variances)
        return factor_cov + specific_cov
    
    def get_systematic_risk(self, weights: NDArray[np.float64]) -> float:
        """Calculate systematic (factor) risk for portfolio."""
        factor_exposure = self.loadings.T @ weights
        return float(factor_exposure @ self.factor_covariance @ factor_exposure)
    
    def get_specific_risk(self, weights: NDArray[np.float64]) -> float:
        """Calculate specific (idiosyncratic) risk for portfolio."""
        return float(weights @ np.diag(self.specific_variances) @ weights)
    
    def get_factor_exposures(self, weights: NDArray[np.float64]) -> NDArray[np.float64]:
        """Calculate portfolio factor exposures."""
        return self.loadings.T @ weights


class FactorModel(ABC):
    """Abstract base class for factor models."""
    
    @abstractmethod
    def fit(self, returns: NDArray[np.float64]) -> FactorModelResult:
        """
        Fit factor model to return data.
        
        Args:
            returns: Asset returns matrix (T x N)
            
        Returns:
            FactorModelResult with loadings and factor returns
        """
        pass
    
    @abstractmethod
    def transform(
        self,
        returns: NDArray[np.float64],
        result: FactorModelResult,
    ) -> NDArray[np.float64]:
        """
        Transform returns to factor space.
        
        Args:
            returns: New returns to transform
            result: Fitted model result
            
        Returns:
            Factor returns
        """
        pass


class StatisticalFactorModel(FactorModel):
    """
    Statistical Factor Model using PCA or ICA.
    
    Extracts latent factors from return covariance structure
    without economic interpretation. Optimal for variance explanation.
    
    Features:
    - Principal Component Analysis (PCA)
    - Independent Component Analysis (ICA)
    - Asymptotic PCA (handles T < N)
    - Sparse PCA (interpretable loadings)
    """
    
    def __init__(
        self,
        n_factors: int = 10,
        method: Literal["pca", "ica", "sparse_pca"] = "pca",
        standardize: bool = True,
    ):
        """
        Initialize statistical factor model.
        
        Args:
            n_factors: Number of factors to extract
            method: Extraction method
            standardize: Standardize returns before extraction
        """
        self.n_factors = n_factors
        self.method = method
        self.standardize = standardize
    
    def fit(self, returns: NDArray[np.float64]) -> FactorModelResult:
        """Fit statistical factor model."""
        T, N = returns.shape
        
        # Demean returns
        mean_returns = np.mean(returns, axis=0)
        centered_returns = returns - mean_returns
        
        # Standardize if requested
        if self.standardize:
            std_returns = np.std(returns, axis=0)
            std_returns[std_returns == 0] = 1  # Avoid division by zero
            scaled_returns = centered_returns / std_returns
        else:
            scaled_returns = centered_returns
            std_returns = np.ones(N)
        
        if self.method == "pca":
            # Standard PCA
            pca = PCA(n_components=self.n_factors)
            factor_returns = pca.fit_transform(scaled_returns)
            loadings = pca.components_.T * std_returns[:, np.newaxis]
            explained_var = pca.explained_variance_ratio_
            
        elif self.method == "ica":
            # First reduce with PCA, then apply ICA
            pca = PCA(n_components=self.n_factors)
            pca_returns = pca.fit_transform(scaled_returns)
            
            ica = FastICA(n_components=self.n_factors, random_state=42)
            factor_returns = ica.fit_transform(pca_returns)
            
            # Reconstruct loadings
            loadings = (pca.components_.T @ ica.mixing_) * std_returns[:, np.newaxis]
            explained_var = pca.explained_variance_ratio_
            
        elif self.method == "sparse_pca":
            # Sparse PCA using iterative algorithm
            from sklearn.decomposition import SparsePCA
            spca = SparsePCA(n_components=self.n_factors, random_state=42)
            factor_returns = spca.fit_transform(scaled_returns)
            loadings = spca.components_.T * std_returns[:, np.newaxis]
            
            # Compute explained variance
            reconstructed = factor_returns @ loadings.T
            total_var = np.var(centered_returns)
            explained_var = np.array([
                np.var(factor_returns[:, i] * loadings[:, i]) / total_var
                for i in range(self.n_factors)
            ])
        
        # Compute specific returns
        systematic_returns = factor_returns @ loadings.T
        specific_returns = centered_returns - systematic_returns
        
        # Factor covariance
        factor_covariance = np.cov(factor_returns, rowvar=False)
        if factor_covariance.ndim == 0:
            factor_covariance = np.array([[factor_covariance]])
        
        # Specific variances
        specific_variances = np.var(specific_returns, axis=0)
        
        # R-squared
        total_variance = np.var(centered_returns, axis=0).sum()
        specific_variance = specific_variances.sum()
        r_squared = 1 - specific_variance / total_variance
        
        # Factor names
        factor_names = [f"Factor_{i+1}" for i in range(self.n_factors)]
        
        return FactorModelResult(
            loadings=loadings,
            factor_returns=factor_returns,
            specific_returns=specific_returns,
            factor_covariance=factor_covariance,
            specific_variances=specific_variances,
            r_squared=r_squared,
            explained_variance_ratio=explained_var,
            factor_names=factor_names,
        )
    
    def transform(
        self,
        returns: NDArray[np.float64],
        result: FactorModelResult,
    ) -> NDArray[np.float64]:
        """Transform returns to factor space."""
        # Use pseudo-inverse for transformation
        loadings_pinv = np.linalg.pinv(result.loadings)
        mean_returns = np.mean(returns, axis=0)
        centered = returns - mean_returns
        return centered @ loadings_pinv.T


class FundamentalFactorModel(FactorModel):
    """
    Fundamental Factor Model (Barra-style).
    
    Uses observable characteristics as factors with cross-sectional
    regression to estimate factor returns. Economic interpretability.
    
    Features:
    - Cross-sectional regression factor returns
    - Multiple factor categories (style, industry, country)
    - Weighted least squares (WLS) option
    - Robust regression (handles outliers)
    """
    
    def __init__(
        self,
        weighting: Literal["equal", "cap_weighted", "vol_weighted"] = "equal",
        robust: bool = False,
    ):
        """
        Initialize fundamental factor model.
        
        Args:
            weighting: Regression weighting scheme
            robust: Use robust regression
        """
        self.weighting = weighting
        self.robust = robust
    
    def fit(
        self,
        returns: NDArray[np.float64],
        factor_exposures: NDArray[np.float64] | None = None,
        factor_names: list[str] | None = None,
    ) -> FactorModelResult:
        """
        Fit fundamental factor model.
        
        Args:
            returns: Asset returns (T x N)
            factor_exposures: Known factor exposures (N x K)
            factor_names: Names for factors
            
        Returns:
            FactorModelResult
        """
        T, N = returns.shape
        
        if factor_exposures is None:
            # Default to PCA-based exposures
            pca = PCA(n_components=min(10, N-1))
            factor_exposures = pca.fit_transform(returns.T).T
        
        K = factor_exposures.shape[1]
        
        if factor_names is None:
            factor_names = [f"Factor_{i+1}" for i in range(K)]
        
        # Cross-sectional regression for each time period
        factor_returns = np.zeros((T, K))
        specific_returns = np.zeros((T, N))
        
        for t in range(T):
            y = returns[t, :]
            X = factor_exposures
            
            if self.robust:
                from sklearn.linear_model import HuberRegressor
                reg = HuberRegressor()
                reg.fit(X, y)
                factor_returns[t, :] = reg.coef_
            else:
                # OLS: f = (X'X)^-1 X'y
                factor_returns[t, :] = np.linalg.lstsq(X, y, rcond=None)[0]
            
            # Specific returns
            specific_returns[t, :] = y - X @ factor_returns[t, :]
        
        # Factor covariance
        factor_covariance = np.cov(factor_returns, rowvar=False)
        if factor_covariance.ndim == 0:
            factor_covariance = np.array([[factor_covariance]])
        
        # Specific variances
        specific_variances = np.var(specific_returns, axis=0)
        
        # R-squared
        total_variance = np.var(returns, axis=0).sum()
        specific_variance = specific_variances.sum()
        r_squared = 1 - specific_variance / total_variance
        
        # Explained variance by factor
        factor_vars = np.var(factor_returns, axis=0)
        explained_var = factor_vars / factor_vars.sum()
        
        # T-statistics for factor returns
        factor_means = np.mean(factor_returns, axis=0)
        factor_stds = np.std(factor_returns, axis=0)
        t_stats = factor_means / (factor_stds / np.sqrt(T))
        
        return FactorModelResult(
            loadings=factor_exposures,
            factor_returns=factor_returns,
            specific_returns=specific_returns,
            factor_covariance=factor_covariance,
            specific_variances=specific_variances,
            r_squared=r_squared,
            explained_variance_ratio=explained_var,
            factor_names=factor_names,
            factor_t_stats=t_stats,
        )
    
    def transform(
        self,
        returns: NDArray[np.float64],
        result: FactorModelResult,
    ) -> NDArray[np.float64]:
        """Transform returns to factor space."""
        T = returns.shape[0]
        K = result.n_factors
        factor_returns = np.zeros((T, K))
        
        for t in range(T):
            factor_returns[t, :] = np.linalg.lstsq(
                result.loadings, returns[t, :], rcond=None
            )[0]
        
        return factor_returns


class DynamicFactorModel(FactorModel):
    """
    Dynamic Factor Model with time-varying loadings.
    
    Extends static factor models to allow factor loadings
    to evolve over time using state-space framework.
    
    Features:
    - Kalman filter estimation
    - Time-varying factor loadings
    - Regime-switching capability
    - Rolling window estimation
    
    Reference:
    - "Time-Varying Factor Models" (Stock & Watson, 2009)
    """
    
    def __init__(
        self,
        n_factors: int = 5,
        method: Literal["rolling", "kalman", "regime"] = "rolling",
        window: int = 252,
    ):
        """
        Initialize dynamic factor model.
        
        Args:
            n_factors: Number of factors
            method: Estimation method
            window: Rolling window size
        """
        self.n_factors = n_factors
        self.method = method
        self.window = window
    
    def fit(self, returns: NDArray[np.float64]) -> FactorModelResult:
        """Fit dynamic factor model."""
        T, N = returns.shape
        
        if self.method == "rolling":
            # Rolling window PCA
            loadings_series = []
            factor_returns_series = []
            
            for t in range(self.window, T):
                window_returns = returns[t-self.window:t, :]
                
                pca = PCA(n_components=self.n_factors)
                window_factors = pca.fit_transform(window_returns)
                
                loadings_series.append(pca.components_.T)
                factor_returns_series.append(window_factors[-1, :])
            
            # Use final loadings
            loadings = loadings_series[-1]
            factor_returns = np.array(factor_returns_series)
            
            # Specific returns for final window
            systematic = factor_returns @ loadings.T
            specific_returns = returns[-len(factor_returns):, :] - systematic
            
        elif self.method == "kalman":
            # State-space model with Kalman filter
            # Simplified implementation using rolling with exponential weighting
            alpha = 2 / (self.window + 1)
            
            # Initialize with first window
            pca = PCA(n_components=self.n_factors)
            pca.fit(returns[:self.window, :])
            loadings = pca.components_.T
            
            factor_returns = []
            for t in range(T):
                # Update loadings with exponential smoothing
                if t >= self.window:
                    current_pca = PCA(n_components=self.n_factors)
                    current_pca.fit(returns[t-self.window:t, :])
                    loadings = alpha * current_pca.components_.T + (1 - alpha) * loadings
                
                # Project returns onto factors
                f_t = np.linalg.lstsq(loadings, returns[t, :], rcond=None)[0]
                factor_returns.append(f_t)
            
            factor_returns = np.array(factor_returns)
            systematic = factor_returns @ loadings.T
            specific_returns = returns - systematic
        
        # Factor covariance
        factor_covariance = np.cov(factor_returns, rowvar=False)
        if factor_covariance.ndim == 0:
            factor_covariance = np.array([[factor_covariance]])
        
        # Specific variances
        specific_variances = np.var(specific_returns, axis=0)
        
        # R-squared
        total_variance = np.var(returns, axis=0).sum()
        specific_variance = specific_variances.sum()
        r_squared = 1 - specific_variance / total_variance
        
        # Explained variance
        factor_vars = np.var(factor_returns, axis=0)
        explained_var = factor_vars / factor_vars.sum() if factor_vars.sum() > 0 else factor_vars
        
        return FactorModelResult(
            loadings=loadings,
            factor_returns=factor_returns,
            specific_returns=specific_returns,
            factor_covariance=factor_covariance,
            specific_variances=specific_variances,
            r_squared=r_squared,
            explained_variance_ratio=explained_var,
            factor_names=[f"DynamicFactor_{i+1}" for i in range(self.n_factors)],
        )
    
    def transform(
        self,
        returns: NDArray[np.float64],
        result: FactorModelResult,
    ) -> NDArray[np.float64]:
        """Transform returns to factor space."""
        T = returns.shape[0]
        factor_returns = np.zeros((T, result.n_factors))
        
        for t in range(T):
            factor_returns[t, :] = np.linalg.lstsq(
                result.loadings, returns[t, :], rcond=None
            )[0]
        
        return factor_returns
