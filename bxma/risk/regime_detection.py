"""
Advanced Regime Detection with HMM and Thermodynamic Sampling
=============================================================

Identifies market regimes using:
- Hidden Markov Models (HMM) for state classification
- Thermodynamic sampling for uncertainty quantification
- Dynamic regime switching for portfolio adaptation

Key Capabilities:
- Real-time regime identification
- Transition probability estimation
- Regime-aware risk forecasting
- Thermodynamic confidence intervals

References:
- Hamilton (1989): A New Approach to the Economic Analysis of Nonstationary Time Series
- Ang & Bekaert (2002): Regime Switches in Interest Rates
- Bulla (2011): Hidden Markov Models with Applications in Finance

Author: BXMA Quant Team
Date: January 2026
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Literal
from enum import Enum, auto
import time


class MarketRegime(Enum):
    """Market regime classifications."""
    BULL = auto()        # Low vol, positive trend
    BEAR = auto()        # High vol, negative trend
    HIGH_VOL = auto()    # High volatility, no clear trend
    LOW_VOL = auto()     # Low volatility, stable
    CRISIS = auto()      # Extreme stress
    RECOVERY = auto()    # Post-crisis bounce
    NEUTRAL = auto()     # Average conditions


@dataclass
class RegimeState:
    """Current regime state with probabilities."""
    
    regime: MarketRegime
    probability: float
    
    # All regime probabilities
    regime_probabilities: dict[MarketRegime, float] = field(default_factory=dict)
    
    # Transition probabilities
    transition_matrix: NDArray[np.float64] | None = None
    
    # Confidence
    entropy: float = 0.0  # Lower = more certain
    thermodynamic_temperature: float = 1.0
    
    # Timing
    regime_start: datetime = field(default_factory=datetime.now)
    expected_duration: float = 0.0  # Days
    
    # Metadata
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class HMMParameters:
    """Parameters for Hidden Markov Model."""
    
    n_regimes: int = 4
    
    # Initial probabilities
    pi: NDArray[np.float64] | None = None
    
    # Transition matrix (n_regimes x n_regimes)
    transition: NDArray[np.float64] | None = None
    
    # Emission parameters (per regime)
    # Gaussian emissions: means and covariances
    means: NDArray[np.float64] | None = None
    covariances: NDArray[np.float64] | None = None
    
    # Fitted
    log_likelihood: float = 0.0
    aic: float = 0.0
    bic: float = 0.0


class GaussianHMM:
    """
    Gaussian Hidden Markov Model for regime detection.
    
    Assumes each regime emits observations from a multivariate
    Gaussian distribution with regime-specific parameters.
    """
    
    def __init__(
        self,
        n_regimes: int = 4,
        n_features: int = 3,
        covariance_type: Literal["full", "diag", "spherical"] = "full",
        n_iter: int = 100,
        tol: float = 1e-4,
    ):
        self.n_regimes = n_regimes
        self.n_features = n_features
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.tol = tol
        
        self.params: HMMParameters | None = None
        self._is_fitted = False
    
    def _initialize_params(self, X: NDArray[np.float64]):
        """Initialize HMM parameters."""
        n_samples = len(X)
        
        # Initial state probabilities (uniform)
        pi = np.ones(self.n_regimes) / self.n_regimes
        
        # Transition matrix (slightly sticky)
        transition = np.ones((self.n_regimes, self.n_regimes)) * 0.1
        np.fill_diagonal(transition, 0.7)
        transition /= transition.sum(axis=1, keepdims=True)
        
        # Emission parameters via K-means-like initialization
        indices = np.random.choice(n_samples, self.n_regimes, replace=False)
        means = X[indices]
        
        # Covariances
        overall_cov = np.cov(X.T) + np.eye(self.n_features) * 0.01
        covariances = np.array([overall_cov.copy() for _ in range(self.n_regimes)])
        
        self.params = HMMParameters(
            n_regimes=self.n_regimes,
            pi=pi,
            transition=transition,
            means=means,
            covariances=covariances,
        )
    
    def _gaussian_pdf(
        self,
        X: NDArray[np.float64],
        mean: NDArray[np.float64],
        cov: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Multivariate Gaussian PDF."""
        n = len(mean)
        diff = X - mean
        
        # Use pseudo-inverse for numerical stability
        cov_inv = np.linalg.pinv(cov)
        det = np.linalg.det(cov) + 1e-10
        
        exponent = -0.5 * np.sum(diff @ cov_inv * diff, axis=1)
        norm_const = 1 / np.sqrt((2 * np.pi) ** n * det)
        
        return norm_const * np.exp(exponent)
    
    def _compute_emission_probs(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute emission probabilities B[t, k] = P(x_t | z_t = k)."""
        n_samples = len(X)
        B = np.zeros((n_samples, self.n_regimes))
        
        for k in range(self.n_regimes):
            B[:, k] = self._gaussian_pdf(
                X,
                self.params.means[k],
                self.params.covariances[k],
            )
        
        # Avoid zeros
        B = np.maximum(B, 1e-300)
        return B
    
    def _forward(
        self,
        B: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Forward algorithm with scaling."""
        n_samples = len(B)
        alpha = np.zeros((n_samples, self.n_regimes))
        scale = np.zeros(n_samples)
        
        # Initialize
        alpha[0] = self.params.pi * B[0]
        scale[0] = alpha[0].sum()
        alpha[0] /= scale[0]
        
        # Forward pass
        for t in range(1, n_samples):
            alpha[t] = (alpha[t-1] @ self.params.transition) * B[t]
            scale[t] = alpha[t].sum()
            if scale[t] > 0:
                alpha[t] /= scale[t]
        
        return alpha, scale
    
    def _backward(
        self,
        B: NDArray[np.float64],
        scale: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Backward algorithm with scaling."""
        n_samples = len(B)
        beta = np.zeros((n_samples, self.n_regimes))
        
        # Initialize
        beta[-1] = 1.0
        
        # Backward pass
        for t in range(n_samples - 2, -1, -1):
            beta[t] = self.params.transition @ (B[t+1] * beta[t+1])
            if scale[t+1] > 0:
                beta[t] /= scale[t+1]
        
        return beta
    
    def _e_step(
        self,
        X: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], float]:
        """E-step: compute responsibilities."""
        B = self._compute_emission_probs(X)
        alpha, scale = self._forward(B)
        beta = self._backward(B, scale)
        
        # State probabilities gamma[t, k] = P(z_t = k | X)
        gamma = alpha * beta
        gamma /= gamma.sum(axis=1, keepdims=True) + 1e-10
        
        # Transition probabilities xi[t, i, j] = P(z_t = i, z_{t+1} = j | X)
        n_samples = len(X)
        xi = np.zeros((n_samples - 1, self.n_regimes, self.n_regimes))
        
        for t in range(n_samples - 1):
            numerator = np.outer(alpha[t], B[t+1] * beta[t+1]) * self.params.transition
            xi[t] = numerator / (numerator.sum() + 1e-10)
        
        # Log likelihood
        log_likelihood = np.sum(np.log(scale + 1e-300))
        
        return gamma, xi, log_likelihood
    
    def _m_step(
        self,
        X: NDArray[np.float64],
        gamma: NDArray[np.float64],
        xi: NDArray[np.float64],
    ):
        """M-step: update parameters."""
        # Initial probabilities
        self.params.pi = gamma[0] / gamma[0].sum()
        
        # Transition matrix
        xi_sum = xi.sum(axis=0)
        self.params.transition = xi_sum / (xi_sum.sum(axis=1, keepdims=True) + 1e-10)
        
        # Emission parameters
        for k in range(self.n_regimes):
            # Weights for regime k
            w = gamma[:, k]
            w_sum = w.sum() + 1e-10
            
            # Mean
            self.params.means[k] = (w[:, np.newaxis] * X).sum(axis=0) / w_sum
            
            # Covariance
            diff = X - self.params.means[k]
            self.params.covariances[k] = (
                diff.T @ (w[:, np.newaxis] * diff) / w_sum +
                np.eye(self.n_features) * 0.01  # Regularization
            )
    
    def fit(self, X: NDArray[np.float64]) -> GaussianHMM:
        """
        Fit HMM using Baum-Welch (EM) algorithm.
        
        Args:
            X: Observations (n_samples, n_features)
            
        Returns:
            self
        """
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        self.n_features = X.shape[1]
        self._initialize_params(X)
        
        prev_ll = -np.inf
        
        for iteration in range(self.n_iter):
            # E-step
            gamma, xi, log_likelihood = self._e_step(X)
            
            # M-step
            self._m_step(X, gamma, xi)
            
            # Check convergence
            if abs(log_likelihood - prev_ll) < self.tol:
                break
            
            prev_ll = log_likelihood
        
        # Store final likelihood
        self.params.log_likelihood = log_likelihood
        
        # Compute information criteria
        n_params = (
            self.n_regimes - 1 +  # pi
            self.n_regimes * (self.n_regimes - 1) +  # transition
            self.n_regimes * self.n_features +  # means
            self.n_regimes * self.n_features * (self.n_features + 1) // 2  # covariances
        )
        n_samples = len(X)
        
        self.params.aic = -2 * log_likelihood + 2 * n_params
        self.params.bic = -2 * log_likelihood + n_params * np.log(n_samples)
        
        self._is_fitted = True
        return self
    
    def predict(self, X: NDArray[np.float64]) -> NDArray[np.int64]:
        """
        Predict most likely regime sequence (Viterbi algorithm).
        
        Args:
            X: Observations
            
        Returns:
            Most likely regime sequence
        """
        if not self._is_fitted:
            raise ValueError("Model not fitted")
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        n_samples = len(X)
        B = self._compute_emission_probs(X)
        
        # Viterbi
        log_B = np.log(B + 1e-300)
        log_trans = np.log(self.params.transition + 1e-300)
        log_pi = np.log(self.params.pi + 1e-300)
        
        # Initialize
        delta = np.zeros((n_samples, self.n_regimes))
        psi = np.zeros((n_samples, self.n_regimes), dtype=np.int64)
        
        delta[0] = log_pi + log_B[0]
        
        # Forward pass
        for t in range(1, n_samples):
            for j in range(self.n_regimes):
                scores = delta[t-1] + log_trans[:, j]
                psi[t, j] = np.argmax(scores)
                delta[t, j] = scores[psi[t, j]] + log_B[t, j]
        
        # Backtrack
        states = np.zeros(n_samples, dtype=np.int64)
        states[-1] = np.argmax(delta[-1])
        
        for t in range(n_samples - 2, -1, -1):
            states[t] = psi[t + 1, states[t + 1]]
        
        return states
    
    def predict_proba(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Predict regime probabilities.
        
        Args:
            X: Observations
            
        Returns:
            Probability matrix (n_samples, n_regimes)
        """
        if not self._is_fitted:
            raise ValueError("Model not fitted")
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        gamma, _, _ = self._e_step(X)
        return gamma


class ThermodynamicRegimeDetector:
    """
    Regime detection with thermodynamic uncertainty quantification.
    
    Uses simulated annealing-style sampling to:
    - Quantify regime uncertainty
    - Smooth regime transitions
    - Avoid rapid switching
    """
    
    def __init__(
        self,
        hmm: GaussianHMM,
        base_temperature: float = 1.0,
        volatility_scaling: bool = True,
    ):
        """
        Args:
            hmm: Fitted HMM model
            base_temperature: Base thermodynamic temperature
            volatility_scaling: Scale temperature with market volatility
        """
        self.hmm = hmm
        self.base_temperature = base_temperature
        self.volatility_scaling = volatility_scaling
        
        # Regime mapping
        self._regime_map: dict[int, MarketRegime] = {
            0: MarketRegime.BULL,
            1: MarketRegime.BEAR,
            2: MarketRegime.HIGH_VOL,
            3: MarketRegime.LOW_VOL,
        }
    
    def detect(
        self,
        X: NDArray[np.float64],
        current_vix: float = 15.0,
    ) -> RegimeState:
        """
        Detect current regime with thermodynamic uncertainty.
        
        Args:
            X: Recent observations (window of data)
            current_vix: Current VIX for temperature scaling
            
        Returns:
            RegimeState with probabilities and confidence
        """
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        # Get HMM probabilities
        gamma = self.hmm.predict_proba(X)
        current_probs = gamma[-1]  # Last observation
        
        # Apply thermodynamic temperature
        if self.volatility_scaling:
            temperature = self.base_temperature * (current_vix / 15.0)
        else:
            temperature = self.base_temperature
        
        # Boltzmann-like softening
        scaled_probs = np.power(current_probs, 1.0 / temperature)
        scaled_probs /= scaled_probs.sum()
        
        # Find most likely regime
        most_likely = np.argmax(scaled_probs)
        regime = self._regime_map.get(most_likely, MarketRegime.NEUTRAL)
        
        # Compute entropy (uncertainty measure)
        entropy = -np.sum(scaled_probs * np.log(scaled_probs + 1e-10))
        
        # Regime probabilities dict
        regime_probs = {
            self._regime_map.get(i, MarketRegime.NEUTRAL): float(p)
            for i, p in enumerate(scaled_probs)
        }
        
        # Expected duration (from transition matrix diagonal)
        expected_duration = 1.0 / (1.0 - self.hmm.params.transition[most_likely, most_likely] + 1e-10)
        
        return RegimeState(
            regime=regime,
            probability=float(scaled_probs[most_likely]),
            regime_probabilities=regime_probs,
            transition_matrix=self.hmm.params.transition,
            entropy=float(entropy),
            thermodynamic_temperature=temperature,
            expected_duration=expected_duration,
        )
    
    def sample_regimes(
        self,
        X: NDArray[np.float64],
        n_samples: int = 1000,
        current_vix: float = 15.0,
    ) -> list[MarketRegime]:
        """
        Sample possible regimes using thermodynamic sampling.
        
        Provides a distribution of possible regimes for
        uncertainty quantification.
        """
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        gamma = self.hmm.predict_proba(X)
        current_probs = gamma[-1]
        
        # Temperature scaling
        if self.volatility_scaling:
            temperature = self.base_temperature * (current_vix / 15.0)
        else:
            temperature = self.base_temperature
        
        # Boltzmann distribution
        scaled_probs = np.power(current_probs, 1.0 / temperature)
        scaled_probs /= scaled_probs.sum()
        
        # Sample
        samples = np.random.choice(
            len(scaled_probs),
            size=n_samples,
            p=scaled_probs,
        )
        
        return [self._regime_map.get(s, MarketRegime.NEUTRAL) for s in samples]


class RegimeAwareRiskForecaster:
    """
    Risk forecasting that adapts to detected regime.
    
    Uses regime probabilities to blend different volatility
    and correlation forecasts.
    """
    
    def __init__(
        self,
        regime_detector: ThermodynamicRegimeDetector,
    ):
        self.regime_detector = regime_detector
        
        # Regime-specific volatility multipliers
        self._vol_multipliers: dict[MarketRegime, float] = {
            MarketRegime.BULL: 0.8,
            MarketRegime.BEAR: 1.5,
            MarketRegime.HIGH_VOL: 2.0,
            MarketRegime.LOW_VOL: 0.6,
            MarketRegime.CRISIS: 3.0,
            MarketRegime.RECOVERY: 1.2,
            MarketRegime.NEUTRAL: 1.0,
        }
        
        # Regime-specific correlation adjustments
        self._corr_adjustments: dict[MarketRegime, float] = {
            MarketRegime.BULL: -0.1,  # Lower correlation in bull
            MarketRegime.BEAR: 0.2,   # Higher correlation in bear
            MarketRegime.HIGH_VOL: 0.3,
            MarketRegime.LOW_VOL: -0.15,
            MarketRegime.CRISIS: 0.5,  # Correlations spike in crisis
            MarketRegime.RECOVERY: 0.1,
            MarketRegime.NEUTRAL: 0.0,
        }
    
    def forecast_volatility(
        self,
        base_volatility: float,
        regime_state: RegimeState,
    ) -> tuple[float, float, float]:
        """
        Forecast volatility given regime state.
        
        Returns:
            Tuple of (expected_vol, lower_bound, upper_bound)
        """
        # Probability-weighted volatility adjustment
        expected_multiplier = 0.0
        for regime, prob in regime_state.regime_probabilities.items():
            expected_multiplier += prob * self._vol_multipliers.get(regime, 1.0)
        
        expected_vol = base_volatility * expected_multiplier
        
        # Uncertainty bounds based on entropy
        uncertainty = 1 + regime_state.entropy * 0.3
        lower_bound = expected_vol / uncertainty
        upper_bound = expected_vol * uncertainty
        
        return expected_vol, lower_bound, upper_bound
    
    def forecast_correlation(
        self,
        base_correlation: NDArray[np.float64],
        regime_state: RegimeState,
    ) -> NDArray[np.float64]:
        """
        Forecast correlation matrix given regime state.
        
        Adjusts correlations based on regime probabilities.
        """
        # Probability-weighted correlation adjustment
        adjustment = 0.0
        for regime, prob in regime_state.regime_probabilities.items():
            adjustment += prob * self._corr_adjustments.get(regime, 0.0)
        
        # Adjust off-diagonal elements
        adjusted = base_correlation.copy()
        n = len(adjusted)
        
        for i in range(n):
            for j in range(i + 1, n):
                # Increase correlation towards 1 or decrease towards 0
                if adjustment > 0:
                    adjusted[i, j] = adjusted[i, j] + (1 - adjusted[i, j]) * adjustment
                else:
                    adjusted[i, j] = adjusted[i, j] * (1 + adjustment)
                adjusted[j, i] = adjusted[i, j]
        
        return adjusted
    
    def forecast_var(
        self,
        portfolio_value: float,
        portfolio_volatility: float,
        regime_state: RegimeState,
        confidence: float = 0.99,
        horizon_days: int = 1,
    ) -> dict:
        """
        Forecast Value-at-Risk given regime state.
        """
        from scipy.stats import norm
        
        # Get regime-adjusted volatility
        adj_vol, vol_lower, vol_upper = self.forecast_volatility(
            portfolio_volatility,
            regime_state,
        )
        
        # Scale for horizon
        sqrt_horizon = np.sqrt(horizon_days)
        
        # VaR calculation
        z_score = norm.ppf(1 - confidence)
        
        var_expected = -portfolio_value * adj_vol * z_score * sqrt_horizon
        var_lower = -portfolio_value * vol_lower * z_score * sqrt_horizon
        var_upper = -portfolio_value * vol_upper * z_score * sqrt_horizon
        
        return {
            "var_expected": var_expected,
            "var_lower_bound": var_lower,
            "var_upper_bound": var_upper,
            "adjusted_volatility": adj_vol,
            "regime": regime_state.regime.name,
            "regime_probability": regime_state.probability,
            "confidence": confidence,
            "horizon_days": horizon_days,
        }


def build_features_for_regime(
    returns: NDArray[np.float64],
    volatility_window: int = 20,
) -> NDArray[np.float64]:
    """
    Build feature matrix for regime detection.
    
    Features:
    - Realized returns
    - Realized volatility
    - Return skewness
    """
    n = len(returns)
    features = np.zeros((n - volatility_window + 1, 3))
    
    for i in range(volatility_window - 1, n):
        window = returns[i - volatility_window + 1:i + 1]
        
        features[i - volatility_window + 1, 0] = returns[i]  # Current return
        features[i - volatility_window + 1, 1] = np.std(window) * np.sqrt(252)  # Annualized vol
        features[i - volatility_window + 1, 2] = (
            np.mean((window - window.mean()) ** 3) /
            (np.std(window) ** 3 + 1e-10)
        )  # Skewness
    
    return features
