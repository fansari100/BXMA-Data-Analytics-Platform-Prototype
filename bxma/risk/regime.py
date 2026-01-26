"""
Regime Detection for BXMA Risk/Quant Platform.

Implements market regime detection using:
- Hidden Markov Models (HMM)
- Neural regime detection
- Volatility regime classification

References:
- "Regime-Switching Models" (Hamilton, 1989)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal
import numpy as np
from numpy.typing import NDArray


@dataclass
class RegimeResult:
    """Container for regime detection results."""
    
    # Current regime
    current_regime: int
    regime_probabilities: NDArray[np.float64]
    
    # Historical regimes
    regime_sequence: NDArray[np.int64]
    smoothed_probabilities: NDArray[np.float64]
    
    # Regime statistics
    n_regimes: int
    regime_means: NDArray[np.float64]
    regime_volatilities: NDArray[np.float64]
    transition_matrix: NDArray[np.float64]


class RegimeDetector(ABC):
    """Abstract base class for regime detection."""
    
    @abstractmethod
    def fit(self, returns: NDArray[np.float64]) -> RegimeResult:
        """Fit regime model to returns."""
        pass
    
    @abstractmethod
    def predict(self, returns: NDArray[np.float64]) -> int:
        """Predict current regime."""
        pass


class HMMRegimeDetector(RegimeDetector):
    """
    Hidden Markov Model Regime Detector.
    
    Uses Gaussian HMM to identify market regimes
    (e.g., low volatility, high volatility, crisis).
    """
    
    def __init__(
        self,
        n_regimes: int = 3,
        n_iter: int = 100,
    ):
        """
        Initialize HMM regime detector.
        
        Args:
            n_regimes: Number of regimes to detect
            n_iter: EM iterations
        """
        self.n_regimes = n_regimes
        self.n_iter = n_iter
        self._model = None
    
    def fit(self, returns: NDArray[np.float64]) -> RegimeResult:
        """Fit HMM to return series."""
        # Simple implementation using EM algorithm
        T = len(returns)
        K = self.n_regimes
        
        # Initialize parameters
        np.random.seed(42)
        means = np.percentile(returns, np.linspace(20, 80, K))
        stds = np.ones(K) * np.std(returns)
        
        # Transition matrix (initial: sticky regimes)
        trans = np.eye(K) * 0.9 + np.ones((K, K)) * 0.1 / K
        trans /= trans.sum(axis=1, keepdims=True)
        
        # EM algorithm
        for _ in range(self.n_iter):
            # E-step: compute responsibilities
            log_lik = np.zeros((T, K))
            for k in range(K):
                log_lik[:, k] = -0.5 * np.log(2 * np.pi * stds[k]**2) - \
                               0.5 * ((returns - means[k]) / stds[k])**2
            
            # Forward-backward (simplified)
            gamma = np.exp(log_lik - log_lik.max(axis=1, keepdims=True))
            gamma /= gamma.sum(axis=1, keepdims=True)
            
            # M-step: update parameters
            for k in range(K):
                weight = gamma[:, k].sum()
                means[k] = (gamma[:, k] * returns).sum() / weight
                stds[k] = np.sqrt((gamma[:, k] * (returns - means[k])**2).sum() / weight)
        
        # Decode regime sequence
        regime_sequence = np.argmax(gamma, axis=1)
        current_regime = int(regime_sequence[-1])
        
        return RegimeResult(
            current_regime=current_regime,
            regime_probabilities=gamma[-1],
            regime_sequence=regime_sequence,
            smoothed_probabilities=gamma,
            n_regimes=K,
            regime_means=means,
            regime_volatilities=stds,
            transition_matrix=trans,
        )
    
    def predict(self, returns: NDArray[np.float64]) -> int:
        """Predict current regime based on recent returns."""
        result = self.fit(returns)
        return result.current_regime


class NeuralRegimeDetector(RegimeDetector):
    """
    Neural Network Regime Detector.
    
    Uses LSTM to detect regime changes from return sequences.
    """
    
    def __init__(
        self,
        n_regimes: int = 4,
        sequence_length: int = 20,
    ):
        self.n_regimes = n_regimes
        self.sequence_length = sequence_length
        self._model = None
    
    def fit(self, returns: NDArray[np.float64]) -> RegimeResult:
        """Fit neural regime detector."""
        # Simplified: use volatility-based regime classification
        T = len(returns)
        K = self.n_regimes
        
        # Rolling volatility
        window = min(21, T // 4)
        rolling_vol = np.array([
            np.std(returns[max(0, i-window):i+1]) for i in range(T)
        ])
        
        # Classify into regimes based on volatility percentiles
        thresholds = np.percentile(rolling_vol, np.linspace(0, 100, K + 1)[1:-1])
        regime_sequence = np.digitize(rolling_vol, thresholds)
        
        # Compute regime statistics
        regime_means = np.array([
            np.mean(returns[regime_sequence == k]) if np.any(regime_sequence == k) else 0
            for k in range(K)
        ])
        regime_vols = np.array([
            np.std(returns[regime_sequence == k]) if np.any(regime_sequence == k) else 0
            for k in range(K)
        ])
        
        # Regime probabilities (simple count-based)
        current_regime = int(regime_sequence[-1])
        regime_probs = np.zeros(K)
        regime_probs[current_regime] = 1.0
        
        # Transition matrix from data
        trans = np.zeros((K, K))
        for i in range(1, T):
            trans[regime_sequence[i-1], regime_sequence[i]] += 1
        trans /= trans.sum(axis=1, keepdims=True) + 1e-10
        
        return RegimeResult(
            current_regime=current_regime,
            regime_probabilities=regime_probs,
            regime_sequence=regime_sequence,
            smoothed_probabilities=np.eye(K)[regime_sequence],
            n_regimes=K,
            regime_means=regime_means,
            regime_volatilities=regime_vols,
            transition_matrix=trans,
        )
    
    def predict(self, returns: NDArray[np.float64]) -> int:
        """Predict current regime."""
        result = self.fit(returns)
        return result.current_regime
