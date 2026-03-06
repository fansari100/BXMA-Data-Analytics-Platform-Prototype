"""
SHAP-Based Explainability for BXMA Data Analytics Platform.

Implements SHAP (SHapley Additive exPlanations) for:
- Risk attribution decomposition
- Portfolio decision explanation
- Factor contribution analysis
- Model-agnostic interpretability

Reference:
- "A Unified Approach to Interpreting Model Predictions"
  (Lundberg & Lee, NIPS 2017)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable
import numpy as np
from numpy.typing import NDArray


@dataclass
class SHAPResult:
    """Container for SHAP analysis results."""
    
    # SHAP values (N_samples x N_features)
    shap_values: NDArray[np.float64]
    
    # Base value (expected model output)
    base_value: float
    
    # Feature names
    feature_names: list[str]
    
    # Feature importances (mean absolute SHAP)
    feature_importances: NDArray[np.float64] | None = None
    
    # Interaction values (optional)
    interaction_values: NDArray[np.float64] | None = None
    
    def get_top_features(self, n: int = 10) -> list[tuple[str, float]]:
        """Get top N most important features."""
        if self.feature_importances is None:
            importances = np.mean(np.abs(self.shap_values), axis=0)
        else:
            importances = self.feature_importances
        
        top_idx = np.argsort(importances)[::-1][:n]
        return [
            (self.feature_names[i], float(importances[i]))
            for i in top_idx
        ]
    
    def explain_prediction(self, sample_idx: int = 0) -> dict:
        """Explain a single prediction."""
        shap = self.shap_values[sample_idx]
        
        contributions = [
            {"feature": self.feature_names[i], "contribution": float(shap[i])}
            for i in np.argsort(np.abs(shap))[::-1]
        ]
        
        return {
            "base_value": self.base_value,
            "prediction": self.base_value + float(np.sum(shap)),
            "contributions": contributions,
        }


class SHAPExplainer:
    """
    SHAP Explainer for ML Models.
    
    Provides model-agnostic interpretability using Shapley values.
    """
    
    def __init__(
        self,
        model: Any = None,
        background_data: NDArray[np.float64] | None = None,
    ):
        """
        Initialize SHAP explainer.
        
        Args:
            model: Model to explain (must have predict method)
            background_data: Background data for KernelSHAP
        """
        self.model = model
        self.background_data = background_data
        self._explainer = None
    
    def explain(
        self,
        X: NDArray[np.float64],
        feature_names: list[str] | None = None,
    ) -> SHAPResult:
        """
        Compute SHAP values for input data.
        
        Args:
            X: Input features (N_samples x N_features)
            feature_names: Names for features
            
        Returns:
            SHAPResult with SHAP values
        """
        try:
            import shap
        except ImportError:
            raise ImportError("SHAP library required for explainability")
        
        N, M = X.shape
        
        if feature_names is None:
            feature_names = [f"Feature_{i}" for i in range(M)]
        
        # Create explainer
        if self._explainer is None:
            if self.background_data is not None:
                self._explainer = shap.KernelExplainer(
                    self.model.predict,
                    self.background_data,
                )
            else:
                # Use provided data as background
                self._explainer = shap.KernelExplainer(
                    self.model.predict,
                    X[:min(100, N)],  # Use subset for efficiency
                )
        
        # Compute SHAP values
        shap_values = self._explainer.shap_values(X)
        
        # Handle multi-output
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        
        # Base value
        base_value = float(self._explainer.expected_value)
        if isinstance(base_value, np.ndarray):
            base_value = float(base_value[0])
        
        # Feature importances
        feature_importances = np.mean(np.abs(shap_values), axis=0)
        
        return SHAPResult(
            shap_values=shap_values,
            base_value=base_value,
            feature_names=feature_names,
            feature_importances=feature_importances,
        )
    
    def plot_summary(self, result: SHAPResult) -> None:
        """Generate SHAP summary plot."""
        try:
            import shap
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("SHAP and matplotlib required for plotting")
        
        shap.summary_plot(
            result.shap_values,
            feature_names=result.feature_names,
            show=False,
        )
        plt.tight_layout()
    
    def plot_force(self, result: SHAPResult, sample_idx: int = 0) -> None:
        """Generate SHAP force plot for single prediction."""
        try:
            import shap
        except ImportError:
            raise ImportError("SHAP required for plotting")
        
        shap.force_plot(
            result.base_value,
            result.shap_values[sample_idx],
            feature_names=result.feature_names,
        )


class RiskAttributionExplainer:
    """
    SHAP-Based Risk Attribution.
    
    Explains portfolio risk contributions using Shapley values,
    providing transparent attribution of risk to individual
    positions and factors.
    """
    
    def __init__(self):
        """Initialize risk attribution explainer."""
        pass
    
    def explain_var(
        self,
        weights: NDArray[np.float64],
        returns: NDArray[np.float64],
        asset_names: list[str] | None = None,
        n_samples: int = 1000,
    ) -> dict:
        """
        Explain VaR using Shapley-based attribution.
        
        Args:
            weights: Portfolio weights
            returns: Historical returns (T x N)
            asset_names: Asset identifiers
            n_samples: Number of coalition samples
            
        Returns:
            Dictionary with risk attribution
        """
        n_assets = len(weights)
        
        if asset_names is None:
            asset_names = [f"Asset_{i}" for i in range(n_assets)]
        
        # Portfolio returns
        portfolio_returns = returns @ weights
        var_95 = np.percentile(portfolio_returns, 5)
        
        # Compute Shapley values for VaR contribution
        # Using sampling-based approximation
        shapley_values = np.zeros(n_assets)
        
        for _ in range(n_samples):
            # Random permutation
            perm = np.random.permutation(n_assets)
            
            # Marginal contribution for each asset in permutation order
            mask = np.zeros(n_assets, dtype=bool)
            prev_var = np.percentile(np.zeros(len(returns)), 5)  # Zero portfolio
            
            for i in perm:
                mask[i] = True
                subset_weights = weights * mask
                subset_returns = returns @ subset_weights
                current_var = np.percentile(subset_returns, 5)
                
                shapley_values[i] += current_var - prev_var
                prev_var = current_var
        
        shapley_values /= n_samples
        
        # Normalize to sum to total VaR
        shapley_values *= var_95 / shapley_values.sum() if shapley_values.sum() != 0 else 1
        
        return {
            "total_var_95": float(var_95),
            "attribution": {
                asset_names[i]: {
                    "shapley_var_contribution": float(shapley_values[i]),
                    "percentage_contribution": float(shapley_values[i] / var_95) * 100 if var_95 != 0 else 0,
                    "weight": float(weights[i]),
                }
                for i in range(n_assets)
            },
            "top_contributors": [
                asset_names[i] for i in np.argsort(shapley_values)[:5]  # Most negative = most risk
            ],
        }
    
    def explain_factor_risk(
        self,
        weights: NDArray[np.float64],
        factor_loadings: NDArray[np.float64],
        factor_covariance: NDArray[np.float64],
        factor_names: list[str] | None = None,
    ) -> dict:
        """
        Explain portfolio risk by factor.
        
        Args:
            weights: Portfolio weights
            factor_loadings: Factor loadings (N_assets x K_factors)
            factor_covariance: Factor covariance (K x K)
            factor_names: Factor names
            
        Returns:
            Factor risk attribution
        """
        n_factors = factor_loadings.shape[1]
        
        if factor_names is None:
            factor_names = [f"Factor_{i}" for i in range(n_factors)]
        
        # Portfolio factor exposures
        factor_exposures = factor_loadings.T @ weights
        
        # Total factor variance
        factor_var = factor_exposures @ factor_covariance @ factor_exposures
        
        # Marginal contribution to factor risk
        marginal_risk = factor_covariance @ factor_exposures
        
        # Risk contribution per factor
        risk_contributions = factor_exposures * marginal_risk
        
        return {
            "total_factor_variance": float(factor_var),
            "total_factor_risk": float(np.sqrt(factor_var)),
            "factor_attribution": {
                factor_names[i]: {
                    "exposure": float(factor_exposures[i]),
                    "risk_contribution": float(risk_contributions[i]),
                    "percentage": float(risk_contributions[i] / factor_var) * 100 if factor_var != 0 else 0,
                }
                for i in range(n_factors)
            },
        }


@dataclass
class DecisionAuditTrail:
    """
    Audit trail for portfolio decisions.
    
    Records all factors considered in each decision
    for transparency and compliance.
    """
    
    decision_id: str
    decision_type: str
    timestamp: str
    
    # Decision factors (average 27+ per decision)
    factors: list[dict] = field(default_factory=list)
    
    # Model outputs
    model_predictions: dict = field(default_factory=dict)
    
    # Risk metrics at decision time
    risk_metrics: dict = field(default_factory=dict)
    
    # Final decision
    decision: dict = field(default_factory=dict)
    
    # Confidence
    confidence_score: float = 0.0
    
    def add_factor(
        self,
        name: str,
        value: float,
        weight: float,
        source: str,
        description: str = "",
    ) -> None:
        """Add a decision factor."""
        self.factors.append({
            "name": name,
            "value": value,
            "weight": weight,
            "source": source,
            "description": description,
        })
    
    def to_dict(self) -> dict:
        """Convert to dictionary for storage."""
        return {
            "decision_id": self.decision_id,
            "decision_type": self.decision_type,
            "timestamp": self.timestamp,
            "n_factors": len(self.factors),
            "factors": self.factors,
            "model_predictions": self.model_predictions,
            "risk_metrics": self.risk_metrics,
            "decision": self.decision,
            "confidence_score": self.confidence_score,
        }
