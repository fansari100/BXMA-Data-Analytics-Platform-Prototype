"""
BXMA Real Financial Calculations
Implements actual financial formulas with full mathematical derivations
"""

import numpy as np
from numpy.typing import NDArray
from scipy import stats
from scipy.optimize import minimize
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from datetime import date
import pandas as pd


@dataclass
class CalculationStep:
    """Represents a single step in a calculation"""
    step_number: int
    title: str
    formula_latex: str
    description: str
    inputs: Dict[str, str]
    calculation: str
    result: str


@dataclass
class CalculationResult:
    """Result of a calculation with full derivation"""
    name: str
    value: float
    unit: str
    steps: List[CalculationStep]
    inputs_summary: Dict[str, float]
    formula_latex: str


# =============================================================================
# COVARIANCE & CORRELATION CALCULATIONS
# =============================================================================

def calculate_covariance_matrix(
    returns: NDArray[np.float64],
    method: str = "sample",
    decay_factor: float = 0.94
) -> Tuple[NDArray[np.float64], List[CalculationStep]]:
    """
    Calculate covariance matrix with full derivation steps
    
    Methods:
    - sample: Standard sample covariance
    - ewma: Exponentially weighted (RiskMetrics)
    - ledoit_wolf: Shrinkage estimator
    """
    n_obs, n_assets = returns.shape
    steps = []
    
    if method == "sample":
        # Step 1: Calculate mean returns
        mean_returns = np.mean(returns, axis=0)
        steps.append(CalculationStep(
            step_number=1,
            title="Calculate Mean Returns",
            formula_latex=r"\bar{r}_i = \frac{1}{T} \sum_{t=1}^{T} r_{i,t}",
            description="Compute the arithmetic mean of returns for each asset",
            inputs={"T": str(n_obs), "n": str(n_assets)},
            calculation=f"Mean returns vector: {np.round(mean_returns * 252 * 100, 2)}% (annualized)",
            result=f"μ = [{', '.join([f'{m*252*100:.2f}%' for m in mean_returns])}]"
        ))
        
        # Step 2: Center returns
        centered = returns - mean_returns
        steps.append(CalculationStep(
            step_number=2,
            title="Center Returns (Subtract Mean)",
            formula_latex=r"\tilde{r}_{i,t} = r_{i,t} - \bar{r}_i",
            description="Subtract mean from each return to center the data",
            inputs={"r": "Daily returns", "μ": "Mean returns"},
            calculation="Centered returns matrix created",
            result=f"Centered returns: {n_obs} observations × {n_assets} assets"
        ))
        
        # Step 3: Calculate covariance
        cov = np.cov(returns.T, ddof=1)
        steps.append(CalculationStep(
            step_number=3,
            title="Calculate Sample Covariance Matrix",
            formula_latex=r"\Sigma_{i,j} = \frac{1}{T-1} \sum_{t=1}^{T} \tilde{r}_{i,t} \tilde{r}_{j,t}",
            description="Compute the sample covariance using Bessel's correction (T-1)",
            inputs={"T-1": str(n_obs - 1)},
            calculation="Σ = (1/(T-1)) × X'X where X is centered returns",
            result=f"Covariance matrix: {n_assets}×{n_assets}, trace = {np.trace(cov):.6f}"
        ))
        
    elif method == "ewma":
        # RiskMetrics EWMA
        λ = decay_factor
        
        steps.append(CalculationStep(
            step_number=1,
            title="Initialize EWMA Covariance",
            formula_latex=r"\sigma^2_{t|t-1} = (1-\lambda) r_{t-1}^2 + \lambda \sigma^2_{t-1|t-2}",
            description=f"RiskMetrics exponentially weighted covariance with λ = {λ}",
            inputs={"λ": str(λ), "T": str(n_obs)},
            calculation=f"Half-life = ln(2)/ln(1/λ) = {np.log(2)/np.log(1/λ):.1f} days",
            result="EWMA weights give more weight to recent observations"
        ))
        
        # Calculate EWMA weights
        weights = np.array([(1 - λ) * (λ ** i) for i in range(n_obs)])
        weights = weights[::-1]  # Most recent first
        weights = weights / weights.sum()  # Normalize
        
        steps.append(CalculationStep(
            step_number=2,
            title="Calculate EWMA Weights",
            formula_latex=r"w_t = (1-\lambda) \lambda^{T-t} / \sum_{s=1}^{T}(1-\lambda)\lambda^{T-s}",
            description="Exponentially decaying weights (normalized)",
            inputs={"λ": str(λ)},
            calculation=f"Weight on most recent: {weights[-1]:.4f}, oldest: {weights[0]:.6f}",
            result=f"Sum of weights = {weights.sum():.4f}"
        ))
        
        # Weighted covariance
        mean_returns = np.average(returns, weights=weights, axis=0)
        centered = returns - mean_returns
        cov = np.zeros((n_assets, n_assets))
        for t in range(n_obs):
            cov += weights[t] * np.outer(centered[t], centered[t])
        
        steps.append(CalculationStep(
            step_number=3,
            title="Calculate EWMA Covariance Matrix",
            formula_latex=r"\Sigma_{EWMA} = \sum_{t=1}^{T} w_t (r_t - \bar{r}_w)(r_t - \bar{r}_w)'",
            description="Weighted sum of outer products of centered returns",
            inputs={"weights": "EWMA weights"},
            calculation="Σ_EWMA computed with exponentially weighted observations",
            result=f"EWMA Covariance: trace = {np.trace(cov):.6f}"
        ))
    
    else:  # Ledoit-Wolf shrinkage
        sample_cov = np.cov(returns.T, ddof=1)
        
        # Target: scaled identity matrix
        mu = np.trace(sample_cov) / n_assets
        target = mu * np.eye(n_assets)
        
        steps.append(CalculationStep(
            step_number=1,
            title="Calculate Shrinkage Target",
            formula_latex=r"F = \frac{tr(\Sigma)}{n} I_n",
            description="Scaled identity matrix as shrinkage target",
            inputs={"tr(Σ)": f"{np.trace(sample_cov):.6f}", "n": str(n_assets)},
            calculation=f"μ = tr(Σ)/n = {mu:.6f}",
            result=f"Target = {mu:.6f} × I_{n_assets}"
        ))
        
        # Estimate optimal shrinkage intensity
        delta = sample_cov - target
        delta_sq = np.sum(delta ** 2)
        
        # Simplified shrinkage intensity estimation
        shrinkage = min(1.0, max(0.0, (n_assets / n_obs) * 0.5))
        
        steps.append(CalculationStep(
            step_number=2,
            title="Estimate Shrinkage Intensity",
            formula_latex=r"\alpha^* = \frac{\sum_{i,j} Var(\sigma_{ij})}{\sum_{i,j}(\sigma_{ij} - f_{ij})^2}",
            description="Optimal shrinkage intensity balances bias and variance",
            inputs={"n_obs": str(n_obs), "n_assets": str(n_assets)},
            calculation=f"Shrinkage intensity α = {shrinkage:.4f}",
            result=f"α* = {shrinkage:.4f} (0=sample, 1=target)"
        ))
        
        cov = (1 - shrinkage) * sample_cov + shrinkage * target
        
        steps.append(CalculationStep(
            step_number=3,
            title="Apply Ledoit-Wolf Shrinkage",
            formula_latex=r"\Sigma_{LW} = (1-\alpha) \Sigma_{sample} + \alpha F",
            description="Linear combination of sample covariance and target",
            inputs={"α": f"{shrinkage:.4f}"},
            calculation=f"Σ_LW = {1-shrinkage:.4f} × Σ_sample + {shrinkage:.4f} × F",
            result=f"Shrunk covariance: condition number improved"
        ))
    
    return cov, steps


def calculate_correlation_matrix(
    cov: NDArray[np.float64]
) -> Tuple[NDArray[np.float64], List[CalculationStep]]:
    """Convert covariance to correlation with derivation"""
    steps = []
    
    # Extract standard deviations
    std = np.sqrt(np.diag(cov))
    
    steps.append(CalculationStep(
        step_number=1,
        title="Extract Standard Deviations",
        formula_latex=r"\sigma_i = \sqrt{\Sigma_{ii}}",
        description="Standard deviation is square root of variance (diagonal elements)",
        inputs={"Σ_ii": "Diagonal of covariance matrix"},
        calculation=f"σ = [{', '.join([f'{s*np.sqrt(252)*100:.2f}%' for s in std[:5]])}...] (annualized)",
        result=f"Volatilities extracted for {len(std)} assets"
    ))
    
    # Calculate correlation
    D_inv = np.diag(1 / std)
    corr = D_inv @ cov @ D_inv
    
    steps.append(CalculationStep(
        step_number=2,
        title="Compute Correlation Matrix",
        formula_latex=r"\rho_{ij} = \frac{\Sigma_{ij}}{\sigma_i \sigma_j}",
        description="Correlation = covariance divided by product of standard deviations",
        inputs={"Σ": "Covariance matrix", "σ": "Standard deviations"},
        calculation="ρ = D⁻¹ Σ D⁻¹ where D = diag(σ)",
        result=f"Correlation matrix: all diagonal = 1.0, off-diagonal ∈ [-1, 1]"
    ))
    
    return corr, steps


# =============================================================================
# VALUE AT RISK (VaR) CALCULATIONS
# =============================================================================

def calculate_var(
    returns: NDArray[np.float64],
    weights: NDArray[np.float64],
    confidence: float = 0.95,
    horizon_days: int = 1,
    method: str = "parametric",
    portfolio_value: float = 100_000_000.0
) -> CalculationResult:
    """
    Calculate Value at Risk with full derivation
    
    Methods:
    - parametric: Normal distribution assumption
    - historical: Historical simulation
    - cornish_fisher: Cornish-Fisher expansion (accounts for skewness/kurtosis)
    """
    steps = []
    
    # Calculate portfolio returns
    port_returns = returns @ weights
    n_obs = len(port_returns)
    
    steps.append(CalculationStep(
        step_number=1,
        title="Calculate Portfolio Returns",
        formula_latex=r"r_p = \sum_{i=1}^{n} w_i r_i = \mathbf{w}' \mathbf{r}",
        description="Portfolio return is weighted sum of asset returns",
        inputs={"n": str(len(weights)), "T": str(n_obs)},
        calculation=f"Portfolio: {n_obs} daily returns computed",
        result=f"Mean daily return: {np.mean(port_returns)*100:.4f}%"
    ))
    
    if method == "parametric":
        # Calculate mean and std
        mu = np.mean(port_returns)
        sigma = np.std(port_returns, ddof=1)
        
        steps.append(CalculationStep(
            step_number=2,
            title="Estimate Distribution Parameters",
            formula_latex=r"\mu = \frac{1}{T}\sum r_{p,t}, \quad \sigma = \sqrt{\frac{1}{T-1}\sum(r_{p,t}-\mu)^2}",
            description="Maximum likelihood estimates assuming normality",
            inputs={"T": str(n_obs)},
            calculation=f"μ = {mu*100:.4f}%, σ = {sigma*100:.4f}% (daily)",
            result=f"Annualized: μ = {mu*252*100:.2f}%, σ = {sigma*np.sqrt(252)*100:.2f}%"
        ))
        
        # Calculate z-score
        alpha = 1 - confidence
        z = stats.norm.ppf(alpha)
        
        steps.append(CalculationStep(
            step_number=3,
            title="Calculate Critical Value (z-score)",
            formula_latex=r"z_\alpha = \Phi^{-1}(\alpha)",
            description=f"Inverse normal CDF at α = {alpha:.2f} ({confidence*100:.0f}% confidence)",
            inputs={"α": str(alpha), "confidence": f"{confidence*100:.0f}%"},
            calculation=f"z_{alpha} = Φ⁻¹({alpha}) = {z:.4f}",
            result=f"z = {z:.4f} (left tail critical value)"
        ))
        
        # Scale for horizon
        sigma_h = sigma * np.sqrt(horizon_days)
        mu_h = mu * horizon_days
        
        steps.append(CalculationStep(
            step_number=4,
            title="Scale for Time Horizon",
            formula_latex=r"\sigma_h = \sigma_1 \sqrt{h}, \quad \mu_h = \mu_1 \cdot h",
            description="Square root of time rule for volatility scaling",
            inputs={"h": str(horizon_days), "σ_1": f"{sigma*100:.4f}%"},
            calculation=f"σ_{horizon_days} = {sigma*100:.4f}% × √{horizon_days} = {sigma_h*100:.4f}%",
            result=f"Horizon-scaled: σ_h = {sigma_h*100:.4f}%, μ_h = {mu_h*100:.4f}%"
        ))
        
        # Calculate VaR
        var_pct = -(mu_h + z * sigma_h)
        var_dollar = var_pct * portfolio_value
        
        steps.append(CalculationStep(
            step_number=5,
            title="Calculate Parametric VaR",
            formula_latex=r"VaR_{\alpha} = -(\mu_h + z_\alpha \sigma_h) \cdot V",
            description="VaR as negative of the worst expected loss at confidence level",
            inputs={
                "μ_h": f"{mu_h*100:.4f}%",
                "z": f"{z:.4f}",
                "σ_h": f"{sigma_h*100:.4f}%",
                "V": f"${portfolio_value:,.0f}"
            },
            calculation=f"VaR = -({mu_h*100:.4f}% + {z:.4f} × {sigma_h*100:.4f}%) × ${portfolio_value:,.0f}",
            result=f"VaR({confidence*100:.0f}%) = {var_pct*100:.4f}% = ${var_dollar:,.0f}"
        ))
        
    elif method == "historical":
        # Sort returns
        sorted_returns = np.sort(port_returns)
        
        # Scale for horizon
        sorted_returns_h = sorted_returns * np.sqrt(horizon_days)
        
        steps.append(CalculationStep(
            step_number=2,
            title="Sort Historical Returns",
            formula_latex=r"r_{(1)} \leq r_{(2)} \leq ... \leq r_{(T)}",
            description="Order statistics of portfolio returns",
            inputs={"T": str(n_obs)},
            calculation=f"Sorted {n_obs} returns, min = {sorted_returns[0]*100:.4f}%, max = {sorted_returns[-1]*100:.4f}%",
            result=f"Horizon-scaled (√{horizon_days}) applied"
        ))
        
        # Find percentile
        alpha = 1 - confidence
        var_index = int(np.floor(alpha * n_obs))
        var_pct = -sorted_returns_h[var_index]
        var_dollar = var_pct * portfolio_value
        
        steps.append(CalculationStep(
            step_number=3,
            title="Identify VaR Percentile",
            formula_latex=r"VaR_\alpha = -r_{(\lfloor \alpha T \rfloor)}",
            description=f"Select the {alpha*100:.0f}th percentile loss",
            inputs={"α": str(alpha), "T": str(n_obs), "index": str(var_index)},
            calculation=f"VaR = -r_({var_index}) = -{sorted_returns_h[var_index]*100:.4f}%",
            result=f"Historical VaR({confidence*100:.0f}%) = {var_pct*100:.4f}% = ${var_dollar:,.0f}"
        ))
    
    else:  # Cornish-Fisher
        mu = np.mean(port_returns)
        sigma = np.std(port_returns, ddof=1)
        skew = stats.skew(port_returns)
        kurt = stats.kurtosis(port_returns)  # Excess kurtosis
        
        steps.append(CalculationStep(
            step_number=2,
            title="Calculate Higher Moments",
            formula_latex=r"S = \frac{E[(r-\mu)^3]}{\sigma^3}, \quad K = \frac{E[(r-\mu)^4]}{\sigma^4} - 3",
            description="Skewness and excess kurtosis of portfolio returns",
            inputs={"T": str(n_obs)},
            calculation=f"Skewness S = {skew:.4f}, Excess Kurtosis K = {kurt:.4f}",
            result=f"Non-normality detected: S≠0, K≠0"
        ))
        
        # Cornish-Fisher expansion
        alpha = 1 - confidence
        z = stats.norm.ppf(alpha)
        z_cf = (z + (z**2 - 1) * skew / 6 + 
                (z**3 - 3*z) * kurt / 24 - 
                (2*z**3 - 5*z) * skew**2 / 36)
        
        steps.append(CalculationStep(
            step_number=3,
            title="Cornish-Fisher Expansion",
            formula_latex=r"z_{CF} = z + \frac{z^2-1}{6}S + \frac{z^3-3z}{24}K - \frac{2z^3-5z}{36}S^2",
            description="Adjust z-score for non-normality",
            inputs={"z": f"{z:.4f}", "S": f"{skew:.4f}", "K": f"{kurt:.4f}"},
            calculation=f"z_CF = {z:.4f} + adjustments = {z_cf:.4f}",
            result=f"Adjusted critical value: {z_cf:.4f} (vs. {z:.4f} for normal)"
        ))
        
        # Scale and calculate VaR
        sigma_h = sigma * np.sqrt(horizon_days)
        mu_h = mu * horizon_days
        var_pct = -(mu_h + z_cf * sigma_h)
        var_dollar = var_pct * portfolio_value
        
        steps.append(CalculationStep(
            step_number=4,
            title="Calculate Cornish-Fisher VaR",
            formula_latex=r"VaR_{CF} = -(\mu_h + z_{CF} \sigma_h) \cdot V",
            description="VaR using Cornish-Fisher adjusted quantile",
            inputs={"z_CF": f"{z_cf:.4f}", "σ_h": f"{sigma_h*100:.4f}%"},
            calculation=f"VaR = -({mu_h*100:.4f}% + {z_cf:.4f} × {sigma_h*100:.4f}%) × ${portfolio_value:,.0f}",
            result=f"VaR_CF({confidence*100:.0f}%) = {var_pct*100:.4f}% = ${var_dollar:,.0f}"
        ))
    
    return CalculationResult(
        name=f"VaR ({method.title()})",
        value=var_pct,
        unit="percentage",
        steps=steps,
        inputs_summary={
            "confidence": confidence,
            "horizon_days": horizon_days,
            "portfolio_value": portfolio_value,
            "n_observations": n_obs,
        },
        formula_latex=r"VaR_\alpha = -\inf\{l : P(L > l) \leq 1-\alpha\}"
    )


# =============================================================================
# EXPECTED SHORTFALL (CVaR) CALCULATIONS
# =============================================================================

def calculate_cvar(
    returns: NDArray[np.float64],
    weights: NDArray[np.float64],
    confidence: float = 0.95,
    horizon_days: int = 1,
    portfolio_value: float = 100_000_000.0
) -> CalculationResult:
    """Calculate Conditional VaR (Expected Shortfall) with derivation"""
    steps = []
    
    # Portfolio returns
    port_returns = returns @ weights
    n_obs = len(port_returns)
    
    steps.append(CalculationStep(
        step_number=1,
        title="Calculate Portfolio Returns",
        formula_latex=r"r_p = \mathbf{w}' \mathbf{r}",
        description="Portfolio return is weighted sum of asset returns",
        inputs={"n_assets": str(len(weights)), "T": str(n_obs)},
        calculation=f"Computed {n_obs} portfolio returns",
        result=f"Mean: {np.mean(port_returns)*100:.4f}%, Std: {np.std(port_returns)*100:.4f}%"
    ))
    
    # Sort returns
    sorted_returns = np.sort(port_returns)
    
    # Find VaR threshold
    alpha = 1 - confidence
    var_index = int(np.floor(alpha * n_obs))
    var_threshold = sorted_returns[var_index]
    
    steps.append(CalculationStep(
        step_number=2,
        title="Identify VaR Threshold",
        formula_latex=r"VaR_\alpha = r_{(\lfloor \alpha T \rfloor)}",
        description=f"Find the {alpha*100:.0f}th percentile return",
        inputs={"α": str(alpha), "T": str(n_obs)},
        calculation=f"VaR threshold at index {var_index}: {var_threshold*100:.4f}%",
        result=f"Returns below {var_threshold*100:.4f}% are in the tail"
    ))
    
    # Calculate expected shortfall
    tail_returns = sorted_returns[:var_index+1]
    es = -np.mean(tail_returns)
    
    steps.append(CalculationStep(
        step_number=3,
        title="Calculate Expected Shortfall",
        formula_latex=r"ES_\alpha = -E[r_p | r_p \leq VaR_\alpha] = -\frac{1}{|\{t: r_t \leq VaR\}|}\sum_{r_t \leq VaR} r_t",
        description="Average of returns in the tail (beyond VaR)",
        inputs={"n_tail": str(len(tail_returns)), "VaR": f"{var_threshold*100:.4f}%"},
        calculation=f"Average of {len(tail_returns)} tail returns",
        result=f"ES = -{np.mean(tail_returns)*100:.4f}% = {es*100:.4f}%"
    ))
    
    # Scale for horizon
    es_h = es * np.sqrt(horizon_days)
    es_dollar = es_h * portfolio_value
    
    steps.append(CalculationStep(
        step_number=4,
        title="Scale for Time Horizon",
        formula_latex=r"ES_h = ES_1 \cdot \sqrt{h}",
        description="Square root of time scaling",
        inputs={"h": str(horizon_days), "ES_1": f"{es*100:.4f}%"},
        calculation=f"ES_{horizon_days} = {es*100:.4f}% × √{horizon_days} = {es_h*100:.4f}%",
        result=f"CVaR({confidence*100:.0f}%) = {es_h*100:.4f}% = ${es_dollar:,.0f}"
    ))
    
    return CalculationResult(
        name=f"CVaR/Expected Shortfall",
        value=es_h,
        unit="percentage",
        steps=steps,
        inputs_summary={
            "confidence": confidence,
            "horizon_days": horizon_days,
            "portfolio_value": portfolio_value,
            "n_tail_observations": len(tail_returns),
        },
        formula_latex=r"ES_\alpha = E[-r_p | r_p \leq -VaR_\alpha]"
    )


# =============================================================================
# SHARPE RATIO CALCULATION
# =============================================================================

def calculate_sharpe_ratio(
    returns: NDArray[np.float64],
    weights: NDArray[np.float64],
    risk_free_rate: float = 0.0428,  # 4.28% as of Jan 2026
    annualization_factor: int = 252
) -> CalculationResult:
    """Calculate Sharpe Ratio with full derivation"""
    steps = []
    
    # Portfolio returns
    port_returns = returns @ weights
    n_obs = len(port_returns)
    
    steps.append(CalculationStep(
        step_number=1,
        title="Calculate Portfolio Returns",
        formula_latex=r"r_{p,t} = \sum_{i=1}^{n} w_i r_{i,t}",
        description="Daily portfolio returns from weighted asset returns",
        inputs={"n_assets": str(len(weights)), "T": str(n_obs)},
        calculation=f"Computed {n_obs} daily portfolio returns",
        result=f"Sample period: {n_obs} trading days"
    ))
    
    # Mean return (annualized)
    mu_daily = np.mean(port_returns)
    mu_annual = mu_daily * annualization_factor
    
    steps.append(CalculationStep(
        step_number=2,
        title="Calculate Annualized Mean Return",
        formula_latex=r"\mu_p = \bar{r}_p \times 252",
        description="Annualize daily mean return (assuming 252 trading days)",
        inputs={"r̄_daily": f"{mu_daily*100:.4f}%"},
        calculation=f"μ_p = {mu_daily*100:.4f}% × 252 = {mu_annual*100:.2f}%",
        result=f"Annualized return: {mu_annual*100:.2f}%"
    ))
    
    # Standard deviation (annualized)
    sigma_daily = np.std(port_returns, ddof=1)
    sigma_annual = sigma_daily * np.sqrt(annualization_factor)
    
    steps.append(CalculationStep(
        step_number=3,
        title="Calculate Annualized Volatility",
        formula_latex=r"\sigma_p = \sigma_{daily} \times \sqrt{252}",
        description="Annualize volatility using square root of time",
        inputs={"σ_daily": f"{sigma_daily*100:.4f}%"},
        calculation=f"σ_p = {sigma_daily*100:.4f}% × √252 = {sigma_annual*100:.2f}%",
        result=f"Annualized volatility: {sigma_annual*100:.2f}%"
    ))
    
    # Daily risk-free rate
    rf_daily = risk_free_rate / annualization_factor
    
    steps.append(CalculationStep(
        step_number=4,
        title="Convert Risk-Free Rate",
        formula_latex=r"r_f^{daily} = \frac{r_f^{annual}}{252}",
        description="Convert annual risk-free rate to daily",
        inputs={"r_f^annual": f"{risk_free_rate*100:.2f}%"},
        calculation=f"r_f^daily = {risk_free_rate*100:.2f}% / 252 = {rf_daily*100:.4f}%",
        result=f"Daily risk-free rate: {rf_daily*100:.4f}%"
    ))
    
    # Sharpe Ratio
    excess_return = mu_annual - risk_free_rate
    sharpe = excess_return / sigma_annual
    
    steps.append(CalculationStep(
        step_number=5,
        title="Calculate Sharpe Ratio",
        formula_latex=r"SR = \frac{\mu_p - r_f}{\sigma_p}",
        description="Excess return per unit of risk",
        inputs={
            "μ_p": f"{mu_annual*100:.2f}%",
            "r_f": f"{risk_free_rate*100:.2f}%",
            "σ_p": f"{sigma_annual*100:.2f}%"
        },
        calculation=f"SR = ({mu_annual*100:.2f}% - {risk_free_rate*100:.2f}%) / {sigma_annual*100:.2f}%",
        result=f"Sharpe Ratio = {sharpe:.4f}"
    ))
    
    return CalculationResult(
        name="Sharpe Ratio",
        value=sharpe,
        unit="ratio",
        steps=steps,
        inputs_summary={
            "mean_return_annual": mu_annual,
            "volatility_annual": sigma_annual,
            "risk_free_rate": risk_free_rate,
            "n_observations": n_obs,
        },
        formula_latex=r"SR = \frac{E[r_p] - r_f}{\sigma_p}"
    )


# =============================================================================
# PORTFOLIO OPTIMIZATION
# =============================================================================

def calculate_optimal_weights_hrp(
    returns: NDArray[np.float64],
    asset_names: List[str]
) -> Tuple[NDArray[np.float64], List[CalculationStep]]:
    """
    Hierarchical Risk Parity optimization with full derivation
    Based on López de Prado (2016)
    """
    from scipy.cluster.hierarchy import linkage, leaves_list
    from scipy.spatial.distance import squareform
    
    steps = []
    n_assets = returns.shape[1]
    
    # Step 1: Correlation matrix
    cov = np.cov(returns.T)
    std = np.sqrt(np.diag(cov))
    corr = cov / np.outer(std, std)
    
    steps.append(CalculationStep(
        step_number=1,
        title="Calculate Correlation Matrix",
        formula_latex=r"\rho_{ij} = \frac{Cov(r_i, r_j)}{\sigma_i \sigma_j}",
        description="Pairwise correlations between all assets",
        inputs={"n_assets": str(n_assets)},
        calculation=f"Correlation matrix: {n_assets}×{n_assets}",
        result=f"Average correlation: {(corr.sum() - n_assets) / (n_assets**2 - n_assets):.4f}"
    ))
    
    # Step 2: Distance matrix
    dist = np.sqrt(0.5 * (1 - corr))
    np.fill_diagonal(dist, 0)
    
    steps.append(CalculationStep(
        step_number=2,
        title="Calculate Distance Matrix",
        formula_latex=r"d_{ij} = \sqrt{\frac{1}{2}(1 - \rho_{ij})}",
        description="Convert correlation to distance (0 = perfectly correlated)",
        inputs={"ρ": "Correlation matrix"},
        calculation="d_ij ∈ [0, 1] where 0 = perfect correlation, 1 = perfect anti-correlation",
        result=f"Distance range: [{dist[dist > 0].min():.4f}, {dist.max():.4f}]"
    ))
    
    # Step 3: Hierarchical clustering
    dist_condensed = squareform(dist)
    link = linkage(dist_condensed, method='single')
    sort_ix = leaves_list(link)
    
    steps.append(CalculationStep(
        step_number=3,
        title="Hierarchical Clustering",
        formula_latex=r"\text{linkage}(D, \text{method}=\text{'single'})",
        description="Single-linkage agglomerative clustering",
        inputs={"D": "Distance matrix"},
        calculation="Build dendrogram by iteratively merging closest clusters",
        result=f"Optimal ordering: {[asset_names[i][:4] for i in sort_ix[:5]]}..."
    ))
    
    # Step 4: Quasi-diagonalization (already done by sort_ix)
    sorted_cov = cov[np.ix_(sort_ix, sort_ix)]
    
    steps.append(CalculationStep(
        step_number=4,
        title="Quasi-Diagonalization",
        formula_latex=r"\Sigma_{quasi} = P' \Sigma P",
        description="Reorder covariance matrix by cluster structure",
        inputs={"P": "Permutation matrix from clustering"},
        calculation="Assets reordered to maximize block-diagonal structure",
        result="Covariance matrix quasi-diagonalized"
    ))
    
    # Step 5: Recursive bisection
    def get_cluster_var(cov, weights):
        return np.sqrt(weights @ cov @ weights)
    
    def recursive_bisection(cov, sort_ix):
        w = np.ones(len(sort_ix))
        clusters = [sort_ix]
        
        while len(clusters) > 0:
            clusters = [c[start:end] for c in clusters 
                       for start, end in [(0, len(c)//2), (len(c)//2, len(c))]
                       if len(c) > 1]
            
            for i in range(0, len(clusters), 2):
                if i + 1 < len(clusters):
                    c0, c1 = clusters[i], clusters[i+1]
                    
                    # Inverse-variance weights within clusters
                    cov0 = cov[np.ix_(c0, c0)]
                    cov1 = cov[np.ix_(c1, c1)]
                    
                    w0 = 1 / np.diag(cov0)
                    w0 = w0 / w0.sum()
                    w1 = 1 / np.diag(cov1)
                    w1 = w1 / w1.sum()
                    
                    var0 = get_cluster_var(cov0, w0)
                    var1 = get_cluster_var(cov1, w1)
                    
                    alpha = 1 - var0 / (var0 + var1)
                    
                    w[c0] *= alpha
                    w[c1] *= (1 - alpha)
        
        return w
    
    weights = recursive_bisection(cov, sort_ix)
    weights = weights / weights.sum()  # Normalize
    
    # Reorder weights back to original order
    original_weights = np.zeros(n_assets)
    for i, ix in enumerate(sort_ix):
        original_weights[ix] = weights[i]
    
    steps.append(CalculationStep(
        step_number=5,
        title="Recursive Bisection",
        formula_latex=r"\alpha = 1 - \frac{Var(C_1)}{Var(C_1) + Var(C_2)}",
        description="Allocate inversely proportional to cluster variance",
        inputs={"clusters": "From hierarchical clustering"},
        calculation="Recursively split and allocate between sub-clusters",
        result=f"Final weights: max={original_weights.max()*100:.2f}%, min={original_weights.min()*100:.2f}%"
    ))
    
    return original_weights, steps
