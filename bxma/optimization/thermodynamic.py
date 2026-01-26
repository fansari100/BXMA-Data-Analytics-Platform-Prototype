"""
Thermodynamic Portfolio Optimization (THRML)
============================================

Physics-inspired portfolio optimization using Ising Hamiltonian formulation
and thermodynamic sampling. This approach treats portfolios as energy states
and uses Boltzmann sampling to explore the solution space.

Key Concepts:
- Assets as "spins" in a magnetic system
- Correlations as magnetic couplings (J_ij)
- Portfolio energy = -λ₁·Return + λ₂·Risk + λ₃·Constraints
- Adaptive temperature scales with market volatility (VIX)
- Block Gibbs Sampling for GPU-accelerated exploration

References:
- Venturelli et al. (2019): Quantum-inspired optimization
- Khoshaman et al. (2020): Portfolio optimization with quantum annealing
- Nature Physics (2025): Thermodynamic machine learning

Author: BXMA Quant Team
Date: January 2026
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass, field
from typing import Literal, Callable
from enum import Enum, auto
import time


@dataclass
class ThermodynamicConfig:
    """Configuration for thermodynamic optimization."""
    
    # Temperature schedule
    initial_temperature: float = 10.0
    final_temperature: float = 0.01
    cooling_rate: float = 0.995  # Geometric cooling
    
    # Sampling parameters
    n_samples: int = 10000
    burn_in: int = 1000
    thin: int = 10
    
    # Block Gibbs parameters
    block_size: int = 10  # Number of assets per block
    n_sweeps_per_sample: int = 5
    
    # Adaptive temperature (VIX-linked)
    adaptive_temperature: bool = True
    vix_baseline: float = 15.0  # Temperature multiplier at this VIX
    vix_sensitivity: float = 0.05  # How much temp increases per VIX point
    
    # Regularization
    entropy_weight: float = 0.01  # Encourage diversification
    sparsity_penalty: float = 0.0  # L1 penalty on weights
    
    # Constraints
    max_position_size: float = 0.20
    min_position_size: float = 0.01
    max_assets: int | None = None  # Cardinality constraint
    sector_limits: dict[str, float] = field(default_factory=dict)


class AnnealingSchedule(Enum):
    """Temperature annealing schedules."""
    GEOMETRIC = auto()
    LINEAR = auto()
    LOGARITHMIC = auto()
    ADAPTIVE = auto()
    CAUCHY = auto()


@dataclass
class ThermodynamicResult:
    """Result of thermodynamic optimization."""
    
    # Optimal portfolio
    weights: NDArray[np.float64]
    expected_return: float
    expected_risk: float
    sharpe_ratio: float
    
    # Energy landscape
    final_energy: float
    ground_state_energy: float
    energy_variance: float
    
    # Thermodynamic properties
    final_temperature: float
    entropy: float
    free_energy: float
    
    # Sampling statistics
    acceptance_rate: float
    n_samples: int
    effective_sample_size: float
    
    # Portfolio characteristics
    n_active_positions: int
    herfindahl_index: float
    max_weight: float
    
    # Performance
    solve_time_ms: float
    
    # Posterior distribution (cloud of portfolios)
    portfolio_samples: NDArray[np.float64] | None = None
    energy_samples: NDArray[np.float64] | None = None


class IsingHamiltonian:
    """
    Ising Hamiltonian formulation for portfolio optimization.
    
    The portfolio energy is defined as:
    H(w) = -λ_ret · w'μ + λ_risk · w'Σw + λ_cons · Violations(w)
    
    Where correlations act as magnetic couplings J_ij = Σ_ij
    """
    
    def __init__(
        self,
        expected_returns: NDArray[np.float64],
        covariance: NDArray[np.float64],
        lambda_return: float = 1.0,
        lambda_risk: float = 1.0,
        lambda_constraint: float = 100.0,
    ):
        self.mu = expected_returns
        self.sigma = covariance
        self.n_assets = len(expected_returns)
        
        # Lagrange multipliers (energy weights)
        self.lambda_return = lambda_return
        self.lambda_risk = lambda_risk
        self.lambda_constraint = lambda_constraint
        
        # Precompute correlation-based couplings
        vols = np.sqrt(np.diag(covariance))
        self.correlation = covariance / np.outer(vols, vols)
        self.couplings = self.correlation  # J_ij
        
        # Constraint functions
        self.constraints: list[Callable[[NDArray], float]] = []
    
    def add_constraint(self, constraint_fn: Callable[[NDArray], float]):
        """Add a constraint function that returns violation magnitude."""
        self.constraints.append(constraint_fn)
    
    def energy(self, weights: NDArray[np.float64]) -> float:
        """
        Compute Hamiltonian energy of a portfolio configuration.
        
        Lower energy = better portfolio
        """
        # Return term (negative because we maximize returns)
        return_energy = -self.lambda_return * (weights @ self.mu)
        
        # Risk term (positive because we minimize risk)
        risk_energy = self.lambda_risk * (weights @ self.sigma @ weights)
        
        # Constraint violation term
        constraint_energy = 0.0
        for constraint_fn in self.constraints:
            violation = constraint_fn(weights)
            constraint_energy += self.lambda_constraint * violation ** 2
        
        return return_energy + risk_energy + constraint_energy
    
    def gradient(self, weights: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Compute energy gradient for continuous relaxation.
        """
        grad_return = -self.lambda_return * self.mu
        grad_risk = 2 * self.lambda_risk * (self.sigma @ weights)
        return grad_return + grad_risk
    
    def local_field(self, weights: NDArray[np.float64], i: int) -> float:
        """
        Compute local field at site i (for Gibbs sampling).
        
        h_i = ∂H/∂w_i
        """
        h_return = -self.lambda_return * self.mu[i]
        h_risk = 2 * self.lambda_risk * (self.sigma[i, :] @ weights)
        return h_return + h_risk


class BlockGibbsSampler:
    """
    GPU-accelerated Block Gibbs Sampler for thermodynamic optimization.
    
    Updates blocks of assets jointly to improve mixing in correlated systems.
    """
    
    def __init__(
        self,
        hamiltonian: IsingHamiltonian,
        config: ThermodynamicConfig,
    ):
        self.H = hamiltonian
        self.config = config
        self.n_assets = hamiltonian.n_assets
        
        # Initialize state
        self.weights = np.ones(self.n_assets) / self.n_assets
        self.temperature = config.initial_temperature
        
        # Statistics
        self.accepted = 0
        self.proposed = 0
        self.energy_history: list[float] = []
    
    def _boltzmann_probability(self, delta_energy: float) -> float:
        """Compute Boltzmann acceptance probability."""
        if delta_energy <= 0:
            return 1.0
        return np.exp(-delta_energy / self.temperature)
    
    def _propose_block_update(
        self,
        block_indices: list[int],
    ) -> NDArray[np.float64]:
        """
        Propose a new configuration for a block of assets.
        
        Uses a Metropolis-within-Gibbs approach with Dirichlet proposals.
        """
        current_block_weights = self.weights[block_indices]
        
        # Dirichlet proposal centered on current weights
        concentration = 100 * current_block_weights + 0.1  # Ensure positive
        proposed_block = np.random.dirichlet(concentration)
        
        # Scale to maintain total block weight
        block_sum = current_block_weights.sum()
        proposed_block = proposed_block * block_sum
        
        return proposed_block
    
    def _sweep(self):
        """
        Perform one sweep over all blocks.
        """
        # Randomize block assignment for this sweep
        indices = np.random.permutation(self.n_assets)
        n_blocks = max(1, self.n_assets // self.config.block_size)
        
        for block_idx in range(n_blocks):
            start = block_idx * self.config.block_size
            end = min(start + self.config.block_size, self.n_assets)
            block_indices = indices[start:end].tolist()
            
            # Current energy
            current_energy = self.H.energy(self.weights)
            
            # Propose new block configuration
            old_block = self.weights[block_indices].copy()
            proposed_block = self._propose_block_update(block_indices)
            
            # Apply proposal
            proposed_weights = self.weights.copy()
            proposed_weights[block_indices] = proposed_block
            
            # Renormalize to sum to 1
            proposed_weights = proposed_weights / proposed_weights.sum()
            proposed_weights = np.clip(proposed_weights, 0, self.config.max_position_size)
            proposed_weights = proposed_weights / proposed_weights.sum()
            
            # Compute new energy
            proposed_energy = self.H.energy(proposed_weights)
            delta_energy = proposed_energy - current_energy
            
            # Metropolis acceptance
            self.proposed += 1
            if np.random.random() < self._boltzmann_probability(delta_energy):
                self.weights = proposed_weights
                self.accepted += 1
    
    def _cool(self, schedule: AnnealingSchedule, iteration: int):
        """Apply cooling schedule."""
        if schedule == AnnealingSchedule.GEOMETRIC:
            self.temperature *= self.config.cooling_rate
        elif schedule == AnnealingSchedule.LINEAR:
            total_iters = self.config.n_samples + self.config.burn_in
            self.temperature = self.config.initial_temperature * (1 - iteration / total_iters)
        elif schedule == AnnealingSchedule.LOGARITHMIC:
            self.temperature = self.config.initial_temperature / np.log(2 + iteration)
        elif schedule == AnnealingSchedule.CAUCHY:
            self.temperature = self.config.initial_temperature / (1 + iteration)
        
        self.temperature = max(self.temperature, self.config.final_temperature)
    
    def sample(
        self,
        vix: float | None = None,
        schedule: AnnealingSchedule = AnnealingSchedule.GEOMETRIC,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Run MCMC sampling to explore portfolio space.
        
        Args:
            vix: Current VIX level for adaptive temperature
            schedule: Temperature annealing schedule
            
        Returns:
            Tuple of (portfolio_samples, energy_samples)
        """
        # Adaptive temperature based on VIX
        if self.config.adaptive_temperature and vix is not None:
            temp_multiplier = 1 + self.config.vix_sensitivity * (vix - self.config.vix_baseline)
            self.temperature = self.config.initial_temperature * max(0.5, temp_multiplier)
        
        portfolio_samples = []
        energy_samples = []
        
        total_iterations = self.config.burn_in + self.config.n_samples * self.config.thin
        
        for iteration in range(total_iterations):
            # Perform sweeps
            for _ in range(self.config.n_sweeps_per_sample):
                self._sweep()
            
            # Cool the system
            self._cool(schedule, iteration)
            
            # Record energy
            current_energy = self.H.energy(self.weights)
            self.energy_history.append(current_energy)
            
            # Collect samples after burn-in, with thinning
            if iteration >= self.config.burn_in:
                if (iteration - self.config.burn_in) % self.config.thin == 0:
                    portfolio_samples.append(self.weights.copy())
                    energy_samples.append(current_energy)
        
        return np.array(portfolio_samples), np.array(energy_samples)


class ThermodynamicOptimizer:
    """
    Main thermodynamic portfolio optimizer.
    
    Implements the THRML (Thermodynamic Hypergraphical Model Library) approach
    for robust, physics-inspired portfolio construction.
    """
    
    def __init__(self, config: ThermodynamicConfig | None = None):
        self.config = config or ThermodynamicConfig()
    
    def optimize(
        self,
        expected_returns: NDArray[np.float64],
        covariance: NDArray[np.float64],
        vix: float | None = None,
        risk_aversion: float = 1.0,
        sector_assignments: list[str] | None = None,
        return_samples: bool = False,
    ) -> ThermodynamicResult:
        """
        Find optimal portfolio using thermodynamic sampling.
        
        Args:
            expected_returns: Expected return vector (n_assets,)
            covariance: Covariance matrix (n_assets, n_assets)
            vix: Current VIX level for adaptive temperature
            risk_aversion: Risk aversion parameter (λ_risk / λ_return)
            sector_assignments: Sector labels for each asset
            return_samples: Whether to return the full posterior
            
        Returns:
            ThermodynamicResult with optimal portfolio and statistics
        """
        start_time = time.time()
        n_assets = len(expected_returns)
        
        # Normalize for numerical stability
        mu_scale = np.std(expected_returns) if np.std(expected_returns) > 0 else 1
        sigma_scale = np.sqrt(np.mean(np.diag(covariance)))
        
        mu_normalized = expected_returns / mu_scale
        sigma_normalized = covariance / (sigma_scale ** 2)
        
        # Create Hamiltonian
        hamiltonian = IsingHamiltonian(
            expected_returns=mu_normalized,
            covariance=sigma_normalized,
            lambda_return=1.0,
            lambda_risk=risk_aversion,
            lambda_constraint=100.0,
        )
        
        # Add constraints
        # Sum to 1
        hamiltonian.add_constraint(lambda w: abs(w.sum() - 1.0))
        
        # Long only
        hamiltonian.add_constraint(lambda w: np.sum(np.maximum(-w, 0)))
        
        # Max position
        hamiltonian.add_constraint(
            lambda w: np.sum(np.maximum(w - self.config.max_position_size, 0))
        )
        
        # Cardinality constraint
        if self.config.max_assets is not None:
            hamiltonian.add_constraint(
                lambda w: max(0, np.sum(w > self.config.min_position_size) - self.config.max_assets)
            )
        
        # Sector constraints
        if sector_assignments and self.config.sector_limits:
            for sector, limit in self.config.sector_limits.items():
                sector_mask = np.array([s == sector for s in sector_assignments])
                hamiltonian.add_constraint(
                    lambda w, m=sector_mask, l=limit: max(0, w[m].sum() - l)
                )
        
        # Initialize sampler
        sampler = BlockGibbsSampler(hamiltonian, self.config)
        
        # Run sampling
        portfolio_samples, energy_samples = sampler.sample(
            vix=vix,
            schedule=AnnealingSchedule.ADAPTIVE if self.config.adaptive_temperature else AnnealingSchedule.GEOMETRIC
        )
        
        # Find ground state (minimum energy portfolio)
        ground_state_idx = np.argmin(energy_samples)
        optimal_weights = portfolio_samples[ground_state_idx]
        
        # Ensure constraints are satisfied
        optimal_weights = np.clip(optimal_weights, 0, self.config.max_position_size)
        optimal_weights = optimal_weights / optimal_weights.sum()
        
        # Clean up small positions
        optimal_weights[optimal_weights < self.config.min_position_size] = 0
        if optimal_weights.sum() > 0:
            optimal_weights = optimal_weights / optimal_weights.sum()
        
        # Compute portfolio metrics
        exp_ret = float(expected_returns @ optimal_weights)
        exp_risk = float(np.sqrt(optimal_weights @ covariance @ optimal_weights))
        sharpe = exp_ret / exp_risk if exp_risk > 0 else 0
        
        # Thermodynamic properties
        final_energy = hamiltonian.energy(optimal_weights)
        ground_state_energy = float(np.min(energy_samples))
        energy_variance = float(np.var(energy_samples))
        
        # Entropy (from Boltzmann distribution)
        boltzmann_weights = np.exp(-energy_samples / sampler.temperature)
        boltzmann_weights = boltzmann_weights / boltzmann_weights.sum()
        entropy = -float(np.sum(boltzmann_weights * np.log(boltzmann_weights + 1e-10)))
        
        # Free energy: F = E - TS
        mean_energy = float(np.mean(energy_samples))
        free_energy = mean_energy - sampler.temperature * entropy
        
        # Portfolio characteristics
        n_active = int(np.sum(optimal_weights > self.config.min_position_size))
        herfindahl = float(np.sum(optimal_weights ** 2))
        
        # Effective sample size (accounting for autocorrelation)
        ess = self._effective_sample_size(energy_samples)
        
        solve_time = (time.time() - start_time) * 1000
        
        return ThermodynamicResult(
            weights=optimal_weights,
            expected_return=exp_ret,
            expected_risk=exp_risk,
            sharpe_ratio=sharpe,
            final_energy=final_energy,
            ground_state_energy=ground_state_energy,
            energy_variance=energy_variance,
            final_temperature=sampler.temperature,
            entropy=entropy,
            free_energy=free_energy,
            acceptance_rate=sampler.accepted / max(1, sampler.proposed),
            n_samples=len(portfolio_samples),
            effective_sample_size=ess,
            n_active_positions=n_active,
            herfindahl_index=herfindahl,
            max_weight=float(np.max(optimal_weights)),
            solve_time_ms=solve_time,
            portfolio_samples=portfolio_samples if return_samples else None,
            energy_samples=energy_samples if return_samples else None,
        )
    
    def _effective_sample_size(self, samples: NDArray[np.float64]) -> float:
        """
        Compute effective sample size accounting for autocorrelation.
        
        ESS = n / (1 + 2 * sum(autocorrelations))
        """
        n = len(samples)
        if n < 10:
            return float(n)
        
        # Compute autocorrelation up to lag 50
        max_lag = min(50, n // 4)
        acf = np.correlate(samples - np.mean(samples), samples - np.mean(samples), mode='full')
        acf = acf[n-1:n+max_lag] / acf[n-1]
        
        # Sum autocorrelations until they become negligible
        tau = 1 + 2 * np.sum(acf[1:])
        
        return n / max(1, tau)
    
    def sample_portfolio_cloud(
        self,
        expected_returns: NDArray[np.float64],
        covariance: NDArray[np.float64],
        n_portfolios: int = 100,
        vix: float | None = None,
    ) -> NDArray[np.float64]:
        """
        Generate a cloud of near-optimal portfolios.
        
        Useful for robust portfolio construction and uncertainty quantification.
        
        Returns:
            Array of shape (n_portfolios, n_assets)
        """
        result = self.optimize(
            expected_returns=expected_returns,
            covariance=covariance,
            vix=vix,
            return_samples=True,
        )
        
        if result.portfolio_samples is None:
            return result.weights.reshape(1, -1)
        
        # Return the lowest-energy portfolios
        if result.energy_samples is not None:
            sorted_idx = np.argsort(result.energy_samples)[:n_portfolios]
            return result.portfolio_samples[sorted_idx]
        
        return result.portfolio_samples[:n_portfolios]


class AdaptiveTemperatureController:
    """
    VIX-linked adaptive temperature controller.
    
    In high-volatility regimes (high VIX), the system operates at higher
    temperature, exploring more diverse portfolios. In low-volatility
    regimes, it "freezes" into deterministic optima.
    """
    
    def __init__(
        self,
        baseline_vix: float = 15.0,
        sensitivity: float = 0.05,
        min_temp_multiplier: float = 0.5,
        max_temp_multiplier: float = 3.0,
    ):
        self.baseline_vix = baseline_vix
        self.sensitivity = sensitivity
        self.min_mult = min_temp_multiplier
        self.max_mult = max_temp_multiplier
    
    def get_temperature_multiplier(self, vix: float) -> float:
        """
        Compute temperature multiplier based on VIX.
        
        T_effective = T_base * multiplier
        """
        multiplier = 1 + self.sensitivity * (vix - self.baseline_vix)
        return np.clip(multiplier, self.min_mult, self.max_mult)
    
    def get_regime_entropy(self, vix: float) -> float:
        """
        Estimate regime entropy (exploration vs exploitation).
        
        High entropy = high exploration (volatile markets)
        Low entropy = exploitation (calm markets)
        """
        mult = self.get_temperature_multiplier(vix)
        # Map multiplier to entropy [0, 1]
        return (mult - self.min_mult) / (self.max_mult - self.min_mult)


# Convenience function
def thermodynamic_optimize(
    expected_returns: NDArray[np.float64],
    covariance: NDArray[np.float64],
    vix: float | None = None,
    risk_aversion: float = 1.0,
    max_position: float = 0.20,
    max_assets: int | None = None,
) -> ThermodynamicResult:
    """
    Convenience function for thermodynamic portfolio optimization.
    
    Args:
        expected_returns: Expected return vector
        covariance: Covariance matrix
        vix: Current VIX level
        risk_aversion: Risk aversion parameter
        max_position: Maximum weight per asset
        max_assets: Maximum number of positions
        
    Returns:
        ThermodynamicResult
    """
    config = ThermodynamicConfig(
        max_position_size=max_position,
        max_assets=max_assets,
        adaptive_temperature=vix is not None,
    )
    
    optimizer = ThermodynamicOptimizer(config)
    return optimizer.optimize(
        expected_returns=expected_returns,
        covariance=covariance,
        vix=vix,
        risk_aversion=risk_aversion,
    )
