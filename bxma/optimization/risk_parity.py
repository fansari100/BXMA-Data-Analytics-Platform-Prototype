"""
Risk Parity and Hierarchical Optimization for BXMA Risk/Quant Platform.

Implements advanced risk-based allocation approaches:
- Equal Risk Contribution (ERC)
- Hierarchical Risk Parity (HRP)
- Nested Clustered Optimization (NCO)
- Factor Risk Parity

These methods are robust to estimation error and perform well out-of-sample.

References:
- "Risk Parity Portfolios" (Maillard, Roncalli & Teiletche, 2010)
- "Building Diversified Portfolios that Outperform Out-of-Sample"
  (Lopez de Prado, 2016)
- "A Robust Estimator of the Efficient Frontier" (Lopez de Prado, 2019)
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
from scipy import optimize
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform
import warnings

from bxma.optimization.classical import (
    PortfolioOptimizer,
    OptimizationConstraints,
    OptimizationResult,
)


class RiskParityOptimizer(PortfolioOptimizer):
    """
    Equal Risk Contribution (ERC) / Risk Parity Optimizer.
    
    Allocates weights such that each asset contributes equally
    to portfolio risk. No reliance on expected returns.
    
    Risk contribution_i = w_i * (Σw)_i / σ_p
    
    Reference:
    - "On the Properties of Equally-Weighted Risk Contributions Portfolios"
      (Maillard, Roncalli & Teiletche, 2010)
    """
    
    def __init__(
        self,
        target_risk_contributions: NDArray[np.float64] | None = None,
        method: str = "slsqp",
        **kwargs
    ):
        """
        Initialize Risk Parity optimizer.
        
        Args:
            target_risk_contributions: Target risk budgets (default: equal)
            method: Optimization method ('slsqp', 'ccd', 'newton')
        """
        super().__init__(**kwargs)
        self.target_risk_contributions = target_risk_contributions
        self.method = method
    
    def optimize(
        self,
        expected_returns: NDArray[np.float64],
        covariance: NDArray[np.float64],
        constraints: OptimizationConstraints | None = None,
    ) -> OptimizationResult:
        """Find risk parity portfolio."""
        import time
        start_time = time.time()
        
        n_assets = len(expected_returns)
        constraints = constraints or OptimizationConstraints()
        
        # Target risk contributions
        if self.target_risk_contributions is None:
            b = np.ones(n_assets) / n_assets  # Equal risk contribution
        else:
            b = self.target_risk_contributions / self.target_risk_contributions.sum()
        
        if self.method == "ccd":
            # Cyclical Coordinate Descent (fast for large portfolios)
            weights = self._ccd_risk_parity(covariance, b)
        else:
            # SLSQP optimization
            weights = self._slsqp_risk_parity(covariance, b, constraints)
        
        solve_time = (time.time() - start_time) * 1000
        
        # Compute metrics
        exp_ret = float(expected_returns @ weights)
        exp_risk = float(np.sqrt(weights @ covariance @ weights))
        sharpe = exp_ret / exp_risk if exp_risk > 0 else 0
        
        risk_contrib = self._compute_risk_contributions(weights, covariance)
        
        return OptimizationResult(
            weights=weights,
            expected_return=exp_ret,
            expected_risk=exp_risk,
            sharpe_ratio=sharpe,
            status="optimal",
            optimal=True,
            solve_time_ms=solve_time,
            risk_contributions=risk_contrib,
        )
    
    def _slsqp_risk_parity(
        self,
        cov: NDArray[np.float64],
        b: NDArray[np.float64],
        constraints: OptimizationConstraints,
    ) -> NDArray[np.float64]:
        """SLSQP-based risk parity optimization."""
        n = len(b)
        
        def risk_budget_objective(w):
            """Sum of squared differences from target risk contributions."""
            port_var = w @ cov @ w
            if port_var <= 0:
                return 1e10
            
            port_std = np.sqrt(port_var)
            marginal = cov @ w / port_std
            risk_contrib = w * marginal
            risk_contrib_pct = risk_contrib / port_std
            
            return np.sum((risk_contrib_pct - b) ** 2)
        
        # Constraints
        cons = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        
        bounds = [(constraints.min_weight, constraints.max_weight) for _ in range(n)]
        
        # Initial guess: inverse volatility
        vols = np.sqrt(np.diag(cov))
        w0 = (1 / vols) / np.sum(1 / vols)
        
        result = optimize.minimize(
            risk_budget_objective,
            w0,
            method='SLSQP',
            bounds=bounds,
            constraints=cons,
            options={'maxiter': 1000, 'ftol': 1e-10}
        )
        
        weights = result.x
        weights = np.maximum(weights, 0)
        weights = weights / weights.sum()
        
        return weights
    
    def _ccd_risk_parity(
        self,
        cov: NDArray[np.float64],
        b: NDArray[np.float64],
        max_iter: int = 1000,
        tol: float = 1e-8,
    ) -> NDArray[np.float64]:
        """
        Cyclical Coordinate Descent for risk parity.
        
        Fast algorithm from Griveau-Billion et al. (2013).
        """
        n = len(b)
        
        # Initialize with inverse volatility
        vols = np.sqrt(np.diag(cov))
        w = (1 / vols) / np.sum(1 / vols)
        
        for iteration in range(max_iter):
            w_old = w.copy()
            
            for i in range(n):
                # Compute terms
                sigma_i = np.sqrt(cov[i, i])
                sum_j = sum(w[j] * cov[i, j] for j in range(n) if j != i)
                
                # Analytical solution
                a = cov[i, i]
                c = sum_j
                
                # Solve quadratic: a*w_i^2 + c*w_i - b_i*sqrt(w'Σw) = 0
                # Approximation: w_i = (-c + sqrt(c^2 + 4*a*b_i*σ_p)) / (2*a)
                port_std = np.sqrt(w @ cov @ w)
                
                discriminant = c**2 + 4 * a * b[i] * port_std
                if discriminant > 0:
                    w[i] = (-c + np.sqrt(discriminant)) / (2 * a)
                else:
                    w[i] = 0
                
                w[i] = max(w[i], 1e-10)
            
            # Normalize
            w = w / w.sum()
            
            # Check convergence
            if np.max(np.abs(w - w_old)) < tol:
                break
        
        return w


class HierarchicalRiskParity(PortfolioOptimizer):
    """
    Hierarchical Risk Parity (HRP).
    
    Machine learning approach to portfolio construction that:
    1. Clusters assets using hierarchical clustering on correlation
    2. Applies quasi-diagonalization to order assets
    3. Recursively bisects portfolio using inverse-variance allocation
    
    Benefits:
    - No need to invert covariance matrix
    - Robust to estimation error
    - No expected return estimates required
    - Out-of-sample performance typically better than Markowitz
    
    Reference:
    - "Building Diversified Portfolios that Outperform Out-of-Sample"
      (Lopez de Prado, 2016)
    """
    
    def __init__(
        self,
        linkage_method: str = "ward",
        **kwargs
    ):
        """
        Initialize HRP optimizer.
        
        Args:
            linkage_method: Hierarchical clustering linkage method
                ('ward', 'single', 'complete', 'average')
        """
        super().__init__(**kwargs)
        self.linkage_method = linkage_method
    
    def optimize(
        self,
        expected_returns: NDArray[np.float64],
        covariance: NDArray[np.float64],
        constraints: OptimizationConstraints | None = None,
    ) -> OptimizationResult:
        """Find HRP portfolio."""
        import time
        start_time = time.time()
        
        n_assets = len(expected_returns)
        
        # Step 1: Correlation-based distance matrix
        corr = self._cov_to_corr(covariance)
        dist = self._correlation_distance(corr)
        
        # Step 2: Hierarchical clustering
        link = linkage(squareform(dist), method=self.linkage_method)
        
        # Step 3: Quasi-diagonalization (reorder assets)
        sorted_idx = self._get_quasi_diag(link)
        
        # Step 4: Recursive bisection
        weights = self._recursive_bisection(covariance, sorted_idx)
        
        solve_time = (time.time() - start_time) * 1000
        
        # Compute metrics
        exp_ret = float(expected_returns @ weights)
        exp_risk = float(np.sqrt(weights @ covariance @ weights))
        sharpe = exp_ret / exp_risk if exp_risk > 0 else 0
        
        risk_contrib = self._compute_risk_contributions(weights, covariance)
        
        return OptimizationResult(
            weights=weights,
            expected_return=exp_ret,
            expected_risk=exp_risk,
            sharpe_ratio=sharpe,
            status="optimal",
            optimal=True,
            solve_time_ms=solve_time,
            risk_contributions=risk_contrib,
        )
    
    def _cov_to_corr(self, cov: NDArray[np.float64]) -> NDArray[np.float64]:
        """Convert covariance to correlation matrix."""
        std = np.sqrt(np.diag(cov))
        corr = cov / np.outer(std, std)
        corr = np.clip(corr, -1, 1)
        return corr
    
    def _correlation_distance(self, corr: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Compute correlation-based distance matrix.
        
        d(i,j) = sqrt(0.5 * (1 - ρ_ij))
        """
        dist = np.sqrt(0.5 * (1 - corr))
        np.fill_diagonal(dist, 0)
        # Ensure perfect symmetry for numerical stability
        dist = (dist + dist.T) / 2
        return dist
    
    def _get_quasi_diag(self, link: NDArray[np.float64]) -> list[int]:
        """
        Quasi-diagonalization: reorder assets to make similar assets adjacent.
        
        Returns sorted index list.
        """
        link = link.astype(int)
        n = link.shape[0] + 1
        
        # Build tree structure
        sorted_idx = [int(link[-1, 0]), int(link[-1, 1])]
        
        while True:
            new_sorted = []
            for i in sorted_idx:
                if i >= n:
                    # This is a cluster, expand it
                    cluster_idx = i - n
                    new_sorted.append(int(link[cluster_idx, 0]))
                    new_sorted.append(int(link[cluster_idx, 1]))
                else:
                    # This is an original asset
                    new_sorted.append(i)
            
            sorted_idx = new_sorted
            
            # Check if all are original assets
            if all(i < n for i in sorted_idx):
                break
        
        return sorted_idx
    
    def _recursive_bisection(
        self,
        cov: NDArray[np.float64],
        sorted_idx: list[int],
    ) -> NDArray[np.float64]:
        """
        Recursive bisection for weight allocation.
        
        Splits portfolio recursively, allocating to each half based on
        inverse cluster variance.
        """
        n = cov.shape[0]
        weights = np.ones(n)
        
        # Reorder covariance
        cov_sorted = cov[np.ix_(sorted_idx, sorted_idx)]
        
        # List of clusters to process (start with all assets)
        clusters = [list(range(n))]
        
        while clusters:
            new_clusters = []
            
            for cluster in clusters:
                if len(cluster) == 1:
                    continue
                
                # Split cluster in half
                mid = len(cluster) // 2
                left = cluster[:mid]
                right = cluster[mid:]
                
                # Compute cluster variances
                left_cov = cov_sorted[np.ix_(left, left)]
                right_cov = cov_sorted[np.ix_(right, right)]
                
                # Inverse variance weighting
                left_var = self._get_cluster_variance(left_cov)
                right_var = self._get_cluster_variance(right_cov)
                
                # Weight split
                alpha = 1 - left_var / (left_var + right_var)
                
                # Update weights
                for i in left:
                    original_idx = sorted_idx[i]
                    weights[original_idx] *= alpha
                for i in right:
                    original_idx = sorted_idx[i]
                    weights[original_idx] *= (1 - alpha)
                
                # Add sub-clusters if they have more than one asset
                if len(left) > 1:
                    new_clusters.append(left)
                if len(right) > 1:
                    new_clusters.append(right)
            
            clusters = new_clusters
        
        # Normalize
        weights = weights / weights.sum()
        return weights
    
    def _get_cluster_variance(self, cov: NDArray[np.float64]) -> float:
        """
        Compute cluster variance using inverse-variance portfolio within cluster.
        """
        # Inverse variance weights within cluster
        ivp = 1 / np.diag(cov)
        ivp = ivp / ivp.sum()
        
        return float(ivp @ cov @ ivp)


class NestedClusteredOptimization(PortfolioOptimizer):
    """
    Nested Clustered Optimization (NCO).
    
    Combines HRP's clustering with mean-variance optimization:
    1. Cluster assets using hierarchical clustering
    2. Run mean-variance optimization within each cluster
    3. Run mean-variance optimization across cluster portfolios
    
    Reduces dimensionality of optimization problem and improves
    robustness to estimation error.
    
    Reference:
    - "A Robust Estimator of the Efficient Frontier" (Lopez de Prado, 2019)
    """
    
    def __init__(
        self,
        n_clusters: int | None = None,
        linkage_method: str = "ward",
        inner_objective: str = "max_sharpe",
        outer_objective: str = "max_sharpe",
        **kwargs
    ):
        """
        Initialize NCO optimizer.
        
        Args:
            n_clusters: Number of clusters (None = auto-detect)
            linkage_method: Clustering linkage method
            inner_objective: Optimization within clusters
            outer_objective: Optimization across clusters
        """
        super().__init__(**kwargs)
        self.n_clusters = n_clusters
        self.linkage_method = linkage_method
        self.inner_objective = inner_objective
        self.outer_objective = outer_objective
    
    def optimize(
        self,
        expected_returns: NDArray[np.float64],
        covariance: NDArray[np.float64],
        constraints: OptimizationConstraints | None = None,
    ) -> OptimizationResult:
        """Find NCO portfolio."""
        import time
        from bxma.optimization.classical import MaxSharpeOptimizer, MinVarianceOptimizer
        
        start_time = time.time()
        n_assets = len(expected_returns)
        constraints = constraints or OptimizationConstraints()
        
        # Step 1: Cluster assets
        corr = self._cov_to_corr(covariance)
        dist = np.sqrt(0.5 * (1 - corr))
        np.fill_diagonal(dist, 0)
        
        link = linkage(squareform(dist), method=self.linkage_method)
        
        # Determine number of clusters
        if self.n_clusters is None:
            # Use gap statistic or simple heuristic
            n_clusters = max(2, min(n_assets // 3, 10))
        else:
            n_clusters = self.n_clusters
        
        cluster_labels = fcluster(link, n_clusters, criterion='maxclust')
        
        # Step 2: Optimize within each cluster
        cluster_weights = {}
        cluster_returns = np.zeros(n_clusters)
        cluster_cov = np.zeros((n_clusters, n_clusters))
        
        for k in range(1, n_clusters + 1):
            mask = cluster_labels == k
            idx = np.where(mask)[0]
            
            if len(idx) == 1:
                cluster_weights[k] = np.array([1.0])
                cluster_returns[k-1] = expected_returns[idx[0]]
                cluster_cov[k-1, k-1] = covariance[idx[0], idx[0]]
            else:
                # Inner optimization
                sub_returns = expected_returns[idx]
                sub_cov = covariance[np.ix_(idx, idx)]
                
                if self.inner_objective == "max_sharpe":
                    opt = MaxSharpeOptimizer(solver=self.solver)
                else:
                    opt = MinVarianceOptimizer(solver=self.solver)
                
                result = opt.optimize(sub_returns, sub_cov, constraints)
                cluster_weights[k] = result.weights
                cluster_returns[k-1] = result.expected_return
                
            # Compute cluster covariance contributions
            for j in range(1, n_clusters + 1):
                mask_j = cluster_labels == j
                idx_j = np.where(mask_j)[0]
                
                w_k = np.zeros(n_assets)
                w_k[idx] = cluster_weights[k] if len(idx) > 1 else 1.0
                
                w_j = np.zeros(n_assets)
                w_j[idx_j] = cluster_weights.get(j, np.array([1.0]))
                
                cluster_cov[k-1, j-1] = w_k @ covariance @ w_j
        
        # Step 3: Optimize across clusters
        if self.outer_objective == "max_sharpe":
            outer_opt = MaxSharpeOptimizer(solver=self.solver)
        else:
            outer_opt = MinVarianceOptimizer(solver=self.solver)
        
        outer_result = outer_opt.optimize(cluster_returns, cluster_cov)
        cluster_alloc = outer_result.weights
        
        # Step 4: Combine weights
        final_weights = np.zeros(n_assets)
        for k in range(1, n_clusters + 1):
            mask = cluster_labels == k
            idx = np.where(mask)[0]
            final_weights[idx] = cluster_alloc[k-1] * cluster_weights[k]
        
        final_weights = final_weights / final_weights.sum()
        
        solve_time = (time.time() - start_time) * 1000
        
        # Compute metrics
        exp_ret = float(expected_returns @ final_weights)
        exp_risk = float(np.sqrt(final_weights @ covariance @ final_weights))
        sharpe = exp_ret / exp_risk if exp_risk > 0 else 0
        
        return OptimizationResult(
            weights=final_weights,
            expected_return=exp_ret,
            expected_risk=exp_risk,
            sharpe_ratio=sharpe,
            status="optimal",
            optimal=True,
            solve_time_ms=solve_time,
            risk_contributions=self._compute_risk_contributions(final_weights, covariance),
        )
    
    def _cov_to_corr(self, cov: NDArray[np.float64]) -> NDArray[np.float64]:
        """Convert covariance to correlation matrix."""
        std = np.sqrt(np.diag(cov))
        corr = cov / np.outer(std, std)
        return np.clip(corr, -1, 1)
