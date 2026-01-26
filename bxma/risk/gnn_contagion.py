"""
Graph Neural Network for Counterparty Contagion Modeling
========================================================

Models the financial system as a graph where:
- Nodes: Banks, funds, assets, counterparties
- Edges: Exposures, correlations, credit relationships

Uses dynamic GNNs to predict contagion effects - how a default in one
entity propagates through the network to impact BXMA's positions.

Key Capabilities:
- Real-time graph updates as exposures change
- Contagion simulation via message passing
- Systemic risk metrics (centrality, clustering)
- Default probability propagation

References:
- Battiston et al. (2016): Complexity theory and financial regulation
- Gai & Kapadia (2010): Contagion in financial networks
- Kipf & Welling (2017): Semi-supervised classification with GCNs
- Xu et al. (2019): How powerful are Graph Neural Networks?

Author: BXMA Quant Team
Date: January 2026
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass, field
from typing import Callable
from enum import Enum, auto
import time


class NodeType(Enum):
    """Types of nodes in the financial graph."""
    BANK = auto()
    FUND = auto()
    COUNTERPARTY = auto()
    ASSET = auto()
    SECTOR = auto()
    COUNTRY = auto()


class EdgeType(Enum):
    """Types of edges representing relationships."""
    CREDIT_EXPOSURE = auto()
    EQUITY_HOLDING = auto()
    DERIVATIVE_COUNTERPARTY = auto()
    PRIME_BROKERAGE = auto()
    CORRELATION = auto()
    SECTOR_MEMBERSHIP = auto()
    GEOGRAPHIC = auto()


@dataclass
class GraphNode:
    """A node in the financial graph."""
    
    id: str
    name: str
    node_type: NodeType
    
    # Node features
    features: NDArray[np.float64] = field(default_factory=lambda: np.zeros(64))
    
    # Financial attributes
    assets_under_management: float = 0.0
    leverage_ratio: float = 1.0
    credit_rating: str = "A"
    default_probability: float = 0.01
    
    # State
    is_defaulted: bool = False
    stress_level: float = 0.0  # 0-1 scale


@dataclass
class GraphEdge:
    """An edge in the financial graph."""
    
    source_id: str
    target_id: str
    edge_type: EdgeType
    
    # Edge attributes
    weight: float = 1.0  # Exposure amount or correlation strength
    features: NDArray[np.float64] = field(default_factory=lambda: np.zeros(16))
    
    # Financial attributes
    notional: float = 0.0
    maturity_days: int = 365
    collateralized: bool = False
    collateral_ratio: float = 0.0


@dataclass
class ContagionResult:
    """Result of contagion simulation."""
    
    # Initial shock
    initial_default_node: str
    initial_shock_size: float
    
    # Propagation
    rounds_to_stabilize: int
    nodes_affected: int
    total_cascade_losses: float
    
    # Node-level impacts
    node_losses: dict[str, float] = field(default_factory=dict)
    node_default_probs: dict[str, float] = field(default_factory=dict)
    
    # Systemic metrics
    contagion_index: float = 0.0  # How "viral" was the shock
    amplification_factor: float = 0.0  # Final loss / initial shock
    
    # Path
    propagation_path: list[list[str]] = field(default_factory=list)
    
    # Timing
    simulation_time_ms: float = 0.0


@dataclass
class SystemicRiskMetrics:
    """Systemic risk metrics for the financial graph."""
    
    # Centrality measures
    degree_centrality: dict[str, float] = field(default_factory=dict)
    betweenness_centrality: dict[str, float] = field(default_factory=dict)
    eigenvector_centrality: dict[str, float] = field(default_factory=dict)
    
    # Clustering
    clustering_coefficient: float = 0.0
    modularity: float = 0.0
    
    # Network structure
    n_nodes: int = 0
    n_edges: int = 0
    density: float = 0.0
    avg_path_length: float = 0.0
    
    # Risk metrics
    concentration_hhi: float = 0.0  # Herfindahl of exposures
    systemically_important: list[str] = field(default_factory=list)


class FinancialGraph:
    """
    The financial network graph.
    
    Maintains the topology and enables efficient updates and queries.
    """
    
    def __init__(self):
        self.nodes: dict[str, GraphNode] = {}
        self.edges: list[GraphEdge] = []
        self.adjacency: dict[str, list[str]] = {}  # node_id -> [neighbor_ids]
        self.edge_index: dict[tuple[str, str], GraphEdge] = {}
    
    def add_node(self, node: GraphNode):
        """Add a node to the graph."""
        self.nodes[node.id] = node
        if node.id not in self.adjacency:
            self.adjacency[node.id] = []
    
    def add_edge(self, edge: GraphEdge):
        """Add an edge to the graph."""
        self.edges.append(edge)
        self.edge_index[(edge.source_id, edge.target_id)] = edge
        
        # Update adjacency
        if edge.source_id not in self.adjacency:
            self.adjacency[edge.source_id] = []
        if edge.target_id not in self.adjacency:
            self.adjacency[edge.target_id] = []
        
        self.adjacency[edge.source_id].append(edge.target_id)
        self.adjacency[edge.target_id].append(edge.source_id)
    
    def get_neighbors(self, node_id: str) -> list[str]:
        """Get neighboring nodes."""
        return self.adjacency.get(node_id, [])
    
    def get_edge(self, source: str, target: str) -> GraphEdge | None:
        """Get edge between two nodes."""
        return self.edge_index.get((source, target)) or self.edge_index.get((target, source))
    
    def to_adjacency_matrix(self) -> NDArray[np.float64]:
        """Convert to dense adjacency matrix."""
        node_ids = list(self.nodes.keys())
        n = len(node_ids)
        id_to_idx = {id: i for i, id in enumerate(node_ids)}
        
        adj = np.zeros((n, n))
        for edge in self.edges:
            i = id_to_idx.get(edge.source_id)
            j = id_to_idx.get(edge.target_id)
            if i is not None and j is not None:
                adj[i, j] = edge.weight
                adj[j, i] = edge.weight
        
        return adj
    
    def to_node_features(self) -> NDArray[np.float64]:
        """Get node feature matrix."""
        return np.stack([node.features for node in self.nodes.values()])


class GraphConvolutionLayer:
    """
    Graph Convolution Layer for message passing.
    
    Implements: H^(l+1) = σ(D^(-1/2) A D^(-1/2) H^(l) W^(l))
    
    Where:
    - A: Adjacency matrix (with self-loops)
    - D: Degree matrix
    - H: Node features
    - W: Learnable weights
    """
    
    def __init__(self, in_features: int, out_features: int):
        self.in_features = in_features
        self.out_features = out_features
        
        # Initialize weights (Xavier initialization)
        self.W = np.random.randn(in_features, out_features) * np.sqrt(2.0 / (in_features + out_features))
        self.bias = np.zeros(out_features)
    
    def forward(
        self,
        X: NDArray[np.float64],
        A: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """
        Forward pass.
        
        Args:
            X: Node features (N, in_features)
            A: Adjacency matrix (N, N)
            
        Returns:
            Updated features (N, out_features)
        """
        # Add self-loops
        A_hat = A + np.eye(A.shape[0])
        
        # Compute degree matrix
        D = np.diag(np.sum(A_hat, axis=1))
        D_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(D) + 1e-8))
        
        # Normalized adjacency
        A_norm = D_inv_sqrt @ A_hat @ D_inv_sqrt
        
        # Message passing
        H = A_norm @ X @ self.W + self.bias
        
        # ReLU activation
        return np.maximum(0, H)


class GraphAttentionLayer:
    """
    Graph Attention Layer with multi-head attention.
    
    Learns to weight neighbor contributions dynamically,
    crucial for identifying which exposures matter most during stress.
    """
    
    def __init__(self, in_features: int, out_features: int, n_heads: int = 4):
        self.in_features = in_features
        self.out_features = out_features
        self.n_heads = n_heads
        self.head_dim = out_features // n_heads
        
        # Multi-head attention parameters
        self.W = np.random.randn(n_heads, in_features, self.head_dim) * 0.1
        self.a = np.random.randn(n_heads, 2 * self.head_dim) * 0.1
    
    def forward(
        self,
        X: NDArray[np.float64],
        A: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """
        Forward pass with attention.
        
        Args:
            X: Node features (N, in_features)
            A: Adjacency matrix (N, N)
            
        Returns:
            Updated features (N, out_features)
        """
        N = X.shape[0]
        heads_out = []
        
        for head in range(self.n_heads):
            # Linear transformation
            H = X @ self.W[head]  # (N, head_dim)
            
            # Compute attention scores
            # For each pair (i, j), compute a^T [h_i || h_j]
            scores = np.zeros((N, N))
            for i in range(N):
                for j in range(N):
                    if A[i, j] > 0 or i == j:
                        concat = np.concatenate([H[i], H[j]])
                        scores[i, j] = np.dot(self.a[head], concat)
            
            # Masked softmax (only over neighbors)
            mask = (A > 0) | np.eye(N, dtype=bool)
            scores = np.where(mask, scores, -1e9)
            attention = np.exp(scores) / (np.sum(np.exp(scores), axis=1, keepdims=True) + 1e-8)
            
            # Aggregate
            out = attention @ H  # (N, head_dim)
            heads_out.append(out)
        
        # Concatenate heads
        return np.concatenate(heads_out, axis=1)


class ContagionGNN:
    """
    Graph Neural Network for Contagion Modeling.
    
    Architecture:
    1. Node embedding layer
    2. Multiple graph convolution layers
    3. Attention layer for exposure weighting
    4. Output layer for default probability
    """
    
    def __init__(
        self,
        node_features: int = 64,
        hidden_dim: int = 128,
        n_layers: int = 3,
        n_attention_heads: int = 4,
    ):
        self.node_features = node_features
        self.hidden_dim = hidden_dim
        
        # Build layers
        self.conv_layers = []
        
        # Input layer
        self.conv_layers.append(GraphConvolutionLayer(node_features, hidden_dim))
        
        # Hidden layers
        for _ in range(n_layers - 2):
            self.conv_layers.append(GraphConvolutionLayer(hidden_dim, hidden_dim))
        
        # Attention layer
        self.attention = GraphAttentionLayer(hidden_dim, hidden_dim, n_attention_heads)
        
        # Output layer (predict default probability)
        self.output_layer = GraphConvolutionLayer(hidden_dim, 1)
    
    def forward(
        self,
        X: NDArray[np.float64],
        A: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """
        Forward pass to predict default probabilities.
        
        Args:
            X: Node features (N, node_features)
            A: Weighted adjacency matrix (N, N)
            
        Returns:
            Default probabilities (N,)
        """
        H = X
        
        # Convolution layers
        for conv in self.conv_layers:
            H = conv.forward(H, A)
        
        # Attention layer
        H = self.attention.forward(H, A)
        
        # Output
        logits = self.output_layer.forward(H, A)
        
        # Sigmoid to get probabilities
        probs = 1 / (1 + np.exp(-logits.flatten()))
        
        return probs
    
    def simulate_contagion(
        self,
        graph: FinancialGraph,
        initial_default: str,
        shock_size: float = 1.0,
        max_rounds: int = 100,
        default_threshold: float = 0.5,
    ) -> ContagionResult:
        """
        Simulate contagion propagation from an initial default.
        
        Args:
            graph: The financial graph
            initial_default: ID of initially defaulting node
            shock_size: Size of initial shock (0-1)
            max_rounds: Maximum simulation rounds
            default_threshold: Probability threshold for default
            
        Returns:
            ContagionResult with propagation details
        """
        start_time = time.time()
        
        # Get graph matrices
        X = graph.to_node_features()
        A = graph.to_adjacency_matrix()
        
        node_ids = list(graph.nodes.keys())
        id_to_idx = {id: i for i, id in enumerate(node_ids)}
        idx_to_id = {i: id for id, i in id_to_idx.items()}
        
        N = len(node_ids)
        
        # Initialize states
        default_probs = np.zeros(N)
        losses = np.zeros(N)
        defaulted = np.zeros(N, dtype=bool)
        
        # Apply initial shock
        if initial_default in id_to_idx:
            idx = id_to_idx[initial_default]
            default_probs[idx] = shock_size
            defaulted[idx] = True
            losses[idx] = graph.nodes[initial_default].assets_under_management * shock_size
        
        propagation_path = [[initial_default]]
        
        # Propagation loop
        for round_num in range(max_rounds):
            # Get current state features
            state_features = np.column_stack([
                X,
                default_probs.reshape(-1, 1),
                losses.reshape(-1, 1),
            ])
            
            # Pad to expected dimensions
            if state_features.shape[1] < self.node_features:
                padding = np.zeros((N, self.node_features - state_features.shape[1]))
                state_features = np.column_stack([state_features, padding])
            else:
                state_features = state_features[:, :self.node_features]
            
            # Forward pass through GNN
            new_probs = self.forward(state_features, A)
            
            # Combine with existing (max of current and propagated)
            default_probs = np.maximum(default_probs, new_probs)
            
            # Check for new defaults
            new_defaults = (default_probs > default_threshold) & ~defaulted
            
            if not np.any(new_defaults):
                break
            
            # Record new defaults
            new_default_ids = [idx_to_id[i] for i in np.where(new_defaults)[0]]
            propagation_path.append(new_default_ids)
            
            # Update losses
            for idx in np.where(new_defaults)[0]:
                node_id = idx_to_id[idx]
                node = graph.nodes[node_id]
                losses[idx] = node.assets_under_management * default_probs[idx]
            
            defaulted |= new_defaults
        
        # Compute results
        node_losses = {idx_to_id[i]: losses[i] for i in range(N) if losses[i] > 0}
        node_default_probs = {idx_to_id[i]: default_probs[i] for i in range(N)}
        
        total_losses = float(np.sum(losses))
        initial_loss = losses[id_to_idx.get(initial_default, 0)] if initial_default in id_to_idx else shock_size
        
        simulation_time = (time.time() - start_time) * 1000
        
        return ContagionResult(
            initial_default_node=initial_default,
            initial_shock_size=shock_size,
            rounds_to_stabilize=round_num + 1,
            nodes_affected=int(np.sum(defaulted)),
            total_cascade_losses=total_losses,
            node_losses=node_losses,
            node_default_probs=node_default_probs,
            contagion_index=np.sum(defaulted) / N,
            amplification_factor=total_losses / max(initial_loss, 1e-10),
            propagation_path=propagation_path,
            simulation_time_ms=simulation_time,
        )


class SystemicRiskAnalyzer:
    """
    Analyzes systemic risk in the financial graph.
    
    Computes various centrality and clustering metrics to identify
    systemically important nodes.
    """
    
    def __init__(self, graph: FinancialGraph):
        self.graph = graph
        self.adj = graph.to_adjacency_matrix()
        self.node_ids = list(graph.nodes.keys())
        self.id_to_idx = {id: i for i, id in enumerate(self.node_ids)}
    
    def compute_degree_centrality(self) -> dict[str, float]:
        """Compute degree centrality for all nodes."""
        degrees = np.sum(self.adj > 0, axis=1)
        max_degree = len(self.node_ids) - 1
        centrality = degrees / max(max_degree, 1)
        
        return {self.node_ids[i]: centrality[i] for i in range(len(self.node_ids))}
    
    def compute_betweenness_centrality(self) -> dict[str, float]:
        """
        Compute betweenness centrality using shortest paths.
        
        Identifies nodes that are "bridges" in the network.
        """
        N = len(self.node_ids)
        betweenness = np.zeros(N)
        
        # Floyd-Warshall for all shortest paths
        dist = np.where(self.adj > 0, 1, np.inf)
        np.fill_diagonal(dist, 0)
        
        for k in range(N):
            for i in range(N):
                for j in range(N):
                    if dist[i, k] + dist[k, j] < dist[i, j]:
                        dist[i, j] = dist[i, k] + dist[k, j]
        
        # Count paths through each node
        for s in range(N):
            for t in range(N):
                if s != t and np.isfinite(dist[s, t]):
                    for v in range(N):
                        if v != s and v != t:
                            if np.isfinite(dist[s, v]) and np.isfinite(dist[v, t]):
                                if abs(dist[s, v] + dist[v, t] - dist[s, t]) < 0.001:
                                    betweenness[v] += 1
        
        # Normalize
        norm = max((N - 1) * (N - 2), 1)
        betweenness = betweenness / norm
        
        return {self.node_ids[i]: betweenness[i] for i in range(N)}
    
    def compute_eigenvector_centrality(self, max_iter: int = 100) -> dict[str, float]:
        """
        Compute eigenvector centrality via power iteration.
        
        Nodes connected to high-centrality nodes have higher centrality.
        """
        N = len(self.node_ids)
        x = np.ones(N) / N
        
        for _ in range(max_iter):
            x_new = self.adj @ x
            norm = np.linalg.norm(x_new)
            if norm > 0:
                x_new = x_new / norm
            
            if np.allclose(x, x_new):
                break
            x = x_new
        
        return {self.node_ids[i]: x[i] for i in range(N)}
    
    def compute_clustering_coefficient(self) -> float:
        """Compute global clustering coefficient."""
        N = len(self.node_ids)
        triangles = 0
        triplets = 0
        
        for i in range(N):
            neighbors = np.where(self.adj[i] > 0)[0]
            k = len(neighbors)
            if k >= 2:
                triplets += k * (k - 1) / 2
                for j in range(len(neighbors)):
                    for l in range(j + 1, len(neighbors)):
                        if self.adj[neighbors[j], neighbors[l]] > 0:
                            triangles += 1
        
        return (3 * triangles) / max(triplets, 1)
    
    def identify_systemically_important(self, top_k: int = 10) -> list[str]:
        """
        Identify systemically important financial institutions (SIFIs).
        
        Uses a composite score of centrality measures and size.
        """
        degree = self.compute_degree_centrality()
        betweenness = self.compute_betweenness_centrality()
        eigenvector = self.compute_eigenvector_centrality()
        
        # Composite score
        scores = {}
        for node_id in self.node_ids:
            node = self.graph.nodes[node_id]
            size_factor = np.log1p(node.assets_under_management) / 20  # Normalize
            
            scores[node_id] = (
                0.3 * degree[node_id] +
                0.3 * betweenness[node_id] +
                0.2 * eigenvector[node_id] +
                0.2 * size_factor
            )
        
        # Sort and return top k
        sorted_nodes = sorted(scores.items(), key=lambda x: -x[1])
        return [node_id for node_id, _ in sorted_nodes[:top_k]]
    
    def compute_metrics(self) -> SystemicRiskMetrics:
        """Compute all systemic risk metrics."""
        N = len(self.node_ids)
        n_edges = len(self.graph.edges)
        max_edges = N * (N - 1) / 2
        
        return SystemicRiskMetrics(
            degree_centrality=self.compute_degree_centrality(),
            betweenness_centrality=self.compute_betweenness_centrality(),
            eigenvector_centrality=self.compute_eigenvector_centrality(),
            clustering_coefficient=self.compute_clustering_coefficient(),
            n_nodes=N,
            n_edges=n_edges,
            density=n_edges / max(max_edges, 1),
            systemically_important=self.identify_systemically_important(),
        )


def build_sample_financial_graph() -> FinancialGraph:
    """Build a sample financial network for testing."""
    graph = FinancialGraph()
    
    # Add major banks
    banks = ["JPM", "GS", "MS", "BAC", "C", "WFC"]
    for bank in banks:
        graph.add_node(GraphNode(
            id=bank,
            name=f"{bank} Bank",
            node_type=NodeType.BANK,
            features=np.random.randn(64),
            assets_under_management=np.random.uniform(1e12, 3e12),
            leverage_ratio=np.random.uniform(10, 20),
        ))
    
    # Add hedge funds
    funds = ["BXMA", "BW", "CITADEL", "AQR", "DE_SHAW"]
    for fund in funds:
        graph.add_node(GraphNode(
            id=fund,
            name=f"{fund} Fund",
            node_type=NodeType.FUND,
            features=np.random.randn(64),
            assets_under_management=np.random.uniform(50e9, 200e9),
            leverage_ratio=np.random.uniform(2, 5),
        ))
    
    # Add prime brokerage relationships
    for fund in funds:
        pb_banks = np.random.choice(banks, size=2, replace=False)
        for bank in pb_banks:
            graph.add_edge(GraphEdge(
                source_id=fund,
                target_id=bank,
                edge_type=EdgeType.PRIME_BROKERAGE,
                weight=np.random.uniform(0.3, 0.8),
                notional=np.random.uniform(10e9, 50e9),
            ))
    
    # Add interbank exposures
    for i, bank1 in enumerate(banks):
        for bank2 in banks[i+1:]:
            if np.random.random() > 0.3:  # 70% connectivity
                graph.add_edge(GraphEdge(
                    source_id=bank1,
                    target_id=bank2,
                    edge_type=EdgeType.CREDIT_EXPOSURE,
                    weight=np.random.uniform(0.2, 0.6),
                    notional=np.random.uniform(5e9, 20e9),
                ))
    
    return graph
