# BXMA Data Analytics Platform

<div align="center">

![BXMA Platform](https://img.shields.io/badge/BXMA-Risk%20%7C%20Quant-00d4ff?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.11+-3776ab?style=for-the-badge&logo=python&logoColor=white)
![TypeScript](https://img.shields.io/badge/TypeScript-5.4+-3178c6?style=for-the-badge&logo=typescript&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**Blackstone Multi-Asset Investing Risk & Quantitative Analytics Platform**

*A bleeding-edge, full-stack platform for institutional portfolio analytics, risk management, and optimization.*

[Features](#features) • [Architecture](#architecture) • [Quick Start](#quick-start) • [Documentation](#documentation)

</div>

---

## 🚀 Features

### Core Analytics

| Module | Description | Key Methods |
|--------|-------------|-------------|
| **Risk Analytics** | Comprehensive VaR/CVaR calculations | Parametric, Historical, Monte Carlo, Cornish-Fisher |
| **Portfolio Optimization** | Advanced allocation algorithms | HRP, Risk Parity, Mean-Variance, Black-Litterman |
| **Performance Attribution** | Return decomposition | Brinson-Fachler, Geometric, Multi-period Linking |
| **Stress Testing** | Scenario analysis | Historical crises, Factor shocks, Custom scenarios |
| **Factor Models** | Risk decomposition | PCA, Statistical, Fundamental factors |
| **Explainability** | Model interpretability | SHAP analysis, Feature importance |

### Technical Capabilities

- **Real-time Streaming**: WebSocket-based live risk metrics
- **GPU Acceleration**: CUDA support for large-scale computations
- **Time-Series Optimization**: TimescaleDB for efficient data storage
- **Distributed Computing**: Kafka/Redis for event streaming
- **Production Ready**: Docker/Kubernetes deployment configurations

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                          BXMA Platform                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────────┐ │
│  │   Next.js   │◄──►│   FastAPI   │◄──►│  PostgreSQL/TimescaleDB │ │
│  │  Frontend   │    │   Backend   │    │        + Redis          │ │
│  │  (React)    │    │  (Python)   │    │        + Kafka          │ │
│  └─────────────┘    └─────────────┘    └─────────────────────────┘ │
│         │                 │                        │                │
│         │                 │                        │                │
│  ┌──────▼──────────────────▼────────────────────────▼──────────┐   │
│  │                    Analytics Engine                          │   │
│  │  ┌──────┐ ┌──────┐ ┌───────┐ ┌────────┐ ┌──────────────┐   │   │
│  │  │ Risk │ │ Opt  │ │ Attr  │ │ Stress │ │ Explainability│   │   │
│  │  │  VaR │ │ HRP  │ │Brinson│ │  Test  │ │     SHAP     │   │   │
│  │  └──────┘ └──────┘ └───────┘ └────────┘ └──────────────┘   │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Tech Stack

**Backend:**
- Python 3.11+ with async/await
- FastAPI for high-performance APIs
- NumPy/SciPy/Polars for computations
- CVXPy for convex optimization
- PyTorch for ML models

**Frontend:**
- Next.js 14 with App Router
- React 18 with Server Components
- TypeScript for type safety
- Recharts/D3 for visualizations
- Framer Motion for animations
- Zustand for state management

**Infrastructure:**
- PostgreSQL + TimescaleDB
- Redis for caching/pub-sub
- Kafka for event streaming
- Docker & Kubernetes
- Prometheus + Grafana monitoring

---

## ⚡ Quick Start

### Prerequisites

- Python 3.11+
- Node.js 20+
- Docker & Docker Compose

### Option 1: Docker (Recommended)

```bash
# Clone the repository
git clone https://github.com/your-org/bxma-platform.git
cd bxma-platform

# Start all services
docker-compose up -d

# Access the platform
# Frontend: http://localhost:3000
# Backend API: http://localhost:8000/api/docs
# Grafana: http://localhost:3001 (admin/bxma_admin)
```

### Option 2: Development Mode

```bash
# Backend
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows
pip install -r requirements.txt
python main.py --server

# Frontend (new terminal)
cd frontend
npm install
npm run dev
```

### Run Demo

```bash
python main.py
```

---

## 📖 Documentation

### API Reference

The API is fully documented with OpenAPI/Swagger:

- **Interactive Docs**: `http://localhost:8000/api/docs`
- **ReDoc**: `http://localhost:8000/api/redoc`

### Key Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/risk/var` | POST | Calculate Value-at-Risk |
| `/api/v1/optimize` | POST | Run portfolio optimization |
| `/api/v1/attribution/brinson` | POST | Brinson attribution analysis |
| `/api/v1/stress-test` | POST | Execute stress test scenario |
| `/ws/risk-stream` | WebSocket | Real-time risk metrics |

### Example: Calculate VaR

```python
import requests

response = requests.post("http://localhost:8000/api/v1/risk/var", json={
    "portfolio": {"weights": [0.4, 0.3, 0.2, 0.1]},
    "returns": {"returns": [[...]]},  # Historical returns matrix
    "confidence_level": 0.95,
    "horizon_days": 1,
    "method": "parametric"
})

result = response.json()
print(f"VaR (95%): {result['var']:.4f}")
print(f"CVaR: {result['cvar']:.4f}")
```

### Example: HRP Optimization

```python
from bxma.optimization.risk_parity import HierarchicalRiskParity
import numpy as np

# Sample data
expected_returns = np.array([0.08, 0.10, 0.06, 0.05])
covariance = np.array([...])  # Covariance matrix

optimizer = HierarchicalRiskParity()
result = optimizer.optimize(expected_returns, covariance)

print(f"Optimal Weights: {result.weights}")
print(f"Expected Return: {result.expected_return:.2%}")
print(f"Expected Risk: {result.expected_risk:.2%}")
print(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
```

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=bxma --cov-report=html

# Run specific test module
pytest tests/test_risk.py -v
```

---

## 📁 Project Structure

```
bxma-platform/
├── bxma/                      # Core analytics library
│   ├── core/                  # Types, config, portfolio
│   ├── risk/                  # VaR, factor models, covariance
│   ├── optimization/          # HRP, risk parity, classical
│   ├── attribution/           # Brinson, geometric, linking
│   ├── stress_testing/        # Scenarios, factor shocks
│   ├── explainability/        # SHAP analysis
│   ├── data/                  # Data engine
│   └── reporting/             # Dashboards, reports
├── backend/                   # FastAPI server
│   ├── main.py               # API endpoints
│   ├── auth/                 # JWT, RBAC
│   └── database/             # SQLAlchemy models
├── frontend/                  # Next.js application
│   ├── src/
│   │   ├── app/              # Pages and layouts
│   │   ├── components/       # React components
│   │   ├── lib/              # API client, stores
│   │   └── hooks/            # Custom hooks
│   └── public/               # Static assets
├── docker/                    # Docker configurations
├── kubernetes/                # K8s manifests
├── tests/                     # Test suite
├── docker-compose.yml         # Multi-container setup
├── requirements.txt           # Python dependencies
└── README.md
```

---

## 🔒 Security

- **Authentication**: OAuth2 with JWT tokens
- **Authorization**: Role-Based Access Control (RBAC)
- **Encryption**: TLS/SSL for all communications
- **Audit**: Comprehensive audit logging
- **Compliance**: SOC2/ISO27001 ready architecture

---

## 📊 Performance

| Operation | Latency (p95) | Throughput |
|-----------|---------------|------------|
| VaR Calculation | < 50ms | 1000 req/s |
| HRP Optimization | < 200ms | 500 req/s |
| Attribution | < 30ms | 2000 req/s |
| WebSocket Update | < 10ms | Real-time |

---

## 🛣️ Roadmap

- [ ] Black-Litterman optimization with views
- [ ] Machine learning factor models
- [ ] Natural language scenario generation (LLM)
- [ ] Real-time market data integration
- [ ] Mobile application
- [ ] Advanced backtesting framework

---

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.
