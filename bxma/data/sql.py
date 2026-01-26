"""
SQL Database Integration
========================

Advanced SQL capabilities for large-scale data operations:
- PostgreSQL/TimescaleDB for time-series
- DuckDB for analytical queries
- Connection pooling
- Query optimization
- Data pipeline integration

CRITICAL REQUIREMENT: Advanced programming skills in SQL required.

Author: BXMA Quant Team
Date: January 2026
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Any, Literal, Generator
from pathlib import Path
import json


@dataclass
class SQLConfig:
    """SQL database configuration."""
    
    # Connection
    driver: Literal["postgresql", "timescaledb", "duckdb", "sqlite"] = "postgresql"
    host: str = "localhost"
    port: int = 5432
    database: str = "bxma"
    username: str = ""
    password: str = ""
    
    # Connection pool
    pool_size: int = 10
    max_overflow: int = 20
    pool_timeout: int = 30
    
    # SSL
    ssl_mode: str = "prefer"
    ssl_cert_path: str | None = None
    
    # Performance
    statement_timeout_ms: int = 30000
    application_name: str = "BXMA_Risk_Platform"
    
    @property
    def connection_string(self) -> str:
        """Build connection string."""
        if self.driver == "duckdb":
            return f"duckdb:///{self.database}"
        elif self.driver == "sqlite":
            return f"sqlite:///{self.database}"
        else:
            return (
                f"{self.driver}://{self.username}:{self.password}@"
                f"{self.host}:{self.port}/{self.database}"
            )


@dataclass
class QueryResult:
    """Result of a SQL query."""
    
    columns: list[str] = field(default_factory=list)
    rows: list[tuple] = field(default_factory=list)
    row_count: int = 0
    
    # Performance
    execution_time_ms: float = 0
    query_plan: str | None = None
    
    def to_dict_list(self) -> list[dict]:
        """Convert to list of dictionaries."""
        return [dict(zip(self.columns, row)) for row in self.rows]
    
    def to_numpy(self) -> NDArray:
        """Convert to numpy array."""
        return np.array(self.rows)


class SQLQueryBuilder:
    """
    SQL query builder for complex analytical queries.
    
    Provides:
    - Parameterized queries
    - Query composition
    - Common financial queries
    """
    
    @staticmethod
    def build_timeseries_query(
        table: str,
        columns: list[str],
        date_column: str = "as_of_date",
        start_date: date | None = None,
        end_date: date | None = None,
        asset_ids: list[str] | None = None,
        order_by: str | None = None,
        limit: int | None = None,
    ) -> tuple[str, dict]:
        """Build a time series query."""
        select_clause = ", ".join(columns)
        query = f"SELECT {select_clause} FROM {table}"
        params: dict[str, Any] = {}
        
        conditions = []
        
        if start_date:
            conditions.append(f"{date_column} >= :start_date")
            params["start_date"] = start_date
        
        if end_date:
            conditions.append(f"{date_column} <= :end_date")
            params["end_date"] = end_date
        
        if asset_ids:
            conditions.append("asset_id = ANY(:asset_ids)")
            params["asset_ids"] = asset_ids
        
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        
        if order_by:
            query += f" ORDER BY {order_by}"
        
        if limit:
            query += f" LIMIT {limit}"
        
        return query, params
    
    @staticmethod
    def build_portfolio_holdings_query(
        portfolio_id: str,
        as_of_date: date,
    ) -> tuple[str, dict]:
        """Build query for portfolio holdings."""
        query = """
        SELECT 
            h.asset_id,
            a.name AS asset_name,
            a.asset_class,
            a.sector,
            h.quantity,
            h.market_value,
            h.weight,
            h.cost_basis,
            h.unrealized_pnl
        FROM portfolio_holdings h
        JOIN assets a ON h.asset_id = a.asset_id
        WHERE h.portfolio_id = :portfolio_id
          AND h.as_of_date = :as_of_date
        ORDER BY h.market_value DESC
        """
        
        return query, {"portfolio_id": portfolio_id, "as_of_date": as_of_date}
    
    @staticmethod
    def build_returns_query(
        asset_ids: list[str],
        start_date: date,
        end_date: date,
        frequency: Literal["daily", "weekly", "monthly"] = "daily",
    ) -> tuple[str, dict]:
        """Build query for asset returns."""
        # Use window functions for return calculation
        query = """
        WITH prices AS (
            SELECT 
                asset_id,
                as_of_date,
                close_price,
                LAG(close_price) OVER (PARTITION BY asset_id ORDER BY as_of_date) AS prev_price
            FROM price_history
            WHERE asset_id = ANY(:asset_ids)
              AND as_of_date BETWEEN :start_date AND :end_date
        )
        SELECT 
            asset_id,
            as_of_date,
            close_price,
            (close_price - prev_price) / NULLIF(prev_price, 0) AS daily_return
        FROM prices
        WHERE prev_price IS NOT NULL
        ORDER BY asset_id, as_of_date
        """
        
        return query, {
            "asset_ids": asset_ids,
            "start_date": start_date,
            "end_date": end_date,
        }
    
    @staticmethod
    def build_risk_metrics_query(
        portfolio_id: str,
        start_date: date,
        end_date: date,
    ) -> tuple[str, dict]:
        """Build query for historical risk metrics."""
        query = """
        SELECT 
            as_of_date,
            portfolio_id,
            var_95,
            var_99,
            cvar_95,
            cvar_99,
            volatility,
            beta,
            sharpe_ratio,
            sortino_ratio,
            max_drawdown,
            tracking_error
        FROM risk_metrics_history
        WHERE portfolio_id = :portfolio_id
          AND as_of_date BETWEEN :start_date AND :end_date
        ORDER BY as_of_date
        """
        
        return query, {
            "portfolio_id": portfolio_id,
            "start_date": start_date,
            "end_date": end_date,
        }
    
    @staticmethod
    def build_factor_exposure_query(
        portfolio_id: str,
        factor_model: str,
        as_of_date: date,
    ) -> tuple[str, dict]:
        """Build query for factor exposures."""
        query = """
        SELECT 
            factor_name,
            factor_category,
            exposure,
            marginal_contribution,
            standalone_risk
        FROM factor_exposures
        WHERE portfolio_id = :portfolio_id
          AND factor_model = :factor_model
          AND as_of_date = :as_of_date
        ORDER BY ABS(exposure) DESC
        """
        
        return query, {
            "portfolio_id": portfolio_id,
            "factor_model": factor_model,
            "as_of_date": as_of_date,
        }
    
    @staticmethod
    def build_correlation_matrix_query(
        asset_ids: list[str],
        lookback_days: int = 252,
    ) -> tuple[str, dict]:
        """Build query for correlation matrix calculation."""
        query = """
        WITH returns AS (
            SELECT 
                asset_id,
                as_of_date,
                (close_price - LAG(close_price) OVER (PARTITION BY asset_id ORDER BY as_of_date)) 
                    / NULLIF(LAG(close_price) OVER (PARTITION BY asset_id ORDER BY as_of_date), 0) AS daily_return
            FROM price_history
            WHERE asset_id = ANY(:asset_ids)
              AND as_of_date >= CURRENT_DATE - :lookback_days
        ),
        return_matrix AS (
            SELECT 
                r1.asset_id AS asset_1,
                r2.asset_id AS asset_2,
                CORR(r1.daily_return, r2.daily_return) AS correlation
            FROM returns r1
            JOIN returns r2 ON r1.as_of_date = r2.as_of_date
            WHERE r1.daily_return IS NOT NULL AND r2.daily_return IS NOT NULL
            GROUP BY r1.asset_id, r2.asset_id
        )
        SELECT * FROM return_matrix
        ORDER BY asset_1, asset_2
        """
        
        return query, {"asset_ids": asset_ids, "lookback_days": lookback_days}


class SQLDataManager:
    """
    SQL data management for the BXMA platform.
    
    Handles:
    - Large dataset operations
    - Time-series data storage
    - Analytical queries
    - Data pipelines
    """
    
    def __init__(self, config: SQLConfig | None = None):
        self.config = config or SQLConfig()
        self._connection = None
        self._query_builder = SQLQueryBuilder()
    
    def connect(self) -> bool:
        """Establish database connection."""
        # In production, use SQLAlchemy or psycopg2
        print(f"Connecting to {self.config.driver}://{self.config.host}:{self.config.port}")
        self._connection = True
        return True
    
    def disconnect(self):
        """Close database connection."""
        self._connection = None
    
    def execute(
        self,
        query: str,
        params: dict | None = None,
        fetch: bool = True,
    ) -> QueryResult:
        """Execute a SQL query."""
        start_time = datetime.now()
        
        # In production, execute actual SQL
        # For now, return sample result
        result = QueryResult(
            columns=["sample_column"],
            rows=[("sample_value",)],
            row_count=1,
            execution_time_ms=(datetime.now() - start_time).total_seconds() * 1000,
        )
        
        return result
    
    def execute_many(
        self,
        query: str,
        params_list: list[dict],
        batch_size: int = 1000,
    ) -> int:
        """Execute query with multiple parameter sets (bulk insert/update)."""
        total_rows = 0
        
        for i in range(0, len(params_list), batch_size):
            batch = params_list[i:i + batch_size]
            total_rows += len(batch)
        
        return total_rows
    
    def stream_query(
        self,
        query: str,
        params: dict | None = None,
        chunk_size: int = 10000,
    ) -> Generator[list[tuple], None, None]:
        """Stream large result sets in chunks."""
        # In production, use server-side cursor
        yield [("sample",)]
    
    def get_portfolio_holdings(
        self,
        portfolio_id: str,
        as_of_date: date,
    ) -> QueryResult:
        """Get portfolio holdings."""
        query, params = self._query_builder.build_portfolio_holdings_query(
            portfolio_id, as_of_date
        )
        return self.execute(query, params)
    
    def get_returns(
        self,
        asset_ids: list[str],
        start_date: date,
        end_date: date,
    ) -> QueryResult:
        """Get asset returns."""
        query, params = self._query_builder.build_returns_query(
            asset_ids, start_date, end_date
        )
        return self.execute(query, params)
    
    def get_risk_metrics_history(
        self,
        portfolio_id: str,
        start_date: date,
        end_date: date,
    ) -> QueryResult:
        """Get historical risk metrics."""
        query, params = self._query_builder.build_risk_metrics_query(
            portfolio_id, start_date, end_date
        )
        return self.execute(query, params)
    
    def save_risk_metrics(
        self,
        portfolio_id: str,
        as_of_date: date,
        metrics: dict[str, float],
    ) -> bool:
        """Save risk metrics to database."""
        query = """
        INSERT INTO risk_metrics_history (
            portfolio_id, as_of_date, var_95, var_99, cvar_95, cvar_99,
            volatility, beta, sharpe_ratio, sortino_ratio, max_drawdown
        ) VALUES (
            :portfolio_id, :as_of_date, :var_95, :var_99, :cvar_95, :cvar_99,
            :volatility, :beta, :sharpe_ratio, :sortino_ratio, :max_drawdown
        )
        ON CONFLICT (portfolio_id, as_of_date) DO UPDATE SET
            var_95 = EXCLUDED.var_95,
            var_99 = EXCLUDED.var_99,
            cvar_95 = EXCLUDED.cvar_95,
            cvar_99 = EXCLUDED.cvar_99,
            volatility = EXCLUDED.volatility,
            beta = EXCLUDED.beta,
            sharpe_ratio = EXCLUDED.sharpe_ratio,
            sortino_ratio = EXCLUDED.sortino_ratio,
            max_drawdown = EXCLUDED.max_drawdown
        """
        
        params = {
            "portfolio_id": portfolio_id,
            "as_of_date": as_of_date,
            **metrics,
        }
        
        self.execute(query, params, fetch=False)
        return True


# Common SQL queries for BXMA operations
BXMA_SQL_QUERIES = {
    "daily_nav": """
        SELECT 
            portfolio_id,
            as_of_date,
            nav,
            nav - LAG(nav) OVER (PARTITION BY portfolio_id ORDER BY as_of_date) AS daily_change,
            (nav - LAG(nav) OVER (PARTITION BY portfolio_id ORDER BY as_of_date)) 
                / NULLIF(LAG(nav) OVER (PARTITION BY portfolio_id ORDER BY as_of_date), 0) AS daily_return
        FROM portfolio_nav
        WHERE as_of_date BETWEEN :start_date AND :end_date
        ORDER BY portfolio_id, as_of_date
    """,
    
    "top_contributors": """
        SELECT 
            asset_id,
            asset_name,
            sector,
            SUM(contribution) AS total_contribution
        FROM performance_attribution
        WHERE portfolio_id = :portfolio_id
          AND as_of_date BETWEEN :start_date AND :end_date
        GROUP BY asset_id, asset_name, sector
        ORDER BY total_contribution DESC
        LIMIT :top_n
    """,
    
    "risk_limit_utilization": """
        SELECT 
            portfolio_id,
            limit_type,
            current_value,
            limit_value,
            (current_value / NULLIF(limit_value, 0)) * 100 AS utilization_pct,
            CASE 
                WHEN current_value > limit_value THEN 'BREACH'
                WHEN current_value > limit_value * 0.9 THEN 'WARNING'
                ELSE 'OK'
            END AS status
        FROM risk_limits
        WHERE as_of_date = CURRENT_DATE
        ORDER BY utilization_pct DESC
    """,
    
    "sector_exposure": """
        SELECT 
            sector,
            SUM(market_value) AS total_market_value,
            SUM(weight) AS total_weight,
            COUNT(*) AS position_count
        FROM portfolio_holdings h
        JOIN assets a ON h.asset_id = a.asset_id
        WHERE h.portfolio_id = :portfolio_id
          AND h.as_of_date = :as_of_date
        GROUP BY sector
        ORDER BY total_weight DESC
    """,
    
    "var_backtesting": """
        WITH var_breaches AS (
            SELECT 
                as_of_date,
                var_95,
                actual_return,
                CASE WHEN actual_return < -var_95 THEN 1 ELSE 0 END AS breach_95
            FROM risk_metrics_history
            WHERE portfolio_id = :portfolio_id
              AND as_of_date BETWEEN :start_date AND :end_date
        )
        SELECT 
            COUNT(*) AS total_days,
            SUM(breach_95) AS breach_count_95,
            (SUM(breach_95)::FLOAT / COUNT(*)) * 100 AS breach_rate_95
        FROM var_breaches
    """,
}


# Database schema for BXMA platform
BXMA_SCHEMA_DDL = """
-- Portfolios
CREATE TABLE IF NOT EXISTS portfolios (
    portfolio_id VARCHAR(50) PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    manager_id VARCHAR(50),
    inception_date DATE,
    benchmark_id VARCHAR(50),
    currency VARCHAR(3) DEFAULT 'USD',
    status VARCHAR(20) DEFAULT 'ACTIVE',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Assets
CREATE TABLE IF NOT EXISTS assets (
    asset_id VARCHAR(50) PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    ticker VARCHAR(20),
    asset_class VARCHAR(50),
    sector VARCHAR(100),
    industry VARCHAR(100),
    country VARCHAR(3),
    currency VARCHAR(3),
    isin VARCHAR(12),
    cusip VARCHAR(9),
    sedol VARCHAR(7),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Portfolio Holdings
CREATE TABLE IF NOT EXISTS portfolio_holdings (
    portfolio_id VARCHAR(50) REFERENCES portfolios(portfolio_id),
    asset_id VARCHAR(50) REFERENCES assets(asset_id),
    as_of_date DATE,
    quantity NUMERIC(20, 6),
    market_value NUMERIC(20, 2),
    weight NUMERIC(10, 6),
    cost_basis NUMERIC(20, 2),
    unrealized_pnl NUMERIC(20, 2),
    PRIMARY KEY (portfolio_id, asset_id, as_of_date)
);

-- Price History
CREATE TABLE IF NOT EXISTS price_history (
    asset_id VARCHAR(50) REFERENCES assets(asset_id),
    as_of_date DATE,
    open_price NUMERIC(20, 6),
    high_price NUMERIC(20, 6),
    low_price NUMERIC(20, 6),
    close_price NUMERIC(20, 6),
    adjusted_close NUMERIC(20, 6),
    volume BIGINT,
    PRIMARY KEY (asset_id, as_of_date)
);

-- Risk Metrics History
CREATE TABLE IF NOT EXISTS risk_metrics_history (
    portfolio_id VARCHAR(50) REFERENCES portfolios(portfolio_id),
    as_of_date DATE,
    var_95 NUMERIC(10, 6),
    var_99 NUMERIC(10, 6),
    cvar_95 NUMERIC(10, 6),
    cvar_99 NUMERIC(10, 6),
    volatility NUMERIC(10, 6),
    beta NUMERIC(10, 6),
    sharpe_ratio NUMERIC(10, 6),
    sortino_ratio NUMERIC(10, 6),
    max_drawdown NUMERIC(10, 6),
    tracking_error NUMERIC(10, 6),
    information_ratio NUMERIC(10, 6),
    PRIMARY KEY (portfolio_id, as_of_date)
);

-- Factor Exposures
CREATE TABLE IF NOT EXISTS factor_exposures (
    portfolio_id VARCHAR(50) REFERENCES portfolios(portfolio_id),
    as_of_date DATE,
    factor_model VARCHAR(50),
    factor_name VARCHAR(100),
    factor_category VARCHAR(50),
    exposure NUMERIC(10, 6),
    marginal_contribution NUMERIC(10, 6),
    standalone_risk NUMERIC(10, 6),
    PRIMARY KEY (portfolio_id, as_of_date, factor_model, factor_name)
);

-- Risk Limits
CREATE TABLE IF NOT EXISTS risk_limits (
    portfolio_id VARCHAR(50) REFERENCES portfolios(portfolio_id),
    as_of_date DATE,
    limit_type VARCHAR(50),
    limit_value NUMERIC(20, 6),
    current_value NUMERIC(20, 6),
    breach_date DATE,
    PRIMARY KEY (portfolio_id, as_of_date, limit_type)
);

-- Performance Attribution
CREATE TABLE IF NOT EXISTS performance_attribution (
    portfolio_id VARCHAR(50) REFERENCES portfolios(portfolio_id),
    as_of_date DATE,
    asset_id VARCHAR(50),
    asset_name VARCHAR(255),
    sector VARCHAR(100),
    contribution NUMERIC(10, 6),
    allocation_effect NUMERIC(10, 6),
    selection_effect NUMERIC(10, 6),
    interaction_effect NUMERIC(10, 6),
    PRIMARY KEY (portfolio_id, as_of_date, asset_id)
);

-- Portfolio NAV
CREATE TABLE IF NOT EXISTS portfolio_nav (
    portfolio_id VARCHAR(50) REFERENCES portfolios(portfolio_id),
    as_of_date DATE,
    nav NUMERIC(20, 2),
    aum NUMERIC(20, 2),
    daily_return NUMERIC(10, 6),
    mtd_return NUMERIC(10, 6),
    ytd_return NUMERIC(10, 6),
    inception_return NUMERIC(10, 6),
    PRIMARY KEY (portfolio_id, as_of_date)
);

-- Create indexes for common queries
CREATE INDEX IF NOT EXISTS idx_price_history_date ON price_history(as_of_date);
CREATE INDEX IF NOT EXISTS idx_holdings_date ON portfolio_holdings(as_of_date);
CREATE INDEX IF NOT EXISTS idx_risk_metrics_date ON risk_metrics_history(as_of_date);
CREATE INDEX IF NOT EXISTS idx_nav_date ON portfolio_nav(as_of_date);

-- Create TimescaleDB hypertables if available
-- SELECT create_hypertable('price_history', 'as_of_date', if_not_exists => TRUE);
-- SELECT create_hypertable('risk_metrics_history', 'as_of_date', if_not_exists => TRUE);
"""
