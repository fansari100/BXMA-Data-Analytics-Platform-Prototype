"""
Database Models for BXMA Data Analytics Platform.

Uses SQLAlchemy 2.0 with async support and TimescaleDB for time-series data.
"""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from typing import Optional
from uuid import UUID, uuid4

from sqlalchemy import (
    Boolean,
    Column,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    JSON,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID as PGUUID
from sqlalchemy.ext.asyncio import AsyncAttrs
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(AsyncAttrs, DeclarativeBase):
    """Base class for all database models."""
    pass


# =============================================================================
# PORTFOLIO MODELS
# =============================================================================

class Portfolio(Base):
    """Portfolio master table."""
    
    __tablename__ = "portfolios"
    
    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    strategy: Mapped[str] = mapped_column(String(100), nullable=False)
    inception_date: Mapped[date] = mapped_column(Date, nullable=False)
    benchmark_id: Mapped[Optional[UUID]] = mapped_column(PGUUID(as_uuid=True), nullable=True)
    currency: Mapped[str] = mapped_column(String(3), default="USD")
    status: Mapped[str] = mapped_column(String(50), default="active")
    metadata_: Mapped[dict] = mapped_column(JSONB, default={})
    
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())
    
    # Relationships
    positions: Mapped[list["Position"]] = relationship(back_populates="portfolio", lazy="selectin")
    snapshots: Mapped[list["PortfolioSnapshot"]] = relationship(back_populates="portfolio", lazy="selectin")


class Position(Base):
    """Individual position within a portfolio."""
    
    __tablename__ = "positions"
    
    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    portfolio_id: Mapped[UUID] = mapped_column(ForeignKey("portfolios.id"), nullable=False)
    security_id: Mapped[UUID] = mapped_column(ForeignKey("securities.id"), nullable=False)
    
    quantity: Mapped[Decimal] = mapped_column(Numeric(20, 8), default=0)
    market_value: Mapped[Decimal] = mapped_column(Numeric(20, 4), default=0)
    weight: Mapped[float] = mapped_column(Float, default=0)
    cost_basis: Mapped[Decimal] = mapped_column(Numeric(20, 4), default=0)
    
    as_of_date: Mapped[date] = mapped_column(Date, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    
    # Relationships
    portfolio: Mapped["Portfolio"] = relationship(back_populates="positions")
    security: Mapped["Security"] = relationship(back_populates="positions")
    
    __table_args__ = (
        UniqueConstraint("portfolio_id", "security_id", "as_of_date", name="uq_position"),
        Index("ix_positions_portfolio_date", "portfolio_id", "as_of_date"),
    )


class PortfolioSnapshot(Base):
    """Point-in-time portfolio snapshot for historical analysis."""
    
    __tablename__ = "portfolio_snapshots"
    
    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    portfolio_id: Mapped[UUID] = mapped_column(ForeignKey("portfolios.id"), nullable=False)
    snapshot_date: Mapped[date] = mapped_column(Date, nullable=False)
    
    total_nav: Mapped[Decimal] = mapped_column(Numeric(20, 4), nullable=False)
    daily_return: Mapped[float] = mapped_column(Float, default=0)
    cumulative_return: Mapped[float] = mapped_column(Float, default=0)
    
    # Risk metrics
    volatility: Mapped[Optional[float]] = mapped_column(Float)
    var_95: Mapped[Optional[float]] = mapped_column(Float)
    var_99: Mapped[Optional[float]] = mapped_column(Float)
    cvar_95: Mapped[Optional[float]] = mapped_column(Float)
    max_drawdown: Mapped[Optional[float]] = mapped_column(Float)
    sharpe_ratio: Mapped[Optional[float]] = mapped_column(Float)
    
    # Position weights snapshot
    weights: Mapped[dict] = mapped_column(JSONB, default={})
    factor_exposures: Mapped[dict] = mapped_column(JSONB, default={})
    
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    
    # Relationships
    portfolio: Mapped["Portfolio"] = relationship(back_populates="snapshots")
    
    __table_args__ = (
        UniqueConstraint("portfolio_id", "snapshot_date", name="uq_snapshot"),
        Index("ix_snapshots_date", "snapshot_date"),
    )


# =============================================================================
# SECURITY MODELS
# =============================================================================

class Security(Base):
    """Security master data."""
    
    __tablename__ = "securities"
    
    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    ticker: Mapped[str] = mapped_column(String(50), unique=True, nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    
    # Identifiers
    cusip: Mapped[Optional[str]] = mapped_column(String(9))
    isin: Mapped[Optional[str]] = mapped_column(String(12))
    sedol: Mapped[Optional[str]] = mapped_column(String(7))
    bloomberg_id: Mapped[Optional[str]] = mapped_column(String(50))
    
    # Classification
    asset_class: Mapped[str] = mapped_column(String(100), nullable=False)
    sector: Mapped[Optional[str]] = mapped_column(String(100))
    industry: Mapped[Optional[str]] = mapped_column(String(100))
    country: Mapped[str] = mapped_column(String(3), default="USA")
    currency: Mapped[str] = mapped_column(String(3), default="USD")
    
    # Status
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())
    
    # Relationships
    positions: Mapped[list["Position"]] = relationship(back_populates="security")
    prices: Mapped[list["PriceHistory"]] = relationship(back_populates="security")


class PriceHistory(Base):
    """Historical price data (TimescaleDB hypertable)."""
    
    __tablename__ = "price_history"
    
    security_id: Mapped[UUID] = mapped_column(ForeignKey("securities.id"), primary_key=True)
    price_date: Mapped[date] = mapped_column(Date, primary_key=True)
    
    open_price: Mapped[Optional[Decimal]] = mapped_column(Numeric(20, 8))
    high_price: Mapped[Optional[Decimal]] = mapped_column(Numeric(20, 8))
    low_price: Mapped[Optional[Decimal]] = mapped_column(Numeric(20, 8))
    close_price: Mapped[Decimal] = mapped_column(Numeric(20, 8), nullable=False)
    adjusted_close: Mapped[Decimal] = mapped_column(Numeric(20, 8), nullable=False)
    volume: Mapped[Optional[int]] = mapped_column(Integer)
    
    # Pre-computed returns
    daily_return: Mapped[Optional[float]] = mapped_column(Float)
    
    # Relationships
    security: Mapped["Security"] = relationship(back_populates="prices")
    
    __table_args__ = (
        Index("ix_price_date", "price_date"),
    )


# =============================================================================
# FACTOR MODELS
# =============================================================================

class FactorDefinition(Base):
    """Factor definition master."""
    
    __tablename__ = "factor_definitions"
    
    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    name: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)
    category: Mapped[str] = mapped_column(String(50), nullable=False)  # macro, style, industry
    description: Mapped[Optional[str]] = mapped_column(Text)
    
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class FactorReturn(Base):
    """Daily factor returns (TimescaleDB hypertable)."""
    
    __tablename__ = "factor_returns"
    
    factor_id: Mapped[UUID] = mapped_column(ForeignKey("factor_definitions.id"), primary_key=True)
    return_date: Mapped[date] = mapped_column(Date, primary_key=True)
    
    factor_return: Mapped[float] = mapped_column(Float, nullable=False)
    
    __table_args__ = (
        Index("ix_factor_return_date", "return_date"),
    )


class FactorExposure(Base):
    """Security factor exposures."""
    
    __tablename__ = "factor_exposures"
    
    security_id: Mapped[UUID] = mapped_column(ForeignKey("securities.id"), primary_key=True)
    factor_id: Mapped[UUID] = mapped_column(ForeignKey("factor_definitions.id"), primary_key=True)
    exposure_date: Mapped[date] = mapped_column(Date, primary_key=True)
    
    exposure: Mapped[float] = mapped_column(Float, nullable=False)
    t_statistic: Mapped[Optional[float]] = mapped_column(Float)
    
    __table_args__ = (
        Index("ix_factor_exposure_date", "exposure_date"),
    )


# =============================================================================
# RISK ANALYTICS
# =============================================================================

class RiskCalculation(Base):
    """Stored risk calculation results."""
    
    __tablename__ = "risk_calculations"
    
    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    portfolio_id: Mapped[UUID] = mapped_column(ForeignKey("portfolios.id"), nullable=False)
    calculation_date: Mapped[date] = mapped_column(Date, nullable=False)
    calculation_type: Mapped[str] = mapped_column(String(50), nullable=False)  # var, cvar, factor_risk
    
    # Parameters
    confidence_level: Mapped[Optional[float]] = mapped_column(Float)
    horizon_days: Mapped[Optional[int]] = mapped_column(Integer)
    method: Mapped[Optional[str]] = mapped_column(String(50))
    
    # Results
    result_value: Mapped[float] = mapped_column(Float, nullable=False)
    component_values: Mapped[Optional[dict]] = mapped_column(JSONB)
    metadata_: Mapped[dict] = mapped_column(JSONB, default={})
    
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    
    __table_args__ = (
        Index("ix_risk_calc_portfolio_date", "portfolio_id", "calculation_date"),
    )


class StressTestResult(Base):
    """Stored stress test results."""
    
    __tablename__ = "stress_test_results"
    
    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    portfolio_id: Mapped[UUID] = mapped_column(ForeignKey("portfolios.id"), nullable=False)
    scenario_name: Mapped[str] = mapped_column(String(255), nullable=False)
    calculation_date: Mapped[date] = mapped_column(Date, nullable=False)
    
    portfolio_impact: Mapped[float] = mapped_column(Float, nullable=False)
    position_impacts: Mapped[dict] = mapped_column(JSONB, default={})
    factor_contributions: Mapped[dict] = mapped_column(JSONB, default={})
    
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


# =============================================================================
# ATTRIBUTION
# =============================================================================

class AttributionResult(Base):
    """Stored attribution analysis results."""
    
    __tablename__ = "attribution_results"
    
    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    portfolio_id: Mapped[UUID] = mapped_column(ForeignKey("portfolios.id"), nullable=False)
    benchmark_id: Mapped[Optional[UUID]] = mapped_column(PGUUID(as_uuid=True))
    
    start_date: Mapped[date] = mapped_column(Date, nullable=False)
    end_date: Mapped[date] = mapped_column(Date, nullable=False)
    
    method: Mapped[str] = mapped_column(String(50), nullable=False)  # brinson, geometric
    
    portfolio_return: Mapped[float] = mapped_column(Float, nullable=False)
    benchmark_return: Mapped[float] = mapped_column(Float, nullable=False)
    active_return: Mapped[float] = mapped_column(Float, nullable=False)
    
    allocation_effect: Mapped[float] = mapped_column(Float, nullable=False)
    selection_effect: Mapped[float] = mapped_column(Float, nullable=False)
    interaction_effect: Mapped[float] = mapped_column(Float, default=0)
    
    segment_attribution: Mapped[dict] = mapped_column(JSONB, default={})
    
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


# =============================================================================
# USERS & AUTHENTICATION
# =============================================================================

class User(Base):
    """User accounts."""
    
    __tablename__ = "users"
    
    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    
    first_name: Mapped[str] = mapped_column(String(100), nullable=False)
    last_name: Mapped[str] = mapped_column(String(100), nullable=False)
    
    role: Mapped[str] = mapped_column(String(50), default="analyst")  # analyst, pm, admin
    department: Mapped[Optional[str]] = mapped_column(String(100))
    
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    is_superuser: Mapped[bool] = mapped_column(Boolean, default=False)
    
    last_login: Mapped[Optional[datetime]] = mapped_column(DateTime)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())


class AuditLog(Base):
    """Audit trail for all actions."""
    
    __tablename__ = "audit_logs"
    
    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    user_id: Mapped[Optional[UUID]] = mapped_column(ForeignKey("users.id"))
    
    action: Mapped[str] = mapped_column(String(100), nullable=False)
    resource_type: Mapped[str] = mapped_column(String(100), nullable=False)
    resource_id: Mapped[Optional[UUID]] = mapped_column(PGUUID(as_uuid=True))
    
    details: Mapped[dict] = mapped_column(JSONB, default={})
    ip_address: Mapped[Optional[str]] = mapped_column(String(45))
    
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    
    __table_args__ = (
        Index("ix_audit_created", "created_at"),
        Index("ix_audit_user", "user_id"),
    )
