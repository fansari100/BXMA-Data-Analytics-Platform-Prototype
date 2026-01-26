-- BXMA Database Initialization Script
-- Creates TimescaleDB extensions and initial schema

-- Enable extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- Create schemas
CREATE SCHEMA IF NOT EXISTS portfolio;
CREATE SCHEMA IF NOT EXISTS risk;
CREATE SCHEMA IF NOT EXISTS analytics;
CREATE SCHEMA IF NOT EXISTS audit;

-- Grant permissions
GRANT ALL PRIVILEGES ON SCHEMA portfolio TO bxma;
GRANT ALL PRIVILEGES ON SCHEMA risk TO bxma;
GRANT ALL PRIVILEGES ON SCHEMA analytics TO bxma;
GRANT ALL PRIVILEGES ON SCHEMA audit TO bxma;

-- Create price_history as TimescaleDB hypertable
CREATE TABLE IF NOT EXISTS portfolio.price_history (
    security_id UUID NOT NULL,
    price_date DATE NOT NULL,
    open_price NUMERIC(20, 8),
    high_price NUMERIC(20, 8),
    low_price NUMERIC(20, 8),
    close_price NUMERIC(20, 8) NOT NULL,
    adjusted_close NUMERIC(20, 8) NOT NULL,
    volume BIGINT,
    daily_return DOUBLE PRECISION,
    PRIMARY KEY (security_id, price_date)
);

-- Convert to hypertable
SELECT create_hypertable(
    'portfolio.price_history', 
    'price_date',
    if_not_exists => TRUE
);

-- Create factor_returns as TimescaleDB hypertable  
CREATE TABLE IF NOT EXISTS risk.factor_returns (
    factor_id UUID NOT NULL,
    return_date DATE NOT NULL,
    factor_return DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (factor_id, return_date)
);

SELECT create_hypertable(
    'risk.factor_returns',
    'return_date',
    if_not_exists => TRUE
);

-- Create portfolio_snapshots as hypertable
CREATE TABLE IF NOT EXISTS analytics.portfolio_snapshots (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    portfolio_id UUID NOT NULL,
    snapshot_date DATE NOT NULL,
    total_nav NUMERIC(20, 4) NOT NULL,
    daily_return DOUBLE PRECISION,
    cumulative_return DOUBLE PRECISION,
    volatility DOUBLE PRECISION,
    var_95 DOUBLE PRECISION,
    var_99 DOUBLE PRECISION,
    cvar_95 DOUBLE PRECISION,
    max_drawdown DOUBLE PRECISION,
    sharpe_ratio DOUBLE PRECISION,
    weights JSONB DEFAULT '{}',
    factor_exposures JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_snapshots_portfolio_date ON analytics.portfolio_snapshots(portfolio_id, snapshot_date);

-- Create audit log table
CREATE TABLE IF NOT EXISTS audit.logs (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    user_id UUID,
    action VARCHAR(100) NOT NULL,
    resource_type VARCHAR(100) NOT NULL,
    resource_id UUID,
    details JSONB DEFAULT '{}',
    ip_address INET,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_audit_created ON audit.logs(created_at);
CREATE INDEX idx_audit_user ON audit.logs(user_id);

-- Create continuous aggregates for performance
CREATE MATERIALIZED VIEW IF NOT EXISTS analytics.daily_portfolio_stats
WITH (timescaledb.continuous) AS
SELECT 
    portfolio_id,
    time_bucket('1 day', snapshot_date) AS bucket,
    AVG(daily_return) as avg_return,
    STDDEV(daily_return) as volatility,
    MIN(daily_return) as min_return,
    MAX(daily_return) as max_return
FROM analytics.portfolio_snapshots
GROUP BY portfolio_id, time_bucket('1 day', snapshot_date)
WITH NO DATA;

-- Add compression policy for older data
SELECT add_compression_policy('portfolio.price_history', INTERVAL '1 year');
SELECT add_compression_policy('risk.factor_returns', INTERVAL '1 year');

-- Create retention policy (optional, for demo keeping all data)
-- SELECT add_retention_policy('portfolio.price_history', INTERVAL '10 years');

COMMENT ON DATABASE bxma IS 'BXMA Risk/Quant Platform Database';
