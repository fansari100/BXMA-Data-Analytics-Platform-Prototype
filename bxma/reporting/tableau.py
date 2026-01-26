"""
Tableau Integration Module
==========================

Provides seamless integration with Tableau for:
- Data export to Tableau-compatible formats
- Tableau Server/Online publishing
- Hyper file generation
- Dashboard embedding

This is a CRITICAL requirement for the BXMA Risk/Quant platform.

Author: BXMA Quant Team
Date: January 2026
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Any, Literal
from pathlib import Path
import json
import csv
import io


@dataclass
class TableauDataSource:
    """Configuration for a Tableau data source."""
    
    name: str
    description: str = ""
    
    # Connection type
    connection_type: Literal["extract", "live", "published"] = "extract"
    
    # Server config (for published sources)
    server_url: str = ""
    site_id: str = ""
    project_name: str = "BXMA Risk Analytics"
    
    # Credentials
    token_name: str = ""
    token_value: str = ""
    
    # Refresh schedule
    refresh_frequency: str = "daily"
    refresh_time: str = "06:00"


@dataclass
class TableauExportConfig:
    """Configuration for Tableau data export."""
    
    output_format: Literal["csv", "hyper", "json", "tde"] = "csv"
    output_path: str = "./tableau_exports"
    
    # Column mappings
    column_aliases: dict[str, str] = field(default_factory=dict)
    
    # Data types
    force_types: dict[str, str] = field(default_factory=dict)
    
    # Filters
    include_columns: list[str] | None = None
    exclude_columns: list[str] | None = None


class TableauExporter:
    """
    Export data to Tableau-compatible formats.
    
    Supports:
    - CSV with proper formatting
    - JSON for Web Data Connector
    - Hyper extracts (via tableauhyperapi)
    """
    
    def __init__(self, config: TableauExportConfig | None = None):
        self.config = config or TableauExportConfig()
        self._ensure_output_dir()
    
    def _ensure_output_dir(self):
        """Create output directory if needed."""
        Path(self.config.output_path).mkdir(parents=True, exist_ok=True)
    
    def export_risk_metrics(
        self,
        data: dict[str, Any],
        filename: str = "risk_metrics",
    ) -> str:
        """
        Export risk metrics data for Tableau.
        
        Args:
            data: Risk metrics dictionary
            filename: Output filename (without extension)
            
        Returns:
            Path to exported file
        """
        # Flatten nested data
        flat_data = self._flatten_dict(data)
        
        # Add metadata
        flat_data["export_timestamp"] = datetime.now().isoformat()
        flat_data["source"] = "BXMA Risk Platform"
        
        return self._export_single_record(flat_data, filename)
    
    def export_portfolio_data(
        self,
        positions: list[dict],
        portfolio_name: str,
        as_of_date: date | None = None,
        filename: str = "portfolio_positions",
    ) -> str:
        """
        Export portfolio positions for Tableau.
        
        Args:
            positions: List of position dictionaries
            portfolio_name: Name of the portfolio
            as_of_date: Valuation date
            filename: Output filename
            
        Returns:
            Path to exported file
        """
        # Enrich positions with metadata
        enriched = []
        for pos in positions:
            record = {
                "portfolio_name": portfolio_name,
                "as_of_date": (as_of_date or date.today()).isoformat(),
                **pos,
            }
            enriched.append(record)
        
        return self._export_records(enriched, filename)
    
    def export_time_series(
        self,
        dates: list[date],
        values: dict[str, list[float]],
        filename: str = "time_series",
    ) -> str:
        """
        Export time series data for Tableau.
        
        Args:
            dates: List of dates
            values: Dictionary of series name -> values
            filename: Output filename
            
        Returns:
            Path to exported file
        """
        records = []
        for i, dt in enumerate(dates):
            record = {"date": dt.isoformat()}
            for series_name, series_values in values.items():
                if i < len(series_values):
                    record[series_name] = series_values[i]
            records.append(record)
        
        return self._export_records(records, filename)
    
    def export_attribution(
        self,
        attribution_data: dict[str, Any],
        filename: str = "attribution",
    ) -> str:
        """
        Export attribution analysis for Tableau.
        
        Args:
            attribution_data: Attribution results
            filename: Output filename
            
        Returns:
            Path to exported file
        """
        # Flatten segment-level attribution
        records = []
        
        if "segment_attribution" in attribution_data:
            for segment, effects in attribution_data["segment_attribution"].items():
                record = {
                    "segment": segment,
                    "allocation_effect": effects.get("allocation", 0),
                    "selection_effect": effects.get("selection", 0),
                    "interaction_effect": effects.get("interaction", 0),
                    "total_effect": sum(effects.values()),
                    "portfolio_return": attribution_data.get("portfolio_return", 0),
                    "benchmark_return": attribution_data.get("benchmark_return", 0),
                    "active_return": attribution_data.get("active_return", 0),
                }
                records.append(record)
        
        return self._export_records(records, filename)
    
    def export_factor_exposures(
        self,
        exposures: dict[str, dict[str, float]],
        filename: str = "factor_exposures",
    ) -> str:
        """
        Export factor exposures for Tableau heatmap.
        
        Args:
            exposures: Dict of asset_id -> {factor: exposure}
            filename: Output filename
            
        Returns:
            Path to exported file
        """
        records = []
        for asset_id, factors in exposures.items():
            for factor, exposure in factors.items():
                records.append({
                    "asset_id": asset_id,
                    "factor": factor,
                    "exposure": exposure,
                })
        
        return self._export_records(records, filename)
    
    def export_var_history(
        self,
        var_data: list[dict],
        filename: str = "var_history",
    ) -> str:
        """
        Export VaR history for Tableau time series.
        
        Args:
            var_data: List of VaR records with dates
            filename: Output filename
            
        Returns:
            Path to exported file
        """
        return self._export_records(var_data, filename)
    
    def _export_single_record(self, data: dict, filename: str) -> str:
        """Export a single record."""
        return self._export_records([data], filename)
    
    def _export_records(self, records: list[dict], filename: str) -> str:
        """Export multiple records."""
        if not records:
            return ""
        
        if self.config.output_format == "csv":
            return self._export_csv(records, filename)
        elif self.config.output_format == "json":
            return self._export_json(records, filename)
        elif self.config.output_format == "hyper":
            return self._export_hyper(records, filename)
        else:
            raise ValueError(f"Unsupported format: {self.config.output_format}")
    
    def _export_csv(self, records: list[dict], filename: str) -> str:
        """Export to CSV format."""
        filepath = Path(self.config.output_path) / f"{filename}.csv"
        
        # Get all columns
        columns = set()
        for record in records:
            columns.update(record.keys())
        columns = sorted(columns)
        
        # Apply column filters
        if self.config.include_columns:
            columns = [c for c in columns if c in self.config.include_columns]
        if self.config.exclude_columns:
            columns = [c for c in columns if c not in self.config.exclude_columns]
        
        # Apply aliases
        header = [self.config.column_aliases.get(c, c) for c in columns]
        
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            
            for record in records:
                row = [record.get(c, "") for c in columns]
                writer.writerow(row)
        
        return str(filepath)
    
    def _export_json(self, records: list[dict], filename: str) -> str:
        """Export to JSON format for Web Data Connector."""
        filepath = Path(self.config.output_path) / f"{filename}.json"
        
        # Structure for Tableau WDC
        output = {
            "metadata": {
                "source": "BXMA Risk Platform",
                "exported_at": datetime.now().isoformat(),
                "record_count": len(records),
            },
            "data": records,
        }
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, default=str)
        
        return str(filepath)
    
    def _export_hyper(self, records: list[dict], filename: str) -> str:
        """
        Export to Tableau Hyper format.
        
        Requires tableauhyperapi package.
        """
        try:
            from tableauhyperapi import (
                HyperProcess, Telemetry, Connection, CreateMode,
                TableDefinition, SqlType, Inserter, TableName
            )
        except ImportError:
            # Fallback to CSV if Hyper API not available
            print("Warning: tableauhyperapi not installed, falling back to CSV")
            return self._export_csv(records, filename)
        
        filepath = Path(self.config.output_path) / f"{filename}.hyper"
        
        # Infer schema from data
        if not records:
            return str(filepath)
        
        sample = records[0]
        columns = []
        
        for key, value in sample.items():
            if isinstance(value, (int, np.integer)):
                sql_type = SqlType.big_int()
            elif isinstance(value, (float, np.floating)):
                sql_type = SqlType.double()
            elif isinstance(value, bool):
                sql_type = SqlType.bool()
            elif isinstance(value, (date, datetime)):
                sql_type = SqlType.date()
            else:
                sql_type = SqlType.text()
            
            col_name = self.config.column_aliases.get(key, key)
            columns.append(TableDefinition.Column(col_name, sql_type))
        
        # Create Hyper file
        with HyperProcess(telemetry=Telemetry.DO_NOT_SEND_USAGE_DATA_TO_TABLEAU) as hyper:
            with Connection(
                hyper.endpoint,
                str(filepath),
                CreateMode.CREATE_AND_REPLACE
            ) as connection:
                table_def = TableDefinition(
                    TableName("Extract", "Extract"),
                    columns
                )
                connection.catalog.create_table(table_def)
                
                with Inserter(connection, table_def) as inserter:
                    for record in records:
                        row = [record.get(col.name.unescaped, None) for col in columns]
                        inserter.add_row(row)
                    inserter.execute()
        
        return str(filepath)
    
    def _flatten_dict(self, d: dict, parent_key: str = "", sep: str = "_") -> dict:
        """Flatten nested dictionary."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep).items())
            elif isinstance(v, (list, np.ndarray)):
                if len(v) > 0 and isinstance(v[0], (int, float, np.number)):
                    # Store as JSON string for arrays
                    items.append((new_key, json.dumps(list(v) if isinstance(v, np.ndarray) else v)))
                else:
                    items.append((new_key, str(v)))
            else:
                items.append((new_key, v))
        return dict(items)


class TableauServerPublisher:
    """
    Publish data sources and workbooks to Tableau Server/Online.
    
    Requires tableauserverclient package.
    """
    
    def __init__(self, config: TableauDataSource):
        self.config = config
        self._server = None
        self._auth = None
    
    def connect(self) -> bool:
        """Connect to Tableau Server."""
        try:
            import tableauserverclient as TSC
            
            self._auth = TSC.PersonalAccessTokenAuth(
                self.config.token_name,
                self.config.token_value,
                self.config.site_id,
            )
            self._server = TSC.Server(self.config.server_url, use_server_version=True)
            self._server.auth.sign_in(self._auth)
            
            return True
            
        except ImportError:
            print("Warning: tableauserverclient not installed")
            return False
        except Exception as e:
            print(f"Failed to connect to Tableau Server: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from Tableau Server."""
        if self._server:
            self._server.auth.sign_out()
    
    def publish_datasource(
        self,
        filepath: str,
        name: str | None = None,
        project_name: str | None = None,
    ) -> str | None:
        """
        Publish a data source to Tableau Server.
        
        Args:
            filepath: Path to .hyper or .tde file
            name: Data source name
            project_name: Target project
            
        Returns:
            Data source ID on success
        """
        if not self._server:
            return None
        
        try:
            import tableauserverclient as TSC
            
            # Find project
            project_name = project_name or self.config.project_name
            all_projects, _ = self._server.projects.get()
            project = next(
                (p for p in all_projects if p.name == project_name),
                None
            )
            
            if not project:
                print(f"Project not found: {project_name}")
                return None
            
            # Create data source
            datasource = TSC.DatasourceItem(project.id, name or self.config.name)
            
            # Publish
            datasource = self._server.datasources.publish(
                datasource,
                filepath,
                TSC.Server.PublishMode.Overwrite,
            )
            
            return datasource.id
            
        except Exception as e:
            print(f"Failed to publish data source: {e}")
            return None
    
    def refresh_datasource(self, datasource_id: str) -> bool:
        """Trigger a refresh of a published data source."""
        if not self._server:
            return False
        
        try:
            self._server.datasources.refresh(datasource_id)
            return True
        except Exception as e:
            print(f"Failed to refresh data source: {e}")
            return False


# Pre-configured exports for common BXMA use cases
BXMA_TABLEAU_EXPORTS = {
    "daily_risk_report": TableauExportConfig(
        output_format="csv",
        output_path="./tableau_exports/daily",
        column_aliases={
            "var_95": "VaR (95%)",
            "var_99": "VaR (99%)",
            "cvar_95": "CVaR (95%)",
            "volatility": "Portfolio Volatility",
            "sharpe_ratio": "Sharpe Ratio",
        },
    ),
    "attribution_report": TableauExportConfig(
        output_format="csv",
        output_path="./tableau_exports/attribution",
        column_aliases={
            "allocation_effect": "Allocation Effect",
            "selection_effect": "Selection Effect",
            "interaction_effect": "Interaction Effect",
        },
    ),
    "factor_analysis": TableauExportConfig(
        output_format="hyper",
        output_path="./tableau_exports/factors",
    ),
}
