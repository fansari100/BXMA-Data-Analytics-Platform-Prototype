"""
Risk Dashboard for BXMA Data Analytics Platform.

Provides real-time interactive dashboards using Plotly Dash:
- Portfolio overview and NAV tracking
- Risk metrics visualization (VaR, CVaR, drawdowns)
- Factor exposure analysis
- Performance attribution
- Stress test results

Designed for cross-functional collaboration with stakeholders.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Any
import numpy as np
from numpy.typing import NDArray


@dataclass
class DashboardConfig:
    """Configuration for risk dashboard."""
    
    port: int = 8050
    debug: bool = False
    title: str = "BXMA Risk Analytics Dashboard"
    refresh_interval_ms: int = 60000  # 1 minute
    theme: str = "dark"


class RiskDashboard:
    """
    Interactive Risk Dashboard using Plotly Dash.
    
    Features:
    - Real-time portfolio monitoring
    - Interactive risk decomposition
    - Factor exposure visualization
    - Attribution analysis
    - Stress testing interface
    """
    
    def __init__(self, config: DashboardConfig | None = None):
        """
        Initialize risk dashboard.
        
        Args:
            config: Dashboard configuration
        """
        self.config = config or DashboardConfig()
        self._app = None
    
    def create_app(self):
        """Create Dash application."""
        try:
            import dash
            from dash import dcc, html, Input, Output, State
            import plotly.graph_objects as go
            import plotly.express as px
        except ImportError:
            raise ImportError("Dash and Plotly required for dashboard")
        
        # Initialize Dash app
        app = dash.Dash(
            __name__,
            title=self.config.title,
            suppress_callback_exceptions=True,
        )
        
        # Layout
        app.layout = html.Div([
            # Header
            html.Div([
                html.H1(self.config.title, style={
                    'color': '#00d4ff',
                    'fontFamily': 'JetBrains Mono, monospace',
                    'marginBottom': '0',
                }),
                html.P("Blackstone Multi-Asset Investing Data Analytics Analytics", style={
                    'color': '#888',
                    'fontSize': '14px',
                }),
            ], style={'padding': '20px', 'borderBottom': '1px solid #333'}),
            
            # Tabs
            dcc.Tabs(id='dashboard-tabs', value='overview', children=[
                dcc.Tab(label='Overview', value='overview'),
                dcc.Tab(label='Risk Metrics', value='risk'),
                dcc.Tab(label='Factor Analysis', value='factors'),
                dcc.Tab(label='Attribution', value='attribution'),
                dcc.Tab(label='Stress Testing', value='stress'),
            ], style={'marginBottom': '20px'}),
            
            # Content
            html.Div(id='tab-content', style={'padding': '20px'}),
            
            # Auto-refresh
            dcc.Interval(
                id='interval-component',
                interval=self.config.refresh_interval_ms,
                n_intervals=0
            ),
            
            # Store for data
            dcc.Store(id='portfolio-data'),
            dcc.Store(id='risk-data'),
            
        ], style={
            'backgroundColor': '#0a0a0a',
            'minHeight': '100vh',
            'fontFamily': 'Inter, sans-serif',
        })
        
        # Callbacks
        @app.callback(
            Output('tab-content', 'children'),
            Input('dashboard-tabs', 'value'),
            Input('interval-component', 'n_intervals'),
        )
        def render_tab(tab, n):
            if tab == 'overview':
                return self._render_overview()
            elif tab == 'risk':
                return self._render_risk_metrics()
            elif tab == 'factors':
                return self._render_factor_analysis()
            elif tab == 'attribution':
                return self._render_attribution()
            elif tab == 'stress':
                return self._render_stress_testing()
            return html.Div("Select a tab")
        
        self._app = app
        return app
    
    def _render_overview(self):
        """Render portfolio overview tab."""
        try:
            from dash import html, dcc
            import plotly.graph_objects as go
        except ImportError:
            return None
        
        # Sample data for demonstration
        dates = [date(2024, 1, i+1) for i in range(30)]
        nav = 100 * np.cumprod(1 + np.random.randn(30) * 0.01)
        
        # NAV chart
        nav_fig = go.Figure()
        nav_fig.add_trace(go.Scatter(
            x=dates, y=nav,
            mode='lines',
            name='Portfolio NAV',
            line=dict(color='#00d4ff', width=2),
            fill='tozeroy',
            fillcolor='rgba(0, 212, 255, 0.1)',
        ))
        nav_fig.update_layout(
            title='Portfolio NAV',
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=40, r=40, t=60, b=40),
        )
        
        return html.Div([
            # KPI Cards
            html.Div([
                self._kpi_card("Total NAV", "$90.2B", "+2.4%", "positive"),
                self._kpi_card("Daily Return", "+0.34%", "+15 bps", "positive"),
                self._kpi_card("VaR (95%)", "1.23%", "-5 bps", "neutral"),
                self._kpi_card("Max Drawdown", "-4.2%", "from peak", "negative"),
            ], style={
                'display': 'grid',
                'gridTemplateColumns': 'repeat(4, 1fr)',
                'gap': '20px',
                'marginBottom': '30px',
            }),
            
            # Charts
            html.Div([
                dcc.Graph(figure=nav_fig, style={'height': '400px'}),
            ]),
        ])
    
    def _render_risk_metrics(self):
        """Render risk metrics tab."""
        try:
            from dash import html, dcc
            import plotly.graph_objects as go
        except ImportError:
            return None
        
        # VaR breakdown chart
        var_fig = go.Figure()
        
        categories = ['VaR 95%', 'VaR 99%', 'CVaR 95%', 'CVaR 99%']
        values = [1.23, 2.14, 1.87, 3.21]
        
        var_fig.add_trace(go.Bar(
            x=categories,
            y=values,
            marker_color=['#00d4ff', '#00d4ff', '#ff6b6b', '#ff6b6b'],
        ))
        var_fig.update_layout(
            title='Risk Metrics (%)',
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
        )
        
        # Risk contribution pie
        contrib_fig = go.Figure()
        contrib_fig.add_trace(go.Pie(
            labels=['Equity', 'Fixed Income', 'Alternatives', 'Currency'],
            values=[45, 25, 20, 10],
            hole=0.4,
            marker_colors=['#00d4ff', '#00ff88', '#ffaa00', '#ff6b6b'],
        ))
        contrib_fig.update_layout(
            title='Risk Contribution by Asset Class',
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
        )
        
        return html.Div([
            html.Div([
                dcc.Graph(figure=var_fig, style={'height': '350px'}),
                dcc.Graph(figure=contrib_fig, style={'height': '350px'}),
            ], style={
                'display': 'grid',
                'gridTemplateColumns': '1fr 1fr',
                'gap': '20px',
            }),
        ])
    
    def _render_factor_analysis(self):
        """Render factor analysis tab."""
        try:
            from dash import html, dcc
            import plotly.graph_objects as go
        except ImportError:
            return None
        
        factors = ['Market', 'Size', 'Value', 'Momentum', 'Quality', 'Volatility']
        exposures = [1.05, 0.12, -0.23, 0.45, 0.31, -0.15]
        
        factor_fig = go.Figure()
        colors = ['#00d4ff' if e > 0 else '#ff6b6b' for e in exposures]
        
        factor_fig.add_trace(go.Bar(
            x=factors,
            y=exposures,
            marker_color=colors,
        ))
        factor_fig.update_layout(
            title='Factor Exposures',
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
        )
        factor_fig.add_hline(y=0, line_dash="dash", line_color="gray")
        
        return html.Div([
            dcc.Graph(figure=factor_fig, style={'height': '400px'}),
        ])
    
    def _render_attribution(self):
        """Render attribution analysis tab."""
        try:
            from dash import html, dcc
            import plotly.graph_objects as go
        except ImportError:
            return None
        
        # Waterfall chart for attribution
        attr_fig = go.Figure(go.Waterfall(
            name="Attribution",
            orientation="v",
            measure=["relative", "relative", "relative", "total"],
            x=["Allocation", "Selection", "Interaction", "Active Return"],
            y=[0.15, 0.28, -0.05, 0.38],
            connector={"line": {"color": "rgba(255,255,255,0.3)"}},
            increasing={"marker": {"color": "#00ff88"}},
            decreasing={"marker": {"color": "#ff6b6b"}},
            totals={"marker": {"color": "#00d4ff"}},
        ))
        attr_fig.update_layout(
            title='Performance Attribution (%)',
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
        )
        
        return html.Div([
            dcc.Graph(figure=attr_fig, style={'height': '400px'}),
        ])
    
    def _render_stress_testing(self):
        """Render stress testing tab."""
        try:
            from dash import html, dcc
            import plotly.graph_objects as go
        except ImportError:
            return None
        
        scenarios = ['2008 Crisis', 'COVID Crash', '2022 Rates', 'EM Crisis', 'Credit Crisis']
        impacts = [-15.2, -12.4, -8.7, -10.3, -11.8]
        
        stress_fig = go.Figure()
        stress_fig.add_trace(go.Bar(
            x=scenarios,
            y=impacts,
            marker_color='#ff6b6b',
        ))
        stress_fig.update_layout(
            title='Stress Test Portfolio Impact (%)',
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
        )
        
        return html.Div([
            dcc.Graph(figure=stress_fig, style={'height': '400px'}),
        ])
    
    def _kpi_card(self, title: str, value: str, change: str, sentiment: str):
        """Create a KPI card component."""
        try:
            from dash import html
        except ImportError:
            return None
        
        colors = {
            'positive': '#00ff88',
            'negative': '#ff6b6b',
            'neutral': '#888',
        }
        
        return html.Div([
            html.P(title, style={
                'color': '#888',
                'fontSize': '12px',
                'marginBottom': '5px',
            }),
            html.H2(value, style={
                'color': '#fff',
                'fontSize': '28px',
                'marginBottom': '5px',
                'fontFamily': 'JetBrains Mono, monospace',
            }),
            html.P(change, style={
                'color': colors.get(sentiment, '#888'),
                'fontSize': '14px',
            }),
        ], style={
            'backgroundColor': '#1a1a1a',
            'padding': '20px',
            'borderRadius': '8px',
            'border': '1px solid #333',
        })
    
    def run(self, host: str = "0.0.0.0"):
        """
        Run the dashboard server.
        
        Args:
            host: Host to bind to
        """
        if self._app is None:
            self.create_app()
        
        self._app.run_server(
            host=host,
            port=self.config.port,
            debug=self.config.debug,
        )
