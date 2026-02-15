"""
Brent Oil Change Point Analysis - Interactive Dashboard
========================================================

A Streamlit dashboard for exploring Bayesian change point detection 
in Brent oil prices with business impact analysis.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import os
from datetime import datetime
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Page configuration
st.set_page_config(
    page_title="Brent Oil Change Point Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, 'data', 'BrentOilPrices.csv')
EVENTS_PATH = os.path.join(BASE_DIR, 'events.csv')
RESULTS_PATH = os.path.join(BASE_DIR, 'data', 'model_results.json')


#  ===================
# Helper Functions
# ====================

@st.cache_data
def load_price_data():
    """Load and preprocess Brent oil price data."""
    try:
        df = pd.read_csv(DATA_PATH)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')
        df.set_index('Date', inplace=True)
        
        # Calculate log returns
        df['Log_Price'] = np.log(df['Price'])
        df['Log_Returns'] = df['Log_Price'].diff()
        df['Volatility_21'] = df['Log_Returns'].rolling(window=21).std() * np.sqrt(252)
        
        return df
    except FileNotFoundError:
        st.error(f"Data file not found: {DATA_PATH}")
        return None


@st.cache_data
def load_events():
    """Load geopolitical and economic events."""
    try:
        events = pd.read_csv(EVENTS_PATH)
        events['Date'] = pd.to_datetime(events['Date'])
        return events
    except FileNotFoundError:
        st.warning(f"Events file not found: {EVENTS_PATH}")
        return pd.DataFrame()


@st.cache_data
def load_model_results():
    """Load model results if available."""
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH, 'r') as f:
            return json.load(f)
    return None


def create_price_chart(df, events, change_points=None, date_range=None):
    """Create interactive price chart with events and change points."""
    if date_range:
        df = df.loc[date_range[0]:date_range[1]]
    
    fig = go.Figure()
    
    # Price line
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['Price'],
        mode='lines',
        name='Brent Oil Price',
        line=dict(color='#1f77b4', width=2),
        hovertemplate='%{x|%Y-%m-%d}<br>Price: $%{y:.2f}<extra></extra>'
    ))
    
    # Add change points if available
    if change_points:
        for cp in change_points:
            cp_date = pd.to_datetime(cp['date'])
            if date_range is None or (date_range[0] <= cp_date <= date_range[1]):
                fig.add_vline(
                    x=cp_date,
                    line_dash="dash",
                    line_color="red",
                    annotation_text="Change Point",
                    annotation_position="top"
                )
    
    # Add events
    if not events.empty:
        for _, event in events.iterrows():
            if date_range is None or (date_range[0] <= event['Date'] <= date_range[1]):
                fig.add_vline(
                    x=event['Date'],
                    line_dash="dot",
                    line_color="gray",
                    opacity=0.5,
                    annotation_text=event['Event'][:20] + "...",
                    annotation_position="top right",
                    annotation_font_size=10
                )
    
    fig.update_layout(
        title="Brent Oil Price History with Change Points and Events",
        xaxis_title="Date",
        yaxis_title="Price (USD/barrel)",
        hovermode='x unified',
        height=500,
        template="plotly_white"
    )
    
    return fig


def create_volatility_chart(df, date_range=None):
    """Create volatility chart."""
    if date_range:
        df = df.loc[date_range[0]:date_range[1]]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['Volatility_21'],
        mode='lines',
        name='21-day Volatility',
        line=dict(color='#ff7f0e', width=2),
        fill='tozeroy',
        fillcolor='rgba(255, 127, 14, 0.2)'
    ))
    
    fig.update_layout(
        title="Annualized Volatility (21-day Rolling)",
        xaxis_title="Date",
        yaxis_title="Volatility",
        hovermode='x unified',
        height=300,
        template="plotly_white"
    )
    
    return fig


# ====================
# Main Application
# ====================

def main():
    """Main dashboard application."""
    
    # Sidebar
    with st.sidebar:
        st.title("📊 Navigation")
        page = st.radio(
            "Select Page",
            ["Overview", "Price Analysis", "Change Points", "Model Diagnostics", "Business Impact"]
        )
        
        st.markdown("---")
        st.markdown("### About")
        st.markdown("""
        This dashboard analyzes Brent oil price change points using Bayesian inference.
        
        **Key Features:**
        - Detects structural breaks in oil prices
        - Correlates with geopolitical events
        - Quantifies regime shifts
        - Provides risk insights
        """)
    
    # Load data
    df = load_price_data()
    events = load_events()
    model_results = load_model_results()
    
    if df is None:
        st.error("Failed to load price data. Please check data files.")
        return
    
    # Page routing
    if page == "Overview":
        show_overview(df, events, model_results)
    elif page == "Price Analysis":
        show_price_analysis(df, events, model_results)
    elif page == "Change Points":
        show_change_points(df, events, model_results)
    elif page == "Model Diagnostics":
        show_diagnostics(model_results)
    elif page == "Business Impact":
        show_business_impact(df, model_results)


def show_overview(df, events, model_results):
    """Overview page with problem statement and key metrics."""
    st.title("🛢️ Brent Oil Change Point Analysis")
    st.markdown("### Quantifying Market Regime Shifts for Risk Management")
    
    # Problem statement
    st.markdown("""
    ## Business Problem
    
    Oil price volatility creates significant risk for:
    - **Portfolio managers** hedging energy exposure
    - **Trading desks** optimizing commodity positions  
    - **Risk officers** setting Value-at-Risk (VaR) parameters
    - **Analysts** forecasting macro trends
    
    **Challenge**: Traditional models assume stationary volatility, failing to capture structural breaks 
    caused by geopolitical shocks (wars, pandemics, OPEC decisions).
    
    **Solution**: Bayesian change point detection identifies regime transitions with probabilistic uncertainty, 
    enabling dynamic risk models.
    """)
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Data Range",
            f"{df.index.min().year} - {df.index.max().year}",
            delta=f"{len(df)} days"
        )
    
    with col2:
        current_price = df['Price'].iloc[-1]
        st.metric(
            "Latest Price",
            f"${current_price:.2f}",
            delta=f"{((current_price / df['Price'].iloc[-30]) - 1) * 100:.1f}% (30d)"
        )
    
    with col3:
        if model_results and 'change_points' in model_results:
            n_change_points = len(model_results['change_points'])
        else:
            n_change_points = "N/A"
        st.metric(
            "Change Points Detected",
            n_change_points
        )
    
    with col4:
        current_vol = df['Volatility_21'].iloc[-1]
        st.metric(
            "Current Volatility",
            f"{current_vol:.1%}",
            delta="Annualized"
        )
    
    # Model status
    st.markdown("---")
    if model_results:
        st.success("✅ Model results available")
        if 'convergence' in model_results:
            if model_results['convergence'].get('r_hat_ok', False):
                st.info("✅ Model convergence: PASSED (R-hat < 1.05)")
            else:
                st.warning("⚠️ Model convergence: CHECK REQUIRED")
    else:
        st.warning("⚠️ Model has not been run yet. Navigate to Model Diagnostics to run analysis.")
    
    # Quick chart
    st.markdown("### Price History Overview")
    fig = create_price_chart(df, events, 
                            model_results['change_points'] if model_results else None)
    st.plotly_chart(fig, use_container_width=True)


def show_price_analysis(df, events, model_results):
    """Detailed price analysis page."""
    st.title("📈 Price Analysis")
    
    # Date range selector
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            "Start Date",
            value=df.index.min(),
            min_value=df.index.min(),
            max_value=df.index.max()
        )
    with col2:
        end_date = st.date_input(
            "End Date",
            value=df.index.max(),
            min_value=df.index.min(),
            max_value=df.index.max()
        )
    
    date_range = (pd.Timestamp(start_date), pd.Timestamp(end_date))
    
    # Price chart
    st.markdown("### Price Chart with Events")
    fig_price = create_price_chart(df, events, 
                                   model_results['change_points'] if model_results else None,
                                   date_range)
    st.plotly_chart(fig_price, use_container_width=True)
    
    # Volatility chart
    st.markdown("### Volatility Analysis")
    fig_vol = create_volatility_chart(df, date_range)
    st.plotly_chart(fig_vol, use_container_width=True)
    
    # Statistics
    st.markdown("### Summary Statistics")
    df_range = df.loc[date_range[0]:date_range[1]]
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Mean Price", f"${df_range['Price'].mean():.2f}")
        st.metric("Std Dev", f"${df_range['Price'].std():.2f}")
    with col2:
        st.metric("Min Price", f"${df_range['Price'].min():.2f}")
        st.metric("Max Price", f"${df_range['Price'].max():.2f}")
    with col3:
        st.metric("Mean Return", f"{df_range['Log_Returns'].mean() * 252:.2%}")
        st.metric("Volatility", f"{df_range['Log_Returns'].std() * np.sqrt(252):.2%}")


def show_change_points(df, events, model_results):
    """Change points analysis page."""
    st.title("🎯 Change Point Detection")
    
    if not model_results or 'change_points' not in model_results:
        st.warning("⚠️ No model results available. Please run the model first.")
        st.markdown("""
        To generate change point detection results:
        ```bash
        python src/run_model.py
        ```
        """)
        return
    
    change_points = model_results['change_points']
    
    st.markdown(f"### Detected {len(change_points)} Change Point(s)")
    
    # Change points table
    if change_points:
        cp_df = pd.DataFrame([
            {
                'Date': cp['date'],
                'Index': cp.get('index', 'N/A'),
                'Credible Interval': f"{cp.get('credible_interval', ['N/A', 'N/A'])[0]} to {cp.get('credible_interval', ['N/A', 'N/A'])[1]}"
            }
            for cp in change_points
        ])
        st.dataframe(cp_df, use_container_width=True)
    
    # Regime parameters
    if 'regime_parameters' in model_results:
        st.markdown("### Regime Parameters")
        regime_df = pd.DataFrame(model_results['regime_parameters'])
        st.dataframe(regime_df, use_container_width=True)
    
    # Event correlation
    st.markdown("### Event Correlation")
    if not events.empty and change_points:
        st.markdown("Detected change points and nearby events:")
        
        for cp in change_points:
            cp_date = pd.to_datetime(cp['date'])
            # Find events within 30 days
            nearby_events = events[
                (events['Date'] >= cp_date - pd.Timedelta(days=30)) &
                (events['Date'] <= cp_date + pd.Timedelta(days=30))
            ]
            
            if not nearby_events.empty:
                st.markdown(f"**Change Point: {cp_date.date()}**")
                for _, event in nearby_events.iterrows():
                    days_diff = (event['Date'] - cp_date).days
                    st.markdown(f"- {event['Event']} ({event['Date'].date()}, {abs(days_diff)} days {'before' if days_diff < 0 else 'after'})")
            else:
                st.markdown(f"**Change Point: {cp_date.date()}** - No events found within ±30 days")


def show_diagnostics(model_results):
    """Model diagnostics page."""
    st.title("🔬 Model Diagnostics")
    
    if not model_results:
        st.warning("⚠️ No model results available.")
        return
    
    # Convergence metrics
    st.markdown("### Convergence Metrics")
    
    if 'convergence' in model_results:
        conv = model_results['convergence']
        
        col1, col2 = st.columns(2)
        with col1:
            r_hat_ok = conv.get('r_hat_ok', False)
            st.metric(
                "R-hat Check",
                "✅ PASSED" if r_hat_ok else "❌ FAILED",
                delta=f"Max: {conv.get('max_r_hat', 'N/A')}"
            )
        
        with col2:
            ess_ok = conv.get('ess_ok', False)
            st.metric(
                "ESS Check",
                "✅ PASSED" if ess_ok else "❌ FAILED"
            )
        
        # Interpretation
        st.markdown("""
        **Convergence Diagnostics:**
        - **R-hat < 1.05**: Chains have converged (recommended)
        - **ESS > 400**: Sufficient effective sample size for inference
        """)
    else:
        st.info("Convergence metrics not available in results.")
    
    # Model assumptions
    st.markdown("---")
    st.markdown("### Model Assumptions & Limitations")
    st.markdown("""
    **Assumptions:**
    1. **Discrete Transitions**: Market regimes change at specific points in time
    2. **Log-Return Stationarity**: Returns are locally stationary within regimes
    3. **Independent Observations**: Daily returns are not auto-correlated
    
    **Limitations:**
    1. **Correlation ≠ Causation**: Statistical breaks may coincide with events but don't prove causation
    2. **Dating Precision**: Market reactions may be anticipatory or lagged
    3. **Omitted Variables**: Model focuses on price; doesn't include supply/demand fundamentals
    """)
    
    st.markdown("---")
    st.markdown("### Methodology")
    st.markdown("""
    This analysis uses **Bayesian change point detection** with PyMC:
    
    - **Model**: Switching mean and variance
    - **Inference**: MCMC (NUTS + Metropolis)
    - **Samples**: 2000 draws × 2 chains (default)
    - **Target**: R-hat < 1.05, ESS > 400
    """)


def show_business_impact(df, model_results):
    """Business impact analysis page."""
    st.title("💼 Business Impact")
    
    st.markdown("""
    ## Risk Management Insights
    
    Understanding regime shifts enables:
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### Portfolio Risk
        - **Dynamic VaR**: Adjust Value-at-Risk by regime
        - **Hedging Strategy**: Time hedge adjustments to regime transitions
        - **Correlation Shifts**: Energy sector betas change across regimes
        """)
    
    with col2:
        st.markdown("""
        ### Trading Applications
        - **Momentum vs Mean Reversion**: Strategy selection by regime
        - **Position Sizing**: Scale exposure based on volatility regime
        - **Stop-Loss Levels**: Adaptive stops reflecting current risk
        """)
    
    # Quantified impacts
    if model_results and 'regime_parameters' in model_results:
        st.markdown("---")
        st.markdown("### Quantified Regime Shifts")
        
        regimes = model_results['regime_parameters']
        if len(regimes) >= 2:
            for i in range(len(regimes) - 1):
                r1 = regimes[i]
                r2 = regimes[i + 1]
                
                mean_shift = (r2['mean'] - r1['mean']) * 252 * 100  # Annualized %
                vol_shift = (r2['std'] - r1['std']) * np.sqrt(252) * 100  # Annualized %
                
                st.markdown(f"""
                **Regime {i+1} → Regime {i+2}:**
                - Mean return shift: **{mean_shift:+.1f}% per year**
                - Volatility shift: **{vol_shift:+.1f}% per year**
                """)
                
                # Risk interpretation
                if abs(vol_shift) > 10:
                    st.warning(f"⚠️ **High Impact**: Volatility changed by {abs(vol_shift):.1f}% - significant risk profile shift")
                elif abs(mean_shift) > 20:
                    st.info(f"📊 **Trend Change**: Mean return shifted by {abs(mean_shift):.1f}% annualized")
    
    # Action items
    st.markdown("---")
    st.markdown("### Recommended Actions")
    st.markdown("""
    1. **Update Risk Models**: Incorporate regime-dependent volatility in VaR calculations
    2. **Review Hedges**: Reassess exposure when approaching potential transition points
    3. **Monitor Leading Indicators**: Track geopolitical events that historically precede regime shifts
    4. **Scenario Planning**: Model portfolio performance under different regime scenarios
    """)


if __name__ == "__main__":
    main()
