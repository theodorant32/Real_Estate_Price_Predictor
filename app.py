import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys
import json
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent / "src"))

from predict import PricePredictor, MarketAnalyzer
from buy_vs_rent import BuyVsRentCalculator, quick_analysis
from recommender import PropertyRecommender, get_recommendations_for_budget
from heatmap import MarketHeatmapGenerator, generate_sample_heatmap_data
from chatbot import PropertyChatbot
from roi_calculator import ROICalculator, PropertyInputs, quick_roi_analysis
from scenario_simulator import ScenarioSimulator, ScenarioInputs, quick_scenario_analysis
from ensemble import AdvancedEnsemblePredictor
from neighborhood_graph import create_sample_graph, NeighborhoodGraph
from explainability import PredictionExplainer, MarketRegimeDetector
from case_studies import get_all_case_studies, compare_case_studies, create_case_study_presentation
from listing_scraper import LiveListingsScraper


# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="Propra - AI-Powered Canadian Real Estate Platform",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# =============================================================================
# CUSTOM CSS - NVIDIA-LEVEL POLISH
# =============================================================================

st.markdown("""
<style>
    /* Import Inter font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Global reset */
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
    }

    /* Smooth scrolling */
    html {
        scroll-behavior: smooth;
    }

    /* Hero section - premium gradient */
    .hero-section {
        background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 50%, #0f0f23 100%);
        padding: 4.5rem 3.5rem;
        border-radius: 1.5rem;
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
        box-shadow: 0 20px 60px rgba(0, 120, 212, 0.15);
    }
    .hero-section::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0; bottom: 0;
        background:
            radial-gradient(circle at 20% 50%, rgba(0, 120, 212, 0.15) 0%, transparent 50%),
            radial-gradient(circle at 80% 80%, rgba(0, 188, 242, 0.1) 0%, transparent 50%);
        pointer-events: none;
    }
    .hero-title {
        font-size: 3.5rem;
        font-weight: 900;
        margin-bottom: 0.75rem;
        position: relative;
        z-index: 1;
        background: linear-gradient(135deg, #ffffff 0%, #a5f3fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        letter-spacing: -0.02em;
        text-shadow: 0 2px 10px rgba(0,0,0,0.3);
    }
    .hero-title span:first-child {
        -webkit-text-fill-color: initial;
        text-shadow: none;
    }
    .hero-subtitle {
        font-size: 1.25rem;
        color: rgba(255, 255, 255, 0.85);
        font-weight: 300;
        position: relative;
        z-index: 1;
        letter-spacing: 0.02em;
    }
    .hero-tag {
        display: inline-flex;
        gap: 0.5rem;
        align-items: center;
        background: rgba(0, 120, 212, 0.2);
        padding: 0.5rem 1.25rem;
        border-radius: 2rem;
        font-size: 0.875rem;
        margin-top: 1.25rem;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(0, 120, 212, 0.3);
        color: rgba(255, 255, 255, 0.9);
    }

    /* Navigation tabs - modern pill style with high contrast */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        padding: 0.75rem;
        background: #1e293b !important;
        border-radius: 12px;
        border: 1px solid #334155;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 12px 20px;
        border-radius: 8px;
        font-weight: 700;
        font-size: 0.9rem;
        transition: all 0.2s ease;
        border: 2px solid transparent;
        color: #94a3b8 !important;
        background: #334155 !important;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background: #475569 !important;
        border-color: #0078D4;
        color: #ffffff !important;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: linear-gradient(135deg, #0078D4 0%, #00BCF2 100%) !important;
        color: #ffffff !important;
        box-shadow: 0 4px 16px rgba(0, 120, 212, 0.5);
        border-color: #0078D4;
    }

    /* Premium cards */
    .result-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        border-radius: 1rem;
        padding: 1.5rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.06);
        border: 1px solid rgba(226, 232, 240, 0.8);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .result-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 40px rgba(0,0,0,0.12);
        border-color: rgba(0, 120, 212, 0.2);
    }

    /* Metric displays - bold and clear */
    .metric-value {
        font-size: 2.75rem;
        font-weight: 800;
        background: linear-gradient(135deg, #0078D4 0%, #00BCF2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        letter-spacing: -0.02em;
    }
    .metric-label {
        font-size: 0.75rem;
        color: #64748b;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.5rem;
    }

    /* Badges - gradient backgrounds with shadows */
    .badge-buy {
        background: linear-gradient(135deg, #22c55e 0%, #16a34a 100%);
        color: white;
        padding: 0.5rem 1.25rem;
        border-radius: 2rem;
        font-weight: 700;
        font-size: 0.875rem;
        display: inline-block;
        box-shadow: 0 4px 12px rgba(34, 197, 94, 0.3);
    }
    .badge-rent {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        color: white;
        padding: 0.5rem 1.25rem;
        border-radius: 2rem;
        font-weight: 700;
        font-size: 0.875rem;
        display: inline-block;
        box-shadow: 0 4px 12px rgba(239, 68, 68, 0.3);
    }
    .badge-hold {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        color: white;
        padding: 0.5rem 1.25rem;
        border-radius: 2rem;
        font-weight: 700;
        font-size: 0.875rem;
        display: inline-block;
        box-shadow: 0 4px 12px rgba(245, 158, 11, 0.3);
    }
    .badge-hot {
        background: linear-gradient(135deg, #dc2626 0%, #ef4444 50%, #f97316 100%);
        color: white;
        padding: 0.35rem 0.85rem;
        border-radius: 1rem;
        font-size: 0.7rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        box-shadow: 0 2px 8px rgba(220, 38, 38, 0.4);
    }
    .badge-warm {
        background: linear-gradient(135deg, #f97316 0%, #f59e0b 100%);
        color: white;
        padding: 0.35rem 0.85rem;
        border-radius: 1rem;
        font-size: 0.7rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        box-shadow: 0 2px 8px rgba(249, 115, 22, 0.4);
    }
    .badge-cooling {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: white;
        padding: 0.35rem 0.85rem;
        border-radius: 1rem;
        font-size: 0.7rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        box-shadow: 0 2px 8px rgba(59, 130, 246, 0.4);
    }
    .badge-cold {
        background: linear-gradient(135deg, #6b7280 0%, #4b5563 100%);
        color: white;
        padding: 0.35rem 0.85rem;
        border-radius: 1rem;
        font-size: 0.7rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        box-shadow: 0 2px 8px rgba(107, 114, 128, 0.4);
    }
    .badge-cold {
        background: linear-gradient(135deg, #6b7280, #4b5563);
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 1rem;
        font-size: 0.75rem;
        font-weight: 700;
    }

    /* Section headers */
    .section-header {
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: 1rem;
        color: #1a1a1a;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* Feature cards */
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin: 1.5rem 0;
    }
    .feature-card {
        background: linear-gradient(135deg, #f8fafc, #f1f5f9);
        padding: 1.25rem;
        border-radius: 0.75rem;
        border: 1px solid #e2e8f0;
    }
    .feature-card h4 {
        margin-bottom: 0.5rem;
        color: #0078D4;
    }

    /* Chatbot messages */
    .chat-message {
        padding: 1rem 1.25rem;
        border-radius: 1rem;
        margin-bottom: 1rem;
        max-width: 85%;
    }
    .chat-user {
        background: linear-gradient(135deg, #0078D4, #00BCF2);
        color: white;
        margin-left: auto;
    }
    .chat-bot {
        background: #f1f5f9;
        color: #1a1a1a;
        border: 1px solid #e2e8f0;
    }

    /* Progress bar */
    .stProgress > div > div {
        background: linear-gradient(90deg, #0078D4, #00BCF2);
    }

    /* Dataframe styling */
    .dataframe {
        border-radius: 0.75rem;
        overflow: hidden;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# MODEL LOADING
# =============================================================================

@st.cache_resource
def load_components():
    model_dir = str(Path(__file__).parent / "models")

    predictor = PricePredictor(model_dir=model_dir)
    model_loaded = False
    try:
        predictor.load_model()
        model_loaded = True
    except FileNotFoundError:
        pass

    heatmap_gen = MarketHeatmapGenerator(predictor=predictor)
    chatbot = PropertyChatbot(predictor=predictor)
    roi_calc = ROICalculator()
    listing_scraper = LiveListingsScraper(cache_hours=24)

    return {
        'predictor': predictor,
        'model_loaded': model_loaded,
        'heatmap': heatmap_gen,
        'chatbot': chatbot,
        'roi': roi_calc,
        'listings': listing_scraper
    }


# =============================================================================
# INVESTOR PERSONAS
# =============================================================================

INVESTOR_PERSONAS = {
    "First-Time Buyer": {
        "description": "Looking for an affordable entry point with good appreciation potential",
        "priority": "appreciation",
        "risk_tolerance": "medium",
        "time_horizon": 10,
        "budget_focus": "condo,townhouse"
    },
    "Long-Term Investor": {
        "description": "Seeking steady cash flow and long-term wealth building",
        "priority": "cash_flow",
        "risk_tolerance": "low",
        "time_horizon": 15,
        "budget_focus": "multi_family,detached"
    },
    "Growth Hunter": {
        "description": "Maximizing appreciation potential, willing to accept higher risk",
        "priority": "growth",
        "risk_tolerance": "high",
        "time_horizon": 5,
        "budget_focus": "condo,townhouse"
    },
    "Flipper": {
        "description": "Short-term holds targeting undervalued properties",
        "priority": "undervalue",
        "risk_tolerance": "high",
        "time_horizon": 2,
        "budget_focus": "condo,townhouse,detached"
    },
    "Balanced": {
        "description": "Equal focus on cash flow and appreciation",
        "priority": "balanced",
        "risk_tolerance": "medium",
        "time_horizon": 7,
        "budget_focus": "townhouse,detached"
    }
}


def get_personalized_recommendations(persona: str, city_data: pd.DataFrame) -> pd.DataFrame:
    """Get recommendations tailored to investor persona."""

    persona_config = INVESTOR_PERSONAS.get(persona, INVESTOR_PERSONAS["Balanced"])
    priority = persona_config["priority"]

    if priority == "appreciation":
        return city_data.nlargest(5, "appreciation_12m")
    elif priority == "cash_flow":
        return city_data.nlargest(5, "rental_yield")
    elif priority == "growth":
        score = city_data["appreciation_12m"] * 0.7 + city_data["investment_score"] / 100 * 0.3
        return city_data.assign(priority_score(score)).nlargest(5, "priority_score")
    elif priority == "undervalue":
        undervalued = city_data[
            (city_data["current_price"] < city_data["current_price"].mean() * 0.85) &
            (city_data["rental_yield"] > city_data["rental_yield"].median())
        ]
        return undervalued.nlargest(5, "investment_score") if len(undervalued) > 0 else city_data.nlargest(5, "investment_score")
    else:  # balanced
        return city_data.nlargest(5, "investment_score")


# =============================================================================
# SIDEBAR TOGGLE
# =============================================================================

# Load components FIRST - needed for entire app
components = load_components()

# Initialize sidebar state
if "sidebar_open" not in st.session_state:
    st.session_state.sidebar_open = False

# Sidebar toggle button in top-right corner
col_space, col_toggle = st.columns([12, 1])
with col_toggle:
    toggle_label = "✕" if st.session_state.sidebar_open else "☰"
    if st.button(toggle_label, key="sidebar_toggle", use_container_width=True, help="Toggle sidebar"):
        st.session_state.sidebar_open = not st.session_state.sidebar_open
        st.rerun()

# Force rerun on every interaction to ensure sidebar state updates
if st.session_state.get('sidebar_needs_rerun', False):
    st.session_state.sidebar_needs_rerun = False
    st.rerun()

# Show sidebar if open
if st.session_state.sidebar_open:
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/house.png", width=80)
        st.title("Propra")
        st.markdown("**AI-Powered Canadian Real Estate Platform**")
        st.markdown("---")

        if components['model_loaded']:
            st.sidebar.success("✅ ML Model Active")
        else:
            st.sidebar.info("ℹ️ Using smart fallback predictions")

        st.sidebar.markdown("---")
        st.sidebar.markdown("""
        ### Features
        - 📊 Market Heatmap
        - 🤖 AI Chatbot
        - 💰 ROI Calculator
        - 📈 Scenario Simulator
        - 🎯 Personalized Recs
        - 🔍 Hidden Gems
        """)

        st.sidebar.markdown("---")
        st.sidebar.markdown("<small>Not financial advice. For informational purposes only.</small>", unsafe_allow_html=True)


# =============================================================================
# HERO SECTION
# =============================================================================

st.markdown("""
<div class="hero-section">
    <div class="hero-title"><span style="-webkit-text-fill-color: initial;">🏠</span> <span style="background: linear-gradient(135deg, #ffffff 0%, #a5f3fc 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">Propra</span></div>
    <div class="hero-subtitle">Canada's AI-Powered Real Estate Intelligence Platform</div>
    <div class="hero-tag">✨ Predictive Analytics &bull; Investment Insights &bull; Market Intelligence</div>
</div>
""", unsafe_allow_html=True)


# =============================================================================
# MAIN NAVIGATION - CONSOLIDATED
# =============================================================================

tabs = st.tabs([
    "🔮 Predict & Simulate",
    "🗺️ Explore Markets",
    "🤖 AI Advisor",
    "💰 Calculate ROI",
    "🏠 Live Listings",
    "🎯 My Recommendations"
])


# =============================================================================
# TAB 1: PREDICT & SIMULATE (Consolidated)
# =============================================================================

with tabs[0]:
    st.markdown("### 🔮 Predict & Simulate")
    st.markdown("AI price predictions with scenario modeling and Monte Carlo simulation.")

    # Sub-tabs for Predict vs Simulate
    pred_sim_tabs = st.tabs(["📈 Price Prediction", "🎲 Scenario Simulator"])

    # ===== PRICE PREDICTION SUB-TAB =====
    with pred_sim_tabs[0]:
        st.markdown("Get ML-powered price forecasts with confidence intervals.")

        col1, col2 = st.columns([2, 1])

        with col1:
            pred_col1, pred_col2 = st.columns(2)

            with pred_col1:
                pred_city = st.selectbox(
                    "City",
                    ["Vancouver", "Toronto", "Calgary", "Burnaby", "Richmond", "North Vancouver"],
                    index=0,
                    key="pred_city"
                )

                pred_type = st.selectbox(
                    "Property Type",
                    ["condo", "townhouse", "detached", "multi_family"],
                    format_func=lambda x: x.replace("_", " ").title(),
                    key="pred_type"
                )

            with pred_col2:
                pred_price = st.number_input(
                    "Current Price ($)",
                    min_value=200000,
                    max_value=10000000,
                    value=750000,
                    step=25000,
                    key="pred_price"
                )

                pred_rent = st.number_input(
                    "Monthly Rent ($)",
                    min_value=500,
                    max_value=20000,
                    value=2600,
                    step=100,
                    key="pred_rent"
                )

        with col2:
            st.markdown("#### Scenario Adjustment")
            rate_adj = st.slider(
                "Rate Change (%)",
                min_value=-2.0,
                max_value=3.0,
                value=0.0,
                step=0.25,
                key="rate_adj"
            )

            horizon = st.slider(
                "Prediction Horizon (months)",
                min_value=6,
                max_value=24,
                value=12,
                step=6,
                key="horizon"
            )

        if st.button("🔮 Get Prediction", type="primary", use_container_width=True, key="get_pred"):
            predictor = components['predictor']

            prediction = predictor.predict_price_change(
                current_price=pred_price,
                city=pred_city,
                property_type=pred_type,
                horizon_months=horizon
            )

            # Rate adjustment
            if rate_adj != 0:
                rate_impact = -0.5 * rate_adj
                adj_appreciation = prediction["predicted_change_pct"] + rate_impact
                adj_price = pred_price * (1 + adj_appreciation / 100)
            else:
                adj_price = prediction["predicted_price_6m"]
                adj_appreciation = prediction["predicted_change_pct"]

            # Buy vs Rent
            bvr = quick_analysis(
                purchase_price=pred_price,
                monthly_rent=pred_rent,
                city=pred_city,
                property_type=pred_type,
                time_horizon_years=5
            )

            rec = bvr["recommendation"]["recommendation"]
            badge = "badge-buy" if "BUY" in rec else "badge-rent" if "RENT" in rec else "badge-hold"
            bvr_rec = "Buy" if "BUY" in rec else "Rent" if "RENT" in rec else "Hold"

            # Risk level
            risk = "Low" if pred_city in ["Vancouver", "North Vancouver", "Burnaby"] else "Medium"
            if pred_type == "condo":
                risk = "Medium" if risk == "Low" else "High"

            # Investment score
            score = min(100, max(0, 50 + adj_appreciation * 5 + (20 if "BUY" in rec else -10)))

            st.markdown("---")

            # Results
            r1, r2, r3, r4 = st.columns(4)

            with r1:
                st.markdown('<div class="metric-label">Predicted Price</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-value">${adj_price:,.0f}</div>', unsafe_allow_html=True)
                st.metric("Expected Change", f"{adj_appreciation:+.1f}%")

            with r2:
                st.markdown('<div class="metric-label">Recommendation</div>', unsafe_allow_html=True)
                st.markdown(f'<span class="{badge}" style="font-size:1.25rem">{bvr_rec}</span>', unsafe_allow_html=True)
                st.markdown(f"Over 5-year horizon")

            with r3:
                st.markdown('<div class="metric-label">Risk Level</div>', unsafe_allow_html=True)
                risk_emoji = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}.get(risk, "⚪")
                st.markdown(f'<span style="font-size:1.5rem">{risk_emoji} {risk}</span>', unsafe_allow_html=True)

            with r4:
                st.markdown('<div class="metric-label">Investment Score</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-value">{score:.0f}/100</div>', unsafe_allow_html=True)
                if score >= 70:
                    st.success("Strong opportunity")
                elif score >= 50:
                    st.info("Moderate potential")
                else:
                    st.warning("Consider alternatives")

            # Price chart
            st.markdown("#### Price Forecast")

            months = [0, 6, 12, 18, 24]
            prices = [pred_price]
            for m in months[1:]:
                p = predictor.predict_price_change(
                    current_price=pred_price,
                    city=pred_city,
                    property_type=pred_type,
                    horizon_months=m
                )
                prices.append(p["predicted_price_6m"] if m == 6 else p["predicted_price_6m"] * (1 + p["predicted_change_pct"] / 100 * ((m - 6) / 6)))

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=["Now"] + [f"{m} mo" for m in months[1:]],
                y=prices,
                mode="lines+markers",
                name="Predicted Price",
                line=dict(color="#0078D4", width=3),
                marker=dict(size=10)
            ))

            fig.add_trace(go.Scatter(
                x=["Now"] + [f"{m} mo" for m in months[1:]] + [f"{m} mo" for m in reversed(months[1:])] + ["Now"],
                y=prices + [p * 1.1 for p in reversed(prices)] + [pred_price * 1.1],
                fill="toself",
                fillcolor="rgba(0,120,212,0.15)",
                line=dict(color="rgba(0,0,0,0)"),
                name="Confidence Range"
            ))

            fig.update_layout(
                height=350,
                xaxis_title="Time",
                yaxis_title="Price ($)",
                showlegend=True,
                yaxis_tickformat="$,.0f"
            )

            st.plotly_chart(fig, use_container_width=True)

    # ===== SCENARIO SIMULATOR SUB-TAB =====
    with pred_sim_tabs[1]:
        st.markdown("Multi-year wealth projection with Monte Carlo simulation.")

        sim_col1, sim_col2 = st.columns(2)

        with sim_col1:
            sim_price = st.number_input(
                "Purchase Price ($)",
                min_value=200000,
                max_value=10000000,
                value=750000,
                step=25000,
                key="sim_price"
            )

            sim_down = st.slider(
                "Down Payment (%)",
                min_value=5,
                max_value=50,
                value=20,
                key="sim_down"
            )

            sim_rent = st.number_input(
                "Monthly Rent ($)",
                min_value=0,
                max_value=50000,
                value=2600,
                step=100,
                help="Set to 0 for owner-occupied",
                key="sim_rent"
            )

        with sim_col2:
            sim_city = st.selectbox(
                "City",
                ["Vancouver", "Toronto", "Calgary", "Burnaby", "Richmond", "North Vancouver"],
                index=0,
                key="sim_city"
            )

            sim_horizon = st.slider(
                "Time Horizon (years)",
                min_value=5,
                max_value=30,
                value=10,
                step=5,
                key="sim_horizon"
            )

            sim_rate = st.number_input(
                "Mortgage Rate (%)",
                min_value=1.0,
                max_value=15.0,
                value=5.0,
                step=0.25,
                key="sim_rate"
            )

        if st.button("📈 Run Simulation", type="primary", use_container_width=True, key="run_sim"):
            results = quick_scenario_analysis(
                purchase_price=sim_price,
                down_payment_pct=sim_down / 100,
                monthly_rent=sim_rent,
                city=sim_city,
                time_horizon_years=sim_horizon
            )

            base = results["base_case"]
            mc = results["monte_carlo"]
            risk = results["risk_metrics"]

            st.markdown("---")

            # Key outcomes
            r1, r2, r3, r4 = st.columns(4)

            with r1:
                st.metric(
                    "Final Property Value",
                    f"${base['final_property_value']:,.0f}",
                    f"{((base['final_property_value']/sim_price)**(1/sim_horizon)-1)*100:.1f}% CAGR"
                )

            with r2:
                st.metric(
                    "Net Sale Proceeds",
                    f"${base['net_sale_proceeds']:,.0f}"
                )

            with r3:
                st.metric(
                    "Total Equity Built",
                    f"${base['total_equity']:,.0f}"
                )

            with r4:
                st.metric(
                    "Cash-on-Cash Return",
                    f"{risk['cash_on_cash_return']:.1f}%"
                )

            # Monte Carlo visualization
            st.markdown("#### Monte Carlo Simulation (1000 trials)")

            fig_mc = go.Figure()
            fig_mc.add_trace(go.Histogram(
                x=mc["net_proceeds_samples"],
                nbinsx=50,
                name="Net Proceeds",
                marker_color="#0078D4",
                opacity=0.7
            ))

            fig_mc.update_layout(
                height=400,
                xaxis_title="Net Sale Proceeds ($)",
                yaxis_title="Frequency",
                showlegend=False
            )

            st.plotly_chart(fig_mc, use_container_width=True)

            # Scenario comparison
            st.markdown("#### Scenario Comparison")
            st.dataframe(
                results["scenario_comparison"].round(2),
                use_container_width=True
            )


# =============================================================================
# TAB 2: EXPLORE MARKETS (Consolidated)
# =============================================================================

with tabs[1]:
    st.markdown("### 🗺️ Explore Markets")
    st.markdown("Discover opportunities with interactive maps and AI-powered undervalued property detection.")

    # Sub-tabs for Heatmap vs Hidden Gems
    explore_tabs = st.tabs(["📊 Market Heatmap", "💎 Hidden Gems"])

    heatmap_data = generate_sample_heatmap_data()
    heatmap_gen = components['heatmap']

    # ===== MARKET HEATMAP SUB-TAB =====
    with explore_tabs[0]:
        # Compact inline selector with description
        col_sel1, col_sel2 = st.columns([1, 3])
        with col_sel1:
            viz_type = st.selectbox(
                "Metric",
                ["Investment Score", "Appreciation (12m)", "Rental Yield", "Cap Rate", "Buy vs Rent Score"]
            )
        with col_sel2:
            metric_info = {
                "Investment Score": "0-100 score combining growth, yield & risk",
                "Appreciation (12m)": "Predicted price growth over 12 months",
                "Rental Yield": "Annual rent as % of property price",
                "Cap Rate": "Net operating income / property value",
                "Buy vs Rent Score": "Higher = better to buy than rent"
            }
            st.markdown(f"<div style='padding: 0.75rem 1rem; color: #64748b; font-size: 0.9rem;'>{metric_info[viz_type]}</div>", unsafe_allow_html=True)

        # Create two columns - map on left, sidebar on right
        col_map, col_sidebar = st.columns([3, 1])

    # Enhanced map with better styling
    if viz_type == "Investment Score":
        color_col = "investment_score"
        colorscale = [[0, "#1e3a5f"], [0.5, "#0891b2"], [1, "#22d3ee"]]
    elif viz_type == "Appreciation (12m)":
        color_col = "appreciation_12m"
        colorscale = [[0, "#dc2626"], [0.5, "#fbbf24"], [1, "#16a34a"]]
    elif viz_type == "Rental Yield":
        color_col = "rental_yield"
        colorscale = [[0, "#7c2d12"], [0.5, "#f97316"], [1, "#fde68a"]]
    elif viz_type == "Cap Rate":
        color_col = "cap_rate"
        colorscale = [[0, "#4c1d95"], [0.5, "#8b5cf6"], [1, "#c4b5fd"]]
    else:
        color_col = "buy_vs_rent_score"
        colorscale = [[0, "#164e63"], [0.5, "#06b6d4"], [1, "#67e8f9"]]

    # Create map with custom styling
    fig_map = go.Figure()

    fig_map.add_trace(go.Scattergeo(
        lat=heatmap_data["latitude"],
        lon=heatmap_data["longitude"],
        marker=dict(
            size=heatmap_data["marker_size"] * 1.5,
            color=heatmap_data[color_col],
            colorscale=colorscale,
            colorbar=dict(
                title=dict(text=viz_type, font=dict(size=12, color="#1a1a1a")),
                tickfont=dict(size=10, color="#1a1a1a"),
                bgcolor="rgba(255,255,255,0.9)",
                bordercolor="#e2e8f0",
                borderwidth=1,
                len=0.4,
                thickness=20
            ),
            line=dict(color="#1a1a1a", width=1.5),
            opacity=0.9
        ),
        text=heatmap_data.apply(
            lambda x: f"<b>{x['city']}</b><br>{x['property_type'].replace('_', ' ').title()}<br>" +
                      f"Price: ${x['current_price']:,.0f}<br>" +
                      f"{viz_type}: {x[color_col]:.1f}" +
                      f"<br>Appreciation: {x['appreciation_12m']:+.1f}%<br>" +
                      f"Yield: {x['rental_yield']:.1f}%",
            axis=1
        ),
        hoverinfo="text",
        hoverlabel=dict(
            bgcolor="white",
            font_size=12,
            font_family="Inter, sans-serif",
            font_color="#1a1a1a",
            bordercolor="#0078D4",
            align="left"
        )
    ))

    fig_map.update_layout(
        title=dict(
            text=f"🗺️ {viz_type} Across Canadian Markets",
            font=dict(size=16, weight=600, color="#1a1a1a"),
            y=0.95,
            x=0.5,
            xanchor="center",
            yanchor="top"
        ),
        geo=dict(
            scope="north america",
            center=dict(lat=56.1304, lon=-106.3468),
            projection_type="mercator",
            bgcolor="rgba(248, 250, 252, 0.95)",
            lakecolor="rgba(59, 130, 246, 0.5)",
            landcolor="rgba(241, 245, 249, 0.95)",
            showland=True,
            showlakes=True,
            countrycolor="#475569",
            countrywidth=2,
            showcoastlines=True,
            coastlinecolor="#475569",
            coastlinewidth=2,
            showsubunits=True,
            subunitcolor="#94a3b8",
            subunitwidth=1,
            lataxis_range=[41, 84],
            lonaxis_range=[-141, -52]
        ),
        height=600,
        margin=dict(l=10, r=10, t=60, b=10),
        paper_bgcolor="rgba(255,255,255,0.95)",
        plot_bgcolor="rgba(255,255,255,0.95)"
    )

    with col_map:
        st.plotly_chart(fig_map, use_container_width=True)

    with col_sidebar:
        # Market summary
        st.markdown("#### Market Summary")

        summary = heatmap_gen.generate_city_summary(heatmap_data)
        st.dataframe(
            summary.round(2),
            use_container_width=True,
            hide_index=True
        )

        # Market Regime Legend - use container to prevent overflow
        with st.container():
            st.markdown("#### Market Regimes")
            st.markdown("""
            <div style="background: #f8fafc; padding: 1rem; border-radius: 8px; font-size: 0.85rem;">
                <div style="margin-bottom: 0.5rem;"><span style="color: #dc2626; font-weight: bold;">🔴 HOT</span> - &gt;5% appreciation, high demand</div>
                <div style="margin-bottom: 0.5rem;"><span style="color: #ea580c; font-weight: bold;">🟠 WARM</span> - 2-5% appreciation, stable</div>
                <div style="margin-bottom: 0.5rem;"><span style="color: #2563eb; font-weight: bold;">🔵 COOLING</span> - 0-2% appreciation, slowing</div>
                <div><span style="color: #6b7280; font-weight: bold;">⚫ COLD</span> - Negative appreciation, low demand</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("#### Top Markets")
        top = heatmap_gen.get_top_markets(heatmap_data)
        for _, row in top.head(5).iterrows():
            regime_badge = {
                "hot": "badge-hot",
                "warm": "badge-warm",
                "cooling": "badge-cooling",
                "cold": "badge-cold"
            }.get(row["market_regime"], "badge-hold")

            # Get appreciation for this property
            prop_data = heatmap_data[
                (heatmap_data["city"] == row["city"]) &
                (heatmap_data["property_type"] == row["property_type"])
            ]
            appreciation = prop_data["appreciation_12m"].values[0] if len(prop_data) > 0 else 0
            rental_yield = prop_data["rental_yield"].values[0] if len(prop_data) > 0 else 0

            # City color based on regime for visual clarity
            city_colors = {
                "hot": "#dc2626",
                "warm": "#ea580c",
                "cooling": "#2563eb",
                "cold": "#374151"
            }
            city_color = city_colors.get(row["market_regime"], "#1a1a1a")

            st.markdown(f"""
            <div class="result-card" style="padding: 1rem; margin-bottom: 0.75rem; border-left: 5px solid {city_color}; background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%); box-shadow: 0 2px 8px rgba(0,0,0,0.08);">
                <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.75rem;">
                    <div style="display:flex;align-items:center;gap:0.5rem;">
                        <span style="font-size:1.25rem; font-weight: 700; color: #1a1a1a; text-shadow: 0 1px 2px rgba(255,255,255,1);">{row['city']}</span>
                        <span style="font-size:0.9rem; color: #6b7280; font-weight: 500; background: #f1f5f9; padding: 0.25rem 0.5rem; border-radius: 4px;">{row['property_type'].replace('_', ' ').title()}</span>
                    </div>
                    <span class="{regime_badge}" style="box-shadow: 0 2px 4px rgba(0,0,0,0.1);">{row['market_regime'].upper()}</span>
                </div>
                <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:0.75rem; background: #f8fafc; padding: 0.75rem; border-radius: 8px;">
                    <div style="text-align:center;">
                        <div style="font-size:0.75rem; color: #64748b; text-transform:uppercase; letter-spacing:0.5px; margin-bottom:0.25rem;">Price</div>
                        <div style="font-size:1.1rem; font-weight: 700; color: #0f172a;">${row['current_price']/1000:.0f}K</div>
                    </div>
                    <div style="text-align:center; border-left:1px solid #e2e8f0; border-right:1px solid #e2e8f0;">
                        <div style="font-size:0.75rem; color: #64748b; text-transform:uppercase; letter-spacing:0.5px; margin-bottom:0.25rem;">Appreciation</div>
                        <div style="font-size:1.1rem; font-weight: 700; color: {'#16a34a' if appreciation>3 else '#d97706' if appreciation>0 else '#dc2626'}">{appreciation:+.1f}%</div>
                    </div>
                    <div style="text-align:center;">
                        <div style="font-size:0.75rem; color: #64748b; text-transform:uppercase; letter-spacing:0.5px; margin-bottom:0.25rem;">Yield</div>
                        <div style="font-size:1.1rem; font-weight: 700; color: #0f172a;">{rental_yield:.1f}%</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ===== HIDDEN GEMS SUB-TAB =====
    with explore_tabs[1]:
        st.markdown("### 💎 Hidden Gems Engine")
        st.markdown("AI-powered detection of undervalued properties with high potential returns.")

        # Get undervalued properties
        undervalued = heatmap_gen.get_undervalued_properties(heatmap_data)

        if len(undervalued) > 0:
            st.success(f"Found {len(undervalued)} potentially undervalued properties!")

            for i, (_, row) in enumerate(undervalued.iterrows()):
                # Calculate realistic discount vs same property type in same city
                city_type_avg = heatmap_data[
                    (heatmap_data['city'] == row['city']) &
                    (heatmap_data['property_type'] == row['property_type'])
                ]['current_price'].median()

                # More realistic discount calculation (typically 5-15% undervalued, not 67%)
                discount_pct = max(0, min(20, (1 - row['current_price'] / city_type_avg) * 100)) if city_type_avg > 0 else 0

                # Fix rent calculation - rental_yield is annual %, so monthly rent = price * (yield/100) / 12
                # Typical yields are 3-6% annually
                estimated_monthly_rent = row['current_price'] * (row['rental_yield'] / 100) / 12

                st.markdown("---")

                # Property card using Streamlit columns
                gem_col1, gem_col2 = st.columns([2, 1])

                with gem_col1:
                    # Header
                    col_header1, col_header2 = st.columns([3, 1])
                    with col_header1:
                        st.markdown(f"### 🔍 {row['city']} - {row['property_type'].replace('_', ' ').title()}")
                        st.caption(f"Undervalue Score: **{row['undervalue_score']:.1f}**")
                    with col_header2:
                        st.markdown("")
                        st.markdown("")
                        if discount_pct >= 10:
                            st.success("**GREAT VALUE**")
                        elif discount_pct >= 5:
                            st.info("**GOOD VALUE**")
                        else:
                            st.caption("**FAIR PRICE**")

                    # Stats grid
                    stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
                    with stat_col1:
                        st.markdown("**Price**")
                        st.markdown(f"${row['current_price']/1000:.0f}K")
                    with stat_col2:
                        st.markdown("**Vs Market**")
                        if discount_pct > 0:
                            st.markdown(f":green[**{discount_pct:.0f}% below**]")
                        else:
                            st.markdown(f"**At market**")
                    with stat_col3:
                        st.markdown("**Yield**")
                        st.markdown(f"**{row['rental_yield']:.1f}%**")
                    with stat_col4:
                        st.markdown("**Score**")
                        st.markdown(f"**{row['investment_score']:.0f}/100**")

                with gem_col2:
                    st.markdown("#### Why It's Interesting")
                    st.markdown(f"- 📊 **Price:** ${row['current_price']:,.0f} vs ${city_type_avg:,.0f} avg for {row['property_type'].replace('_', ' ')}s in {row['city']}")
                    st.markdown(f"- 💰 **Est. Rent:** ${estimated_monthly_rent:,.0f}/month ({row['rental_yield']:.1f}% gross yield)")
                    st.markdown(f"- 📈 **12mo Forecast:** {row['appreciation_12m']:+.1f}%")
                    st.markdown(f"- 🌡️ **Market:** {row['market_regime'].title()}")

                st.markdown("")
        else:
            st.info("No undervalued properties detected in current market conditions. Market appears fairly valued overall.")


# =============================================================================
# TAB 3: AI ADVISOR
# =============================================================================

with tabs[2]:
    st.markdown("### AI Property Advisor")
    st.markdown("Ask questions about Canadian real estate and get instant, data-backed answers.")

    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Initialize inline analysis states
    if "inline_roi_property" not in st.session_state:
        st.session_state.inline_roi_property = None
    if "inline_ai_query" not in st.session_state:
        st.session_state.inline_ai_query = None
    if "inline_compare_city" not in st.session_state:
        st.session_state.inline_compare_city = None

    chatbot = components['chatbot']

    # Display chat history
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(f'<div class="chat-message chat-user">{msg["content"]}</div>', unsafe_allow_html=True)
        else:
            # Render markdown content properly
            st.markdown(f'<div class="chat-message chat-bot">', unsafe_allow_html=True)
            st.markdown(msg["content"])  # Render markdown
            st.markdown(f"""
            <div style="margin-top:0.75rem;font-size:0.85rem;color:#64748b;border-top:1px solid #e2e8f0;padding-top:0.5rem;">
                <strong>Confidence:</strong> {msg["confidence"]}
                {f' | <strong>Try:</strong> {", ".join(msg["follow_ups"][:3])}' if msg.get("follow_ups") else ''}
            </div>
            </div>
            """, unsafe_allow_html=True)

    # Chat input
    query = st.text_input(
        "Ask a question...",
        placeholder="e.g., 'Where should I buy for 6% ROI?' or 'Vancouver vs Calgary for investment'",
        key="chat_input"
    )

    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("💬 Send", type="primary", use_container_width=True):
            if query:
                # Add user message
                st.session_state.chat_history.append({"role": "user", "content": query})

                # Get response
                response = chatbot.respond(query)

                # Add bot response
                st.session_state.chat_history.append({
                    "role": "bot",
                    "content": response.answer,
                    "confidence": response.confidence,
                    "follow_ups": response.follow_up_questions
                })

                st.rerun()

    # Quick questions
    st.markdown("#### Quick Questions")
    quick_qs = [
        "Where should I buy for 6% ROI?",
        "Vancouver vs Calgary for investment",
        "Should I buy a condo or townhouse?",
        "How much house can I afford with $100k income?",
        "What's a good rental yield?",
        "Is now a good time to buy?"
    ]

    cols = st.columns(3)
    for i, q in enumerate(quick_qs):
        with cols[i % 3]:
            if st.button(q, use_container_width=True, key=f"qq_{i}"):
                st.session_state.chat_history.append({"role": "user", "content": q})
                response = chatbot.respond(q)
                st.session_state.chat_history.append({
                    "role": "bot",
                    "content": response.answer,
                    "confidence": response.confidence,
                    "follow_ups": response.follow_up_questions
                })
                st.rerun()

    # Clear history
    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()


# =============================================================================
# TAB 4: ROI CALCULATOR
# =============================================================================

with tabs[3]:
    st.markdown("### ROI Calculator")
    st.markdown("Calculate rental yield, cap rate, cash-on-cash return, and total ROI for investment properties.")

    col1, col2 = st.columns(2)

    with col1:
        roi_price = st.number_input(
            "Purchase Price ($)",
            min_value=100000,
            max_value=10000000,
            value=750000,
            step=25000,
            key="roi_price"
        )

        roi_rent = st.number_input(
            "Monthly Rent ($)",
            min_value=500,
            max_value=50000,
            value=2600,
            step=100,
            key="roi_rent"
        )

        roi_down = st.slider(
            "Down Payment (%)",
            min_value=5,
            max_value=50,
            value=20,
            key="roi_down"
        )

    with col2:
        roi_city = st.selectbox(
            "City",
            ["Vancouver", "Toronto", "Calgary", "Burnaby", "Richmond", "North Vancouver"],
            index=0,
            key="roi_city"
        )

        roi_type = st.selectbox(
            "Property Type",
            ["condo", "townhouse", "detached", "multi_family"],
            format_func=lambda x: x.replace("_", " ").title(),
            key="roi_type"
        )

        roi_strata = st.number_input(
            "Monthly Strata Fee ($)",
            min_value=0,
            max_value=2000,
            value={"condo": 500, "townhouse": 350, "detached": 0, "multi_family": 200}.get(roi_type, 0),
            step=50,
            key="roi_strata"
        )

    if st.button("📊 Calculate ROI", type="primary", use_container_width=True, key="calc_roi"):
        inputs = PropertyInputs(
            purchase_price=roi_price,
            monthly_rent=roi_rent,
            down_payment_pct=roi_down / 100,
            strata_fee_monthly=roi_strata
        )

        roi_calc = components['roi']
        metrics = roi_calc.calculate_all_metrics(inputs)
        recommendation = roi_calc.get_recommendation(metrics)

        st.markdown("---")

        # Grade and recommendation
        r1, r2, r3 = st.columns(3)

        with r1:
            st.markdown("#### Investment Grade")
            grade = recommendation["grade"].split(" - ")[0]
            grade_color = {"A": "#22c55e", "B": "#84cc16", "C": "#f59e0b", "D": "#ef4444", "F": "#dc2626"}.get(grade, "#6b7280")
            st.markdown(f'<div style="font-size:3rem;font-weight:800;color:{grade_color}">{grade}</div>', unsafe_allow_html=True)
            st.markdown(recommendation["grade"])

        with r2:
            st.markdown("#### Recommendation")
            rec_badge = {"BUY": "badge-buy", "HOLD": "badge-hold", "AVOID": "badge-rent"}.get(
                recommendation["recommendation"], "badge-hold"
            )
            st.markdown(f'<span class="{rec_badge}" style="font-size:1.5rem">{recommendation["recommendation"]}</span>', unsafe_allow_html=True)
            st.markdown(f"Confidence: {recommendation['confidence']}")

        with r3:
            st.markdown("#### Key Metrics")
            st.metric("Gross Yield", f"{metrics['gross_yield']:.2f}%")
            st.metric("Cap Rate", f"{metrics['cap_rate']:.2f}%")
            st.metric("Cash-on-Cash", f"{metrics['cash_on_cash_return']:.2f}%")

        # Detailed metrics
        st.markdown("#### Detailed Analysis")

        det1, det2 = st.columns(2)

        with det1:
            st.markdown("**Cash Flow**")
            st.markdown(f"""
            - Annual Rent: ${metrics['annual_rent']:,.0f}
            - Operating Expenses: ${metrics['annual_operating_expenses']:,.0f}
            - Net Operating Income: ${metrics['noi']:,.0f}
            - Annual Mortgage: ${metrics['annual_mortgage']:,.0f}
            - **Annual Cash Flow: ${metrics['annual_cash_flow']:,.0f}**
            - Monthly Cash Flow: ${metrics['monthly_cash_flow']:,.0f}
            """)

        with det2:
            st.markdown("**Returns**")
            st.markdown(f"""
            - Total Cash Invested: ${metrics['total_cash_invested']:,.0f}
            - Annual Appreciation: ${metrics['annual_appreciation']:,.0f}
            - Principal Paydown: ${metrics['first_year_principal_paydown']:,.0f}
            - **Total ROI: {metrics['total_roi']:.2f}%**
            - DSCR: {metrics['dscr']:.2f}
            """)

        # Reasoning
        st.markdown("#### Analysis")
        for reason in recommendation["reasoning"]:
            st.markdown(f"• {reason}")


# =============================================================================
# TAB 4: LIVE LISTINGS
# =============================================================================

with tabs[4]:
    st.markdown("### 🏠 Live Listings")
    st.markdown("Current properties for sale from public sources. Updated daily.")

    # Initialize live listings state
    if "live_listings" not in st.session_state:
        st.session_state.live_listings = None
    if "listings_last_fetched" not in st.session_state:
        st.session_state.listings_last_fetched = None

    # Fetch button
    col_fetch, col_info = st.columns([1, 4])
    with col_fetch:
        if st.button("🔄 Refresh Listings", use_container_width=True):
            with st.spinner("Fetching live listings from public sources..."):
                scraper = components['listings']
                st.session_state.live_listings = scraper.fetch_listings()
                st.session_state.listings_last_fetched = datetime.now()

    if st.session_state.listings_last_fetched:
        st.info(f"Last updated: {st.session_state.listings_last_fetched.strftime('%Y-%m-%d %H:%M')}")

    # Show listings
    if st.session_state.live_listings is not None and len(st.session_state.live_listings) > 0:
        listings_df = st.session_state.live_listings

        # Filters
        filter_col1, filter_col2, filter_col3, filter_col4 = st.columns(4)
        with filter_col1:
            selected_city = st.selectbox("City", sorted(listings_df['city'].unique()))
        with filter_col2:
            selected_type = st.selectbox("Property Type", sorted(listings_df['property_type'].unique()))
        with filter_col3:
            max_price = st.number_input("Max Price", min_value=0, value=2000000, step=100000)
        with filter_col4:
            min_beds = st.selectbox("Min Beds", [0, 1, 2, 3, 4], index=0)

        filtered = listings_df[
            (listings_df['city'] == selected_city) &
            (listings_df['property_type'] == selected_type) &
            (listings_df['price'] <= max_price) &
            (listings_df['bedrooms'] >= min_beds)
        ]

        if len(filtered) > 0:
            st.markdown(f"#### {len(filtered)} Properties Found")

            for idx, (_, row) in enumerate(filtered.iterrows()):
                # Calculate investment metrics
                estimated_rent = row.get('estimated_monthly_rent', int(row['price'] * 0.04 / 12))
                gross_yield = (estimated_rent * 12) / row['price'] * 100

                # Property images
                images = row.get('images', [])
                description = row.get('description', '')
                walk_score = row.get('walk_score', 0)
                transit_score = row.get('transit_score', 0)
                neighborhood = row.get('neighborhood', 'Unknown')
                days_on_market = row.get('days_on_market', 0)

                with st.container():
                    st.markdown(f"""
                    <div class="result-card" style="border-left: 5px solid #0078D4;">
                        <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:1rem;">
                            <div>
                                <div style="font-size:1.25rem; font-weight:700; color:#0f172a;">
                                    {row.get('address', f"{row['city']} Property")}
                                </div>
                                <div style="color:#64748b; font-size:0.9rem; margin-top:0.25rem;">
                                    {neighborhood}, {row['city']} • {row['property_type'].replace('_', ' ').title()}
                                </div>
                            </div>
                            <div style="text-align:right;">
                                <div style="font-size:1.75rem; font-weight:800; color:#0078D4;">${row['price']/1000:.0f}K</div>
                                <div style="font-size:0.8rem; color:#22c55e; font-weight:600;">{gross_yield:.1f}% Gross Yield</div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # Property images gallery
                    if images:
                        img_cols = st.columns(4)
                        for i, img_url in enumerate(images[:4]):
                            with img_cols[i]:
                                st.image(img_url, use_container_width=True)

                    # Property details
                    details_col1, details_col2, details_col3, details_col4 = st.columns(4)
                    with details_col1:
                        st.metric("Price", f"${row['price']:,.0f}")
                    with details_col2:
                        st.metric("Beds / Baths", f"{row.get('bedrooms', 'N/A')} / {row.get('bathrooms', 'N/A')}")
                    with details_col3:
                        sqft = row.get('sqft', 0)
                        st.metric("Size", f"{sqft:,} sqft" if sqft else "N/A")
                    with details_col4:
                        st.metric("Days on Market", f"{days_on_market}")

                    # Accessibility scores
                    access_col1, access_col2 = st.columns(2)
                    with access_col1:
                        walk_emoji = "🚶" if walk_score >= 70 else "🚶‍♂️" if walk_score >= 50 else "🚗"
                        st.markdown(f"**{walk_emoji} Walk Score:** {walk_score}/100 " +
                                    f"({'Very Walkable' if walk_score >= 70 else 'Somewhat Walkable' if walk_score >= 50 else 'Car Dependent'})")
                    with access_col2:
                        transit_emoji = "🚇" if transit_score >= 70 else "🚌" if transit_score >= 50 else "🚗"
                        st.markdown(f"**{transit_emoji} Transit Score:** {transit_score}/100 " +
                                    f"({'Excellent Transit' if transit_score >= 70 else 'Good Transit' if transit_score >= 50 else 'Minimal Transit'})")

                    # Property description
                    if description:
                        st.markdown(f"""
                        <div style="background:#f8fafc;padding:1rem;border-radius:8px;margin:1rem 0;">
                            <strong style="color:#0f172a;">📝 About This Property</strong>
                            <p style="color:#475569;margin:0.5rem 0 0 0;">{description}</p>
                        </div>
                        """, unsafe_allow_html=True)

                    # Investment metrics
                    st.markdown("---")
                    invest_col1, invest_col2, invest_col3 = st.columns(3)
                    with invest_col1:
                        st.markdown(f"**💰 Est. Monthly Rent:** ${estimated_rent:,}")
                    with invest_col2:
                        st.markdown(f"**📈 Gross Yield:** {gross_yield:.1f}%")
                    with invest_col3:
                        st.markdown(f"**🏷️ Price/Sqft:** ${row['price']/row.get('sqft', 1):.0f}")

                    # Analyze button
                    if st.button("🔍 Full Investment Analysis", key=f"analyze_listing_{idx}", use_container_width=True, type="primary"):
                        st.session_state.analyzing_listing_idx = idx if st.session_state.get('analyzing_listing_idx') != idx else None
                        st.rerun()

                    # Inline ROI Analysis (when triggered)
                    if st.session_state.get('analyzing_listing_idx') == idx:
                        st.markdown("**🧮 Quick Investment Analysis**")
                        roi_inputs = PropertyInputs(
                            purchase_price=row['price'],
                            monthly_rent=estimated_rent,
                            down_payment_pct=0.20
                        )
                        roi_metrics = components['roi'].calculate_all_metrics(roi_inputs)
                        roi_grade = components['roi'].get_investment_grade(roi_metrics)

                        roi_col1, roi_col2, roi_col3, roi_col4 = st.columns(4)
                        with roi_col1:
                            st.metric("Cap Rate", f"{roi_metrics['cap_rate']:.2f}%")
                        with roi_col2:
                            st.metric("Cash-on-Cash", f"{roi_metrics['cash_on_cash_return']:.2f}%")
                        with roi_col3:
                            st.metric("DSCR", f"{roi_metrics['dscr']:.2f}")
                        with roi_col4:
                            st.metric("Grade", roi_grade)

                        # Price prediction
                        pred = components['predictor'].predict_price_change(
                            current_price=row['price'],
                            city=row['city'],
                            property_type=row['property_type'],
                            horizon_months=12
                        )
                        st.markdown(f"""
                        - **12-Month Prediction**: ${pred['predicted_price_6m']:,.0f} ({pred['predicted_change_pct']:+.1f}%)
                        - **Market Regime**: {pred['market_regime'].upper()}
                        - **Confidence Range**: ${pred['confidence_lower']:,.0f} - ${pred['confidence_upper']:,.0f}
                        """)

                        if st.button("Close Analysis", key=f"close_analysis_{idx}"):
                            st.session_state.analyzing_listing_idx = None
                            st.rerun()
                        st.markdown("---")

                    st.markdown("---")
        else:
            st.info("No properties match your filters. Try adjusting your criteria.")
    else:
        st.markdown("""
        <div style="text-align:center; padding:3rem;">
            <div style="font-size:3rem; margin-bottom:1rem;">🏠</div>
            <div style="font-size:1.25rem; color:#64748b;">Click "Refresh Listings" to fetch current properties from public sources</div>
            <div style="font-size:0.9rem; color:#94a3b8; margin-top:0.5rem;">Data cached for 24 hours for performance</div>
        </div>
        """, unsafe_allow_html=True)

        # Show summary if available
        scraper = components['listings']
        summary = scraper.get_listing_summary()
        if summary['total'] > 0:
            st.markdown(f"**Cached:** {summary['total']} listings across {len(summary['cities'])} cities")


# =============================================================================
# TAB 5: MY RECOMMENDATIONS
# =============================================================================

with tabs[5]:
    st.markdown("### 🎯 My Recommendations")
    st.markdown("Get property recommendations tailored to your investment goals and risk profile.")

    # Initialize expanded property state
    if "expanded_property" not in st.session_state:
        st.session_state.expanded_property = None

    # Investor persona selector
    persona = st.selectbox(
        "Investor Profile",
        list(INVESTOR_PERSONAS.keys()),
        format_func=lambda x: f"{x} - {INVESTOR_PERSONAS[x]['description']}"
    )

    st.markdown(f"**Strategy:** {INVESTOR_PERSONAS[persona]['priority'].replace('_', ' ').title()} focus | "
                f"**Risk Tolerance:** {INVESTOR_PERSONAS[persona]['risk_tolerance'].title()} | "
                f"**Time Horizon:** {INVESTOR_PERSONAS[persona]['time_horizon']} years")

    # Generate heatmap data
    heatmap_data = generate_sample_heatmap_data()

    # Get personalized recommendations
    recs = get_personalized_recommendations(persona, heatmap_data)

    if len(recs) > 0:
        st.markdown("#### Top Recommendations for Your Profile")

        for i, (_, row) in enumerate(recs.head(5).iterrows()):
            is_expanded = st.session_state.expanded_property == i

            col1, col2 = st.columns([3, 1])

            regime = row['market_regime']
            regime_badge = "badge-hot" if regime == "hot" else "badge-warm" if regime == "warm" else "badge-cooling" if regime == "cooling" else "badge-cold"
            regime_colors = {"hot": "#dc2626", "warm": "#ea580c", "cooling": "#2563eb", "cold": "#374151"}
            accent_color = regime_colors.get(regime, "#0078D4")

            with col1:
                st.markdown(f"""
                <div class="result-card" style="border-left: 5px solid {accent_color};">
                    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:1rem;">
                        <div>
                            <div style="font-size:1.5rem; font-weight:800; color:#0f172a; margin:0;">{row['city']}</div>
                            <div style="font-size:0.9rem; color:#64748b; font-weight:600; background:#f1f5f9; padding:0.25rem 0.75rem; border-radius:4px; display:inline-block; margin-top:0.5rem;">
                                {row['property_type'].replace('_', ' ').title()}
                            </div>
                        </div>
                        <span class="{regime_badge}" style="box-shadow:0 2px 8px rgba(0,0,0,0.15);">{regime.upper()}</span>
                    </div>
                    <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:0.75rem; background:linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%); padding:1rem; border-radius:10px; border:1px solid #e2e8f0;">
                        <div style="text-align:center;">
                            <div style="font-size:0.7rem; color:#64748b; text-transform:uppercase; letter-spacing:0.05em; margin-bottom:0.5rem;">Price</div>
                            <div style="font-size:1.25rem; font-weight:800; color:#0f172a;">${row['current_price']/1000:.0f}K</div>
                        </div>
                        <div style="text-align:center; border-left:1px solid #e2e8f0; border-right:1px solid #e2e8f0;">
                            <div style="font-size:0.7rem; color:#64748b; text-transform:uppercase; letter-spacing:0.05em; margin-bottom:0.5rem;">Appreciation</div>
                            <div style="font-size:1.25rem; font-weight:800; color:{'#16a34a' if row['appreciation_12m']>3 else '#d97706' if row['appreciation_12m']>0 else '#dc2626'}">
                                {row['appreciation_12m']:+.1f}%
                            </div>
                        </div>
                        <div style="text-align:center;">
                            <div style="font-size:0.7rem; color:#64748b; text-transform:uppercase; letter-spacing:0.05em; margin-bottom:0.5rem;">Yield</div>
                            <div style="font-size:1.25rem; font-weight:800; color:#0f172a;">{row['rental_yield']:.1f}%</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                st.metric("Investment Score", f"{row['investment_score']:.0f}/100")
                st.metric("Buy/Rent Score", f"{row['buy_vs_rent_score']:.0f}/100")

                btn_label = "▼ Hide Analysis" if is_expanded else "▲ Analyze"
                if st.button(btn_label, key=f"rec_{i}", use_container_width=True):
                    if is_expanded:
                        st.session_state.expanded_property = None
                    else:
                        st.session_state.expanded_property = i
                    st.rerun()

            if is_expanded:
                st.markdown(f"""
                <div style="background:linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
                    padding:1.5rem; border-radius:12px; margin:0.5rem 0 1.5rem 0;
                    border:1px solid #e2e8f0; box-shadow:0 4px 12px rgba(0,0,0,0.08);">
                    <h4 style="margin-top:0; color:#0f172a;">📊 Investment Analysis: {row['city']} {row['property_type'].replace('_', ' ').title()}</h4>
                </div>
                """, unsafe_allow_html=True)

                analysis_col1, analysis_col2, analysis_col3 = st.columns(3)

                with analysis_col1:
                    st.markdown("**📈 Price Forecast**")
                    st.markdown(f"- Current: ${row['current_price']:,.0f}")
                    st.markdown(f"- 12mo Predicted: ${row['current_price'] * (1 + row['appreciation_12m']/100):,.0f}")
                    st.markdown(f"- Appreciation: {row['appreciation_12m']:+.1f}%")

                with analysis_col2:
                    monthly_rent = row['current_price'] * row['rental_yield'] / 12
                    st.markdown("**💰 Cash Flow**")
                    st.markdown(f"- Est. Monthly Rent: ${monthly_rent:,.0f}")
                    st.markdown(f"- Gross Yield: {row['rental_yield']:.1f}%")
                    st.markdown(f"- Cap Rate: ~{row['rental_yield'] * 0.7:.1f}%")

                with analysis_col3:
                    st.markdown("**⚖️ Recommendation**")
                    if row['investment_score'] >= 70:
                        st.success("**STRONG BUY**")
                    elif row['investment_score'] >= 50:
                        st.info("**BUY**")
                    else:
                        st.warning("**HOLD**")

                st.markdown("**⚡ Quick Actions:**")
                action_col1, action_col2, action_col3 = st.columns(3)
                with action_col1:
                    if st.button("🧮 Calculate ROI", key=f"roi_{i}", use_container_width=True):
                        st.session_state.inline_roi_property = {
                            'price': row['current_price'],
                            'rent': monthly_rent,
                            'city': row['city'],
                            'property_type': row['property_type']
                        }
                        st.rerun()
                with action_col2:
                    if st.button("🤖 Ask AI", key=f"ai_{i}", use_container_width=True):
                        st.session_state.inline_ai_query = f"Analyze {row['city']} {row['property_type'].replace('_', ' ')} as investment - price ${row['current_price']:,.0f}, yield {row['rental_yield']:.1f}%, investment score {row['investment_score']:.0f}/100"
                        st.rerun()
                with action_col3:
                    if st.button("📍 Compare Cities", key=f"compare_{i}", use_container_width=True):
                        st.session_state.inline_compare_city = row['city']
                        st.rerun()

                # Inline ROI Calculator (when triggered)
                if st.session_state.get('inline_roi_property') and st.session_state.inline_roi_property.get('city') == row['city']:
                    st.markdown("---")
                    st.markdown("**🧮 Quick ROI Analysis**")
                    roi_prop = st.session_state.inline_roi_property
                    roi_inputs = PropertyInputs(
                        purchase_price=roi_prop['price'],
                        monthly_rent=roi_prop['rent'],
                        down_payment_pct=0.20
                    )
                    roi_metrics = components['roi'].calculate_all_metrics(roi_inputs)
                    roi_grade = components['roi'].get_investment_grade(roi_metrics)

                    roi_col1, roi_col2, roi_col3, roi_col4 = st.columns(4)
                    with roi_col1:
                        st.metric("Cap Rate", f"{roi_metrics['cap_rate']:.2f}%")
                    with roi_col2:
                        st.metric("Cash-on-Cash", f"{roi_metrics['cash_on_cash_return']:.2f}%")
                    with roi_col3:
                        st.metric("DSCR", f"{roi_metrics['dscr']:.2f}")
                    with roi_col4:
                        st.metric("Grade", roi_grade)

                    if st.button("Close ROI", key="close_inline_roi"):
                        st.session_state.inline_roi_property = None
                        st.rerun()
                    st.markdown("---")

                # Inline AI Analysis (when triggered)
                if st.session_state.get('inline_ai_query') and row['city'] in st.session_state.inline_ai_query:
                    st.markdown("---")
                    st.markdown("**🤖 AI Investment Analysis**")

                    ai_query = st.session_state.inline_ai_query
                    with st.spinner("AI is analyzing..."):
                        ai_response = components['chatbot'].get_response(ai_query)

                    st.markdown(ai_response)

                    if st.button("Close AI Analysis", key="close_inline_ai"):
                        st.session_state.inline_ai_query = None
                        st.rerun()
                    st.markdown("---")

                # Inline City Comparison (when triggered)
                if st.session_state.get('inline_compare_city'):
                    st.markdown("---")
                    st.markdown(f"**📍 {st.session_state.inline_compare_city} vs Other Markets**")

                    compare_city = st.session_state.inline_compare_city
                    city_data = heatmap_data[heatmap_data['city'] == compare_city]
                    other_cities = heatmap_data[heatmap_data['city'] != compare_city].groupby('city').first().reset_index()

                    if len(city_data) > 0 and len(other_cities) > 0:
                        comparison_df = pd.concat([
                            city_data[['city', 'property_type', 'current_price', 'appreciation_12m', 'rental_yield', 'investment_score']],
                            other_cities[['city', 'property_type', 'current_price', 'appreciation_12m', 'rental_yield', 'investment_score']].head(5)
                        ], ignore_index=True)

                        st.dataframe(comparison_df.round(2), use_container_width=True)

                        # Visual comparison
                        viz_col1, viz_col2 = st.columns(2)
                        with viz_col1:
                            fig = px.bar(comparison_df, x='city', y='investment_score',
                                        title='Investment Score Comparison',
                                        color='investment_score', color_continuous_scale='RdYlGn')
                            st.plotly_chart(fig, use_container_width=True)
                        with viz_col2:
                            fig = px.scatter(comparison_df, x='current_price', y='appreciation_12m',
                                            size='rental_yield', color='city',
                                            title='Price vs Appreciation',
                                            hover_data=['property_type'])
                            st.plotly_chart(fig, use_container_width=True)

                    if st.button("Close Comparison", key="close_inline_compare"):
                        st.session_state.inline_compare_city = None
                        st.rerun()
                    st.markdown("---")

                st.markdown("")
    else:
        st.info("No matching properties found. Try adjusting your criteria.")

    # Budget filter
    st.markdown("---")
    st.markdown("### Filter by Budget")

    budget_col1, budget_col2 = st.columns(2)

    with budget_col1:
        min_budget = st.number_input("Min Budget ($)", min_value=0, max_value=10000000, value=300000, step=50000, key="min_budget")
    with budget_col2:
        max_budget = st.number_input("Max Budget ($)", min_value=0, max_value=10000000, value=1500000, step=50000, key="max_budget")

    filtered = heatmap_data[(heatmap_data["current_price"] >= min_budget) & (heatmap_data["current_price"] <= max_budget)]

    if len(filtered) > 0:
        st.dataframe(filtered[["city", "property_type", "current_price", "appreciation_12m", "rental_yield", "investment_score"]].round(2), use_container_width=True)
