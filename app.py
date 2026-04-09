import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys
import json

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

    /* Navigation tabs - modern pill style */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        padding: 0.5rem;
        background: #f1f5f9;
        border-radius: 12px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 14px 28px;
        border-radius: 10px;
        font-weight: 600;
        font-size: 0.95rem;
        transition: all 0.2s ease;
        border: none;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(0, 120, 212, 0.08);
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: linear-gradient(135deg, #0078D4 0%, #00BCF2 100%);
        color: white;
        box-shadow: 0 4px 12px rgba(0, 120, 212, 0.3);
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

    return {
        'predictor': predictor,
        'model_loaded': model_loaded,
        'heatmap': heatmap_gen,
        'chatbot': chatbot,
        'roi': roi_calc
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
# SIDEBAR
# =============================================================================

st.sidebar.image("https://img.icons8.com/color/96/000000/house.png", width=80)
st.sidebar.title("Propra")
st.sidebar.markdown("**AI-Powered Canadian Real Estate Platform**")
st.sidebar.markdown("---")

components = load_components()

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
    <div class="hero-title">🏠 Propra</div>
    <div class="hero-subtitle">Canada's AI-Powered Real Estate Intelligence Platform</div>
    <div class="hero-tag">✨ Predictive Analytics &bull; Investment Insights &bull; Market Intelligence</div>
</div>
""", unsafe_allow_html=True)


# =============================================================================
# MAIN NAVIGATION
# =============================================================================

tabs = st.tabs([
    "🔮 Price Predictor",
    "🗺️ Market Heatmap",
    "🤖 AI Advisor",
    "💰 ROI Calculator",
    "📈 Scenario Simulator",
    "🎯 My Recommendations",
    "💎 Hidden Gems",
    "📚 Case Studies",
    "🧠 Neighborhood Analysis"
])


# =============================================================================
# TAB 1: PRICE PREDICTOR
# =============================================================================

with tabs[0]:
    st.markdown("### AI Price Prediction")
    st.markdown("Get ML-powered price forecasts with confidence intervals and market analysis.")

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

    if st.button("🔮 Get Prediction", type="primary", use_container_width=True):
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
        if "BUY" in rec:
            badge = "badge-buy"
            bvr_rec = "Buy"
        elif "RENT" in rec:
            badge = "badge-rent"
            bvr_rec = "Rent"
        else:
            badge = "badge-hold"
            bvr_rec = "Hold"

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
            st.markdown(f"Based on market volatility")

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

        # Confidence interval
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


# =============================================================================
# TAB 2: MARKET HEATMAP
# =============================================================================

with tabs[1]:
    st.markdown("### Market Heatmap")
    st.markdown("Visualize market opportunities across Canadian cities with interactive heatmaps.")

    heatmap_data = generate_sample_heatmap_data()
    heatmap_gen = components['heatmap']

    # Visualization type selector
    viz_type = st.selectbox(
        "Visualization Type",
        ["Investment Score", "Appreciation (12m)", "Rental Yield", "Cap Rate", "Buy vs Rent Score"]
    )

    col1, col2 = st.columns([2, 1])

    # Metric description cards
    metric_info = {
        "Investment Score": ("0-100 score combining growth, yield & risk", "🎯", "#0078D4"),
        "Appreciation (12m)": ("Predicted price growth over 12 months", "📈", "#22c55e"),
        "Rental Yield": ("Annual rent as % of property price", "💰", "#f59e0b"),
        "Cap Rate": ("Net operating income / property value", "📊", "#8b5cf6"),
        "Buy vs Rent Score": ("Higher = better to buy than rent", "⚖️", "#06b6d4")
    }

    info_color = metric_info[viz_type][2]
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, {info_color}15 0%, {info_color}08 100%);
                padding: 1rem 1.25rem; border-radius: 12px; margin-bottom: 1.5rem;
                border: 1px solid {info_color}30; display: flex; align-items: center; gap: 0.75rem;">
        <span style="font-size: 1.5rem;">{metric_info[viz_type][1]}</span>
        <div>
            <div style="font-weight: 600; color: #1a1a1a; margin-bottom: 0.25rem;">{viz_type}</div>
            <div style="font-size: 0.875rem; color: #6b7280;">{metric_info[viz_type][0]}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

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
            center=dict(lat=52, lon=-105),
            projection_type="mercator",
            bgcolor="rgba(248, 250, 252, 0.8)",
            lakecolor="rgba(186, 221, 255, 0.6)",
            landcolor="rgba(241, 245, 249, 0.8)",
            showland=True,
            showlakes=True,
            countrycolor="#94a3b8",
            countrywidth=1,
            showcoastlines=True,
            coastlinecolor="#64748b",
            coastlinewidth=1.5,
            showsubunits=True,
            subunitcolor="#cbd5e1",
            subunitwidth=1
        ),
        height=550,
        margin=dict(l=20, r=20, t=60, b=20),
        paper_bgcolor="rgba(255,255,255,0.95)",
        plot_bgcolor="rgba(255,255,255,0.95)"
    )

    st.plotly_chart(fig_map, use_container_width=True)

    with col2:
        # Market summary
        st.markdown("#### Market Summary")

        summary = heatmap_gen.generate_city_summary(heatmap_data)
        st.dataframe(
            summary.round(2),
            use_container_width=True,
            hide_index=True
        )

        # Market Regime Legend
        with st.expander("📊 What do market regimes mean?", expanded=False):
            st.markdown("""
            | Regime | Description | What It Means |
            |--------|-------------|---------------|
            | 🔴 **HOT** | >5% appreciation, low inventory | High demand, rapid price growth, expect competition |
            | 🟠 **WARM** | 2-5% appreciation, stable market | Steady growth, balanced buyer/seller conditions |
            | 🔵 **COOLING** | 0-2% appreciation, rising inventory | Slowing demand, buyer gaining advantage |
            | ⚫ **COLD** | Negative appreciation, high inventory | Low demand, price declines, distressed opportunities |
            """)

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


# =============================================================================
# TAB 3: AI CHATBOT
# =============================================================================

with tabs[2]:
    st.markdown("### AI Property Advisor")
    st.markdown("Ask questions about Canadian real estate and get instant, data-backed answers.")

    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    chatbot = components['chatbot']

    # Display chat history
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(f'<div class="chat-message chat-user">{msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="chat-message chat-bot">
                {msg["content"]}
                <div style="margin-top:0.75rem;font-size:0.85rem;color:#666;border-top:1px solid #e2e8f0;padding-top:0.5rem;">
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
# TAB 5: SCENARIO SIMULATOR
# =============================================================================

with tabs[4]:
    st.markdown("### Investment Scenario Simulator")
    st.markdown("Multi-year wealth projection with Monte Carlo uncertainty modeling.")

    col1, col2 = st.columns(2)

    with col1:
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

    with col2:
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
                f"${base['net_sale_proceeds']:,.0f}",
                "After selling costs"
            )

        with r3:
            cagr_val = mc['cagr_mean'] * 100 if not np.isnan(mc['cagr_mean']) else 0
            cagr_std = mc['cagr_std'] * 100 if not np.isnan(mc['cagr_std']) else 0
            st.metric(
                "Expected CAGR",
                f"{cagr_val:.1f}%",
                f"±{cagr_std:.1f}%"
            )

        with r4:
            st.metric(
                "Probability >5% Return",
                f"{mc['probability_beat_5pct']*100:.0f}%",
                risk["risk_level"] + " Risk"
            )

        # Projection chart
        st.markdown("#### Wealth Projection Over Time")

        years = base["years"]
        property_values = base["property_values"]
        equity = base["equity"]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=years,
            y=property_values,
            mode="lines+markers",
            name="Property Value",
            line=dict(color="#0078D4", width=3)
        ))
        fig.add_trace(go.Scatter(
            x=years,
            y=equity,
            mode="lines+markers",
            name="Equity",
            line=dict(color="#22c55e", width=3)
        ))

        fig.update_layout(
            height=400,
            xaxis_title="Year",
            yaxis_title="Value ($)",
            yaxis_tickformat="$,.0f",
            showlegend=True
        )

        st.plotly_chart(fig, use_container_width=True)

        # Monte Carlo results
        st.markdown("#### Monte Carlo Analysis")

        mc1, mc2, mc3 = st.columns(3)

        with mc1:
            st.markdown("**Value Range (95% Confidence)**")
            st.markdown(f"""
            - 5th Percentile: ${mc['final_value_p5']:,.0f}
            - Expected: ${mc['final_value_mean']:,.0f}
            - 95th Percentile: ${mc['final_value_p95']:,.0f}
            """)

        with mc2:
            st.markdown("**Return Probabilities**")
            st.markdown(f"""
            - Positive Return: {mc['probability_positive_return']*100:.0f}%
            - Beat 5% CAGR: {mc['probability_beat_5pct']*100:.0f}%
            - Beat 10% CAGR: {mc['probability_beat_10pct']*100:.0f}%
            """)

        with mc3:
            st.markdown("**Risk Metrics**")
            st.markdown(f"""
            - Risk Level: {risk['risk_level']}
            - Risk Score: {risk['risk_score']:.0f}/100
            - Volatility: {risk['volatility']*100:.1f}%
            """)

        # Scenario comparison
        st.markdown("#### Scenario Comparison")
        st.dataframe(
            results["scenario_comparison"].round(2),
            use_container_width=True
        )


# =============================================================================
# TAB 6: PERSONALIZED RECOMMENDATIONS
# =============================================================================

with tabs[5]:
    st.markdown("### Personalized Recommendations")
    st.markdown("Get property recommendations tailored to your investment goals and risk profile.")

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
            col1, col2, col3 = st.columns([2, 1, 1])

            with col1:
                st.markdown(f"""
                <div class="result-card">
                    <div style="display:flex;justify-content:space-between;align-items:center;">
                        <div>
                            <h3 style="margin:0">{row['city']}</h3>
                            <small style="color:#666">{row['property_type'].replace('_', ' ').title()}</small>
                        </div>
                        <span class="badge-{'hot' if row['market_regime']=='hot' else 'warm' if row['market_regime']=='warm' else 'cooling'}">
                            {row['market_regime'].upper()}
                        </span>
                    </div>
                    <div style="margin-top:1rem;display:grid;grid-template-columns:repeat(3,1fr);gap:0.5rem;">
                        <div>
                            <small style="color:#666">Price</small><br>
                            <strong>${row['current_price']:,.0f}</strong>
                        </div>
                        <div>
                            <small style="color:#666">Appreciation</small><br>
                            <strong style="color:{'#22c55e' if row['appreciation_12m']>3 else '#f59e0b' if row['appreciation_12m']>0 else '#ef4444'}">
                                {row['appreciation_12m']:+.1f}%
                            </strong>
                        </div>
                        <div>
                            <small style="color:#666">Yield</small><br>
                            <strong>{row['rental_yield']:.2f}%</strong>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                st.metric("Investment Score", f"{row['investment_score']:.0f}/100")
                st.metric("Buy/Rent Score", f"{row['buy_vs_rent_score']:.0f}/100")

            with col3:
                if st.button("Analyze", key=f"rec_{i}"):
                    st.session_state.chat_history.append({
                        "role": "user",
                        "content": f"Tell me about {row['city']} {row['property_type']} investment"
                    })
                    st.success("Switch to AI Advisor tab for details!")

            st.markdown("")
    else:
        st.info("No matching properties found. Try adjusting your criteria.")

    # Budget filter
    st.markdown("---")
    st.markdown("### Filter by Budget")

    budget_col1, budget_col2 = st.columns(2)

    with budget_col1:
        min_budget = st.number_input(
            "Min Budget ($)",
            min_value=0,
            max_value=10000000,
            value=300000,
            step=50000,
            key="min_budget"
        )

    with budget_col2:
        max_budget = st.number_input(
            "Max Budget ($)",
            min_value=0,
            max_value=10000000,
            value=1500000,
            step=50000,
            key="max_budget"
        )

    filtered = heatmap_data[
        (heatmap_data["current_price"] >= min_budget) &
        (heatmap_data["current_price"] <= max_budget)
    ]

    if len(filtered) > 0:
        st.dataframe(
            filtered[["city", "property_type", "current_price", "appreciation_12m", "rental_yield", "investment_score"]].round(2),
            use_container_width=True
        )


# =============================================================================
# TAB 7: HIDDEN GEMS
# =============================================================================

with tabs[6]:
    st.markdown("### Hidden Gems Engine")
    st.markdown("AI-powered detection of undervalued properties with high potential returns.")

    heatmap_data = generate_sample_heatmap_data()
    heatmap_gen = components['heatmap']

    # Get undervalued properties
    undervalued = heatmap_gen.get_undervalued_properties(heatmap_data)

    if len(undervalued) > 0:
        st.success(f"Found {len(undervalued)} potentially undervalued properties!")

        for i, (_, row) in enumerate(undervalued.iterrows()):
            st.markdown("---")

            col1, col2 = st.columns([2, 1])

            with col1:
                st.markdown(f"""
                <div class="result-card">
                    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:1rem;">
                        <div>
                            <h3 style="margin:0">🔍 {row['city']} - {row['property_type'].replace('_', ' ').title()}</h3>
                            <small style="color:#666">Undervalue Score: {row['undervalue_score']:.1f}</small>
                        </div>
                        <span class="badge-buy">POTENTIAL GEM</span>
                    </div>

                    <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:1rem;">
                        <div>
                            <small style="color:#666">Current Price</small><br>
                            <strong style="font-size:1.25rem">${row['current_price']:,.0f}</strong>
                        </div>
                        <div>
                            <small style="color:#666">vs Market Avg</small><br>
                            <strong style="font-size:1.25rem;color:#22c55e">{(1 - row['current_price']/heatmap_data['current_price'].mean())*100:.1f}% below</strong>
                        </div>
                        <div>
                            <small style="color:#666">Rental Yield</small><br>
                            <strong style="font-size:1.25rem">{row['rental_yield']:.2f}%</strong>
                        </div>
                        <div>
                            <small style="color:#666">Investment Score</small><br>
                            <strong style="font-size:1.25rem">{row['investment_score']:.0f}/100</strong>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                st.markdown("**Why It's Undervalued:**")
                st.markdown(f"""
                - Price is {((1 - row['current_price']/heatmap_data['current_price'].mean())*100):.0f}% below market average
                - Rental yield of {row['rental_yield']:.2f}% exceeds median
                - Market regime: {row['market_regime'].title()}
                """)

                if st.button("View Details", key=f"gem_{i}", use_container_width=True):
                    st.info("Use ROI Calculator tab for detailed analysis!")
    else:
        st.info("No undervalued properties detected in current market conditions.")

    # Market opportunities summary
    st.markdown("---")
    st.markdown("### Market Opportunities Summary")

    opp1, opp2, opp3 = st.columns(3)

    with opp1:
        st.markdown("**Hot Markets**")
        hot = heatmap_gen.get_hot_markets(heatmap_data)
        if len(hot) > 0:
            for _, row in hot.head(3).iterrows():
                st.markdown(f"- {row['city']} {row['property_type'].replace('_', ' ').title()}: {row['appreciation_12m']:.1f}%")
        else:
            st.markdown("No hot markets currently")

    with opp2:
        st.markdown("**Highest Yield**")
        highest_yield = heatmap_data.nlargest(3, "rental_yield")
        for _, row in highest_yield.iterrows():
            st.markdown(f"- {row['city']} {row['property_type'].replace('_', ' ').title()}: {row['rental_yield']:.2f}%")

    with opp3:
        st.markdown("**Best Investment Scores**")
        best_score = heatmap_data.nlargest(3, "investment_score")
        for _, row in best_score.iterrows():
            st.markdown(f"- {row['city']} {row['property_type'].replace('_', ' ').title()}: {row['investment_score']:.0f}/100")


# =============================================================================
# TAB 8: CASE STUDIES
# =============================================================================

with tabs[7]:
    st.markdown("### Real Estate Case Studies")
    st.markdown("Deep-dive analysis of real investment properties with historical performance and lessons learned.")

    case_studies = get_all_case_studies()

    # Case study selector
    selected_id = st.selectbox(
        "Select Case Study",
        [cs.id for cs in case_studies],
        format_func=lambda x: next(cs.title for cs in case_studies if cs.id == x)
    )

    selected_cs = next(cs for cs in case_studies if cs.id == selected_id)
    presentation = create_case_study_presentation(selected_cs)

    st.markdown("---")

    # Overview
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown(f"#### {selected_cs.title}")
        st.markdown(selected_cs.description)

        st.markdown("**Investment Thesis:**")
        st.markdown(selected_cs.investment_thesis)

    with col2:
        st.markdown("**Key Metrics**")
        metrics = selected_cs.key_metrics
        st.metric("Purchase Price", f"${metrics['purchase_price']:,}")
        st.metric("Current Value", f"${metrics['current_value']:,}")
        st.metric("Total Appreciation", metrics['total_appreciation'])
        st.metric("Annualized Return", metrics['annualized_appreciation'])
        st.metric("Gross Yield", metrics['gross_yield'])
        st.metric("Cap Rate", metrics['cap_rate'])
        st.metric("Total ROI", metrics['total_roi'])

    st.markdown("")

    # Charts
    if selected_cs.charts:
        st.markdown("#### Performance Charts")
        if "price_history" in selected_cs.charts:
            st.plotly_chart(selected_cs.charts["price_history"], use_container_width=True)
        if "rent_growth" in selected_cs.charts:
            st.plotly_chart(selected_cs.charts["rent_growth"], use_container_width=True)

    # Outcome and lessons
    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Outcome")
        st.markdown(selected_cs.outcome)

    with col2:
        st.markdown("#### Lessons Learned")
        for lesson in selected_cs.lessons:
            st.markdown(f"• {lesson}")

    # Forecasts
    st.markdown("---")
    st.markdown("#### ML Forecasts")

    forecast_cols = st.columns(3)
    for i, (horizon, forecast) in enumerate(selected_cs.predictions.items()):
        with forecast_cols[i]:
            st.markdown(f"**{horizon.replace('_', ' ').title()}**")
            st.metric("Predicted Price", f"${forecast['price']:,}")
            st.caption(f"Confidence: {forecast['confidence'].title()}")

    # Comparison table
    st.markdown("---")
    st.markdown("### Compare All Case Studies")

    comparison_df = compare_case_studies(case_studies)
    st.dataframe(comparison_df, use_container_width=True)


# =============================================================================
# TAB 9: NEIGHBORHOOD ANALYSIS
# =============================================================================

with tabs[8]:
    st.markdown("### Neighborhood Graph Analysis")
    st.markdown("Graph-based modeling of spatial relationships between properties, amenities, and neighborhoods.")

    # Create sample graph
    @st.cache_resource
    def get_neighborhood_graph():
        return create_sample_graph()

    graph = get_neighborhood_graph()

    st.markdown("---")

    # Graph statistics
    stat1, stat2, stat3, stat4 = st.columns(4)

    with stat1:
        st.metric("Properties", len(graph.properties))

    with stat2:
        st.metric("Amenities", len(graph.amenities))

    with stat3:
        st.metric("Neighborhoods", len(graph.neighborhoods))

    with stat4:
        st.metric("Graph Edges", graph.graph.number_of_edges())

    st.markdown("")

    # Property scores
    st.markdown("### Location Scores")
    st.markdown("Walkability, transit access, and overall location quality based on graph analysis.")

    scores = graph.compute_property_scores()

    score_data = []
    for prop_id, prop_scores in scores.items():
        prop = graph.properties[prop_id]
        score_data.append({
            "Property": f"{prop.neighborhood} - {prop.property_type}",
            "Price": prop.price,
            "Walkability": prop_scores["walkability_score"],
            "Transit Score": prop_scores["transit_score"],
            "Connectivity": prop_scores["connectivity_score"],
            "Location Score": prop_scores["overall_location_score"]
        })

    score_df = pd.DataFrame(score_data)
    st.dataframe(score_df, use_container_width=True)

    # Neighborhood summaries
    st.markdown("### Neighborhood Summaries")

    neigh_cols = st.columns(len(graph.neighborhoods))

    for i, (neigh_id, neigh) in enumerate(graph.neighborhoods.items()):
        with neigh_cols[i]:
            summary = graph.get_neighborhood_summary(neigh_id)
            if "error" not in summary:
                st.markdown(f"**{summary['name']}**")
                st.caption(summary['city'])
                st.markdown(f"Properties: {summary['property_count']}")
                st.markdown(f"Avg Price: ${summary['avg_price']:,.0f}")
                st.markdown(f"Walkability: {summary['walkability_score']}")
                st.markdown(f"Location Score: {summary['avg_location_score']}")

    # Market regime detection
    st.markdown("---")
    st.markdown("### Market Regime Detection")

    detector = MarketRegimeDetector()
    regimes = detector.define_regimes()

    regime_cols = st.columns(4)
    for i, (regime, info) in enumerate(regimes.items()):
        with regime_cols[i % 4]:
            st.markdown(f"**{regime.upper()}**")
            st.caption(info["description"])
            st.markdown(f"**Action:** {info['investor_action']}")

    # Interactive regime detector
    st.markdown("#### Test Market Regime Detection")

    reg_col1, reg_col2, reg_col3 = st.columns(3)

    with reg_col1:
        test_momentum = st.slider("Price Momentum (%)", -5.0, 10.0, 3.0, 0.5)
    with reg_col2:
        test_dom = st.slider("Days on Market", 10, 150, 45, 5)
    with reg_col3:
        test_inventory = st.slider("Inventory Change (%)", -30, 30, 0, 5)

    test_metrics = {
        "price_momentum": test_momentum,
        "days_on_market": test_dom,
        "inventory_change": test_inventory
    }

    detected_regime, confidence, details = detector.detect_regime(test_metrics)

    st.markdown(f"**Detected Regime:** :{regimes[detected_regime]['color']}[{detected_regime.upper()}] ({confidence:.0%} confidence)")
    st.markdown(f"**Recommendation:** {details['investor_action']}")


# =============================================================================
# FOOTER
# =============================================================================

st.markdown("---")
st.markdown("""
<div style="text-align:center;color:#666;padding:2rem;">
    <p><strong>⚠️ Disclaimer:</strong> This platform provides informational analysis only and does not constitute financial advice.
    Real estate investments carry risks. Always consult with qualified professionals before making investment decisions.</p>
    <p style="font-size:0.875rem;margin-top:1rem;">
    Built with ❤️ for Canadian real estate investors | Powered by Machine Learning
    </p>
</div>
""", unsafe_allow_html=True)
