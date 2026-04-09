import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from predict import PricePredictor, MarketAnalyzer
from buy_vs_rent import BuyVsRentCalculator, BuyVsRentInputs, quick_analysis
from recommender import PropertyRecommender, BuyerProfile, get_recommendations_for_budget


# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="Propra - AI-Powered Canadian Real Estate",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# =============================================================================
# CUSTOM CSS FOR POLISHED LOOK
# =============================================================================

st.markdown("""
<style>
    /* Hide default Streamlit elements for cleaner look */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Hero section styling */
    .hero-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 3rem;
        border-radius: 1rem;
        color: white;
        margin-bottom: 2rem;
    }
    .hero-title {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .hero-subtitle {
        font-size: 1.2rem;
        opacity: 0.9;
    }

    /* Result cards */
    .result-card {
        background: white;
        border-radius: 0.75rem;
        padding: 1.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }

    /* Investment score display */
    .score-high { color: #22c55e; font-weight: bold; }
    .score-medium { color: #f59e0b; font-weight: bold; }
    .score-low { color: #ef4444; font-weight: bold; }

    /* Recommendation badges */
    .badge-buy {
        background: #22c55e;
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-weight: 600;
    }
    .badge-rent {
        background: #ef4444;
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-weight: 600;
    }
    .badge-hold {
        background: #f59e0b;
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# MODEL LOADING
# =============================================================================

@st.cache_resource
def load_model():
    model_dir = str(Path(__file__).parent / "models")

    predictor = PricePredictor(model_dir=model_dir)
    try:
        predictor.load_model()
        return predictor, True
    except FileNotFoundError:
        return predictor, False


# =============================================================================
# INVESTMENT SCORE CALCULATOR
# =============================================================================

def calculate_investment_score(prediction, buy_vs_rent_result, risk_level):
    """Calculate 0-100 investment score from multiple factors."""

    # Growth score (0-40 points) - based on predicted appreciation
    appreciation = prediction.get("predicted_change_pct", 0)
    if appreciation >= 5:
        growth_score = 40
    elif appreciation >= 3:
        growth_score = 30
    elif appreciation >= 1:
        growth_score = 20
    elif appreciation >= 0:
        growth_score = 10
    else:
        growth_score = 0

    # Buy vs Rent score (0-30 points)
    recommendation = buy_vs_rent_result.get("recommend", {}).get("recommendation", "")
    if "STRONG_BUY" in recommendation:
        bvr_score = 30
    elif "MODERATE_BUY" in recommendation:
        bvr_score = 25
    elif "WEAK_BUY" in recommendation:
        bvr_score = 20
    elif "STRONG_RENT" in recommendation:
        bvr_score = 0
    elif "MODERATE_RENT" in recommendation:
        bvr_score = 5
    else:
        bvr_score = 10

    # Risk score (0-30 points)
    if risk_level == "Low":
        risk_score = 30
    elif risk_level == "Medium":
        risk_score = 20
    else:
        risk_score = 10

    total_score = growth_score + bvr_score + risk_score

    # Determine confidence
    if total_score >= 70:
        confidence = "High"
    elif total_score >= 50:
        confidence = "Medium"
    else:
        confidence = "Low"

    return total_score, confidence


def get_risk_level(city, property_type, appreciation):
    """Determine risk level based on market factors."""

    # Base risk by property type
    type_risk = {
        "condo": "medium",
        "townhouse": "low",
        "detached": "medium",
        "multi_family": "low"
    }

    # Adjust by city market stability
    stable_markets = ["Vancouver", "North Vancouver", "Burnaby"]
    volatile_markets = ["Calgary"]

    base_risk = type_risk.get(property_type, "medium")

    if city in volatile_markets:
        if base_risk == "low":
            base_risk = "medium"
        else:
            base_risk = "high"
    elif city in stable_markets and base_risk == "medium":
        base_risk = "low"

    # Adjust for extreme appreciation predictions
    if appreciation > 6:
        base_risk = "high"  # High growth = higher risk
    elif appreciation < -2:
        base_risk = "high"  # Declining market = higher risk

    return base_risk.capitalize()


# =============================================================================
# REASONING GENERATOR
# =============================================================================

def generate_reasoning(city, property_type, prediction, rates_info=None):
    """Generate human-readable explanation for the prediction."""

    reasons = []

    appreciation = prediction.get("predicted_change_pct", 0)

    # Market trend based on appreciation
    if appreciation > 4:
        reasons.append(f"Strong demand expected in {city} market")
    elif appreciation > 2:
        reasons.append(f"Steady growth projected for {city} {property_type}s")
    elif appreciation > 0:
        reasons.append(f"Modest appreciation expected in {city}")
    else:
        reasons.append(f"Market cooling expected in {city}")

    # Property type insights
    type_insights = {
        "condo": "Condo market shows high demand from first-time buyers",
        "townhouse": "Townhouses offer balanced space and affordability",
        "detached": "Detached homes remain premium long-term investments",
        "multi_family": "Multi-family properties benefit from rental income potential"
    }
    reasons.append(type_insights.get(property_type, ""))

    # Interest rate context
    if rates_info:
        mortgage_rate = rates_info.get("mortgage_rate", 5.0)
        if mortgage_rate < 4:
            reasons.append("Current mortgage rates are favorable for buyers")
        elif mortgage_rate > 6:
            reasons.append("Higher rates may cool short-term demand")

    return reasons


# =============================================================================
# SIDEBAR
# =============================================================================

st.sidebar.title("🏠 Propra")
st.sidebar.markdown("---")
st.sidebar.markdown("**AI-Powered Canadian Real Estate**")
st.sidebar.markdown("Predict prices, compare neighborhoods, and get smart buy/rent recommendations.")
st.sidebar.markdown("---")

# Model status
_, model_loaded = load_model()
if model_loaded:
    st.sidebar.success("✅ ML Model loaded")
else:
    st.sidebar.info("ℹ️ Using fallback predictions")


# =============================================================================
# HERO SECTION
# =============================================================================

st.markdown("""
<div class="hero-section">
    <div class="hero-title">🏠 Propra</div>
    <div class="hero-subtitle">Predict Canadian property prices and decide whether to buy or rent — powered by AI</div>
</div>
""", unsafe_allow_html=True)


# =============================================================================
# MAIN INPUT FORM
# =============================================================================

st.markdown("### Start Your Analysis")
st.markdown("Select your property details to get instant AI-powered insights.")

col1, col2 = st.columns([2, 1])

with col1:
    input_col1, input_col2 = st.columns(2)

    with input_col1:
        city = st.selectbox(
            "📍 City",
            ["Vancouver", "Toronto", "Calgary", "Burnaby", "Richmond", "North Vancouver"],
            index=0
        )

        property_type = st.selectbox(
            "🏢 Property Type",
            ["condo", "townhouse", "detached", "multi_family"],
            format_func=lambda x: x.replace("_", " ").title()
        )

    with input_col2:
        current_price = st.number_input(
            "💰 Current Price ($)",
            min_value=200000,
            max_value=10000000,
            value=750000,
            step=25000
        )

        monthly_rent = st.number_input(
            "🏠 Equivalent Monthly Rent ($)",
            min_value=500,
            max_value=20000,
            value=2600,
            step=100,
            help="Optional - used for Buy vs Rent calculation"
        )

# Scenario slider
st.markdown("### What-If Scenario")
st.markdown("Adjust mortgage rates to see how market changes affect your prediction.")

rate_adjustment = st.slider(
    "Mortgage Rate Change",
    min_value=-2.0,
    max_value=3.0,
    value=0.0,
    step=0.25,
    format="%.2f%%",
    help="Simulate what happens if rates go up or down"
)

# Analyze button
analyze_btn = st.button("🔮 Get Prediction", type="primary", use_container_width=True)


# =============================================================================
# RESULTS SECTION
# =============================================================================

if analyze_btn:
    predictor, model_loaded = load_model()

    # Get prediction
    prediction = predictor.predict_price_change(
        current_price=current_price,
        city=city,
        property_type=property_type,
        horizon_months=12
    )

    # Apply scenario adjustment
    if rate_adjustment != 0:
        # Higher rates typically reduce appreciation by ~0.5% per 1% rate increase
        rate_impact = -0.5 * rate_adjustment
        adjusted_appreciation = prediction["predicted_change_pct"] + rate_impact
        adjusted_price = current_price * (1 + adjusted_appreciation / 100)
    else:
        adjusted_price = prediction["predicted_price_6m"]
        adjusted_appreciation = prediction["predicted_change_pct"]

    # Get buy vs rent analysis
    bvr_results = quick_analysis(
        purchase_price=current_price,
        monthly_rent=monthly_rent,
        city=city,
        property_type=property_type,
        time_horizon_years=5
    )

    # Determine recommendation
    rec = bvr_results["recommendation"]["recommendation"]
    if "BUY" in rec:
        buy_rent_rec = "Buy"
        rec_badge = "badge-buy"
    elif "RENT" in rec:
        buy_rent_rec = "Rent"
        rec_badge = "badge-rent"
    else:
        buy_rent_rec = "Hold"
        rec_badge = "badge-hold"

    # Get risk level
    risk_level = get_risk_level(city, property_type, adjusted_appreciation)

    # Calculate investment score
    investment_score, confidence = calculate_investment_score(
        {"predicted_change_pct": adjusted_appreciation},
        {"recommend": {"recommendation": rec}},
        risk_level
    )

    # Get reasoning
    try:
        from src.rates import get_current_rates
        rates_info = get_current_rates()
        mortgage_rate = rates_info.get("sources", {}).get("mortgage_rates", {}).get("rates_by_type", {}).get("fixed_5yr", 5.0)
    except:
        mortgage_rate = 5.0
        rates_info = None

    reasoning = generate_reasoning(city, property_type, prediction, rates_info)

    st.markdown("---")

    # =============================================================================
    # PRIMARY RESULTS
    # =============================================================================

    st.markdown("### Your Results")

    # Top row: Price prediction, Buy/Rent, Risk
    result_col1, result_col2, result_col3 = st.columns(3)

    with result_col1:
        st.markdown("#### Predicted Price (12 months)")
        st.markdown(f"<div style='font-size: 2rem; font-weight: 700; color: #667eea'>${adjusted_price:,.0f}</div>", unsafe_allow_html=True)
        delta_str = f"{adjusted_appreciation:+.1f}%"
        if adjusted_appreciation >= 0:
            st.metric(label="Expected Change", value=delta_str, delta=delta_str)
        else:
            st.metric(label="Expected Change", value=delta_str, delta=delta_str, delta_color="inverse")

    with result_col2:
        st.markdown("#### Buy vs Rent")
        st.markdown(f"<span class='{rec_badge}' style='font-size: 1.5rem'>{buy_rent_rec}</span>", unsafe_allow_html=True)
        st.markdown(f"Over a 5-year horizon, **{buy_rent_rec.lower()}** builds more wealth")

    with result_col3:
        st.markdown("#### Risk Level")
        risk_emoji = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}.get(risk_level, "⚪")
        st.markdown(f"<span style='font-size: 1.5rem'>{risk_emoji} {risk_level}</span>", unsafe_allow_html=True)
        if risk_level == "Low":
            st.markdown("Stable market with predictable trends")
        elif risk_level == "Medium":
            st.markdown("Moderate volatility expected")
        else:
            st.markdown("Higher uncertainty in projections")

    st.markdown("")

    # Investment Score
    st.markdown("### Investment Score")

    score_col1, score_col2 = st.columns([1, 2])

    with score_col1:
        # Gauge chart for score
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=investment_score,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Score", 'font': {'size': 16}},
            number={'font': {'size': 40, 'color': 'white'}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "white"},
                'bar': {'color': "white"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 40], 'color': '#ef4444'},
                    {'range': [40, 70], 'color': '#f59e0b'},
                    {'range': [70, 100], 'color': '#22c55e'}
                ],
            }
        ))
        fig_gauge.update_layout(height=200, paper_bgcolor={'color': 'white'}, font={'color': 'darkgray'})
        st.plotly_chart(fig_gauge, use_container_width=True)

    with score_col2:
        st.markdown(f"**Score: {investment_score}/100**")
        st.markdown(f"**Confidence: {confidence}**")
        st.markdown("")
        st.markdown("**Score Breakdown:**")
        st.markdown(f"- Growth Potential: Based on {adjusted_appreciation:+.1f}% predicted appreciation")
        st.markdown(f"- Buy/Rent Signal: {buy_rent_rec} recommendation from financial analysis")
        st.markdown(f"- Risk Assessment: {risk_level} risk based on market stability")
        st.markdown("")
        if investment_score >= 70:
            st.success("🎯 Strong investment opportunity")
        elif investment_score >= 50:
            st.info("📈 Moderate investment potential")
        else:
            st.warning("⚠️ Consider waiting or explore other options")

    st.markdown("")

    # =============================================================================
    # REASONING SECTION
    # =============================================================================

    st.markdown("### Why This Prediction?")

    st.markdown("**Analysis powered by:**")
    for reason in reasoning:
        if reason:
            st.markdown(f"• {reason}")

    st.markdown("")

    # =============================================================================
    # PRICE TREND CHART
    # =============================================================================

    st.markdown("### Price Forecast")

    # Build timeline data
    months = [0, 6, 12, 18]
    base_price = current_price

    # Get predictions for each horizon
    prices = [base_price]
    for horizon in [6, 12, 18]:
        pred = predictor.predict_price_change(
            current_price=current_price,
            city=city,
            property_type=property_type,
            horizon_months=horizon
        )
        prices.append(pred["predicted_price_6m"] if horizon == 6 else pred["predicted_price_6m"] * (1 + pred["predicted_change_pct"] / 100 * (horizon / 6)))

    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(
        x=["Now", "6 mo", "12 mo", "18 mo"],
        y=prices,
        mode="lines+markers",
        name="Predicted Price",
        line=dict(color="#667eea", width=3),
        marker=dict(size=8)
    ))

    fig_trend.update_layout(
        title=f"Price Forecast for {city} {property_type.title()}",
        xaxis_title="Time",
        yaxis_title="Price ($)",
        height=300,
        yaxis_tickformat="$,.0f"
    )

    st.plotly_chart(fig_trend, use_container_width=True)

    # =============================================================================
    # SCENARIO IMPACT
    # =============================================================================

    if rate_adjustment != 0:
        st.markdown("---")
        st.markdown("### Scenario Impact")

        impact_col1, impact_col2 = st.columns(2)

        with impact_col1:
            st.markdown("**Base Case**")
            st.markdown(f"Current rates → ${prediction['predicted_price_6m']:,.0f}")

        with impact_col2:
            direction = "increase" if rate_adjustment > 0 else "decrease"
            st.markdown(f"**With {abs(rate_adjustment)}% rate {direction}**")
            st.markdown(f"Adjusted → ${adjusted_price:,.0f}")

        difference = adjusted_price - prediction["predicted_price_6m"]
        st.markdown(f"**Impact:** ${difference:+,.0f} ({difference/prediction['predicted_price_6m']*100:+.1f}%)")

    st.markdown("")

    # =============================================================================
    # NEIGHBORHOOD COMPARISON
    # =============================================================================

    st.markdown("### Nearby Markets Comparison")
    st.markdown("See how similar areas are performing:")

    # Get comparison data
    try:
        analyzer = MarketAnalyzer(predictor)
        comparison_df = analyzer.compare_markets()

        # Filter to same property type
        type_comparison = comparison_df[comparison_df["property_type"] == property_type].head(5)

        if len(type_comparison) > 0:
            fig_comparison = px.bar(
                type_comparison,
                x="city",
                y="predicted_appreciation_6m",
                color="predicted_appreciation_6m",
                color_continuous_scale=["#ef4444", "#f59e0b", "#22c55e"],
                title=f"6-Month Appreciation by City ({property_type.title()})"
            )
            fig_comparison.update_layout(height=300, showlegend=False)
            st.plotly_chart(fig_comparison, use_container_width=True)
        else:
            st.info("Comparison data not available for this property type.")
    except Exception as e:
        st.info("Market comparison data not available.")


# =============================================================================
# FOOTER
# =============================================================================

st.markdown("---")
st.markdown("<div style='text-align: center; color: gray; font-size: 0.9rem'>⚠️ Not financial advice. For informational purposes only.</div>", unsafe_allow_html=True)
