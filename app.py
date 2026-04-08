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


# Page config
st.set_page_config(
    page_title="Canadian Real Estate Decision Engine",
    page_icon="🏠",
    layout="wide"
)

@st.cache_resource
def load_model():
    # Use absolute path based on app.py location (works on Railway)
    model_dir = str(Path(__file__).parent / "models")
    predictor = PricePredictor(model_dir=model_dir)
    try:
        predictor.load_model()
        return predictor, True
    except FileNotFoundError:
        return predictor, False


@st.cache_data
def get_market_comparison():
    predictor, loaded = load_model()
    if loaded:
        analyzer = MarketAnalyzer(predictor)
        return analyzer.compare_markets()
    else:
        return None


# Initialize session state
if "predictor" not in st.session_state:
    st.session_state.predictor, st.session_state.model_loaded = load_model()


# =============================================================================
# SIDEBAR NAVIGATION
# =============================================================================

st.sidebar.title("🏠 Canadian Real Estate")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    ["Home", "Price Predictor", "Buy vs Rent", "Property Recommender", "Market Comparison"],
    index=0
)

st.sidebar.markdown("---")

# Model status
if st.session_state.model_loaded:
    st.sidebar.success("✅ Model loaded")
else:
    st.sidebar.warning("⚠️ Model not found - using fallback predictions")
    st.sidebar.markdown("Run `python src/train.py` to train the model")


# =============================================================================
# HOME PAGE
# =============================================================================

if page == "Home":
    st.title("🏠 Canadian Real Estate Decision Engine")

    st.markdown("""
    ## Make Data-Driven Real Estate Decisions

    This tool combines machine learning price predictions with Canadian-specific
    financial analysis to help you decide:

    1. **What will this property cost in 6 months?**
    2. **Which asset type is the better buy?**
    3. **Should I buy or rent?**

    ---
    """)

    # Quick stats
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            label="Cities Covered",
            value="6",
            help="Vancouver, Burnaby, Richmond, North Vancouver, Toronto, Calgary"
        )

    with col2:
        st.metric(
            label="Property Types",
            value="4",
            help="Detached, Townhouse, Condo, Multi-family"
        )

    with col3:
        st.metric(
            label="Prediction Horizon",
            value="6 months",
            help="ML model predicts prices 6 months ahead"
        )

    st.markdown("---")

    # Feature highlights
    st.markdown("### What You Can Do")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        #### 🔮 Price Predictor
        - Get 6-month price forecasts
        - By city and property type
        - With confidence intervals
        """)

    with col2:
        st.markdown("""
        #### 📊 Buy vs Rent
        - Canadian tax logic (BC PTT, CMHC)
        - Strata fees, maintenance
        - Break-even analysis
        """)

    with col3:
        st.markdown("""
        #### 📈 Market Comparison
        - Compare all cities side-by-side
        - See top markets by appreciation
        - Buy/Hold/Sell recommendations
        """)

    # Sample predictions
    st.markdown("---")
    st.markdown("### Sample Predictions (Today)")

    predictor = st.session_state.predictor

    sample_props = [
        {"city": "Vancouver", "property_type": "condo", "current_price": 750000},
        {"city": "Burnaby", "property_type": "townhouse", "current_price": 950000},
        {"city": "Calgary", "property_type": "detached", "current_price": 650000},
        {"city": "Toronto", "property_type": "multi_family", "current_price": 1200000},
    ]

    sample_data = []
    for prop in sample_props:
        pred = predictor.predict_price_change(**prop)
        sample_data.append({
            "City": prop["city"],
            "Type": prop["property_type"].capitalize(),
            "Current Price": f"${prop['current_price']:,}",
            "Predicted (6m)": f"${pred['predicted_price_6m']:,}",
            "Change": f"{pred['predicted_change_pct']:+.1f}%"
        })

    st.table(pd.DataFrame(sample_data))


# =============================================================================
# PRICE PREDICTOR PAGE
# =============================================================================

elif page == "Price Predictor":
    st.title("🔮 Price Predictor")

    st.markdown("""
    Get ML-powered price predictions for Canadian real estate.
    The model predicts prices 6 months into the future.
    """)

    # Input form
    col1, col2 = st.columns(2)

    with col1:
        city = st.selectbox(
            "City",
            ["Vancouver", "Burnaby", "Richmond", "North Vancouver", "Toronto", "Calgary"],
            index=0
        )

        property_type = st.selectbox(
            "Property Type",
            ["condo", "townhouse", "detached", "multi_family"],
            index=0
        )

    with col2:
        current_price = st.number_input(
            "Current Price ($)",
            min_value=100000,
            max_value=10000000,
            value=750000,
            step=50000
        )

        time_horizon = st.selectbox(
            "Prediction Horizon",
            ["6 months", "12 months", "18 months"],
            index=0
        )

    predictor = st.session_state.predictor

    horizon_map = {"6 months": 6, "12 months": 12, "18 months": 18}
    pred = predictor.predict_price_change(
        current_price=current_price,
        city=city,
        property_type=property_type,
        horizon_months=horizon_map.get(time_horizon, 6)
    )

    st.markdown("---")

    # Results
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            label="Current Price",
            value=f"${pred['current_price']:,}"
        )

    with col2:
        st.metric(
            label="Predicted Price (6m)",
            value=f"${pred['predicted_price_6m']:,}",
            delta=f"{pred['predicted_change_pct']:+.2f}%"
        )

    with col3:
        st.metric(
            label="Expected Change",
            value=f"${pred['predicted_price_6m'] - pred['current_price']:,}"
        )

    # Confidence interval
    st.markdown("### Confidence Interval")

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=["Current", "Predicted (6m)"],
        y=[pred["current_price"], pred["predicted_price_6m"]],
        mode="lines+markers",
        name="Price",
        line=dict(color="#1f77b4", width=3),
        marker=dict(size=10)
    ))

    fig.add_trace(go.Scatter(
        x=["Predicted (6m)"],
        y=[pred["confidence_upper"]],
        mode="markers",
        marker=dict(color="green", size=15, symbol="triangle-up"),
        name="Upper Bound"
    ))

    fig.add_trace(go.Scatter(
        x=["Predicted (6m)"],
        y=[pred["confidence_lower"]],
        mode="markers",
        marker=dict(color="red", size=15, symbol="triangle-down"),
        name="Lower Bound"
    ))

    fig.update_layout(
        title=f"Price Prediction: {city} {property_type.capitalize()}",
        yaxis_title="Price ($)",
        height=400,
        showlegend=True
    )

    st.plotly_chart(fig, use_container_width=True)

    # Details
    with st.expander("Prediction Details"):
        st.write(f"""
        - **City**: {city}
        - **Property Type**: {property_type}
        - **Current Price**: ${pred['current_price']:,}
        - **Predicted Price (6m)**: ${pred['predicted_price_6m']:,}
        - **Expected Change**: {pred['predicted_change_pct']:+.2f}%
        - **95% Confidence Range**: ${pred['confidence_lower']:,} - ${pred['confidence_upper']:,}
        """)


# =============================================================================
# BUY VS RENT PAGE
# =============================================================================

elif page == "Buy vs Rent":
    st.title("📊 Buy vs Rent Calculator")

    st.markdown("""
    Compare the financial outcomes of buying vs renting with Canadian-specific costs.
    Includes BC Property Transfer Tax, CMHC insurance, strata fees, and more.
    """)

    # Input sections
    st.markdown("### Purchase Details")

    col1, col2, col3 = st.columns(3)

    with col1:
        purchase_price = st.number_input(
            "Purchase Price ($)",
            min_value=200000,
            max_value=10000000,
            value=750000,
            step=50000
        )

        down_payment_pct = st.slider(
            "Down Payment (%)",
            min_value=5,
            max_value=50,
            value=20
        )

    with col2:
        monthly_rent = st.number_input(
            "Equivalent Monthly Rent ($)",
            min_value=500,
            max_value=20000,
            value=2600,
            step=100
        )

        property_type = st.selectbox(
            "Property Type",
            ["condo", "townhouse", "detached", "multi_family"],
            key="bvr_type"
        )

    with col3:
        time_horizon = st.slider(
            "Time Horizon (years)",
            min_value=1,
            max_value=30,
            value=5
        )

        city = st.selectbox(
            "City",
            ["Vancouver", "Burnaby", "Richmond", "North Vancouver", "Toronto", "Calgary"],
            key="bvr_city"
        )

    # Advanced options
    with st.expander("Advanced Options"):
        col1, col2 = st.columns(2)

        with col1:
            mortgage_rate = st.number_input(
                "Mortgage Rate (%)",
                min_value=1.0,
                max_value=15.0,
                value=5.0,
                step=0.25
            ) / 100

            appreciation_rate = st.number_input(
                "Expected Appreciation (%)",
                min_value=-5.0,
                max_value=15.0,
                value=3.0,
                step=0.5
            ) / 100

        with col2:
            strata_fee = st.number_input(
                "Monthly Strata Fee ($)",
                min_value=0,
                max_value=2000,
                value={"condo": 500, "townhouse": 350, "detached": 0}.get(property_type, 0),
                step=50
            )

            investment_return = st.number_input(
                "Investment Return Rate (%)",
                min_value=1.0,
                max_value=15.0,
                value=7.0,
                step=0.5
            ) / 100

    # Run analysis
    results = quick_analysis(
        purchase_price=purchase_price,
        monthly_rent=monthly_rent,
        city=city,
        property_type=property_type,
        time_horizon_years=time_horizon,
        appreciation_rate=appreciation_rate
    )

    st.markdown("---")

    # Recommendation banner
    rec = results["recommendation"]
    rec_color = {
        "STRONG_BUYING": "🟢",
        "MODERATE_BUYING": "🟡",
        "WEAK_BUYING": "🟡",
        "STRONG_RENTING": "🔴",
        "MODERATE_RENTING": "🟠",
        "WEAK_RENTING": "🟠"
    }.get(rec["recommendation"], "⚪")

    st.markdown(f"### {rec_color} Recommendation: {rec['recommendation'].replace('_', ' ')}")
    for reason in rec["reasoning"]:
        st.markdown(f"  - {reason}")

    # Comparison metrics
    st.markdown("### Financial Comparison")

    col1, col2, col3 = st.columns(3)

    comp = results["comparison"]

    with col1:
        st.metric(
            label="Net Worth (Buying)",
            value=f"${comp['net_worth']['buying']:,.0f}"
        )

    with col2:
        st.metric(
            label="Net Worth (Renting)",
            value=f"${comp['net_worth']['renting']:,.0f}"
        )

    with col3:
        delta = comp["net_worth"]["difference"]
        st.metric(
            label="Difference",
            value=f"${delta:,.0f}",
            delta=f"{'Buying' if delta > 0 else 'Renting'} advantage"
        )

    # Charts
    st.markdown("### Wealth Over Time")

    # Build wealth timeline
    wealth_data = []
    for year in range(1, time_horizon + 1):
        buying_equity = results["buying"]["annual_costs"][year-1]["equity"]
        renting_invest = results["renting"]["annual_costs"][year-1]["investment_balance"]
        wealth_data.append({
            "Year": year,
            "Buying": buying_equity,
            "Renting": renting_invest
        })

    wealth_df = pd.DataFrame(wealth_data)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=wealth_df["Year"],
        y=wealth_df["Buying"],
        mode="lines+markers",
        name="Buying (Equity)",
        line=dict(color="#1f77b4", width=3)
    ))
    fig.add_trace(go.Scatter(
        x=wealth_df["Year"],
        y=wealth_df["Renting"],
        mode="lines+markers",
        name="Renting (Investment)",
        line=dict(color="#ff7f0e", width=3)
    ))

    fig.update_layout(
        title="Wealth Accumulation: Buying vs Renting",
        xaxis_title="Year",
        yaxis_title="Net Worth ($)",
        height=400
    )

    st.plotly_chart(fig, use_container_width=True)

    # Cost breakdown
    st.markdown("### Cost Breakdown")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Buying Costs**")
        buying = results["buying"]
        st.write(f"""
        - Initial (Down + PTT + CMHC): ${buying['initial_costs']['total']:,.0f}
        - Total Mortgage Payments: ${buying['total_costs'] - buying['initial_costs']['total']:,.0f}
        - Property Tax (avg/year): ${buying['annual_costs'][0]['property_tax']:,.0f}
        - Strata Fees (avg/year): ${buying['annual_costs'][0]['strata_fees']:,.0f}
        - Maintenance (avg/year): ${buying['annual_costs'][0]['maintenance']:,.0f}
        """)

    with col2:
        st.markdown("**Renting Costs**")
        renting = results["renting"]
        st.write(f"""
        - Initial Investment: ${renting['initial_investment']:,.0f}
        - Total Rent Paid: ${renting['total_rent_paid']:,.0f}
        - Renters Insurance: ${renting['annual_costs'][0]['insurance']:,.0f}/year
        - Final Investment Balance: ${renting['final_investment_balance']:,.0f}
        """)

    # Break-even info
    if comp["break_even_year"]:
        st.success(f"💡 Break-even: Buying becomes more valuable than renting in **Year {comp['break_even_year']}**")
    else:
        st.info(f"💡 Over this {time_horizon}-year horizon, renting provides better wealth accumulation")


# =============================================================================
# PROPERTY RECOMMENDER PAGE
# =============================================================================

elif page == "Property Recommender":
    st.title("💰 Property Recommender")

    st.markdown("""
    Get personalized property recommendations based on your budget and income.
    The recommender considers affordability, predicted appreciation, and payment comfort.
    """)

    # Show current rates
    st.markdown("### 📊 Current Rates")
    try:
        from src.rates import get_current_rates
        rates = get_current_rates()
        mortgage = rates.get("sources", {}).get("mortgage_rates", {})
        derived = rates.get("derived", {})

        col_rates1, col_rates2, col_rates3 = st.columns(3)
        with col_rates1:
            rate_5yr = mortgage.get("rates_by_type", {}).get("fixed_5yr", 4.59)
            st.metric("5-Year Fixed Rate", f"{rate_5yr}%")
        with col_rates2:
            stress = derived.get("stress_test_rate", 5.25)
            st.metric("Stress Test Rate", f"{stress}%")
        with col_rates3:
            prime = rates.get("sources", {}).get("boc", {}).get("prime", 5.45)
            st.metric("Prime Rate", f"{prime}%")
    except Exception:
        st.info("Using default rates (5.25% stress test)")

    st.markdown("---")

    # Financial profile inputs
    st.markdown("### Your Financial Profile")

    col1, col2, col3 = st.columns(3)

    with col1:
        annual_income = st.number_input(
            "Annual Household Income ($)",
            min_value=30000,
            max_value=1000000,
            value=100000,
            step=10000
        )

        other_debt = st.number_input(
            "Other Monthly Debt ($)",
            min_value=0,
            max_value=10000,
            value=0,
            step=100,
            help="Car payments, student loans, credit cards, etc."
        )

        # First-time buyer status
        is_ftb = st.checkbox(
            "First-Time Home Buyer",
            value=True,
            help="Never owned a home in the past 5 years. Qualifies for PTT exemption (up to $835K) and FHSA benefits."
        )

    with col2:
        down_payment = st.number_input(
            "Available Down Payment ($)",
            min_value=20000,
            max_value=2000000,
            value=150000,
            step=10000
        )

        fhsa_balance = st.number_input(
            "FHSA Balance ($)",
            min_value=0,
            max_value=80000,
            value=0,
            step=5000,
            help="First Home Savings Account balance. Can be used tax-free for down payment. Max $8K/year, $40K lifetime."
        )

        risk_tolerance = st.selectbox(
            "Risk Tolerance",
            ["conservative", "moderate", "aggressive"],
            index=1,
            help="Conservative: prioritize affordability. Aggressive: prioritize appreciation."
        )

    with col3:
        time_horizon = st.slider(
            "Time Horizon (years)",
            min_value=1,
            max_value=30,
            value=5
        )

        property_types = st.multiselect(
            "Property Types to Consider",
            ["condo", "townhouse", "detached", "multi_family"],
            default=["condo", "townhouse", "detached", "multi_family"]
        )

    # Run recommendations
    if st.button("Get Recommendations", type="primary"):
        # Get recommendations with first-time buyer status
        from src.recommender import PropertyRecommender, BuyerProfile

        profile = BuyerProfile(
            annual_income=annual_income,
            available_down_payment=down_payment,
            other_monthly_debt=other_debt,
            risk_tolerance=risk_tolerance,
            is_first_time_buyer=is_ftb,
            fhsa_balance=fhsa_balance,
            property_types=property_types
        )

        recommender = PropertyRecommender()
        affordability = recommender.affordability_calc.calculate_max_purchase_price(profile)
        recommendations = recommender.get_top_recommendations(profile, n=10)
        type_comparison = recommender.compare_property_types(profile)

        results = {
            "profile": {
                "annual_income": annual_income,
                "down_payment": down_payment,
                "other_monthly_debt": other_debt,
                "risk_tolerance": risk_tolerance,
                "is_first_time_buyer": is_ftb,
                "fhsa_balance": fhsa_balance
            },
            "affordability": affordability,
            "recommendations": recommendations,
            "property_type_summary": type_comparison.to_dict() if type_comparison is not None else None
        }

        # Display affordability
        st.markdown("### 📊 Affordability Analysis")

        aff = results["affordability"]

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                label="Max Purchase Price",
                value=f"${aff['max_purchase_price']:,.0f}"
            )

        with col2:
            st.metric(
                label="Max Mortgage",
                value=f"${aff['max_mortgage_amount']:,.0f}"
            )

        with col3:
            st.metric(
                label="Required Down Payment",
                value=f"${aff['required_down_payment']:,.0f}"
            )

        with col4:
            st.metric(
                label="Est. Monthly Payment",
                value=f"${aff['estimated_monthly_payment']:,.0f}"
            )

        # Constraints breakdown
        with st.expander("View Affordability Constraints"):
            st.write("""
            **CMHC Guidelines Used:**
            - **GDS (Gross Debt Service):** Housing costs ≤ 32% of gross income
            - **TDS (Total Debt Service):** Total debt ≤ 40% of gross income
            - **Stress Test Rate:** 5.25% (OSFI requirement)
            """)

            constraints = aff["constraints"]
            st.write(f"""
            - Max price by GDS constraint: ${constraints['gds_max_price']:,.0f}
            - Max price by TDS constraint: ${constraints['tds_max_price']:,.0f}
            - Max price by down payment: ${constraints['down_payment_max_price']:,.0f}
            """)

        # Top recommendations
        st.markdown("### 🏆 Top Recommendations")

        recommendations = results["recommendations"]

        if len(recommendations) == 0:
            st.warning("No properties found within your criteria. Try increasing your budget or expanding property types.")
        else:
            # Display as cards
            for i, rec in enumerate(recommendations, 1):
                strength_colors = {
                    "Excellent Choice": "🟢",
                    "Great Option": "🟢",
                    "Worth Considering": "🟡",
                    "Think Twice": "🟠"
                }

                with st.container():
                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        st.markdown(f"**{i}. {rec['city']}**")
                        st.markdown(f"{rec['property_type'].capitalize()}")

                    with col2:
                        st.metric("Price", f"${rec['price']:,.0f}")

                    with col3:
                        st.metric("Monthly", f"${rec['total_monthly']:,.0f}")
                        st.metric("Appreciation", f"{rec['predicted_appreciation']}%")

                    with col4:
                        st.markdown(f"{strength_colors.get(rec['strength'], '')} **{rec['strength']}**")

                    st.markdown(f"*{rec['reasoning']}*")
                    st.divider()

            # Property type summary
            st.markdown("### 📈 Property Type Comparison")

            type_summary = results["property_type_summary"]
            if type_summary:
                type_df = pd.DataFrame(type_summary)
                if len(type_df) > 0:
                    type_df.columns = ["Property Type", "Avg Price", "Avg Monthly Payment",
                                       "Avg Total Monthly", "Avg Appreciation (%)",
                                       "Affordable Options", "Stretch Options"]

                    st.dataframe(
                        type_df.style.format({
                            "Avg Price": "${:,.0f}",
                            "Avg Monthly Payment": "${:,.0f}",
                            "Avg Total Monthly": "${:,.0f}",
                            "Avg Appreciation (%)": "{:.1f}%"
                        }),
                        hide_index=True,
                        use_container_width=True
                    )

    # Quick preset examples
    st.markdown("---")
    st.markdown("### Quick Examples")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("First-Time Buyer\n$75K income, $80K down", use_container_width=True):
            st.session_state.example_ftb = True

    with col2:
        if st.button("Young Family\n$150K income, $200K down", use_container_width=True):
            st.session_state.example_family = True

    with col3:
        if st.button("Dual Income\n$250K income, $500K down", use_container_width=True):
            st.session_state.example_dual = True

    if st.session_state.get("example_ftb"):
        annual_income = 75000
        down_payment = 80000
        st.session_state.example_ftb = False
        st.rerun()
    elif st.session_state.get("example_family"):
        annual_income = 150000
        down_payment = 200000
        st.session_state.example_family = False
        st.rerun()
    elif st.session_state.get("example_dual"):
        annual_income = 250000
        down_payment = 500000
        st.session_state.example_dual = False
        st.rerun()


# =============================================================================
# MARKET COMPARISON PAGE
# =============================================================================

elif page == "Market Comparison":
    st.title("📈 Market Comparison")

    st.markdown("""
    Compare real estate markets across cities and property types.
    See which markets are predicted to outperform.
    """)

    # Get comparison data
    comparison_df = get_market_comparison()

    if comparison_df is not None:
        # Display top markets
        st.markdown("### Top Markets by Predicted Appreciation")

        top_markets = comparison_df.head(10)[["city", "property_type", "predicted_appreciation_6m", "recommendation"]]
        top_markets.columns = ["City", "Property Type", "6-Month Appreciation (%)", "Recommendation"]

        st.dataframe(
            top_markets.style.format({"6-Month Appreciation (%)": "{:.1f}%"})
                         .background_gradient(subset=["6-Month Appreciation (%)"], cmap="RdYlGn"),
            hide_index=True
        )

        # Visualization
        st.markdown("### Appreciation by City and Property Type")

        fig = px.bar(
            comparison_df,
            x="city",
            y="predicted_appreciation_6m",
            color="property_type",
            barmode="group",
            title="Predicted 6-Month Appreciation by City and Property Type",
            labels={"predicted_appreciation_6m": "Appreciation (%)", "city": "City"}
        )

        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

        # Recommendation breakdown
        st.markdown("### Recommendation Breakdown")

        rec_counts = comparison_df["recommendation"].value_counts()

        fig = px.pie(
            values=rec_counts.values,
            names=rec_counts.index,
            title="Market Recommendations Distribution"
        )

        st.plotly_chart(fig, use_container_width=True)

        # Full data
        with st.expander("View Full Data"):
            st.dataframe(comparison_df)

    else:
        st.warning("""
        Market comparison data not available. This feature requires a trained model.

        Run the following to train the model:
        ```
        python src/train.py
        ```
        """)


# =============================================================================
# FOOTER
# =============================================================================

st.sidebar.markdown("---")
st.sidebar.markdown("⚠️ Not financial advice. For informational purposes only.")
