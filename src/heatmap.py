"""
Market Heatmap Dashboard Components

Generates data for interactive visualizations:
- Price heatmaps by city/neighborhood
- ROI heatmaps
- Rental yield maps
- Market regime indicators
- Appreciation forecasts
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class MarketMetrics:
    """Market metrics for a city/property combination."""
    city: str
    property_type: str
    current_price: float
    predicted_price_6m: float
    predicted_price_12m: float
    appreciation_6m: float
    appreciation_12m: float
    rental_yield: float
    cap_rate: float
    price_to_rent_ratio: float
    market_regime: str  # hot, warm, cooling, cold
    risk_level: str  # low, medium, high
    investment_score: float  # 0-100
    buy_vs_rent_score: float  # 0-100


class MarketHeatmapGenerator:
    """Generate data for market heatmap visualizations."""

    def __init__(self, predictor=None, roi_calculator=None):
        self.predictor = predictor
        self.roi_calculator = roi_calculator

        # City coordinates for map visualization (approximate centroids)
        self.city_coordinates = {
            "Vancouver": {"lat": 49.2827, "lng": -123.1207},
            "Burnaby": {"lat": 49.2488, "lng": -122.9805},
            "Richmond": {"lat": 49.1666, "lng": -123.1336},
            "North Vancouver": {"lat": 49.3163, "lng": -123.0693},
            "Toronto": {"lat": 43.6532, "lng": -79.3832},
            "Calgary": {"lat": 51.0447, "lng": -114.0719},
        }

        # Property type markers
        self.property_markers = {
            "condo": {"icon": "circle", "size": 10},
            "townhouse": {"icon": "square", "size": 12},
            "detached": {"icon": "triangle", "size": 14},
            "multi_family": {"icon": "diamond", "size": 12},
        }

    def get_market_metrics(
        self,
        city: str,
        property_type: str,
        current_price: float,
        monthly_rent: float
    ) -> MarketMetrics:
        """Calculate comprehensive market metrics for a property."""

        from src.roi_calculator import ROICalculator, PropertyInputs

        # Get price prediction
        if self.predictor:
            pred_6m = self.predictor.predict_price_change(
                current_price=current_price,
                city=city,
                property_type=property_type,
                horizon_months=6
            )
            pred_12m = self.predictor.predict_price_change(
                current_price=current_price,
                city=city,
                property_type=property_type,
                horizon_months=12
            )
        else:
            # Fallback
            appreciation_rates = {
                ("Vancouver", "condo"): 0.018,
                ("Vancouver", "townhouse"): 0.038,
                ("Vancouver", "detached"): 0.025,
                ("Calgary", "condo"): 0.035,
                ("Calgary", "townhouse"): 0.055,
                ("Calgary", "detached"): 0.045,
            }
            rate = appreciation_rates.get((city, property_type), 0.03)
            pred_6m = {"predicted_price_6m": current_price * (1 + rate * 0.5), "predicted_change_pct": rate * 50}
            pred_12m = {"predicted_price_6m": current_price * (1 + rate), "predicted_change_pct": rate * 100}

        # Get ROI metrics
        if self.roi_calculator:
            roi_inputs = PropertyInputs(
                purchase_price=current_price,
                monthly_rent=monthly_rent,
                down_payment_pct=0.20
            )
            roi_metrics = self.roi_calculator.calculate_all_metrics(roi_inputs)
            rental_yield = roi_metrics["gross_yield"]
            cap_rate = roi_metrics["cap_rate"]
        else:
            rental_yield = (monthly_rent * 12 / current_price) * 100
            cap_rate = rental_yield * 0.7  # Rough estimate

        appreciation_6m = pred_6m.get("predicted_change_pct", 0)
        appreciation_12m = pred_12m.get("predicted_change_pct", 0)

        # Price to rent ratio
        price_to_rent = current_price / (monthly_rent * 12)

        # Market regime based on appreciation
        if appreciation_12m > 5:
            market_regime = "hot"
        elif appreciation_12m > 2:
            market_regime = "warm"
        elif appreciation_12m > 0:
            market_regime = "cooling"
        else:
            market_regime = "cold"

        # Risk level
        risk_score = 50
        if city in ["Vancouver", "North Vancouver"]:
            risk_score -= 15  # More stable
        elif city in ["Calgary"]:
            risk_score += 10  # More volatile

        if property_type == "condo":
            risk_score += 5  # More volatile
        elif property_type == "detached":
            risk_score -= 5  # More stable

        if risk_score < 30:
            risk_level = "low"
        elif risk_score < 60:
            risk_level = "medium"
        else:
            risk_level = "high"

        # Investment score (0-100)
        investment_score = self._calculate_investment_score(
            appreciation_12m, rental_yield, risk_level
        )

        # Buy vs Rent score
        buy_vs_rent_score = self._calculate_buy_vs_rent_score(
            price_to_rent, appreciation_12m, cap_rate
        )

        return MarketMetrics(
            city=city,
            property_type=property_type,
            current_price=current_price,
            predicted_price_6m=pred_6m["predicted_price_6m"],
            predicted_price_12m=pred_12m["predicted_price_6m"],
            appreciation_6m=appreciation_6m,
            appreciation_12m=appreciation_12m,
            rental_yield=rental_yield,
            cap_rate=cap_rate,
            price_to_rent_ratio=price_to_rent,
            market_regime=market_regime,
            risk_level=risk_level,
            investment_score=investment_score,
            buy_vs_rent_score=buy_vs_rent_score
        )

    def _calculate_investment_score(
        self,
        appreciation: float,
        rental_yield: float,
        risk_level: str
    ) -> float:
        """Calculate 0-100 investment score."""

        # Growth score (0-40)
        if appreciation >= 6:
            growth_score = 40
        elif appreciation >= 4:
            growth_score = 30
        elif appreciation >= 2:
            growth_score = 20
        elif appreciation >= 0:
            growth_score = 10
        else:
            growth_score = 5

        # Yield score (0-30)
        if rental_yield >= 5:
            yield_score = 30
        elif rental_yield >= 3.5:
            yield_score = 20
        elif rental_yield >= 2:
            yield_score = 10
        else:
            yield_score = 5

        # Risk score (0-30)
        risk_scores = {"low": 30, "medium": 20, "high": 10}
        risk_score = risk_scores.get(risk_level, 15)

        return min(100, growth_score + yield_score + risk_score)

    def _calculate_buy_vs_rent_score(
        self,
        price_to_rent: float,
        appreciation: float,
        cap_rate: float
    ) -> float:
        """Calculate 0-100 buy vs rent score (higher = better to buy)."""

        score = 50  # Neutral start

        # Price to rent adjustment
        if price_to_rent < 15:
            score += 20  # Very favorable to buy
        elif price_to_rent < 20:
            score += 10
        elif price_to_rent > 30:
            score -= 20  # Very unfavorable to buy
        elif price_to_rent > 25:
            score -= 10

        # Appreciation adjustment
        if appreciation > 5:
            score += 15
        elif appreciation > 2:
            score += 5
        elif appreciation < 0:
            score -= 10

        # Cap rate adjustment
        if cap_rate > 4:
            score += 15
        elif cap_rate > 2:
            score += 5

        return max(0, min(100, score))

    def generate_heatmap_data(
        self,
        properties: List[Dict]
    ) -> pd.DataFrame:
        """Generate heatmap data from property list."""

        results = []

        for prop in properties:
            metrics = self.get_market_metrics(
                city=prop["city"],
                property_type=prop["property_type"],
                current_price=prop["current_price"],
                monthly_rent=prop.get("monthly_rent", 2500)
            )

            # Get coordinates
            coords = self.city_coordinates.get(metrics.city, {"lat": 0, "lng": 0})

            # Add jitter for overlapping properties
            lat_jitter = np.random.uniform(-0.02, 0.02)
            lng_jitter = np.random.uniform(-0.02, 0.02)

            results.append({
                "city": metrics.city,
                "property_type": metrics.property_type,
                "latitude": coords["lat"] + lat_jitter,
                "longitude": coords["lng"] + lng_jitter,
                "current_price": metrics.current_price,
                "predicted_price_12m": metrics.predicted_price_12m,
                "appreciation_12m": metrics.appreciation_12m,
                "rental_yield": metrics.rental_yield,
                "cap_rate": metrics.cap_rate,
                "price_to_rent_ratio": metrics.price_to_rent_ratio,
                "market_regime": metrics.market_regime,
                "risk_level": metrics.risk_level,
                "investment_score": metrics.investment_score,
                "buy_vs_rent_score": metrics.buy_vs_rent_score,
                "marker_type": self.property_markers[metrics.property_type]["icon"],
                "marker_size": self.property_markers[metrics.property_type]["size"],
            })

        return pd.DataFrame(results)

    def generate_city_summary(self, heatmap_df: pd.DataFrame) -> pd.DataFrame:
        """Generate city-level summary statistics."""

        summary = heatmap_df.groupby("city").agg({
            "current_price": "mean",
            "predicted_price_12m": "mean",
            "appreciation_12m": "mean",
            "rental_yield": "mean",
            "investment_score": "mean",
            "buy_vs_rent_score": "mean",
        }).reset_index()

        summary.columns = [
            "City", "Avg Price", "Avg Predicted Price (12m)",
            "Avg Appreciation (%)", "Avg Rental Yield (%)",
            "Avg Investment Score", "Avg Buy/Rent Score"
        ]

        return summary

    def generate_property_type_summary(self, heatmap_df: pd.DataFrame) -> pd.DataFrame:
        """Generate property type summary statistics."""

        summary = heatmap_df.groupby("property_type").agg({
            "current_price": "mean",
            "appreciation_12m": "mean",
            "rental_yield": "mean",
            "investment_score": "mean",
        }).reset_index()

        summary.columns = [
            "Property Type", "Avg Price", "Avg Appreciation (%)",
            "Avg Rental Yield (%)", "Avg Investment Score"
        ]

        return summary

    def get_top_markets(self, heatmap_df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
        """Get top N markets by investment score."""

        top = heatmap_df.nlargest(n, "investment_score")
        return top[["city", "property_type", "current_price", "investment_score", "market_regime"]]

    def get_hot_markets(self, heatmap_df: pd.DataFrame) -> pd.DataFrame:
        """Get markets in 'hot' regime."""

        hot = heatmap_df[heatmap_df["market_regime"] == "hot"]
        return hot.sort_values("appreciation_12m", ascending=False)

    def get_undervalued_properties(self, heatmap_df: pd.DataFrame) -> pd.DataFrame:
        """Find potentially undervalued properties (high yield, low price)."""

        # Score based on rental yield and below-average price
        avg_price = heatmap_df["current_price"].mean()

        undervalued = heatmap_df[
            (heatmap_df["current_price"] < avg_price * 0.8) &
            (heatmap_df["rental_yield"] > heatmap_df["rental_yield"].median())
        ].copy()

        undervalued["undervalue_score"] = (
            (1 - undervalued["current_price"] / avg_price) * 50 +
            undervalued["rental_yield"] * 10
        )

        return undervalued.sort_values("undervalue_score", ascending=False)


def generate_sample_heatmap_data() -> pd.DataFrame:
    """Generate sample data for demo purposes."""

    sample_properties = [
        {"city": "Vancouver", "property_type": "condo", "current_price": 750000, "monthly_rent": 2600},
        {"city": "Vancouver", "property_type": "townhouse", "current_price": 950000, "monthly_rent": 3200},
        {"city": "Vancouver", "property_type": "detached", "current_price": 2000000, "monthly_rent": 5500},
        {"city": "Burnaby", "property_type": "condo", "current_price": 620000, "monthly_rent": 2200},
        {"city": "Burnaby", "property_type": "townhouse", "current_price": 850000, "monthly_rent": 2900},
        {"city": "Richmond", "property_type": "condo", "current_price": 580000, "monthly_rent": 2000},
        {"city": "Richmond", "property_type": "detached", "current_price": 1400000, "monthly_rent": 4000},
        {"city": "North Vancouver", "property_type": "townhouse", "current_price": 900000, "monthly_rent": 3000},
        {"city": "North Vancouver", "property_type": "detached", "current_price": 1700000, "monthly_rent": 4800},
        {"city": "Toronto", "property_type": "condo", "current_price": 700000, "monthly_rent": 2500},
        {"city": "Toronto", "property_type": "townhouse", "current_price": 900000, "monthly_rent": 3100},
        {"city": "Toronto", "property_type": "detached", "current_price": 1600000, "monthly_rent": 4500},
        {"city": "Calgary", "property_type": "condo", "current_price": 320000, "monthly_rent": 1600},
        {"city": "Calgary", "property_type": "townhouse", "current_price": 450000, "monthly_rent": 2000},
        {"city": "Calgary", "property_type": "detached", "current_price": 650000, "monthly_rent": 2600},
    ]

    generator = MarketHeatmapGenerator()
    return generator.generate_heatmap_data(sample_properties)


def main():
    print("=" * 70)
    print("MARKET HEATMAP DATA GENERATOR")
    print("=" * 70)

    # Generate sample data
    df = generate_sample_heatmap_data()

    print("\n--- SAMPLE HEATMAP DATA ---")
    print(df.head(10).to_string(index=False))

    generator = MarketHeatmapGenerator()

    print("\n--- CITY SUMMARY ---")
    city_summary = generator.generate_city_summary(df)
    print(city_summary.to_string(index=False))

    print("\n--- TOP MARKETS BY INVESTMENT SCORE ---")
    top_markets = generator.get_top_markets(df)
    print(top_markets.to_string(index=False))

    print("\n--- HOT MARKETS ---")
    hot_markets = generator.get_hot_markets(df)
    if len(hot_markets) > 0:
        print(hot_markets[["city", "property_type", "appreciation_12m"]].to_string(index=False))
    else:
        print("No hot markets currently")

    print("\n--- UNDervalued PROPERTIES ---")
    undervalued = generator.get_undervalued_properties(df)
    if len(undervalued) > 0:
        print(undervalued[["city", "property_type", "current_price", "rental_yield", "undervalue_score"]].to_string(index=False))
    else:
        print("No undervalued properties found")


if __name__ == "__main__":
    main()
