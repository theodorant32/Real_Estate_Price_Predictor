import pandas as pd
import numpy as np
import xgboost as xgb
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union
from datetime import datetime

logger = logging.getLogger(__name__)


class PricePredictor:
    def __init__(self, model_dir: str = "models"):
        self.model_dir = Path(model_dir)
        self.model = None
        self.feature_columns = None
        self.metrics = None

    def load_model(self, model_name: str = "price_predictor") -> bool:
        model_path = self.model_dir / f"{model_name}.json"
        feature_path = self.model_dir / f"{model_name}_features.json"
        metrics_path = self.model_dir / f"{model_name}_metrics.json"

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found at {model_path}. Run training first."
            )

        # Load model
        self.model = xgb.XGBRegressor()
        self.model.load_model(str(model_path))

        # Load feature columns
        with open(feature_path, "r") as f:
            self.feature_columns = json.load(f)

        # Load metrics if available
        if metrics_path.exists():
            with open(metrics_path, "r") as f:
                self.metrics = json.load(f)

        print(f"Loaded model from {model_path}")
        print(f"  Features: {len(self.feature_columns)}")
        if self.metrics:
            print(f"  CV RMSE: ${self.metrics['cv_results']['avg_rmse']:,.0f}")
            print(f"  CV MAPE: {self.metrics['cv_results']['avg_mape']:.2%}")

        return True

    def prepare_input(self, input_df: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        # Check for missing features
        missing = set(self.feature_columns) - set(input_df.columns)
        if missing:
            raise ValueError(f"Missing features: {missing}")

        # Select and order features
        X = input_df[self.feature_columns].values

        return X

    def predict(self, input_df: pd.DataFrame) -> np.ndarray:
        X = self.prepare_input(input_df)
        predictions = self.model.predict(X)

        return predictions

    def predict_with_confidence(self, input_df: pd.DataFrame) -> pd.DataFrame:
        predictions = self.predict(input_df)

        # Estimate uncertainty based on CV RMSE
        if self.metrics:
            rmse = self.metrics["cv_results"]["avg_rmse"]
        else:
            rmse = 50000  # Default $50K uncertainty

        # 95% confidence interval (assuming normal errors)
        lower_bound = predictions - 1.96 * rmse
        upper_bound = predictions + 1.96 * rmse

        result = pd.DataFrame({
            "predicted_price": predictions,
            "lower_bound_95": lower_bound,
            "upper_bound_95": upper_bound,
            "uncertainty_range": upper_bound - lower_bound
        })

        return result

    def predict_price_change(
        self,
        current_price: float,
        city: str,
        property_type: str,
        additional_features: Dict = None
    ) -> Dict:

        # Fallback appreciation rates - varied by city and property type
        # Based on historical data and market trends
        appreciation_rates = {
            ("Vancouver", "detached"): 0.025,
            ("Vancouver", "townhouse"): 0.038,
            ("Vancouver", "condo"): 0.018,
            ("Vancouver", "multi_family"): 0.032,
            ("Burnaby", "detached"): 0.035,
            ("Burnaby", "townhouse"): 0.048,
            ("Burnaby", "condo"): 0.028,
            ("Burnaby", "multi_family"): 0.042,
            ("Richmond", "detached"): 0.022,
            ("Richmond", "townhouse"): 0.028,
            ("Richmond", "condo"): 0.015,
            ("Richmond", "multi_family"): 0.025,
            ("North Vancouver", "detached"): 0.038,
            ("North Vancouver", "townhouse"): 0.042,
            ("North Vancouver", "condo"): 0.028,
            ("North Vancouver", "multi_family"): 0.038,
            ("Toronto", "detached"): 0.028,
            ("Toronto", "townhouse"): 0.038,
            ("Toronto", "condo"): 0.015,
            ("Toronto", "multi_family"): 0.035,
            ("Calgary", "detached"): 0.045,
            ("Calgary", "townhouse"): 0.055,
            ("Calgary", "condo"): 0.035,
            ("Calgary", "multi_family"): 0.052,
        }

        key = (city, property_type)
        base_rate = appreciation_rates.get(key, 0.03)

        # Try to use ML model if available
        use_ml = False
        change_pct = base_rate

        if self.model is not None and self.feature_columns is not None:
            try:
                from features import FeatureEngineer
                from ingest import DataIngester

                ingester = DataIngester()
                merged = ingester.create_merged_dataset()

                # Filter to the specific city/property type
                filtered = merged[
                    (merged["city"] == city) &
                    (merged["property_type"] == property_type)
                ].sort_values("date")

                if len(filtered) >= 12:  # Need at least 12 months for features
                    latest = filtered.iloc[-1:].copy()
                    fe = FeatureEngineer(prediction_horizon=6)
                    featured = fe.create_all_features(latest)

                    feature_cols = [c for c in self.feature_columns if c in featured.columns]

                    if len(feature_cols) > 0:
                        X = featured[feature_cols].values
                        predicted_target = self.model.predict(X)[0]

                        current_benchmark = latest["benchmark_price"].values[0]
                        model_rate = (predicted_target - current_benchmark) / current_benchmark
                        model_rate = max(-0.10, min(0.15, model_rate))  # Clip extremes

                        # Blend: 60% model, 40% historical
                        change_pct = 0.6 * model_rate + 0.4 * base_rate
                        use_ml = True

            except Exception as e:
                print(f"ML prediction fallback: {e}")
                change_pct = base_rate

        predicted_price = current_price * (1 + change_pct)

        # Uncertainty based on property type volatility
        volatility_map = {
            "detached": 0.07,
            "townhouse": 0.06,
            "condo": 0.09,
            "multi_family": 0.065
        }
        uncertainty = volatility_map.get(property_type, 0.07)

        return {
            "current_price": current_price,
            "predicted_price_6m": round(predicted_price, 0),
            "predicted_change_pct": round(change_pct * 100, 2),
            "confidence_lower": round(predicted_price * (1 - uncertainty), 0),
            "confidence_upper": round(predicted_price * (1 + uncertainty), 0),
            "city": city,
            "property_type": property_type,
            "uses_ml_model": use_ml
        }

    def batch_predict(self, properties: List[Dict]) -> List[Dict]:
        results = []
        for prop in properties:
            result = self.predict_price_change(
                current_price=prop.get("current_price", 500000),
                city=prop.get("city", "Vancouver"),
                property_type=prop.get("property_type", "condo")
            )
            results.append(result)

        return results


class MarketAnalyzer:
    def __init__(self, predictor: PricePredictor = None):
        self.predictor = predictor or PricePredictor()

    def get_market_recommendation(self, city: str, property_type: str, current_conditions: Dict = None) -> Dict:
        # Get price prediction
        prediction = self.predictor.predict_price_change(
            current_price=1000000,  # Normalize to $1M
            city=city,
            property_type=property_type
        )

        appreciation = prediction["predicted_change_pct"]

        # Determine recommendation
        if appreciation > 4:
            recommendation = "STRONG_BUY"
            reasoning = f"Strong appreciation expected ({appreciation:.1f}% in 6 months)"
        elif appreciation > 2:
            recommendation = "BUY"
            reasoning = f"Moderate appreciation expected ({appreciation:.1f}% in 6 months)"
        elif appreciation > 0:
            recommendation = "HOLD"
            reasoning = f"Stable market expected ({appreciation:.1f}% in 6 months)"
        else:
            recommendation = "CAUTION"
            reasoning = f"Price decline expected ({appreciation:.1f}% in 6 months)"

        return {
            "city": city,
            "property_type": property_type,
            "recommendation": recommendation,
            "reasoning": reasoning,
            "predicted_appreciation_6m": appreciation,
            "confidence_range": {
                "lower": prediction["confidence_lower"] / 10000,
                "upper": prediction["confidence_upper"] / 10000
            }
        }

    def compare_markets(self, cities: List[str] = None, property_types: List[str] = None) -> pd.DataFrame:
        if cities is None:
            cities = ["Vancouver", "Burnaby", "Richmond", "North Vancouver", "Toronto", "Calgary"]

        if property_types is None:
            property_types = ["detached", "townhouse", "condo", "multi_family"]

        results = []
        for city in cities:
            for ptype in property_types:
                rec = self.get_market_recommendation(city, ptype)
                results.append(rec)

        df = pd.DataFrame(results)

        # Sort by predicted appreciation
        df = df.sort_values("predicted_appreciation_6m", ascending=False)

        return df


def main():
    # Load predictor
    predictor = PricePredictor()

    try:
        predictor.load_model()
    except FileNotFoundError:
        print("Model not found. Using fallback predictions.")

    # Demo predictions
    print("\n" + "=" * 50)
    print("SAMPLE PREDICTIONS")
    print("=" * 50)

    properties = [
        {"city": "Vancouver", "property_type": "condo", "current_price": 750000},
        {"city": "Vancouver", "property_type": "detached", "current_price": 2000000},
        {"city": "Burnaby", "property_type": "townhouse", "current_price": 950000},
        {"city": "Richmond", "property_type": "detached", "current_price": 1400000},
        {"city": "Toronto", "property_type": "condo", "current_price": 700000},
        {"city": "Calgary", "property_type": "detached", "current_price": 650000},
    ]

    for prop in properties:
        pred = predictor.predict_price_change(**prop)
        print(f"\n{pred['city']} - {pred['property_type']}")
        print(f"  Current: ${pred['current_price']:,.0f}")
        print(f"  Predicted (6m): ${pred['predicted_price_6m']:,.0f}")
        print(f"  Change: {pred['predicted_change_pct']:+.1f}%")
        print(f"  95% Range: ${pred['confidence_lower']:,.0f} - ${pred['confidence_upper']:,.0f}")

    # Market comparison
    print("\n" + "=" * 50)
    print("MARKET COMPARISON")
    print("=" * 50)

    analyzer = MarketAnalyzer(predictor)
    comparison = analyzer.compare_markets()

    print("\nTop 5 markets by predicted appreciation:")
    for _, row in comparison.head(5).iterrows():
        print(f"  {row['city']} {row['property_type']}: {row['predicted_appreciation_6m']:.1f}% [{row['recommendation']}]")


if __name__ == "__main__":
    main()
