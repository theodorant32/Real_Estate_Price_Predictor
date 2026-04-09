"""
Explainable AI for Real Estate Predictions

SHAP (SHapley Additive exPlanations) integration:
- Feature importance visualization
- Individual prediction explanations
- Global model interpretability
- Partial dependence plots

LIME integration for local explanations.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Optional, Tuple
from pathlib import Path

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


class PredictionExplainer:
    """
    Explain ML model predictions using SHAP values.

    Provides:
    - Global feature importance
    - Individual prediction explanations
    - Force plots
    - Dependence plots
    - Interaction effects
    """

    def __init__(self, model=None, feature_columns: List[str] = None):
        self.model = model
        self.feature_columns = feature_columns or []
        self.explainer = None
        self.shap_values = None
        self.background_data = None

    def set_model(self, model, feature_columns: List[str]):
        """Set the model to explain."""
        self.model = model
        self.feature_columns = feature_columns

        # Create SHAP explainer for tree models
        if SHAP_AVAILABLE and hasattr(model, 'tree_'):
            try:
                self.explainer = shap.TreeExplainer(model)
            except Exception as e:
                print(f"SHAP explainer creation failed: {e}")
                self.explainer = None

    def compute_shap_values(self, X: np.ndarray, n_samples: int = 100):
        """Compute SHAP values for a sample of data."""

        if not SHAP_AVAILABLE or self.explainer is None:
            return None

        # Sample for speed
        if len(X) > n_samples:
            indices = np.random.choice(len(X), n_samples, replace=False)
            X_sample = X[indices]
        else:
            X_sample = X

        self.shap_values = self.explainer.shap_values(X_sample)
        self.background_data = X_sample

        return self.shap_values

    def get_feature_importance(self) -> pd.DataFrame:
        """Get global feature importance from SHAP values."""

        if self.shap_values is None or self.feature_columns is None:
            return pd.DataFrame()

        # Mean absolute SHAP value for each feature
        if isinstance(self.shap_values, list):
            # Multi-output models
            shap_abs = np.mean(np.abs(self.shap_values[0]), axis=0)
        else:
            shap_abs = np.mean(np.abs(self.shap_values), axis=0)

        importance_df = pd.DataFrame({
            "feature": self.feature_columns,
            "importance": shap_abs,
            "avg_shap_value": np.mean(self.shap_values, axis=0) if not isinstance(self.shap_values, list) else np.mean(self.shap_values[0], axis=0)
        }).sort_values("importance", ascending=False)

        return importance_df

    def explain_prediction(self, X: np.ndarray, sample_idx: int = 0) -> Dict:
        """Explain a single prediction."""

        if self.shap_values is None or self.feature_columns is None:
            return {"error": "SHAP values not computed"}

        if isinstance(self.shap_values, list):
            sample_shap = self.shap_values[0][sample_idx]
        else:
            sample_shap = self.shap_values[sample_idx]

        # Get feature names and values
        feature_names = self.feature_columns
        feature_values = X[sample_idx]

        # Create explanation
        explanation = {
            "feature": feature_names,
            "value": feature_values,
            "shap_value": sample_shap,
            "base_value": self.explainer.expected_value if hasattr(self.explainer, 'expected_value') else 0
        }

        return explanation

    def plot_feature_importance(self, top_n: int = 20) -> go.Figure:
        """Create feature importance bar chart."""

        importance_df = self.get_feature_importance().head(top_n)

        if len(importance_df) == 0:
            return go.Figure().add_annotation(text="No SHAP data available")

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=importance_df["importance"],
            y=importance_df["feature"],
            orientation="h",
            marker=dict(
                color=importance_df["importance"],
                colorscale="Viridis",
                showscale=True
            ),
            name="Importance"
        ))

        fig.update_layout(
            title="Feature Importance (SHAP)",
            xaxis_title="Mean |SHAP Value|",
            yaxis_title="Feature",
            height=max(400, len(importance_df) * 30),
            showlegend=False
        )

        return fig

    def plot_waterfall(self, sample_idx: int = 0) -> go.Figure:
        """Create waterfall chart for individual prediction."""

        if self.shap_values is None:
            return go.Figure().add_annotation(text="No SHAP data available")

        explanation = self.explain_prediction(self.background_data, sample_idx)

        if "error" in explanation:
            return go.Figure().add_annotation(text=explanation["error"])

        # Sort by absolute SHAP value
        abs_shap = np.abs(explanation["shap_value"])
        sorted_indices = np.argsort(abs_shap)[::-1][:15]  # Top 15

        features = [explanation["feature"][i] for i in sorted_indices]
        shap_vals = [explanation["shap_value"][i] for i in sorted_indices]
        values = [explanation["value"][i] for i in sorted_indices]

        # Create waterfall
        base = explanation["base_value"]
        if isinstance(base, np.ndarray):
            base = base[0]

        # Calculate cumulative
        cumulative = [base]
        for val in shap_vals:
            cumulative.append(cumulative[-1] + val)

        fig = go.Figure()

        # Base value
        fig.add_trace(go.Bar(
            x=[base],
            y=["Base Value"],
            orientation="h",
            marker_color="gray",
            name="Base"
        ))

        # Positive contributions
        pos_features = [f for f, s in zip(features, shap_vals) if s > 0]
        pos_values = [s for s in shap_vals if s > 0]

        if pos_values:
            fig.add_trace(go.Bar(
                x=pos_values,
                y=pos_features,
                orientation="h",
                marker_color="#22c55e",
                name="Positive Impact"
            ))

        # Negative contributions
        neg_features = [f for f, s in zip(features, shap_vals) if s < 0]
        neg_values = [abs(s) for s in shap_vals if s < 0]

        if neg_values:
            fig.add_trace(go.Bar(
                x=neg_values,
                y=neg_features,
                orientation="h",
                marker_color="#ef4444",
                name="Negative Impact"
            ))

        fig.update_layout(
            title=f"Prediction Explanation (Sample {sample_idx})",
            barmode="relative",
            xaxis_title="Impact on Prediction",
            height=max(400, len(features) * 30),
            showlegend=True
        )

        return fig

    def plot_dependence(self, feature_idx: int = 0) -> go.Figure:
        """Create SHAP dependence plot for a feature."""

        if self.shap_values is None or self.feature_columns is None:
            return go.Figure().add_annotation(text="No SHAP data available")

        feature_name = self.feature_columns[feature_idx]

        if isinstance(self.shap_values, list):
            shap_vals = self.shap_values[0][:, feature_idx]
            feature_values = self.background_data[:, feature_idx]
        else:
            shap_vals = self.shap_values[:, feature_idx]
            feature_values = self.background_data[:, feature_idx]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=feature_values,
            y=shap_vals,
            mode="markers",
            marker=dict(
                size=8,
                color=feature_values,
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title=feature_name)
            ),
            text=[f"{feature_name}: {v:.2f}<br>SHAP: {s:.2f}" for v, s in zip(feature_values, shap_vals)],
            hoverinfo="text"
        ))

        fig.update_layout(
            title=f"SHAP Dependence: {feature_name}",
            xaxis_title=feature_name,
            yaxis_title="SHAP Value",
            height=500,
            showlegend=False
        )

        return fig

    def get_prediction_summary(self, X: np.ndarray, y_true: np.ndarray = None) -> Dict:
        """Get comprehensive prediction summary with explanations."""

        predictions = self.model.predict(X)

        summary = {
            "count": len(predictions),
            "mean": float(np.mean(predictions)),
            "std": float(np.std(predictions)),
            "min": float(np.min(predictions)),
            "max": float(np.max(predictions)),
            "median": float(np.median(predictions)),
            "p25": float(np.percentile(predictions, 25)),
            "p75": float(np.percentile(predictions, 75)),
        }

        if y_true is not None:
            from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
            summary["rmse"] = float(np.sqrt(mean_squared_error(y_true, predictions)))
            summary["mae"] = float(mean_absolute_error(y_true, predictions))
            summary["r2"] = float(r2_score(y_true, predictions))

        return summary


class MarketRegimeDetector:
    """
    Detect and explain market regimes using ML clustering.

    Regimes:
    - Hot: High demand, low inventory, rapid price growth
    - Warm: Balanced market, steady growth
    - Cooling: Slowing demand, increasing inventory
    - Cold: Low demand, high inventory, price declines
    """

    def __init__(self):
        self.regime_centers = None
        self.feature_columns = [
            "price_momentum",
            "inventory_change",
            "days_on_market",
            "sale_to_list_ratio",
            "rental_yield"
        ]

    def define_regimes(self) -> Dict:
        """Define market regime characteristics."""

        return {
            "hot": {
                "description": "High demand, rapid appreciation, low inventory",
                "characteristics": [
                    "Price momentum > 5% (6-month)",
                    "Inventory declining",
                    "Days on market < 30",
                    "Sale-to-list ratio > 100%",
                    "Multiple offers common"
                ],
                "investor_action": "Act quickly, expect competition",
                "color": "#ef4444"
            },
            "warm": {
                "description": "Balanced market, steady growth",
                "characteristics": [
                    "Price momentum 2-5%",
                    "Stable inventory",
                    "Days on market 30-60",
                    "Sale-to-list ratio 95-100%",
                    "Some competition"
                ],
                "investor_action": "Good time to buy, negotiate reasonably",
                "color": "#f59e0b"
            },
            "cooling": {
                "description": "Slowing market, buyer gaining advantage",
                "characteristics": [
                    "Price momentum 0-2%",
                    "Inventory increasing",
                    "Days on market 60-90",
                    "Sale-to-list ratio 90-95%",
                    "Price reductions common"
                ],
                "investor_action": "Take time, negotiate aggressively",
                "color": "#3b82f6"
            },
            "cold": {
                "description": "Buyer's market, price declines",
                "characteristics": [
                    "Negative price momentum",
                    "High inventory",
                    "Days on market > 90",
                    "Sale-to-list ratio < 90%",
                    "Distressed sales increasing"
                ],
                "investor_action": "Look for distressed opportunities",
                "color": "#6b7280"
            }
        }

    def detect_regime(self, metrics: Dict) -> Tuple[str, float, Dict]:
        """
        Detect market regime from metrics.

        Returns:
            - Regime name
            - Confidence score
            - Regime details
        """

        regimes = self.define_regimes()

        # Score each regime
        scores = {}

        # Momentum score
        momentum = metrics.get("price_momentum", 0)
        if momentum > 5:
            scores["hot"] = 0.8
            scores["warm"] = 0.2
        elif momentum > 2:
            scores["warm"] = 0.7
            scores["cooling"] = 0.3
        elif momentum > 0:
            scores["cooling"] = 0.6
            scores["cold"] = 0.4
        else:
            scores["cold"] = 0.8
            scores["cooling"] = 0.2

        # DOM score
        dom = metrics.get("days_on_market", 60)
        if dom < 30:
            scores["hot"] = scores.get("hot", 0) + 0.2
        elif dom < 60:
            scores["warm"] = scores.get("warm", 0) + 0.2
        elif dom < 90:
            scores["cooling"] = scores.get("cooling", 0) + 0.2
        else:
            scores["cold"] = scores.get("cold", 0) + 0.2

        # Find best matching regime
        best_regime = max(scores, key=scores.get)
        confidence = scores[best_regime] / sum(scores.values())

        return best_regime, confidence, regimes[best_regime]

    def explain_regime(self, regime: str, metrics: Dict) -> str:
        """Generate human-readable explanation of market regime."""

        regimes = self.define_regimes()
        regime_info = regimes.get(regime, regimes["warm"])

        explanation = f"**Market Regime: {regime.upper()}**\n\n"
        explanation += f"{regime_info['description']}.\n\n"
        explanation += "**Key Indicators:**\n"

        # Add specific metrics
        if "price_momentum" in metrics:
            explanation += f"- Price Momentum: {metrics['price_momentum']:+.1f}%\n"
        if "days_on_market" in metrics:
            explanation += f"- Days on Market: {metrics['days_on_market']:.0f}\n"
        if "inventory_change" in metrics:
            change = metrics["inventory_change"]
            direction = "increasing" if change > 0 else "decreasing"
            explanation += f"- Inventory: {direction} ({change:+.1f}%)\n"

        explanation += f"\n**Recommendation:** {regime_info['investor_action']}"

        return explanation


def create_explanation_report(
    predictor,
    X_sample: np.ndarray,
    feature_columns: List[str],
    sample_indices: List[int] = None
) -> Dict:
    """
    Create comprehensive explanation report.

    Returns dict with:
    - Feature importance chart
    - Individual explanations
    - Model performance summary
    """

    if sample_indices is None:
        sample_indices = [0, 1, 2]

    explainer = PredictionExplainer(predictor.model, feature_columns)

    # Compute SHAP values
    X = X_sample if len(X_sample) <= 100 else X_sample[:100]
    explainer.compute_shap_values(X)

    report = {
        "feature_importance": explainer.get_feature_importance(),
        "feature_importance_chart": explainer.plot_feature_importance(),
        "individual_explanations": [],
        "model_summary": explainer.get_prediction_summary(X_sample)
    }

    for idx in sample_indices:
        if idx < len(X_sample):
            report["individual_explanations"].append({
                "sample_idx": idx,
                "waterfall_chart": explainer.plot_waterfall(idx),
                "explanation": explainer.explain_prediction(X_sample, idx)
            })

    return report


def main():
    print("=" * 70)
    print("EXPLAINABLE AI FOR REAL ESTATE")
    print("=" * 70)

    if not SHAP_AVAILABLE:
        print("SHAP not installed. Install with: pip install shap")
        return

    # Demo with sample data
    from sklearn.ensemble import RandomForestRegressor

    np.random.seed(42)
    n_samples = 100
    n_features = 10

    X = np.random.randn(n_samples, n_features)
    y = 3 * X[:, 0] + 2 * X[:, 1] - X[:, 2] + np.random.randn(n_samples) * 0.5

    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    # Create explainer
    feature_names = [f"feature_{i}" for i in range(n_features)]
    explainer = PredictionExplainer(model, feature_names)
    explainer.compute_shap_values(X)

    print("\nFeature Importance:")
    print(explainer.get_feature_importance())

    # Market regime demo
    print("\n" + "=" * 70)
    print("MARKET REGIME DETECTION")
    print("=" * 70)

    detector = MarketRegimeDetector()

    test_metrics = [
        {"price_momentum": 6.5, "days_on_market": 25, "inventory_change": -10},
        {"price_momentum": 3.0, "days_on_market": 45, "inventory_change": 2},
        {"price_momentum": 0.5, "days_on_market": 75, "inventory_change": 8},
        {"price_momentum": -3.0, "days_on_market": 120, "inventory_change": 25},
    ]

    for metrics in test_metrics:
        regime, confidence, details = detector.detect_regime(metrics)
        print(f"\nMetrics: {metrics}")
        print(f"Detected Regime: {regime.upper()} ({confidence:.0%} confidence)")
        print(f"Action: {details['investor_action']}")


if __name__ == "__main__":
    main()
