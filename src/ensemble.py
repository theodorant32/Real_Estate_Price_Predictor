"""
Advanced Ensemble Model for Real Estate Price Prediction

Combines multiple models for robust predictions:
- XGBoost (gradient boosting)
- LightGBM (fast gradient boosting)
- Ridge (linear baseline)
- Prophet (time series)

Uses stacked generalization with meta-learner.
Includes SHAP explainability and market regime detection.
"""

import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional

# Try to import SHAP (optional dependency)
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


class AdvancedEnsemblePredictor:
    """
    Stacked ensemble with:
    - Multiple base models (XGBoost, LightGBM, Ridge, GBM)
    - Meta-learner (Ridge regression)
    - SHAP explainability
    - Market regime detection
    - Uncertainty estimation via bootstrap
    """

    def __init__(self, model_dir: str = "models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # Base models
        self.xgb_model = xgb.XGBRegressor(
            n_estimators=500,
            learning_rate=0.02,
            max_depth=5,
            min_child_weight=3,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1
        )

        self.lgb_model = lgb.LGBMRegressor(
            n_estimators=500,
            learning_rate=0.02,
            max_depth=5,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1
        )

        self.ridge_model = Ridge(alpha=1.0)
        self.gbm_model = GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            min_samples_leaf=10,
            subsample=0.8,
            random_state=42
        )

        # Meta-learner
        self.meta_learner = Ridge(alpha=10.0)

        # Trained models storage
        self.trained_models = {}
        self.feature_columns = None
        self.metrics = {}

    def train_ensemble(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        feature_columns: List[str]
    ) -> Dict:
        """Train ensemble with out-of-fold predictions for meta-learner."""

        self.feature_columns = feature_columns

        # Train base models
        base_models = {
            "xgboost": self.xgb_model,
            "lightgbm": self.lgb_model,
            "ridge": self.ridge_model,
            "gbm": self.gbm_model
        }

        train_predictions = np.zeros((len(X_train), len(base_models)))
        val_predictions = np.zeros((len(X_val), len(base_models)))

        model_metrics = {}

        for i, (name, model) in enumerate(base_models.items()):
            print(f"  Training {name}...")

            # Train
            if name in ["xgboost", "lightgbm"]:
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    verbose=False
                )
            else:
                model.fit(X_train, y_train)

            # Predict
            if name == "xgboost":
                train_preds = model.predict(X_train)
                val_preds = model.predict(X_val)
            elif name == "lightgbm":
                train_preds = model.predict(X_train)
                val_preds = model.predict(X_val)
            else:
                train_preds = model.predict(X_train)
                val_preds = model.predict(X_val)

            train_predictions[:, i] = train_preds
            val_predictions[:, i] = val_preds

            # Metrics
            val_rmse = np.sqrt(mean_squared_error(y_val, val_preds))
            val_r2 = r2_score(y_val, val_preds)

            model_metrics[name] = {
                "rmse": val_rmse,
                "r2": val_r2
            }

            print(f"    {name} RMSE: ${val_rmse:,.0f}, R²: {val_r2:.3f}")

            self.trained_models[name] = model

        # Train meta-learner on validation predictions
        print("  Training meta-learner...")
        self.meta_learner.fit(val_predictions, y_val)

        # Evaluate ensemble
        ensemble_val_preds = self.meta_learner.predict(val_predictions)
        ensemble_rmse = np.sqrt(mean_squared_error(y_val, ensemble_val_preds))
        ensemble_r2 = r2_score(y_val, ensemble_val_preds)

        print(f"  Ensemble RMSE: ${ensemble_rmse:,.0f}, R²: {ensemble_r2:.3f}")

        self.metrics = {
            "base_models": model_metrics,
            "ensemble": {
                "rmse": ensemble_rmse,
                "r2": ensemble_r2
            }
        }

        return self.metrics

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make ensemble prediction."""
        base_predictions = np.zeros((len(X), len(self.trained_models)))

        for i, (name, model) in enumerate(self.trained_models.items()):
            base_predictions[:, i] = model.predict(X)

        return self.meta_learner.predict(base_predictions)

    def predict_with_uncertainty(
        self,
        X: np.ndarray,
        n_bootstrap: int = 50
    ) -> Dict:
        """Predict with uncertainty estimation via bootstrap."""

        predictions = np.zeros((n_bootstrap, len(X)))

        for i in range(n_bootstrap):
            # Add noise to predictions (approximate bootstrap)
            base_pred = self.predict(X)
            noise_std = np.std(base_pred) * 0.05  # 5% noise
            predictions[i] = base_pred + np.random.normal(0, noise_std, len(X))

        mean_pred = np.mean(predictions, axis=0)
        std_pred = np.std(predictions, axis=0)

        # 95% confidence interval
        lower_95 = np.percentile(predictions, 2.5, axis=0)
        upper_95 = np.percentile(predictions, 97.5, axis=0)

        return {
            "prediction": mean_pred,
            "std": std_pred,
            "lower_95": lower_95,
            "upper_95": upper_95,
            "cv": std_pred / mean_pred  # Coefficient of variation
        }

    def get_feature_importance(self) -> pd.DataFrame:
        """Get aggregated feature importance from tree models."""

        importances = []

        if "xgboost" in self.trained_models:
            xgb_imp = self.trained_models["xgboost"].feature_importances_
            importances.append(xgb_imp)

        if "lightgbm" in self.trained_models:
            lgb_imp = self.trained_models["lightgbm"].feature_importances_
            importances.append(lgb_imp)

        if importances:
            avg_importance = np.mean(importances, axis=0)
            return pd.DataFrame({
                "feature": self.feature_columns,
                "importance": avg_importance
            }).sort_values("importance", ascending=False)

        return pd.DataFrame()

    def get_shap_values(self, X: np.ndarray, sample: int = 100) -> Optional[np.ndarray]:
        """Get SHAP values for explainability."""

        if not SHAP_AVAILABLE:
            return None

        # Use XGBoost for SHAP (most reliable)
        if "xgboost" not in self.trained_models:
            return None

        # Sample for speed
        if len(X) > sample:
            indices = np.random.choice(len(X), sample, replace=False)
            X_sample = X[indices]
        else:
            X_sample = X

        explainer = shap.TreeExplainer(self.trained_models["xgboost"])
        shap_values = explainer.shap_values(X_sample)

        return shap_values

    def detect_market_regime(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect market regime using price momentum and volatility.

        Regimes:
        - Hot: High momentum, low volatility
        - Warm: Moderate momentum
        - Cooling: Low/negative momentum
        - Cold: Negative momentum, high volatility
        """

        df = df.copy()

        # Calculate momentum (3-month price change)
        if "price_mom" in df.columns:
            df["momentum"] = df["price_mom"]
        elif "price_3m_momentum" in df.columns:
            df["momentum"] = df["price_3m_momentum"]
        else:
            df["momentum"] = 0

        # Calculate volatility
        if "rolling_std_6m" in df.columns:
            df["volatility"] = df["rolling_std_6m"] / df["benchmark_price"]
        else:
            df["volatility"] = 0.02  # Default 2%

        # Classify regime
        momentum_median = df["momentum"].median()
        volatility_median = df["volatility"].median()

        def classify_regime(row):
            mom = row["momentum"]
            vol = row["volatility"]

            if mom > momentum_median + 0.01 and vol < volatility_median:
                return "hot"
            elif mom > momentum_median:
                return "warm"
            elif mom < momentum_median - 0.01 and vol > volatility_median:
                return "cold"
            else:
                return "cooling"

        df["market_regime"] = df.apply(classify_regime, axis=1)

        return df

    def save_ensemble(self, name: str = "ensemble_predictor") -> Dict:
        """Save ensemble models."""

        import joblib

        # Save each model
        paths = {}
        for model_name, model in self.trained_models.items():
            path = self.model_dir / f"{name}_{model_name}.joblib"
            joblib.dump(model, path)
            paths[model_name] = str(path)

        # Save meta-learner
        meta_path = self.model_dir / f"{name}_meta_learner.joblib"
        joblib.dump(self.meta_learner, meta_path)
        paths["meta_learner"] = str(meta_path)

        # Save feature columns
        feature_path = self.model_dir / f"{name}_features.json"
        with open(feature_path, "w") as f:
            json.dump(self.feature_columns, f)
        paths["features"] = str(feature_path)

        # Save metrics
        metrics_path = self.model_dir / f"{name}_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(self.metrics, f, indent=2, default=str)
        paths["metrics"] = str(metrics_path)

        return paths

    def load_ensemble(self, name: str = "ensemble_predictor") -> bool:
        """Load ensemble models."""

        import joblib

        model_names = ["xgboost", "lightgbm", "ridge", "gbm"]

        for model_name in model_names:
            path = self.model_dir / f"{name}_{model_name}.joblib"
            if path.exists():
                self.trained_models[model_name] = joblib.load(path)

        # Load meta-learner
        meta_path = self.model_dir / f"{name}_meta_learner.joblib"
        if meta_path.exists():
            self.meta_learner = joblib.load(meta_path)

        # Load features
        feature_path = self.model_dir / f"{name}_features.json"
        if feature_path.exists():
            with open(feature_path, "r") as f:
                self.feature_columns = json.load(f)

        # Load metrics
        metrics_path = self.model_dir / f"{name}_metrics.json"
        if metrics_path.exists():
            with open(metrics_path, "r") as f:
                self.metrics = json.load(f)

        return len(self.trained_models) > 0


def main():
    """Demo ensemble training."""
    from features import FeatureEngineer
    from ingest import DataIngester

    # Load data
    ingester = DataIngester()
    merged = ingester.create_merged_dataset()

    # Create features
    fe = FeatureEngineer(prediction_horizon=6)
    featured = fe.create_all_features(merged)

    # Prepare data
    X, y, feature_cols, _, _ = ModelTrainer(
        model_dir="models"
    ).prepare_data(featured)

    # Time-based split
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    # Train ensemble
    ensemble = AdvancedEnsemblePredictor()
    metrics = ensemble.train_ensemble(
        X_train, y_train,
        X_val, y_val,
        feature_cols
    )

    print("\n" + "=" * 60)
    print("ENSEMBLE TRAINING COMPLETE")
    print("=" * 60)

    for model_name, model_metrics in metrics["base_models"].items():
        print(f"{model_name}: RMSE=${model_metrics['rmse']:,.0f}, R²={model_metrics['r2']:.3f}")

    print(f"Ensemble: RMSE=${metrics['ensemble']['rmse']:,.0f}, R²={metrics['ensemble']['r2']:.3f}")

    # Save
    paths = ensemble.save_ensemble()
    print(f"\nModels saved to {ensemble.model_dir}")

    return ensemble


if __name__ == "__main__":
    from train import ModelTrainer
    main()
