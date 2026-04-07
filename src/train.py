import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_percentage_error,
    mean_absolute_error,
    r2_score
)
from pathlib import Path
import joblib
import json
from typing import Dict, Tuple, List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PurgedTimeSeriesSplit:
    def __init__(
        self,
        n_splits: int = 5,
        embargo_periods: int = 6,
        test_size: int = 12
    ):
        self.n_splits = n_splits
        self.embargo_periods = embargo_periods
        self.test_size = test_size

    def split(self, X: np.ndarray, dates: pd.Series = None) -> List[Tuple[np.ndarray, np.ndarray]]:
        n_samples = len(X)

        # Calculate fold size based on time (not samples)
        # Each fold should have roughly equal time spans
        min_train_size = int(n_samples * 0.5)  # Start with 50% for training
        fold_size = int((n_samples - min_train_size) / self.n_splits)

        splits = []
        for i in range(self.n_splits):
            # Training end for this fold
            train_end = min_train_size + i * fold_size

            # Embargo: skip embargo_periods months
            val_start = train_end + self.embargo_periods

            # Validation end
            val_end = val_start + self.test_size

            if val_end > n_samples:
                break

            train_idx = np.arange(0, train_end)
            val_idx = np.arange(val_start, min(val_end, n_samples))

            if len(train_idx) > 0 and len(val_idx) > 0:
                splits.append((train_idx, val_idx))

        return splits


class ModelTrainer:
    # Minimum data requirements
    MIN_SAMPLES = 100
    MIN_TIMEPOINTS = 24  # At least 2 years of monthly data

    def __init__(
        self,
        model_dir: str = "models",
        n_splits: int = 5,
        prediction_horizon: int = 6,
        embargo_periods: int = 6,
        holdout_months: int = 12
    ):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.n_splits = n_splits
        self.prediction_horizon = prediction_horizon
        self.embargo_periods = embargo_periods
        self.holdout_months = holdout_months

        # XGBoost hyperparameters with REGULARIZATION
        self.model_params = {
            "n_estimators": 300,  # Reduced from 500
            "learning_rate": 0.03,  # Reduced from 0.05
            "max_depth": 4,  # Reduced from 6 (prevent overfitting)
            "min_child_weight": 5,  # Increased from default 1
            "subsample": 0.7,  # Reduced from 0.8
            "colsample_bytree": 0.7,  # Reduced from 0.8
            "colsample_bylevel": 0.8,  # Additional regularization
            "reg_alpha": 0.1,  # L1 regularization (NEW)
            "reg_lambda": 1.0,  # L2 regularization (NEW)
            "gamma": 0.1,  # Minimum loss reduction for splits (NEW)
            "early_stopping_rounds": 50,
            "random_state": 42,
            "n_jobs": -1
        }

        self.model = None
        self.feature_columns = None
        self.training_metrics = {}
        self.holdout_metrics = None

    def check_data_quality(
        self,
        df: pd.DataFrame,
        target_col: str
    ) -> Dict:
        issues = []
        warnings = []

        n_samples = len(df)
        if n_samples < self.MIN_SAMPLES:
            issues.append(f"Too few samples: {n_samples} < {self.MIN_SAMPLES}")

        # Check time coverage
        if "date" in df.columns:
            time_range = (df["date"].max() - df["date"].min()).days / 30
            if time_range < self.MIN_TIMEPOINTS:
                warnings.append(
                    f"Short time series: {time_range:.0f} months < {self.MIN_TIMEPOINTS} recommended"
                )

        # Check target distribution
        target = df[target_col].dropna()
        if len(target) < n_samples * 0.8:
            issues.append(f"Too much missing target data: {len(target)}/{n_samples}")

        # Check for target leakage (extreme values)
        if len(target) > 0:
            z_scores = np.abs((target - target.mean()) / (target.std() + 1e-8))
            if (z_scores > 5).any():
                warnings.append(
                    f"Extreme outliers in target: {(z_scores > 5).sum()} samples with |z| > 5"
                )

        # Check class balance for directional prediction
        target_returns = df.get("target_return_6m", pd.Series([])).dropna()
        if len(target_returns) > 0:
            positive_pct = (target_returns > 0).mean()
            if positive_pct < 0.3 or positive_pct > 0.7:
                warnings.append(
                    f"Imbalanced target returns: {positive_pct:.1%} positive"
                )

        return {
            "n_samples": n_samples,
            "issues": issues,
            "warnings": warnings,
            "can_proceed": len(issues) == 0
        }

    def prepare_data(
        self,
        df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, List[str], pd.DataFrame]:
        # Columns to exclude - comprehensive list to prevent leakage
        exclude_cols = {
            # Identifiers and temporal
            "date", "city", "property_type",
            # Target variables (obvious leakage)
            "target_price_6m", "target_return_6m",
            # Categorical variables that get encoded
            "market_tightness", "rate_regime",
            # Current price (would be leakage - we're predicting it)
            "benchmark_price",
            # Any future-looking features
            "target_price_1m", "target_price_3m", "target_price_12m",
        }

        # Get feature columns (only numeric, non-excluded)
        feature_cols = [
            col for col in df.columns
            if col not in exclude_cols
            and df[col].dtype in ["float64", "int64", "int32", "float32", "uint8"]
        ]

        target_col = "target_price_6m"

        # Drop rows with NaN in features or target
        valid_mask = df[target_col].notna()
        for col in feature_cols:
            valid_mask &= df[col].notna()

        X = df.loc[valid_mask, feature_cols].values
        y = df.loc[valid_mask, target_col].values
        dates = df.loc[valid_mask, "date"].copy()

        # Keep metadata for analysis
        meta = df.loc[valid_mask, ["city", "property_type", "benchmark_price"]].copy()

        logger.info(f"Prepared data: X shape = {X.shape}, y shape = {y.shape}")
        logger.info(f"  Dropped {len(df) - len(X)} rows with missing values")

        return X, y, feature_cols, meta, dates

    def create_holdout_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        meta: pd.DataFrame,
        dates: pd.Series
    ) -> Dict:
        n_samples = len(X)
        holdout_size = min(
            self.holdout_months * 6,  # Approximate rows per month
            int(n_samples * 0.2)  # Max 20% of data
        )

        train_idx = np.arange(0, n_samples - holdout_size)
        test_idx = np.arange(n_samples - holdout_size, n_samples)

        return {
            "X_train": X[train_idx],
            "y_train": y[train_idx],
            "X_test": X[test_idx],
            "y_test": y[test_idx],
            "meta_train": meta.iloc[train_idx].copy(),
            "meta_test": meta.iloc[test_idx].copy(),
            "dates_train": dates.iloc[train_idx].copy(),
            "dates_test": dates.iloc[test_idx].copy(),
        }

    def calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        actual_prices: np.ndarray = None
    ) -> Dict:
        metrics = {}

        # Standard regression metrics
        metrics["rmse"] = np.sqrt(mean_squared_error(y_true, y_pred))
        metrics["mae"] = mean_absolute_error(y_true, y_pred)
        metrics["mape"] = mean_absolute_percentage_error(y_true, y_pred, sample_weight=(y_true != 0))
        metrics["r2"] = r2_score(y_true, y_pred)

        # Directional accuracy (can we predict price increases vs decreases?)
        if actual_prices is not None:
            pred_prices = y_pred
            actual_changes = actual_prices[1:] - actual_prices[:-1]
            pred_changes = pred_prices[1:] - pred_prices[:-1]

            # For target returns
            if len(y_true) > 1:
                pred_direction = np.sign(y_pred)
                actual_direction = np.sign(y_true)
                metrics["directional_accuracy"] = (pred_direction == actual_direction).mean()

        # Calibration: mean prediction error
        metrics["mean_prediction_error"] = (y_pred - y_true).mean()
        metrics["prediction_bias"] = np.median(y_pred - y_true)

        # Economic significance
        if actual_prices is not None and len(actual_prices) > 0:
            mean_price = actual_prices.mean()
            metrics["rmse_pct"] = metrics["rmse"] / mean_price
            metrics["mae_pct"] = metrics["mae"] / mean_price

        return metrics

    def time_series_cv(
        self,
        X: np.ndarray,
        y: np.ndarray,
        meta: pd.DataFrame,
        dates: pd.Series
    ) -> Dict:
        cv = PurgedTimeSeriesSplit(
            n_splits=self.n_splits,
            embargo_periods=self.embargo_periods
        )

        splits = cv.split(X, dates)
        fold_metrics = []
        all_val_predictions = []
        all_val_actuals = []

        logger.info(f"\nRunning {self.n_splits}-fold purged time-series CV...")
        logger.info(f"  Embargo period: {self.embargo_periods} months")

        for fold, (train_idx, val_idx) in enumerate(splits):
            logger.info(f"\n  Fold {fold + 1}/{len(splits)}")
            logger.info(f"    Train: {len(train_idx)} samples, Val: {len(val_idx)} samples")

            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # Train model with regularization
            model = xgb.XGBRegressor(**self.model_params)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )

            # Predict
            y_pred = model.predict(X_val)

            # Calculate metrics
            metrics = self.calculate_metrics(y_val, y_pred)

            # Get meta for this fold
            val_meta = meta.iloc[val_idx].copy()
            val_meta["pred"] = y_pred
            val_meta["actual"] = y_val

            # Error by city and property type
            city_errors = val_meta.groupby("city").apply(
                lambda x: np.sqrt(mean_squared_error(x["actual"], x["pred"]))
                if len(x) > 1 else np.nan
            ).to_dict()

            type_errors = val_meta.groupby("property_type").apply(
                lambda x: np.sqrt(mean_squared_error(x["actual"], x["pred"]))
                if len(x) > 1 else np.nan
            ).to_dict()

            fold_metrics.append({
                "fold": fold + 1,
                "train_size": len(train_idx),
                "val_size": len(val_idx),
                "rmse": metrics["rmse"],
                "mae": metrics["mae"],
                "mape": metrics["mape"],
                "r2": metrics["r2"],
                "directional_accuracy": metrics.get("directional_accuracy"),
                "rmse_by_city": city_errors,
                "rmse_by_property_type": type_errors
            })

            all_val_predictions.extend(y_pred)
            all_val_actuals.extend(y_val)

            logger.info(f"    RMSE: ${metrics['rmse']:,.0f}")
            logger.info(f"    MAPE: {metrics['mape']:.2%}")
            logger.info(f"    R²: {metrics['r2']:.3f}")

        # Aggregate metrics
        avg_rmse = np.mean([m["rmse"] for m in fold_metrics])
        avg_mape = np.mean([m["mape"] for m in fold_metrics])
        avg_r2 = np.mean([m["r2"] for m in fold_metrics])

        # Handle None values in directional accuracy
        directional_accs = [m["directional_accuracy"] for m in fold_metrics if m.get("directional_accuracy") is not None]
        avg_directional = np.mean(directional_accs) if directional_accs else None

        # Aggregate city/property errors
        all_cities = set()
        all_types = set()
        for m in fold_metrics:
            all_cities.update(m["rmse_by_city"].keys())
            all_types.update(m["rmse_by_property_type"].keys())

        city_rmse_avg = {
            city: np.nanmean([m["rmse_by_city"].get(city) for m in fold_metrics if city in m["rmse_by_city"]])
            for city in all_cities
        }

        type_rmse_avg = {
            ptype: np.nanmean([m["rmse_by_property_type"].get(ptype) for m in fold_metrics if ptype in m["rmse_by_property_type"]])
            for ptype in all_types
        }

        # Calculate overall metrics on all CV predictions
        all_val_actuals = np.array(all_val_actuals)
        all_val_predictions = np.array(all_val_predictions)
        overall_metrics = self.calculate_metrics(all_val_actuals, all_val_predictions)

        return {
            "fold_metrics": fold_metrics,
            "avg_rmse": avg_rmse,
            "avg_mape": avg_mape,
            "avg_r2": avg_r2,
            "avg_directional_accuracy": avg_directional,
            "overall_cv_metrics": overall_metrics,
            "rmse_by_city": city_rmse_avg,
            "rmse_by_property_type": type_rmse_avg
        }

    def evaluate_on_holdout(
        self,
        model: xgb.XGBRegressor,
        X_test: np.ndarray,
        y_test: np.ndarray,
        meta_test: pd.DataFrame
    ) -> Dict:
        y_pred = model.predict(X_test)

        metrics = self.calculate_metrics(y_test, y_pred, meta_test["benchmark_price"].values)

        # Add breakdown by city and property type
        test_meta = meta_test.copy()
        test_meta["pred"] = y_pred
        test_meta["actual"] = y_test

        city_metrics = {}
        for city in test_meta["city"].unique():
            city_data = test_meta[test_meta["city"] == city]
            if len(city_data) > 1:
                city_metrics[city] = self.calculate_metrics(
                    city_data["actual"].values,
                    city_data["pred"].values
                )

        type_metrics = {}
        for ptype in test_meta["property_type"].unique():
            type_data = test_meta[test_meta["property_type"] == ptype]
            if len(type_data) > 1:
                type_metrics[ptype] = self.calculate_metrics(
                    type_data["actual"].values,
                    type_data["pred"].values
                )

        return {
            "overall": metrics,
            "by_city": city_metrics,
            "by_property_type": type_metrics,
            "n_test_samples": len(y_test)
        }

    def train_final_model(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_columns: List[str]
    ) -> xgb.XGBRegressor:
        logger.info("\nTraining final model...")

        # Remove early stopping params (no validation set)
        final_params = {k: v for k, v in self.model_params.items() if k != "early_stopping_rounds"}

        self.model = xgb.XGBRegressor(**final_params)
        self.model.fit(X, y, verbose=False)
        self.feature_columns = feature_columns

        logger.info(f"  Model trained with {len(feature_columns)} features")

        return self.model

    def get_feature_importance(self) -> pd.DataFrame:
        if self.model is None:
            raise ValueError("No model trained yet")

        importance_df = pd.DataFrame({
            "feature": self.feature_columns,
            "importance": self.model.feature_importances_
        }).sort_values("importance", ascending=False)

        return importance_df

    def check_overfitting(self, cv_metrics: Dict, holdout_metrics: Dict) -> Dict:
        cv_rmse = cv_metrics.get("avg_rmse", float("inf"))
        holdout_rmse = holdout_metrics.get("overall", {}).get("rmse", float("inf"))

        cv_mape = cv_metrics.get("avg_mape", 1.0)
        holdout_mape = holdout_metrics.get("overall", {}).get("mape", 1.0)

        cv_r2 = cv_metrics.get("avg_r2", 0)
        holdout_r2 = holdout_metrics.get("overall", {}).get("r2", 0)

        # Calculate degradation
        rmse_degradation = (holdout_rmse - cv_rmse) / (cv_rmse + 1e-8)
        mape_degradation = (holdout_mape - cv_mape) / (cv_mape + 1e-8)
        r2_drop = cv_r2 - holdout_r2

        # Flags
        warnings = []
        if rmse_degradation > 0.2:
            warnings.append(f"⚠ RMSE degraded by {rmse_degradation:.1%} on holdout")
        if mape_degradation > 0.2:
            warnings.append(f"⚠ MAPE degraded by {mape_degradation:.1%} on holdout")
        if r2_drop > 0.1:
            warnings.append(f"⚠ R² dropped by {r2_drop:.2f} on holdout")

        return {
            "cv_rmse": cv_rmse,
            "holdout_rmse": holdout_rmse,
            "rmse_degradation_pct": rmse_degradation * 100,
            "cv_mape": cv_mape,
            "holdout_mape": holdout_mape,
            "mape_degradation_pct": mape_degradation * 100,
            "cv_r2": cv_r2,
            "holdout_r2": holdout_r2,
            "r2_drop": r2_drop,
            "warnings": warnings,
            "overfitting_risk": "high" if len(warnings) >= 2 else "medium" if len(warnings) >= 1 else "low"
        }

    def save_model(self, model_name: str = "price_predictor") -> Dict:
        if self.model is None:
            raise ValueError("No model trained yet")

        model_path = self.model_dir / f"{model_name}.json"
        self.model.save_model(str(model_path))

        feature_path = self.model_dir / f"{model_name}_features.json"
        with open(feature_path, "w") as f:
            json.dump(self.feature_columns, f)

        metrics_path = self.model_dir / f"{model_name}_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(self.training_metrics, f, indent=2, default=str)

        logger.info(f"\nSaved model to {model_path}")
        logger.info(f"Saved features to {feature_path}")
        logger.info(f"Saved metrics to {metrics_path}")

        return {
            "model_path": str(model_path),
            "feature_path": str(feature_path),
            "metrics_path": str(metrics_path)
        }

    def train(
        self,
        df: pd.DataFrame,
        model_name: str = "price_predictor"
    ) -> Dict:
        # Step 1: Data quality check
        logger.info("=" * 60)
        logger.info("DATA QUALITY CHECK")
        logger.info("=" * 60)

        quality = self.check_data_quality(df, "target_price_6m")
        logger.info(f"Samples: {quality['n_samples']}")

        for warning in quality["warnings"]:
            logger.warning(f"  {warning}")

        if not quality["can_proceed"]:
            logger.error("Data quality issues prevent training:")
            for issue in quality["issues"]:
                logger.error(f"  - {issue}")
            raise ValueError(f"Data quality check failed: {quality['issues']}")

        # Step 2: Prepare data
        logger.info("\n" + "=" * 60)
        logger.info("DATA PREPARATION")
        logger.info("=" * 60)

        X, y, feature_cols, meta, dates = self.prepare_data(df)

        # Step 3: Create holdout split
        logger.info("\n" + "=" * 60)
        logger.info("HOLDOUT TEST SET")
        logger.info("=" * 60)

        split = self.create_holdout_split(X, y, meta, dates)
        logger.info(f"Training samples: {len(split['X_train'])}")
        logger.info(f"Test samples: {len(split['X_test'])}")

        # Step 4: Cross-validation
        cv_results = self.time_series_cv(
            split["X_train"],
            split["y_train"],
            split["meta_train"],
            split["dates_train"]
        )

        # Step 5: Train final model on training portion only
        self.train_final_model(
            split["X_train"],
            split["y_train"],
            feature_cols
        )

        # Step 6: Evaluate on holdout
        logger.info("\n" + "=" * 60)
        logger.info("HOLDOUT EVALUATION")
        logger.info("=" * 60)

        self.holdout_metrics = self.evaluate_on_holdout(
            self.model,
            split["X_test"],
            split["y_test"],
            split["meta_test"]
        )

        logger.info(f"Holdout RMSE: ${self.holdout_metrics['overall']['rmse']:,.0f}")
        logger.info(f"Holdout MAPE: {self.holdout_metrics['overall']['mape']:.2%}")
        logger.info(f"Holdout R²: {self.holdout_metrics['overall']['r2']:.3f}")

        # Step 7: Check overfitting
        logger.info("\n" + "=" * 60)
        logger.info("OVERFITTING CHECK")
        logger.info("=" * 60)

        overfit_check = self.check_overfitting(cv_results, self.holdout_metrics)
        logger.info(f"CV RMSE: ${overfit_check['cv_rmse']:,.0f}")
        logger.info(f"Holdout RMSE: ${overfit_check['holdout_rmse']:,.0f}")
        logger.info(f"Degradation: {overfit_check['rmse_degradation_pct']:.1f}%")
        logger.info(f"Overfitting risk: {overfit_check['overfitting_risk']}")

        for warning in overfit_check["warnings"]:
            logger.warning(f"  {warning}")

        # Store all metrics
        self.training_metrics = {
            "cv_results": {
                "avg_rmse": cv_results["avg_rmse"],
                "avg_mape": cv_results["avg_mape"],
                "avg_r2": cv_results["avg_r2"],
                "avg_directional_accuracy": cv_results.get("avg_directional_accuracy"),
                "rmse_by_city": cv_results["rmse_by_city"],
                "rmse_by_property_type": cv_results["rmse_by_property_type"],
                "overall_cv_metrics": cv_results["overall_cv_metrics"],
                "fold_metrics": [
                    {k: v for k, v in m.items() if k not in ["rmse_by_city", "rmse_by_property_type"]}
                    for m in cv_results["fold_metrics"]
                ]
            },
            "holdout_results": self.holdout_metrics,
            "overfitting_check": overfit_check,
            "model_params": self.model_params,
            "n_samples": len(X),
            "n_features": len(feature_cols),
            "data_quality": quality
        }

        # Print summary
        logger.info("\n" + "=" * 60)
        logger.info("TRAINING SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Average CV RMSE: ${cv_results['avg_rmse']:,.0f}")
        logger.info(f"Average CV MAPE: {cv_results['avg_mape']:.2%}")
        logger.info(f"Average CV R²: {cv_results['avg_r2']:.3f}")
        logger.info(f"Holdout RMSE: ${self.holdout_metrics['overall']['rmse']:,.0f}")
        logger.info(f"Holdout MAPE: {self.holdout_metrics['overall']['mape']:.2%}")
        logger.info(f"Holdout R²: {self.holdout_metrics['overall']['r2']:.3f}")

        # Feature importance
        importance = self.get_feature_importance()
        logger.info("\nTop 15 Features:")
        for _, row in importance.head(15).iterrows():
            logger.info(f"  {row['feature']}: {row['importance']:.4f}")

        # Save model
        paths = self.save_model(model_name)

        return {
            **paths,
            "metrics": self.training_metrics,
            "feature_importance": importance.to_dict()
        }


def main():
    from pathlib import Path

    data_dir = Path("data/processed")
    featured_path = data_dir / "featured_data.csv"

    if not featured_path.exists():
        logger.error(f"Featured data not found at {featured_path}")
        logger.error("Run: python src/features.py first")
        return None

    logger.info(f"Loading featured data from {featured_path}...")
    df = pd.read_csv(featured_path, parse_dates=["date"])

    trainer = ModelTrainer(
        n_splits=5,
        embargo_periods=6,  # 6 months embargo
        holdout_months=12   # Last 12 months as holdout
    )
    results = trainer.train(df)

    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Model saved to: {results['model_path']}")
    logger.info(f"Features saved to: {results['feature_path']}")

    return results


if __name__ == "__main__":
    main()
