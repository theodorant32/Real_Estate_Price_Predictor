"""
Continuous Training Module

Handles scheduled model retraining with:
- Model versioning
- Performance tracking
- Automatic retraining triggers
- Model comparison and promotion
"""

import os
import json
import shutil
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import joblib
import logging

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from train import ModelTrainer
from predict import PricePredictor
from validate import ValidationReport
from config import ModelConfig, logger

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelRegistry:
    """
    Manages model versions, metadata, and lifecycle.
    """

    def __init__(self, models_dir: str = "models"):
        """
        Args:
            models_dir: Directory for model storage
        """
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)

        self.registry_file = self.models_dir / "model_registry.json"
        self.performance_file = self.models_dir / "performance_history.json"

        # Initialize registry if not exists
        self._init_registry()

    def _init_registry(self):
        """Initialize registry files."""
        if not self.registry_file.exists():
            registry = {
                "current_model": None,
                "models": {},
                "last_updated": None
            }
            self._save_registry(registry)

        if not self.performance_file.exists():
            self._save_performance_history([])

    def _save_registry(self, registry: Dict):
        """Save registry to disk."""
        with open(self.registry_file, 'w') as f:
            json.dump(registry, f, indent=2, default=str)

    def _load_registry(self) -> Dict:
        """Load registry from disk."""
        with open(self.registry_file, 'r') as f:
            return json.load(f)

    def _save_performance_history(self, history: List):
        """Save performance history to disk."""
        with open(self.performance_file, 'w') as f:
            json.dump(history, f, indent=2, default=str)

    def _load_performance_history(self) -> List:
        """Load performance history from disk."""
        with open(self.performance_file, 'r') as f:
            return json.load(f)

    def register_new_model(
        self,
        model_path: str,
        metrics: Dict,
        training_data_info: Dict = None
    ) -> str:
        """
        Register a newly trained model.

        Args:
            model_path: Path to the model file
            metrics: Training metrics (RMSE, MAPE, etc.)
            training_data_info: Info about training data

        Returns:
            Model version ID
        """
        registry = self._load_registry()

        # Generate version ID
        version = len(registry["models"]) + 1
        version_id = f"v{version:03d}"
        timestamp = datetime.now().isoformat()

        # Model metadata
        model_info = {
            "version_id": version_id,
            "path": str(model_path),
            "timestamp": timestamp,
            "metrics": metrics,
            "training_data_info": training_data_info or {},
            "status": "staging"  # staging, production, archived
        }

        # Register model
        registry["models"][version_id] = model_info

        # If first model, make it current
        if registry["current_model"] is None:
            registry["current_model"] = version_id
            model_info["status"] = "production"

        registry["last_updated"] = timestamp

        self._save_registry(registry)

        # Log performance
        self._log_performance(version_id, metrics, timestamp)

        logger.info(f"Registered new model: {version_id}")
        return version_id

    def _log_performance(self, version_id: str, metrics: Dict, timestamp: str):
        """Log model performance for tracking."""
        history = self._load_performance_history()

        history.append({
            "version_id": version_id,
            "timestamp": timestamp,
            "metrics": {
                "rmse": metrics.get("avg_rmse"),
                "mape": metrics.get("avg_mape"),
                "rmse_by_city": metrics.get("rmse_by_city"),
                "rmse_by_property_type": metrics.get("rmse_by_property_type")
            }
        })

        # Keep last 100 entries
        history = history[-100:]

        self._save_performance_history(history)

    def promote_model(self, version_id: str) -> bool:
        """
        Promote a model to production.

        Args:
            version_id: Model version to promote

        Returns:
            True if successful
        """
        registry = self._load_registry()

        if version_id not in registry["models"]:
            logger.error(f"Model {version_id} not found")
            return False

        # Demote current production model
        if registry["current_model"]:
            current = registry["current_model"]
            registry["models"][current]["status"] = "archived"

        # Promote new model
        registry["models"][version_id]["status"] = "production"
        registry["current_model"] = version_id
        registry["last_updated"] = datetime.now().isoformat()

        self._save_registry(registry)

        logger.info(f"Promoted model {version_id} to production")
        return True

    def get_current_model(self) -> Optional[Dict]:
        """Get current production model info."""
        registry = self._load_registry()

        if registry["current_model"] is None:
            return None

        return registry["models"].get(registry["current_model"])

    def get_model_versions(self) -> List[Dict]:
        """Get all model versions."""
        registry = self._load_registry()
        return list(registry["models"].values())

    def compare_models(self, version_ids: List[str] = None) -> pd.DataFrame:
        """
        Compare model versions.

        Args:
            version_ids: List of versions to compare (default: all)

        Returns:
            DataFrame with comparison metrics
        """
        registry = self._load_registry()

        if version_ids is None:
            version_ids = list(registry["models"].keys())

        comparisons = []
        for vid in version_ids:
            if vid in registry["models"]:
                model = registry["models"][vid]
                metrics = model.get("metrics", {})
                comparisons.append({
                    "version": vid,
                    "rmse": metrics.get("avg_rmse"),
                    "mape": metrics.get("avg_mape"),
                    "status": model.get("status"),
                    "timestamp": model.get("timestamp"),
                    "samples": model.get("training_data_info", {}).get("n_samples")
                })

        return pd.DataFrame(comparisons)

    def get_performance_trend(self, n_versions: int = 10) -> pd.DataFrame:
        """
        Get performance trend over recent models.

        Args:
            n_versions: Number of recent versions to analyze

        Returns:
            DataFrame with performance trend
        """
        history = self._load_performance_history()

        if not history:
            return pd.DataFrame()

        df = pd.DataFrame(history)
        df = df.tail(n_versions)

        return df


class ContinuousTrainer:
    """
    Manages continuous training pipeline.
    """

    def __init__(
        self,
        models_dir: str = "models",
        data_dir: str = "data/processed"
    ):
        """
        Args:
            models_dir: Directory for model storage
            data_dir: Directory for processed data
        """
        self.models_dir = Path(models_dir)
        self.data_dir = Path(data_dir)
        self.registry = ModelRegistry(models_dir)
        self.config = ModelConfig()

    def check_retrain_needed(self) -> Tuple[bool, str]:
        """
        Check if model needs retraining.

        Triggers retrain if:
        1. Last training was > 7 days ago
        2. New data points > threshold
        3. Performance degradation detected

        Returns:
            (needs_retrain, reason) tuple
        """
        current_model = self.registry.get_current_model()

        # No model exists
        if current_model is None:
            return True, "No model exists"

        # Check time since last training
        last_trained = datetime.fromisoformat(current_model["timestamp"])
        days_since = (datetime.now() - last_trained).days

        if days_since >= self.config.retrain_frequency_days:
            return True, f"Scheduled retrain ({days_since} days since last training)"

        # Check for new data
        new_data_count = self._count_new_data_points(last_trained)
        if new_data_count >= self.config.min_new_data_points:
            return True, f"New data available ({new_data_count} new points)"

        # Check performance degradation (would need live predictions)
        # This is a placeholder for production implementation

        return False, "Model is current"

    def _count_new_data_points(self, since: datetime) -> int:
        """Count new data points since given date."""
        featured_file = self.data_dir / "featured_data.csv"

        if not featured_file.exists():
            return 0

        try:
            df = pd.read_csv(featured_file, parse_dates=["date"])

            # Count rows with date > since
            new_rows = df[df["date"] > since]
            return len(new_rows)

        except Exception as e:
            logger.error(f"Failed to count new data: {e}")
            return 0

    def run_training_pipeline(self, force: bool = False) -> Dict:
        """
        Run the complete training pipeline.

        Args:
            force: Force retraining even if not needed

        Returns:
            Training results dictionary
        """
        # Check if retrain needed
        needs_retrain, reason = self.check_retrain_needed()

        if not needs_retrain and not force:
            logger.info(f"Retrain not needed: {reason}")
            return {"status": "skipped", "reason": reason}

        logger.info(f"Starting training: {reason}")

        # Load data
        featured_file = self.data_dir / "featured_data.csv"
        if not featured_file.exists():
            logger.error("Featured data not found. Run: python src/features.py")
            return {"status": "failed", "error": "No data"}

        logger.info(f"Loading data from {featured_file}")
        df = pd.read_csv(featured_file, parse_dates=["date"])

        # Train model
        trainer = ModelTrainer(model_dir=str(self.models_dir))

        logger.info("Training model...")
        results = trainer.train(df, model_name="price_predictor_staging")

        # Validate model before registering
        logger.info("Running validation checks...")
        validator = ValidationReport(models_dir=str(self.models_dir))
        val_report = validator.generate_report(model_name="price_predictor_staging")

        # Check for critical issues
        critical_issues = []
        for check_name, check_result in val_report.get("checks", {}).items():
            critical_issues.extend(check_result.get("issues", []))

        if critical_issues:
            logger.warning(f"Model has {len(critical_issues)} critical issues:")
            for issue in critical_issues:
                logger.warning(f"  - {issue}")
            # Still register but mark as potentially problematic
            version_id = self.registry.register_new_model(
                model_path=str(self.models_dir / "price_predictor_staging.json"),
                metrics=results["metrics"]["cv_results"],
                training_data_info={
                    "n_samples": results["metrics"]["n_samples"],
                    "n_features": results["metrics"]["n_features"],
                    "date_range": str(df["date"].min()) + " to " + str(df["date"].max()),
                    "validation_issues": critical_issues
                }
            )
        else:
            logger.info("Validation passed - no critical issues")
            version_id = self.registry.register_new_model(
                model_path=str(self.models_dir / "price_predictor_staging.json"),
                metrics=results["metrics"]["cv_results"],
                training_data_info={
                    "n_samples": results["metrics"]["n_samples"],
                    "n_features": results["metrics"]["n_features"],
                    "date_range": str(df["date"].min()) + " to " + str(df["date"].max())
                }
            )

        # Compare with current model
        comparison = self._compare_with_current(version_id)

        # Auto-promote if better or no current model
        if comparison["should_promote"]:
            self.registry.promote_model(version_id)
            logger.info(f"Auto-promoted {version_id} to production")

        return {
            "status": "completed",
            "version_id": version_id,
            "metrics": results["metrics"]["cv_results"],
            "comparison": comparison
        }

    def _compare_with_current(self, new_version_id: str) -> Dict:
        """
        Compare new model with current production model.

        Args:
            new_version_id: New model version

        Returns:
            Comparison results with promotion recommendation
        """
        current = self.registry.get_current_model()

        if current is None:
            return {"should_promote": True, "reason": "No current model"}

        registry = self.registry._load_registry()
        new_model = registry["models"][new_version_id]

        current_rmse = current.get("metrics", {}).get("avg_rmse", float('inf'))
        new_rmse = new_model.get("metrics", {}).get("avg_rmse", float('inf'))

        current_mape = current.get("metrics", {}).get("avg_mape", float('inf'))
        new_mape = new_model.get("metrics", {}).get("avg_mape", float('inf'))

        # Promote if RMSE or MAPE improved
        improvement = (current_rmse - new_rmse) / current_rmse if current_rmse > 0 else 0

        should_promote = (
            new_rmse < current_rmse or  # Better RMSE
            new_mape < current_mape or  # Better MAPE
            improvement > 0.05  # 5% improvement threshold
        )

        return {
            "should_promote": should_promote,
            "current_rmse": current_rmse,
            "new_rmse": new_rmse,
            "improvement_pct": improvement * 100
        }

    def get_model_status(self) -> Dict:
        """Get current model status and health."""
        current = self.registry.get_current_model()

        if current is None:
            return {
                "status": "no_model",
                "message": "No model trained yet"
            }

        # Get performance trend
        trend = self.registry.get_performance_trend()

        # Calculate health metrics
        recent_mape = trend["metrics"].apply(lambda x: x.get("mape")).tail(3).mean() if len(trend) > 0 else None

        health = "good"
        if recent_mape:
            if recent_mape > 0.20:
                health = "warning"
            elif recent_mape > 0.30:
                health = "critical"

        return {
            "status": "active",
            "current_version": current["version_id"],
            "last_trained": current["timestamp"],
            "health": health,
            "metrics": current.get("metrics"),
            "training_samples": current.get("training_data_info", {}).get("n_samples")
        }


def run_scheduled_retrain():
    """Entry point for scheduled retraining (e.g., cron job)."""
    print("=" * 60)
    print("CONTINUOUS TRAINING PIPELINE")
    print("=" * 60)

    trainer = ContinuousTrainer()

    # Run training
    results = trainer.run_training_pipeline()

    print(f"\nStatus: {results['status']}")

    if results["status"] == "completed":
        print(f"Version: {results['version_id']}")
        print(f"RMSE: ${results['metrics']['avg_rmse']:,.0f}")
        print(f"MAPE: {results['metrics']['avg_mape']:.2%}")

        if results.get("comparison", {}).get("should_promote"):
            print("✓ Model promoted to production")
        else:
            print("○ Model kept in staging")

    elif results["status"] == "skipped":
        print(f"Reason: {results['reason']}")

    return results


def main():
    """Demo continuous training."""
    print("=" * 60)
    print("MODEL REGISTRY STATUS")
    print("=" * 60)

    trainer = ContinuousTrainer()
    status = trainer.get_model_status()

    print(f"\nCurrent Model: {status.get('current_version', 'None')}")
    print(f"Health: {status.get('health', 'N/A')}")
    print(f"Training Samples: {status.get('training_samples', 'N/A')}")

    if status.get("metrics"):
        print(f"\nMetrics:")
        print(f"  RMSE: ${status['metrics'].get('avg_rmse', 0):,.0f}")
        print(f"  MAPE: {status['metrics'].get('avg_mape', 0):.2%}")

    # Check if retrain needed
    print("\n" + "=" * 60)
    print("RETRAIN CHECK")
    print("=" * 60)

    needs_retrain, reason = trainer.check_retrain_needed()
    print(f"Needs Retrain: {needs_retrain}")
    print(f"Reason: {reason}")

    # Show model versions
    print("\n" + "=" * 60)
    print("MODEL VERSIONS")
    print("=" * 60)

    comparison = trainer.registry.compare_models()
    if len(comparison) > 0:
        print(comparison.to_string(index=False))
    else:
        print("No models registered yet")


if __name__ == "__main__":
    main()
