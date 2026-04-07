"""
Automated ML Pipeline for Canadian Real Estate Price Predictor

This pipeline:
1. Fetches fresh data from all sources
2. Creates features
3. Retrains the model
4. Validates performance against previous model
5. Deploys if better, otherwise rolls back

Usage:
    python src/pipeline.py [--force-refresh] [--skip-validation]

Schedule:
    Weekly on Sundays at 2am:
    0 2 * * 0 cd /path/to/Propra && python src/pipeline.py
"""

import sys
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional
import argparse

import pandas as pd
import numpy as np

from ingest import DataIngester
from features import FeatureEngineer
from train import ModelTrainer
from predict import PricePredictor


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('pipeline')


class ModelRegistry:
    """Track model versions and performance."""

    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.registry_path = self.models_dir / "model_registry.json"
        self.registry = self._load_registry()

    def _load_registry(self) -> Dict:
        if self.registry_path.exists():
            with open(self.registry_path, 'r') as f:
                data = json.load(f)
                # Convert old format (dict) to new format (list)
                if isinstance(data.get("models"), dict):
                    models_list = []
                    for version, info in data["models"].items():
                        info["version"] = version
                        info["status"] = info.get("status", "archived")
                        models_list.append(info)
                    data["models"] = models_list
                return data
        return {"models": [], "current_model": None}

    def save_registry(self):
        with open(self.registry_path, 'w') as f:
            json.dump(self.registry, f, indent=2)

    def register_model(
        self,
        version: str,
        metrics: Dict,
        model_path: str,
        features_path: str,
        training_samples: int,
        training_date_range: tuple
    ):
        """Register a new model version."""
        entry = {
            "version": version,
            "created_at": datetime.now().isoformat(),
            "metrics": metrics,
            "model_path": model_path,
            "features_path": features_path,
            "training_samples": training_samples,
            "training_date_range": {
                "start": str(training_date_range[0]),
                "end": str(training_date_range[1])
            },
            "status": "candidate"
        }

        self.registry["models"].append(entry)
        self.save_registry()
        return entry

    def promote_to_current(self, version: str):
        """Promote a model version to production."""
        for model in self.registry["models"]:
            if model["version"] == version:
                model["status"] = "production"
                self.registry["current_model"] = version
            elif model["status"] == "production":
                model["status"] = "archived"

        self.save_registry()
        logger.info(f"Promoted model {version} to production")

    def get_current_model(self) -> Optional[Dict]:
        """Get the current production model."""
        if self.registry["current_model"] is None:
            return None

        for model in self.registry["models"]:
            ver = model.get("version") or model.get("version_id")
            if ver == self.registry["current_model"]:
                return model

        return None

    def should_deploy(
        self,
        new_metrics: Dict,
        min_improvement: float = 0.02
    ) -> tuple:
        """
        Decide if new model should be deployed.

        Returns (should_deploy: bool, reason: str)
        """
        current = self.get_current_model()

        if current is None:
            return True, "No current model - deploying first model"

        # Handle both metric formats
        current_m = current.get("metrics", {})
        if "holdout" in current_m:
            current_rmse = current_m["holdout"].get("rmse", float('inf'))
            current_mape = current_m["holdout"].get("mape", float('inf'))
        elif "cv_results" in current_m:
            current_rmse = current_m["cv_results"].get("avg_rmse", float('inf'))
            current_mape = current_m["cv_results"].get("avg_mape", float('inf'))
        else:
            current_rmse = float('inf')
            current_mape = float('inf')

        new_rmse = new_metrics.get("holdout", {}).get("rmse", float('inf'))
        new_mape = new_metrics.get("holdout", {}).get("mape", float('inf'))

        # Check for significant improvement
        rmse_improvement = (current_rmse - new_rmse) / current_rmse if current_rmse > 0 else 0
        mape_improvement = (current_mape - new_mape) / current_mape if current_mape > 0 else 0

        if rmse_improvement >= min_improvement:
            return True, f"RMSE improved by {rmse_improvement:.1%}"

        if mape_improvement >= min_improvement:
            return True, f"MAPE improved by {mape_improvement:.1%}"

        # Check for degradation
        if new_rmse > current_rmse * 1.05:
            return False, f"RMSE degraded by {(new_rmse - current_rmse) / current_rmse:.1%}"

        if new_mape > current_mape * 1.05:
            return False, f"MAPE degraded by {(new_mape - current_mape) / current_mape:.1%}"

        return True, "Performance similar to current - deploying for freshness"


class Pipeline:
    """Main ML pipeline orchestrator."""

    def __init__(self, data_dir: str = "data", models_dir: str = "models"):
        self.data_dir = Path(data_dir)
        self.models_dir = Path(models_dir)
        self.ingester = DataIngester(data_dir)
        self.trainer = ModelTrainer(models_dir)
        self.registry = ModelRegistry(models_dir)

    def run(
        self,
        force_refresh: bool = False,
        skip_validation: bool = False,
        min_improvement: float = 0.02
    ) -> Dict:
        """
        Run the full pipeline.

        Args:
            force_refresh: Re-fetch all data from sources
            skip_validation: Deploy regardless of performance
            min_improvement: Minimum improvement to deploy

        Returns:
            Pipeline results dictionary
        """
        logger.info("=" * 60)
        logger.info("STARTING ML PIPELINE")
        logger.info(f"Force refresh: {force_refresh}")
        logger.info(f"Skip validation: {skip_validation}")
        logger.info("=" * 60)

        results = {
            "started_at": datetime.now().isoformat(),
            "status": "running",
            "steps": {}
        }

        try:
            # Step 1: Data Ingestion
            logger.info("\n[STEP 1/5] Fetching data from sources...")
            data = self.ingester.ingest_all(force_refresh=force_refresh)
            results["steps"]["ingestion"] = {
                "status": "success",
                "records": {k: len(v) for k, v in data.items()}
            }

            # Step 2: Create merged dataset
            logger.info("\n[STEP 2/5] Creating merged dataset...")
            merged = self.ingester.create_merged_dataset(data)
            merged_path = self.data_dir / "processed" / "merged_data.csv"
            merged.to_csv(merged_path, index=False)
            results["steps"]["merge"] = {
                "status": "success",
                "shape": list(merged.shape),
                "date_range": [str(merged["date"].min()), str(merged["date"].max())]
            }

            # Step 3: Feature engineering
            logger.info("\n[STEP 3/5] Creating features...")
            fe = FeatureEngineer(prediction_horizon=6)
            featured = fe.create_all_features(merged)
            featured_path = self.data_dir / "processed" / "featured_data.csv"
            featured.to_csv(featured_path, index=False)
            results["steps"]["features"] = {
                "status": "success",
                "shape": list(featured.shape),
                "feature_count": len(fe.get_feature_columns(featured))
            }

            # Step 4: Train model
            logger.info("\n[STEP 4/5] Training model...")
            version = datetime.now().strftime("v%Y%m%d_%H%M%S")
            train_results = self.trainer.train(featured, model_name="price_predictor")

            results["steps"]["training"] = {
                "status": "success",
                "version": version,
                "cv_rmse": train_results["metrics"]["cv_results"]["avg_rmse"],
                "cv_mape": train_results["metrics"]["cv_results"]["avg_mape"],
                "holdout_rmse": train_results["metrics"]["holdout_results"]["overall"]["rmse"],
                "holdout_mape": train_results["metrics"]["holdout_results"]["overall"]["mape"],
                "holdout_r2": train_results["metrics"]["holdout_results"]["overall"]["r2"],
            }

            # Step 5: Model validation and deployment
            logger.info("\n[STEP 5/5] Validating and deploying model...")

            metrics_for_validation = {
                "cv_results": train_results["metrics"]["cv_results"],
                "holdout": train_results["metrics"]["holdout_results"]["overall"]
            }
            should_deploy, reason = self.registry.should_deploy(
                metrics_for_validation,
                min_improvement=min_improvement
            )

            results["steps"]["validation"] = {
                "status": "success",
                "should_deploy": should_deploy,
                "reason": reason
            }

            if should_deploy or skip_validation:
                # Register and promote
                metrics_for_registry = {
                    "cv_results": train_results["metrics"]["cv_results"],
                    "holdout": train_results["metrics"]["holdout_results"]["overall"]
                }
                self.registry.register_model(
                    version=version,
                    metrics=metrics_for_registry,
                    model_path=str(self.models_dir / f"price_predictor.json"),
                    features_path=str(self.models_dir / f"price_predictor_features.json"),
                    training_samples=len(featured),
                    training_date_range=(
                        featured["date"].min(),
                        featured["date"].max()
                    )
                )
                self.registry.promote_to_current(version)

                results["deployment"] = {
                    "status": "deployed",
                    "version": version,
                    "reason": reason
                }
                logger.info(f"Model {version} deployed: {reason}")
            else:
                results["deployment"] = {
                    "status": "skipped",
                    "reason": reason
                }
                logger.warning(f"Model NOT deployed: {reason}")

            results["status"] = "completed"
            results["completed_at"] = datetime.now().isoformat()

        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
            results["status"] = "failed"
            results["error"] = str(e)

        # Save pipeline results
        results_path = self.data_dir.parent / "logs" / f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        results_path.parent.mkdir(parents=True, exist_ok=True)
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        logger.info(f"\nPipeline results saved to {results_path}")
        logger.info("=" * 60)
        logger.info(f"PIPELINE COMPLETED - Status: {results['status']}")
        logger.info("=" * 60)

        return results


def main():
    parser = argparse.ArgumentParser(description='ML Pipeline for Real Estate Predictor')
    parser.add_argument('--force-refresh', action='store_true', help='Re-fetch all data')
    parser.add_argument('--skip-validation', action='store_true', help='Deploy regardless of performance')
    parser.add_argument('--min-improvement', type=float, default=0.02, help='Minimum improvement to deploy')

    args = parser.parse_args()

    pipeline = Pipeline()
    results = pipeline.run(
        force_refresh=args.force_refresh,
        skip_validation=args.skip_validation,
        min_improvement=args.min_improvement
    )

    # Exit with appropriate code
    sys.exit(0 if results["status"] == "completed" else 1)


if __name__ == "__main__":
    main()
