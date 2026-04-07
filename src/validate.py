import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime


class ValidationReport:
    def __init__(self, model_dir: str = "models", data_dir: str = "data/processed"):
        self.model_dir = Path(model_dir)
        self.data_dir = Path(data_dir)

    def load_model_metrics(self, model_name: str = "price_predictor") -> Optional[Dict]:
        metrics_path = self.model_dir / f"{model_name}_metrics.json"

        if not metrics_path.exists():
            return None

        with open(metrics_path, "r") as f:
            return json.load(f)

    def check_data_leakage(self, df: pd.DataFrame) -> Dict:
        issues = []
        warnings = []

        # Columns that are EXPECTED (not leakage)
        expected_cols = {
            "target_price_6m", "target_return_6m",  # These ARE the target
            "benchmark_price",  # Current price - should be excluded from features but not leakage
        }

        # Check for obvious leakage columns (excluding expected ones)
        leakage_patterns = [
            "target_", "future_", "next_", "lead_", "forward_"
        ]

        for col in df.columns:
            if col in expected_cols:
                continue

            col_lower = col.lower()
            for leak_pattern in leakage_patterns:
                if leak_pattern in col_lower and "lag" not in col_lower:
                    issues.append(f"Potential leakage column: '{col}'")

        # Check for high correlations with target
        # High correlation alone isn't leakage - it could be a good predictive feature
        # We only warn about suspiciously high correlations (>0.99) with non-lag features
        target_col = "target_price_6m"
        if target_col in df.columns:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if col == target_col or col in expected_cols:
                    continue

                corr = df[[col, target_col]].dropna().corr().iloc[0, 1]
                # Only warn about very high correlations with non-lag features
                if abs(corr) > 0.99 and "lag" not in col.lower() and "rolling" not in col.lower():
                    warnings.append(
                        f"Very high correlation with target: '{col}' (r={corr:.3f}) - verify no leakage"
                    )

        return {
            "passed": len(issues) == 0,
            "issues": issues,
            "warnings": warnings,
            "status": "PASS" if len(issues) == 0 else "FAIL"
        }

    def check_overfitting(self, metrics: Dict) -> Dict:
        issues = []
        warnings = []

        cv_results = metrics.get("cv_results", {})
        holdout_results = metrics.get("holdout_results", {})
        overfit_check = metrics.get("overfitting_check", {})

        if not cv_results or not holdout_results:
            return {
                "status": "UNKNOWN",
                "message": "Holdout metrics not available"
            }

        cv_rmse = cv_results.get("avg_rmse", 0)
        holdout_rmse = holdout_results.get("overall", {}).get("rmse", 0)

        cv_mape = cv_results.get("avg_mape", 0)
        holdout_mape = holdout_results.get("overall", {}).get("mape", 0)

        cv_r2 = cv_results.get("avg_r2", 0)
        holdout_r2 = holdout_results.get("overall", {}).get("r2", 0)

        # Check degradation
        if cv_rmse > 0:
            rmse_degradation = (holdout_rmse - cv_rmse) / cv_rmse
            if rmse_degradation > 0.5:
                issues.append(f"Severe RMSE degradation: {rmse_degradation:.1%}")
            elif rmse_degradation > 0.2:
                warnings.append(f"Moderate RMSE degradation: {rmse_degradation:.1%}")

        if cv_mape > 0:
            mape_degradation = (holdout_mape - cv_mape) / cv_mape
            if mape_degradation > 0.5:
                issues.append(f"Severe MAPE degradation: {mape_degradation:.1%}")
            elif mape_degradation > 0.2:
                warnings.append(f"Moderate MAPE degradation: {mape_degradation:.1%}")

        if cv_r2 > 0:
            r2_drop = cv_r2 - holdout_r2
            if r2_drop > 0.3:
                issues.append(f"Severe R² drop: {r2_drop:.2f}")
            elif r2_drop > 0.1:
                warnings.append(f"Moderate R² drop: {r2_drop:.2f}")

        # Check model complexity vs data size
        model_params = metrics.get("model_params", {})
        n_samples = metrics.get("n_samples", 0)
        n_features = metrics.get("n_features", 0)

        if n_samples > 0 and n_features > 0:
            samples_per_feature = n_samples / n_features
            if samples_per_feature < 10:
                issues.append(
                    f"Too few samples per feature: {samples_per_feature:.1f} (need >10)"
                )
            elif samples_per_feature < 50:
                warnings.append(
                    f"Low samples per feature: {samples_per_feature:.1f} (prefer >50)"
                )

        return {
            "passed": len(issues) == 0,
            "issues": issues,
            "warnings": warnings,
            "status": "PASS" if len(issues) == 0 else "FAIL",
            "cv_rmse": cv_rmse,
            "holdout_rmse": holdout_rmse,
            "cv_r2": cv_r2,
            "holdout_r2": holdout_r2
        }

    def check_regularization(self, metrics: Dict) -> Dict:
        issues = []
        warnings = []

        model_params = metrics.get("model_params", {})

        # Check for regularization parameters
        if "reg_alpha" not in model_params or model_params.get("reg_alpha", 0) == 0:
            warnings.append("No L1 regularization (reg_alpha=0)")

        if "reg_lambda" not in model_params or model_params.get("reg_lambda", 0) == 0:
            warnings.append("No L2 regularization (reg_lambda=0)")

        # Check model complexity
        max_depth = model_params.get("max_depth", 6)
        if max_depth > 8:
            warnings.append(f"High max_depth ({max_depth}) may overfit")

        if model_params.get("subsample", 1.0) > 0.9:
            warnings.append("High subsample - consider reducing for regularization")

        if model_params.get("colsample_bytree", 1.0) > 0.9:
            warnings.append("High colsample_bytree - consider reducing")

        # Check learning rate and estimators
        learning_rate = model_params.get("learning_rate", 0.1)
        n_estimators = model_params.get("n_estimators", 100)

        if learning_rate > 0.1 and n_estimators < 100:
            warnings.append("High learning rate with few estimators may underfit")

        if learning_rate < 0.01 and n_estimators > 1000:
            warnings.append("Very low learning rate - training may be slow")

        return {
            "passed": len(issues) == 0,
            "issues": issues,
            "warnings": warnings,
            "status": "PASS" if len(issues) == 0 else "WARN",
            "regularization_params": {
                "reg_alpha": model_params.get("reg_alpha", 0),
                "reg_lambda": model_params.get("reg_lambda", 0),
                "max_depth": max_depth,
                "subsample": model_params.get("subsample", 1.0),
                "colsample_bytree": model_params.get("colsample_bytree", 1.0)
            }
        }

    def check_metric_robustness(self, metrics: Dict) -> Dict:
        issues = []
        warnings = []

        cv_results = metrics.get("cv_results", {})

        # Check for multiple metrics
        has_rmse = "avg_rmse" in cv_results
        has_mape = "avg_mape" in cv_results
        has_r2 = "avg_r2" in cv_results
        has_directional = cv_results.get("avg_directional_accuracy") is not None

        if not has_rmse:
            issues.append("Missing RMSE metric")

        if not has_mape and not has_r2:
            warnings.append("Consider adding relative metrics (MAPE or R²)")

        if not has_directional:
            warnings.append("Consider tracking directional accuracy for price predictions")

        # Check fold consistency
        fold_metrics = cv_results.get("fold_metrics", [])
        if len(fold_metrics) > 1:
            fold_rmses = [f.get("rmse", 0) for f in fold_metrics]
            fold_cv = np.std(fold_rmses) / (np.mean(fold_rmses) + 1e-8)

            if fold_cv > 0.3:
                warnings.append(f"High fold variance (CV={fold_cv:.2f}) - unstable model")

        # Check for city/property breakdown
        if not cv_results.get("rmse_by_city"):
            warnings.append("No per-city error breakdown")

        if not cv_results.get("rmse_by_property_type"):
            warnings.append("No per-property-type error breakdown")

        return {
            "passed": len(issues) == 0,
            "issues": issues,
            "warnings": warnings,
            "status": "PASS" if len(issues) == 0 else "WARN",
            "metrics_tracked": {
                "rmse": has_rmse,
                "mape": has_mape,
                "r2": has_r2,
                "directional_accuracy": has_directional
            }
        }

    def check_data_sufficiency(self, metrics: Dict) -> Dict:
        issues = []
        warnings = []

        n_samples = metrics.get("n_samples", 0)
        n_features = metrics.get("n_features", 0)
        data_quality = metrics.get("data_quality", {})

        # Minimum samples
        if n_samples < 100:
            issues.append(f"Too few samples: {n_samples} (need >100)")
        elif n_samples < 500:
            warnings.append(f"Limited samples: {n_samples} (prefer >500)")

        # Samples per feature
        if n_features > 0:
            ratio = n_samples / n_features
            if ratio < 10:
                issues.append(f"Too few samples per feature: {ratio:.1f}")
            elif ratio < 50:
                warnings.append(f"Low samples per feature: {ratio:.1f}")

        # Check data quality warnings
        quality_warnings = data_quality.get("warnings", [])
        for w in quality_warnings:
            warnings.append(f"Data quality: {w}")

        quality_issues = data_quality.get("issues", [])
        for i in quality_issues:
            issues.append(f"Data quality: {i}")

        return {
            "passed": len(issues) == 0,
            "issues": issues,
            "warnings": warnings,
            "status": "PASS" if len(issues) == 0 else "FAIL",
            "n_samples": n_samples,
            "n_features": n_features
        }

    def check_real_world_validity(self, metrics: Dict) -> Dict:
        issues = []
        warnings = []

        holdout = metrics.get("holdout_results")

        if not holdout:
            issues.append("No holdout test set evaluation")
        else:
            holdout_rmse = holdout.get("overall", {}).get("rmse")
            holdout_n = holdout.get("n_test_samples", 0)

            if holdout_n < 50:
                warnings.append(f"Small holdout set: {holdout_n} samples")

            # Check if holdout metrics are reasonable
            if holdout_rmse:
                # Assuming house prices, RMSE should be < 20% of typical price
                # This is a rough heuristic
                pass  # Would need context about typical prices

        # Check for temporal validation
        cv_results = metrics.get("cv_results", {})
        fold_metrics = cv_results.get("fold_metrics", [])

        if len(fold_metrics) < 3:
            warnings.append("Few CV folds - may not capture temporal patterns")

        # Check for embargo in CV
        model_params = metrics.get("model_params", {})
        # We can't directly check embargo, but we can note if CV was used
        if not cv_results:
            issues.append("No cross-validation performed")

        return {
            "passed": len(issues) == 0,
            "issues": issues,
            "warnings": warnings,
            "status": "PASS" if len(issues) == 0 else "FAIL",
            "has_holdout": holdout is not None,
            "has_cv": bool(cv_results)
        }

    def generate_report(
        self,
        df: pd.DataFrame = None,
        model_name: str = "price_predictor"
    ) -> Dict:
        report = {
            "model_name": model_name,
            "generated_at": datetime.now().isoformat(),
            "checks": {},
            "summary": {}
        }

        # Load metrics
        metrics = self.load_model_metrics(model_name)

        if not metrics:
            report["summary"]["status"] = "FAIL"
            report["summary"]["message"] = f"Model '{model_name}' not found"
            return report

        # Run all checks
        if df is not None:
            report["checks"]["data_leakage"] = self.check_data_leakage(df)

        report["checks"]["overfitting"] = self.check_overfitting(metrics)
        report["checks"]["regularization"] = self.check_regularization(metrics)
        report["checks"]["metric_robustness"] = self.check_metric_robustness(metrics)
        report["checks"]["data_sufficiency"] = self.check_data_sufficiency(metrics)
        report["checks"]["real_world_validity"] = self.check_real_world_validity(metrics)

        # Calculate summary
        all_checks = list(report["checks"].values())
        passed = sum(1 for c in all_checks if c.get("passed", False))
        failed = sum(1 for c in all_checks if c.get("status") == "FAIL")
        warnings = sum(
            len(c.get("warnings", [])) + len(c.get("issues", []))
            for c in all_checks
        )

        if failed > 0:
            overall_status = "FAIL"
        elif passed == len(all_checks):
            overall_status = "PASS"
        else:
            overall_status = "WARN"

        report["summary"] = {
            "status": overall_status,
            "checks_passed": f"{passed}/{len(all_checks)}",
            "total_issues": failed,
            "total_warnings": warnings
        }

        return report

    def print_report(self, report: Dict):
        print("=" * 70)
        print(f"MODEL VALIDATION REPORT: {report['model_name']}")
        print(f"Generated: {report['generated_at']}")
        print("=" * 70)

        summary = report.get("summary", {})
        print(f"\nOVERALL STATUS: {summary.get('status', 'UNKNOWN')}")
        print(f"Checks Passed: {summary.get('checks_passed', 'N/A')}")
        print(f"Issues: {summary.get('total_issues', 0)}")
        print(f"Warnings: {summary.get('total_warnings', 0)}")

        for check_name, check_result in report.get("checks", {}).items():
            status = check_result.get("status", "UNKNOWN")
            status_icon = {"PASS": "[OK]", "FAIL": "[FAIL]", "WARN": "[WARN]"}.get(status, "[?]")

            print(f"\n{status_icon} {check_name.upper().replace('_', ' ')}")

            for issue in check_result.get("issues", []):
                print(f"    ISSUE: {issue}")

            for warning in check_result.get("warnings", []):
                print(f"    WARNING: {warning}")

        print("\n" + "=" * 70)


def main():
    validator = ValidationReport()

    # Try to load featured data for leakage check
    featured_path = Path("data/processed/featured_data.csv")
    df = None
    if featured_path.exists():
        df = pd.read_csv(featured_path, parse_dates=["date"])
        print(f"Loaded featured data: {len(df)} rows")

    # Generate report
    report = validator.generate_report(df)
    validator.print_report(report)

    return report


if __name__ == "__main__":
    main()
