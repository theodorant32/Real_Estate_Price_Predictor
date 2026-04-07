import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from buy_vs_rent import BuyVsRentCalculator, BuyVsRentInputs, quick_analysis
from predict import PricePredictor, MarketAnalyzer


@dataclass
class BuyerProfile:
    # Income
    annual_income: float  # Gross household income
    available_down_payment: float  # Total savings for down payment + closing costs
    other_monthly_debt: float = 0  # Car payments, student loans, etc.

    # Preferences
    max_monthly_payment: float = None  # Optional max payment comfort level
    time_horizon_years: int = 5  # How long planning to stay
    risk_tolerance: str = "moderate"  # conservative, moderate, aggressive

    # Property preferences
    min_bedrooms: int = 1
    property_types: List[str] = None  # Options: condo, townhouse, detached, multi_family

    # First-time buyer status (affects taxes, CMHC, FHSA)
    is_first_time_buyer: bool = True  # Never owned a home in past 5 years

    # FHSA (First Home Savings Account)
    fhsa_balance: float = 0  # Current FHSA balance available for down payment

    def __post_init__(self):
        if self.property_types is None:
            self.property_types = ["condo", "townhouse", "detached", "multi_family"]

        # Default max payment to 32% of gross income (CMHC guideline)
        if self.max_monthly_payment is None:
            self.max_monthly_payment = self.annual_income * 0.32 / 12


class AffordabilityCalculator:
    def __init__(
        self,
        mortgage_rate: float = 0.05,
        stress_test_rate: float = None,
        property_tax_rate: float = 0.003,
        heating_cost_monthly: float = 150
    ):
        self.mortgage_rate = mortgage_rate
        self.stress_test_rate = stress_test_rate or max(mortgage_rate + 0.02, 0.0525)
        self.property_tax_rate = property_tax_rate
        self.heating_cost_monthly = heating_cost_monthly

    def calculate_max_purchase_price(
        self,
        profile: BuyerProfile,
        include_strata: bool = True,
        avg_strata: float = 400
    ) -> Dict:
        monthly_income = profile.annual_income / 12

        # ===== GDS CONSTRAINT (32% rule) =====
        # GDS = (Mortgage Payment + Property Tax + Heating + 50% Strata) / Gross Income
        max_gds_payment = monthly_income * 0.32

        # Subtract fixed costs to get max mortgage payment
        max_mortgage_from_gds = (
            max_gds_payment
            - (self.heating_cost_monthly)
            - (avg_strata * 0.5 if include_strata else 0)
        )

        # Calculate max mortgage amount (using stress test rate)
        max_mortgage_gds = self._mortgage_to_principal(
            max_mortgage_from_gds,
            self.stress_test_rate,
            25  # 25-year amortization
        )

        # ===== TDS CONSTRAINT (40% rule) =====
        # TDS = (Mortgage + Property Tax + Heating + Strata + Other Debt) / Gross Income
        max_tds_payment = monthly_income * 0.40

        max_mortgage_from_tds = (
            max_tds_payment
            - self.heating_cost_monthly
            - (avg_strata if include_strata else 0)
            - profile.other_monthly_debt
        )

        max_mortgage_tds = self._mortgage_to_principal(
            max_mortgage_from_tds,
            self.stress_test_rate,
            25
        )

        # Take the more restrictive constraint
        max_mortgage = min(max_mortgage_gds, max_mortgage_tds)

        # Effective down payment includes FHSA
        effective_down_payment = profile.available_down_payment + profile.fhsa_balance

        # Handle edge case: zero income means no mortgage qualification
        if max_mortgage <= 0 or max_mortgage_gds <= 0:
            # Cash purchase only - limited by down payment
            return {
                "max_purchase_price": round(effective_down_payment / 0.20, 0),
                "max_mortgage_amount": 0,
                "required_down_payment": round(effective_down_payment, 0),
                "effective_down_payment": round(effective_down_payment, 0),
                "fhsa_used": round(profile.fhsa_balance, 0),
                "ftb_ptt_exemption": profile.is_first_time_buyer,
                "estimated_monthly_payment": 0,
                "gds_limit": 0,
                "tds_limit": 0,
                "constraints": {
                    "gds_max_price": 0,
                    "tds_max_price": 0,
                    "down_payment_max_price": round(effective_down_payment / 0.20, 0)
                },
                "note": "Cash purchase only (no income to qualify for mortgage)"
            }

        # Also constrained by down payment (need 20% to avoid CMHC)
        max_purchase_by_down_payment = effective_down_payment / 0.20

        # Calculate max purchase price
        # Need to account for closing costs (PTT, legal, etc.)
        # First-time buyers save on PTT (exemption up to $835K in BC)
        if profile.is_first_time_buyer:
            ptt_estimate = 0  # Will get exemption
            closing_cost_rate = 0.015  # Just legal, inspection (no PTT)
        else:
            ptt_estimate = max_purchase_by_down_payment * 0.02  # ~2% PTT estimate
            closing_cost_rate = 0.035  # PTT + legal + inspection

        max_purchase_by_mortgage = max_mortgage / (1 - effective_down_payment / max_mortgage)

        # Take the more restrictive constraint
        max_purchase_price = min(max_purchase_by_mortgage, max_purchase_by_down_payment)

        # Calculate actual monthly payment at current rates
        actual_monthly_payment = self._principal_to_mortgage(
            max_purchase_price * 0.80,  # 20% down
            self.mortgage_rate,
            25
        )

        # FHSA benefit (tax-free contribution)
        fhsa_benefit = profile.fhsa_balance * 0.30  # Approximate tax savings

        return {
            "max_purchase_price": round(max_purchase_price, 0),
            "max_mortgage_amount": round(max_mortgage, 0),
            "required_down_payment": round(max_purchase_price * 0.20, 0),
            "effective_down_payment": round(effective_down_payment, 0),
            "fhsa_used": round(profile.fhsa_balance, 0),
            "ftb_ptt_exemption": profile.is_first_time_buyer,
            "estimated_monthly_payment": round(actual_monthly_payment, 0),
            "gds_limit": round(max_gds_payment, 0),
            "tds_limit": round(max_tds_payment, 0),
            "constraints": {
                "gds_max_price": round(max_mortgage_gds / 0.80, 0),
                "tds_max_price": round(max_mortgage_tds / 0.80, 0),
                "down_payment_max_price": round(max_purchase_by_down_payment, 0)
            }
        }

    def _mortgage_to_principal(
        self,
        monthly_payment: float,
        annual_rate: float,
        amortization_years: int
    ) -> float:
        """Convert monthly payment to mortgage principal."""
        if monthly_payment <= 0:
            return 0

        monthly_rate = (1 + annual_rate / 2) ** (2 / 12) - 1
        n_payments = amortization_years * 12

        if monthly_rate == 0:
            return monthly_payment * n_payments

        principal = monthly_payment * (1 - (1 + monthly_rate) ** (-n_payments)) / monthly_rate
        return principal

    def _principal_to_mortgage(
        self,
        principal: float,
        annual_rate: float,
        amortization_years: int
    ) -> float:
        """Convert mortgage principal to monthly payment."""
        if principal <= 0:
            return 0

        monthly_rate = (1 + annual_rate / 2) ** (2 / 12) - 1
        n_payments = amortization_years * 12

        if monthly_rate == 0:
            return principal / n_payments

        payment = principal * monthly_rate / (1 - (1 + monthly_rate) ** (-n_payments))
        return payment


class PropertyRecommender:
    def __init__(
        self,
        predictor: PricePredictor = None,
        current_prices: Dict = None
    ):
        self.predictor = predictor or PricePredictor()
        try:
            self.predictor.load_model()
            self.model_loaded = True
        except FileNotFoundError:
            self.model_loaded = False

        # Default current prices (approximate 2024-2025 values)
        self.current_prices = current_prices or {
            ("Vancouver", "condo"): 750000,
            ("Vancouver", "townhouse"): 950000,
            ("Vancouver", "detached"): 1800000,
            ("Vancouver", "multi_family"): 1400000,
            ("Burnaby", "condo"): 620000,
            ("Burnaby", "townhouse"): 850000,
            ("Burnaby", "detached"): 1500000,
            ("Burnaby", "multi_family"): 1150000,
            ("Richmond", "condo"): 580000,
            ("Richmond", "townhouse"): 800000,
            ("Richmond", "detached"): 1400000,
            ("Richmond", "multi_family"): 1100000,
            ("North Vancouver", "condo"): 650000,
            ("North Vancouver", "townhouse"): 900000,
            ("North Vancouver", "detached"): 1700000,
            ("North Vancouver", "multi_family"): 1300000,
            ("Toronto", "condo"): 700000,
            ("Toronto", "townhouse"): 900000,
            ("Toronto", "detached"): 1600000,
            ("Toronto", "multi_family"): 1200000,
            ("Calgary", "condo"): 320000,
            ("Calgary", "townhouse"): 450000,
            ("Calgary", "detached"): 650000,
            ("Calgary", "multi_family"): 550000,
        }

        self.affordability_calc = AffordabilityCalculator()

    def get_affordable_properties(
        self,
        profile: BuyerProfile
    ) -> pd.DataFrame:
        # Calculate max budget
        affordability = self.affordability_calc.calculate_max_purchase_price(profile)
        max_price = affordability["max_purchase_price"]

        # Build list of affordable options
        options = []
        for city in ["Vancouver", "Burnaby", "Richmond", "North Vancouver", "Toronto", "Calgary"]:
            for ptype in profile.property_types:
                key = (city, ptype)
                if key not in self.current_prices:
                    continue

                price = self.current_prices[key]

                # Skip if over budget
                if price > max_price * 1.10:  # Allow 10% stretch
                    continue

                # Get appreciation prediction
                if self.model_loaded:
                    pred = self.predictor.predict_price_change(
                        current_price=price,
                        city=city,
                        property_type=ptype
                    )
                    appreciation = pred["predicted_change_pct"]
                    confidence_lower = pred["confidence_lower"]
                    confidence_upper = pred["confidence_upper"]
                else:
                    # Fallback estimates
                    appreciation = self._estimate_appreciation(city, ptype)
                    confidence_lower = price * 0.92
                    confidence_upper = price * 1.08

                # Calculate affordability metrics
                down_payment = price * 0.20
                mortgage = price - down_payment
                monthly_payment = self.affordability_calc._principal_to_mortgage(
                    mortgage,
                    self.affordability_calc.mortgage_rate,
                    25
                )

                # Estimate strata by property type (multi_family has lower strata, often none)
                strata_est = {"condo": 500, "townhouse": 350, "detached": 0, "multi_family": 150}.get(ptype, 0)

                # Total monthly housing cost
                property_tax = price * 0.003 / 12
                total_monthly = monthly_payment + property_tax + strata_est

                # Payment as % of income
                payment_ratio = total_monthly / (profile.annual_income / 12)

                options.append({
                    "city": city,
                    "property_type": ptype,
                    "price": price,
                    "down_payment_20pct": down_payment,
                    "monthly_payment": monthly_payment,
                    "strata_estimate": strata_est,
                    "total_monthly_cost": total_monthly,
                    "payment_to_income_pct": payment_ratio * 100,
                    "predicted_appreciation_6m": appreciation,
                    "confidence_lower": confidence_lower,
                    "confidence_upper": confidence_upper,
                    "affordable": price <= max_price,
                    "stretch_required": price > max_price
                })

        if len(options) == 0:
            return pd.DataFrame()

        df = pd.DataFrame(options)

        # Sort by appreciation (best investment first)
        if "predicted_appreciation_6m" in df.columns:
            df = df.sort_values("predicted_appreciation_6m", ascending=False)

        return df

    def _estimate_appreciation(self, city: str, property_type: str) -> float:
        """Fallback appreciation estimates."""
        estimates = {
            ("Vancouver", "condo"): 2.0,
            ("Vancouver", "townhouse"): 4.0,
            ("Vancouver", "detached"): 3.0,
            ("Vancouver", "multi_family"): 4.5,
            ("Burnaby", "condo"): 3.0,
            ("Burnaby", "townhouse"): 5.0,
            ("Burnaby", "detached"): 4.0,
            ("Burnaby", "multi_family"): 5.5,
            ("Richmond", "condo"): 2.0,
            ("Richmond", "townhouse"): 3.0,
            ("Richmond", "detached"): 2.5,
            ("Richmond", "multi_family"): 3.5,
            ("North Vancouver", "condo"): 3.0,
            ("North Vancouver", "townhouse"): 4.0,
            ("North Vancouver", "detached"): 4.0,
            ("North Vancouver", "multi_family"): 4.5,
            ("Toronto", "condo"): 2.0,
            ("Toronto", "townhouse"): 4.0,
            ("Toronto", "detached"): 3.0,
            ("Toronto", "multi_family"): 4.5,
            ("Calgary", "condo"): 4.0,
            ("Calgary", "townhouse"): 6.0,
            ("Calgary", "detached"): 5.0,
            ("Calgary", "multi_family"): 6.5,
        }
        return estimates.get((city, property_type), 3.0)

    def compare_property_types(
        self,
        profile: BuyerProfile,
        city: str = None
    ) -> pd.DataFrame:
        affordable = self.get_affordable_properties(profile)

        if city:
            affordable = affordable[affordable["city"] == city]

        if len(affordable) == 0:
            return pd.DataFrame()

        # Summary by property type
        summary = affordable.groupby("property_type").agg({
            "price": "mean",
            "monthly_payment": "mean",
            "total_monthly_cost": "mean",
            "predicted_appreciation_6m": "mean",
            "affordable": "sum",
            "stretch_required": "sum"
        }).reset_index()

        summary.columns = [
            "Property Type",
            "Avg Price",
            "Avg Monthly Payment",
            "Avg Total Monthly",
            "Avg Appreciation (%)",
            "Affordable Options",
            "Stretch Options"
        ]

        return summary

    def get_top_recommendations(
        self,
        profile: BuyerProfile,
        n: int = 5
    ) -> List[Dict]:
        affordable = self.get_affordable_properties(profile)

        if len(affordable) == 0:
            return []

        # Calculate recommendation score
        # Higher appreciation = better
        # Lower payment ratio = better
        # Affordable = better than stretch

        max_appreciation = affordable["predicted_appreciation_6m"].max()
        min_appreciation = affordable["predicted_appreciation_6m"].min()
        max_payment_ratio = affordable["payment_to_income_pct"].max()
        min_payment_ratio = affordable["payment_to_income_pct"].min()

        def calculate_score(row):
            # Appreciation score (0-100)
            if max_appreciation == min_appreciation:
                appreciation_score = 50
            else:
                appreciation_score = (
                    (row["predicted_appreciation_6m"] - min_appreciation)
                    / (max_appreciation - min_appreciation)
                    * 100
                )

            # Payment ratio score (lower is better, 0-100)
            if max_payment_ratio == min_payment_ratio:
                payment_score = 50
            else:
                payment_score = (
                    100 - (row["payment_to_income_pct"] - min_payment_ratio)
                    / (max_payment_ratio - min_payment_ratio) * 100
                )

            # Affordability bonus
            affordable_bonus = 20 if row["affordable"] else -10

            # Risk tolerance adjustment
            if profile.risk_tolerance == "aggressive":
                appreciation_weight = 0.6
                payment_weight = 0.2
            elif profile.risk_tolerance == "conservative":
                appreciation_weight = 0.2
                payment_weight = 0.6
            else:  # moderate
                appreciation_weight = 0.4
                payment_weight = 0.4

            total_score = (
                appreciation_score * appreciation_weight
                + payment_score * payment_weight
                + affordable_bonus
            )

            return total_score

        affordable["recommendation_score"] = affordable.apply(calculate_score, axis=1)

        # Get top N
        top = affordable.nlargest(n, "recommendation_score")

        recommendations = []
        for _, row in top.iterrows():
            score = row["recommendation_score"]
            if score >= 70:
                strength = "Excellent Choice"
            elif score >= 50:
                strength = "Great Option"
            elif score >= 30:
                strength = "Worth Considering"
            else:
                strength = "Think Twice"

            recommendations.append({
                "city": row["city"],
                "property_type": row["property_type"],
                "price": row["price"],
                "monthly_payment": row["monthly_payment"],
                "total_monthly": row["total_monthly_cost"],
                "predicted_appreciation": round(row["predicted_appreciation_6m"]),
                "strength": strength,
                "affordable": row["affordable"],
                "reasoning": self._generate_reasoning(row, profile)
            })

        return recommendations

    def _generate_reasoning(self, row: pd.Series, profile: BuyerProfile) -> str:
        reasons = []

        appreciation = round(row["predicted_appreciation_6m"])
        if appreciation >= 5:
            reasons.append(f"Strong appreciation potential ({appreciation}%)")
        elif appreciation >= 3:
            reasons.append(f"Good appreciation expected ({appreciation}%)")

        # Affordability
        if row["affordable"]:
            reasons.append("Within budget")
            if row["payment_to_income_pct"] < 25:
                reasons.append("Comfortable payment-to-income ratio")
        else:
            reasons.append("Requires stretching budget")

        # Property type specific
        if row["property_type"] == "condo":
            reasons.append("Lower entry point, good for first-time buyers")
        elif row["property_type"] == "townhouse":
            reasons.append("Balance of space and affordability")
        elif row["property_type"] == "detached":
            reasons.append("Maximum land value, best long-term appreciation")
        elif row["property_type"] == "multi_family":
            reasons.append("Rental income potential from additional units")

        return "; ".join(reasons)


def get_recommendations_for_budget(
    annual_income: float,
    down_payment: float,
    other_debt: float = 0,
    risk_tolerance: str = "moderate",
    n_recommendations: int = 5
) -> Dict:
    profile = BuyerProfile(
        annual_income=annual_income,
        available_down_payment=down_payment,
        other_monthly_debt=other_debt,
        risk_tolerance=risk_tolerance
    )

    recommender = PropertyRecommender()

    # Get affordability analysis
    affordability = recommender.affordability_calc.calculate_max_purchase_price(profile)

    # Get recommendations
    recommendations = recommender.get_top_recommendations(profile, n=n_recommendations)

    # Get property type comparison
    type_comparison = recommender.compare_property_types(profile)

    return {
        "profile": {
            "annual_income": annual_income,
            "down_payment": down_payment,
            "other_monthly_debt": other_debt,
            "risk_tolerance": risk_tolerance
        },
        "affordability": affordability,
        "recommendations": recommendations,
        "property_type_summary": type_comparison.to_dict() if type_comparison is not None else None
    }


def main():

    print("=" * 70)
    print("CANADIAN PROPERTY RECOMMENDER")
    print("=" * 70)

    # Example 1: Young professional, first-time buyer
    print("\n" + "=" * 70)
    print("PROFILE 1: Young Professional (First-Time Buyer)")
    print("=" * 70)
    print("Income: $85,000/year")
    print("Down Payment: $100,000")
    print("Other Debt: $500/month (car payment)")
    print("Risk Tolerance: Moderate")

    profile1 = BuyerProfile(
        annual_income=85000,
        available_down_payment=100000,
        other_monthly_debt=500,
        risk_tolerance="moderate"
    )

    recommender = PropertyRecommender()
    affordability1 = recommender.affordability_calc.calculate_max_purchase_price(profile1)

    print(f"\n--- Affordability Analysis ---")
    print(f"Max Purchase Price: ${affordability1['max_purchase_price']:,.0f}")
    print(f"Max Mortgage: ${affordability1['max_mortgage_amount']:,.0f}")
    print(f"Required Down Payment: ${affordability1['required_down_payment']:,.0f}")
    print(f"Estimated Monthly Payment: ${affordability1['estimated_monthly_payment']:,.0f}")

    recommendations1 = recommender.get_top_recommendations(profile1, n=5)

    print(f"\n--- Top Recommendations ---")
    for i, rec in enumerate(recommendations1, 1):
        print(f"\n{i}. {rec['city']} - {rec['property_type'].capitalize()}")
        print(f"   Price: ${rec['price']:,.0f}")
        print(f"   Monthly Payment: ${rec['monthly_payment']:,.0f}")
        print(f"   Total Monthly (incl. strata): ${rec['total_monthly']:,.0f}")
        print(f"   Predicted Appreciation: {rec['predicted_appreciation']:.1f}%")
        print(f"   Recommendation: {rec['strength']}")
        print(f"   Why: {rec['reasoning']}")

    # Example 2: Dual-income family
    print("\n" + "=" * 70)
    print("PROFILE 2: Dual-Income Family")
    print("=" * 70)
    print("Income: $180,000/year")
    print("Down Payment: $400,000")
    print("Other Debt: $1,200/month (car + student loans)")
    print("Risk Tolerance: Aggressive")

    profile2 = BuyerProfile(
        annual_income=180000,
        available_down_payment=400000,
        other_monthly_debt=1200,
        risk_tolerance="aggressive"
    )

    affordability2 = recommender.affordability_calc.calculate_max_purchase_price(profile2)

    print(f"\n--- Affordability Analysis ---")
    print(f"Max Purchase Price: ${affordability2['max_purchase_price']:,.0f}")
    print(f"Max Mortgage: ${affordability2['max_mortgage_amount']:,.0f}")
    print(f"Required Down Payment: ${affordability2['required_down_payment']:,.0f}")

    recommendations2 = recommender.get_top_recommendations(profile2, n=5)

    print(f"\n--- Top Recommendations ---")
    for i, rec in enumerate(recommendations2, 1):
        print(f"\n{i}. {rec['city']} - {rec['property_type'].capitalize()}")
        print(f"   Price: ${rec['price']:,.0f}")
        print(f"   Monthly Payment: ${rec['monthly_payment']:,.0f}")
        print(f"   Total Monthly: ${rec['total_monthly']:,.0f}")
        print(f"   Predicted Appreciation: {rec['predicted_appreciation']:.1f}%")
        print(f"   Recommendation: {rec['strength']}")

    # Property type comparison
    print("\n" + "=" * 70)
    print("PROPERTY TYPE COMPARISON (Profile 2)")
    print("=" * 70)

    type_comp = recommender.compare_property_types(profile2)
    if type_comp is not None:
        print(type_comp.to_string(index=False))


if __name__ == "__main__":
    main()
