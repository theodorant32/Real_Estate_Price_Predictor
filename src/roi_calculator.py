"""
Rental Yield and ROI Calculator for Canadian Real Estate

Calculates key investment metrics:
- Gross Rental Yield
- Net Rental Yield (Cap Rate)
- Cash-on-Cash Return
- Total ROI (including appreciation)
- DSCR (Debt Service Coverage Ratio)
"""

from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np


@dataclass
class PropertyInputs:
    purchase_price: float
    monthly_rent: float
    down_payment_pct: float = 0.20
    mortgage_rate: float = 0.05
    amortization_years: int = 25

    # Costs
    property_tax_rate: float = 0.003  # 0.3% of value
    strata_fee_monthly: float = 0
    insurance_monthly: float = 100
    maintenance_rate: float = 0.01  # 1% of value annually
    vacancy_rate: float = 0.05  # 5% vacancy buffer
    property_management_rate: float = 0.08  # 8% of rent (optional)

    # Appreciation
    appreciation_rate: float = 0.03

    # Time horizon
    time_horizon_years: int = 5


class ROICalculator:
    def __init__(self):
        pass

    def calculate_mortgage_payment(self, principal: float, annual_rate: float, amortization_years: int) -> float:
        """Calculate monthly mortgage payment (Canadian semi-annual compounding)."""
        if principal <= 0:
            return 0.0

        monthly_rate = (1 + annual_rate / 2) ** (2 / 12) - 1
        n_payments = amortization_years * 12

        if monthly_rate == 0:
            return principal / n_payments

        payment = principal * monthly_rate / (1 - (1 + monthly_rate) ** (-n_payments))
        return payment

    def calculate_all_metrics(self, inputs: PropertyInputs) -> Dict:
        """Calculate comprehensive rental investment metrics."""

        # ===== BASIC METRICS =====
        purchase_price = inputs.purchase_price
        monthly_rent = inputs.monthly_rent
        annual_rent = monthly_rent * 12

        down_payment = purchase_price * inputs.down_payment_pct
        mortgage_amount = purchase_price - down_payment

        # ===== RENTAL YIELD =====
        # Gross Rental Yield = Annual Rent / Purchase Price
        gross_yield = (annual_rent / purchase_price) * 100

        # ===== OPERATING EXPENSES =====
        property_tax = purchase_price * inputs.property_tax_rate
        strata_fees = inputs.strata_fee_monthly * 12
        insurance = inputs.insurance_monthly * 12
        maintenance = purchase_price * inputs.maintenance_rate
        vacancy_loss = annual_rent * inputs.vacancy_rate
        property_management = annual_rent * inputs.property_management_rate

        total_operating_expenses = (
            property_tax + strata_fees + insurance +
            maintenance + vacancy_loss + property_management
        )

        # ===== NET OPERATING INCOME (NOI) =====
        effective_gross_income = annual_rent - vacancy_loss
        noi = effective_gross_income - (
            property_tax + strata_fees + insurance +
            maintenance + property_management
        )

        # ===== CAP RATE (Net Rental Yield) =====
        # Cap Rate = NOI / Purchase Price
        cap_rate = (noi / purchase_price) * 100

        # ===== CASH FLOW =====
        monthly_mortgage = self.calculate_mortgage_payment(
            mortgage_amount, inputs.mortgage_rate, inputs.amortization_years
        )
        annual_mortgage = monthly_mortgage * 12

        # Annual cash flow (before taxes)
        annual_cash_flow = noi - annual_mortgage
        monthly_cash_flow = annual_cash_flow / 12

        # ===== CASH-ON-CASH RETURN =====
        # CoC = Annual Cash Flow / Total Cash Invested
        closing_costs = purchase_price * 0.02  # ~2% for legal, inspection, etc.
        total_cash_invested = down_payment + closing_costs

        cash_on_cash_return = (annual_cash_flow / total_cash_invested) * 100 if total_cash_invested > 0 else 0

        # ===== DEBT SERVICE COVERAGE RATIO (DSCR) =====
        # DSCR = NOI / Annual Debt Service
        annual_debt_service = annual_mortgage
        dscr = noi / annual_debt_service if annual_debt_service > 0 else float('inf')

        # ===== TOTAL ROI (including appreciation + principal paydown) =====
        # Over time horizon
        appreciation = purchase_price * inputs.appreciation_rate

        # Principal paid in first year
        first_year_interest = mortgage_amount * inputs.mortgage_rate
        first_year_principal = annual_mortgage - first_year_interest

        # Total annual return = Cash flow + Appreciation + Principal paydown
        total_annual_return = annual_cash_flow + appreciation + first_year_principal
        total_roi = (total_annual_return / total_cash_invested) * 100 if total_cash_invested > 0 else 0

        # ===== 1% RULE CHECK =====
        # Rule of thumb: Monthly rent should be >= 1% of purchase price
        one_percent_rule = (monthly_rent / purchase_price) * 100
        passes_one_percent = one_percent_rule >= 0.8  # More realistic for Canada

        # ===== AFFORDABILITY METRICS =====
        price_to_rent_ratio = purchase_price / annual_rent
        # < 15 = buy, 15-20 = neutral, > 20 = rent (US standard)
        # For Canada, adjust: < 20 = buy, 20-25 = neutral, > 25 = rent

        return {
            "purchase_price": purchase_price,
            "down_payment": down_payment,
            "mortgage_amount": mortgage_amount,
            "monthly_mortgage_payment": monthly_mortgage,

            # Yield metrics
            "gross_yield": gross_yield,
            "cap_rate": cap_rate,
            "cash_on_cash_return": cash_on_cash_return,
            "total_roi": total_roi,

            # Cash flow
            "annual_rent": annual_rent,
            "effective_gross_income": effective_gross_income,
            "noi": noi,
            "annual_operating_expenses": total_operating_expenses,
            "annual_mortgage": annual_mortgage,
            "annual_cash_flow": annual_cash_flow,
            "monthly_cash_flow": monthly_cash_flow,

            # Risk metrics
            "dscr": dscr,
            "vacancy_loss": vacancy_loss,
            "passes_one_percent_rule": passes_one_percent,
            "one_percent_ratio": one_percent_rule,

            # Price metrics
            "price_to_rent_ratio": price_to_rent_ratio,

            # Appreciation
            "annual_appreciation": appreciation,
            "first_year_principal_paydown": first_year_principal,

            # Cash invested
            "total_cash_invested": total_cash_invested,
            "closing_costs": closing_costs,
        }

    def get_investment_grade(self, metrics: Dict) -> str:
        """Assign investment grade based on metrics."""

        score = 0

        # Cash-on-Cash scoring (max 30 points)
        coc = metrics["cash_on_cash_return"]
        if coc >= 8:
            score += 30
        elif coc >= 5:
            score += 20
        elif coc >= 2:
            score += 10
        elif coc >= 0:
            score += 5

        # Cap rate scoring (max 25 points)
        cap = metrics["cap_rate"]
        if cap >= 5:
            score += 25
        elif cap >= 3.5:
            score += 18
        elif cap >= 2.5:
            score += 10
        elif cap >= 1.5:
            score += 5

        # Cash flow scoring (max 25 points)
        if metrics["annual_cash_flow"] > 0:
            score += 25
        elif metrics["annual_cash_flow"] > -5000:
            score += 15
        elif metrics["annual_cash_flow"] > -10000:
            score += 5

        # DSCR scoring (max 20 points)
        dscr = metrics["dscr"]
        if dscr >= 1.25:
            score += 20
        elif dscr >= 1.0:
            score += 15
        elif dscr >= 0.8:
            score += 5

        # Convert score to grade
        if score >= 80:
            return "A - Excellent Investment"
        elif score >= 65:
            return "B - Good Investment"
        elif score >= 50:
            return "C - Average Investment"
        elif score >= 35:
            return "D - Below Average"
        else:
            return "F - Poor Investment"

    def get_recommendation(self, metrics: Dict) -> Dict:
        """Generate investment recommendation."""

        grade = self.get_investment_grade(metrics)
        coc = metrics["cash_on_cash_return"]
        cash_flow = metrics["annual_cash_flow"]

        reasoning = []

        # Cash flow insight
        if cash_flow > 0:
            reasoning.append(f"Positive cash flow of ${cash_flow:,.0f}/year")
        else:
            reasoning.append(f"Negative cash flow of ${cash_flow:,.0f}/year (common in high-appreciation markets)")

        # Yield insight
        if metrics["cap_rate"] >= 4:
            reasoning.append(f"Strong cap rate of {metrics['cap_rate']:.1f}%")
        elif metrics["cap_rate"] >= 2.5:
            reasoning.append(f"Moderate cap rate of {metrics['cap_rate']:.1f}%")
        else:
            reasoning.append(f"Low cap rate of {metrics['cap_rate']:.1f}% - banking on appreciation")

        # Cash-on-cash insight
        if coc >= 5:
            reasoning.append(f"Excellent cash-on-cash return of {coc:.1f}%")
        elif coc >= 0:
            reasoning.append(f"Positive cash-on-cash return of {coc:.1f}%")
        else:
            reasoning.append(f"Negative cash-on-cash return of {coc:.1f}%")

        # DSCR insight
        dscr = metrics["dscr"]
        if dscr >= 1.25:
            reasoning.append("Strong DSCR - comfortable mortgage coverage")
        elif dscr >= 1.0:
            reasoning.append("Adequate DSCR - NOI covers debt service")
        else:
            reasoning.append(f"Weak DSCR ({dscr:.2f}) - requires additional income")

        # 1% rule
        if metrics["passes_one_percent_rule"]:
            reasoning.append("Passes 1% rule screening")

        # Final recommendation
        if grade.startswith("A") or grade.startswith("B"):
            recommendation = "BUY"
            confidence = "High" if grade.startswith("A") else "Medium"
        elif grade.startswith("C"):
            recommendation = "HOLD"
            confidence = "Medium"
        else:
            recommendation = "AVOID"
            confidence = "High"

        return {
            "grade": grade,
            "recommendation": recommendation,
            "confidence": confidence,
            "reasoning": reasoning,
            "score_breakdown": {
                "cash_on_cash": coc,
                "cap_rate": metrics["cap_rate"],
                "monthly_cash_flow": metrics["monthly_cash_flow"],
                "dscr": dscr
            }
        }


def quick_roi_analysis(
    purchase_price: float,
    monthly_rent: float,
    down_payment_pct: float = 0.20,
    city: str = None
) -> Dict:
    """Quick ROI analysis with city-specific defaults."""

    # City-specific appreciation estimates
    appreciation_rates = {
        "Vancouver": 0.03,
        "Toronto": 0.03,
        "Calgary": 0.05,
        "Burnaby": 0.04,
        "Richmond": 0.025,
        "North Vancouver": 0.035
    }

    # City-specific strata estimates
    strata_estimates = {
        "Vancouver": {"condo": 550, "townhouse": 350, "detached": 0},
        "Toronto": {"condo": 650, "townhouse": 400, "detached": 0},
        "Calgary": {"condo": 350, "townhouse": 250, "detached": 0}
    }

    appreciation = appreciation_rates.get(city, 0.03)

    inputs = PropertyInputs(
        purchase_price=purchase_price,
        monthly_rent=monthly_rent,
        down_payment_pct=down_payment_pct,
        appreciation_rate=appreciation
    )

    calculator = ROICalculator()
    metrics = calculator.calculate_all_metrics(inputs)
    recommendation = calculator.get_recommendation(metrics)

    return {
        "metrics": metrics,
        "recommendation": recommendation
    }


def main():
    print("=" * 70)
    print("RENTAL YIELD & ROI CALCULATOR")
    print("=" * 70)

    # Test scenarios
    scenarios = [
        {
            "name": "Vancouver 2BR Condo",
            "purchase_price": 750000,
            "monthly_rent": 2600,
            "city": "Vancouver"
        },
        {
            "name": "Calgary Detached",
            "purchase_price": 650000,
            "monthly_rent": 2400,
            "city": "Calgary"
        },
        {
            "name": "Toronto Condo",
            "purchase_price": 700000,
            "monthly_rent": 2500,
            "city": "Toronto"
        }
    ]

    for scenario in scenarios:
        name = scenario.pop("name")
        print(f"\n{'=' * 70}")
        print(f"PROPERTY: {name}")
        print("=" * 70)

        results = quick_roi_analysis(**scenario)
        metrics = results["metrics"]
        rec = results["recommendation"]

        print(f"\n--- INVESTMENT GRADE: {rec['grade']} ---")
        print(f"RECOMMENDATION: {rec['recommendation']} (Confidence: {rec['confidence']})")

        print(f"\n--- YIELD METRICS ---")
        print(f"  Gross Yield: {metrics['gross_yield']:.2f}%")
        print(f"  Cap Rate: {metrics['cap_rate']:.2f}%")
        print(f"  Cash-on-Cash Return: {metrics['cash_on_cash_return']:.2f}%")
        print(f"  Total ROI: {metrics['total_roi']:.2f}%")

        print(f"\n--- CASH FLOW ---")
        print(f"  Monthly Rent: ${metrics['annual_rent']/12:,.0f}")
        print(f"  Monthly Mortgage: ${metrics['monthly_mortgage_payment']:,.0f}")
        print(f"  Monthly Cash Flow: ${metrics['monthly_cash_flow']:,.0f}")
        print(f"  Annual Cash Flow: ${metrics['annual_cash_flow']:,.0f}")

        print(f"\n--- RISK METRICS ---")
        print(f"  DSCR: {metrics['dscr']:.2f}")
        print(f"  Passes 1% Rule: {'Yes' if metrics['passes_one_percent_rule'] else 'No'}")
        print(f"  Price-to-Rent Ratio: {metrics['price_to_rent_ratio']:.1f}")

        print(f"\n--- REASONING ---")
        for reason in rec["reasoning"]:
            print(f"  • {reason}")


if __name__ == "__main__":
    main()
