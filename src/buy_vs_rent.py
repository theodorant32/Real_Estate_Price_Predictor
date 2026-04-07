import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class BuyVsRentInputs:
    # Purchase details
    purchase_price: float
    down_payment_pct: float = 0.20  # 20% default

    # Mortgage details
    mortgage_rate: float = 0.05  # 5% annual
    amortization_years: int = 25
    payment_frequency: str = "monthly"

    # Property details
    property_tax_rate: float = 0.003  # 0.3% of assessed value (Vancouver)
    strata_fee_monthly: float = 0  # $0 for detached, $400-800 for condos
    maintenance_rate: float = 0.01  # 1% of value per year
    home_insurance_monthly: float = 100

    # Rental income (for multi-family or mortgage helper)
    monthly_rental_income: float = 0  # Income from secondary suite, laneway house, or multi-unit
    rental_income_vacancy_rate: float = 0.05  # 5% vacancy buffer

    # Rental alternative
    monthly_rent: float = 2500
    rent_inflation_rate: float = 0.03  # 3% annual
    renters_insurance_monthly: float = 20

    # Investment opportunity cost
    investment_return_rate: float = 0.07  # 7% annual if invested instead

    # Time horizon
    time_horizon_years: int = 5

    # Appreciation forecast (from ML model)
    appreciation_rate: float = 0.03  # Annual appreciation

    # Transaction costs
    buying_agent_rate: float = 0.0  # Usually paid by seller in Canada
    selling_agent_rate: float = 0.035  # 3.5% typical in BC

    # Tax flags
    is_principal_residence: bool = True  # Capital gains tax-free
    is_first_time_buyer: bool = False  # FHSA benefits, PTT exemption
    foreign_buyer: bool = False  # Foreign buyer ban + tax

    # First-time buyer specific
    fhsa_contribution: float = 0  # FHSA amount used for down payment
    ptt_exemption_eligible: bool = False  # BC FTB PTT exemption


class BuyVsRentCalculator:
    def __init__(self):
        pass

    def calculate_bc_ptt(
        self,
        purchase_price: float,
        is_foreign_buyer: bool = False,
        is_first_time_buyer: bool = False
    ) -> float:
        # First-time buyer exemption thresholds (BC 2024)
        fttb_full_exemption_threshold = 835_000
        fttb_partial_exemption_threshold = 860_000

        # Calculate base PTT
        ptt = 0.0

        # First $200K at 1%
        ptt += min(purchase_price, 200_000) * 0.01

        # $200K to $2M at 2%
        if purchase_price > 200_000:
            ptt += (min(purchase_price, 2_000_000) - 200_000) * 0.02

        # Above $2M at 3%
        if purchase_price > 2_000_000:
            ptt += (purchase_price - 2_000_000) * 0.03

        # Apply first-time buyer exemption
        if is_first_time_buyer:
            if purchase_price <= fttb_full_exemption_threshold:
                return 0.0  # Full exemption
            elif purchase_price <= fttb_partial_exemption_threshold:
                # Partial exemption (linear phase-out)
                exemption_ratio = (
                    (fttb_partial_exemption_threshold - purchase_price)
                    / (fttb_partial_exemption_threshold - fttb_full_exemption_threshold)
                )
                ptt *= (1 - exemption_ratio)

        # Foreign buyer additional tax (20% of purchase price)
        if is_foreign_buyer:
            ptt += purchase_price * 0.20

        return ptt

    def calculate_cmhc_insurance(self, mortgage_amount: float, down_payment_pct: float) -> float:
        if down_payment_pct >= 0.20:
            return 0.0

        if down_payment_pct >= 0.15:
            rate = 0.028
        elif down_payment_pct >= 0.10:
            rate = 0.031
        else:
            rate = 0.040

        return mortgage_amount * rate

    def calculate_monthly_mortgage_payment(self, principal: float, annual_rate: float, amortization_years: int) -> float:
        if principal <= 0:
            return 0.0

        # Convert to monthly rate (Canadian semi-annual compounding)
        monthly_rate = (1 + annual_rate / 2) ** (2 / 12) - 1

        n_payments = amortization_years * 12

        if monthly_rate == 0:
            return principal / n_payments

        payment = principal * monthly_rate / (1 - (1 + monthly_rate) ** (-n_payments))

        return payment

    def run_analysis(self, inputs: BuyVsRentInputs) -> Dict:
        # ===== BUYING SCENARIO =====
        buying_results = self._analyze_buying(inputs)

        # ===== RENTING SCENARIO =====
        renting_results = self._analyze_renting(inputs)

        # ===== COMPARISON =====
        comparison = self._compare_scenarios(buying_results, renting_results, inputs)

        return {
            "inputs": {
                "purchase_price": inputs.purchase_price,
                "down_payment": inputs.purchase_price * inputs.down_payment_pct,
                "monthly_rent": inputs.monthly_rent,
                "time_horizon_years": inputs.time_horizon_years
            },
            "buying": buying_results,
            "renting": renting_results,
            "comparison": comparison,
            "recommendation": self._get_recommendation(comparison)
        }

    def _analyze_buying(self, inputs: BuyVsRentInputs) -> Dict:

        # Initial costs
        down_payment = inputs.purchase_price * inputs.down_payment_pct

        # PTT with first-time buyer exemption
        ptt = self.calculate_bc_ptt(
            inputs.purchase_price,
            inputs.foreign_buyer,
            inputs.is_first_time_buyer
        )

        # CMHC with first-time buyer discount (if applicable)
        cmhc = self.calculate_cmhc_insurance(
            inputs.purchase_price - down_payment,
            inputs.down_payment_pct
        )

        # FHSA contribution reduces effective down payment needed
        fhsa_benefit = inputs.fhsa_contribution

        # Effective initial cost (FHSA money is "free" from tax perspective)
        total_initial_cost = down_payment + ptt + cmhc - fhsa_benefit

        # Mortgage
        mortgage_principal = inputs.purchase_price - down_payment + cmhc
        monthly_payment = self.calculate_monthly_mortgage_payment(
            mortgage_principal,
            inputs.mortgage_rate,
            inputs.amortization_years
        )

        # Build year-by-year cash flow
        years = list(range(1, inputs.time_horizon_years + 1))

        cash_flows = []
        remaining_balance = mortgage_principal
        property_value = inputs.purchase_price

        for year in years:
            # Mortgage payments (annual)
            annual_mortgage = monthly_payment * 12

            # Split between interest and principal
            interest_component = remaining_balance * inputs.mortgage_rate
            principal_component = annual_mortgage - interest_component

            # Update balance (can't go negative)
            remaining_balance = max(0, remaining_balance - principal_component)

            # Property appreciation
            property_value *= (1 + inputs.appreciation_rate)

            # Annual costs
            property_tax = property_value * inputs.property_tax_rate
            strata_fees = inputs.strata_fee_monthly * 12
            maintenance = property_value * inputs.maintenance_rate
            insurance = inputs.home_insurance_monthly * 12

            # Rental income (for multi-family or mortgage helper)
            effective_rental_income = (
                inputs.monthly_rental_income * 12 *
                (1 - inputs.rental_income_vacancy_rate)
            )

            total_annual_cost = (
                annual_mortgage + property_tax + strata_fees +
                maintenance + insurance - effective_rental_income
            )

            # Equity built
            equity = property_value - remaining_balance

            cash_flows.append({
                "year": year,
                "mortgage_payment": annual_mortgage,
                "interest_paid": interest_component,
                "principal_paid": principal_component,
                "property_tax": property_tax,
                "strata_fees": strata_fees,
                "maintenance": maintenance,
                "insurance": insurance,
                "total_cost": total_annual_cost,
                "remaining_balance": remaining_balance,
                "property_value": property_value,
                "equity": equity
            })

        # Sale costs
        selling_costs = property_value * inputs.selling_agent_rate

        # Net proceeds at sale
        sale_proceeds = property_value - remaining_balance - selling_costs

        # Total costs over horizon
        total_costs = sum(cf["total_cost"] for cf in cash_flows) + total_initial_cost

        # Net worth at end
        final_net_worth = sale_proceeds

        return {
            "initial_costs": {
                "down_payment": down_payment,
                "ptt": ptt,
                "cmhc": cmhc,
                "total": total_initial_cost
            },
            "mortgage": {
                "principal": mortgage_principal,
                "monthly_payment": monthly_payment,
                "remaining_balance_at_sale": remaining_balance
            },
            "annual_costs": cash_flows,
            "sale_proceeds": {
                "property_value": property_value,
                "remaining_mortgage": remaining_balance,
                "selling_costs": selling_costs,
                "net_proceeds": sale_proceeds
            },
            "total_costs": total_costs,
            "final_net_worth": final_net_worth,
            "equity_built": equity
        }

    def _analyze_renting(self, inputs: BuyVsRentInputs) -> Dict:

        # Initial investment (down payment + closing costs saved)
        # In rental scenario, this money is invested instead
        initial_investment = (
            inputs.purchase_price * inputs.down_payment_pct
        )  # Down payment not spent

        # Build year-by-year
        years = list(range(1, inputs.time_horizon_years + 1))

        cash_flows = []
        investment_balance = initial_investment
        monthly_rent = inputs.monthly_rent

        for year in years:
            # Rent increases with inflation
            annual_rent = monthly_rent * 12
            monthly_rent *= (1 + inputs.rent_inflation_rate)

            # Investment growth
            investment_return = investment_balance * inputs.investment_return_rate
            investment_balance += investment_return

            # Annual costs
            rent_cost = annual_rent
            insurance = inputs.renters_insurance_monthly * 12

            total_annual_cost = rent_cost + insurance

            cash_flows.append({
                "year": year,
                "rent_paid": annual_rent,
                "insurance": insurance,
                "total_cost": total_annual_cost,
                "investment_return": investment_return,
                "investment_balance": investment_balance
            })

        return {
            "initial_investment": initial_investment,
            "annual_costs": cash_flows,
            "final_investment_balance": investment_balance,
            "total_rent_paid": sum(cf["rent_paid"] for cf in cash_flows),
            "final_net_worth": investment_balance
        }

    def _compare_scenarios(self, buying: Dict, renting: Dict, inputs: BuyVsRentInputs) -> Dict:

        # Net worth comparison
        buying_net_worth = buying["final_net_worth"]
        renting_net_worth = renting["final_net_worth"]

        net_worth_difference = buying_net_worth - renting_net_worth

        # Total costs comparison
        buying_total_cost = buying["total_costs"]
        renting_total_cost = renting["total_rent_paid"]

        # Break-even analysis
        if buying_total_cost > renting_total_cost:
            cost_advantage = "renting"
            cost_savings = buying_total_cost - renting_total_cost
        else:
            cost_advantage = "buying"
            cost_savings = renting_total_cost - buying_total_cost

        # Monthly cost comparison (average)
        avg_monthly_buying = buying_total_cost / (inputs.time_horizon_years * 12)
        avg_monthly_renting = renting_total_cost / (inputs.time_horizon_years * 12)

        # Find break-even year
        break_even_year = None
        for i, cf in enumerate(buying["annual_costs"]):
            buying_cumulative = sum(
                buying["annual_costs"][j]["total_cost"]
                for j in range(i + 1)
            ) + buying["initial_costs"]["total"]

            renting_cumulative = sum(
                renting["annual_costs"][j]["total_cost"]
                for j in range(i + 1)
            )

            # Compare equity vs investment
            buying_equity = cf["equity"]
            renting_investment = renting["annual_costs"][i]["investment_balance"]

            if buying_equity > renting_investment and break_even_year is None:
                break_even_year = i + 1

        return {
            "net_worth": {
                "buying": buying_net_worth,
                "renting": renting_net_worth,
                "difference": net_worth_difference,
                "winner": "buying" if net_worth_difference > 0 else "renting"
            },
            "total_costs": {
                "buying": buying_total_cost,
                "renting": renting_total_cost,
                "advantage": cost_advantage,
                "savings": cost_savings
            },
            "monthly_costs": {
                "buying_avg": avg_monthly_buying,
                "renting_avg": avg_monthly_renting
            },
            "break_even_year": break_even_year,
            "time_horizon_years": inputs.time_horizon_years
        }

    def _get_recommendation(self, comparison: Dict) -> Dict:

        net_worth_winner = comparison["net_worth"]["winner"]
        cost_advantage = comparison["total_costs"]["advantage"]
        break_even = comparison["break_even_year"]

        # Strength of recommendation
        net_worth_diff = abs(comparison["net_worth"]["difference"])
        purchase_price_context = net_worth_diff / 1000000  # Normalize to $1M

        if purchase_price_context > 0.1:
            strength = "STRONG"
        elif purchase_price_context > 0.05:
            strength = "MODERATE"
        else:
            strength = "WEAK"

        recommendation = f"{strength}_{net_worth_winner.upper()}"

        reasoning = []

        if net_worth_winner == "buying":
            reasoning.append(
                f"Buying builds ${comparison['net_worth']['difference']:,.0f} more wealth "
                f"over {comparison['time_horizon_years']} years"
            )
            if break_even:
                reasoning.append(f"Wealth advantage begins in year {break_even}")
        else:
            reasoning.append(
                f"Renting + investing builds ${abs(comparison['net_worth']['difference']):,.0f} "
                f"more wealth over {comparison['time_horizon_years']} years"
            )

        if comparison["total_costs"]["advantage"] == "buying":
            reasoning.append(
                f"Buying costs ${comparison['total_costs']['savings']:,.0f} less in total expenses"
            )
        else:
            reasoning.append(
                f"Renting costs ${comparison['total_costs']['savings']:,.0f} less in total expenses"
            )

        return {
            "recommendation": recommendation,
            "reasoning": reasoning,
            "confidence": "high" if purchase_price_context > 0.1 else "medium"
        }


def quick_analysis(
    purchase_price: float,
    monthly_rent: float,
    city: str = "Vancouver",
    property_type: str = "condo",
    time_horizon_years: int = 5,
    appreciation_rate: float = None
) -> Dict:
    # Default appreciation by city
    appreciation_defaults = {
        "Vancouver": 0.03,
        "Burnaby": 0.04,
        "Richmond": 0.025,
        "North Vancouver": 0.035,
        "Toronto": 0.03,
        "Calgary": 0.05
    }

    if appreciation_rate is None:
        appreciation_rate = appreciation_defaults.get(city, 0.03)

    # Default strata fees by property type
    strata_defaults = {
        "condo": 500,
        "townhouse": 350,
        "detached": 0,
        "multi_family": 150
    }

    # Default rental income potential by property type (monthly)
    rental_income_defaults = {
        "condo": 0,
        "townhouse": 0,
        "detached": 1500,  # Basement suite potential
        "multi_family": 2500  # Additional unit(s)
    }

    # Calculate down payment (20% default)
    down_payment = purchase_price * 0.20

    # Get current mortgage rate estimate
    mortgage_rate = 0.05  # 5% current estimate

    inputs = BuyVsRentInputs(
        purchase_price=purchase_price,
        down_payment_pct=0.20,
        mortgage_rate=mortgage_rate,
        strata_fee_monthly=strata_defaults.get(property_type, 0),
        monthly_rent=monthly_rent,
        time_horizon_years=time_horizon_years,
        appreciation_rate=appreciation_rate,
        monthly_rental_income=rental_income_defaults.get(property_type, 0)
    )

    calculator = BuyVsRentCalculator()
    results = calculator.run_analysis(inputs)

    return results


def main():

    print("=" * 60)
    print("CANADIAN BUY VS RENT CALCULATOR")
    print("=" * 60)

    # Example scenarios
    scenarios = [
        {
            "name": "Vancouver 2BR Condo",
            "purchase_price": 750_000,
            "monthly_rent": 2_600,
            "city": "Vancouver",
            "property_type": "condo",
            "time_horizon_years": 5
        },
        {
            "name": "Burnaby Townhouse",
            "purchase_price": 950_000,
            "monthly_rent": 2_850,
            "city": "Burnaby",
            "property_type": "townhouse",
            "time_horizon_years": 5
        },
        {
            "name": "Richmond Detached",
            "purchase_price": 1_400_000,
            "monthly_rent": 3_500,
            "city": "Richmond",
            "property_type": "detached",
            "time_horizon_years": 10
        },
        {
            "name": "Calgary Detached",
            "purchase_price": 650_000,
            "monthly_rent": 2_200,
            "city": "Calgary",
            "property_type": "detached",
            "time_horizon_years": 5
        }
    ]

    for scenario in scenarios:
        name = scenario.pop("name")

        print(f"\n{'=' * 60}")
        print(f"SCENARIO: {name}")
        print("=" * 60)

        results = quick_analysis(**scenario)

        print(f"\nPurchase Price: ${results['inputs']['purchase_price']:,.0f}")
        print(f"Down Payment (20%): ${results['inputs']['down_payment']:,.0f}")
        print(f"Monthly Rent: ${results['inputs']['monthly_rent']:,.0f}")
        print(f"Time Horizon: {results['inputs']['time_horizon_years']} years")

        print(f"\n--- BUYING ---")
        print(f"  Initial Costs: ${results['buying']['initial_costs']['total']:,.0f}")
        print(f"    - Down Payment: ${results['buying']['initial_costs']['down_payment']:,.0f}")
        print(f"    - BC PTT: ${results['buying']['initial_costs']['ptt']:,.0f}")
        print(f"    - CMHC Insurance: ${results['buying']['initial_costs']['cmhc']:,.0f}")
        print(f"  Monthly Mortgage: ${results['buying']['mortgage']['monthly_payment']:,.0f}")
        print(f"  Total Costs (all-in): ${results['buying']['total_costs']:,.0f}")
        print(f"  Net Worth at End: ${results['buying']['final_net_worth']:,.0f}")

        print(f"\n--- RENTING ---")
        print(f"  Initial Investment: ${results['renting']['initial_investment']:,.0f}")
        print(f"  Total Rent Paid: ${results['renting']['total_rent_paid']:,.0f}")
        print(f"  Net Worth at End: ${results['renting']['final_net_worth']:,.0f}")

        print(f"\n--- COMPARISON ---")
        comp = results["comparison"]
        print(f"  Net Worth Winner: {comp['net_worth']['winner'].upper()}")
        print(f"    Buying: ${comp['net_worth']['buying']:,.0f}")
        print(f"    Renting: ${comp['net_worth']['renting']:,.0f}")
        print(f"    Difference: ${comp['net_worth']['difference']:,.0f}")
        print(f"  Break-even Year: {comp['break_even_year'] or 'N/A'}")

        print(f"\n--- RECOMMENDATION ---")
        rec = results["recommendation"]
        print(f"  {rec['recommendation']}")
        for reason in rec["reasoning"]:
            print(f"    • {reason}")


if __name__ == "__main__":
    main()
