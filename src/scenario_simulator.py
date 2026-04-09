"""
Investment Scenario Simulator

Multi-year wealth projection with:
- Mortgage amortization
- Property appreciation
- Rental income growth
- Tax implications
- Opportunity cost analysis
- Monte Carlo uncertainty modeling

Shows 5/10/15/20 year outcomes with interactive scenarios.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from scipy import stats


@dataclass
class ScenarioInputs:
    # Property
    purchase_price: float
    down_payment_pct: float = 0.20
    property_type: str = "condo"
    city: str = "Vancouver"

    # Mortgage
    mortgage_rate: float = 0.05
    amortization_years: int = 25
    payment_frequency: str = "monthly"

    # Costs
    property_tax_rate: float = 0.003
    strata_fee_monthly: float = 0
    insurance_monthly: float = 100
    maintenance_rate: float = 0.01

    # Rental (if applicable)
    monthly_rent: float = 0  # Rental income
    rent_inflation_rate: float = 0.03

    # Investment alternative
    investment_return_rate: float = 0.07  # If money invested elsewhere

    # Scenarios
    appreciation_rate: float = 0.03
    appreciation_std: float = 0.02  # Volatility for Monte Carlo

    # Time
    time_horizon_years: int = 10

    # Tax
    marginal_tax_rate: float = 0.30
    capital_gains_inclusion: float = 0.50  # 50% in Canada


class ScenarioSimulator:
    def __init__(self, n_simulations: int = 1000):
        self.n_simulations = n_simulations

    def run_base_scenario(self, inputs: ScenarioInputs) -> Dict:
        """Run deterministic base case scenario."""

        years = list(range(1, inputs.time_horizon_years + 1))
        n_years = len(years)

        # Initial values
        down_payment = inputs.purchase_price * inputs.down_payment_pct
        mortgage_principal = inputs.purchase_price - down_payment

        # Calculate monthly mortgage (Canadian semi-annual compounding)
        monthly_rate = (1 + inputs.mortgage_rate / 2) ** (2 / 12) - 1
        n_payments = inputs.amortization_years * 12
        monthly_mortgage = (
            mortgage_principal * monthly_rate /
            (1 - (1 + monthly_rate) ** (-n_payments))
        )

        # Initialize arrays
        property_values = np.zeros(n_years + 1)
        mortgage_balances = np.zeros(n_years + 1)
        equity = np.zeros(n_years + 1)
        annual_cash_flows = np.zeros(n_years)
        cumulative_cash_flows = np.zeros(n_years)

        property_values[0] = inputs.purchase_price
        mortgage_balances[0] = mortgage_principal
        equity[0] = down_payment

        running_cash = -down_payment  # Initial cash outflow
        current_rent = inputs.monthly_rent * 12

        for i, year in enumerate(years):
            idx = i

            # Property appreciation
            property_values[idx + 1] = property_values[idx] * (1 + inputs.appreciation_rate)

            # Mortgage payments
            annual_mortgage = monthly_mortgage * 12
            interest_portion = mortgage_balances[idx] * inputs.mortgage_rate
            principal_portion = annual_mortgage - interest_portion
            mortgage_balances[idx + 1] = max(0, mortgage_balances[idx] - principal_portion)

            # Equity
            equity[idx + 1] = property_values[idx + 1] - mortgage_balances[idx + 1]

            # Annual costs
            property_tax = property_values[idx] * inputs.property_tax_rate
            strata = inputs.strata_fee_monthly * 12
            insurance = inputs.insurance_monthly * 12
            maintenance = inputs.purchase_price * inputs.maintenance_rate

            # Rental income (if applicable)
            rental_income = current_rent * (1 - 0.05)  # 5% vacancy buffer

            # Net cash flow
            if inputs.monthly_rent > 0:
                # Investment property
                annual_cash_flows[idx] = rental_income - annual_mortgage - property_tax - strata - insurance - maintenance
            else:
                # Owner-occupied (imputed rent benefit not counted)
                annual_cash_flows[idx] = -annual_mortgage - property_tax - strata - insurance - maintenance

            cumulative_cash_flows[idx] = running_cash
            running_cash += annual_cash_flows[idx]
            current_rent *= (1 + inputs.rent_inflation_rate)

        # Final sale scenario
        sale_price = property_values[-1]
        remaining_mortgage = mortgage_balances[-1]
        selling_costs = sale_price * 0.035  # Real estate fees ~3.5%

        # Capital gains tax (if investment property)
        if inputs.monthly_rent > 0:
            capital_gain = sale_price - inputs.purchase_price
            taxable_gain = capital_gain * inputs.capital_gains_inclusion
            capital_gains_tax = taxable_gain * inputs.marginal_tax_rate
        else:
            capital_gains_tax = 0  # Principal residence exemption

        net_sale_proceeds = sale_price - remaining_mortgage - selling_costs - capital_gains_tax

        # Total cash invested
        initial_cash = down_payment + (inputs.purchase_price * 0.02)  # Closing costs ~2%

        # IRR calculation
        all_cash_flows = [-initial_cash] + list(annual_cash_flows)
        all_cash_flows[-1] += net_sale_proceeds

        try:
            irr = np.irr(all_cash_flows)
        except:
            irr = 0.0

        # CAGR (handle negative/zero final wealth)
        final_wealth = net_sale_proceeds + sum(annual_cash_flows)
        if final_wealth > 0 and initial_cash > 0:
            cagr = (final_wealth / initial_cash) ** (1 / inputs.time_horizon_years) - 1
        else:
            cagr = 0.0

        return {
            "property_values": property_values.tolist(),
            "mortgage_balances": mortgage_balances.tolist(),
            "equity": equity.tolist(),
            "annual_cash_flows": annual_cash_flows.tolist(),
            "cumulative_cash_flows": cumulative_cash_flows.tolist(),
            "final_property_value": property_values[-1],
            "final_equity": equity[-1],
            "net_sale_proceeds": net_sale_proceeds,
            "total_cash_invested": initial_cash,
            "total_return": final_wealth - initial_cash,
            "irr": irr,
            "cagr": cagr,
            "years": [0] + years
        }

    def run_monte_carlo(self, inputs: ScenarioInputs) -> Dict:
        """Run Monte Carlo simulation for uncertainty analysis."""

        final_values = []
        cagrs = []
        irr_values = []

        for _ in range(self.n_simulations):
            # Sample appreciation rate from normal distribution
            sampled_appreciation = np.random.normal(
                inputs.appreciation_rate,
                inputs.appreciation_std
            )
            sampled_appreciation = np.clip(sampled_appreciation, -0.10, 0.15)

            # Create modified inputs
            modified_inputs = ScenarioInputs(
                purchase_price=inputs.purchase_price,
                down_payment_pct=inputs.down_payment_pct,
                mortgage_rate=inputs.mortgage_rate,
                amortization_years=inputs.amortization_years,
                property_tax_rate=inputs.property_tax_rate,
                strata_fee_monthly=inputs.strata_fee_monthly,
                insurance_monthly=inputs.insurance_monthly,
                maintenance_rate=inputs.maintenance_rate,
                monthly_rent=inputs.monthly_rent,
                rent_inflation_rate=inputs.rent_inflation_rate,
                investment_return_rate=inputs.investment_return_rate,
                appreciation_rate=sampled_appreciation,
                time_horizon_years=inputs.time_horizon_years,
                marginal_tax_rate=inputs.marginal_tax_rate,
                capital_gains_inclusion=inputs.capital_gains_inclusion
            )

            # Run scenario
            result = self.run_base_scenario(modified_inputs)
            final_values.append(result["final_property_value"])
            cagrs.append(result["cagr"])
            irr_values.append(result["irr"])

        # Calculate statistics
        final_values = np.array(final_values)
        cagrs = np.array(cagrs)
        irr_values = np.array(irr_values)

        return {
            "final_value_mean": np.mean(final_values),
            "final_value_median": np.median(final_values),
            "final_value_std": np.std(final_values),
            "final_value_p5": np.percentile(final_values, 5),
            "final_value_p95": np.percentile(final_values, 95),
            "cagr_mean": np.mean(cagrs),
            "cagr_std": np.std(cagrs),
            "cagr_p5": np.percentile(cagrs, 5),
            "cagr_p95": np.percentile(cagrs, 95),
            "irr_mean": np.mean(irr_values),
            "irr_median": np.median(irr_values),
            "probability_positive_return": np.mean(final_values > 0),
            "probability_beat_5pct": np.mean(cagrs > 0.05),
            "probability_beat_10pct": np.mean(cagrs > 0.10),
        }

    def compare_scenarios(
        self,
        inputs: ScenarioInputs,
        scenarios: List[Dict]
    ) -> pd.DataFrame:
        """Compare multiple what-if scenarios."""

        results = []

        for scenario in scenarios:
            scenario_inputs = ScenarioInputs(
                purchase_price=inputs.purchase_price,
                down_payment_pct=inputs.down_payment_pct,
                mortgage_rate=scenario.get("mortgage_rate", inputs.mortgage_rate),
                amortization_years=inputs.amortization_years,
                property_tax_rate=inputs.property_tax_rate,
                strata_fee_monthly=inputs.strata_fee_monthly,
                insurance_monthly=inputs.insurance_monthly,
                maintenance_rate=inputs.maintenance_rate,
                monthly_rent=inputs.monthly_rent,
                rent_inflation_rate=inputs.rent_inflation_rate,
                investment_return_rate=inputs.investment_return_rate,
                appreciation_rate=scenario.get("appreciation_rate", inputs.appreciation_rate),
                time_horizon_years=inputs.time_horizon_years,
                marginal_tax_rate=inputs.marginal_tax_rate,
                capital_gains_inclusion=inputs.capital_gains_inclusion
            )

            result = self.run_base_scenario(scenario_inputs)
            monte_carlo = self.run_monte_carlo(scenario_inputs)

            results.append({
                "Scenario": scenario.get("name", "Base"),
                "Appreciation": scenario.get("appreciation_rate", inputs.appreciation_rate) * 100,
                "Mortgage Rate": scenario.get("mortgage_rate", inputs.mortgage_rate) * 100,
                "Final Value": result["final_property_value"],
                "Net Proceeds": result["net_sale_proceeds"],
                "CAGR": result["cagr"] * 100,
                "IRR": result["irr"] * 100,
                "CAGR (P5)": monte_carlo["cagr_p5"] * 100,
                "CAGR (P95)": monte_carlo["cagr_p95"] * 100,
                "Prob >5%": monte_carlo["probability_beat_5pct"] * 100,
            })

        return pd.DataFrame(results)

    def get_risk_metrics(self, inputs: ScenarioInputs, monte_carlo_result: Dict) -> Dict:
        """Calculate risk metrics."""

        # Value at Risk (VaR)
        var_95 = monte_carlo_result["final_value_p5"]
        expected_shortfall = var_95 * 0.9  # Approximate

        # Volatility
        volatility = monte_carlo_result["final_value_std"] / monte_carlo_result["final_value_mean"]

        # Sharpe-like ratio (excess return / volatility)
        risk_free = 0.03  # Assume 3% risk-free rate
        excess_return = monte_carlo_result["cagr_mean"] - risk_free
        sharpe = excess_return / volatility if volatility > 0 else 0

        # Risk score (0-100, lower is riskier)
        risk_score = 100 * (1 - volatility) * (1 + sharpe) / 2
        risk_score = max(0, min(100, risk_score))

        return {
            "var_95": var_95,
            "expected_shortfall": expected_shortfall,
            "volatility": volatility,
            "sharpe_ratio": sharpe,
            "risk_score": risk_score,
            "risk_level": "Low" if risk_score > 70 else "Medium" if risk_score > 40 else "High"
        }


def quick_scenario_analysis(
    purchase_price: float,
    down_payment_pct: float = 0.20,
    monthly_rent: float = 0,
    city: str = "Vancouver",
    time_horizon_years: int = 10
) -> Dict:
    """Quick scenario analysis with city-specific defaults."""

    # City-specific defaults
    city_defaults = {
        "Vancouver": {"appreciation": 0.03, "strata": {"condo": 500, "townhouse": 350}},
        "Toronto": {"appreciation": 0.03, "strata": {"condo": 600, "townhouse": 400}},
        "Calgary": {"appreciation": 0.05, "strata": {"condo": 350, "townhouse": 250}},
        "Burnaby": {"appreciation": 0.04, "strata": {"condo": 450, "townhouse": 300}},
    }

    defaults = city_defaults.get(city, {"appreciation": 0.03, "strata": {}})

    inputs = ScenarioInputs(
        purchase_price=purchase_price,
        down_payment_pct=down_payment_pct,
        monthly_rent=monthly_rent,
        appreciation_rate=defaults["appreciation"],
        time_horizon_years=time_horizon_years
    )

    simulator = ScenarioSimulator(n_simulations=500)
    base_result = simulator.run_base_scenario(inputs)
    monte_carlo = simulator.run_monte_carlo(inputs)
    risk_metrics = simulator.get_risk_metrics(inputs, monte_carlo)

    # What-if scenarios
    scenarios_to_compare = [
        {"name": "Base Case", "appreciation_rate": defaults["appreciation"], "mortgage_rate": 0.05},
        {"name": "High Growth", "appreciation_rate": defaults["appreciation"] * 1.5, "mortgage_rate": 0.05},
        {"name": "Low Growth", "appreciation_rate": defaults["appreciation"] * 0.5, "mortgage_rate": 0.05},
        {"name": "High Rates", "appreciation_rate": defaults["appreciation"] * 0.7, "mortgage_rate": 0.07},
        {"name": "Recession", "appreciation_rate": -0.02, "mortgage_rate": 0.04},
    ]

    comparison_df = simulator.compare_scenarios(inputs, scenarios_to_compare)

    return {
        "base_case": base_result,
        "monte_carlo": monte_carlo,
        "risk_metrics": risk_metrics,
        "scenario_comparison": comparison_df,
        "inputs": inputs
    }


def main():
    print("=" * 70)
    print("INVESTMENT SCENARIO SIMULATOR")
    print("=" * 70)

    # Example property
    results = quick_scenario_analysis(
        purchase_price=750000,
        down_payment_pct=0.20,
        monthly_rent=2600,
        city="Vancouver",
        time_horizon_years=10
    )

    base = results["base_case"]
    mc = results["monte_carlo"]
    risk = results["risk_metrics"]

    print(f"\n--- BASE CASE PROJECTION (10 Years) ---")
    print(f"  Purchase Price: ${results['inputs'].purchase_price:,.0f}")
    print(f"  Down Payment: ${results['inputs'].purchase_price * results['inputs'].down_payment_pct:,.0f}")
    print(f"  Final Property Value: ${base['final_property_value']:,.0f}")
    print(f"  Net Sale Proceeds: ${base['net_sale_proceeds']:,.0f}")
    print(f"  Total Return: ${base['total_return']:,.0f}")
    print(f"  CAGR: {base['cagr']*100:.2f}%")
    print(f"  IRR: {base['irr']*100:.2f}%")

    print(f"\n--- MONTE CARLO ANALYSIS ({results['monte_carlo']['n_simulations'] if 'n_simulations' in results['monte_carlo'] else 500} simulations) ---")
    print(f"  Expected Final Value: ${mc['final_value_mean']:,.0f}")
    print(f"  Median Final Value: ${mc['final_value_median']:,.0f}")
    print(f"  5th Percentile: ${mc['final_value_p5']:,.0f}")
    print(f"  95th Percentile: ${mc['final_value_p95']:,.0f}")
    print(f"  Expected CAGR: {mc['cagr_mean']*100:.2f}% (±{mc['cagr_std']*100:.1f}%)")
    print(f"  Probability of >5% CAGR: {mc['probability_beat_5pct']*100:.1f}%")
    print(f"  Probability of >10% CAGR: {mc['probability_beat_10pct']*100:.1f}%")

    print(f"\n--- RISK METRICS ---")
    print(f"  Risk Level: {risk['risk_level']}")
    print(f"  Risk Score: {risk['risk_score']:.1f}/100")
    print(f"  Volatility: {risk['volatility']*100:.1f}%")
    print(f"  Sharpe Ratio: {risk['sharpe_ratio']:.2f}")
    print(f"  VaR (95%): ${risk['var_95']:,.0f}")

    print(f"\n--- SCENARIO COMPARISON ---")
    print(results["scenario_comparison"].to_string(index=False))


if __name__ == "__main__":
    main()
