"""
AI Property Advisor Chatbot

Natural language interface for real estate queries.
Uses rule-based reasoning with ML-backed insights.

Example queries:
- "Where should I buy for 6% ROI?"
- "Is Vancouver or Calgary better for investment?"
- "What property type has the best rental yield?"
- "Should I buy a condo with $100k down payment?"
"""

import re
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class ChatResponse:
    answer: str
    confidence: str  # high, medium, low
    data: Dict
    follow_up_questions: List[str]


class PropertyChatbot:
    """
    Rule-based chatbot with ML-backed responses.

    Handles queries about:
    - Investment recommendations
    - City comparisons
    - Property type advice
    - Affordability calculations
    - Market insights
    """

    def __init__(self, predictor=None, roi_calculator=None, heatmap_generator=None):
        self.predictor = predictor
        self.roi_calculator = roi_calculator
        self.heatmap_generator = heatmap_generator

        # Intent patterns
        self.intent_patterns = {
            "investment_recommendation": [
                r"where should i buy",
                r"best (city|place|area) for (investment|roi|return)",
                r"recommend (me|a) (property|city)",
                r"what (city|area) has (the )?best roi",
                r"show me (good|great|best) investments?",
            ],
            "city_comparison": [
                r"(vancouver|toronto|calgary|burnaby|richmond) (vs|versus|or) (vancouver|toronto|calgary|burnaby|richmond)",
                r"compare (vancouver|toronto|calgary|burnaby|richmond) (and|with|to) (vancouver|toronto|calgary|burnaby|richmond)",
                r"which is better (vancouver|toronto|calgary|burnaby|richmond) or (vancouver|toronto|calgary|burnaby|richmond)",
            ],
            "property_type_advice": [
                r"should i buy (a )?(condo|townhouse|detached)",
                r"(condo|townhouse|detached) (vs|versus|or) (condo|townhouse|detached)",
                r"what property type (is best|should i buy|has best)",
                r"best property type for (investment|roi|rental)",
            ],
            "affordability": [
                r"how much (house|property) can i afford",
                r"what can i afford with",
                r"can i afford (a|a )?\$?(\d+)",
                r"budget (for|of) \$?(\d+)",
            ],
            "rental_yield": [
                r"(rental yield|cap rate|cash flow)",
                r"rental income (for|from)",
                r"how much rent (can i get|should i charge)",
            ],
            "market_outlook": [
                r"(market|prices) (going up|will rise|outlook|forecast|prediction)",
                r"will (prices|the market) (go up|rise|fall|crash)",
                r"is (now|this) a good time to buy",
            ],
            "roi_calculation": [
                r"(roi|return on investment|profit) (for|on) ",
                r"what's (the )?roi (for|on)",
                r"calculate (my|the) (roi|return)",
            ],
        }

        # City data cache
        self.city_data = {
            "vancouver": {"appreciation": 3.0, "rental_yield": 3.5, "risk": "low"},
            "toronto": {"appreciation": 3.0, "rental_yield": 3.8, "risk": "low"},
            "calgary": {"appreciation": 5.0, "rental_yield": 5.5, "risk": "medium"},
            "burnaby": {"appreciation": 4.0, "rental_yield": 4.0, "risk": "low"},
            "richmond": {"appreciation": 2.5, "rental_yield": 3.8, "risk": "low"},
            "north vancouver": {"appreciation": 3.5, "rental_yield": 3.5, "risk": "low"},
        }

        self.property_data = {
            "condo": {"appreciation": 2.5, "rental_yield": 4.0, "entry": "low"},
            "townhouse": {"appreciation": 3.5, "rental_yield": 3.8, "entry": "medium"},
            "detached": {"appreciation": 3.5, "rental_yield": 3.0, "entry": "high"},
            "multi_family": {"appreciation": 4.0, "rental_yield": 5.0, "entry": "high"},
        }

    def detect_intent(self, query: str) -> Tuple[str, float]:
        """Detect user intent from query."""

        query_lower = query.lower()
        best_intent = "unknown"
        best_confidence = 0.0

        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                match = re.search(pattern, query_lower)
                if match:
                    # Calculate confidence based on match quality
                    confidence = min(1.0, len(match.group(0)) / len(query_lower) * 1.5)
                    if confidence > best_confidence:
                        best_intent = intent
                        best_confidence = confidence

        return best_intent, best_confidence

    def extract_entities(self, query: str) -> Dict:
        """Extract entities from query."""

        entities = {
            "cities": [],
            "property_types": [],
            "numbers": [],
            "roi_target": None,
        }

        query_lower = query.lower()

        # Extract cities
        for city in self.city_data.keys():
            if city in query_lower:
                entities["cities"].append(city.title())

        # Extract property types
        for ptype in self.property_data.keys():
            if ptype in query_lower:
                entities["property_types"].append(ptype)

        # Extract numbers (budget, ROI target, etc.)
        number_patterns = [
            r"\$(\d+(?:,\d{3})*(?:\.\d{2})?)",  # $1,000,000 or $50000
            r"(\d+(?:,\d{3})*(?:\.\d{2})?)\s*(?:percent|%)",  # 6% or 6 percent
            r"(\d+)%",  # Just 6%
            r"(\d{2,3})\s*(?:percent|%)",  # 6 percent
        ]

        for pattern in number_patterns:
            matches = re.findall(pattern, query_lower)
            for match in matches:
                num_str = match.replace(",", "")
                try:
                    num = float(num_str)
                    entities["numbers"].append(num)

                    # Check if it's an ROI target (small numbers like 6, 8, 10)
                    if 1 <= num <= 20 and "%" in query or "percent" in query or "roi" in query_lower:
                        entities["roi_target"] = num
                except ValueError:
                    pass

        return entities

    def respond(self, query: str, user_context: Dict = None) -> ChatResponse:
        """Generate response to user query."""

        intent, confidence = self.detect_intent(query)
        entities = self.extract_entities(query)

        # Route to appropriate handler
        handlers = {
            "investment_recommendation": self._handle_investment_recommendation,
            "city_comparison": self._handle_city_comparison,
            "property_type_advice": self._handle_property_type_advice,
            "affordability": self._handle_affordability,
            "rental_yield": self._handle_rental_yield,
            "market_outlook": self._handle_market_outlook,
            "roi_calculation": self._handle_roi_calculation,
            "unknown": self._handle_unknown,
        }

        handler = handlers.get(intent, self._handle_unknown)
        response = handler(entities, user_context or {})

        return ChatResponse(
            answer=response["answer"],
            confidence="high" if confidence > 0.7 else "medium" if confidence > 0.4 else "low",
            data=response.get("data", {}),
            follow_up_questions=response.get("follow_up", [])
        )

    def _handle_investment_recommendation(self, entities: Dict, context: Dict) -> Dict:
        """Handle investment recommendation queries."""

        roi_target = entities.get("roi_target", 6.0)

        # Find cities matching criteria
        matching_cities = []
        for city, data in self.city_data.items():
            if data["appreciation"] >= roi_target - 2:  # Within 2% of target
                matching_cities.append({
                    "city": city.title(),
                    "appreciation": data["appreciation"],
                    "rental_yield": data["rental_yield"],
                    "risk": data["risk"]
                })

        # Sort by appreciation
        matching_cities.sort(key=lambda x: x["appreciation"], reverse=True)

        if matching_cities:
            top_city = matching_cities[0]
            answer = (
                f"Based on your criteria for {roi_target}%+ ROI, I recommend looking at **{top_city['city']}**.\n\n"
                f"**Why {top_city['city']}?**\n"
                f"- Predicted appreciation: {top_city['appreciation']}%\n"
                f"- Average rental yield: {top_city['rental_yield']}%\n"
                f"- Risk level: {top_city['risk'].title()}\n\n"
            )

            if len(matching_cities) > 1:
                answer += f"**Alternative:** {matching_cities[1]['city']} also shows strong potential at {matching_cities[1]['appreciation']}% predicted growth."

            follow_up = [
                f"What's the best property type in {top_city['city']}?",
                f"Show me specific properties in {top_city['city']}",
                "Calculate my affordability for this market"
            ]
        else:
            answer = (
                f"Finding markets with {roi_target}%+ ROI is challenging in current conditions. "
                f"Consider adjusting your target or exploring emerging neighborhoods.\n\n"
                f"**Top alternatives:**\n"
                f"- Calgary: 5.0% predicted appreciation, 5.5% rental yield\n"
                f"- Burnaby: 4.0% predicted appreciation, 4.0% rental yield"
            )
            follow_up = [
                "Show me Calgary properties",
                "What's a realistic ROI target?",
                "How can I improve my investment returns?"
            ]

        return {"answer": answer, "follow_up": follow_up}

    def _handle_city_comparison(self, entities: Dict, context: Dict) -> Dict:
        """Handle city vs city comparison queries."""

        cities = entities.get("cities", [])

        if len(cities) < 2:
            # Default comparison
            cities = ["Vancouver", "Calgary"]

        city1, city2 = cities[0].lower(), cities[1].lower()

        data1 = self.city_data.get(city1, {"appreciation": 3.0, "rental_yield": 3.5, "risk": "medium"})
        data2 = self.city_data.get(city2, {"appreciation": 3.0, "rental_yield": 3.5, "risk": "medium"})

        # Determine winner
        score1 = data1["appreciation"] * 0.5 + data1["rental_yield"] * 0.3 + (10 - {"low": 0, "medium": 5, "high": 10}.get(data1["risk"], 5)) * 0.2
        score2 = data2["appreciation"] * 0.5 + data2["rental_yield"] * 0.3 + (10 - {"low": 0, "medium": 5, "high": 10}.get(data2["risk"], 5)) * 0.2

        if score1 > score2:
            winner = cities[0]
            margin = score1 - score2
        else:
            winner = cities[1]
            margin = score2 - score1

        answer = (
            f"## {cities[0]} vs {cities[1]}\n\n"
            f"**Verdict: {winner}** (by {margin:.1f} points)\n\n"
            f"### {cities[0]}\n"
            f"- Predicted appreciation: {data1['appreciation']}%\n"
            f"- Rental yield: {data1['rental_yield']}%\n"
            f"- Risk: {data1['risk'].title()}\n\n"
            f"### {cities[1]}\n"
            f"- Predicted appreciation: {data2['appreciation']}%\n"
            f"- Rental yield: {data2['rental_yield']}%\n"
            f"- Risk: {data2['risk'].title()}\n\n"
        )

        if data1["appreciation"] > data2["appreciation"]:
            answer += f"**For appreciation:** {cities[0]} wins ({data1['appreciation']}% vs {data2['appreciation']}%)\n"
        else:
            answer += f"**For appreciation:** {cities[1]} wins ({data2['appreciation']}% vs {data1['appreciation']}%)\n"

        if data1["rental_yield"] > data2["rental_yield"]:
            answer += f"**For rental income:** {cities[0]} wins ({data1['rental_yield']}% vs {data2['rental_yield']}%)\n"
        else:
            answer += f"**For rental income:** {cities[1]} wins ({data2['rental_yield']}% vs {data1['rental_yield']}%)\n"

        follow_up = [
            f"Show me properties in {winner}",
            f"What's the average price in {winner}?",
            f"Calculate ROI for {winner}"
        ]

        return {"answer": answer, "follow_up": follow_up}

    def _handle_property_type_advice(self, entities: Dict, context: Dict) -> Dict:
        """Handle property type advice queries."""

        property_types = entities.get("property_types", list(self.property_data.keys()))

        # Score each property type
        scored = []
        for ptype in property_types:
            data = self.property_data.get(ptype, {"appreciation": 3.0, "rental_yield": 3.5, "entry": "medium"})
            score = data["appreciation"] * 0.4 + data["rental_yield"] * 0.4 + (10 if data["entry"] == "low" else 5 if data["entry"] == "medium" else 0) * 0.2
            scored.append({
                "type": ptype,
                "score": score,
                **data
            })

        scored.sort(key=lambda x: x["score"], reverse=True)
        winner = scored[0]

        answer = (
            f"## Best Property Type Recommendation\n\n"
            f"**Winner: {winner['type'].title()}**\n\n"
            f"### Comparison:\n\n"
        )

        for item in scored:
            answer += f"**{item['type'].title()}**\n"
            answer += f"- Appreciation: {item['appreciation']}%\n"
            answer += f"- Rental Yield: {item['rental_yield']}%\n"
            answer += f"- Entry Point: {item['entry'].title()}\n\n"

        # Property-specific advice
        advice = {
            "condo": "Best for first-time buyers and investors seeking lower entry costs with decent rental income.",
            "townhouse": "Great balance of space, affordability, and appreciation potential.",
            "detached": "Best long-term appreciation but requires significant capital. Land value drives growth.",
            "multi_family": "Highest rental income potential. Ideal for experienced investors seeking cash flow."
        }

        answer += f"**Advice:** {advice.get(winner['type'], '')}"

        follow_up = [
            f"Show me {winner['type']} properties",
            "What's the average price for this type?",
            "Calculate mortgage for this property type"
        ]

        return {"answer": answer, "follow_up": follow_up}

    def _handle_affordability(self, entities: Dict, context: Dict) -> Dict:
        """Handle affordability queries."""

        # Extract budget from entities
        budget = None
        for num in entities.get("numbers", []):
            if num > 50000:  # Likely a budget
                budget = num
                break

        if budget is None:
            budget = context.get("budget", 750000)

        # Calculate affordability based on income if available
        income = context.get("income", 100000)
        down_payment = context.get("down_payment", 150000)

        # CMHC guidelines: GDS <= 32%, TDS <= 40%
        max_monthly_gds = income * 0.32 / 12
        max_monthly_tds = income * 0.40 / 12

        # Estimate max mortgage (at 5% stress test)
        stress_rate = 0.0525
        monthly_rate = (1 + stress_rate / 2) ** (2 / 12) - 1
        amortization = 25 * 12

        max_mortgage_gds = max_monthly_gds * (1 - (1 + monthly_rate) ** (-amortization)) / monthly_rate
        max_mortgage_tds = max_monthly_tds * (1 - (1 + monthly_rate) ** (-amortization)) / monthly_rate

        max_mortgage = min(max_mortgage_gds, max_mortgage_tds)
        max_price = max_mortgage + down_payment

        answer = (
            f"## Affordability Analysis\n\n"
            f"Based on your profile:\n\n"
            f"- **Annual Income:** ${income:,.0f}\n"
            f"- **Down Payment:** ${down_payment:,.0f}\n\n"
            f"### You Can Afford:\n\n"
            f"**Maximum Purchase Price: ~${max_price:,.0f}**\n\n"
            f"Breakdown:\n"
            f"- Max Mortgage: ${max_mortgage:,.0f}\n"
            f"- Down Payment: ${down_payment:,.0f}\n\n"
            f"### Monthly Payment Estimate:\n"
            f"- Mortgage (5%): ~${max_monthly_gds * 0.8:,.0f}/month\n"
            f"- Property Tax: ~${max_price * 0.003 / 12:,.0f}/month\n"
            f"- Total: ~${max_monthly_gds * 0.8 + max_price * 0.003 / 12:,.0f}/month\n\n"
        )

        if budget and budget < max_price:
            answer += f"Your budget of ${budget:,.0f} is **within your affordability range**."
        elif budget and budget > max_price:
            answer += f"Your budget of ${budget:,.0f} **exceeds your calculated affordability**. Consider adjusting expectations or increasing income/down payment."

        follow_up = [
            "Show me properties in my budget",
            "How can I increase my borrowing power?",
            "Calculate my exact mortgage payment"
        ]

        return {"answer": answer, "follow_up": follow_up}

    def _handle_rental_yield(self, entities: Dict, context: Dict) -> Dict:
        """Handle rental yield queries."""

        answer = (
            f"## Rental Yield Analysis\n\n"
            f"**Rental yield** is calculated as: (Annual Rent / Property Price) × 100\n\n"
            f"### Average Yields by City:\n\n"
        )

        for city, data in sorted(self.city_data.items(), key=lambda x: x[1]["rental_yield"], reverse=True):
            answer += f"- **{city.title()}:** {data['rental_yield']}%\n"

        answer += "\n### Average Yields by Property Type:\n\n"

        for ptype, data in sorted(self.property_data.items(), key=lambda x: x[1]["rental_yield"], reverse=True):
            answer += f"- **{ptype.title()}:** {data['rental_yield']}%\n"

        answer += "\n**Key Insights:**\n"
        answer += "- Calgary offers the highest rental yields in Canada\n"
        answer += "- Multi-family properties generate the best cash flow\n"
        answer += "- Condos offer good entry-level yields with lower capital requirements\n\n"
        answer += "**Target:** Aim for 4%+ gross yield for positive cash flow investments."

        follow_up = [
            "Calculate yield for a specific property",
            "What's a good cap rate?",
            "Show me high-yield properties"
        ]

        return {"answer": answer, "follow_up": follow_up}

    def _handle_market_outlook(self, entities: Dict, context: Dict) -> Dict:
        """Handle market outlook/forecast queries."""

        answer = (
            f"## Market Outlook\n\n"
            f"### Current Market Conditions:\n\n"
            f"**Interest Rates:** Bank of Canada overnight rate at 5.0% (as of early 2025)\n"
            f"- 5-year fixed mortgage rates: ~5.0-5.5%\n"
            f"- Stress test rate: 5.25%\n\n"
            f"### Price Forecasts (12-month):\n\n"
        )

        for city, data in self.city_data.items():
            answer += f"- **{city.title()}:** +{data['appreciation']}% predicted\n"

        answer += "\n### Market Regime:\n\n"
        answer += "The Canadian real estate market is in a **transition phase**:\n\n"
        answer += "- **Vancouver/Toronto:** Stabilizing after rate hikes, low inventory supporting prices\n"
        answer += "- **Calgary:** Strong growth driven by interprovincial migration and affordability\n"
        answer += "- **Overall:** Balanced market conditions with selective opportunities\n\n"
        answer += "**Is now a good time to buy?**\n\n"
        answer += "For long-term holders (5+ years): **Yes** - time in market beats timing market\n"
        answer += "For short-term flippers: **Caution** - appreciation may be modest near-term\n\n"
        answer += "**Strategy:** Focus on cash-flowing properties in growth markets like Calgary."

        follow_up = [
            "Will prices go up in 2025?",
            "Is there a risk of a market crash?",
            "Best markets to buy right now"
        ]

        return {"answer": answer, "follow_up": follow_up}

    def _handle_roi_calculation(self, entities: Dict, context: Dict) -> Dict:
        """Handle ROI calculation queries."""

        price = context.get("price", 750000)
        down_payment = context.get("down_payment", 150000)
        monthly_rent = context.get("monthly_rent", 2600)

        # Calculate ROI
        annual_rent = monthly_rent * 12
        gross_yield = (annual_rent / price) * 100

        # Estimate expenses (~30% of rent)
        expenses = annual_rent * 0.30
        noi = annual_rent - expenses

        cap_rate = (noi / price) * 100

        # Cash-on-cash
        mortgage = price - down_payment
        annual_mortgage = mortgage * 0.05  # ~5% interest
        cash_flow = noi - annual_mortgage

        coc_return = (cash_flow / down_payment) * 100 if down_payment > 0 else 0

        # Total ROI (including appreciation)
        appreciation = 3.0  # Default assumption
        principal_paydown = annual_mortgage * 0.3  # ~30% goes to principal early on
        total_return = cash_flow + (price * appreciation / 100) + principal_paydown
        total_roi = (total_return / down_payment) * 100 if down_payment > 0 else 0

        answer = (
            f"## ROI Calculation\n\n"
            f"### Property Details:\n"
            f"- Purchase Price: ${price:,.0f}\n"
            f"- Down Payment: ${down_payment:,.0f}\n"
            f"- Monthly Rent: ${monthly_rent:,.0f}\n\n"
            f"### Returns:\n\n"
            f"- **Gross Rental Yield:** {gross_yield:.1f}%\n"
            f"- **Cap Rate (Net Yield):** {cap_rate:.1f}%\n"
            f"- **Cash-on-Cash Return:** {coc_return:.1f}%\n"
            f"- **Total ROI (incl. appreciation):** {total_roi:.1f}%\n\n"
            f"### Annual Cash Flow:\n"
            f"- Rental Income: ${annual_rent:,.0f}\n"
            f"- Expenses (~30%): ${expenses:,.0f}\n"
            f"- Net Operating Income: ${noi:,.0f}\n"
            f"- Mortgage Interest: ${annual_mortgage:,.0f}\n"
            f"- **Cash Flow: ${cash_flow:,.0f}**\n"
        )

        if cash_flow > 0:
            answer += "\n✅ **Positive cash flow property**"
        else:
            answer += "\n⚠️ **Negative cash flow** - banking on appreciation"

        follow_up = [
            "How can I improve the ROI?",
            "What if I put 20% down instead?",
            "Compare with a different property"
        ]

        return {"answer": answer, "follow_up": follow_up}

    def _handle_unknown(self, entities: Dict, context: Dict) -> Dict:
        """Handle unknown/unrecognized queries."""

        answer = (
            "I'm here to help with Canadian real estate investment questions!\n\n"
            "**I can help you with:**\n\n"
            "1. **Investment Recommendations** - 'Where should I buy for 6% ROI?'\n"
            "2. **City Comparisons** - 'Vancouver vs Calgary for investment'\n"
            "3. **Property Type Advice** - 'Should I buy a condo or townhouse?'\n"
            "4. **Affordability** - 'How much can I afford with $100k income?'\n"
            "5. **Rental Yield** - 'What's a good cap rate?'\n"
            "6. **Market Outlook** - 'Will prices go up in 2025?'\n"
            "7. **ROI Calculations** - 'Calculate ROI for a $750k property'\n\n"
            "**Try asking:**\n"
            "- 'Where should I buy for the best ROI?'\n"
            "- 'Compare Vancouver and Calgary'\n"
            "- 'What property type has the best rental yield?'"
        )

        follow_up = [
            "Where should I buy for 6% ROI?",
            "Vancouver vs Calgary - which is better?",
            "How much property can I afford?"
        ]

        return {"answer": answer, "follow_up": follow_up}


def main():
    print("=" * 70)
    print("PROPERTY CHATBOT DEMO")
    print("=" * 70)

    bot = PropertyChatbot()

    test_queries = [
        "Where should I buy for 6% ROI?",
        "Vancouver vs Calgary for investment",
        "Should I buy a condo or townhouse?",
        "How much house can I afford with $100k income?",
        "What's a good rental yield?",
        "Is now a good time to buy?",
        "Calculate ROI for a $750k property",
        "Show me the best investment properties",
    ]

    for query in test_queries:
        print(f"\n📝 Query: {query}")
        print("-" * 50)
        response = bot.respond(query)
        print(f"💬 Response:\n{response.answer}")
        print(f"\n🔍 Confidence: {response.confidence}")
        print(f"💡 Follow-ups: {response.follow_up_questions}")


if __name__ == "__main__":
    main()
