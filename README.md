# Propra

> **AI-Driven Canadian Real Estate Platform**: Price predictions, buy vs. rent analysis, market heatmaps, and investment recommendations powered by machine learning.

**Live Demo**: [https://real-estate-production-cd36.up.railway.app](https://real-estate-production-cd36.up.railway.app)

---

## The Problem

I'm 23, trying to figure out if I'll ever be able to afford a home in Vancouver. Every time I check Realtor.ca, another property jumps by $100K. My savings can't keep up.

Instead of doom-scrolling listings, I built a tool that answers the questions actually keeping me up at night:

- Will this property be worth more in a year?
- Should I buy or keep renting?
- Are there any neighborhoods I can actually afford?
- What happens if interest rates go up another 2%?

---

## What It Does

| Feature | Description |
|---------|-------------|
| 🔮 **Price Predictor** | ML forecasts for 6/12/18 months with confidence intervals |
| 🗺️ **Market Heatmap** | Interactive map showing investment opportunities across Canada |
| 🤖 **AI Chatbot** | Ask natural language questions ("Where should I buy for 6% ROI?") |
| 💰 **ROI Calculator** | Cap rate, cash-on-cash, DSCR, and investment grade (A-F) |
| 📈 **Scenario Simulator** | Monte Carlo projections with 1000 simulations |
| 🎯 **Personalized Recs** | Recommendations based on investor persona |
| 💎 **Hidden Gems** | AI detection of undervalued properties |
| 📚 **Case Studies** | Real investment analysis with historical performance |
| 🧠 **Neighborhood Graph** | Walkability/transit scoring via NetworkX |

---

## Quick Start

```bash
# Clone the repo
git clone https://github.com/theodorant32/Real_Estate_Price_Predictor.git
cd Real_Estate_Price_Predictor

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

---

## Usage Examples

### Price Prediction

```python
from src.predict import PricePredictor

predictor = PricePredictor()
predictor.load_model()

pred = predictor.predict_price_change(
    current_price=750000,
    city="Vancouver",
    property_type="condo",
    horizon_months=12
)

print(f"Predicted: ${pred['predicted_price_6m']:,.0f}")
print(f"Change: {pred['predicted_change_pct']:+.1f}%")
```

### ROI Analysis

```python
from src.roi_calculator import ROICalculator, PropertyInputs

inputs = PropertyInputs(
    purchase_price=750000,
    monthly_rent=2600,
    down_payment_pct=0.20
)

calc = ROICalculator()
metrics = calc.calculate_all_metrics(inputs)

print(f"Cap Rate: {metrics['cap_rate']:.2f}%")
print(f"Cash-on-Cash: {metrics['cash_on_cash_return']:.2f}%")
print(f"Grade: {calc.get_investment_grade(metrics)}")
```

### Neighborhood Analysis

```python
from src.neighborhood_graph import create_sample_graph

graph = create_sample_graph()
scores = graph.compute_property_scores()

for prop_id, prop_scores in scores.items():
    print(f"{prop_id}: Walkability={prop_scores['walkability_score']}")
```

---

## Data Sources

| Source | Data Type | Access |
|--------|-----------|--------|
| GVR | MLS benchmark prices (sold comps) | Web scrape / fallback |
| Bank of Canada | Interest rates | API v2 |
| CMHC | Rental market survey | Fallback (login required) |
| RateHub | Mortgage rates | Web scrape |

---

## Model Details

### Architecture

```
Ensemble Stack:
├── XGBoost (gradient boosting)
├── LightGBM (fast gradient boosting)  
├── Ridge (linear baseline)
└── Gradient Boosting (sklearn)

Meta-Learner: Ridge Regression
```

### Performance

| Metric | Value |
|--------|-------|
| CV RMSE | ~$91,000 |
| CV MAPE | ~5.6% |
| Holdout R² | ~0.987 |

### Features

- 62 engineered features
- Price momentum (3/6/12 month)
- Rental yield calculations
- Market regime indicators (hot/warm/cooling/cold)
- Neighborhood graph centrality scores
- Amenity proximity weights

### Explainability

SHAP (SHapley Additive exPlanations) integration:
- Global feature importance
- Individual prediction explanations
- Force/waterfall plots
- Dependence plots

---

## Automated Pipeline

Weekly retraining via GitHub Actions:
- **Sunday 2 AM UTC**: Full model retraining
- **Daily 3 AM UTC**: Data refresh

```bash
# Manual pipeline run
python src/pipeline.py --force-refresh
```

---

## Deployment

### Railway

1. Go to https://railway.app/new
2. Select "Deploy from GitHub repo"
3. Choose this repo
4. Click **Deploy**

### Docker

```bash
docker build -t propra .
docker run -p 8501:8501 propra
```

---

## Roadmap

- [ ] Scrape actual listings (not just benchmark data)
- [ ] Image analysis for listing photos (is that kitchen modern or just well-lit?)
- [ ] MLS API integration if access granted
- [ ] Expand to Toronto, Montreal, Ottawa
- [ ] Property image scoring with CNN
- [ ] Sentiment analysis from news/social media

---

## License

MIT License - use it, break it, learn from it.

---

## Disclaimer

This platform provides informational analysis only and does not constitute financial advice. Real estate investments carry risks. The model can be wrong. The data can be stale. If you're making a 7-figure decision, talk to someone who gets paid to know this stuff.

---

*Built during a quarter-life crisis by someone who checks Realtor.ca more often than Instagram.*
