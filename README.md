# Propra - AI-Powered Canadian Real Estate Platform

**Canada's most advanced real estate investment platform** - combining machine learning predictions, interactive visualizations, graph-based neighborhood analysis, and AI-driven insights.

Think Zillow + Redfin + Reonomy, but built for Canadian markets with cutting-edge ML.

---

## 🌐 Live Demo

**Try it now:** [https://real-estate-production-cd36.up.railway.app](https://real-estate-production-cd36.up.railway.app)

*Note: First load may take 30-60 seconds as Railway spins up from free tier sleep.*

---

## ✨ Features Overview

| Feature | Description |
|---------|-------------|
| 🔮 **Price Predictor** | ML-powered forecasts with confidence intervals |
| 🗺️ **Market Heatmap** | Interactive geographic visualization |
| 🤖 **AI Chatbot** | Natural language property queries |
| 💰 **ROI Calculator** | Cap rate, cash-on-cash, DSCR metrics |
| 📈 **Scenario Simulator** | Monte Carlo wealth projections |
| 🎯 **Personalized Recs** | Investor persona-based recommendations |
| 💎 **Hidden Gems** | AI undervalued property detection |
| 📚 **Case Studies** | Real investment deep-dives |
| 🧠 **Neighborhood Graph** | Spatial relationship modeling |

---

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Train the model (optional, uses fallback if not available)
python src/train.py

# Run the app
streamlit run app.py
```

---

## 📍 Coverage

| Cities | Property Types |
|--------|----------------|
| Vancouver | Detached |
| Toronto | Townhouse |
| Calgary | Condo |
| Burnaby | Multi-family |
| Richmond | |
| North Vancouver | |

---

## 🎯 Investment Scoring System

Every property gets a 0-100 score combining:

| Component | Max Points | Description |
|-----------|------------|-------------|
| Growth Potential | 40 | ML-predicted appreciation |
| Buy/Rent Signal | 30 | Financial analysis recommendation |
| Risk Assessment | 30 | Market stability evaluation |

**Score Guide:**
- 🟢 **70-100:** Strong investment opportunity
- 🟡 **40-70:** Moderate investment potential
- 🔴 **0-40:** Consider waiting or explore alternatives

---

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| CV RMSE | ~$91,000 |
| CV MAPE | ~5.6% |
| Holdout R² | ~0.987 |

---

## 🔧 Technical Architecture

### Ensemble Model

```
Base Models:
├── XGBoost (gradient boosting)
├── LightGBM (fast gradient boosting)
├── Ridge (linear baseline)
└── Gradient Boosting (sklearn)

Meta-Learner: Ridge Regression
```

### Feature Engineering

- 62 engineered features
- Price momentum (3/6/12 month)
- Rental yield calculations
- Market regime indicators
- Neighborhood graph centrality
- Amenity proximity scores

### Explainability (SHAP)

- Global feature importance
- Individual prediction explanations
- Force/waterfall plots
- Dependence plots

---

## 🤖 Automated Pipeline

Weekly retraining via GitHub Actions:
- **Sunday 2 AM UTC:** Full model retraining
- **Daily 3 AM UTC:** Data refresh

```bash
# Manual pipeline run
python src/pipeline.py --force-refresh
```

---

## 📁 Project Structure

```
propra/
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies
├── .gitignore                  # Git ignore rules
├── src/
│   ├── predict.py              # Price prediction module
│   ├── features.py             # Feature engineering
│   ├── train.py                # Model training
│   ├── ingest.py               # Data ingestion
│   ├── pipeline.py             # Automated ML pipeline
│   ├── heatmap.py              # Market heatmap generation
│   ├── chatbot.py              # AI property advisor
│   ├── roi_calculator.py       # Investment metrics
│   ├── scenario_simulator.py   # Monte Carlo simulations
│   ├── ensemble.py             # Stacked ensemble model
│   ├── neighborhood_graph.py   # Graph-based modeling
│   ├── explainability.py       # SHAP explanations
│   ├── case_studies.py         # Investment case studies
│   ├── recommender.py          # Property recommendations
│   └── buy_vs_rent.py          # Buy vs rent calculator
├── data/
│   ├── raw/                    # Source data
│   └── processed/              # Merged + featured data
├── models/                     # Trained model files
└── .github/workflows/          # CI/CD pipelines
```

---

## 📦 Data Sources

| Source | Data Type | Access |
|--------|-----------|--------|
| GVR | MLS benchmark prices | Web scrape / fallback |
| Bank of Canada | Interest rates | API v2 |
| CMHC | Rental market survey | Fallback |
| RateHub | Mortgage rates | Web scrape |
| BoC News | Sentiment analysis | Web scrape |

---

## 💻 API Usage

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

## 🚀 Deploy Your Own

### Railway (Recommended)

1. Go to https://railway.app/new
2. Select "Deploy from GitHub repo"
3. Choose this repo
4. Click **Deploy** → Wait for build
5. Click **Generate Domain**

### Docker

```bash
docker build -t propra .
docker run -p 8501:8501 propra
```

---

## 🔒 Security

**.gitignore** excludes:
- `.env` files
- API keys and credentials
- Model checkpoints
- Sensitive data

**Never commit:**
- `.env` or `.env.local`
- `credentials.json`
- `api_keys.txt`

---

## 📚 Case Studies

Included in the platform:

1. **Downtown Vancouver Condo** - Long-term hold analysis
2. **Calgary Detached Home** - Growth market play
3. **Burnaby Townhouse** - Balanced investment

Each includes:
- Historical performance data
- Investment thesis
- Lessons learned
- ML forecasts

---

## 🧠 Neighborhood Graph Analysis

Graph-based modeling captures:
- Property-to-amenity distances
- Walkability scores
- Transit accessibility
- Neighborhood similarity
- Spatial autocorrelation

**Amenity Types Tracked:**
- Schools (elementary, high school)
- Transit (SkyTrain, bus, commuter rail)
- Parks and recreation
- Shopping and grocery
- Healthcare facilities
- Restaurants

---

## 📄 License

MIT License

---

## ⚠️ Disclaimer

For informational purposes only. Not financial advice. Real estate investments carry risks. Always consult with qualified professionals before making investment decisions.

---

## 📧 Contact

For questions or feedback, open an issue on GitHub.
