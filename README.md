# Propra - AI-Powered Canadian Real Estate

**Predict Canadian property prices and decide whether to buy or rent — powered by AI.**

Think Zillow, but smarter: ML price predictions, buy vs rent guidance, and market scenario simulation — all in one interactive dashboard.

---

## 🌐 Live Demo

**Try it now:** [https://real-estate-production-cd36.up.railway.app](https://real-estate-production-cd36.up.railway.app)

*Note: First load may take 30-60 seconds as Railway spins up from free tier sleep.*

---

## ✨ Features

### 1. 🔮 Price Prediction
Get AI-powered forecasts for 6/12/18 month prices with confidence intervals.

**Example:** *"Your Vancouver condo: $750K → predicted $812K in 12 months"*

### 2. 📊 Buy vs Rent Calculator
Canadian-specific analysis with BC PTT, CMHC rules, strata fees, and tax implications.

**Example:** *"Buying scores 72/100, Renting scores 54/100 → Buy recommended"*

### 3. 📈 Scenario Simulation
Test "what-if" scenarios: What happens if mortgage rates rise by 1%? See instant impact on predictions.

**Example:** *"+1% interest rate → your property predicted: $765K (down from $790K)"*

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

## 🎯 Investment Score

Every property gets a 0-100 score combining:
- **Growth Potential** (40 pts) - ML-predicted appreciation
- **Buy/Rent Signal** (30 pts) - Financial analysis recommendation  
- **Risk Assessment** (30 pts) - Market stability evaluation

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
├── src/
│   ├── scrapers.py      # Web scrapers (GVR, BoC, CMHC, RateHub)
│   ├── ingest.py        # Data ingestion pipeline
│   ├── features.py      # Feature engineering (62 features)
│   ├── train.py         # Model training with time-series CV
│   ├── predict.py       # Inference with ML model
│   ├── pipeline.py      # Automated ML pipeline
│   ├── buy_vs_rent.py   # Financial calculator
│   └── recommender.py   # Property recommendations
├── data/
│   ├── raw/             # Source data (CSV)
│   └── processed/       # Merged + featured data
├── models/              # Saved XGBoost models
├── .github/workflows/   # CI/CD for automated retraining
└── app.py               # Streamlit dashboard
```

---

## 📦 Data Sources

| Source | Data Type | Access |
|--------|-----------|--------|
| GVR | MLS benchmark prices (sold comps) | Web scrape / fallback |
| Bank of Canada | Interest rates | API v2 |
| CMHC | Rental market survey | Fallback (login required) |
| RateHub | Mortgage rates | Web scrape |
| BoC News | Sentiment analysis | Web scrape |

---

## 🚀 Deploy Your Own

### Railway (Recommended)

1. Go to https://railway.app/new
2. Select "Deploy from GitHub repo"
3. Choose this repo
4. Click **Deploy** → Wait for build
5. Click **Generate Domain** to get your public URL

Your app is live at `https://your-app-production.up.railway.app`

### Hugging Face Spaces (Free)

1. Go to https://huggingface.co/spaces
2. Create new Space → Streamlit
3. Connect GitHub repo

---

## 💻 API Usage

```python
from src.predict import PricePredictor
from src.recommender import PropertyRecommender, BuyerProfile

# Price prediction
predictor = PricePredictor()
predictor.load_model()

pred = predictor.predict_price_change(
    current_price=750000,
    city="Vancouver",
    property_type="condo",
    horizon_months=12
)

# Property recommendations
recommender = PropertyRecommender()
profile = BuyerProfile(
    annual_income=100000,
    available_down_payment=150000
)
recs = recommender.get_top_recommendations(profile, n=5)
```

---

## 📄 License

MIT

---

## ⚠️ Disclaimer

For informational purposes only. Not financial advice. Always consult a qualified financial advisor before making real estate decisions.
