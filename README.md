# Canadian Real Estate Price Predictor

ML-powered property price forecasting with buy-vs-rent analysis for Canadian markets.

## Live Demo

**Try it now:** [https://propra-production.up.railway.app](https://propra-production.up.railway.app)

## Quick Start

```bash
pip install -r requirements.txt
python src/pipeline.py
streamlit run app.py
```

## Features

| Feature | Description |
|---------|-------------|
| **Price Prediction** | XGBoost model predicting prices 6/12/18 months ahead |
| **Buy vs Rent** | Canadian-specific calculator (BC PTT, CMHC, strata fees) |
| **Market Comparison** | City-by-city analysis with buy/hold/sell signals |
| **Property Recommender** | Budget-based recommendations with CMHC affordability rules |

## Coverage

- **Cities:** Vancouver, Burnaby, Richmond, North Vancouver, Toronto, Calgary
- **Property Types:** Detached, Townhouse, Condo, Multi-family (duplex/triplex/fourplex)

## Model Performance

| Metric | Value |
|--------|-------|
| CV RMSE | ~$91,000 |
| CV MAPE | ~5.6% |
| Holdout R² | ~0.987 |

## Automated Pipeline

Weekly retraining via GitHub Actions:
- **Sunday 2 AM UTC:** Full model retraining
- **Daily 3 AM UTC:** Data refresh

```bash
# Manual run
python src/pipeline.py --force-refresh
```

## Project Structure

```
├── src/
│   ├── scrapers.py    # Web scrapers (GVR, BoC, CMHC, RateHub)
│   ├── ingest.py      # Data ingestion pipeline
│   ├── features.py    # Feature engineering (62 features)
│   ├── train.py       # Model training with time-series CV
│   ├── predict.py     # Inference with ML model
│   ├── pipeline.py    # Automated ML pipeline
│   ├── buy_vs_rent.py # Financial calculator
│   └── recommender.py # Property recommendations
├── data/
│   ├── raw/           # Source data (CSV)
│   └── processed/     # Merged + featured data
├── models/            # Saved XGBoost models
├── .github/workflows/ # CI/CD for automated retraining
└── app.py             # Streamlit dashboard
```

## Data Sources

| Source | Data Type | Access |
|--------|-----------|--------|
| GVR | MLS benchmark prices (sold comps) | Web scrape / fallback |
| Bank of Canada | Interest rates | API v2 |
| CMHC | Rental market survey | Fallback (login required) |
| RateHub | Mortgage rates | Web scrape |
| BoC News | Sentiment analysis | Web scrape |

## API Usage

```python
from src.predict import PricePredictor
from src.recommender import PropertyRecommender, BuyerProfile

predictor = PricePredictor()
predictor.load_model()

pred = predictor.predict_price_change(
    current_price=750000,
    city="Vancouver",
    property_type="condo",
    horizon_months=12
)

recommender = PropertyRecommender()
profile = BuyerProfile(annual_income=100000, available_down_payment=150000)
recs = recommender.get_top_recommendations(profile, n=5)
```

## Deploy Your Own

### Railway (Recommended)

1. Go to https://railway.app/new
2. Select "Deploy from GitHub repo"
3. Choose this repo
4. Deploy

### Hugging Face Spaces (Free)

1. Go to https://huggingface.co/spaces
2. Create new Space → Streamlit
3. Connect GitHub repo

## License

MIT

## Disclaimer

For informational purposes only. Not financial advice.
