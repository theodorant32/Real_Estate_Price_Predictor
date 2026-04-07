# Canadian Real Estate Price Predictor

ML-powered property price forecasting (6-month horizon) with buy-vs-rent analysis for Canadian markets.

## Quick Start

```bash
pip install -r requirements.txt
python src/pipeline.py
streamlit run app.py
```

## Features

| Feature | Description |
|---------|-------------|
| **Price Prediction** | XGBoost model predicting prices 6 months ahead |
| **Buy vs Rent** | Canadian-specific calculator (BC PTT, CMHC, strata fees) |
| **Market Comparison** | City-by-city analysis with buy/hold/sell signals |
| **Property Recommender** | Budget-based recommendations with CMHC affordability rules |

## Coverage

- **Cities**: Vancouver, Burnaby, Richmond, North Vancouver, Toronto, Calgary
- **Property Types**: Detached, Townhouse, Condo, Multi-family (duplex/triplex/fourplex)

## Model Performance

| Metric | Value |
|--------|-------|
| CV RMSE | ~$91,000 |
| CV MAPE | ~5.6% |
| Holdout R² | ~0.987 |

## Automated Pipeline

Weekly retraining via GitHub Actions:
- **Sunday 2 AM UTC**: Full model retraining
- **Daily 3 AM UTC**: Data refresh

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
| GVR | MLS benchmark prices | Web scrape / fallback |
| Bank of Canada | Interest rates | API v2 |
| CMHC | Rental market survey | Fallback (login required) |
| RateHub | Mortgage rates | Web scrape |

## API Usage

```python
from src.predict import PricePredictor
from src.recommender import PropertyRecommender, BuyerProfile

predictor = PricePredictor()
predictor.load_model()

pred = predictor.predict_price_change(
    current_price=750000,
    city="Vancouver",
    property_type="condo"
)

recommender = PropertyRecommender()
profile = BuyerProfile(annual_income=100000, available_down_payment=150000)
recs = recommender.get_top_recommendations(profile, n=5)
```

## License

MIT

## Disclaimer

For informational purposes only. Not financial advice.
