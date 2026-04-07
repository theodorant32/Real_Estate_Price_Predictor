# Canadian Real Estate Price Predictor

ML-powered property price forecasting (6-month horizon) with buy-vs-rent analysis for Canadian markets.

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Run pipeline
python src/ingest.py && python src/features.py && python src/train.py

# Dashboard
streamlit run app.py
```

## Features

| Feature | Description |
|---------|-------------|
| **Price Prediction** | XGBoost model predicting prices 6 months ahead |
| **Buy vs Rent** | Canadian-specific calculator (PTT, CMHC, strata) |
| **Market Comparison** | Compare cities with buy/hold/sell recommendations |
| **Property Recommender** | Budget-based recommendations with CMHC affordability |

## Coverage

- **Cities**: Vancouver, Burnaby, Richmond, North Vancouver, Toronto, Calgary
- **Property Types**: Detached, Townhouse, Condo

## Model Details

**Validation**: Purged time-series CV (5 folds, 6-month embargo) + holdout test set

**Regularization**: L1/L2, max_depth=4, dropout via colsample

**Metrics**: RMSE, MAPE, R², directional accuracy

```python
from src.predict import PricePredictor

predictor = PricePredictor()
predictor.load_model()
pred = predictor.predict_price_change(
    current_price=750000,
    city="Vancouver",
    property_type="condo"
)
```

## Project Structure

```
├── src/
│   ├── ingest.py      # Data ingestion
│   ├── features.py    # Feature engineering
│   ├── train.py       # Model training (with CV + holdout)
│   ├── predict.py     # Inference
│   ├── buy_vs_rent.py # Financial calculator
│   └── recommender.py # Recommendations
├── data/
│   ├── raw/           # Source data
│   └── processed/     # Cleaned data
├── models/            # Saved models + metrics
└── app.py             # Streamlit dashboard
```

## Data Sources

- GVR (benchmark prices), CMHC (rental data), Bank of Canada (rates), Statistics Canada

## Disclaimer

For informational purposes only. Not financial advice.

## License

MIT
