import os
import logging
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import json


# =============================================================================
# PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"
CONFIG_DIR = PROJECT_ROOT / "config"

# Ensure directories exist
for d in [DATA_DIR, MODELS_DIR, LOGS_DIR, CONFIG_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# =============================================================================
# LOGGING SETUP
# =============================================================================

def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers
    if logger.handlers:
        return logger

    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_formatter = logging.Formatter(
        '%(levelname)s - %(message)s'
    )

    # File handler
    log_file = LOGS_DIR / f"{datetime.now().strftime('%Y-%m-%d')}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)
    file_handler.setFormatter(file_formatter)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(console_formatter)

    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


# Create default logger
logger = setup_logger("re_engine")


# =============================================================================
# TAX AND FEE CONSTANTS (BC/CANADA)
# =============================================================================

@dataclass
class TaxConstants:
    # BC Property Transfer Tax (PTT) tiers
    ptt_tiers: List[tuple] = field(default_factory=lambda: [
        (200_000, 0.01),      # 1% on first $200K
        (2_000_000, 0.02),    # 2% on portion up to $2M
        (float('inf'), 0.03)  # 3% on portion above $2M
    ])

    # Foreign buyer additional tax (20% of purchase price)
    foreign_buyer_tax_rate: float = 0.20

    # BC Speculation and Vacancy Tax
    speculation_tax_rate: float = 0.005  # 0.5% for Canadian residents

    # Vancouver Empty Homes Tax
    empty_homes_tax_rate: float = 0.03  # 3% of assessed value

    # Property tax rate (typical Vancouver)
    property_tax_rate: float = 0.003  # 0.3% of assessed value annually

    # CMHC insurance rates (based on down payment %)
    cmhc_rates: List[tuple] = field(default_factory=lambda: [
        (0.20, 0.00),   # 20%+ down: 0%
        (0.15, 0.028),  # 15-19.99%: 2.80%
        (0.10, 0.031),  # 10-14.99%: 3.10%
        (0.05, 0.040),  # 5-9.99%: 4.00%
    ])

    # First-Time Home Buyer PTT exemption (BC)
    fttb_ptt_exemption_max_price: float = 835_000  # 2024 threshold
    fttb_ptt_partial_exemption_max: float = 860_000  # Partial exemption up to

    # FHSA (First Home Savings Account) limits
    fhsa_annual_limit: float = 8_000
    fhsa_lifetime_limit: float = 40_000

    # GST on new builds
    gst_rate: float = 0.05
    gst_rebate_max_price: float = 450_000  # Full rebate below this
    gst_rebate_phase_out_max: float = 475_000  # No rebate above this


# =============================================================================
# CMHC GUIDELINES
# =============================================================================

@dataclass
class CMHCGuidelines:
    # GDS (Gross Debt Service) ratio limit
    max_gds_ratio: float = 0.32  # 32%

    # TDS (Total Debt Service) ratio limit
    max_tds_ratio: float = 0.40  # 40%

    # Stress test rate (OSFI requirement)
    stress_test_rate: float = 0.0525  # 5.25% minimum
    stress_test_addition: float = 0.02  # Contract rate + 2%

    # Minimum down payment
    min_down_payment_pct: float = 0.05  # 5% minimum

    # Default insurance threshold (mandatory below this)
    mandatory_insurance_threshold: float = 0.20  # 20% down

    # Maximum amortization (insured mortgages)
    max_amortization_insured: int = 25

    # Maximum amortization (uninsured mortgages)
    max_amortization_uninsured: int = 30


# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

@dataclass
class ModelConfig:
    # XGBoost hyperparameters
    n_estimators: int = 500
    learning_rate: float = 0.05
    max_depth: int = 6
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    early_stopping_rounds: int = 50

    # Training configuration
    n_cv_splits: int = 5
    prediction_horizon_months: int = 6

    # Minimum data requirements
    min_training_samples: int = 100
    min_features: int = 10

    # Retraining triggers
    retrain_frequency_days: int = 7  # Weekly retraining
    min_new_data_points: int = 30  # Minimum new rows to trigger retrain
    performance_drop_threshold: float = 0.20  # Retrain if performance drops >20%


# =============================================================================
# DATA SOURCES
# =============================================================================

DATA_SOURCES = {
    "gvr": {
        "name": "Greater Vancouver REALTORS",
        "url": "https://www.greatervancouverrealtors.com/market-watch/market-reports",
        "format": "CSV/PDF",
        "frequency": "monthly",
        "fields": ["date", "city", "property_type", "benchmark_price", "sales",
                   "new_listings", "active_listings", "days_on_market"]
    },
    "cmhc": {
        "name": "CMHC Rental Market Survey",
        "url": "https://cmhc-schl.gc.ca",
        "format": "CSV",
        "frequency": "annual",
        "fields": ["date", "city", "bedroom_type", "avg_rent", "vacancy_rate"]
    },
    "boc": {
        "name": "Bank of Canada",
        "url": "https://www.bankofcanada.ca/rates/interest-rates/canadian-interest-rates/",
        "format": "CSV",
        "frequency": "daily",
        "fields": ["date", "overnight_rate", "prime_rate", "mortgage_rate_5yr_fixed"]
    },
    "statscan": {
        "name": "Statistics Canada",
        "url": "https://www150.statcan.gc.ca",
        "format": "CSV",
        "frequency": "quarterly",
        "fields": ["date", "city", "population_growth_yoy", "housing_starts",
                   "unemployment_rate"]
    },
    "ratehub": {
        "name": "Ratehub.ca (Mortgage Rates)",
        "url": "https://www.ratehub.ca",
        "format": "HTML",
        "frequency": "daily",
        "fields": ["lender", "rate_type", "term", "rate", "special_rate"]
    }
}


# =============================================================================
# CITIES AND PROPERTY TYPES
# =============================================================================

CITIES = [
    "Vancouver",
    "Burnaby",
    "Richmond",
    "North Vancouver",
    "Toronto",
    "Calgary"
]

PROPERTY_TYPES = ["condo", "townhouse", "detached"]

# City-specific metadata
CITY_METADATA = {
    "Vancouver": {
        "province": "BC",
        "region": "Lower Mainland",
        "population": 675218,
        "avg_price_detached": 1800000,
        "avg_price_condo": 750000
    },
    "Burnaby": {
        "province": "BC",
        "region": "Lower Mainland",
        "population": 249125,
        "avg_price_detached": 1500000,
        "avg_price_condo": 620000
    },
    "Richmond": {
        "province": "BC",
        "region": "Lower Mainland",
        "population": 209937,
        "avg_price_detached": 1400000,
        "avg_price_condo": 580000
    },
    "North Vancouver": {
        "province": "BC",
        "region": "Lower Mainland",
        "population": 88168,
        "avg_price_detached": 1700000,
        "avg_price_condo": 650000
    },
    "Toronto": {
        "province": "ON",
        "region": "GTA",
        "population": 2794356,
        "avg_price_detached": 1600000,
        "avg_price_condo": 700000
    },
    "Calgary": {
        "province": "AB",
        "region": "Calgary Region",
        "population": 1336000,
        "avg_price_detached": 650000,
        "avg_price_condo": 320000
    }
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def calculate_ptt(purchase_price: float, is_ftb: bool = False,
                  is_foreign_buyer: bool = False) -> float:
    constants = TaxConstants()

    # First-time buyer exemption
    if is_ftb and purchase_price <= constants.fttb_ptt_exemption_max_price:
        return 0.0

    # Partial exemption for first-time buyers
    if is_ftb and purchase_price <= constants.fttb_ptt_partial_exemption_max:
        exemption_ratio = (
            constants.fttb_ptt_partial_exemption_max - purchase_price
        ) / (
            constants.fttb_ptt_partial_exemption_max -
            constants.fttb_ptt_exemption_max_price
        )
        # Calculate full PTT and apply partial exemption
        full_ptt = calculate_ptt(purchase_price, is_ftb=False,
                                 is_foreign_buyer=is_foreign_buyer)
        return full_ptt * (1 - exemption_ratio)

    # Standard PTT calculation
    ptt = 0.0
    remaining = purchase_price

    for tier_max, rate in constants.ptt_tiers:
        taxable = min(remaining, tier_max - (purchase_price - remaining))
        if taxable > 0:
            ptt += taxable * rate
            remaining -= taxable
        if remaining <= 0:
            break

    # Simpler calculation
    ptt = 0
    if purchase_price <= 200_000:
        ptt = purchase_price * 0.01
    elif purchase_price <= 2_000_000:
        ptt = 200_000 * 0.01 + (purchase_price - 200_000) * 0.02
    else:
        ptt = 200_000 * 0.01 + 1_800_000 * 0.02 + (purchase_price - 2_000_000) * 0.03

    # Foreign buyer additional tax
    if is_foreign_buyer:
        ptt += purchase_price * constants.foreign_buyer_tax_rate

    return ptt


def calculate_cmhc_premium(mortgage_amount: float,
                           down_payment_pct: float) -> float:
    constants = TaxConstants()

    if down_payment_pct >= 0.20:
        return 0.0

    for min_down, rate in constants.cmhc_rates:
        if down_payment_pct >= min_down:
            return mortgage_amount * rate

    return mortgage_amount * 0.040  # Default to highest rate


def get_config() -> Dict:
    return {
        "paths": {
            "project_root": str(PROJECT_ROOT),
            "data_dir": str(DATA_DIR),
            "models_dir": str(MODELS_DIR),
            "logs_dir": str(LOGS_DIR)
        },
        "tax_constants": TaxConstants().__dict__,
        "cmhc_guidelines": CMHCGuidelines().__dict__,
        "model_config": ModelConfig().__dict__,
        "data_sources": DATA_SOURCES,
        "cities": CITIES,
        "property_types": PROPERTY_TYPES
    }


def save_config(path: str = None):
    if path is None:
        path = CONFIG_DIR / "config.json"

    config = get_config()

    # Convert paths back to strings
    with open(path, 'w') as f:
        json.dump(config, f, indent=2, default=str)

    logger.info(f"Configuration saved to {path}")


def load_config(path: str = None) -> Dict:
    if path is None:
        path = CONFIG_DIR / "config.json"

    if not Path(path).exists():
        logger.warning(f"Config file not found at {path}, using defaults")
        return get_config()

    with open(path, 'r') as f:
        config = json.load(f)

    logger.info(f"Configuration loaded from {path}")
    return config


# Initialize config on import
save_config()
