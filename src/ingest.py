import os
import io
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Optional
from bs4 import BeautifulSoup


class DataIngester:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        # Cities we track
        self.cities = [
            "Vancouver", "Burnaby", "Richmond", "North Vancouver",
            "Toronto", "Calgary"
        ]

        # Property types
        self.property_types = ["detached", "townhouse", "condo"]

    # =========================================================================
    # GVR Data - Greater Vancouver REALTORS
    # =========================================================================

    def fetch_gvr_data(self, save: bool = True) -> pd.DataFrame:
        print("Fetching GVR benchmark price data...")

        # Try to load cached data first
        cached_path = self.raw_dir / "gvr_benchmark.csv"
        if cached_path.exists():
            print(f"  Loading cached GVR data from {cached_path}")
            return pd.read_csv(cached_path, parse_dates=["date"])

        # Generate realistic historical data based on actual GVR reports
        # This is a placeholder - in production you'd scrape the actual data
        data = self._generate_gvr_placeholder()

        if save:
            data.to_csv(cached_path, index=False)
            print(f"  Saved GVR data to {cached_path}")

        return data

    def _generate_gvr_placeholder(self) -> pd.DataFrame:
        import numpy as np

        # Base prices by city and property type (Jan 2020 levels, approximate)
        base_prices = {
            ("Vancouver", "detached"): 1_800_000,
            ("Vancouver", "townhouse"): 950_000,
            ("Vancouver", "condo"): 720_000,
            ("Burnaby", "detached"): 1_500_000,
            ("Burnaby", "townhouse"): 850_000,
            ("Burnaby", "condo"): 620_000,
            ("Richmond", "detached"): 1_400_000,
            ("Richmond", "townhouse"): 800_000,
            ("Richmond", "condo"): 580_000,
            ("North Vancouver", "detached"): 1_700_000,
            ("North Vancouver", "townhouse"): 900_000,
            ("North Vancouver", "condo"): 650_000,
            ("Toronto", "detached"): 1_600_000,
            ("Toronto", "townhouse"): 900_000,
            ("Toronto", "condo"): 700_000,
            ("Calgary", "detached"): 650_000,
            ("Calgary", "townhouse"): 450_000,
            ("Calgary", "condo"): 320_000,
        }

        # Generate monthly data from Jan 2020 to present
        dates = pd.date_range("2020-01-01", datetime.now(), freq="MS")

        records = []
        np.random.seed(42)

        for city in self.cities:
            for prop_type in self.property_types:
                key = (city, prop_type)
                if key not in base_prices:
                    continue

                base = base_prices[key]

                # Create realistic price trajectory
                # 2020: slight dip (COVID)
                # 2021-2022: huge run-up
                # 2023: correction (rate hikes)
                # 2024-2025: gradual recovery

                for date in dates:
                    year = date.year
                    month = date.month

                    # Time index from start
                    t = (date - dates[0]).days / 30

                    # Base trend with regime changes
                    if date < pd.Timestamp("2021-01-01"):
                        # COVID period - slight decline
                        trend = 1.0 - 0.02 * (t / 12)
                    elif date < pd.Timestamp("2022-07-01"):
                        # Hot market - rapid appreciation
                        trend = 0.98 + 0.15 * ((t - 12) / 18)
                    elif date < pd.Timestamp("2023-07-01"):
                        # Rate hike correction
                        trend = 1.13 - 0.08 * ((t - 30) / 12)
                    else:
                        # Recovery period
                        trend = 1.05 + 0.03 * ((t - 42) / 24)

                    # Add seasonality (spring market stronger)
                    seasonal = 1.0 + 0.02 * np.sin(2 * np.pi * (month - 3) / 12)

                    # Add noise
                    noise = 1.0 + 0.01 * np.random.randn()

                    price = base * trend * seasonal * noise

                    # Generate correlated metrics
                    sales = int(50 + 30 * np.sin(2 * np.pi * (month - 4) / 12) + np.random.randn() * 10)
                    new_listings = int(sales * 1.5 + np.random.randn() * 15)
                    active_listings = int(new_listings * 2 + np.random.randn() * 20)
                    days_on_market = int(14 + 7 * np.random.randn())

                    # Ensure positive values
                    sales = max(5, sales)
                    new_listings = max(10, new_listings)
                    active_listings = max(20, active_listings)
                    days_on_market = max(3, days_on_market)

                    sales_to_active = sales / active_listings

                    records.append({
                        "date": date,
                        "city": city,
                        "property_type": prop_type,
                        "benchmark_price": round(price, 0),
                        "sales": sales,
                        "new_listings": new_listings,
                        "active_listings": active_listings,
                        "days_on_market": days_on_market,
                        "sales_to_active_ratio": round(sales_to_active, 3)
                    })

        return pd.DataFrame(records)

    # =========================================================================
    # CMHC Data - Rental Market Survey
    # =========================================================================

    def fetch_cmhc_data(self, save: bool = True) -> pd.DataFrame:
        print("Fetching CMHC rental market data...")

        cached_path = self.raw_dir / "cmhc_rental.csv"
        if cached_path.exists():
            print(f"  Loading cached CMHC data from {cached_path}")
            return pd.read_csv(cached_path, parse_dates=["date"])

        data = self._generate_cmhc_placeholder()

        if save:
            data.to_csv(cached_path, index=False)
            print(f"  Saved CMHC data to {cached_path}")

        return data

    def _generate_cmhc_placeholder(self) -> pd.DataFrame:
        import numpy as np

        # Base rents by city and bedroom type (2020 levels)
        base_rents = {
            ("Vancouver", "1br"): 2_100,
            ("Vancouver", "2br"): 3_200,
            ("Burnaby", "1br"): 1_800,
            ("Burnaby", "2br"): 2_700,
            ("Richmond", "1br"): 1_750,
            ("Richmond", "2br"): 2_600,
            ("North Vancouver", "1br"): 1_900,
            ("North Vancouver", "2br"): 2_800,
            ("Toronto", "1br"): 2_000,
            ("Toronto", "2br"): 2_900,
            ("Calgary", "1br"): 1_300,
            ("Calgary", "2br"): 1_800,
        }

        # Annual data (CMHC publishes once per year in October)
        years = list(range(2020, 2026))

        records = []
        np.random.seed(43)

        for city in self.cities:
            for bedroom in ["1br", "2br"]:
                key = (city, bedroom)
                if key not in base_rents:
                    continue

                base = base_rents[key]

                for year in years:
                    # Rent growth trajectory
                    if year == 2020:
                        growth = 1.0
                    elif year == 2021:
                        growth = 1.02  # COVID suppression
                    elif year == 2022:
                        growth = 1.08  # Post-COVID rebound
                    elif year == 2023:
                        growth = 1.15  # Strong demand
                    elif year == 2024:
                        growth = 1.22
                    else:  # 2025
                        growth = 1.28

                    # Vacancy rate (tighter in Vancouver)
                    if city == "Vancouver":
                        vacancy = 0.01 + 0.005 * np.random.randn()
                    elif city == "Toronto":
                        vacancy = 0.015 + 0.005 * np.random.randn()
                    elif city == "Calgary":
                        vacancy = 0.03 + 0.01 * np.random.randn()
                    else:
                        vacancy = 0.02 + 0.005 * np.random.randn()

                    vacancy = max(0.005, min(0.10, vacancy))

                    records.append({
                        "date": pd.Timestamp(f"{year}-10-01"),
                        "city": city,
                        "bedroom_type": bedroom,
                        "avg_rent": round(base * growth + np.random.randn() * 50, 0),
                        "vacancy_rate": round(vacancy, 3)
                    })

        return pd.DataFrame(records)

    # =========================================================================
    # Bank of Canada Data - Interest Rates
    # =========================================================================

    def fetch_boc_data(self, save: bool = True) -> pd.DataFrame:
        print("Fetching Bank of Canada rate data...")

        cached_path = self.raw_dir / "boc_rates.csv"
        if cached_path.exists():
            print(f"  Loading cached BoC data from {cached_path}")
            return pd.read_csv(cached_path, parse_dates=["date"])

        data = self._generate_boc_placeholder()

        if save:
            data.to_csv(cached_path, index=False)
            print(f"  Saved BoC data to {cached_path}")

        return data

    def _generate_boc_placeholder(self) -> pd.DataFrame:
        import numpy as np

        # Monthly data from 2020 to present
        dates = pd.date_range("2020-01-01", datetime.now(), freq="MS")

        records = []

        for date in dates:
            # Overnight rate trajectory (based on actual BoC history)
            if date < pd.Timestamp("2020-03-01"):
                overnight = 1.75
            elif date < pd.Timestamp("2020-04-01"):
                overnight = 0.25  # COVID emergency cut
            elif date < pd.Timestamp("2022-03-01"):
                overnight = 0.25  # Held at bottom
            elif date < pd.Timestamp("2022-06-01"):
                overnight = 0.50  # First hikes
            elif date < pd.Timestamp("2022-07-01"):
                overnight = 1.00
            elif date < pd.Timestamp("2022-09-01"):
                overnight = 1.75
            elif date < pd.Timestamp("2022-10-01"):
                overnight = 2.50
            elif date < pd.Timestamp("2022-12-01"):
                overnight = 3.75
            elif date < pd.Timestamp("2023-02-01"):
                overnight = 4.50  # Peak
            elif date < pd.Timestamp("2024-06-01"):
                overnight = 4.50  # Held at peak
            elif date < pd.Timestamp("2024-09-01"):
                overnight = 4.25  # Cutting cycle begins
            elif date < pd.Timestamp("2025-01-01"):
                overnight = 3.75
            else:
                overnight = 3.25

            # Prime rate = overnight + ~2%
            prime = overnight + 2.2

            # 5-year fixed mortgage rate (scraped from ratehub typically)
            if date < pd.Timestamp("2020-03-01"):
                mortgage_5yr = 3.0
            elif date < pd.Timestamp("2021-01-01"):
                mortgage_5yr = 2.0  # COVID lows
            elif date < pd.Timestamp("2022-03-01"):
                mortgage_5yr = 2.5
            elif date < pd.Timestamp("2023-01-01"):
                mortgage_5yr = 4.5  # Rapid increase
            elif date < pd.Timestamp("2023-07-01"):
                mortgage_5yr = 5.5
            elif date < pd.Timestamp("2024-01-01"):
                mortgage_5yr = 6.0  # Peak
            elif date < pd.Timestamp("2025-01-01"):
                mortgage_5yr = 5.0
            else:
                mortgage_5yr = 4.5

            records.append({
                "date": date,
                "overnight_rate": overnight,
                "prime_rate": prime,
                "mortgage_rate_5yr_fixed": mortgage_5yr
            })

        return pd.DataFrame(records)

    # =========================================================================
    # Statistics Canada Data
    # =========================================================================

    def fetch_statscan_data(self, save: bool = True) -> pd.DataFrame:
        print("Fetching Statistics Canada data...")

        cached_path = self.raw_dir / "statscan.csv"
        if cached_path.exists():
            print(f"  Loading cached StatsCan data from {cached_path}")
            return pd.read_csv(cached_path, parse_dates=["date"])

        data = self._generate_statscan_placeholder()

        if save:
            data.to_csv(cached_path, index=False)
            print(f"  Saved StatsCan data to {cached_path}")

        return data

    def _generate_statscan_placeholder(self) -> pd.DataFrame:
        import numpy as np

        # Annual/quarterly data
        quarters = pd.date_range("2020-01-01", datetime.now(), freq="QE")

        # Base metrics by city
        city_metrics = {
            "Vancouver": {"pop_growth": 0.015, "housing_starts": 8000, "unemployment": 0.06},
            "Burnaby": {"pop_growth": 0.020, "housing_starts": 2000, "unemployment": 0.055},
            "Richmond": {"pop_growth": 0.018, "housing_starts": 1500, "unemployment": 0.055},
            "North Vancouver": {"pop_growth": 0.012, "housing_starts": 800, "unemployment": 0.05},
            "Toronto": {"pop_growth": 0.025, "housing_starts": 25000, "unemployment": 0.065},
            "Calgary": {"pop_growth": 0.030, "housing_starts": 12000, "unemployment": 0.08},
        }

        records = []
        np.random.seed(44)

        for quarter in quarters:
            year = quarter.year

            for city in self.cities:
                metrics = city_metrics.get(city, {})

                # Add some variation over time
                if year == 2020:
                    # COVID impact
                    pop_adj = 0.5
                    starts_adj = 0.6
                    unemp_adj = 1.5
                elif year == 2021:
                    # Recovery
                    pop_adj = 0.7
                    starts_adj = 0.8
                    unemp_adj = 1.2
                elif year == 2022:
                    # Strong recovery
                    pop_adj = 1.0
                    starts_adj = 1.0
                    unemp_adj = 1.0
                elif year == 2023:
                    # Rate hike slowdown
                    pop_adj = 1.3  # Immigration surge
                    starts_adj = 0.9
                    unemp_adj = 1.0
                elif year == 2024:
                    # Continued immigration
                    pop_adj = 1.4
                    starts_adj = 1.0
                    unemp_adj = 1.1
                else:  # 2025
                    pop_adj = 1.3
                    starts_adj = 1.1
                    unemp_adj = 1.0

                pop_growth = metrics.get("pop_growth", 0.015) * pop_adj + 0.005 * np.random.randn()
                housing_starts = int(metrics.get("housing_starts", 5000) * starts_adj + np.random.randn() * 500)
                unemployment = min(0.15, max(0.03, metrics.get("unemployment", 0.06) * unemp_adj + 0.01 * np.random.randn()))

                records.append({
                    "date": quarter,
                    "city": city,
                    "population_growth_yoy": round(pop_growth, 4),
                    "housing_starts": max(100, housing_starts),
                    "unemployment_rate": round(unemployment, 3)
                })

        return pd.DataFrame(records)

    # =========================================================================
    # Main Ingestion Pipeline
    # =========================================================================

    def ingest_all(self, force_refresh: bool = False) -> dict:
        if force_refresh:
            # Delete cached files
            for f in self.raw_dir.glob("*.csv"):
                f.unlink()

        return {
            "gvr": self.fetch_gvr_data(save=True),
            "cmhc": self.fetch_cmhc_data(save=True),
            "boc": self.fetch_boc_data(save=True),
            "statscan": self.fetch_statscan_data(save=True)
        }

    def create_merged_dataset(self, data: dict = None) -> pd.DataFrame:
        if data is None:
            data = self.ingest_all()

        gvr = data["gvr"]
        cmhc = data["cmhc"]
        boc = data["boc"]
        statscan = data["statscan"]

        print("Merging datasets...")

        # Start with GVR data (most granular - monthly by city + property_type)
        merged = gvr.copy()

        # Merge Bank of Canada rates (national, by date only)
        merged = merged.merge(boc, on="date", how="left")

        # Merge CMHC rental data (by city + date, need to aggregate bedroom types)
        cmhc_pivot = cmhc.pivot_table(
            index=["city", "date"],
            columns="bedroom_type",
            values=["avg_rent", "vacancy_rate"]
        ).reset_index()

        cmhc_pivot.columns = [
            "city", "date",
            "avg_rent_1br", "avg_rent_2br",
            "vacancy_rate_1br", "vacancy_rate_2br"
        ]

        merged = merged.merge(cmhc_pivot, on=["city", "date"], how="left")

        # Merge StatsCan data (quarterly) using merge_asof for nearest date match
        # StatsCan data is quarterly, GVR is monthly - need to match by quarter
        merged = merged.sort_values(["date", "city"]).reset_index(drop=True)
        statscan = statscan.sort_values(["date", "city"]).reset_index(drop=True)

        merged = pd.merge_asof(
            merged,
            statscan,
            on="date",
            by="city",
            direction="backward"  # Use most recent quarter data
        )

        # Forward fill to propagate any remaining gaps within groups
        merged = merged.sort_values(["city", "property_type", "date"])
        for col in merged.columns:
            if col not in ["city", "property_type", "date"]:
                merged[col] = merged.groupby(["city", "property_type"])[col].transform(
                    lambda x: x.ffill()
                )

        print(f"  Merged dataset shape: {merged.shape}")
        print(f"  Date range: {merged['date'].min()} to {merged['date'].max()}")
        print(f"  Cities: {merged['city'].unique().tolist()}")
        print(f"  Property types: {merged['property_type'].unique().tolist()}")

        return merged


def main():
    ingester = DataIngester()

    # Fetch all data
    data = ingester.ingest_all(force_refresh=False)

    # Create merged dataset
    merged = ingester.create_merged_dataset(data)

    # Save processed data
    output_path = ingester.processed_dir / "merged_data.csv"
    merged.to_csv(output_path, index=False)
    print(f"\nSaved merged dataset to {output_path}")

    return merged


if __name__ == "__main__":
    main()
