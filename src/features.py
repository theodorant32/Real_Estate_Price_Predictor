import pandas as pd
import numpy as np
from typing import List, Tuple
from pathlib import Path


class FeatureEngineer:
    def __init__(self, prediction_horizon: int = 6):
        self.prediction_horizon = prediction_horizon

    def create_target(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Shift price forward by prediction_horizon months within each group
        df["target_price_6m"] = df.groupby(["city", "property_type"])[
            "benchmark_price"
        ].shift(-self.prediction_horizon)

        # Also create the return percentage (more stable target)
        df["target_return_6m"] = (
            df["target_price_6m"] - df["benchmark_price"]
        ) / df["benchmark_price"]

        return df

    def create_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Sort for proper lag calculation
        df = df.sort_values(["city", "property_type", "date"])

        # ===== LAG FEATURES =====
        # shift(n) looks n periods into the PAST - no leakage
        lag_months = [1, 3, 6, 12]
        for lag in lag_months:
            df[f"price_lag_{lag}m"] = df.groupby(
                ["city", "property_type"]
            )["benchmark_price"].shift(lag)

        # ===== MOMENTUM FEATURES =====
        # Month-over-month change (uses current and 1 month ago)
        df["price_mom"] = df.groupby(["city", "property_type"])[
            "benchmark_price"
        ].pct_change(1)

        # Year-over-year change (uses current and 12 months ago)
        df["price_yoy"] = df.groupby(["city", "property_type"])[
            "benchmark_price"
        ].pct_change(12)

        # 3-month momentum
        df["price_3m_momentum"] = (
            df["benchmark_price"] - df["price_lag_3m"]
        ) / df["price_lag_3m"]

        # 6-month momentum
        df["price_6m_momentum"] = (
            df["benchmark_price"] - df["price_lag_6m"]
        ) / df["price_lag_6m"]

        # ===== ROLLING STATISTICS =====
        # CRITICAL: Use min_periods=1 but only look at past data
        # rolling(n) by default is right-aligned (looks at past n periods)
        # This is SAFE - no future data leakage

        # 3-month rolling mean (past 3 months including current)
        df["rolling_mean_3m"] = df.groupby(["city", "property_type"])[
            "benchmark_price"
        ].transform(lambda x: x.rolling(3, min_periods=1).mean())

        # 12-month rolling mean
        df["rolling_mean_12m"] = df.groupby(["city", "property_type"])[
            "benchmark_price"
        ].transform(lambda x: x.rolling(12, min_periods=1).mean())

        # 6-month rolling std (volatility)
        df["rolling_std_6m"] = df.groupby(["city", "property_type"])[
            "benchmark_price"
        ].transform(lambda x: x.rolling(6, min_periods=1).std())

        # Price vs rolling mean (deviation)
        df["price_vs_3m_mean"] = (
            df["benchmark_price"] - df["rolling_mean_3m"]
        ) / df["rolling_mean_3m"]

        df["price_vs_12m_mean"] = (
            df["benchmark_price"] - df["rolling_mean_12m"]
        ) / df["rolling_mean_12m"]

        return df

    def create_supply_demand_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Sales to active ratio is already in raw data, but let's add derived metrics

        # Absorption rate (sales / new listings)
        df["absorption_rate"] = df["sales"] / df["new_listings"].replace(0, np.nan)

        # Months of supply (active listings / sales) - inverse of absorption
        df["months_of_supply"] = df["active_listings"] / df["sales"].replace(0, np.nan)

        # Cap extreme values
        df["months_of_supply"] = df["months_of_supply"].clip(upper=24)

        # Lag features for sales and listings
        df["sales_lag_1m"] = df.groupby(["city", "property_type"])["sales"].shift(1)
        df["listings_lag_1m"] = df.groupby(["city", "property_type"])[
            "new_listings"
        ].shift(1)

        # Sales momentum
        df["sales_yoy"] = df.groupby(["city", "property_type"])["sales"].pct_change(12)

        # Listings growth
        df["listings_yoy"] = df.groupby(["city", "property_type"])[
            "new_listings"
        ].pct_change(12)

        # Market tightness indicator
        # < 12% = buyer's market, 12-20% = balanced, > 20% = seller's market
        df["market_tightness"] = pd.cut(
            df["sales_to_active_ratio"],
            bins=[0, 0.12, 0.20, 1.0],
            labels=["buyers_market", "balanced", "sellers_market"]
        )

        return df

    def create_rate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Rate changes over different periods
        df["rate_change_3m"] = df["overnight_rate"] - df["overnight_rate"].shift(3)
        df["rate_change_6m"] = df["overnight_rate"] - df["overnight_rate"].shift(6)
        df["rate_change_12m"] = df["overnight_rate"] - df["overnight_rate"].shift(12)

        # Stress test rate (OSFI: contract rate + 2% or 5.25%, whichever is higher)
        df["stress_test_rate"] = df["mortgage_rate_5yr_fixed"].apply(
            lambda x: max(x + 0.02, 0.0525)
        )

        # Rate regime indicator
        df["rate_regime"] = pd.cut(
            df["overnight_rate"],
            bins=[-np.inf, 1.0, 3.0, 5.0, np.inf],
            labels=["very_low", "low", "moderate", "high"]
        )

        # Affordability shock (rate increase * price level)
        df["affordability_shock"] = (
            df["rate_change_12m"] * df["benchmark_price"] / 100000
        )

        return df

    def create_rental_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Average rent (use 2BR as primary, or average of 1BR and 2BR)
        df["avg_rent"] = (df["avg_rent_1br"] + df["avg_rent_2br"]) / 2

        # Average vacancy rate
        df["vacancy_rate"] = (df["vacancy_rate_1br"] + df["vacancy_rate_2br"]) / 2

        # Price to rent ratio (annual)
        # Higher = more expensive to buy relative to rent
        df["price_to_rent_ratio"] = (
            df["benchmark_price"] / (df["avg_rent_2br"] * 12)
        )

        # Rent yield (inverse of price-to-rent)
        df["rent_yield"] = (df["avg_rent_2br"] * 12) / df["benchmark_price"]

        # Rent YoY change (rent inflation)
        df["rent_yoy_change"] = df.groupby(["city", "property_type"])[
            "avg_rent_2br"
        ].pct_change(12)

        # Vacancy trend
        df["vacancy_trend"] = df["vacancy_rate"] - df["vacancy_rate"].shift(1)

        return df

    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Extract date components
        df["month"] = df["date"].dt.month
        df["quarter"] = df["date"].dt.quarter
        df["year"] = df["date"].dt.year
        df["dayofweek"] = df["date"].dt.dayofweek

        # Seasonality indicators
        df["is_spring"] = (df["month"].isin([3, 4, 5, 6])).astype(int)
        df["is_fall"] = (df["month"].isin([9, 10, 11])).astype(int)
        df["is_summer"] = (df["month"].isin([7, 8])).astype(int)
        df["is_winter"] = (df["month"].isin([12, 1, 2])).astype(int)

        # Policy regime flags
        df["post_rate_hike"] = (df["date"] >= pd.Timestamp("2022-07-01")).astype(int)
        df["foreign_buyer_ban"] = (df["date"] >= pd.Timestamp("2023-01-01")).astype(int)

        # Time trend (continuous)
        df["time_index"] = (df["date"] - df["date"].min()).dt.days

        return df

    def create_city_property_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # City-level aggregates (market-wide, not property-type specific)
        city_agg = df.groupby(["city", "date"]).agg({
            "benchmark_price": "mean",
            "sales": "sum",
            "new_listings": "sum",
            "active_listings": "sum"
        }).reset_index()

        city_agg.columns = [
            "city", "date",
            "city_avg_price", "city_total_sales",
            "city_total_listings", "city_total_active"
        ]

        df = df.merge(city_agg, on=["city", "date"], how="left")

        # Property type market share
        df["property_type_share"] = df["sales"] / df["city_total_sales"].replace(0, np.nan)

        # Relative price (vs city average)
        df["price_vs_city_avg"] = (
            df["benchmark_price"] - df["city_avg_price"]
        ) / df["city_avg_price"]

        return df

    def create_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        print("Creating features...")

        # Create target first
        df = self.create_target(df)
        print("  - Target variable created")

        # Price features
        df = self.create_price_features(df)
        print("  - Price features created")

        # Supply & demand
        df = self.create_supply_demand_features(df)
        print("  - Supply/demand features created")

        # Rate features
        df = self.create_rate_features(df)
        print("  - Rate features created")

        # Rental features
        df = self.create_rental_features(df)
        print("  - Rental features created")

        # Temporal features
        df = self.create_temporal_features(df)
        print("  - Temporal features created")

        # City/property features
        df = self.create_city_property_features(df)
        print("  - City/property features created")

        # Encode categorical variables
        df = self.encode_categoricals(df)
        print("  - Categorical variables encoded")

        return df

    def encode_categoricals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # One-hot encode property type
        property_dummies = pd.get_dummies(
            df["property_type"],
            prefix="prop",
            drop_first=False
        )

        # One-hot encode city
        city_dummies = pd.get_dummies(
            df["city"],
            prefix="city",
            drop_first=False
        )

        # One-hot encode market tightness
        if "market_tightness" in df.columns:
            tightness_dummies = pd.get_dummies(
                df["market_tightness"],
                prefix="tightness",
                drop_first=False
            )
            df = pd.concat([df, tightness_dummies], axis=1)

        # One-hot encode rate regime
        if "rate_regime" in df.columns:
            regime_dummies = pd.get_dummies(
                df["rate_regime"],
                prefix="regime",
                drop_first=False
            )
            df = pd.concat([df, regime_dummies], axis=1)

        # Add encoded dummies
        df = pd.concat([df, property_dummies, city_dummies], axis=1)

        return df

    def get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        # Columns to exclude
        exclude_cols = {
            "date", "city", "property_type",
            "target_price_6m", "target_return_6m",
            "market_tightness", "rate_regime",  # These are encoded
        }

        feature_cols = [
            col for col in df.columns
            if col not in exclude_cols
            and not df[col].isna().all()
        ]

        return feature_cols

    def prepare_train_test_split(
        self,
        df: pd.DataFrame,
        train_end_date: str = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if train_end_date is None:
            # Use 80/20 time-based split
            max_date = df["date"].max()
            min_date = df["date"].min()
            split_date = min_date + 0.8 * (max_date - min_date)
        else:
            split_date = pd.Timestamp(train_end_date)

        train_df = df[df["date"] <= split_date].copy()
        test_df = df[df["date"] > split_date].copy()

        print(f"Train/test split date: {split_date.date()}")
        print(f"  Train: {len(train_df)} rows ({train_df['date'].min().date()} to {train_df['date'].max().date()})")
        print(f"  Test: {len(test_df)} rows ({test_df['date'].min().date()} to {test_df['date'].max().date()})")

        return train_df, test_df


def main():
    from ingest import DataIngester

    # Load merged data
    ingester = DataIngester()
    merged = ingester.create_merged_dataset()

    # Create features
    fe = FeatureEngineer(prediction_horizon=6)
    featured = fe.create_all_features(merged)

    # Save processed data
    output_path = ingester.processed_dir / "featured_data.csv"
    featured.to_csv(output_path, index=False)
    print(f"\nSaved featured dataset to {output_path}")
    print(f"Total features: {len(fe.get_feature_columns(featured))}")

    return featured


if __name__ == "__main__":
    main()
