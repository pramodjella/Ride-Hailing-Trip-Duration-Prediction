#!/usr/bin/env python
"""Improved ETA model: LightGBM with rich feature engineering.

Key improvements over baseline:
1. Zone-pair historical statistics (median, p25, p75 duration)
2. Geo features (haversine distance from zone centroids)
3. Rich temporal features (cyclical encoding, rush hour flags, holiday flags)
4. Zone-level frequency features
5. LightGBM for faster training and often better generalization

Run:
    python train_improved.py
"""

from __future__ import annotations

import json
import math
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from urllib.request import urlretrieve
import zipfile
import os

DATA_DIR = Path(__file__).parent / "data"
MODEL_PATH = Path(__file__).parent / "model.pkl"
ZONE_STATS_PATH = Path(__file__).parent / "zone_stats.pkl"

# ----------- Zone geo data helpers -----------

def download_zone_shapefile() -> Path:
    """Download and extract the NYC taxi zone shapefile."""
    zip_path = DATA_DIR / "taxi_zones.zip"
    shape_dir = DATA_DIR / "taxi_zones"
    if shape_dir.exists():
        return shape_dir
    if not zip_path.exists():
        print("  Downloading taxi zone shapefile...")
        urlretrieve("https://d37ci6vzurychx.cloudfront.net/misc/taxi_zones.zip", zip_path)
    shape_dir.mkdir(exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(shape_dir)
    return shape_dir


def load_zone_centroids() -> dict[int, tuple[float, float]]:
    """Load zone centroids (lat/lon) from the shapefile."""
    try:
        import geopandas as gpd
        shape_dir = download_zone_shapefile()
        # Find the .shp file
        shp_files = list(shape_dir.rglob("*.shp"))
        if not shp_files:
            print("  WARNING: No .shp file found, falling back to lookup CSV")
            return _load_zone_centroids_from_csv()
        gdf = gpd.read_file(shp_files[0])
        # Convert to lat/lon (EPSG:4326)
        gdf = gdf.to_crs(epsg=4326)
        centroids = {}
        for _, row in gdf.iterrows():
            zone_id = int(row["LocationID"]) if "LocationID" in row else int(row["OBJECTID"])
            c = row.geometry.centroid
            centroids[zone_id] = (c.y, c.x)  # (lat, lon)
        return centroids
    except Exception as e:
        print(f"  WARNING: Could not load shapefile ({e}), falling back to CSV")
        return _load_zone_centroids_from_csv()


def _load_zone_centroids_from_csv() -> dict[int, tuple[float, float]]:
    """Fallback: approximate centroids from the zone lookup CSV."""
    csv_path = DATA_DIR / "taxi_zone_lookup.csv"
    if not csv_path.exists():
        urlretrieve("https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv", csv_path)
    df = pd.read_csv(csv_path)
    # No lat/lon in this CSV, so we can't compute centroids
    # Return empty dict, haversine features won't be used
    return {}


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Haversine distance in kilometers."""
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ----------- Feature engineering -----------

NYC_HOLIDAYS_2023 = {
    "2023-01-01", "2023-01-02", "2023-01-16", "2023-02-20",
    "2023-05-29", "2023-06-19", "2023-07-04", "2023-09-04",
    "2023-10-09", "2023-11-10", "2023-11-23", "2023-11-24",
    "2023-12-25", "2023-12-26", "2023-12-31",
}


def compute_zone_pair_stats(train: pd.DataFrame) -> dict:
    """Pre-compute zone-pair statistics from training data."""
    print("  Computing zone-pair statistics...")
    stats = {}

    # Zone-pair stats
    grouped = train.groupby(["pickup_zone", "dropoff_zone"])["duration_seconds"]
    pair_stats = grouped.agg(["median", "mean", "count"]).reset_index()
    pair_stats.columns = ["pickup_zone", "dropoff_zone", "pair_median", "pair_mean", "pair_count"]

    # Quantiles
    pair_q25 = grouped.quantile(0.25).reset_index()
    pair_q25.columns = ["pickup_zone", "dropoff_zone", "pair_q25"]
    pair_q75 = grouped.quantile(0.75).reset_index()
    pair_q75.columns = ["pickup_zone", "dropoff_zone", "pair_q75"]

    pair_stats = pair_stats.merge(pair_q25, on=["pickup_zone", "dropoff_zone"])
    pair_stats = pair_stats.merge(pair_q75, on=["pickup_zone", "dropoff_zone"])

    # Zone-level stats (pickup)
    pu_stats = train.groupby("pickup_zone")["duration_seconds"].agg(["median", "mean", "count"]).reset_index()
    pu_stats.columns = ["zone", "pu_median", "pu_mean", "pu_count"]

    # Zone-level stats (dropoff)
    do_stats = train.groupby("dropoff_zone")["duration_seconds"].agg(["median", "mean", "count"]).reset_index()
    do_stats.columns = ["zone", "do_median", "do_mean", "do_count"]

    # Hour-of-day stats per zone pair
    ts = pd.to_datetime(train["requested_at"])
    train_with_hour = train.copy()
    train_with_hour["hour"] = ts.dt.hour
    hour_pair = train_with_hour.groupby(["pickup_zone", "dropoff_zone", "hour"])["duration_seconds"].median().reset_index()
    hour_pair.columns = ["pickup_zone", "dropoff_zone", "hour", "hour_pair_median"]

    stats["pair_stats"] = pair_stats
    stats["pu_stats"] = pu_stats
    stats["do_stats"] = do_stats
    stats["hour_pair_stats"] = hour_pair
    stats["global_median"] = float(train["duration_seconds"].median())
    stats["global_mean"] = float(train["duration_seconds"].mean())

    return stats


def engineer_features(df: pd.DataFrame, zone_stats: dict, centroids: dict[int, tuple[float, float]]) -> pd.DataFrame:
    """Comprehensive feature engineering."""
    ts = pd.to_datetime(df["requested_at"])

    features = pd.DataFrame()

    # --- Basic features ---
    features["pickup_zone"] = df["pickup_zone"].astype("int32")
    features["dropoff_zone"] = df["dropoff_zone"].astype("int32")
    features["passenger_count"] = df["passenger_count"].astype("int8")

    # --- Temporal features ---
    features["hour"] = ts.dt.hour.astype("int8")
    features["dow"] = ts.dt.dayofweek.astype("int8")
    features["month"] = ts.dt.month.astype("int8")
    features["day"] = ts.dt.day.astype("int8")

    # Cyclical encoding for hour
    features["hour_sin"] = np.sin(2 * np.pi * ts.dt.hour / 24).astype("float32")
    features["hour_cos"] = np.cos(2 * np.pi * ts.dt.hour / 24).astype("float32")

    # Cyclical encoding for day of week
    features["dow_sin"] = np.sin(2 * np.pi * ts.dt.dayofweek / 7).astype("float32")
    features["dow_cos"] = np.cos(2 * np.pi * ts.dt.dayofweek / 7).astype("float32")

    # Cyclical encoding for month
    features["month_sin"] = np.sin(2 * np.pi * ts.dt.month / 12).astype("float32")
    features["month_cos"] = np.cos(2 * np.pi * ts.dt.month / 12).astype("float32")

    # Rush hour flags
    hour = ts.dt.hour
    features["is_morning_rush"] = ((hour >= 7) & (hour <= 9)).astype("int8")
    features["is_evening_rush"] = ((hour >= 16) & (hour <= 19)).astype("int8")
    features["is_night"] = ((hour >= 22) | (hour <= 5)).astype("int8")
    features["is_weekend"] = (ts.dt.dayofweek >= 5).astype("int8")

    # Holiday flag
    date_str = ts.dt.strftime("%Y-%m-%d")
    features["is_holiday"] = date_str.isin(NYC_HOLIDAYS_2023).astype("int8")

    # Minute of day (captures more granular time patterns)
    features["minute_of_day"] = (ts.dt.hour * 60 + ts.dt.minute).astype("int16")

    # --- Zone-pair stats features ---
    pair_stats = zone_stats["pair_stats"]
    pu_stats = zone_stats["pu_stats"]
    do_stats = zone_stats["do_stats"]
    hour_pair_stats = zone_stats["hour_pair_stats"]
    global_median = zone_stats["global_median"]

    # Merge zone-pair stats
    merge_df = df[["pickup_zone", "dropoff_zone"]].copy()
    merge_df = merge_df.merge(pair_stats, on=["pickup_zone", "dropoff_zone"], how="left")
    features["pair_median"] = merge_df["pair_median"].fillna(global_median).astype("float32")
    features["pair_mean"] = merge_df["pair_mean"].fillna(zone_stats["global_mean"]).astype("float32")
    features["pair_count"] = merge_df["pair_count"].fillna(0).astype("float32")
    features["pair_q25"] = merge_df["pair_q25"].fillna(global_median * 0.6).astype("float32")
    features["pair_q75"] = merge_df["pair_q75"].fillna(global_median * 1.5).astype("float32")
    features["pair_iqr"] = (features["pair_q75"] - features["pair_q25"]).astype("float32")

    # Log of pair count (trip frequency as a feature)
    features["log_pair_count"] = np.log1p(features["pair_count"]).astype("float32")

    # Merge pickup zone stats
    merge_pu = df[["pickup_zone"]].copy()
    merge_pu = merge_pu.merge(pu_stats, left_on="pickup_zone", right_on="zone", how="left")
    features["pu_median"] = merge_pu["pu_median"].fillna(global_median).astype("float32")
    features["pu_count"] = merge_pu["pu_count"].fillna(0).astype("float32")

    # Merge dropoff zone stats
    merge_do = df[["dropoff_zone"]].copy()
    merge_do = merge_do.merge(do_stats, left_on="dropoff_zone", right_on="zone", how="left")
    features["do_median"] = merge_do["do_median"].fillna(global_median).astype("float32")
    features["do_count"] = merge_do["do_count"].fillna(0).astype("float32")

    # Hour-specific zone-pair median
    merge_hour = df[["pickup_zone", "dropoff_zone"]].copy()
    merge_hour["hour"] = ts.dt.hour.values
    merge_hour = merge_hour.merge(
        hour_pair_stats,
        on=["pickup_zone", "dropoff_zone", "hour"],
        how="left"
    )
    features["hour_pair_median"] = merge_hour["hour_pair_median"].fillna(features["pair_median"]).astype("float32")

    # Ratio: hour_pair_median / pair_median (how much does this hour differ from average?)
    features["hour_pair_ratio"] = (features["hour_pair_median"] / (features["pair_median"] + 1)).astype("float32")

    # Same zone flag (very short trips)
    features["same_zone"] = (df["pickup_zone"] == df["dropoff_zone"]).astype("int8")

    # --- Geo features ---
    if centroids:
        pu_lats = df["pickup_zone"].map(lambda z: centroids.get(z, (40.75, -73.97))[0]).astype("float32")
        pu_lons = df["pickup_zone"].map(lambda z: centroids.get(z, (40.75, -73.97))[1]).astype("float32")
        do_lats = df["dropoff_zone"].map(lambda z: centroids.get(z, (40.75, -73.97))[0]).astype("float32")
        do_lons = df["dropoff_zone"].map(lambda z: centroids.get(z, (40.75, -73.97))[1]).astype("float32")

        features["pu_lat"] = pu_lats
        features["pu_lon"] = pu_lons
        features["do_lat"] = do_lats
        features["do_lon"] = do_lons

        # Haversine distance
        dlat = np.radians(do_lats - pu_lats)
        dlon = np.radians(do_lons - pu_lons)
        a = np.sin(dlat / 2) ** 2 + np.cos(np.radians(pu_lats)) * np.cos(np.radians(do_lats)) * np.sin(dlon / 2) ** 2
        features["haversine_km"] = (6371.0 * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))).astype("float32")

        # Manhattan distance proxy (sum of abs lat/lon differences)
        features["manhattan_proxy"] = (np.abs(do_lats - pu_lats) + np.abs(do_lons - pu_lons)).astype("float32")

        # Direction (bearing)
        features["bearing"] = np.degrees(np.arctan2(
            np.sin(np.radians(do_lons - pu_lons)) * np.cos(np.radians(do_lats)),
            np.cos(np.radians(pu_lats)) * np.sin(np.radians(do_lats)) -
            np.sin(np.radians(pu_lats)) * np.cos(np.radians(do_lats)) * np.cos(np.radians(do_lons - pu_lons))
        )).astype("float32")

    return features


def main() -> None:
    train_path = DATA_DIR / "train.parquet"
    dev_path = DATA_DIR / "dev.parquet"
    for p in (train_path, dev_path):
        if not p.exists():
            raise SystemExit(f"Missing {p.name}. Run `python data/download_data.py` first.")

    print("Loading data...")
    train = pd.read_parquet(train_path)
    dev = pd.read_parquet(dev_path)
    print(f"  train: {len(train):,} rows")
    print(f"  dev:   {len(dev):,} rows")

    # --- Additional data cleaning ---
    print("\nAdditional data cleaning...")
    # Remove extreme outliers (< 1min or > 2 hours for most trips)
    q99 = train["duration_seconds"].quantile(0.99)
    q01 = train["duration_seconds"].quantile(0.01)
    before = len(train)
    train = train[(train["duration_seconds"] >= q01) & (train["duration_seconds"] <= q99)].reset_index(drop=True)
    print(f"  Removed {before - len(train):,} outlier rows (keeping [{q01:.0f}s, {q99:.0f}s])")

    # --- Compute zone statistics from training data ---
    print("\nComputing zone statistics...")
    zone_stats = compute_zone_pair_stats(train)

    # --- Load geo data ---
    print("\nLoading zone centroids...")
    centroids = load_zone_centroids()
    print(f"  Loaded {len(centroids)} zone centroids")

    # --- Feature engineering ---
    print("\nEngineering features...")
    t0 = time.time()
    X_train = engineer_features(train, zone_stats, centroids)
    y_train = train["duration_seconds"].to_numpy()
    X_dev = engineer_features(dev, zone_stats, centroids)
    y_dev = dev["duration_seconds"].to_numpy()
    print(f"  Feature engineering: {time.time() - t0:.1f}s")
    print(f"  Features: {list(X_train.columns)}")
    print(f"  Feature count: {X_train.shape[1]}")

    # --- Train LightGBM ---
    print("\nTraining LightGBM...")
    train_data = lgb.Dataset(X_train, label=y_train)
    dev_data = lgb.Dataset(X_dev, label=y_dev, reference=train_data)

    params = {
        "objective": "regression_l1",  # MAE objective (directly optimizes for the metric)
        "metric": "mae",
        "num_leaves": 255,
        "max_depth": -1,
        "learning_rate": 0.05,
        "n_estimators": 2000,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_samples": 100,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "n_jobs": -1,
        "random_state": 42,
        "verbose": -1,
    }

    callbacks = [
        lgb.log_evaluation(100),
        lgb.early_stopping(50),
    ]

    t0 = time.time()
    model = lgb.train(
        params,
        train_data,
        num_boost_round=2000,
        valid_sets=[dev_data],
        valid_names=["dev"],
        callbacks=callbacks,
    )
    print(f"  Trained in {time.time() - t0:.0f}s  (best iteration: {model.best_iteration})")

    # --- Evaluate ---
    preds = model.predict(X_dev, num_iteration=model.best_iteration)
    mae = float(np.mean(np.abs(preds - y_dev)))
    print(f"\nDev MAE: {mae:.1f} seconds")

    # --- Save model + zone stats + centroids ---
    model_bundle = {
        "model": model,
        "zone_stats": zone_stats,
        "centroids": centroids,
        "feature_names": list(X_train.columns),
    }
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model_bundle, f)
    print(f"Saved model bundle to {MODEL_PATH}")

    # --- Feature importance ---
    print("\nTop 15 features by importance:")
    importance = model.feature_importance(importance_type="gain")
    feature_names = X_train.columns.tolist()
    sorted_idx = np.argsort(importance)[::-1]
    for i in sorted_idx[:15]:
        print(f"  {feature_names[i]:30s}  {importance[i]:,.0f}")


if __name__ == "__main__":
    main()
