"""Submission interface — this is what Gobblecube's grader imports.

The grader will call `predict` once per held-out request. The signature below
is fixed; everything else (model type, preprocessing, etc.) is yours to change.
"""

from __future__ import annotations

import math
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

_MODEL_PATH = Path(__file__).parent / "model.pkl"

with open(_MODEL_PATH, "rb") as _f:
    _BUNDLE = pickle.load(_f)

# Support both old XGBoost model and new LightGBM bundle
if isinstance(_BUNDLE, dict) and "model" in _BUNDLE:
    _MODEL = _BUNDLE["model"]
    _ZONE_STATS = _BUNDLE["zone_stats"]
    _CENTROIDS = _BUNDLE["centroids"]
    _FEATURE_NAMES = _BUNDLE["feature_names"]
    _IS_IMPROVED = True
else:
    _MODEL = _BUNDLE
    _IS_IMPROVED = False
    if hasattr(_MODEL, "get_booster"):
        _MODEL.get_booster().feature_names = None

# NYC holidays used during training
NYC_HOLIDAYS_2023 = {
    "2023-01-01", "2023-01-02", "2023-01-16", "2023-02-20",
    "2023-05-29", "2023-06-19", "2023-07-04", "2023-09-04",
    "2023-10-09", "2023-11-10", "2023-11-23", "2023-11-24",
    "2023-12-25", "2023-12-26", "2023-12-31",
    # 2024 holidays for eval
    "2024-01-01", "2024-01-15", "2024-02-19",
    "2024-05-27", "2024-06-19", "2024-07-04", "2024-09-02",
    "2024-10-14", "2024-11-11", "2024-11-28", "2024-11-29",
    "2024-12-25", "2024-12-26", "2024-12-31",
}


def _predict_improved(request: dict) -> float:
    """Predict using improved LightGBM model with rich features."""
    ts = datetime.fromisoformat(request["requested_at"])
    pz = int(request["pickup_zone"])
    dz = int(request["dropoff_zone"])
    pc = int(request["passenger_count"])
    hour = ts.hour
    dow = ts.weekday()
    month = ts.month
    day = ts.day

    # Build feature dict
    feats = {}
    feats["pickup_zone"] = pz
    feats["dropoff_zone"] = dz
    feats["passenger_count"] = pc
    feats["hour"] = hour
    feats["dow"] = dow
    feats["month"] = month
    feats["day"] = day

    # Cyclical features
    feats["hour_sin"] = math.sin(2 * math.pi * hour / 24)
    feats["hour_cos"] = math.cos(2 * math.pi * hour / 24)
    feats["dow_sin"] = math.sin(2 * math.pi * dow / 7)
    feats["dow_cos"] = math.cos(2 * math.pi * dow / 7)
    feats["month_sin"] = math.sin(2 * math.pi * month / 12)
    feats["month_cos"] = math.cos(2 * math.pi * month / 12)

    # Rush hour / time flags
    feats["is_morning_rush"] = 1 if 7 <= hour <= 9 else 0
    feats["is_evening_rush"] = 1 if 16 <= hour <= 19 else 0
    feats["is_night"] = 1 if hour >= 22 or hour <= 5 else 0
    feats["is_weekend"] = 1 if dow >= 5 else 0

    # Holiday
    date_str = ts.strftime("%Y-%m-%d")
    feats["is_holiday"] = 1 if date_str in NYC_HOLIDAYS_2023 else 0

    # Minute of day
    feats["minute_of_day"] = hour * 60 + ts.minute

    # Zone-pair stats
    pair_stats = _ZONE_STATS["pair_stats"]
    global_median = _ZONE_STATS["global_median"]
    global_mean = _ZONE_STATS["global_mean"]

    pair_row = pair_stats[
        (pair_stats["pickup_zone"] == pz) & (pair_stats["dropoff_zone"] == dz)
    ]
    if len(pair_row) > 0:
        row = pair_row.iloc[0]
        feats["pair_median"] = float(row["pair_median"])
        feats["pair_mean"] = float(row["pair_mean"])
        feats["pair_count"] = float(row["pair_count"])
        feats["pair_q25"] = float(row["pair_q25"])
        feats["pair_q75"] = float(row["pair_q75"])
    else:
        feats["pair_median"] = global_median
        feats["pair_mean"] = global_mean
        feats["pair_count"] = 0.0
        feats["pair_q25"] = global_median * 0.6
        feats["pair_q75"] = global_median * 1.5

    feats["pair_iqr"] = feats["pair_q75"] - feats["pair_q25"]
    feats["log_pair_count"] = math.log1p(feats["pair_count"])

    # Pickup zone stats
    pu_stats = _ZONE_STATS["pu_stats"]
    pu_row = pu_stats[pu_stats["zone"] == pz]
    if len(pu_row) > 0:
        feats["pu_median"] = float(pu_row.iloc[0]["pu_median"])
        feats["pu_count"] = float(pu_row.iloc[0]["pu_count"])
    else:
        feats["pu_median"] = global_median
        feats["pu_count"] = 0.0

    # Dropoff zone stats
    do_stats = _ZONE_STATS["do_stats"]
    do_row = do_stats[do_stats["zone"] == dz]
    if len(do_row) > 0:
        feats["do_median"] = float(do_row.iloc[0]["do_median"])
        feats["do_count"] = float(do_row.iloc[0]["do_count"])
    else:
        feats["do_median"] = global_median
        feats["do_count"] = 0.0

    # Hour-pair stats
    hour_pair_stats = _ZONE_STATS["hour_pair_stats"]
    hp_row = hour_pair_stats[
        (hour_pair_stats["pickup_zone"] == pz) &
        (hour_pair_stats["dropoff_zone"] == dz) &
        (hour_pair_stats["hour"] == hour)
    ]
    if len(hp_row) > 0:
        feats["hour_pair_median"] = float(hp_row.iloc[0]["hour_pair_median"])
    else:
        feats["hour_pair_median"] = feats["pair_median"]

    feats["hour_pair_ratio"] = feats["hour_pair_median"] / (feats["pair_median"] + 1)

    # Same zone flag
    feats["same_zone"] = 1 if pz == dz else 0

    # Geo features
    if _CENTROIDS:
        pu_lat, pu_lon = _CENTROIDS.get(pz, (40.75, -73.97))
        do_lat, do_lon = _CENTROIDS.get(dz, (40.75, -73.97))
        feats["pu_lat"] = pu_lat
        feats["pu_lon"] = pu_lon
        feats["do_lat"] = do_lat
        feats["do_lon"] = do_lon

        # Haversine
        dlat = math.radians(do_lat - pu_lat)
        dlon = math.radians(do_lon - pu_lon)
        a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(pu_lat)) * math.cos(math.radians(do_lat)) * math.sin(dlon / 2) ** 2
        feats["haversine_km"] = 6371.0 * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        # Manhattan proxy
        feats["manhattan_proxy"] = abs(do_lat - pu_lat) + abs(do_lon - pu_lon)

        # Bearing
        feats["bearing"] = math.degrees(math.atan2(
            math.sin(math.radians(do_lon - pu_lon)) * math.cos(math.radians(do_lat)),
            math.cos(math.radians(pu_lat)) * math.sin(math.radians(do_lat)) -
            math.sin(math.radians(pu_lat)) * math.cos(math.radians(do_lat)) * math.cos(math.radians(do_lon - pu_lon))
        ))

    # Build feature array in the same order as training
    x = np.array([[feats.get(f, 0.0) for f in _FEATURE_NAMES]], dtype=np.float32)
    return float(_MODEL.predict(x)[0])


def _predict_baseline(request: dict) -> float:
    """Original baseline prediction."""
    ts = datetime.fromisoformat(request["requested_at"])
    x = np.array(
        [[
            int(request["pickup_zone"]),
            int(request["dropoff_zone"]),
            ts.hour,
            ts.weekday(),
            ts.month,
            int(request["passenger_count"]),
        ]],
        dtype=np.int32,
    )
    return float(_MODEL.predict(x)[0])


def predict(request: dict) -> float:
    """Predict trip duration in seconds.

    Input schema:
        {
            "pickup_zone":     int,   # NYC taxi zone, 1-265
            "dropoff_zone":    int,
            "requested_at":    str,   # ISO 8601 datetime
            "passenger_count": int,
        }
    """
    if _IS_IMPROVED:
        return _predict_improved(request)
    else:
        return _predict_baseline(request)
