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

    # Pre-build hash-based lookup tables for fast per-request inference
    _PAIR_LOOKUP = {}
    ps = _ZONE_STATS["pair_stats"]
    for _, r in ps.iterrows():
        key = (int(r["pickup_zone"]), int(r["dropoff_zone"]))
        _PAIR_LOOKUP[key] = (
            float(r["pair_median"]), float(r["pair_mean"]),
            float(r["pair_count"]), float(r["pair_q25"]), float(r["pair_q75"])
        )

    _PU_LOOKUP = {}
    for _, r in _ZONE_STATS["pu_stats"].iterrows():
        _PU_LOOKUP[int(r["zone"])] = (float(r["pu_median"]), float(r["pu_count"]))

    _DO_LOOKUP = {}
    for _, r in _ZONE_STATS["do_stats"].iterrows():
        _DO_LOOKUP[int(r["zone"])] = (float(r["do_median"]), float(r["do_count"]))

    _HOUR_PAIR_LOOKUP = {}
    for _, r in _ZONE_STATS["hour_pair_stats"].iterrows():
        key = (int(r["pickup_zone"]), int(r["dropoff_zone"]), int(r["hour"]))
        _HOUR_PAIR_LOOKUP[key] = float(r["hour_pair_median"])

    _GLOBAL_MEDIAN = _ZONE_STATS["global_median"]
    _GLOBAL_MEAN = _ZONE_STATS["global_mean"]

    del _ZONE_STATS  # free memory
else:
    _MODEL = _BUNDLE
    _IS_IMPROVED = False
    if hasattr(_MODEL, "get_booster"):
        _MODEL.get_booster().feature_names = None

# NYC holidays
NYC_HOLIDAYS = {
    "2023-01-01", "2023-01-02", "2023-01-16", "2023-02-20",
    "2023-05-29", "2023-06-19", "2023-07-04", "2023-09-04",
    "2023-10-09", "2023-11-10", "2023-11-23", "2023-11-24",
    "2023-12-25", "2023-12-26", "2023-12-31",
    "2024-01-01", "2024-01-15", "2024-02-19",
    "2024-05-27", "2024-06-19", "2024-07-04", "2024-09-02",
    "2024-10-14", "2024-11-11", "2024-11-28", "2024-11-29",
    "2024-12-25", "2024-12-26", "2024-12-31",
}

# Pre-compute sin/cos tables for cyclical features
_HOUR_SIN = [math.sin(2 * math.pi * h / 24) for h in range(24)]
_HOUR_COS = [math.cos(2 * math.pi * h / 24) for h in range(24)]
_DOW_SIN = [math.sin(2 * math.pi * d / 7) for d in range(7)]
_DOW_COS = [math.cos(2 * math.pi * d / 7) for d in range(7)]
_MONTH_SIN = [math.sin(2 * math.pi * m / 12) for m in range(13)]
_MONTH_COS = [math.cos(2 * math.pi * m / 12) for m in range(13)]


def _predict_improved(request: dict) -> float:
    """Predict using improved LightGBM model with rich features.
    Optimized with hash-based lookups for fast per-request inference.
    """
    ts = datetime.fromisoformat(request["requested_at"])
    pz = int(request["pickup_zone"])
    dz = int(request["dropoff_zone"])
    pc = int(request["passenger_count"])
    hour = ts.hour
    dow = ts.weekday()
    month = ts.month
    day = ts.day

    # Zone-pair stats via hash lookup
    pair_key = (pz, dz)
    if pair_key in _PAIR_LOOKUP:
        p_med, p_mean, p_count, p_q25, p_q75 = _PAIR_LOOKUP[pair_key]
    else:
        p_med = _GLOBAL_MEDIAN
        p_mean = _GLOBAL_MEAN
        p_count = 0.0
        p_q25 = _GLOBAL_MEDIAN * 0.6
        p_q75 = _GLOBAL_MEDIAN * 1.5

    # Pickup / dropoff zone stats
    pu_med, pu_count = _PU_LOOKUP.get(pz, (_GLOBAL_MEDIAN, 0.0))
    do_med, do_count = _DO_LOOKUP.get(dz, (_GLOBAL_MEDIAN, 0.0))

    # Hour-pair stats
    hp_key = (pz, dz, hour)
    hp_med = _HOUR_PAIR_LOOKUP.get(hp_key, p_med)

    # Build feature array directly — order must match _FEATURE_NAMES
    feats = [
        pz,                         # pickup_zone
        dz,                         # dropoff_zone
        pc,                         # passenger_count
        hour,                       # hour
        dow,                        # dow
        month,                      # month
        day,                        # day
        _HOUR_SIN[hour],            # hour_sin
        _HOUR_COS[hour],            # hour_cos
        _DOW_SIN[dow],              # dow_sin
        _DOW_COS[dow],              # dow_cos
        _MONTH_SIN[month],          # month_sin
        _MONTH_COS[month],          # month_cos
        1 if 7 <= hour <= 9 else 0,   # is_morning_rush
        1 if 16 <= hour <= 19 else 0, # is_evening_rush
        1 if hour >= 22 or hour <= 5 else 0,  # is_night
        1 if dow >= 5 else 0,       # is_weekend
        1 if ts.strftime("%Y-%m-%d") in NYC_HOLIDAYS else 0,  # is_holiday
        hour * 60 + ts.minute,      # minute_of_day
        p_med,                      # pair_median
        p_mean,                     # pair_mean
        p_count,                    # pair_count
        p_q25,                      # pair_q25
        p_q75,                      # pair_q75
        p_q75 - p_q25,             # pair_iqr
        math.log1p(p_count),       # log_pair_count
        pu_med,                    # pu_median
        pu_count,                  # pu_count
        do_med,                    # do_median
        do_count,                  # do_count
        hp_med,                    # hour_pair_median
        hp_med / (p_med + 1),      # hour_pair_ratio
        1 if pz == dz else 0,     # same_zone
    ]

    # Geo features
    if _CENTROIDS:
        pu_lat, pu_lon = _CENTROIDS.get(pz, (40.75, -73.97))
        do_lat, do_lon = _CENTROIDS.get(dz, (40.75, -73.97))
        feats.append(pu_lat)
        feats.append(pu_lon)
        feats.append(do_lat)
        feats.append(do_lon)

        # Haversine
        dlat = math.radians(do_lat - pu_lat)
        dlon = math.radians(do_lon - pu_lon)
        a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(pu_lat)) * math.cos(math.radians(do_lat)) * math.sin(dlon / 2) ** 2
        feats.append(6371.0 * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a)))  # haversine_km

        feats.append(abs(do_lat - pu_lat) + abs(do_lon - pu_lon))  # manhattan_proxy

        feats.append(math.degrees(math.atan2(
            math.sin(math.radians(do_lon - pu_lon)) * math.cos(math.radians(do_lat)),
            math.cos(math.radians(pu_lat)) * math.sin(math.radians(do_lat)) -
            math.sin(math.radians(pu_lat)) * math.cos(math.radians(do_lat)) * math.cos(math.radians(do_lon - pu_lon))
        )))  # bearing

    x = np.array([feats], dtype=np.float32)
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
