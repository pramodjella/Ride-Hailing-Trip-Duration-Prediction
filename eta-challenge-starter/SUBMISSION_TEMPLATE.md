# ETA Challenge Submission

## Your final score

Dev MAE: **258.4 s** (vs baseline 354.2 s — a **96 s improvement**)

---

## Your approach, in one paragraph

Built a LightGBM model with 40 engineered features, replacing the baseline's 6-feature XGBoost.
The single biggest improvement came from **zone-pair historical statistics** — pre-computing
median/mean/q25/q75 trip durations for every (pickup, dropoff) zone pair from the training data.
This alone captures most of the spatial structure (the baseline's README notes that a naive 10-line
zone-pair lookup already beats the GBT baseline). On top of that, I added **hour-specific zone-pair
medians** (captures rush-hour congestion patterns per route), **geo features** from the official
NYC taxi zone shapefile (haversine distance, Manhattan-distance proxy, bearing between centroids),
and **rich temporal features** (cyclical sin/cos encoding for hour/dow/month, rush-hour flags,
weekend/holiday flags, minute-of-day). The model uses `regression_l1` (MAE) as the objective
to directly optimize the scoring metric. Outlier trips (<1st and >99th percentile) were removed
from training.

## What you tried that didn't work

1. **Using the full feature set without zone-pair stats**: Pure temporal + geo features without
   the historical statistics only marginally beat the baseline. The zone-pair statistics are the
   dominant feature by far (hour_pair_median has ~300M importance vs ~5M for the next feature).

2. **XGBoost with the same features**: Tried XGBoost with identical features — LightGBM trained
   faster and produced marginally better results on this dataset size (36M rows).

## Where AI tooling sped you up most

- **Feature engineering design**: AI helped rapidly enumerate which features would matter for
  ride-hailing ETA prediction, particularly the insight that zone-pair historical statistics
  would dominate and that the baseline's poor performance was due to missing this obvious signal.
- **Debugging the predict.py optimization**: Converting from per-row DataFrame lookups to
  hash-based dictionaries required careful alignment of feature names — AI caught ordering bugs.
- **Dockerfile and requirements tuning**: Quickly generated correct dependency specs.

## Next experiments

1. **Day-of-week × hour interaction features**: Different weekday rush hours vs weekend patterns
2. **Weather data integration**: Join NOAA hourly weather observations (rain/snow significantly
   impact NYC taxi durations)
3. **Time-window zone-pair stats**: Use hour-of-day AND day-of-week specific medians
4. **Neural network embedding approach**: Learn zone embeddings via a small MLP that captures
   latent spatial relationships beyond simple centroids
5. **Ensemble**: Blend LightGBM with a separate zone-pair-median fallback for rare routes

## How to reproduce

```bash
# Setup
python -m venv .venv && .venv\Scripts\activate  # Windows
pip install -r requirements.txt

# Download data (~500 MB, one-time)
python data/download_data.py

# Train improved model (produces model.pkl)
python train_improved.py

# Grade on Dev set
python grade.py

# Run contract tests
python -m pytest tests/
```

---

_Total time spent on this challenge: ~3 hours._
