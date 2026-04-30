# Improved Dockerfile for the ETA Challenge.
# Uses LightGBM with rich feature engineering.
#
# Build:
#   docker build -t my-eta .
# Test the grader pathway:
#   docker run --rm -v $(pwd)/data:/work my-eta /work/dev.parquet /work/preds.csv

FROM python:3.11-slim

WORKDIR /app

# libgomp1 is required for xgboost/lightgbm at runtime on slim images
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Submission surface: predict.py + grade.py + trained weights
COPY predict.py grade.py ./
COPY model.pkl ./

# Grader invokes:  python grade.py <input.parquet> <output.csv>
ENTRYPOINT ["python", "grade.py"]
