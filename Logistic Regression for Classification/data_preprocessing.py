"""Data preprocessing for the Weather dataset.
This script mirrors the steps from the logistic regression tutorial but avoids
pandas inplace warnings, prevents memory errors, and uses the updated scikit‑learn API.
"""

import os
import gc  # Garbage collection to free up RAM dynamically
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

# ---------------------------------------------------------------------------
# Load raw CSV (same file used in the tutorial)
# ---------------------------------------------------------------------------
DATA_DIR = "."
CSV_PATH = os.path.join(DATA_DIR, "weatherAUS.csv")

if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"Could not find {CSV_PATH}. Please ensure it is in the correct directory.")

print("Loading dataset...")
raw_df = pd.read_csv(CSV_PATH)

# ---------------------------------------------------------------------------
# Drop rows where target columns are missing
# ---------------------------------------------------------------------------
raw_df = raw_df.dropna(subset=["RainToday", "RainTomorrow"]).reset_index(drop=True)

# ---------------------------------------------------------------------------
# Feature Engineering: Handle the Date column to avoid memory explosion
# ---------------------------------------------------------------------------
# Converting string dates to months captures seasonality without creating 140k columns
raw_df["Date"] = pd.to_datetime(raw_df["Date"])
raw_df["Month"] = raw_df["Date"].dt.month
raw_df = raw_df.drop(columns=["Date"])

# ---------------------------------------------------------------------------
# Identify numeric and categorical columns (excluding the target column)
# ---------------------------------------------------------------------------
numeric_cols = raw_df.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols = [c for c in numeric_cols if c != "RainTomorrow"]

categorical_cols = raw_df.select_dtypes(exclude=[np.number]).columns.tolist()
categorical_cols = [c for c in categorical_cols if c not in ["RainTomorrow"]]

# ---------------------------------------------------------------------------
# Fill missing numeric values with median (no inplace warning)
# ---------------------------------------------------------------------------
for col in numeric_cols:
    median_val = raw_df[col].median()
    raw_df[col] = raw_df[col].fillna(median_val)

# ---------------------------------------------------------------------------
# One‑hot encode categorical columns
# ---------------------------------------------------------------------------
encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
cat_encoded = encoder.fit_transform(raw_df[categorical_cols])
cat_feature_names = encoder.get_feature_names_out(categorical_cols)
cat_df = pd.DataFrame(cat_encoded, columns=cat_feature_names, index=raw_df.index)

# ---------------------------------------------------------------------------
# Scale numeric columns to [0, 1]
# ---------------------------------------------------------------------------
scaler = MinMaxScaler()
num_scaled = scaler.fit_transform(raw_df[numeric_cols])
num_df = pd.DataFrame(num_scaled, columns=numeric_cols, index=raw_df.index)

# ---------------------------------------------------------------------------
# Combine features with the target column & Clear Memory
# ---------------------------------------------------------------------------
print("Combining processed features...")
final_df = pd.concat([num_df, cat_df, raw_df[["RainTomorrow"]]], axis=1)

# Clean up intermediate dataframes to free up RAM immediately
del raw_df, num_df, cat_df
gc.collect()

# ---------------------------------------------------------------------------
# Split into train/validation/test sets
# ---------------------------------------------------------------------------
train_val_df, test_df = train_test_split(
    final_df,
    test_size=0.2,
    random_state=42,
    stratify=final_df["RainTomorrow"],
)

del final_df
gc.collect()

train_df, val_df = train_test_split(
    train_val_df,
    test_size=0.25,
    random_state=42,
    stratify=train_val_df["RainTomorrow"],
)

del train_val_df
gc.collect()

print("Train:", train_df.shape, "Validation:", val_df.shape, "Test:", test_df.shape)

# ---------------------------------------------------------------------------
# Save each split to CSV files sequentially to optimize memory footprint
# ---------------------------------------------------------------------------
OUTPUT_DIR = "./processed_weather"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Saving train.csv...")
train_df.to_csv(os.path.join(OUTPUT_DIR, "train.csv"), index=False, chunksize=10000)
del train_df
gc.collect()

print("Saving validation.csv...")
val_df.to_csv(os.path.join(OUTPUT_DIR, "validation.csv"), index=False, chunksize=10000)
del val_df
gc.collect()

print("Saving test.csv...")
test_df.to_csv(os.path.join(OUTPUT_DIR, "test.csv"), index=False, chunksize=10000)
del test_df
gc.collect()

print("Datasets successfully saved to:", OUTPUT_DIR)