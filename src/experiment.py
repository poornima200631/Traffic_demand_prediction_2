import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score

# ==========================
# LOAD DATA
# ==========================
print("Loading data...")
train = pd.read_csv("data/raw/train.csv")

# ==========================
# MISSING VALUES
# ==========================
train["RoadType"] = train["RoadType"].fillna("Unknown")
train["Weather"] = train["Weather"].fillna("Unknown")
train["Temperature"] = train["Temperature"].fillna(train["Temperature"].median())

# ==========================
# TIME FEATURES
# ==========================
train[["hour", "minute"]] = train["timestamp"].str.split(":", expand=True).astype(int)
train["time_slot"] = train["hour"] * 4 + train["minute"] // 15
train["rush_hour"] = ((train["hour"].between(7,10)) | (train["hour"].between(17,20))).astype(int)

# Cyclical encoding
train["hour_sin"] = np.sin(2 * np.pi * train["hour"] / 24)
train["hour_cos"] = np.cos(2 * np.pi * train["hour"] / 24)

# ==========================
# FIX DAY & WEEKEND
# ==========================
train["day_of_week"] = train["day"] % 7
train["is_weekend"] = train["day_of_week"].isin([5, 6]).astype(int)

# ==========================
# GEOHASH CHARACTERS
# ==========================
train["geo_c4"] = train["geohash"].str[3]
train["geo_c5"] = train["geohash"].str[4]
train["geo_c6"] = train["geohash"].str[5]
train["geo_c45"] = train["geohash"].str[3:5]

# ==========================
# INTERACTION FEATURES
# ==========================
train["road_lane"] = train["RoadType"] + "_" + train["NumberofLanes"].astype(str)
train["road_weather"] = train["RoadType"] + "_" + train["Weather"]
train["lane_weather"] = train["NumberofLanes"].astype(str) + "_" + train["Weather"]
train["high_capacity_road"] = (train["NumberofLanes"] >= 4).astype(int)

# ==========================
# SPLIT TRAIN (DAY 48) & VAL (DAY 49)
# ==========================
train_df = train[train["day"] == 48].copy()
val_df = train[train["day"] == 49].copy()

print(f"Train (Day 48) shape: {train_df.shape}")
print(f"Val (Day 49) shape: {val_df.shape}")

# ==========================
# TARGET ENCODING (NO LEAKAGE)
# ==========================
def robust_target_encode(train_df, val_df, col, target, m=10):
    global_mean = train_df[target].mean()
    
    # 5-Fold out-of-fold for train_df
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    train_encoded = np.zeros(len(train_df))
    
    for tr_idx, val_idx in kf.split(train_df):
        X_tr = train_df.iloc[tr_idx]
        X_val = train_df.iloc[val_idx]
        
        stats = X_tr.groupby(col)[target].agg(["count", "mean"])
        smoothed_means = (stats["count"] * stats["mean"] + m * global_mean) / (stats["count"] + m)
        train_encoded[val_idx] = X_val[col].map(smoothed_means).fillna(global_mean)
        
    # Map val_df using full train_df
    stats_full = train_df.groupby(col)[target].agg(["count", "mean"])
    smoothed_means_full = (stats_full["count"] * stats_full["mean"] + m * global_mean) / (stats_full["count"] + m)
    val_encoded = val_df[col].map(smoothed_means_full).fillna(global_mean)
    
    return train_encoded, val_encoded

train_df["geo_te"], val_df["geo_te"] = robust_target_encode(train_df, val_df, "geohash", "demand")
train_df["road_te"], val_df["road_te"] = robust_target_encode(train_df, val_df, "RoadType", "demand")

# Let's also do a smoothed target encoding for the interaction geo_time to see if it helps!
train_df["geo_time"] = train_df["geohash"] + "_" + train_df["time_slot"].astype(str)
val_df["geo_time"] = val_df["geohash"] + "_" + val_df["time_slot"].astype(str)
train_df["geo_time_te"], val_df["geo_time_te"] = robust_target_encode(train_df, val_df, "geo_time", "demand", m=5)

# Target encoding for other interactions
train_df["road_lane_te"], val_df["road_lane_te"] = robust_target_encode(train_df, val_df, "road_lane", "demand")
train_df["road_weather_te"], val_df["road_weather_te"] = robust_target_encode(train_df, val_df, "road_weather", "demand")
train_df["lane_weather_te"], val_df["lane_weather_te"] = robust_target_encode(train_df, val_df, "lane_weather", "demand")

# Drop geo_time raw column so it is not in the model
train_df = train_df.drop(columns=["geo_time"])
val_df = val_df.drop(columns=["geo_time"])

# ==========================
# DROP UNNECESSARY COLUMNS
# ==========================
drop_cols = ["Index", "timestamp", "hour", "minute", "demand"]
X_train = train_df.drop(columns=drop_cols)
y_train = train_df["demand"]

X_val = val_df.drop(columns=drop_cols)
y_val = val_df["demand"]

# ==========================
# CATEGORICAL FEATURES
# ==========================
cat_features = [
    "geohash",
    "RoadType",
    "LargeVehicles",
    "Landmarks",
    "Weather",
    "geo_c4",
    "geo_c5",
    "geo_c6",
    "geo_c45",
    "road_lane",
    "road_weather",
    "lane_weather"
]

# We need a validation set inside train_df for early stopping
# Let's split train_df into 80% train and 20% internal validation
from sklearn.model_selection import train_test_split
X_tr, X_val_internal, y_tr, y_val_internal = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, shuffle=True
)

# ==========================
# MODEL
# ==========================
print("Training CatBoost model...")
model = CatBoostRegressor(
    iterations=2000,
    learning_rate=0.08,
    depth=6,
    l2_leaf_reg=5,
    loss_function="RMSE",
    eval_metric="R2",
    verbose=200,
    early_stopping_rounds=200,
    random_seed=42
)

model.fit(
    X_tr, y_tr,
    eval_set=(X_val_internal, y_val_internal),
    cat_features=cat_features,
    use_best_model=True
)

# Predict on Day 49 validation set
preds_val = model.predict(X_val)
r2_baseline = r2_score(y_val, preds_val)
print(f"\nBaseline R2 on Day 49 (no adjustment): {r2_baseline:.5f}")

# ==========================
# MULTIPLIER OPTIMIZATION
# ==========================
best_r2 = -1
best_mult = 1.0

for mult in np.linspace(1.0, 2.5, 31):
    adjusted_preds = preds_val * mult
    r2 = r2_score(y_val, adjusted_preds)
    print(f"Multiplier: {mult:.2f} -> R2 Score: {r2:.5f}")
    if r2 > best_r2:
        best_r2 = r2
        best_mult = mult

print(f"\nOptimal Multiplier: {best_mult:.2f} with R2 Score: {best_r2:.5f}")
