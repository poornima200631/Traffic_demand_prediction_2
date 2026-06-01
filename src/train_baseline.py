import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score

# ==========================
# LOAD DATA
# ==========================
train = pd.read_csv("data/raw/train.csv")
test = pd.read_csv("data/raw/test.csv")

# ==========================
# MISSING VALUES
# ==========================
for df in [train, test]:
    df["RoadType"] = df["RoadType"].fillna("Unknown")
    df["Weather"] = df["Weather"].fillna("Unknown")
    df["Temperature"] = df["Temperature"].fillna(train["Temperature"].median())

# ==========================
# DAY FEATURES
# ==========================
day_map = {
    "Monday": 0, "Tuesday": 1, "Wednesday": 2,
    "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6
}

for df in [train, test]:
    df["day_num"] = df["day"].map(day_map)
    df["is_weekend"] = (df["day_num"] >= 5).astype(int)

# ==========================
# TIME FEATURES
# ==========================
for df in [train, test]:
    df[["hour", "minute"]] = df["timestamp"].str.split(":", expand=True).astype(int)

    df["time_slot"] = df["hour"] * 4 + df["minute"] // 15

    df["rush_hour"] = (
        df["hour"].between(7, 10) |
        df["hour"].between(17, 20)
    ).astype(int)

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    df["is_night"] = ((df["hour"] >= 22) | (df["hour"] <= 5)).astype(int)
    df["is_morning"] = df["hour"].between(6, 11).astype(int)
    df["is_evening"] = df["hour"].between(17, 21).astype(int)

# ==========================
# INTERACTION FEATURES
# ==========================
for df in [train, test]:
    df["road_lane"] = df["RoadType"].astype(str) + "_" + df["NumberofLanes"].astype(str)
    df["geo_road"] = df["geohash"].astype(str) + "_" + df["RoadType"].astype(str)
    df["geo_time"] = df["geohash"].astype(str) + "_" + df["time_slot"].astype(str)
    df["geo_hour"] = df["geohash"].astype(str) + "_" + df["hour"].astype(str)
    df["road_weather"] = df["RoadType"].astype(str) + "_" + df["Weather"].astype(str)
    df["lane_weather"] = df["NumberofLanes"].astype(str) + "_" + df["Weather"].astype(str)

    df["high_capacity_road"] = (df["NumberofLanes"] >= 4).astype(int)

# ==========================
# GLOBAL STAT FEATURES (FIXED SAFE VERSION)
# ==========================

# GEO FEATURES
geo_mean_map = train.groupby("geohash")["demand"].mean()
geo_std_map = train.groupby("geohash")["demand"].std()

train["geo_mean"] = train["geohash"].map(geo_mean_map)
train["geo_std"] = train["geohash"].map(geo_std_map)

test["geo_mean"] = test["geohash"].map(geo_mean_map)
test["geo_std"] = test["geohash"].map(geo_std_map)

# ROAD FEATURES
road_mean_map = train.groupby("RoadType")["demand"].mean()

train["road_mean"] = train["RoadType"].map(road_mean_map)
test["road_mean"] = test["RoadType"].map(road_mean_map)

# HANDLE MISSING VALUES (IMPORTANT)
for df in [train, test]:
    df["geo_mean"] = df["geo_mean"].fillna(train["demand"].mean())
    df["geo_std"] = df["geo_std"].fillna(0)
    df["road_mean"] = df["road_mean"].fillna(train["demand"].mean())

# ==========================
# TARGET ENCODING (SAFE)
# ==========================
def target_encode(train, test, col, target):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    global_mean = train[target].mean()

    train_encoded = np.zeros(len(train))
    test_encoded = np.zeros(len(test))

    for tr_idx, val_idx in kf.split(train):
        X_tr, X_val = train.iloc[tr_idx], train.iloc[val_idx]

        means = X_tr.groupby(col)[target].mean()

        train_encoded[val_idx] = X_val[col].map(means).fillna(global_mean)
        test_encoded += test[col].map(means).fillna(global_mean) / 5

    return train_encoded, test_encoded


train["geo_te"], test["geo_te"] = target_encode(train, test, "geohash", "demand")
train["road_te"], test["road_te"] = target_encode(train, test, "RoadType", "demand")

# ==========================
# CLEAN FEATURES
# ==========================
drop_cols = ["demand", "Index", "timestamp", "hour", "minute"]
X = train.drop(columns=drop_cols)
y = train["demand"]

# ==========================
# CRITICAL FIX: FORCE ALL STRING CATS
# ==========================
cat_features = [
    "geohash",
    "RoadType",
    "LargeVehicles",
    "Landmarks",
    "Weather",
    "road_lane",
    "geo_road",
    "geo_time",
    "geo_hour",
    "road_weather",
    "lane_weather"
]

for col in cat_features:
    X[col] = X[col].astype(str)
    test[col] = test[col].astype(str)

# ==========================
# TRAIN VALID SPLIT
# ==========================
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==========================
# MODEL (STABLE BEST VERSION)
# ==========================
model = CatBoostRegressor(
    iterations=6000,
    learning_rate=0.04,
    depth=6,
    loss_function="RMSE",
    eval_metric="R2",
    random_strength=2,
    bagging_temperature=1,
    l2_leaf_reg=5,
    verbose=500,
    early_stopping_rounds=300
)

# ==========================
# TRAIN
# ==========================
model.fit(
    X_train, y_train,
    eval_set=(X_val, y_val),
    cat_features=cat_features,
    use_best_model=True
)

# ==========================
# VALIDATION
# ==========================
preds = model.predict(X_val)
print("\nR2 Score:", r2_score(y_val, preds))

# ==========================
# FEATURE IMPORTANCE
# ==========================
importance = model.get_feature_importance()

print("\nTop Features:")
for f, imp in sorted(zip(X.columns, importance), key=lambda x: x[1], reverse=True)[:20]:
    print(f"{f}: {imp:.2f}")

# ==========================
# FINAL TEST PREDICTION (FIXED)
# ==========================
test_preds = model.predict(test[X.columns])

submission = pd.DataFrame({
    "Index": pd.read_csv("data/raw/test.csv")["Index"],
    "demand": test_preds
})

submission.to_csv("submissions/submission.csv", index=False)

model.save_model("models/catboost_final.cbm")

print("\nDone successfully!")