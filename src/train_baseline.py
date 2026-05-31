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
train["RoadType"] = train["RoadType"].fillna("Unknown")
train["Weather"] = train["Weather"].fillna("Unknown")
train["Temperature"] = train["Temperature"].fillna(train["Temperature"].median())

test["RoadType"] = test["RoadType"].fillna("Unknown")
test["Weather"] = test["Weather"].fillna("Unknown")
test["Temperature"] = test["Temperature"].fillna(train["Temperature"].median())

# ==========================
# BASIC FEATURES
# ==========================
day_map = {
    "Monday": 0, "Tuesday": 1, "Wednesday": 2,
    "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6
}

train["day_num"] = train["day"].map(day_map)
test["day_num"] = test["day"].map(day_map)

train["is_weekend"] = (train["day_num"] >= 5).astype(int)
test["is_weekend"] = (test["day_num"] >= 5).astype(int)

# ==========================
# TIME FEATURES
# ==========================
train[["hour", "minute"]] = train["timestamp"].str.split(":", expand=True).astype(int)
test[["hour", "minute"]] = test["timestamp"].str.split(":", expand=True).astype(int)

train["time_slot"] = train["hour"] * 4 + train["minute"] // 15
test["time_slot"] = test["hour"] * 4 + test["minute"] // 15

train["rush_hour"] = ((train["hour"].between(7,10)) | (train["hour"].between(17,20))).astype(int)
test["rush_hour"] = ((test["hour"].between(7,10)) | (test["hour"].between(17,20))).astype(int)

# Cyclical encoding
train["hour_sin"] = np.sin(2 * np.pi * train["hour"] / 24)
train["hour_cos"] = np.cos(2 * np.pi * train["hour"] / 24)

test["hour_sin"] = np.sin(2 * np.pi * test["hour"] / 24)
test["hour_cos"] = np.cos(2 * np.pi * test["hour"] / 24)

# ==========================
# INTERACTION FEATURES
# ==========================
train["road_lane"] = train["RoadType"] + "_" + train["NumberofLanes"].astype(str)
test["road_lane"] = test["RoadType"] + "_" + test["NumberofLanes"].astype(str)

train["geo_road"] = train["geohash"] + "_" + train["RoadType"]
test["geo_road"] = test["geohash"] + "_" + test["RoadType"]

train["geo_time"] = train["geohash"] + "_" + train["time_slot"].astype(str)
test["geo_time"] = test["geohash"] + "_" + test["time_slot"].astype(str)

train["geo_hour"] = train["geohash"] + "_" + train["hour"].astype(str)
test["geo_hour"] = test["geohash"] + "_" + test["hour"].astype(str)

train["road_weather"] = train["RoadType"] + "_" + train["Weather"]
test["road_weather"] = test["RoadType"] + "_" + test["Weather"]

train["lane_weather"] = train["NumberofLanes"].astype(str) + "_" + train["Weather"]
test["lane_weather"] = test["NumberofLanes"].astype(str) + "_" + test["Weather"]

# ==========================
# IMPORTANT FIX
# ==========================
train["high_capacity_road"] = (train["NumberofLanes"] >= 4).astype(int)
test["high_capacity_road"] = (test["NumberofLanes"] >= 4).astype(int)

# ==========================
# TARGET ENCODING (NO LEAKAGE)
# ==========================
def target_encode(train, test, col, target):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    train_encoded = np.zeros(len(train))
    test_encoded = np.zeros(len(test))

    for tr_idx, val_idx in kf.split(train):
        X_tr, X_val = train.iloc[tr_idx], train.iloc[val_idx]

        means = X_tr.groupby(col)[target].mean()

        train_encoded[val_idx] = X_val[col].map(means)
        test_encoded += test[col].map(means).fillna(train[target].mean()) / 5

    return train_encoded, test_encoded


train["geo_te"], test["geo_te"] = target_encode(train, test, "geohash", "demand")
train["road_te"], test["road_te"] = target_encode(train, test, "RoadType", "demand")

# ==========================
# SPLIT FEATURES
# ==========================
X = train.drop(columns=["demand"])
y = train["demand"]

X = X.drop(columns=["Index", "timestamp", "hour", "minute"])

# ==========================
# CATEGORICAL FEATURES (FIXED)
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

# ==========================
# TRAIN/VALID SPLIT
# ==========================
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

# ==========================
# MODEL
# ==========================
model = CatBoostRegressor(
    iterations=8000,
    learning_rate=0.03,
    depth=8,
    l2_leaf_reg=3,
    random_strength=3,
    bagging_temperature=1,
    loss_function="RMSE",
    eval_metric="R2",
    verbose=500,
    early_stopping_rounds=500
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

print("\nFeature Importance:")
for f, imp in sorted(zip(X.columns, importance), key=lambda x: x[1], reverse=True):
    print(f"{f}: {imp:.2f}")

# ==========================
# TEST PREPARATION
# ==========================
test["geo_te"] = test["geohash"].map(train.groupby("geohash")["demand"].mean()).fillna(train["demand"].mean())
test["road_te"] = test["RoadType"].map(train.groupby("RoadType")["demand"].mean()).fillna(train["demand"].mean())

test = test[X.columns]

# ==========================
# PREDICTION
# ==========================
test_preds = model.predict(test)

submission = pd.DataFrame({
    "Index": pd.read_csv("data/raw/test.csv")["Index"],
    "demand": test_preds
})

submission.to_csv("submissions/submission.csv", index=False)

model.save_model("models/catboost_v1.cbm")

print("\nDone!")