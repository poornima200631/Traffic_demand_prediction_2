import pandas as pd
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Load data
train = pd.read_csv("data/raw/train.csv")
test = pd.read_csv("data/raw/test.csv")

# Missing values
train["RoadType"] = train["RoadType"].fillna("Unknown")
train["Weather"] = train["Weather"].fillna("Unknown")
train["Temperature"] = train["Temperature"].fillna(
    train["Temperature"].median()
)

# Timestamp features
train[["hour", "minute"]] = train["timestamp"].str.split(
    ":", expand=True
).astype(int)

train["time_slot"] = (
    train["hour"] * 4
    + train["minute"] // 15
)

train["road_lane"] = (
    train["RoadType"].astype(str)
    + "_"
    + train["NumberofLanes"].astype(str)
)

train["geo_road"] = (
    train["geohash"]
    + "_"
    + train["RoadType"].astype(str)
)
train["high_capacity_road"] = (
    (train["NumberofLanes"] >= 4).astype(str)
)
# Features
X = train.drop(columns=["demand"])
y = train["demand"]

# Remove Index
X = X.drop(columns=[
    "Index",
    "timestamp",
    "hour",
    "minute"
])

# Categorical columns
cat_features = [
    "geohash",
    "RoadType",
    "LargeVehicles",
    "Landmarks",
    "Weather",
    "road_lane",
    "geo_road",
    "high_capacity_road"
]

# Split
X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42

)
# Model
model = CatBoostRegressor(
    iterations=1000,
    depth=10,
    learning_rate=0.03,
    loss_function="RMSE",
    eval_metric="R2",
    random_seed=42,
    verbose=100
)

# Train
model.fit(
    X_train,
    y_train,
    cat_features=cat_features
)

# Validation
preds = model.predict(X_val)

print("\nR2 Score:")
print(r2_score(y_val, preds))

# Feature Importance
importance = model.get_feature_importance()

print("\nFeature Importance:")
for feature, imp in sorted(
    zip(X.columns, importance),
    key=lambda x: x[1],
    reverse=True
):
    print(f"{feature}: {imp:.2f}")
# ==========================
# PREPARE TEST DATA
# ==========================

test["RoadType"] = test["RoadType"].fillna("Unknown")
test["Weather"] = test["Weather"].fillna("Unknown")
test["Temperature"] = test["Temperature"].fillna(
    train["Temperature"].median()
)

test[["hour", "minute"]] = test["timestamp"].str.split(
    ":", expand=True
).astype(int)

test["time_slot"] = (
    test["hour"] * 4
    + test["minute"] // 15
)

test["road_lane"] = (
    test["RoadType"].astype(str)
    + "_"
    + test["NumberofLanes"].astype(str)
)

test["geo_road"] = (
    test["geohash"]
    + "_"
    + test["RoadType"].astype(str)
)

test["high_capacity_road"] = (
    (test["NumberofLanes"] >= 4).astype(str)
)

# Save Index before dropping
test_index = test["Index"]

# Keep same columns as training
test = test.drop(columns=[
    "Index",
    "timestamp",
    "hour",
    "minute"
])

test = test[X.columns]

# ==========================
# PREDICT
# ==========================

test_predictions = model.predict(test)

submission = pd.DataFrame({
    "Index": test_index,
    "demand": test_predictions
})

submission.to_csv(
    "submissions/submission.csv",
    index=False
)

print("\nSubmission file created!")

model.save_model("models/catboost_v1.cbm")