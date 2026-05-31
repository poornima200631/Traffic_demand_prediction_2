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

# Day features

day_map = {
    "Monday": 0,
    "Tuesday": 1,
    "Wednesday": 2,
    "Thursday": 3,
    "Friday": 4,
    "Saturday": 5,
    "Sunday": 6
}

train["day_num"] = train["day"].map(day_map)

train["is_weekend"] = (
    train["day_num"] >= 5
).astype(int)

# Timestamp features
train[["hour", "minute"]] = train["timestamp"].str.split(
    ":", expand=True
).astype(int)

train["time_slot"] = (
    train["hour"] * 4
    + train["minute"] // 15
)

train["rush_hour"] = (
    (
        (train["hour"] >= 7)
        & (train["hour"] <= 10)
    )
    |
    (
        (train["hour"] >= 17)
        & (train["hour"] <= 20)
    )
).astype(int)

train["part_of_day"] = pd.cut(
    train["hour"],
    bins=[0,6,12,17,21,24],
    labels=[
        "Night",
        "Morning",
        "Afternoon",
        "Evening",
        "LateNight"
    ],
    include_lowest=True
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


train["geo_time"] = (
    train["geohash"]
    + "_"
    + train["time_slot"].astype(str)
)  


train["high_capacity_road"] = (
    (train["NumberofLanes"] >= 4).astype(str)
)

# Average demand per geohash

geo_mean = train.groupby(
    "geohash"
)["demand"].mean()

train["geo_avg_demand"] = (
    train["geohash"]
    .map(geo_mean)
)

# Average demand per road type

road_mean = train.groupby(
    "RoadType"
)["demand"].mean()

train["road_avg_demand"] = (
    train["RoadType"]
    .map(road_mean)
)



print(train[[
    "geo_avg_demand",
    "road_avg_demand",
]].isnull().sum())
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
    "geo_time",
    "high_capacity_road",
    "part_of_day"
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
    iterations=3000,
    depth=8,
    learning_rate=0.02,
    loss_function="RMSE",
    eval_metric="R2",
    random_seed=42,
    verbose=200
)

# Train
model.fit(
    X_train,
    y_train,
    eval_set=(X_val, y_val),
    cat_features=cat_features,
    use_best_model=True,
    early_stopping_rounds=300
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
test["day_num"] = test["day"].map(day_map)

test["is_weekend"] = (
    test["day_num"] >= 5
).astype(int)

test[["hour", "minute"]] = test["timestamp"].str.split(
    ":", expand=True
).astype(int)

test["time_slot"] = (
    test["hour"] * 4
    + test["minute"] // 15
)
test["rush_hour"] = (
    (
        (test["hour"] >= 7)
        & (test["hour"] <= 10)
    )
    |
    (
        (test["hour"] >= 17)
        & (test["hour"] <= 20)
    )
).astype(int)

test["part_of_day"] = pd.cut(
    test["hour"],
    bins=[0,6,12,17,21,24],
    labels=[
        "Night",
        "Morning",
        "Afternoon",
        "Evening",
        "LateNight"
    ],
    include_lowest=True
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

test["geo_time"] = (
    test["geohash"]
    + "_"
    + test["time_slot"].astype(str)
)

test["high_capacity_road"] = (
    (test["NumberofLanes"] >= 4).astype(str)
)

test["geo_avg_demand"] = (
    test["geohash"]
    .map(geo_mean)
)

test["road_avg_demand"] = (
    test["RoadType"]
    .map(road_mean)
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


