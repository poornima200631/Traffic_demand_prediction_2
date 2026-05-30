import pandas as pd

train = pd.read_csv("data/raw/train.csv")
test = pd.read_csv("data/raw/test.csv")

print("="*50)
print("TRAIN SHAPE")
print(train.shape)

print("\nTEST SHAPE")
print(test.shape)

print("\nCOLUMNS")
print(train.columns.tolist())

print("\nFIRST 5 ROWS")
print(train.head())

print("\nINFO")
print(train.info())

print("\nMISSING VALUES")
print(train.isnull().sum())

print("\nNUMERICAL SUMMARY")
print(train.describe())
print(train["day"].value_counts())

print("\nTIMESTAMP SAMPLE")
print(train["timestamp"].head(20))

print("\nUNIQUE TIMESTAMPS")
print(train["timestamp"].nunique())

print("\nROAD TYPES")
print(train["RoadType"].value_counts(dropna=False))

print("\nWEATHER")
print(train["Weather"].value_counts(dropna=False))

print("\nUNIQUE GEOHASHES")
print(train["geohash"].nunique())