import pandas as pd
import numpy as np

from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import RandomForestRegressor

# Load dataset
df = pd.read_csv("data/processed/model_ready_data.csv")

if "date" in df.columns:
    df = df.drop(columns=["date"])

if "stn_code" in df.columns:
    df = df.drop(columns=["stn_code"])

# Encode categorical columns
df = pd.get_dummies(df, drop_first=True)

# Features and target
y = df["pm2_5"]
X = df.drop(columns=["pm2_5"])

# Random Forest model
model = RandomForestRegressor(
    n_estimators=200,
    max_depth=15,
    random_state=42,
    n_jobs=-1
)

kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

r2_scores = cross_val_score(
    model,
    X,
    y,
    cv=kf,
    scoring="r2"
)

rmse_scores = np.sqrt(
    -cross_val_score(
        model,
        X,
        y,
        cv=kf,
        scoring="neg_mean_squared_error"
    )
)

print("\nRandom Forest Results")
print("R² Scores:", r2_scores)
print("Mean R²:", r2_scores.mean())
print("Std Dev:", r2_scores.std())

print("\nRMSE Scores:", rmse_scores)
print("Average RMSE:", rmse_scores.mean())