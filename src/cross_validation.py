import pandas as pd
import numpy as np

from sklearn.model_selection import KFold, cross_val_score
from xgboost import XGBRegressor

# Load dataset
df = pd.read_csv("data/processed/model_ready_data.csv")

# Remove columns exactly as in training
if "date" in df.columns:
    df = df.drop(columns=["date"])

if "stn_code" in df.columns:
    df = df.drop(columns=["stn_code"])

# Convert categorical columns to numbers
df = pd.get_dummies(df, drop_first=True)

# Target
y = df["pm2_5"]

# Features
X = df.drop(columns=["pm2_5"])

# Same model as training
model = XGBRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

scores = cross_val_score(
    model,
    X,
    y,
    cv=kf,
    scoring="r2"
)

print("\nFold Scores:")
print(scores)

print("\nMean R2:", np.mean(scores))
print("Std Dev:", np.std(scores))

rmse_scores = np.sqrt(
    -cross_val_score(
        model,
        X,
        y,
        cv=kf,
        scoring="neg_mean_squared_error"
    )
)

print("\nRMSE Scores:")
print(rmse_scores)

print("\nAverage RMSE:", rmse_scores.mean())