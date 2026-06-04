import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle
import os

df = pd.read_csv("data/processed/combined_dataset.csv")

# Same steps as data_preprocessing.py
df = df.rename(columns={'pm2_5_x': 'pm2_5'})
if 'pm2_5_y' in df.columns:
    df.drop(columns=['pm2_5_y'], inplace=True)

df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(by=['stn_code', 'date']).reset_index(drop=True)

# Feature engineering
df['dayofweek'] = df['date'].dt.dayofweek
df['season'] = df['date'].dt.month.map(
    lambda m: 1 if m in [12, 1, 2] else 2 if m in [3, 4, 5] else 3 if m in [6, 7, 8] else 4
)

grouped = df.groupby('stn_code')
df['pm2_5_lag1'] = grouped['pm2_5'].shift(1)
df['pm2_5_lag2'] = grouped['pm2_5'].shift(2)
if 'temperature' in df.columns:
    df['temp_lag1'] = grouped['temperature'].shift(1)
df['pm2_5_roll3'] = grouped['pm2_5'].transform(lambda x: x.rolling(3, min_periods=1).mean())
df['pm2_5_roll7'] = grouped['pm2_5'].transform(lambda x: x.rolling(7, min_periods=1).mean())

df = df.bfill()

# Scaling logic
exclude_cols = ['date', 'stn_code', 'state', 'location', 'type', 'pm2_5']
scale_cols = [col for col in df.select_dtypes(include=np.number).columns if col not in exclude_cols]

scaler = StandardScaler()
scaler.fit(df[scale_cols])

os.makedirs('models', exist_ok=True)
with open('models/base_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
with open('models/scale_cols.pkl', 'wb') as f:
    pickle.dump(scale_cols, f)

print("Base scaler saved successfully!")
