import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
import pickle

DATA_PATH = os.path.join('data', 'processed')
MODELS_PATH = 'models'
os.makedirs(MODELS_PATH, exist_ok=True)


# =========================
# MERGE DATASETS
# =========================
def merge_datasets():

    print("Loading datasets...")

    pm25_df = pd.read_csv(
        os.path.join(DATA_PATH, 'cleaned_data.csv'),
        parse_dates=['date']
    )

    weather_df = pd.read_csv(
        os.path.join(DATA_PATH, 'AirQualityData_cleaned.csv'),
        parse_dates=['timestamp']
    )

    # Create date column
    weather_df['date'] = weather_df['timestamp'].dt.normalize()

    # Shift weather year to 2015
    weather_df['date'] = weather_df['date'].apply(
        lambda d: d.replace(year=2015, day=28)
        if d.month == 2 and d.day == 29
        else d.replace(year=2015)
    )

    print("Aggregating weather data...")

    drop_cols = ['year', 'month', 'day', 'hour', 'dayofweek']

    num_cols = [
        col for col in weather_df.select_dtypes(include=np.number).columns
        if col not in drop_cols
    ]

    daily_weather = (
        weather_df
        .groupby('date')[num_cols]
        .mean()
        .reset_index()
    )

    print("Merging datasets...")

    combined_df = pd.merge(
        pm25_df,
        daily_weather,
        on='date',
        how='inner'
    )

    output_path = os.path.join(DATA_PATH, 'combined_dataset.csv')
    combined_df.to_csv(output_path, index=False)

    print(f"Combined shape: {combined_df.shape}")

    return combined_df


# =========================
# CLEAN + FEATURE ENGINEERING
# =========================
def clean_combined_data(df):

    print("Cleaning data...")

    # Rename/drop duplicate PM2.5 columns
    df = df.rename(columns={'pm2_5_x': 'pm2_5'})

    if 'pm2_5_y' in df.columns:
        df.drop(columns=['pm2_5_y'], inplace=True)

    # Sort
    df['date'] = pd.to_datetime(df['date'])

    df = df.sort_values(
        by=['stn_code', 'date']
    ).reset_index(drop=True)

    # Fill missing values
    df = (
        df.groupby('stn_code')
        .apply(lambda x: x.ffill().bfill())
        .reset_index(drop=True)
    )

    # =========================
    # FEATURE ENGINEERING
    # =========================

    print("Creating features...")

    df['dayofweek'] = df['date'].dt.dayofweek

    df['season'] = df['date'].dt.month.map(
        lambda m: (
            1 if m in [12, 1, 2]
            else 2 if m in [3, 4, 5]
            else 3 if m in [6, 7, 8]
            else 4
        )
    )

    grouped = df.groupby('stn_code')

    # Lag features
    df['pm2_5_lag1'] = grouped['pm2_5'].shift(1)
    df['pm2_5_lag2'] = grouped['pm2_5'].shift(2)

    if 'temperature' in df.columns:
        df['temp_lag1'] = grouped['temperature'].shift(1)

    # Rolling averages
    df['pm2_5_roll3'] = grouped['pm2_5'].transform(
        lambda x: x.rolling(3, min_periods=1).mean()
    )

    df['pm2_5_roll7'] = grouped['pm2_5'].transform(
        lambda x: x.rolling(7, min_periods=1).mean()
    )

    # Fill NaNs from lagging
    df = df.bfill()

    # =========================
    # FEATURE SCALING
    # =========================

    print("Scaling features...")

    exclude_cols = [
        'date', 'stn_code', 'state',
        'location', 'type',
        'pm2_5'
    ]

    scale_cols = [
        col for col in df.select_dtypes(include=np.number).columns
        if col not in exclude_cols
    ]

    scaler = StandardScaler()

    df[scale_cols] = scaler.fit_transform(df[scale_cols])

    # Save the base scaler for deployment
    with open(os.path.join(MODELS_PATH, 'base_scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    
    # Save the list of columns that were scaled so we know what to transform in app.py
    with open(os.path.join(MODELS_PATH, 'scale_cols.pkl'), 'wb') as f:
        pickle.dump(scale_cols, f)

    output_path = os.path.join(DATA_PATH, 'model_ready_data.csv')

    df.to_csv(output_path, index=False)

    print(f"Final shape: {df.shape}")
    print("Model-ready dataset and base scaler saved.")

    return df


# =========================
# MAIN
# =========================
if __name__ == "__main__":

    combined_df = merge_datasets()

    final_df = clean_combined_data(combined_df)