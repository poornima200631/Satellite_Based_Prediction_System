import joblib
import pickle
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

model = joblib.load("models/xgboost_tuned.pkl")
scaler = joblib.load("models/scaler_tuned.pkl")
encoders = pickle.load(open("models/encoder.pkl", "rb"))
feature_list = list(scaler.feature_names_in_)

# Create base input
base_input = {col: 0.0 for col in feature_list}

# Fill some important features
base_input['pm2_5_roll3'] = 80.0
base_input['pm2_5_roll7'] = 80.0
base_input['pm2_5_lag1'] = 80.0
base_input['pm2_5_lag2'] = 80.0
base_input['rspm'] = 50.0
base_input['temperature'] = 30.0

print("Predictions across states:")
# Test for different states
for state in encoders['state'].keys():
    base_input['state'] = encoders['state'][state]
    df_input = pd.DataFrame([base_input])[feature_list]
    scaled = scaler.transform(df_input)
    pred = model.predict(scaled)[0]
    print(f"State: {state:25s} -> Prediction: {pred:.6f}")
