import pickle
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

model = pickle.load(open("models/random_forest_optimized.pkl", "rb"))
rf_scaler = pickle.load(open("models/rf_scaler.pkl", "rb"))
base_scaler = pickle.load(open("models/base_scaler.pkl", "rb"))
encoders = pickle.load(open("models/encoder.pkl", "rb"))
feature_list = pickle.load(open("models/features.pkl", "rb"))
scale_cols = pickle.load(open("models/scale_cols.pkl", "rb"))

def predict(temp, hum, wind, so2, no2, rspm, pm_hist):
    input_data = {col: 0.0 for col in feature_list}
    input_data['state'] = encoders['state']['Delhi']
    input_data['location'] = encoders['location']['Delhi']
    input_data['type'] = encoders['type']['Industrial Area']
    
    input_data['season'] = 2
    input_data['year'] = 2024
    input_data['month'] = 6
    input_data['day'] = 15
    
    input_data['temperature'] = temp
    input_data['humidity'] = hum
    input_data['windspeed'] = wind
    input_data['so2'] = so2
    input_data['no2'] = no2
    input_data['rspm'] = rspm
    
    input_data['pm2_5_lag1'] = pm_hist
    input_data['pm2_5_lag2'] = pm_hist
    input_data['pm2_5_roll3'] = pm_hist
    input_data['pm2_5_roll7'] = pm_hist

    df_input = pd.DataFrame([input_data])[feature_list]
    df_input[scale_cols] = base_scaler.transform(df_input[scale_cols])
    scaled = rf_scaler.transform(df_input)
    return model.predict(scaled)[0]

print("Case A (Clean Air, Hist PM=20):", predict(20, 40, 10, 5, 10, 20, 20.0))
print("Case B (Polluted, Hist PM=150):", predict(45, 90, 1, 100, 150, 500, 150.0))
