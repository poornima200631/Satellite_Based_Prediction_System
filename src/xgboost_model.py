import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

# ==============================
# 1. Load dataset
# ==============================
df = pd.read_csv("data/processed/model_ready_data.csv")

# ==============================
# 2. Clean dataset
# ==============================
if 'date' in df.columns:
    df = df.drop(columns=['date'])

if 'stn_code' in df.columns:
    df = df.drop(columns=['stn_code'])

df = df.dropna()

# ==============================
# 3. Features & target
# ==============================
X = df.drop(columns=['pm2_5'])
y = df['pm2_5']

feature_columns = X.columns.tolist()

# ==============================
# 4. Time-based split
# ==============================
df = df.reset_index(drop=True)

split_index = int(len(df) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

# ==============================
# 5. Baseline Model
# ==============================
baseline = LinearRegression()
baseline.fit(X_train, y_train)

baseline_pred = baseline.predict(X_test)

print("\n===== Baseline Model =====")
print("RMSE:", np.sqrt(mean_squared_error(y_test, baseline_pred)))
print("R2 Score:", r2_score(y_test, baseline_pred))

# ==============================
# 6. XGBoost Model
# ==============================
model = XGBRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

model.fit(X_train, y_train)

# ==============================
# 7. Predictions
# ==============================
y_pred = model.predict(X_test)

# ==============================
# 8. Evaluation
# ==============================
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n===== XGBoost Model =====")
print("MAE:", mae)
print("RMSE:", rmse)
print("R2 Score:", r2)

# ==============================
# 9. Feature Importance
# ==============================
importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values(by='Importance', ascending=False)

print("\nTop Features:")
print(importance.head(10))

plt.figure(figsize=(10,6))
plt.barh(importance['Feature'][:10], importance['Importance'][:10])
plt.gca().invert_yaxis()
plt.title("Feature Importance")
plt.show()

# ==============================
# 10. Residual Plot
# ==============================
residuals = y_test - y_pred

plt.figure(figsize=(8,5))
plt.hist(residuals, bins=50)
plt.title("Residual Distribution")
plt.xlabel("Error")
plt.ylabel("Frequency")
plt.show()

# ==============================
# 11. Save model + features
# ==============================
joblib.dump(model, "models/xgboost_model.pkl")
joblib.dump(feature_columns, "models/features.pkl")

print("\nModel saved successfully!")

# ==============================
# 12. SAFE PREDICTION FUNCTION (FIXED)
# ==============================
def predict_pm25(input_dict):
    model = joblib.load("models/xgboost_model.pkl")
    features = joblib.load("models/features.pkl")

    input_df = pd.DataFrame([input_dict])

    # Add missing columns safely
    for col in features:
        if col not in input_df.columns:
            input_df[col] = 0

    # Ensure correct order
    input_df = input_df[features]

    return model.predict(input_df)[0]

# ==============================
# 13. TEST PREDICTION
# ==============================
print("\nSample Prediction:")
print(predict_pm25({
    "temperature": 30,
    "humidity": 70,
    "no2": 40,
    "co": 1.2,
    "so2": 15
}))