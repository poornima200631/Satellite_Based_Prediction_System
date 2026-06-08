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
# 4. Standard Shuffled Split
# ==============================
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Drop extra columns not used in tuned model
drop_cols = ['state', 'location', 'type']
X = X.drop(columns=[c for c in drop_cols if c in X.columns])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==============================
# 5. Baseline Model
# ==============================
baseline = LinearRegression()
baseline.fit(X_train_scaled, y_train)

baseline_pred = baseline.predict(X_test_scaled)

print("\n===== Baseline Model =====")
print("RMSE:", np.sqrt(mean_squared_error(y_test, baseline_pred)))
print("R2 Score:", r2_score(y_test, baseline_pred))

# ==============================
# 6. XGBoost Model (Tuned)
# ==============================
model = XGBRegressor(
    n_estimators=180,
    learning_rate=0.13,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=1.0,
    gamma=0.1,
    random_state=42
)

model.fit(X_train_scaled, y_train)

# ==============================
# 7. Predictions
# ==============================
y_pred = model.predict(X_test_scaled)

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
# 10. Save Model & Expose predict function
# ==============================
import joblib

joblib.dump(model, "models/xgboost_model.pkl")
joblib.dump(scaler, "models/scaler_tuned.pkl")
print("\nModel saved successfully!")

def predict_pm25(input_dict):
    """
    Predict PM2.5 using the trained XGBoost model.
    """
    input_df = pd.DataFrame([input_dict])
    for col in X.columns:
        if col not in input_df.columns:
            input_df[col] = 0.0
    input_df = input_df[X.columns]
    scaled_input = scaler.transform(input_df)
    return model.predict(scaled_input)[0]

if __name__ == "__main__":
    print("\nSample Prediction:")
    print(predict_pm25({
        "temperature": 30,
        "humidity": 60,
        "pm2_5_roll3": 45,
        "no2": 20,
        "so2": 15
    }))