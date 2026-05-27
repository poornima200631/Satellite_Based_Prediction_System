import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ==========================================
# STEP 1 : LOAD DATASET
# ==========================================

print("Loading Dataset...")

df = pd.read_csv("data/processed/model_ready_data.csv")

print("Dataset Loaded Successfully!")
print(df.head())

# ==========================================
# STEP 2 : REMOVE UNNECESSARY COLUMNS
# ==========================================

if "date" in df.columns:
    df = df.drop("date", axis=1)

if "stn_code" in df.columns:
    df = df.drop("stn_code", axis=1)

# ==========================================
# STEP 3 : FEATURES AND TARGET
# ==========================================

y = df["pm2_5"]

X = df.drop("pm2_5", axis=1)

print("\nFeatures:")
print(X.columns)

# ==========================================
# STEP 4 : TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTrain Size:", X_train.shape)
print("Test Size :", X_test.shape)

# ==========================================
# STEP 5 : FEATURE SCALING
# ==========================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

print("\nScaling Completed!")

# ==========================================
# STEP 6 : DECISION TREE
# ==========================================

print("\nTraining Decision Tree...")

dt_model = DecisionTreeRegressor(
    max_depth=10,
    random_state=42
)

dt_model.fit(X_train, y_train)

print("Decision Tree Training Completed!")

# predictions
dt_predictions = dt_model.predict(X_test)

# metrics
dt_mae = mean_absolute_error(y_test, dt_predictions)
dt_mse = mean_squared_error(y_test, dt_predictions)
dt_rmse = np.sqrt(dt_mse)
dt_r2 = r2_score(y_test, dt_predictions)

# ==========================================
# STEP 7 : RANDOM FOREST
# ==========================================

print("\nTraining Random Forest...")

rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

print("Random Forest Training Completed!")

# predictions
rf_predictions = rf_model.predict(X_test)

# metrics
rf_mae = mean_absolute_error(y_test, rf_predictions)
rf_mse = mean_squared_error(y_test, rf_predictions)
rf_rmse = np.sqrt(rf_mse)
rf_r2 = r2_score(y_test, rf_predictions)

# ==========================================
# STEP 8 : GRADIENT BOOSTING
# ==========================================

print("\nTraining Gradient Boosting...")

gb_model = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)

gb_model.fit(X_train, y_train)

print("Gradient Boosting Training Completed!")

# predictions
gb_predictions = gb_model.predict(X_test)

# metrics
gb_mae = mean_absolute_error(y_test, gb_predictions)
gb_mse = mean_squared_error(y_test, gb_predictions)
gb_rmse = np.sqrt(gb_mse)
gb_r2 = r2_score(y_test, gb_predictions)

# ==========================================
# STEP 9 : RESULTS COMPARISON
# ==========================================

print("\n===================================")
print("DECISION TREE RESULTS")
print("===================================")

print("MAE  :", dt_mae)
print("MSE  :", dt_mse)
print("RMSE :", dt_rmse)
print("R2   :", dt_r2)

print("\n===================================")
print("RANDOM FOREST RESULTS")
print("===================================")

print("MAE  :", rf_mae)
print("MSE  :", rf_mse)
print("RMSE :", rf_rmse)
print("R2   :", rf_r2)

print("\n===================================")
print("GRADIENT BOOSTING RESULTS")
print("===================================")

print("MAE  :", gb_mae)
print("MSE  :", gb_mse)
print("RMSE :", gb_rmse)
print("R2   :", gb_r2)

# ==========================================
# STEP 10 : SAVE MODELS
# ==========================================

print("\nSaving Models...")

with open("decision_tree_model.pkl", "wb") as f:
    pickle.dump(dt_model, f)

with open("random_forest_model.pkl", "wb") as f:
    pickle.dump(rf_model, f)

with open("gradient_boosting_model.pkl", "wb") as f:
    pickle.dump(gb_model, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("Models Saved Successfully!")

# ==========================================
# STEP 11 : SAMPLE PREDICTIONS
# ==========================================

print("\nSome Predictions:\n")

for i in range(5):

    print("Actual Value :", y_test.iloc[i])

    print("Decision Tree      :", dt_predictions[i])

    print("Random Forest      :", rf_predictions[i])

    print("Gradient Boosting  :", gb_predictions[i])

    print("--------------------------------------")