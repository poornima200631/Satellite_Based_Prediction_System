import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ==========================================
# STEP 1 : LOAD DATASET
# ==========================================

print("Loading dataset...")

df = pd.read_csv("data/processed/model_ready_data.csv")

print("Dataset Loaded Successfully!")
print(df.head())

# ==========================================
# STEP 2 : REMOVE UNNECESSARY COLUMNS
# ==========================================

# remove columns if they exist
if "date" in df.columns:
    df = df.drop("date", axis=1)

if "stn_code" in df.columns:
    df = df.drop("stn_code", axis=1)

# ==========================================
# STEP 3 : FEATURES AND TARGET
# ==========================================

# target column
y = df["pm2_5"]

# all remaining columns become features
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
print("Test Size:", X_test.shape)

# ==========================================
# STEP 5 : FEATURE SCALING
# ==========================================

scaler = StandardScaler()

# fit + transform training data
X_train = scaler.fit_transform(X_train)

# only transform testing data
X_test = scaler.transform(X_test)

print("\nScaling Completed!")

# ==========================================
# STEP 6 : LINEAR REGRESSION
# ==========================================

print("\nTraining Linear Regression...")

lr_model = LinearRegression()

lr_model.fit(X_train, y_train)

print("Linear Regression Training Completed!")

# predictions
lr_predictions = lr_model.predict(X_test)

# metrics
lr_mae = mean_absolute_error(y_test, lr_predictions)
lr_mse = mean_squared_error(y_test, lr_predictions)
lr_rmse = np.sqrt(lr_mse)
lr_r2 = r2_score(y_test, lr_predictions)

# ==========================================
# STEP 7 : RIDGE REGRESSION
# ==========================================

print("\nTraining Ridge Regression...")

ridge_model = Ridge(alpha=1.0)

ridge_model.fit(X_train, y_train)

print("Ridge Regression Training Completed!")

# predictions
ridge_predictions = ridge_model.predict(X_test)

# metrics
ridge_mae = mean_absolute_error(y_test, ridge_predictions)
ridge_mse = mean_squared_error(y_test, ridge_predictions)
ridge_rmse = np.sqrt(ridge_mse)
ridge_r2 = r2_score(y_test, ridge_predictions)

# ==========================================
# STEP 8 : RESULTS COMPARISON
# ==========================================

print("\n===================================")
print("LINEAR REGRESSION RESULTS")
print("===================================")

print("MAE  :", lr_mae)
print("MSE  :", lr_mse)
print("RMSE :", lr_rmse)
print("R2   :", lr_r2)

print("\n===================================")
print("RIDGE REGRESSION RESULTS")
print("===================================")

print("MAE  :", ridge_mae)
print("MSE  :", ridge_mse)
print("RMSE :", ridge_rmse)
print("R2   :", ridge_r2)

# ==========================================
# STEP 9 : SAVE MODELS
# ==========================================

print("\nSaving Models...")

with open("linear_model.pkl", "wb") as f:
    pickle.dump(lr_model, f)

with open("ridge_model.pkl", "wb") as f:
    pickle.dump(ridge_model, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("Models Saved Successfully!")

# ==========================================
# STEP 10 : SAMPLE PREDICTIONS
# ==========================================

print("\nSome Predictions:")

for i in range(5):
    print("Actual:", y_test.iloc[i])
    print("Linear Prediction:", lr_predictions[i])
    print("Ridge Prediction :", ridge_predictions[i])
    print("--------------------------------")