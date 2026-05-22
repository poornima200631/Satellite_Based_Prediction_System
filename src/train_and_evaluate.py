import os
import pickle
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Try to import visualization libraries
try:
    import matplotlib
    matplotlib.use('Agg')  # Force non-interactive Agg backend
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_VIS = True
except ImportError:
    HAS_VIS = False


def load_data(file_path: str) -> pd.DataFrame:
    """
    Loads the dataset from the specified CSV path.
    """
    print("=" * 80)
    print(f"[STEP 1] LOADING DATASET FROM: {file_path}")
    print("=" * 80)
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Error: Dataset not found at: {file_path}. Please check the path and run preprocessing first."
        )
        
    df = pd.read_csv(file_path)
    print(f"-> Dataset loaded successfully! Shape: {df.shape[0]} rows, {df.shape[1]} columns.\n")
    return df


def preprocess_data(df: pd.DataFrame, target_col: str = 'pm2_5'):
    """
    Cleans, splits, and scales the dataset.
    - Drops 'date' and 'stn_code' columns.
    - Splits into Train and Test sets.
    - Standardizes the predictor features.
    """
    print("=" * 80)
    print("[STEP 2] PREPROCESSING & SCALING DATA")
    print("=" * 80)
    
    processed_df = df.copy()
    
    # Drop unnecessary columns safely
    columns_to_drop = ['date', 'stn_code']
    dropped_cols = []
    for col in columns_to_drop:
        if col in processed_df.columns:
            processed_df = processed_df.drop(columns=[col])
            dropped_cols.append(col)
            
    if dropped_cols:
        print(f"-> Dropped identifier columns: {dropped_cols}")
        
    # Handle non-numeric columns if any exist
    non_numeric_cols = processed_df.select_dtypes(exclude=[np.number]).columns.tolist()
    if target_col in non_numeric_cols:
        non_numeric_cols.remove(target_col)
    if non_numeric_cols:
        print(f"Warning: Dropping non-numeric features: {non_numeric_cols}")
        processed_df = processed_df.drop(columns=non_numeric_cols)
        
    # Check target column existence
    if target_col not in processed_df.columns:
        raise KeyError(f"Error: Target column '{target_col}' not found in the dataset!")
        
    # Features and target split
    X = processed_df.drop(columns=[target_col])
    y = processed_df[target_col]
    
    feature_names = X.columns.tolist()
    print(f"-> Target variable: '{target_col}'")
    print(f"-> Input features ({len(feature_names)}): {feature_names}")
    
    # Train-test split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"-> Train split size: {X_train.shape[0]} samples")
    print(f"-> Test split size: {X_test.shape[0]} samples")
    
    # Standardize features (highly recommended for both Linear and Ridge regression)
    print("-> Fitting and applying StandardScaler...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("-> Feature scaling completed successfully!\n")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_names


def train_linear_regression(X_train: np.ndarray, y_train: pd.Series) -> LinearRegression:
    """
    Trains a baseline Linear Regression model.
    """
    print("=" * 80)
    print("[STEP 3A] TRAINING LINEAR REGRESSION (BASELINE)")
    print("=" * 80)
    
    start_time = time.time()
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    duration_ms = (time.time() - start_time) * 1000
    
    print(f"-> Linear Regression trained successfully in {duration_ms:.2f} ms!")
    print(f"-> LR Intercept: {lr_model.intercept_:.4f} ug/m3\n")
    return lr_model


def train_ridge_regression(X_train: np.ndarray, y_train: pd.Series) -> Ridge:
    """
    Performs 5-Fold Cross-Validation to tune Ridge Regression's alpha penalty,
    then fits the model with the best alpha.
    """
    print("=" * 80)
    print("[STEP 3B] TRAINING RIDGE REGRESSION & TUNING HYPERPARAMETERS (ALPHA)")
    print("=" * 80)
    
    candidate_alphas = [0.001, 0.01, 0.1, 1.0, 10.0, 50.0, 65.79, 100.0, 500.0, 1000.0]
    best_alpha = None
    best_mse = float('inf')
    
    print("-> Starting 5-Fold Cross-Validation search:")
    print(f"   {'Alpha':<12} | {'Mean Validation MSE':<22} | {'Validation RMSE':<18}")
    print("   " + "-" * 58)
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    for alpha in candidate_alphas:
        cv_mses = []
        for train_idx, val_idx in kf.split(X_train):
            X_tr, X_va = X_train[train_idx], X_train[val_idx]
            y_tr, y_va = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            fold_model = Ridge(alpha=alpha)
            fold_model.fit(X_tr, y_tr)
            fold_pred = fold_model.predict(X_va)
            cv_mses.append(mean_squared_error(y_va, fold_pred))
            
        mean_cv_mse = np.mean(cv_mses)
        mean_cv_rmse = np.sqrt(mean_cv_mse)
        print(f"   {alpha:<12.3f} | {mean_cv_mse:<22.4f} | {mean_cv_rmse:<18.4f}")
        
        if mean_cv_mse < best_mse:
            best_mse = mean_cv_mse
            best_alpha = alpha
            
    print("   " + "-" * 58)
    print(f"-> CV Complete! Best Alpha chosen: {best_alpha}")
    print(f"-> Lowest CV Mean Squared Error: {best_mse:.4f}")
    
    # Train the final Ridge model with the best alpha parameter
    print(f"-> Fitting final Ridge Regression model with alpha={best_alpha}...")
    start_time = time.time()
    ridge_model = Ridge(alpha=best_alpha)
    ridge_model.fit(X_train, y_train)
    duration_ms = (time.time() - start_time) * 1000
    
    print(f"-> Ridge Regression trained successfully in {duration_ms:.2f} ms!\n")
    return ridge_model


def evaluate_and_compare(
    lr_model: LinearRegression,
    ridge_model: Ridge,
    X_test: np.ndarray,
    y_test: pd.Series
) -> dict:
    """
    Evaluates both models side-by-side on the test dataset.
    """
    print("=" * 80)
    print("[STEP 4] COMPARING MODEL PERFORMANCE ON TEST SPLIT")
    print("=" * 80)
    
    # Predictions
    y_pred_lr = lr_model.predict(X_test)
    y_pred_ridge = ridge_model.predict(X_test)
    
    # Calculate Metrics
    models_metrics = {}
    for name, y_pred in [("Linear Regression", y_pred_lr), ("Ridge Regression", y_pred_ridge)]:
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        models_metrics[name] = {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
            'predictions': y_pred
        }
        
    # Print Performance Comparison Table
    print(f"   {'Model Name':<25} | {'MAE':<10} | {'MSE':<12} | {'RMSE':<10} | {'R2 Score':<10}")
    print("   " + "-" * 75)
    for name, metrics in models_metrics.items():
        print(
            f"   {name:<25} | {metrics['mae']:<10.4f} | {metrics['mse']:<12.4f} | "
            f"{metrics['rmse']:<10.4f} | {metrics['r2']:<10.4f}"
        )
    print("   " + "-" * 75)
    print("-> Model evaluation and side-by-side comparison completed!\n")
    
    return models_metrics


def save_artifacts(
    lr_model: LinearRegression,
    ridge_model: Ridge,
    scaler: StandardScaler,
    output_dir: str = 'models'
):
    """
    Saves the trained models and standard scaler to disk.
    """
    print("=" * 80)
    print("[STEP 5] SAVING PIPELINE ARTIFACTS")
    print("=" * 80)
    
    os.makedirs(output_dir, exist_ok=True)
    
    lr_path = os.path.join(output_dir, 'linear_regression_model.pkl')
    ridge_path = os.path.join(output_dir, 'ridge_regression_model.pkl')
    scaler_path = os.path.join(output_dir, 'scaler.pkl')
    
    # Save Linear Regression
    with open(lr_path, 'wb') as f:
        pickle.dump(lr_model, f)
    print(f"-> Saved trained Linear Regression model to: {lr_path}")
    
    # Save Ridge Regression
    with open(ridge_path, 'wb') as f:
        pickle.dump(ridge_model, f)
    print(f"-> Saved trained Ridge Regression model to: {ridge_path}")
    
    # Save Scaler
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"-> Saved fitted StandardScaler to: {scaler_path}\n")


def generate_comparison_plots(
    y_test: pd.Series,
    lr_preds: np.ndarray,
    ridge_preds: np.ndarray,
    output_dir: str = 'reports/figures'
):
    """
    Generates diagnostic comparison figures for both algorithms.
    """
    if not HAS_VIS:
        print("Warning: Matplotlib or Seaborn not installed. Skipping plot generation.")
        return
        
    print("=" * 80)
    print("[STEP 6] GENERATING DIAGNOSTIC COMPARISON PLOTS")
    print("=" * 80)
    
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="darkgrid")
    
    # Plot 1: True vs Predicted (Side by Side)
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)
    
    # Linear Regression subplot
    axes[0].scatter(y_test, lr_preds, alpha=0.5, color='#3b82f6', edgecolors='k', s=40, label='LR Predictions')
    lims = [min(y_test.min(), lr_preds.min()), max(y_test.max(), lr_preds.max())]
    axes[0].plot(lims, lims, 'r--', alpha=0.8, linewidth=2, label='Perfect Fit')
    axes[0].set_xlabel('Actual PM2.5 (ug/m3)', fontweight='bold')
    axes[0].set_ylabel('Predicted PM2.5 (ug/m3)', fontweight='bold')
    axes[0].set_title('Linear Regression: Actual vs. Predicted', fontweight='bold')
    axes[0].legend()
    
    # Ridge Regression subplot
    axes[1].scatter(y_test, ridge_preds, alpha=0.5, color='#8b5cf6', edgecolors='k', s=40, label='Ridge Predictions')
    lims_ridge = [min(y_test.min(), ridge_preds.min()), max(y_test.max(), ridge_preds.max())]
    axes[1].plot(lims_ridge, lims_ridge, 'r--', alpha=0.8, linewidth=2, label='Perfect Fit')
    axes[1].set_xlabel('Actual PM2.5 (ug/m3)', fontweight='bold')
    axes[1].set_title('Ridge Regression: Actual vs. Predicted', fontweight='bold')
    axes[1].legend()
    
    plt.suptitle('Model Comparison: Actual vs. Predicted PM2.5 Values', fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    actual_vs_pred_path = os.path.join(output_dir, 'actual_vs_predicted.png')
    plt.savefig(actual_vs_pred_path, dpi=300)
    plt.close()
    print(f"-> Saved Actual vs Predicted comparison plot to: {actual_vs_pred_path}")
    
    # Plot 2: Residuals Plot (Comparison)
    lr_residuals = y_test - lr_preds
    ridge_residuals = y_test - ridge_preds
    
    plt.figure(figsize=(12, 6))
    plt.scatter(lr_preds, lr_residuals, alpha=0.4, color='#3b82f6', edgecolors='k', s=30, label='Linear Regression')
    plt.scatter(ridge_preds, ridge_residuals, alpha=0.4, color='#10b981', edgecolors='k', s=30, label='Ridge Regression')
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.8, linewidth=2)
    plt.xlabel('Predicted PM2.5 (ug/m3)', fontweight='bold')
    plt.ylabel('Residuals (Actual - Predicted)', fontweight='bold')
    plt.title('Residual Analysis Comparison Plot', fontweight='bold', pad=15)
    plt.legend()
    plt.tight_layout()
    
    residuals_path = os.path.join(output_dir, 'residuals_plot.png')
    plt.savefig(residuals_path, dpi=300)
    plt.close()
    print(f"-> Saved Residuals comparison plot to: {residuals_path}\n")


def main():
    dataset_path = os.path.join('data', 'processed', 'model_ready_data.csv')
    
    try:
        # 1. Load Data
        df = load_data(dataset_path)
        
        # 2. Preprocess Data & Standardize
        X_train, X_test, y_train, y_test, scaler, feature_names = preprocess_data(df)
        
        # 3. Train Models
        lr_model = train_linear_regression(X_train, y_train)
        ridge_model = train_ridge_regression(X_train, y_train)
        
        # 4. Compare & Evaluate both models
        metrics = evaluate_and_compare(lr_model, ridge_model, X_test, y_test)
        
        # 5. Save Artifacts (scaler, LR model, Ridge model)
        save_artifacts(lr_model, ridge_model, scaler)
        
        # 6. Generate Diagnostic visual plots
        generate_comparison_plots(
            y_test, 
            metrics['Linear Regression']['predictions'], 
            metrics['Ridge Regression']['predictions']
        )
        
        print("Success: Modular ML Pipeline with both Linear & Ridge Regression completed successfully!")
        print("=" * 80)
        
    except Exception as e:
        print(f"\nError: Pipeline failed with error: {str(e)}")
        raise


if __name__ == '__main__':
    main()
