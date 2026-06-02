"""
hyperparameter_tuning.py
=============================================================================
Hyperparameter Tuning Pipeline for Satellite-Based Air Quality Prediction.
Tunes Random Forest (Person 1) and XGBoost (Person 2) using RandomizedSearchCV
and GridSearchCV, compares baseline vs. optimized models, and saves the results.
=============================================================================
"""

import os
import time
import json
import joblib
import numpy as np
import pandas as pd
import warnings

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_VIS = True
except ImportError:
    HAS_VIS = False


def load_and_preprocess_data(file_path: str = 'data/processed/model_ready_data.csv'):
    """Loads and preprocesses the dataset for modeling."""
    print("=" * 80)
    print(f"[STEP 1] LOADING DATASET FROM: {file_path}")
    print("=" * 80)

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Dataset not found at '{file_path}'. "
            "Please ensure data preprocessing has been run first."
        )

    df = pd.read_csv(file_path)
    print(f"-> Dataset loaded. Shape: {df.shape[0]} rows x {df.shape[1]} columns.\n")

    print("=" * 80)
    print("[STEP 2] PREPROCESSING & SPLITTING")
    print("=" * 80)

    processed_df = df.copy()

    # Drop identifier columns
    drop_cols = ['date', 'stn_code', 'state', 'location', 'type']
    dropped = [c for c in drop_cols if c in processed_df.columns]
    if dropped:
        processed_df.drop(columns=dropped, inplace=True)
        print(f"-> Dropped columns: {dropped}")

    # Identify non-numeric columns
    non_numeric = processed_df.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        print(f"-> Dropping non-numeric columns: {non_numeric}")
        processed_df.drop(columns=non_numeric, inplace=True)

    target_col = 'pm2_5'
    if target_col not in processed_df.columns:
        raise KeyError(f"Target column '{target_col}' not found in dataset!")

    X = processed_df.drop(columns=[target_col])
    y = processed_df[target_col]
    feature_names = X.columns.tolist()

    print(f"-> Target variable: '{target_col}'")
    print(f"-> Number of features: {len(feature_names)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"-> Train samples: {X_train.shape[0]}")
    print(f"-> Test samples: {X_test.shape[0]}")

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("-> Features standardized with StandardScaler.\n")

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_names


def tune_random_forest(X_train, y_train):
    """Performs RandomizedSearchCV and GridSearchCV tuning for Random Forest."""
    print("=" * 80)
    print("[STEP 3] PERSON 1 - RANDOM FOREST HYPERPARAMETER TUNING")
    print("=" * 80)

    # 1. Baseline Random Forest
    print("-> Training baseline Random Forest with default hyperparameters...")
    baseline_rf = RandomForestRegressor(random_state=42, n_jobs=-1)
    t0 = time.time()
    baseline_rf.fit(X_train, y_train)
    print(f"-> Baseline RF trained in {time.time() - t0:.2f}s.\n")

    # 2. RandomizedSearchCV (Broad Search)
    print("-> Running RandomizedSearchCV for Random Forest...")
    param_dist = {
        'n_estimators': [50, 100, 150, 200],
        'max_depth': [5, 10, 15, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', 0.5, 0.8],
        'bootstrap': [True]
    }

    rf_base = RandomForestRegressor(random_state=42, n_jobs=-1)
    
    t0 = time.time()
    rf_random = RandomizedSearchCV(
        estimator=rf_base,
        param_distributions=param_dist,
        n_iter=12,
        cv=3,
        scoring='neg_mean_squared_error',
        random_state=42,
        n_jobs=1,  # Keep n_jobs=1 for outer CV to avoid overhead, inner RF uses n_jobs=-1
        verbose=1
    )
    rf_random.fit(X_train, y_train)
    random_time = time.time() - t0
    best_params_random = rf_random.best_params_
    print(f"-> RandomizedSearchCV completed in {random_time:.2f}s.")
    print(f"-> Best parameters from RandomizedSearch: {best_params_random}\n")

    # 3. GridSearchCV (Narrow Search around best params)
    print("-> Running GridSearchCV for Random Forest (narrow grid search)...")
    
    # Construct grid around best params
    grid_params = {}
    
    # n_estimators
    best_n = best_params_random['n_estimators']
    grid_params['n_estimators'] = [max(10, best_n - 30), best_n, best_n + 30]
    
    # max_depth
    best_depth = best_params_random['max_depth']
    if best_depth is None:
        grid_params['max_depth'] = [15, 20, None]
    else:
        grid_params['max_depth'] = [max(1, best_depth - 3), best_depth, best_depth + 3]
        
    # min_samples_split
    best_split = best_params_random['min_samples_split']
    grid_params['min_samples_split'] = sorted(list(set([max(2, best_split - 1), best_split, best_split + 2])))
    
    # Fixed parameters from RandomizedSearch to keep grid size small
    grid_params['min_samples_leaf'] = [best_params_random['min_samples_leaf']]
    grid_params['max_features'] = [best_params_random['max_features']]
    grid_params['bootstrap'] = [best_params_random['bootstrap']]

    print(f"-> GridSearchCV parameter grid: {grid_params}")
    
    t0 = time.time()
    rf_grid = GridSearchCV(
        estimator=rf_base,
        param_grid=grid_params,
        cv=3,
        scoring='neg_mean_squared_error',
        n_jobs=1,
        verbose=1
    )
    rf_grid.fit(X_train, y_train)
    grid_time = time.time() - t0
    best_params_grid = rf_grid.best_params_
    best_rf_model = rf_grid.best_estimator_
    print(f"-> GridSearchCV completed in {grid_time:.2f}s.")
    print(f"-> Best parameters from GridSearch: {best_params_grid}\n")

    return baseline_rf, rf_random.best_estimator_, best_rf_model, best_params_random, best_params_grid


def tune_xgboost(X_train, y_train):
    """Performs RandomizedSearchCV and GridSearchCV tuning for XGBoost."""
    print("=" * 80)
    print("[STEP 4] PERSON 2 - XGBOOST HYPERPARAMETER TUNING")
    print("=" * 80)

    # 1. Baseline XGBoost
    print("-> Training baseline XGBoost with default hyperparameters...")
    baseline_xgb = XGBRegressor(random_state=42, n_jobs=-1)
    t0 = time.time()
    baseline_xgb.fit(X_train, y_train)
    print(f"-> Baseline XGBoost trained in {time.time() - t0:.2f}s.\n")

    # 2. RandomizedSearchCV (Broad Search)
    print("-> Running RandomizedSearchCV for XGBoost...")
    param_dist = {
        'n_estimators': [50, 100, 150, 200],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'max_depth': [3, 5, 7, 9],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'gamma': [0, 0.1, 0.2]
    }

    xgb_base = XGBRegressor(random_state=42, n_jobs=-1)
    
    t0 = time.time()
    xgb_random = RandomizedSearchCV(
        estimator=xgb_base,
        param_distributions=param_dist,
        n_iter=12,
        cv=3,
        scoring='neg_mean_squared_error',
        random_state=42,
        n_jobs=1,
        verbose=1
    )
    xgb_random.fit(X_train, y_train)
    random_time = time.time() - t0
    best_params_random = xgb_random.best_params_
    print(f"-> RandomizedSearchCV completed in {random_time:.2f}s.")
    print(f"-> Best parameters from RandomizedSearch: {best_params_random}\n")

    # 3. GridSearchCV (Narrow Search around best params)
    print("-> Running GridSearchCV for XGBoost (narrow grid search)...")
    
    # Construct grid around best params
    grid_params = {}
    
    # n_estimators
    best_n = best_params_random['n_estimators']
    grid_params['n_estimators'] = [max(10, best_n - 30), best_n, best_n + 30]
    
    # learning_rate
    best_lr = best_params_random['learning_rate']
    grid_params['learning_rate'] = [max(0.01, best_lr * 0.7), best_lr, best_lr * 1.3]
    
    # max_depth
    best_depth = best_params_random['max_depth']
    grid_params['max_depth'] = sorted(list(set([max(1, best_depth - 1), best_depth, best_depth + 1])))
    
    # Fixed parameters from RandomizedSearch to keep grid size small
    grid_params['subsample'] = [best_params_random['subsample']]
    grid_params['colsample_bytree'] = [best_params_random['colsample_bytree']]
    grid_params['gamma'] = [best_params_random['gamma']]

    print(f"-> GridSearchCV parameter grid: {grid_params}")
    
    t0 = time.time()
    xgb_grid = GridSearchCV(
        estimator=xgb_base,
        param_grid=grid_params,
        cv=3,
        scoring='neg_mean_squared_error',
        n_jobs=1,
        verbose=1
    )
    xgb_grid.fit(X_train, y_train)
    grid_time = time.time() - t0
    best_params_grid = xgb_grid.best_params_
    best_xgb_model = xgb_grid.best_estimator_
    print(f"-> GridSearchCV completed in {grid_time:.2f}s.")
    print(f"-> Best parameters from GridSearch: {best_params_grid}\n")

    return baseline_xgb, xgb_random.best_estimator_, best_xgb_model, best_params_random, best_params_grid


def evaluate_models(models_dict, X_test, y_test):
    """Evaluates multiple models on the test set and prints comparison."""
    print("=" * 80)
    print("[STEP 5] EVALUATING ALL MODELS ON THE TEST SPLIT")
    print("=" * 80)

    results = {}
    header = f"   {'Model Name':<30} | {'MAE':>10} | {'MSE':>10} | {'RMSE':>10} | {'R² Score':>10}"
    print(header)
    print("   " + "-" * 78)

    for name, model in models_dict.items():
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        results[name] = {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
            'predictions': y_pred
        }
        
        print(f"   {name:<30} | {mae:>10.4f} | {mse:>10.4f} | {rmse:>10.4f} | {r2:>10.4f}")

    print("   " + "-" * 78)
    print()
    return results


def save_tuning_artifacts(models, params, scaler, output_dir='models'):
    """Saves final tuned models and hyperparameters to disk."""
    print("=" * 80)
    print("[STEP 6] SAVING OPTIMIZED MODELS AND PARAMETERS")
    print("=" * 80)
    os.makedirs(output_dir, exist_ok=True)

    # Save models
    for name, model in models.items():
        filename = f"{name}_tuned.pkl"
        path = os.path.join(output_dir, filename)
        joblib.dump(model, path)
        print(f"-> Saved {name} model to {path}")

    # Save scaler
    scaler_path = os.path.join(output_dir, "scaler_tuned.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"-> Saved scaler to {scaler_path}")

    # Save parameters to JSON
    params_path = os.path.join(output_dir, "hyperparameters_summary.json")
    with open(params_path, 'w') as f:
        json.dump(params, f, indent=4)
    print(f"-> Saved tuning parameters summary to {params_path}\n")


def generate_plots(results, y_test, output_dir='reports/figures'):
    """Generates comparison visualizations."""
    if not HAS_VIS:
        print("Warning: matplotlib/seaborn not installed. Skipping plot generation.")
        return

    print("=" * 80)
    print("[STEP 7] GENERATING COMPARISON PLOTS")
    print("=" * 80)
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style='darkgrid', palette='muted')

    # R2 comparison
    plt.figure(figsize=(10, 6))
    names = [name for name in results.keys() if 'Baseline' not in name or 'RF' in name or 'XGB' in name]
    r2_scores = [results[name]['r2'] for name in names]
    
    # Sort by R2 score
    sorted_idx = np.argsort(r2_scores)
    sorted_names = [names[i] for i in sorted_idx]
    sorted_r2 = [r2_scores[i] for i in sorted_idx]

    colors = ['#f87171', '#fbbf24', '#34d399', '#60a5fa', '#a78bfa', '#f472b6']
    bars = plt.barh(sorted_names, sorted_r2, color=colors[:len(names)], edgecolor='black')
    
    # Add values on bars
    for bar in bars:
        width = bar.get_width()
        plt.text(width - 0.05 if width > 0.1 else width + 0.01, 
                 bar.get_y() + bar.get_height()/2, 
                 f"{width:.4f}", 
                 va='center', ha='right' if width > 0.1 else 'left', 
                 fontweight='bold', color='white' if width > 0.1 else 'black')

    plt.xlabel('R² Score', fontweight='bold', fontsize=12)
    plt.title('Model R² Performance Comparison (Tuned vs. Baseline)', fontweight='bold', fontsize=14, pad=15)
    plt.xlim(0, 1.0)
    plt.tight_layout()
    p1 = os.path.join(output_dir, 'hyperparameter_tuning_r2_comparison.png')
    plt.savefig(p1, dpi=200)
    plt.close()
    print(f"-> Saved: {p1}")

    # Actual vs Predicted comparison for final grid search models
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)
    
    rf_opt_name = 'Random Forest (GridSearch)'
    xgb_opt_name = 'XGBoost (GridSearch)'
    
    if rf_opt_name in results and xgb_opt_name in results:
        # RF
        y_pred_rf = results[rf_opt_name]['predictions']
        r2_rf = results[rf_opt_name]['r2']
        rmse_rf = results[rf_opt_name]['rmse']
        
        axes[0].scatter(y_test, y_pred_rf, alpha=0.5, s=30, color='#10b981', edgecolors='k', linewidths=0.2)
        lims = [min(y_test.min(), y_pred_rf.min()), max(y_test.max(), y_pred_rf.max())]
        axes[0].plot(lims, lims, 'r--', lw=2, label='Perfect Fit')
        axes[0].set_xlabel('Actual PM2.5 (µg/m³)', fontweight='bold')
        axes[0].set_ylabel('Predicted PM2.5 (µg/m³)', fontweight='bold')
        axes[0].set_title(f'{rf_opt_name}\nR²={r2_rf:.4f} | RMSE={rmse_rf:.2f}', fontweight='bold')
        axes[0].legend()
        
        # XGB
        y_pred_xgb = results[xgb_opt_name]['predictions']
        r2_xgb = results[xgb_opt_name]['r2']
        rmse_xgb = results[xgb_opt_name]['rmse']
        
        axes[1].scatter(y_test, y_pred_xgb, alpha=0.5, s=30, color='#3b82f6', edgecolors='k', linewidths=0.2)
        lims = [min(y_test.min(), y_pred_xgb.min()), max(y_test.max(), y_pred_xgb.max())]
        axes[1].plot(lims, lims, 'r--', lw=2, label='Perfect Fit')
        axes[1].set_xlabel('Actual PM2.5 (µg/m³)', fontweight='bold')
        axes[1].set_title(f'{xgb_opt_name}\nR²={r2_xgb:.4f} | RMSE={rmse_xgb:.2f}', fontweight='bold')
        axes[1].legend()
        
        plt.suptitle('Optimized Models: Actual vs. Predicted PM2.5', fontsize=15, fontweight='bold', y=0.98)
        plt.tight_layout()
        p2 = os.path.join(output_dir, 'hyperparameter_tuning_predictions.png')
        plt.savefig(p2, dpi=200)
        plt.close()
        print(f"-> Saved: {p2}\n")


def main():
    start_time = time.time()
    try:
        # Load and preprocess
        X_train, X_test, y_train, y_test, scaler, feature_names = load_and_preprocess_data()

        # Tune Random Forest
        rf_base, rf_rand, rf_grid, rf_rand_params, rf_grid_params = tune_random_forest(X_train, y_train)

        # Tune XGBoost
        xgb_base, xgb_rand, xgb_grid, xgb_rand_params, xgb_grid_params = tune_xgboost(X_train, y_train)

        # Assemble and evaluate
        models = {
            'Random Forest (Baseline)': rf_base,
            'Random Forest (RandomizedSearch)': rf_rand,
            'Random Forest (GridSearch)': rf_grid,
            'XGBoost (Baseline)': xgb_base,
            'XGBoost (RandomizedSearch)': xgb_rand,
            'XGBoost (GridSearch)': xgb_grid
        }
        
        results = evaluate_models(models, X_test, y_test)

        # Save artifacts
        save_tuning_artifacts(
            models={
                'random_forest': rf_grid,
                'xgboost': xgb_grid
            },
            params={
                'random_forest': {
                    'randomized_search_best_params': rf_rand_params,
                    'grid_search_best_params': rf_grid_params
                },
                'xgboost': {
                    'randomized_search_best_params': xgb_rand_params,
                    'grid_search_best_params': xgb_grid_params
                }
            },
            scaler=scaler
        )

        # Generate plots
        generate_plots(results, y_test)

        total_elapsed = time.time() - start_time
        print("=" * 80)
        print(f"SUCCESS: Hyperparameter Tuning Pipeline completed in {total_elapsed/60:.2f} minutes!")
        print("=" * 80)
        
        # Output summary of best parameters explicitly
        print("\n*** BEST PARAMETERS FOUND ***")
        print("----------------------------------------")
        print("1. Random Forest (Person 1):")
        print(f"   RandomizedSearchCV: {rf_rand_params}")
        print(f"   GridSearchCV      : {rf_grid_params}")
        print("\n2. XGBoost (Person 2):")
        print(f"   RandomizedSearchCV: {xgb_rand_params}")
        print(f"   GridSearchCV      : {xgb_grid_params}")
        print("----------------------------------------\n")

    except Exception as e:
        print(f"\n[ERROR] Pipeline failed: {e}")
        raise


if __name__ == '__main__':
    main()
