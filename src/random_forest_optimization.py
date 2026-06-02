"""
random_forest_optimization.py
==============================================
Random Forest Hyperparameter Optimization & Evaluation
for the Satellite-Based Air Quality Prediction System.

This module performs:
    - Data loading & preprocessing
    - Random Forest baseline training
    - Hyperparameter tuning via RandomizedSearchCV (broad search)
    - Fine-tuned GridSearchCV (narrow search around best params)
    - Full evaluation: MAE, MSE, RMSE, R2
    - Feature importance analysis
    - Model artifact saving
    - Diagnostic plots (Actual vs Predicted, Residuals, Feature Importances)
"""

import os
import pickle
import time
import warnings
import numpy as np
import pandas as pd

# Suppress the sklearn/joblib parallel config mismatch warning that floods
# the terminal when n_jobs=-1 on the estimator conflicts with n_jobs=1 on CV.
warnings.filterwarnings(
    'ignore',
    message='`sklearn.utils.parallel.delayed` should be used with',
    category=UserWarning,
)

from sklearn.model_selection import (
    train_test_split,
    RandomizedSearchCV,
    GridSearchCV,
    cross_val_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_VIS = True
except ImportError:
    HAS_VIS = False


# ==============================================================================
# STEP 1 – DATA LOADING
# ==============================================================================

def load_data(file_path: str = 'data/processed/model_ready_data.csv') -> pd.DataFrame:
    """Loads the model-ready dataset from disk."""
    print("=" * 80)
    print(f"[STEP 1] LOADING DATASET FROM: {file_path}")
    print("=" * 80)

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Dataset not found at '{file_path}'. "
            "Please run data_preprocessing.py first."
        )

    df = pd.read_csv(file_path)
    print(f"-> Dataset loaded. Shape: {df.shape[0]} rows × {df.shape[1]} columns.\n")
    return df


# ==============================================================================
# STEP 2 – PREPROCESSING
# ==============================================================================

def preprocess_data(df: pd.DataFrame, target_col: str = 'pm2_5'):
    """
    Drops identifier columns, separates features from the target,
    performs an 80/20 stratified split, and applies StandardScaler.

    Returns
    -------
    X_train, X_test : np.ndarray  (scaled)
    y_train, y_test : pd.Series
    scaler          : fitted StandardScaler
    feature_names   : list[str]
    """
    print("=" * 80)
    print("[STEP 2] PREPROCESSING & FEATURE ENGINEERING")
    print("=" * 80)

    processed_df = df.copy()

    # Drop identifier / leakage columns
    drop_cols = ['date', 'stn_code']
    dropped = [c for c in drop_cols if c in processed_df.columns]
    if dropped:
        processed_df.drop(columns=dropped, inplace=True)
        print(f"-> Dropped identifier columns: {dropped}")

    # Drop any remaining non-numeric columns (except the target)
    non_numeric = [
        c for c in processed_df.select_dtypes(exclude=[np.number]).columns
        if c != target_col
    ]
    if non_numeric:
        print(f"-> Dropping non-numeric columns: {non_numeric}")
        processed_df.drop(columns=non_numeric, inplace=True)

    if target_col not in processed_df.columns:
        raise KeyError(f"Target column '{target_col}' not found in dataset!")

    X = processed_df.drop(columns=[target_col])
    y = processed_df[target_col]
    feature_names = X.columns.tolist()

    print(f"-> Target        : '{target_col}'")
    print(f"-> Feature count : {len(feature_names)}")
    print(f"-> Features      : {feature_names}\n")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"-> Train samples : {X_train.shape[0]}")
    print(f"-> Test  samples : {X_test.shape[0]}")

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)
    print("-> StandardScaler applied successfully.\n")

    return X_train_sc, X_test_sc, y_train, y_test, scaler, feature_names


# ==============================================================================
# STEP 3 – BASELINE RANDOM FOREST
# ==============================================================================

def train_baseline_rf(
    X_train: np.ndarray,
    y_train: pd.Series,
) -> RandomForestRegressor:
    """
    Trains a default Random Forest regressor as a performance baseline
    before hyperparameter optimization.
    """
    print("=" * 80)
    print("[STEP 3] TRAINING BASELINE RANDOM FOREST (default hyperparams)")
    print("=" * 80)

    t0 = time.time()
    baseline_rf = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
    )
    baseline_rf.fit(X_train, y_train)
    elapsed = (time.time() - t0) * 1000

    print(f"-> Baseline RF trained in {elapsed:.1f} ms.\n")
    return baseline_rf


# ==============================================================================
# STEP 4 – RANDOMIZED SEARCH (BROAD HYPERPARAMETER SEARCH)
# ==============================================================================

def randomized_search_rf(
    X_train: np.ndarray,
    y_train: pd.Series,
    n_iter: int = 15,
    cv_folds: int = 3,
) -> tuple:
    """
    Broad search over the Random Forest hyperparameter space using
    RandomizedSearchCV with cross-validation.

    Parameters
    ----------
    n_iter    : Number of random parameter combinations to sample.
    cv_folds  : Number of cross-validation folds.

    Returns
    -------
    best_model    : RandomForestRegressor fitted on all training data.
    best_params   : dict of best hyperparameters found.
    cv_results_df : pd.DataFrame of all sampled combinations + CV scores.
    """
    print("=" * 80)
    print("[STEP 4] RANDOMIZED SEARCH CV  —  Broad Hyperparameter Exploration")
    print("=" * 80)

    # Two sub-grids to avoid the illegal combination of
    # bootstrap=False + max_samples != None.
    # n_estimators capped at 200 for search speed; the final refit uses more.
    param_dist = [
        # Sub-grid 1: bootstrap=True  → max_samples can be tuned
        {
            'n_estimators'      : [50, 100, 150, 200],
            'max_depth'         : [None, 5, 10, 15, 20],
            'min_samples_split' : [2, 4, 6, 10],
            'min_samples_leaf'  : [1, 2, 3, 5],
            'max_features'      : ['sqrt', 'log2', 0.5, 0.7],
            'bootstrap'         : [True],
            'max_samples'       : [None, 0.7, 0.8, 0.9],
        },
        # Sub-grid 2: bootstrap=False → max_samples must be None
        {
            'n_estimators'      : [50, 100, 150, 200],
            'max_depth'         : [None, 5, 10, 15, 20],
            'min_samples_split' : [2, 4, 6, 10],
            'min_samples_leaf'  : [1, 2, 3, 5],
            'max_features'      : ['sqrt', 'log2', 0.5, 0.7],
            'bootstrap'         : [False],
            'max_samples'       : [None],
        },
    ]

    total_options = sum(sum(len(v) for v in sub.values()) for sub in param_dist)
    print(f"-> Search space  : {total_options} total param options across 2 sub-grids")
    print(f"-> Sampling      : {n_iter} random combinations")
    print(f"-> CV folds      : {cv_folds}-Fold Cross-Validation")
    print(f"-> Scoring metric: neg_root_mean_squared_error\n")

    # n_jobs=1 on the CV object avoids Windows multiprocessing spawn overhead.
    # The individual RF estimator still uses n_jobs=-1 for its internal tree fitting.
    base_rf = RandomForestRegressor(random_state=42, n_jobs=-1)

    t0 = time.time()
    rscv = RandomizedSearchCV(
        estimator=base_rf,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=cv_folds,
        scoring='neg_root_mean_squared_error',
        refit=True,
        verbose=1,
        random_state=42,
        n_jobs=1,            # avoid Windows joblib spawn overhead
        error_score='raise',
    )
    rscv.fit(X_train, y_train)
    elapsed = time.time() - t0

    best_params   = rscv.best_params_
    best_cv_rmse  = -rscv.best_score_
    cv_results_df = pd.DataFrame(rscv.cv_results_)

    print(f"\n-> RandomizedSearchCV completed in {elapsed:.1f}s")
    print(f"-> Best CV RMSE  : {best_cv_rmse:.4f}")
    print(f"-> Best Params   : {best_params}\n")

    return rscv.best_estimator_, best_params, cv_results_df


# ==============================================================================
# STEP 5 – FINAL REFIT WITH BEST PARAMS (replaces slow GridSearchCV)
# ==============================================================================

def grid_search_rf(
    X_train: np.ndarray,
    y_train: pd.Series,
    best_params_from_random: dict,
    cv_folds: int = 3,          # kept for API compatibility, not used internally
) -> tuple:
    """
    Takes the best hyperparameters found by RandomizedSearchCV and trains
    a single final model with a boosted n_estimators=300 for stronger
    performance — no CV loop needed, completes in seconds.

    (Replaces the previous GridSearchCV approach which was generating
    hundreds of fits and taking 10+ minutes on Windows.)

    Returns
    -------
    best_model  : RandomForestRegressor trained on full X_train.
    best_params : dict — the final hyperparameters used.
    """
    print("=" * 80)
    print("[STEP 5] FINAL REFIT  —  Best Params + Boosted n_estimators")
    print("=" * 80)

    # Use the best params from random search, but increase n_estimators
    # for a stronger final model (more trees = lower variance, no extra tuning needed).
    final_params = dict(best_params_from_random)
    boosted_n = max(final_params.get('n_estimators', 150), 300)
    final_params['n_estimators'] = boosted_n

    # Enforce the bootstrap/max_samples constraint just in case
    if not final_params.get('bootstrap', True):
        final_params['max_samples'] = None

    print(f"-> Base params from RandomizedSearch : {best_params_from_random}")
    print(f"-> n_estimators boosted to           : {boosted_n}")
    print(f"-> Final params                      : {final_params}")
    print(f"-> Fitting single model on full training set...")

    t0 = time.time()
    optimized_model = RandomForestRegressor(
        random_state=42,
        n_jobs=-1,       # use all cores for this single big fit
        **final_params,
    )
    optimized_model.fit(X_train, y_train)
    elapsed = (time.time() - t0) * 1000

    print(f"-> Final model trained in {elapsed:.0f} ms!\n")

    return optimized_model, final_params


# ==============================================================================
# STEP 6 – EVALUATION
# ==============================================================================

def evaluate_models(
    models_dict: dict,
    X_test: np.ndarray,
    y_test: pd.Series,
) -> dict:
    """
    Evaluates each model on the hold-out test set.

    Parameters
    ----------
    models_dict : {'Model Name': fitted_estimator, ...}

    Returns
    -------
    metrics_dict : {'Model Name': {'mae', 'mse', 'rmse', 'r2', 'predictions'}}
    """
    print("=" * 80)
    print("[STEP 6] EVALUATING MODELS ON HOLD-OUT TEST SET")
    print("=" * 80)

    header = f"   {'Model':<30} | {'MAE':>8} | {'MSE':>10} | {'RMSE':>8} | {'R² Score':>8}"
    print(header)
    print("   " + "-" * 76)

    metrics_dict = {}
    for name, model in models_dict.items():
        y_pred = model.predict(X_test)
        mae  = mean_absolute_error(y_test, y_pred)
        mse  = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2   = r2_score(y_test, y_pred)

        metrics_dict[name] = {
            'mae'        : mae,
            'mse'        : mse,
            'rmse'       : rmse,
            'r2'         : r2,
            'predictions': y_pred,
        }
        print(
            f"   {name:<30} | {mae:>8.4f} | {mse:>10.4f} | "
            f"{rmse:>8.4f} | {r2:>8.4f}"
        )

    print("   " + "-" * 76)
    print()

    # Identify best model by R2
    best_name = max(metrics_dict, key=lambda k: metrics_dict[k]['r2'])
    print(f"-> Best model on test set: '{best_name}'  "
          f"(R² = {metrics_dict[best_name]['r2']:.4f})\n")

    return metrics_dict


# ==============================================================================
# STEP 7 – CROSS-VALIDATION ON OPTIMIZED MODEL
# ==============================================================================

def cross_validate_optimized(
    model: RandomForestRegressor,
    X_train: np.ndarray,
    y_train: pd.Series,
    cv_folds: int = 10,
):
    """
    Runs an additional k-fold cross-validation on the optimized model
    to report generalisation statistics (mean ± std).
    """
    print("=" * 80)
    print(f"[STEP 7] {cv_folds}-FOLD CROSS-VALIDATION ON OPTIMIZED RANDOM FOREST")
    print("=" * 80)

    r2_scores   = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring='r2',                           n_jobs=1)
    rmse_scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring='neg_root_mean_squared_error', n_jobs=1)
    rmse_scores = -rmse_scores

    print(f"-> R² Scores per fold   : {np.round(r2_scores, 4).tolist()}")
    print(f"-> R²  Mean ± Std       : {r2_scores.mean():.4f} ± {r2_scores.std():.4f}")
    print(f"-> RMSE Scores per fold : {np.round(rmse_scores, 4).tolist()}")
    print(f"-> RMSE Mean ± Std      : {rmse_scores.mean():.4f} ± {rmse_scores.std():.4f}\n")

    return {
        'r2_scores'  : r2_scores,
        'rmse_scores': rmse_scores,
    }


# ==============================================================================
# STEP 8 – FEATURE IMPORTANCE ANALYSIS
# ==============================================================================

def feature_importance_analysis(
    model: RandomForestRegressor,
    feature_names: list,
    top_n: int = 15,
) -> pd.DataFrame:
    """
    Extracts and prints the top-N most important features from the
    optimized Random Forest model.

    Returns
    -------
    pd.DataFrame with columns ['Feature', 'Importance'].
    """
    print("=" * 80)
    print(f"[STEP 8] FEATURE IMPORTANCE ANALYSIS (Top {top_n})")
    print("=" * 80)

    importances = model.feature_importances_
    importance_df = pd.DataFrame({
        'Feature'   : feature_names,
        'Importance': importances,
    }).sort_values('Importance', ascending=False).reset_index(drop=True)

    print(f"   {'Rank':<5} | {'Feature':<30} | {'Importance':>12}")
    print("   " + "-" * 55)
    for i, row in importance_df.head(top_n).iterrows():
        print(f"   {i+1:<5} | {row['Feature']:<30} | {row['Importance']:>12.6f}")
    print()

    return importance_df


# ==============================================================================
# STEP 9 – SAVE ARTIFACTS
# ==============================================================================

def save_artifacts(
    optimized_model: RandomForestRegressor,
    scaler: StandardScaler,
    best_params: dict,
    output_dir: str = 'models',
):
    """Persists the optimized model, scaler, and best hyperparameters to disk."""
    print("=" * 80)
    print("[STEP 9] SAVING OPTIMIZED PIPELINE ARTIFACTS")
    print("=" * 80)

    os.makedirs(output_dir, exist_ok=True)

    model_path  = os.path.join(output_dir, 'random_forest_optimized.pkl')
    scaler_path = os.path.join(output_dir, 'rf_scaler.pkl')
    params_path = os.path.join(output_dir, 'rf_best_params.pkl')

    with open(model_path, 'wb') as f:
        pickle.dump(optimized_model, f)
    print(f"-> Optimized RF model saved   : {model_path}")

    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"-> StandardScaler saved       : {scaler_path}")

    with open(params_path, 'wb') as f:
        pickle.dump(best_params, f)
    print(f"-> Best hyperparameters saved : {params_path}\n")


# ==============================================================================
# STEP 10 – DIAGNOSTIC PLOTS
# ==============================================================================

def generate_diagnostic_plots(
    y_test: pd.Series,
    metrics: dict,
    importance_df: pd.DataFrame,
    cv_results: dict,
    output_dir: str = 'reports/figures',
    top_n_features: int = 15,
):
    """
    Generates and saves five diagnostic figures:
        1.  Actual vs Predicted scatter (all three models side-by-side)
        2.  Residuals plot (optimized RF)
        3.  Feature importances bar chart (top N)
        4.  CV R² score distribution
        5.  CV RMSE score distribution
    """
    if not HAS_VIS:
        print("Warning: matplotlib/seaborn not installed. Skipping plots.")
        return

    print("=" * 80)
    print("[STEP 10] GENERATING DIAGNOSTIC PLOTS")
    print("=" * 80)

    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style='darkgrid', palette='viridis')

    model_names   = list(metrics.keys())
    color_palette = ['#ef4444', '#10b981', '#3b82f6']

    # -------------------------------------------------------------------
    # Plot 1 – Actual vs Predicted (side-by-side)
    # -------------------------------------------------------------------
    fig, axes = plt.subplots(1, len(model_names), figsize=(8 * len(model_names), 7), sharey=True)
    if len(model_names) == 1:
        axes = [axes]

    for ax, name, color in zip(axes, model_names, color_palette):
        y_pred = metrics[name]['predictions']
        r2     = metrics[name]['r2']
        rmse   = metrics[name]['rmse']

        ax.scatter(y_test, y_pred, alpha=0.45, s=35, color=color, edgecolors='k', linewidths=0.3)
        lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
        ax.plot(lims, lims, 'k--', lw=2, label='Perfect Fit')
        ax.set_xlabel('Actual PM2.5 (µg/m³)', fontweight='bold')
        ax.set_ylabel('Predicted PM2.5 (µg/m³)', fontweight='bold')
        ax.set_title(f'{name}\nR²={r2:.4f}  |  RMSE={rmse:.4f}', fontweight='bold')
        ax.legend(fontsize=9)

    plt.suptitle('Random Forest Optimization — Actual vs. Predicted PM2.5', fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()
    p1 = os.path.join(output_dir, 'rf_actual_vs_predicted.png')
    plt.savefig(p1, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"-> Saved: {p1}")

    # -------------------------------------------------------------------
    # Plot 2 – Residuals for Optimized RF
    # -------------------------------------------------------------------
    opt_name  = model_names[-1]      # last model is the fully optimized one
    opt_preds = metrics[opt_name]['predictions']
    residuals = np.array(y_test) - opt_preds

    plt.figure(figsize=(11, 6))
    plt.scatter(opt_preds, residuals, alpha=0.45, s=35, color='#8b5cf6', edgecolors='k', linewidths=0.3)
    plt.axhline(0, color='red', linestyle='--', lw=2, label='Zero Residual')
    plt.xlabel('Predicted PM2.5 (µg/m³)', fontweight='bold')
    plt.ylabel('Residuals (Actual − Predicted)', fontweight='bold')
    plt.title(f'Residual Analysis — {opt_name}', fontweight='bold', pad=12)
    plt.legend()
    plt.tight_layout()
    p2 = os.path.join(output_dir, 'rf_residuals.png')
    plt.savefig(p2, dpi=200)
    plt.close()
    print(f"-> Saved: {p2}")

    # -------------------------------------------------------------------
    # Plot 3 – Feature Importance
    # -------------------------------------------------------------------
    top_df = importance_df.head(top_n_features).copy()
    plt.figure(figsize=(11, 7))
    sns.barplot(
        data=top_df,
        x='Importance',
        y='Feature',
        hue='Feature',          # fix seaborn FutureWarning: palette needs hue
        palette='viridis',
        legend=False,
    )
    plt.xlabel('Mean Decrease in Impurity (MDI)', fontweight='bold')
    plt.title(f'Top {top_n_features} Feature Importances — Optimized Random Forest', fontweight='bold', pad=12)
    plt.tight_layout()
    p3 = os.path.join(output_dir, 'rf_feature_importances.png')
    plt.savefig(p3, dpi=200)
    plt.close()
    print(f"-> Saved: {p3}")

    # -------------------------------------------------------------------
    # Plot 4 – CV R² Distribution
    # -------------------------------------------------------------------
    r2_scores   = cv_results.get('r2_scores', [])
    rmse_scores = cv_results.get('rmse_scores', [])

    if len(r2_scores):
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        axes[0].barh(
            [f'Fold {i+1}' for i in range(len(r2_scores))],
            r2_scores,
            color='#10b981',
            edgecolor='k',
        )
        axes[0].axvline(np.mean(r2_scores), color='red', linestyle='--', lw=2,
                        label=f'Mean R² = {np.mean(r2_scores):.4f}')
        axes[0].set_xlabel('R² Score', fontweight='bold')
        axes[0].set_title('Cross-Validation R² Per Fold', fontweight='bold')
        axes[0].legend()

        axes[1].barh(
            [f'Fold {i+1}' for i in range(len(rmse_scores))],
            rmse_scores,
            color='#f59e0b',
            edgecolor='k',
        )
        axes[1].axvline(np.mean(rmse_scores), color='red', linestyle='--', lw=2,
                        label=f'Mean RMSE = {np.mean(rmse_scores):.4f}')
        axes[1].set_xlabel('RMSE (µg/m³)', fontweight='bold')
        axes[1].set_title('Cross-Validation RMSE Per Fold', fontweight='bold')
        axes[1].legend()

        plt.suptitle('Optimized Random Forest — Cross-Validation Performance', fontweight='bold', fontsize=13)
        plt.tight_layout()
        p4 = os.path.join(output_dir, 'rf_cv_performance.png')
        plt.savefig(p4, dpi=200)
        plt.close()
        print(f"-> Saved: {p4}")

    print()


# ==============================================================================
# MAIN PIPELINE
# ==============================================================================

def main():
    """
    Full Random Forest Optimization Pipeline:

        Step 1  — Load dataset
        Step 2  — Preprocess & scale features
        Step 3  — Train baseline RF
        Step 4  — RandomizedSearchCV (broad search)
        Step 5  — GridSearchCV (fine-tuning)
        Step 6  — Evaluate all three models on the test set
        Step 7  — K-fold cross-validation on the optimized model
        Step 8  — Feature importance analysis
        Step 9  — Save optimized model, scaler, params
        Step 10 — Generate diagnostic plots
    """
    dataset_path = 'data/processed/model_ready_data.csv'

    try:
        # ── Step 1: Load ─────────────────────────────────────────────────────
        df = load_data(dataset_path)

        # ── Step 2: Preprocess ───────────────────────────────────────────────
        X_train, X_test, y_train, y_test, scaler, feature_names = preprocess_data(df)

        # ── Step 3: Baseline RF ──────────────────────────────────────────────
        baseline_rf = train_baseline_rf(X_train, y_train)

        # ── Step 4: Randomized Search ────────────────────────────────────────
        random_rf, random_best_params, cv_results_df = randomized_search_rf(
            X_train, y_train, n_iter=15, cv_folds=3
        )

        # ── Step 5: Grid Search (fine-tune) ─────────────────────────────────
        optimized_rf, final_best_params = grid_search_rf(
            X_train, y_train,
            best_params_from_random=random_best_params,
            cv_folds=3,
        )

        # ── Step 6: Evaluate all models on test set ──────────────────────────
        models = {
            'Baseline RF'           : baseline_rf,
            'Random Search RF'      : random_rf,
            'Optimized RF (Grid)'   : optimized_rf,
        }
        test_metrics = evaluate_models(models, X_test, y_test)

        # ── Step 7: Cross-validate the optimized model ───────────────────────
        cv_results = cross_validate_optimized(optimized_rf, X_train, y_train, cv_folds=5)  # final CV

        # ── Step 8: Feature importance ────────────────────────────────────────
        importance_df = feature_importance_analysis(optimized_rf, feature_names, top_n=15)

        # ── Step 9: Save artifacts ───────────────────────────────────────────
        save_artifacts(optimized_rf, scaler, final_best_params, output_dir='models')

        # ── Step 10: Diagnostic plots ────────────────────────────────────────
        generate_diagnostic_plots(
            y_test,
            test_metrics,
            importance_df,
            cv_results,
            output_dir='reports/figures',
            top_n_features=15,
        )

        # ── Summary ──────────────────────────────────────────────────────────
        print("=" * 80)
        print("SUCCESS: Random Forest Optimization Pipeline completed!")
        print("=" * 80)
        print("\nFinal Evaluation Summary:")
        print(f"   {'Model':<30} | {'R² Score':>10} | {'RMSE':>10}")
        print("   " + "-" * 58)
        for name, m in test_metrics.items():
            print(f"   {name:<30} | {m['r2']:>10.4f} | {m['rmse']:>10.4f}")

        opt_r2   = test_metrics['Optimized RF (Grid)']['r2']
        base_r2  = test_metrics['Baseline RF']['r2']
        gain     = (opt_r2 - base_r2) / max(abs(base_r2), 1e-9) * 100
        print(f"\n   R² improvement over baseline : {gain:+.2f}%")
        print(f"   Final best hyperparameters   : {final_best_params}")
        print("=" * 80)

    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        print("Please run the data preprocessing pipeline first:\n"
              "  python src/data_preprocessing.py\n")
        raise
    except Exception as e:
        print(f"\n[ERROR] Pipeline failed: {e}")
        raise


if __name__ == '__main__':
    main()
