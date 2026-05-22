import os
import pickle
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
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
    Cleans and splits the dataset for Gradient Boosting Regressor.
    Note: Tree-based models are scale-invariant, so scaling features is not required.
    Keeping features in their raw unit range makes feature importance highly interpretable!
    """
    print("=" * 80)
    print("[STEP 2] PREPROCESSING DATA (SPLITTING FEATURES & TARGET)")
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
    print(f"-> Test split size: {X_test.shape[0]} samples\n")
    
    return X_train, X_test, y_train, y_test, feature_names


def train_gradient_boosting(X_train: pd.DataFrame, y_train: pd.Series) -> GradientBoostingRegressor:
    """
    Trains a Gradient Boosting Regressor (Person 2 Model).
    - Sequential boosting ensemble.
    """
    print("=" * 80)
    print("[STEP 3] TRAINING GRADIENT BOOSTING REGRESSOR (PERSON 2)")
    print("=" * 80)
    
    start_time = time.time()
    gb_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    gb_model.fit(X_train, y_train)
    duration_ms = (time.time() - start_time) * 1000
    
    print(f"-> Gradient Boosting trained successfully in {duration_ms:.2f} ms!\n")
    return gb_model


def evaluate_model(
    model: GradientBoostingRegressor,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> dict:
    """
    Evaluates the Gradient Boosting model on the test dataset.
    """
    print("=" * 80)
    print("[STEP 4] MODEL PERFORMANCE EVALUATION (GRADIENT BOOSTING)")
    print("=" * 80)
    
    # Predict
    y_pred = model.predict(X_test)
    
    # Calculate Metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    # Print Performance Metrics Table
    print(f"   {'Evaluation Metric':<30} | Value")
    print("   " + "-" * 45)
    print(f"   {'Mean Absolute Error (MAE)':<30} : {mae:.4f} ug/m3")
    print(f"   {'Mean Squared Error (MSE)':<30} : {mse:.4f}")
    print(f"   {'Root Mean Sq. Error (RMSE)':<30} : {rmse:.4f} ug/m3")
    print(f"   {'R-squared Score (R2)':<30} : {r2:.4f}")
    print("   " + "-" * 45)
    print("-> Model performance calculated successfully!\n")
    
    return {
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'r2': r2,
        'predictions': y_pred
    }


def save_artifacts(model: GradientBoostingRegressor, output_dir: str = 'models'):
    """
    Saves the trained Gradient Boosting model to disk.
    """
    print("=" * 80)
    print("[STEP 5] SAVING GRADIENT BOOSTING MODEL ARTIFACT")
    print("=" * 80)
    
    os.makedirs(output_dir, exist_ok=True)
    
    file_path = os.path.join(output_dir, "gradient_boosting_model.pkl")
    with open(file_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"-> Saved trained Gradient Boosting model to: {file_path}")
    print("\n-> Model saved successfully!\n")


def generate_plots(
    model: GradientBoostingRegressor,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    y_pred: np.ndarray,
    feature_names: list,
    output_dir: str = 'reports/figures'
):
    """
    Generates and saves diagnostic evaluation plots:
    - Actual vs. Predicted values.
    - Feature Importance plot.
    """
    if not HAS_VIS:
        print("Warning: Matplotlib or Seaborn not installed. Skipping plot generation.")
        return
        
    print("=" * 80)
    print("[STEP 6] GENERATING DIAGNOSTIC PLOTS (GRADIENT BOOSTING)")
    print("=" * 80)
    
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="darkgrid")
    
    # Plot 1: True vs Predicted
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.6, color='#8b5cf6', edgecolors='k', s=45, label='GB Predictions')
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    plt.plot(lims, lims, 'r--', alpha=0.8, linewidth=2, label='Perfect Fit')
    plt.xlabel('Actual PM2.5 (ug/m3)', fontweight='bold')
    plt.ylabel('Predicted PM2.5 (ug/m3)', fontweight='bold')
    plt.title('Gradient Boosting: Actual vs. Predicted PM2.5 Values', fontweight='bold', pad=15)
    plt.legend()
    plt.tight_layout()
    
    actual_vs_pred_path = os.path.join(output_dir, 'actual_vs_predicted_trees.png')
    plt.savefig(actual_vs_pred_path, dpi=300)
    plt.close()
    print(f"-> Saved Actual vs Predicted plot to: {actual_vs_pred_path}")
    
    # Plot 2: Feature Importance (Top 10)
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:10]  # Get top 10 features
    
    top_importances = importances[indices]
    top_names = [feature_names[i] for i in indices]
    
    plt.figure(figsize=(12, 7))
    sns.barplot(x=top_importances, y=top_names, palette='viridis')
    plt.title('Gradient Boosting - Top 10 Feature Importances', fontweight='bold', fontsize=14, pad=15)
    plt.xlabel('Relative Importance', fontweight='bold')
    plt.tight_layout()
    
    importance_path = os.path.join(output_dir, 'tree_feature_importances.png')
    plt.savefig(importance_path, dpi=300)
    plt.close()
    print(f"-> Saved Feature Importance plot to: {importance_path}\n")


def main():
    dataset_path = os.path.join('data', 'processed', 'model_ready_data.csv')
    
    try:
        # 1. Load Data
        df = load_data(dataset_path)
        
        # 2. Preprocess Data
        X_train, X_test, y_train, y_test, feature_names = preprocess_data(df)
        
        # 3. Train Model
        gb_model = train_gradient_boosting(X_train, y_train)
        
        # 4. Evaluate Model
        metrics = evaluate_model(gb_model, X_test, y_test)
        
        # 5. Save Artifact (.pkl file)
        save_artifacts(gb_model)
        
        # 6. Generate visualizations
        generate_plots(gb_model, X_test, y_test, metrics['predictions'], feature_names)
        
        print("Success: Day 9 Gradient Boosting Regressor Pipeline completed successfully!")
        print("=" * 80)
        
    except Exception as e:
        print(f"\nError: Pipeline failed with error: {str(e)}")
        raise


if __name__ == '__main__':
    main()
