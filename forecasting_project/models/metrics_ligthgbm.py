import pandas as pd
import numpy as np
import lightgbm as lgb
from pathlib import Path
from tqdm import tqdm
from loguru import logger
from sktime.performance_metrics.forecasting import MeanSquaredError, MeanAbsolutePercentageError, MeanAbsoluteScaledError, MeanSquaredScaledError
from bayes_opt import BayesianOptimization

# --- Configuration ---
FILENAME_SELLOUT = 'df_sellout_filtered.parquet'
KEY_COLUMNS = ['anio_semana', 'id_pdv', 'id_sku']
TRUE_COLUMN = 'cantidad_vendida'
TARGET_PDVS = ['0027', '0028', '0035', '0050', '0053']
FORECAST_WINDOW = 1
MIN_TRAIN_SIZE = 12  # Minimum weeks to train a model for the first prediction
VALIDATION_WEEKS = 4 # Weeks to use for the validation set for optimization

# --- Feature Configuration ---
FEATURES = ['semana', 'anio', 'lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_8', 'rolling_mean_4', 'rolling_std_4']

# --- Metric Definitions ---
rmse = MeanSquaredError(square_root=True)
smape = MeanAbsolutePercentageError(symmetric=True)
mase = MeanAbsoluteScaledError(sp=FORECAST_WINDOW)
rmsse = MeanSquaredScaledError(sp=FORECAST_WINDOW)

def wmape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculates Weighted Mean Absolute Percentage Error (WMAPE)."""
    sum_y_true = np.sum(y_true)
    if sum_y_true == 0:
        return 0
    return np.sum(np.abs(y_true - y_pred)) / sum_y_true

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_train: np.ndarray | None) -> dict:
    """Calculates all forecasting metrics for a given group."""
    if y_train is None or len(y_train) < 2:
        return {
            'rmse': rmse(y_true, y_pred), 'smape': smape(y_true, y_pred),
            'wmape': wmape(y_true, y_pred), 'rmsse': np.nan, 'mase': np.nan
        }
    return {
        'rmse': rmse(y_true, y_pred), 'smape': smape(y_true, y_pred),
        'wmape': wmape(y_true, y_pred), 'rmsse': rmsse(y_true, y_pred, y_train=y_train),
        'mase': mase(y_true, y_pred, y_train=y_train)
    }

def create_features(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """Creates time-series features from the main dataframe."""
    df_copy = df.copy()
    df_copy['semana'] = df_copy['anio_semana'] % 100
    df_copy['anio'] = df_copy['anio_semana'] // 100
    for lag in [1, 2, 3, 4, 8]:
        df_copy[f'lag_{lag}'] = df_copy.groupby(['id_pdv', 'id_sku'])[target_col].shift(lag)
    shifted_sales = df_copy.groupby(['id_pdv', 'id_sku'])[target_col].shift(1)
    df_copy['rolling_mean_4'] = shifted_sales.rolling(window=4, min_periods=1).mean()
    df_copy['rolling_std_4'] = shifted_sales.rolling(window=4, min_periods=1).std()
    return df_copy

def model_lightgbm(df_group: pd.DataFrame, features: list, target_col: str, initial_window: int, lgb_params: dict) -> list:
    """Performs expanding window backtesting for a single time series using LightGBM with optimized parameters."""
    df_group = df_group.sort_values('anio_semana').reset_index(drop=True)
    df_featured = create_features(df_group, target_col)
    predictions = []
    if len(df_featured) <= initial_window:
        return []

    for i in range(initial_window, len(df_featured)):
        train_df = df_featured.iloc[:i]
        test_df = df_featured.iloc[i:i+1]
        train_df_clean = train_df.dropna(subset=features)
        if train_df_clean.empty or test_df[features].isnull().values.any():
            continue
        X_train, y_train = train_df_clean[features], train_df_clean[target_col]
        X_test, y_test = test_df[features], test_df[target_col]
        
        # Use the optimized parameters to train the model
        model = lgb.LGBMRegressor(**lgb_params)
        model.fit(X_train, y_train)
        prediction_value = model.predict(X_test)
        
        pred_record = {
            'anio_semana': test_df['anio_semana'].iloc[0], 'id_pdv': test_df['id_pdv'].iloc[0],
            'id_sku': test_df['id_sku'].iloc[0], target_col: y_test.iloc[0],
            'prediction': np.maximum(0, np.round(prediction_value[0])).astype(int)
        }
        predictions.append(pred_record)
    return predictions

if __name__ == "__main__":
    # 1. Read and prepare the data
    logger.info("--- Data Loading and Preparation ---")
    df_sellout = pd.read_parquet(FILENAME_SELLOUT)
    df_sellout_per_week = df_sellout.groupby(KEY_COLUMNS).agg({TRUE_COLUMN: 'sum'}).reset_index()
    df_filtered = df_sellout_per_week[df_sellout_per_week['id_pdv'].isin(TARGET_PDVS)].copy()
    if df_filtered.empty:
        logger.warning(f"No data found for PDVs {TARGET_PDVS}. Exiting.")
        exit()

    # 2. Create features and split data for optimization
    logger.info(f"--- Hyperparameter Optimization using {VALIDATION_WEEKS} validation weeks ---")
    df_featured = create_features(df_filtered, TRUE_COLUMN).dropna(subset=FEATURES)
    unique_weeks = sorted(df_featured['anio_semana'].unique())
    
    # Define the split point for training and validation
    split_point = unique_weeks[-VALIDATION_WEEKS]
    train_opt_df = df_featured[df_featured['anio_semana'] < split_point]
    val_opt_df = df_featured[df_featured['anio_semana'] >= split_point]
    
    X_train_opt, y_train_opt = train_opt_df[FEATURES], train_opt_df[TRUE_COLUMN]
    X_val_opt, y_val_opt = val_opt_df[FEATURES], val_opt_df[TRUE_COLUMN]
    
    # 3. Define the objective function for Bayesian Optimization
    def lgbm_objective(n_estimators, num_leaves, learning_rate, reg_alpha, reg_lambda, max_depth):
        # BayesianOptimization passes floats, so we need to convert some to int
        params = {
            'n_estimators': int(n_estimators), 'num_leaves': int(num_leaves),
            'learning_rate': learning_rate, 'reg_alpha': reg_alpha,
            'reg_lambda': reg_lambda, 'max_depth': int(max_depth),
            'objective': 'regression_l1', 'random_state': 42, 'verbose': -1
        }
        model = lgb.LGBMRegressor(**params)
        model.fit(X_train_opt, y_train_opt)
        preds = model.predict(X_val_opt)
        
        # We want to minimize error, so we return the negative error (to maximize)
        return -wmape(y_val_opt, preds)

    # Define the hyperparameter search space
    pbounds = {
        'n_estimators': (50, 250), 'num_leaves': (10, 50),
        'learning_rate': (0.01, 0.2), 'reg_alpha': (0.0, 1.0),
        'reg_lambda': (0.0, 1.0), 'max_depth': (3, 10)
    }

    optimizer = BayesianOptimization(f=lgbm_objective, pbounds=pbounds, random_state=42, verbose=2)
    logger.info("Running Bayesian Optimization...")
    optimizer.maximize(init_points=5, n_iter=15) # init_points: random exploration, n_iter: exploitation
    
    best_params = optimizer.max['params']
    # Convert float params to int where needed for the final model
    best_params['n_estimators'] = int(best_params['n_estimators'])
    best_params['num_leaves'] = int(best_params['num_leaves'])
    best_params['max_depth'] = int(best_params['max_depth'])
    best_params.update({'objective': 'regression_l1', 'random_state': 42, 'verbose': -1})
    
    logger.success(f"Optimal Hyperparameters Found: {best_params}")

    # 4. Run final backtesting with the best hyperparameters
    logger.info("--- Final Backtesting with Optimal Parameters ---")
    all_predictions_list = []
    grouped_data = df_filtered.groupby(['id_pdv', 'id_sku'])
    for (pdv_id, sku), df_group in tqdm(grouped_data, desc="Backtesting PDV-SKU combinations"):
        group_predictions = model_lightgbm(
            df_group=df_group, features=FEATURES, target_col=TRUE_COLUMN,
            initial_window=MIN_TRAIN_SIZE, lgb_params=best_params
        )
        if group_predictions:
            all_predictions_list.extend(group_predictions)

    if not all_predictions_list:
        logger.error("No predictions were generated during backtesting.")
        exit()

    # 5. Consolidate results and calculate final metrics
    logger.info("--- Consolidating Results and Calculating Metrics ---")
    predictions_df = pd.DataFrame(all_predictions_list)
    y_train_lookup = {
        (pdv_id, sku): group[TRUE_COLUMN].values
        for (pdv_id, sku), group in df_sellout_per_week.groupby(['id_pdv', 'id_sku'])
    }
    metrics_list = []
    for (pdv_id, sku), pred_group in tqdm(predictions_df.groupby(['id_pdv', 'id_sku']), desc="Calculating Metrics"):
        full_ts_data_np = y_train_lookup.get((pdv_id, sku))
        y_train_for_scaling = full_ts_data_np[:-len(pred_group)] if full_ts_data_np is not None else None
        
        metrics_dict = calculate_metrics(
            y_true=pred_group[TRUE_COLUMN].values, y_pred=pred_group['prediction'].values,
            y_train=y_train_for_scaling if y_train_for_scaling is not None and len(y_train_for_scaling) > 0 else None
        )
        metrics_dict['id_pdv'] = pdv_id
        metrics_dict['id_sku'] = sku
        metrics_list.append(metrics_dict)
        
    metrics_df = pd.DataFrame(metrics_list)

    # 6. Export DataFrames to Parquet
    pred_filename = 'predictions_lightgbm_bayesian_opt.parquet'
    metrics_filename = 'metrics_lightgbm_bayesian_opt.parquet'
    
    logger.info(f"Exporting backtest predictions to {pred_filename}")
    predictions_df.to_parquet(pred_filename, index=False)
    
    logger.info(f"Exporting metrics to {metrics_filename}")
    metrics_df.to_parquet(metrics_filename, index=False)
    
    logger.info("Script finished successfully.")
    print("\n--- Backtest Predictions DataFrame Head ---")
    print(predictions_df.head())
    print(f"\nTotal backtest predictions made: {len(predictions_df)}")
    
    print("\n--- Metrics DataFrame Head ---")
    print(metrics_df.head())