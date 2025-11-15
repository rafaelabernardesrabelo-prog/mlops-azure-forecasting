import pandas as pd
from pathlib import Path
from tqdm import tqdm
from sktime.performance_metrics.forecasting import MeanSquaredError, MeanAbsolutePercentageError, MeanAbsoluteScaledError, MeanSquaredScaledError
from sktime.split import ExpandingWindowSplitter
from sktime.forecasting.theta import ThetaForecaster
from sktime.forecasting.compose import TransformedTargetForecaster
from sktime.transformations.series.detrend import Deseasonalizer
from loguru import logger
from multiprocessing import Pool, cpu_count
import numpy as np
from functools import partial

# FILENAME_FORECAST = 'data/data/farmaciasp/processed/forecast/Forecast_Bebés.parquet'
FILENAME_FORECAST = 'df_sellout_per_week_with_theta.parquet'
FILENAME_SELLOUT = 'df_sellout_filtered.parquet'
KEY_COLUMNS = ['anio_semana', 'id_pdv', 'id_sku']
FORECAST_COLUMN = 'theta'
TRUE_COLUMN = 'cantidad_vendida'
FORECAST_WINDOW = 1

rmse = MeanSquaredError(square_root=True)
smape = MeanAbsolutePercentageError(symmetric=True)
mase = MeanAbsoluteScaledError(sp=FORECAST_WINDOW)
rmsse = MeanSquaredScaledError(sp=FORECAST_WINDOW)

def wmape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Weighted Mean Absolute Percentage Error (WMAPE).

    Parameters
    ----------
    y_true : np.ndarray
        Array of true values.
    y_pred : np.ndarray
        Array of predicted values.

    Returns
    -------
    float
        WMAPE score.
    """
    sum_y_true = sum(abs(y_true))
    if (sum_y_true == 0) & (sum(abs(y_pred)) == 0):
        wmape_score = 0
    elif sum_y_true == 0:
        wmape_score = np.inf
    else:
        wmape_score = sum(abs(y_true - y_pred)) / sum_y_true
    return wmape_score


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_train: np.ndarray | None) -> dict:
    """
    Calculate forecasting metrics for a given group.

    Parameters
    ----------
    y_true : np.ndarray
        Array of true values.
    y_pred : np.ndarray
        Array of predicted values.
    y_train : np.ndarray or None
        Array of training values, or None if not available.

    Returns
    -------
    dict
        Dictionary with metric names and their values.
    """
    y_true = np.ravel(y_true)
    y_pred = np.ravel(y_pred)

    if y_train is None:
        return {
            'rmse': rmse(y_true, y_pred),
            'smape': smape(y_true, y_pred),
            'wmape': wmape(y_true, y_pred),
            'rmsse': np.nan,  # No training data available
            'mase': np.nan   # No training data available
    }

    y_train = np.ravel(y_train)
    return {
        'rmse': rmse(y_true, y_pred),
        'smape': smape(y_true, y_pred),
        'wmape': wmape(y_true, y_pred),
        'rmsse': rmsse(y_true, y_pred, y_train=y_train),
        'mase': mase(y_true, y_pred, y_train=y_train)
    }


def process_group(
    group_data: tuple,
    df_sellout_per_week: pd.DataFrame,
    true_column: str,
    forecast_column: str
) -> tuple | None:
    """
    Processes a single group DataFrame and calculates metrics.

    Parameters
    ----------
    group_data : tuple
        Tuple containing (group_key, group_df), where group_key is (id_pdv, id_sku) and group_df is the DataFrame for the group.
    df_sellout_per_week : pd.DataFrame
        DataFrame containing sellout and forecast data per week.
    TRUE_COLUMN : str
        Name of the column with true values.
    FORECAST_COLUMN : str
        Name of the column with forecast values.

    Returns
    -------
    tuple or None
        Tuple (id_pdv, id_sku, metrics_dict) if metrics can be calculated, otherwise None.
    """
    name, df_group = group_data
    id_pdv, id_sku = name  # name is a tuple (id_pdv, id_sku)
    # id_pdv = name  # name is a tuple (id_pdv, id_sku)

    y_true = df_group[true_column].values
    y_pred = df_group[forecast_column].values

    # y_train: valores de TRUE_COLUMN onde FORECAST_COLUMN é nulo para o mesmo grupo
    y_train = df_sellout_per_week[(df_sellout_per_week['id_pdv'] == id_pdv) &
                                  (df_sellout_per_week['id_sku'] == id_sku) &
                                  (df_sellout_per_week[forecast_column].isna())][true_column].values

    # Skip groups with empty y_train
    if len(y_train) < 2:
        y_train = None

    if not df_group.empty:
        return (id_pdv, id_sku, calculate_metrics(y_true, y_pred, y_train))
        # return (id_pdv,  calculate_metrics(y_true, y_pred, y_train))
    return None

def model_theta(id_pdv, id_sku, df_sellout_per_week):
    """
    Model Theta for a specific pdv and sku.
        
    Parameters
    ----------
    df_sellout_per_week : pd.DataFrame
        DataFrame containing sellout and forecast data per week.
    id_pdv : str
        ID of the point of sale (pdv).
    id_sku : str
        ID of the stock keeping unit (sku).
    """
    # Filter the DataFrame for the specific pdv and sku
    df_sellout_per_week_pdv_sku = df_sellout_per_week[(df_sellout_per_week['id_pdv'] == id_pdv) & (df_sellout_per_week['id_sku'] == id_sku)]
    # Quantidade de semanas até o primeiro Forecast
    initial_window = (
        df_sellout_per_week_pdv_sku
        .sort_values(by=['anio_semana'])
        [FORECAST_COLUMN]
        .notna()
        .cumsum()
        .eq(1)
        .reset_index()
        .idxmax()[FORECAST_COLUMN]
    )
    splitter = ExpandingWindowSplitter(initial_window=initial_window, step_length=1)
    # forecaster = ThetaForecaster(deseasonalize=False)  # Explicitly disable seasonality
    forecaster = TransformedTargetForecaster(steps=[
        ("deseasonalizer", Deseasonalizer()),  # Explicitly enable deseasonalization
        ("theta", ThetaForecaster(deseasonalize=False))
    ])
    pred = {}
    fcst = {}
    for i, j in splitter.split(df_sellout_per_week_pdv_sku[TRUE_COLUMN]):
        y = df_sellout_per_week_pdv_sku[TRUE_COLUMN].iloc[i].values
        # Ensure there are enough data points for the deseasonalizer
        if len(y) < 2:
            continue
        y_pred = forecaster.fit_predict(y, fh=[1])
        # pred[tuple(j)] = y_pred  # Use tuple(j) to ensure hashable keys
        fcst.update({df_sellout_per_week_pdv_sku.iloc[j[0].item()].name : round(y_pred[0].item(),0)})
    pred['id_pdv'] = id_pdv
    pred['id_sku'] = id_sku
    pred['forecast'] = fcst
    return pred

def initial_sale(df, id_pdv, id_sku):
    """
    Get the initial sale week for a specific pdv and sku.
        
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing sellout data.
    id_pdv : str
        ID of the point of sale (pdv).
    id_sku : str
        ID of the stock keeping unit (sku).
    """
    df_filtered = df[(df['id_pdv'] == id_pdv) & (df['id_sku'] == id_sku)]
    if df_filtered.empty:
        return None
    return df_filtered['anio_semana'].min()

if __name__ == "__main__":
    # Read the Parquet files
    logger.info(f"Reading Parquet files: {FILENAME_SELLOUT} and {FILENAME_FORECAST}")
    df_sellout = pd.read_parquet(FILENAME_SELLOUT)
    df_forecast = pd.read_parquet(FILENAME_FORECAST)
    # Aggregate sellout data per week for comparison with forecast
    logger.info("Aggregating sellout data per week for comparison with forecast.")
    df_sellout_per_week = df_sellout.groupby(KEY_COLUMNS).agg({TRUE_COLUMN: 'sum'}).reset_index()
    # Merge sellout data with forecast data
    logger.info("Merging sellout data with forecast data.")
    df_sellout_per_week = df_sellout_per_week.merge(df_forecast[KEY_COLUMNS + [FORECAST_COLUMN]], on=KEY_COLUMNS, how='left')
    logger.info("Sellout + Forecast column dataframe shape after merging: %s", df_sellout_per_week.shape)

    # Size of forecast values
    logger.info(f"Size of forecast values: {df_sellout_per_week[FORECAST_COLUMN].notna().sum():,d}".replace(',','.'))
    logger.info(f"Size of train values: {df_sellout_per_week[FORECAST_COLUMN].isna().sum():,d}".replace(',','.'))

    # Get the groupby object
    df_metrics = df_sellout_per_week.dropna()
    groups = df_metrics.groupby(['id_pdv', 'id_sku'])
    # groups = df_metrics.groupby(['id_pdv'])
    len_groups = len(groups)
    print(f"Number of unique groups: {len_groups}")

    # Multiprocessign for metrics calculation
    process_group_partial = partial(process_group, df_sellout_per_week=df_sellout_per_week, true_column=TRUE_COLUMN, forecast_column=FORECAST_COLUMN)
    with Pool() as pool:
        logger.info("Starting parallel processing of metrics calculation.")
        results = list(tqdm(pool.imap_unordered(process_group_partial, groups), total=len_groups, desc="Processing groups"))

    # Filter out None results and convert to DataFrame
    metrics = pd.DataFrame([
        {'id_pdv': res[0], 'id_sku': res[1], **res[2]} for res in results if res is not None
        # {'id_pdv': res[0], **res[1]} for res in results if res is not None
    ])

    print(metrics.shape)
    print(metrics.head())
    metrics.to_parquet('metrics_theta_pdv_sku.parquet', index=False)

    
    # # Populate the forecast value in the DataFrame in a column 'theta'
    # row_idx = df_sellout_per_week_pdv_sku.iloc[j].index
    # if y_pred is not None:
    #     df_sellout_per_week_pdv_sku.loc[row_idx, 'theta'] = round(y_pred[0][0],0)

    # Multiprocessing for model_theta calculation
    # def model_theta_partial(group):
    #     id_pdv, id_sku = group
    #     return model_theta(id_pdv, id_sku, df_sellout_per_week)

    # def model_theta_wrapper(group_tuple):
    #     id_pdv, id_sku = group_tuple
    #     return model_theta(id_pdv, id_sku, df_sellout_per_week)

    # model_groups = df_metrics[['id_pdv', 'id_sku']].drop_duplicates().itertuples(index=False, name=None)
    # with Pool() as pool:
    #     logger.info("Starting parallel processing of model_theta calculation.")
    #     theta_results = list(tqdm(pool.imap_unordered(model_theta_wrapper, model_groups), total=len_groups, desc="Processing model_theta"))

    # for group in tqdm(model_groups):
    #     id_pdv, id_sku = group
    #     # Call the model_theta function for each group
    #     theta_result = model_theta(id_pdv, id_sku, df_sellout_per_week)

    # # Combine results into a DataFrame or process as needed
    # all_forecasts = []
    # for res in theta_results:
    #     if res and 'forecast' in res:
    #         # Create a temporary DataFrame from the forecast dictionary
    #         df_forecast = pd.DataFrame.from_dict(res['forecast'], orient='index', columns=['theta'])
    #         df_forecast['id_pdv'] = res['id_pdv']
    #         df_forecast['id_sku'] = res['id_sku']
    #         all_forecasts.append(df_forecast)

    # # Concatenate all the small DataFrames into one
    # if all_forecasts:
    #     theta_metrics = pd.concat(all_forecasts)
    # else:
    #     theta_metrics = pd.DataFrame()

    # df_sellout_per_week = df_sellout_per_week.merge(theta_metrics['theta'], left_index=True, right_index=True, how='left')
    # df_sellout_per_week.to_parquet('df_sellout_per_week_with_theta.parquet')
    # print(df_sellout_per_week.head())
