import pandas as pd
from tqdm import tqdm
import yaml
from typing import List, Tuple
from .time_series_data import TimeSeriesData, TimeSeriesDataset
from multiprocessing import Pool
from functools import partial

def _process_series(df_timeseries: pd.DataFrame, date_column: str, target_column: str, feature_columns: List[str], frequency: str) -> TimeSeriesData:
    """
    Processes a single time series from a dataframe group.
    """
    series_id = df_timeseries['unique_id'].iloc[0]
    target_df = df_timeseries[[date_column, target_column]].copy()
        
    covariates_df = None
    if feature_columns:
        covariates_df = df_timeseries[[date_column] + feature_columns].copy()

    ts_data = TimeSeriesData(
        series_id=series_id,
        data=target_df,
        date_column=date_column,
        target_column=target_column,
        freq=frequency,
        covariates_cols=covariates_df
    )
    return ts_data

def load_and_prepare_data(config: dict) -> TimeSeriesDataset:
    """
    Loads data from a parquet file, filters it, and creates a TimeSeriesDataset.

    Args:
        config (dict): The configuration dictionary.

    Returns:
        TimeSeriesDataset: An object containing all time series from the data.
    """
    data_path = config['data_path']
    date_column = config['date_column']
    target_column = config['target_column']
    key_columns = config.get('key_columns', [])
    feature_columns = config.get('feature_columns', [])
    frequency = config.get('frequency', None)
    
    df = pd.read_parquet(data_path)
    
    df[date_column] = pd.to_datetime(df[date_column])
    df['unique_id'] = df[key_columns].apply('_'.join, axis=1)

    groups = [group for _, group in df.groupby('unique_id')]
    
    process_func = partial(_process_series, date_column=date_column, target_column=target_column, feature_columns=feature_columns, frequency=frequency)

    with Pool() as pool:
        results_iterator = pool.imap_unordered(process_func, groups)
        time_series_list = list(tqdm(results_iterator, total=len(groups), desc="Preparing time series data"))

    return TimeSeriesDataset(time_series=time_series_list)
