import pandas as pd
from statsmodels.tsa.stattools import acf
from scipy.signal import argrelmax
from scipy.stats import norm
import numpy as np
import math
from loguru import logger

def _bartlett_formula(r: np.ndarray, m: int, length: int) -> float:
    """
    Computes the standard error of `r` at order `m` with respect to `length` according to Bartlett's formula.
    """
    if m == 1:
        return math.sqrt(1 / length)
    else:
        return math.sqrt((1 + 2 * sum(map(lambda x: x**2, r[: m - 1]))) / length)

def analyze_seasonality(
    series_df: pd.DataFrame, 
    target_column: str,
    max_lag: int = 48, 
    alpha: float = 0.05
) -> tuple[bool, int | None]:
    """
    Analyzes the seasonality of a time series using its Auto-Correlation Function (ACF).

    Args:
        series_df (pd.DataFrame): The DataFrame containing the time series.
        target_column (str): The name of the target column.
        max_lag (int): The maximal lag to check for seasonality.
        alpha (float): The significance level.
    """
    try:
        ts_values = series_df[target_column].to_numpy()

        if np.unique(ts_values).shape[0] == 1:
            logger.debug("Series is constant, no seasonality.")
            return False, None

        # Calculate ACF using statsmodels
        r = acf(ts_values, nlags=max_lag, fft=False)
        r = np.asarray(r)

        # Find local maxima in the ACF plot
        candidates = argrelmax(r)[0]

        if len(candidates) == 0:
            logger.debug("No local maxima found in ACF. Series is not seasonal.")
            return False, None

        # Remove r[0], the auto-correlation at lag 0
        r_short = r[1:]

        # Significance test
        band_upper = np.mean(r_short) + norm.ppf(1 - alpha / 2) * np.var(r_short)

        for candidate in candidates:
            # candidate-1 because r_short is r[1:]
            stat = _bartlett_formula(r_short, candidate - 1, len(ts_values))
            if r[candidate] > stat * band_upper:
                logger.debug(f"Result: Series IS seasonal with period {candidate}.")
                return True, candidate
        
        logger.debug("Result: Series is NOT seasonal (no significant peaks in ACF).")
        return False, None

    except Exception as e:
        logger.warning(f"Could not perform seasonality analysis: {e}")
        return False, None
