from abc import ABC, abstractmethod
from typing import Optional
import pandas as pd

class AbstractForecastingModel(ABC):
    """
    An abstract base class for forecasting models.
    This class is library-agnostic and works with pandas DataFrames.
    """

    @abstractmethod
    def __init__(self, model_params: dict):
        """
        Initializes the forecasting model.

        Args:
            model_params (dict): A dictionary of hyperparameters for the model.
        """
        self.model_params = model_params
        self.model = None
    @abstractmethod
    def fit(self, target_series: pd.Series, covariates: Optional[pd.DataFrame] = None):
        """
        Fits the model to the target series.

        Args:
            target_series (pd.Series): The time series to train the model on.
                                          Must contain at least a date and a target column.
            covariates (Optional[pd.DataFrame], optional): Covariates. Defaults to None.
        """
        raise NotImplementedError
    
    @abstractmethod
    def predict(self, pred_index: pd.DatetimeIndex, covariates: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Generates forecasts for a given horizon.

        Args:
            pred_index (pd.DatetimeIndex): The forecast horizon.
            covariates (Optional[pd.DataFrame], optional): Covariates for the forecast period.
                                                      Defaults to None.

        Returns:
            pd.DataFrame: A DataFrame containing the predictions.
        """
        raise NotImplementedError

    @abstractmethod
    def save_model(self, path: str):
        """
        Saves the model's state to a file.

        Args:
            path (str): The path where the model should be saved.
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def load_model(cls, path: str) -> 'AbstractForecastingModel':
        """
        Loads a model from a file.

        Args:
            path (str): The path from where to load the model.

        Returns:
            AbstractForecastingModel: An instance of the loaded model.
        """
        raise NotImplementedError

    def get_params(self) -> dict:
        """
        Returns the hyperparameters of the model.

        Returns:
            dict: The model's hyperparameters.
        """
        return self.model_params
