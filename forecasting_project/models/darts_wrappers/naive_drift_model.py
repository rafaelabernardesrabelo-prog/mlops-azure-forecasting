import pandas as pd
from typing import Optional
from darts import TimeSeries
from darts.models import NaiveDrift
from models.base_model import AbstractForecastingModel
import pickle

class NaiveDriftModel(AbstractForecastingModel):
    def __init__(self, model_params: dict, date_column: str, target_column: str):
        super().__init__(model_params)
        self.date_column = date_column
        self.target_column = target_column
        self.model = NaiveDrift(**self.model_params)

    def fit(self, target_series: pd.DataFrame, covariates: Optional[pd.DataFrame] = None):
        target_ts = TimeSeries.from_dataframe(target_series, self.date_column, self.target_column)
        self.model.fit(target_ts)

    def predict(self, n: int, covariates: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        predictions = self.model.predict(n)
        return predictions.to_dataframe()

    def save_model(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load_model(cls, path: str) -> 'NaiveDriftModel':
        with open(path, 'rb') as f:
            return pickle.load(f)
