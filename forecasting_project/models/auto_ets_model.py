import pandas as pd
from darts import TimeSeries
from darts.models import AutoETS
from models.base_model import AbstractForecastingModel
from typing import Optional
import pickle

class AutoETSModel(AbstractForecastingModel):
    def __init__(self, model_params: dict, date_column: str, target_column: str):
        super().__init__(model_params)
        self.date_column = date_column
        self.target_column = target_column
        self.model = AutoETS(**self.model_params)

    def fit(self, target_series: pd.DataFrame, covariates: Optional[pd.DataFrame] = None):
        target_ts = TimeSeries.from_dataframe(target_series, self.date_column, self.target_column)
        
        covariates_ts = None
        if covariates is not None:
            covariates_ts = TimeSeries.from_dataframe(covariates, self.date_column)

        self.model.fit(target_ts, future_covariates=covariates_ts)

    def predict(self, n: int, covariates: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        covariates_ts = None
        if covariates is not None:
            covariates_ts = TimeSeries.from_dataframe(covariates, self.date_column)

        predictions = self.model.predict(n, future_covariates=covariates_ts)
        return predictions.to_dataframe()

    def save_model(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load_model(cls, path: str) -> 'AutoETSModel':
        with open(path, 'rb') as f:
            return pickle.load(f)
