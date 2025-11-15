import pandas as pd
from typing import Optional
from darts import TimeSeries
from darts.models import Prophet
from models.base_model import AbstractForecastingModel
import pickle

class ProphetModel(AbstractForecastingModel):
    def __init__(self, model_params: dict, date_column: str, target_column: str):
        super().__init__(model_params)
        self.date_column = date_column
        self.target_column = target_column
        self.model = Prophet(**self.model_params)

    def fit(self, target_series: pd.DataFrame, covariates: Optional[pd.DataFrame] = None):
        # Prophet requires columns to be named 'ds' and 'y'
        prophet_df = target_series.rename(columns={self.date_column: 'ds', self.target_column: 'y'})
        
        # Darts' Prophet wrapper handles the conversion internally, but we need to ensure the column names are right for the wrapper
        target_ts = TimeSeries.from_dataframe(prophet_df, 'ds', 'y')

        covariates_ts = None
        if covariates is not None:
            covariates_ts = TimeSeries.from_dataframe(covariates, self.date_column)

        self.model.fit(target_ts, future_covariates=covariates_ts)

    def predict(self, n: int, covariates: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        covariates_ts = None
        if covariates is not None:
            covariates_ts = TimeSeries.from_dataframe(covariates, self.date_column)

        predictions = self.model.predict(n, future_covariates=covariates_ts)
        
        # Convert back to original column names
        predictions_df = predictions.to_dataframe()
        predictions_df = predictions_df.rename(columns={'yhat': self.target_column})
        return predictions_df

    def save_model(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load_model(cls, path: str) -> 'ProphetModel':
        with open(path, 'rb') as f:
            return pickle.load(f)
