import pandas as pd
import numpy as np
from typing import Optional
from statsforecast.models import AutoARIMA
from models.base_model import AbstractForecastingModel
import pickle

class AutoARIMAModel(AbstractForecastingModel):
    def __init__(self, model_params: dict):
        super().__init__(model_params)
        self.model = None  # Model will be initialized in fit()

    def fit(self, target_series: pd.Series, covariates: Optional[pd.DataFrame] = None):

        # Instantiate AutoARIMA with prepared parameters
        model = AutoARIMA(**self.model_params)

        # Fit the AutoARIMA model; use positional args since the model's fit expects (target_series, exogenous)
        self.model = model.fit(
            target_series.to_numpy(),
            covariates.to_numpy() if covariates is not None else None
        )

    def predict(self, pred_index: pd.DatetimeIndex, covariates: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        if self.model is None:
            raise RuntimeError("The model must be fitted before making predictions.")
            
        predictions = self.model.predict(
            h=pred_index.shape[0],
            X=covariates.to_numpy() if covariates is not None else None)
        
        predictions_df = pd.DataFrame(index=pred_index)
        predictions_df['y_pred'] = predictions['mean']

        # Return a DataFrame with date and prediction columns
        return predictions_df

    def save_model(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load_model(cls, path: str) -> 'AutoARIMAModel':
        with open(path, 'rb') as f:
            return pickle.load(f)