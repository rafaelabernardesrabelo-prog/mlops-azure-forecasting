from dataclasses import dataclass, field
from .seasonality import analyze_seasonality
from typing import Optional, List
import pandas as pd
from functools import cached_property

@dataclass
class TimeSeriesData:
    """Encapsula uma série temporal e seus metadados."""
    series_id: str  # Um identificador único para a série
    data: pd.DataFrame
    date_column: str
    target_column: str
    
    # Metadados que podem ser preenchidos por funções de análise
    freq: Optional[str] = None
    covariates_cols: Optional[List[str]] = field(default_factory=list)

    def __post_init__(self):
        self._validate_columns()
        self.min_date = self.data[self.date_column].min()
        self.max_date = self.data[self.date_column].max()
    
    def _validate_columns(self):
        """Verifica se as colunas especificadas existem no DataFrame."""
        required_cols = {self.date_column, self.target_column}
        if self.covariates_cols:
            required_cols.update(self.covariates_cols)
        
        missing_cols = required_cols - set(self.data.columns)
        if missing_cols:
            raise ValueError(f"Colunas não encontradas nos dados: {sorted(list(missing_cols))}")

    @cached_property
    def is_seasonal(self) -> bool:
        """Verifica se a série é sazonal. O resultado é cacheado."""
        is_seasonal, _ = analyze_seasonality(self.data, self.target_column)
        return is_seasonal

    @cached_property
    def seasonality_period(self) -> Optional[int]:
        """Retorna o período de sazonalidade, se houver. O resultado é cacheado."""
        _, seasonality_period = analyze_seasonality(self.data, self.target_column)
        return seasonality_period
    

@dataclass
class TimeSeriesDataset:
    """Encapsula um conjunto de dados de séries temporais."""
    time_series: List[TimeSeriesData] = field(default_factory=list)

    def __len__(self):
        return len(self.time_series)

    def __getitem__(self, idx):
        return self.time_series[idx]