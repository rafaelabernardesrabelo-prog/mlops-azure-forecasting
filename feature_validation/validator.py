from typing import Dict, List
import pandas as pd
from loguru import logger
from tqdm import tqdm
from .schema import FeatureSchema


class ValidationResult:
    def __init__(self) -> None:
        self.issues: Dict[str, List[str]] = {
            "missing_columns": [],
            "wrong_dtypes": [],
            "missing_dates": [],
            "missing_target": [],
            "periodicity": [],
        }

    def add(self, category: str, message: str) -> None:
        if category not in self.issues:
            self.issues[category] = []
        self.issues[category].append(message)

    def is_clean(self) -> bool:
        return all(len(v) == 0 for v in self.issues.values())

    def as_dict(self) -> Dict[str, List[str]]:
        return self.issues


class FeatureValidator:
    """Validador com métodos pequenos — fácil de estender (OCP) e testar (SRP)."""
    def __init__(self, schema: FeatureSchema):
        self.schema = schema

    def validate(self, df: pd.DataFrame) -> ValidationResult:
        result = ValidationResult()

        if not self._check_required_columns(df, result):
            # se faltam colunas essenciais, aborta etapas dependentes
            return result

        self._check_dtypes(df, result)
        self._check_target_nulls(df, result)
        self._ensure_date_dtype(df, result)
        self._check_missing_dates(df, result)
        self._check_periodicity(df, result)

        return result

    def _required_columns(self) -> set:
        cols = set(self.schema.features or [])
        if self.schema.target:
            cols.add(self.schema.target)
        if self.schema.date_column:
            cols.add(self.schema.date_column)
        if self.schema.key_columns:
            [cols.add(k) for k in self.schema.key_columns]
        logger.info('Required columns: {}', cols)
        return cols

    def _check_required_columns(self, df: pd.DataFrame, result: ValidationResult) -> bool:
        logger.info("Checking required columns...")
        expected = self._required_columns()
        missing = expected - set(df.columns)
        if missing:
            logger.error("missing_columns", f"missing: {', '.join(sorted(missing))}")
            result.add("missing_columns", f"missing: {', '.join(sorted(missing))}")
            return False
        return True

    def _check_dtypes(self, df: pd.DataFrame, result: ValidationResult) -> None:
        logger.info("Checking data types...")
        for col, expected in (self.schema.dtypes or {}).items():
            if col not in df.columns:
                continue
            actual = str(df[col].dtype)
            if expected in ("datetime", "date"):
                if not pd.api.types.is_datetime64_any_dtype(df[col]):
                    logger.error("wrong_dtypes", f"{col}: expected {expected}, got {actual}")
                    result.add("wrong_dtypes", f"{col}: expected {expected}, got {actual}")
            else:
                # comparação simples por substring para flexibilidade (e.g. "float" dentro de "float64")
                if expected not in actual:
                    logger.error("wrong_dtypes", f"{col}: expected {expected}, got {actual}")
                    result.add("wrong_dtypes", f"{col}: expected {expected}, got {actual}")

    def _check_target_nulls(self, df: pd.DataFrame, result: ValidationResult) -> None:
        logger.info("Checking target nulls...")
        t = self.schema.target
        if t and df[t].isnull().any():
            logger.error("missing_target", f"{t} has {int(df[t].isnull().sum())} null(s)")
            result.add("missing_target", f"{t} has {int(df[t].isnull().sum())} null(s)")

    def _ensure_date_dtype(self, df: pd.DataFrame, result: ValidationResult) -> None:
        logger.info("Ensuring date column dtype...")
        date_col = self.schema.date_column
        if not date_col:
            return
        if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
            try:
                df[date_col] = pd.to_datetime(df[date_col], errors="raise")
            except Exception as e:
                logger.error("missing_dates", f"cannot convert {date_col} to datetime: {e}")
                result.add("missing_dates", f"cannot convert {date_col} to datetime: {e}")

    def _check_missing_dates(self, df: pd.DataFrame, result: ValidationResult) -> None:
        date_col = self.schema.date_column
        if not date_col:
            return
        if df[date_col].isnull().any():
            result.add("missing_dates", f"{date_col} has {int(df[date_col].isnull().sum())} null(s)")

    def _check_periodicity(self, df: pd.DataFrame, result: ValidationResult) -> List[str]:
        logger.info("Checking periodicity (strict, gap-based)...")
        freq_str = (self.schema.frequency or "").lower()
        date_col = self.schema.date_column
        if not freq_str or not date_col:
            return []

        freq_map = {"daily": "D", "weekly": "W", "monthly": "M"}
        if freq_str not in freq_map:
            logger.warning(f"Unsupported frequency for periodicity check: {freq_str}")
            return []
        pd_freq = freq_map[freq_str]

        key_cols = self.schema.key_columns or []
        failed_groups: List[str] = []

        def fmt_group_name(key_cols: List[str], name) -> str:
            if not key_cols:
                return "GLOBAL"
            if not isinstance(name, tuple):
                name = (name,)
            parts = [f"{k}={v}" for k, v in zip(key_cols, name)]
            return ",".join(parts)

        if key_cols:
            group_iter = df.groupby(key_cols)
        else:
            group_iter = [(None, df)]

        for gname, gdf in tqdm(group_iter, desc='Checking groups for periodicity'):
            # Get unique, sorted, non-null dates for the group
            unique_dates = gdf[date_col].dropna().unique()
            if len(unique_dates) < 2:
                continue  # Not enough data to check for gaps

            sorted_dates = pd.Series(unique_dates).sort_values()
            periods = sorted_dates.dt.to_period(pd_freq)

            # Check for duplicate dates within the same period (e.g., two dates in the same week for 'weekly')
            if periods.duplicated().any():
                group_id = fmt_group_name(key_cols, gname)
                msg = f"{group_id}: found multiple dates within the same period for frequency '{freq_str}'"
                logger.error("periodicity: {}", msg)
                result.add("periodicity", msg)
                failed_groups.append(group_id)
                continue # No need to check for gaps if there are duplicates

            # Check for gaps between periods
            # periods is a Series of pandas Period objects; compute ordinal differences to avoid calling Period as a function
            period_ordinals = pd.Series([p.ordinal for p in periods])
            period_diffs = period_ordinals.diff().dropna()
            if not period_diffs.eq(1).all():
                group_id = fmt_group_name(key_cols, gname)
                gaps = int(period_diffs[period_diffs != 1].count())
                msg = f"{group_id}: found {gaps} gaps in the time series for frequency '{freq_str}'"
                logger.error("periodicity: {}", msg)
                result.add("periodicity", msg)
                failed_groups.append(group_id)

        return failed_groups
