import pandas as pd
from pathlib import Path
from loguru import logger
from tqdm import tqdm
import numpy as np
from joblib import Parallel, delayed
import shutil

from metrics import calculate_metrics, wmape
from sktime.performance_metrics.forecasting import MeanSquaredError, MeanAbsolutePercentageError

pd.set_option('display.max_columns', None)

smape = MeanAbsolutePercentageError(symmetric=True)
rmse = MeanSquaredError(square_root=True)

def load_all_parquet_from_folder(folder:Path):
    """Load all parquet files from a folder and concatenate them into a single DataFrame."""
    # Check folder
    if folder.exists() is False or folder.is_dir() is False:
        raise ValueError(f'Folder {folder} does not exist or is not a directory.')
    
    # List of all parquet files
    metrics_files = list(folder.glob('*.parquet'))
    logger.info(f'Found {len(metrics_files)} parquet files in folder {folder}.')
    return pd.concat([pd.read_parquet(f) for f in metrics_files], ignore_index=True)

def reference_check(df):
    """Check if doesn't have missing weeks"""

    # Sanity Check
    df_copy = df.copy()
    df_copy['semana'] = df_copy['anio_semana'] % 100
    df_copy['anio'] = df_copy['anio_semana'] // 100
    df_copy['date'] = pd.to_datetime(df_copy['anio'].astype(str) + df_copy['semana'].astype(str) + '0', format='%Y%W%w')
    df_sanity_check = df_copy.query("date <= '31.12.2024'").groupby(['id_pdv', 'id_sku']).agg(
        dt_min = ('date', 'min'), 
        dt_max = ('date', 'max'), 
        qtd_samples = ('date', 'nunique'),
        ).reset_index()
    df_sanity_check['qtd_check'] = (df_sanity_check['dt_max'] - df_sanity_check['dt_min']).dt.days // 7
    df_sanity_check['check'] = df_sanity_check['qtd_samples'] == (df_sanity_check['qtd_check'] + 1)
    
    if df_sanity_check['check'].sum() != df_sanity_check['check'].shape[0]:
        raise ValueError("Date sanity check failed: Some groups have missing weeks in the date range.")
    
def merge_with_reference(df_reference:pd.DataFrame, df_predictions:pd.DataFrame):
    # Merge with reference to ensure all dates are present
    df_reference_predictions = pd.DataFrame()
    for model_name in tqdm(df_predictions['model_name'].unique()):

        # Merge with reference to ensure all dates are present
        df_pred_model = df_reference.merge(
            df_predictions[df_predictions['model_name'] == model_name].drop(columns=['cantidad_vendida']),
            on=['id_pdv', 'id_sku', 'anio_semana'], 
            how='left'
            )
        
        # Fill eventualy missing values after the merge
        df_pred_model['model_name'] = model_name
        
        # Logging missing values
        if df_pred_model['prediction_int'].isna().any():
            qtd_na = df_pred_model['prediction_int'].isna().sum()
            logger.warning(f'Model {model_name} has missing {qtd_na} predictions.')

        df_reference_predictions = pd.concat([df_reference_predictions, df_pred_model])
    return df_reference_predictions

def prepare_data_for_metrics(df, df_reference, df_train_lim):
    """Prepare data for metrics calculation."""
    id_pdv = df['id_pdv']
    id_sku = df['id_sku']
    anio_semana = df['anio_semana']
    dt_min, dt_max = df_train_lim.query("(id_pdv == @id_pdv) & (id_sku == @id_sku)")[['dt_min', 'dt_max']].values[0]
    
    if df['keep_forecast'] is False:
        return {
            'id_pdv': id_pdv,
            'id_sku': id_sku,
            'anio_semana': anio_semana,
            'rmse': np.nan,
            'smape': np.nan,
            'wmape': np.nan,
            'rmsse': np.nan,
            'mase': np.nan
        }

    # Filter training data for this PDV/SKU combination
    train_mask = (
        (df_reference['anio_semana'].values >= dt_min) & 
        (df_reference['anio_semana'].values < dt_max) & 
        (df_reference['anio_semana'].values < anio_semana) & 
        (df_reference['id_pdv'].values == id_pdv) & 
        (df_reference['id_sku'].values == id_sku)
    )
    y_train = df_reference[train_mask]['cantidad_vendida'].to_numpy()

    y_true = df['cantidad_vendida']
    y_pred = df['prediction_int']

    # RMSSE and MASE require at least 2 training points for sp=1
    if len(y_train) < 2:
        y_train = None

    metrics = calculate_metrics(y_true, y_pred, y_train)
    metrics['id_pdv'] = id_pdv
    metrics['id_sku'] = id_sku
    metrics['anio_semana'] = anio_semana

    return metrics

def check_range_date(df:pd.DataFrame, dt_ini:int, dt_end:int, time_col:str) -> pd.DataFrame:
    logger.info(f'Filtering data between {dt_ini} and {dt_end} on column {time_col}.')
    df['between_range'] = df[time_col].between(dt_ini, dt_end)
    return df
    
def check_first_nonzero(df:pd.DataFrame, true_col:str, time_col:str, group_cols:list[str]) -> pd.DataFrame:
    """
    For each group, sets 'valid_date' to False for dates before the first non-zero value of `true_col`,
    and True for the rest.
    """
    logger.info(f'Checking first non-zero in column {true_col} for groups {group_cols}.')
    df_copy = df.copy()
    # Ensure sorting for cumsum to work correctly within groups
    df_copy = df_copy.sort_values(by=group_cols + [time_col])
        
    # Create a flag that is 1 if the sale is non-zero, 0 otherwise.
    # Then, for each group, calculate the cumulative sum.
    # The cumulative sum will be 0 for all rows until the first non-zero sale.
    is_selling = (df_copy.groupby(group_cols)[true_col].transform('cumsum') > 0)

    df_copy['first_nonzero'] = is_selling
    return df_copy
    
def check_consecutive_nonzero(df: pd.DataFrame, true_col: str, time_col: str, group_cols: list[str], n_intervals: int) -> pd.DataFrame:
    """
    For each group, check for 'n_intervals' consecutive periods with 0 value in 'true_col'.
    Marks 'is_active' as False if a row is part of a sequence of 'n_intervals' or more zeros,
    and True otherwise.
    """
    logger.info(f'Checking for {n_intervals} consecutive non-zero values in column {true_col} for groups {group_cols}.')
    if n_intervals <= 0:
        raise ValueError("n_intervals must be a positive integer.")

    if (df[true_col] < 0).any():
        logger.warning(f"A coluna '{true_col}' contém valores negativos. Isso pode distorcer o resultado da verificação de zeros consecutivos.")

    df_copy = df.copy()
    # Ensure data is sorted for the rolling window to be correct
    df_copy = df_copy.sort_values(by=group_cols + [time_col])

    # Calculate the sum of 'true_col' over a rolling window of size 'n_intervals' for each group.
    # If the sum is 0, it means all values in the window were 0.
    rolling_sum = df_copy.groupby(group_cols)[true_col].transform(
        lambda x: x.rolling(window=n_intervals, min_periods=1).sum()
    )

    # 'is_active' is True if the sum over the window is not zero.
    df_copy['consecutive_nonzero'] = (rolling_sum > 0)
        
    return df_copy
    
def keep_forecast(df):
    df['keep_forecast'] = df['first_nonzero'] & df['consecutive_nonzero'] & df['between_range']
    return df


if __name__ == '__main__':
    #TODO: Parametrizar os modelos do seletor
    #TODO: Fazer com pares de modelos

    # Parameters
    PATH = Path(__file__).parent.parent
    PREDICTIONS_FOLDER = PATH / 'data/farmaciasp/output/predictions'
    DF_REFERENCE = PATH / 'data/farmaciasp/output/df_resampled.parquet'
    METRICS_FOLDER = PATH / 'data/farmaciasp/output/metrics'
    # Load all parquet files from folder
    # df_predictions = load_all_parquet_from_folder(PREDICTIONS_FOLDER)
    
    # Load the reference file (filled with 0 for no data)
    df_reference = pd.read_parquet(DF_REFERENCE)
    reference_check(df_reference)
    
    # Check if product is in the catalog
    # Select period of analysis
    DT_INI = 202401
    DT_END = 202452
    TIME_COL = 'anio_semana'
    TRUE_COL = 'cantidad_vendida'
    GROUP_COLS = ['id_pdv', 'id_sku']
    

    df_reference = check_range_date(df_reference, DT_INI, DT_END, TIME_COL)
    df_reference_2024 = check_first_nonzero(df_reference.loc[df_reference['between_range']], TRUE_COL, TIME_COL, GROUP_COLS)
    df_reference_2024 = check_consecutive_nonzero(df_reference_2024, TRUE_COL, TIME_COL, GROUP_COLS, n_intervals=12)
    df_reference_2024 = keep_forecast(df_reference_2024)

    # Dataframe with the first and last date of sale for each id_pdv and id_sku
    # This will be used to filter the training data for each id_pdv and id_sku
    df_train_lim = df_reference.query("cantidad_vendida >0").groupby(['id_pdv', 'id_sku']).agg(
        dt_min=('anio_semana', 'min'), 
        dt_max=('anio_semana', 'max')
        ).reset_index()

    # # Para cada arquivo em PREDICTIONS_FOLDER calcule suas métricas
    # df_metrics = pd.DataFrame()
    # for parquet_file in PREDICTIONS_FOLDER.glob('*.parquet'):
    #     df_pred = pd.read_parquet(parquet_file)
    #     # Guarantee all dates are present
    #     df_2024_predictions = merge_with_reference(df_reference_2024, df_pred)

    #     model_name = df_2024_predictions['model_name'].iloc[0]
        
    #     # ls_metrics = []
    #     # for _, row in tqdm(df_2024_predictions.dropna(subset=['prediction_int']).iterrows(), 
    #     #                    total=df_2024_predictions.shape[0], 
    #     #                    desc=f'Preparing data for metrics {model_name}'):
    #     #     metrics = prepare_data_for_metrics(row, df_reference, df_train_lim)
    #     #     ls_metrics.append(metrics)

    #     # Apply smape calculation in parallel using joblib
    #     num_cores = -1  # Use all available cores
    #     df_iter = df_2024_predictions.dropna(subset=['prediction_int'])
    #     results = list(Parallel(n_jobs=num_cores, backend='multiprocessing')(delayed(prepare_data_for_metrics)(row, df_reference, df_train_lim) for _, row in tqdm(df_iter.iterrows(), total=len(df_iter), desc=f'Calculating metrics for {model_name}')))
    #     df_metrics = pd.DataFrame(results)
    #     df_2024_predictions = df_2024_predictions.merge(
    #         df_metrics,
    #         on=['id_pdv', 'id_sku', 'anio_semana'], 
    #         how='left'
    #         ) 

    #     # Exporta cálculo das métricas
    #     filename_model_metrics = METRICS_FOLDER / f'metrics_{model_name}.parquet'
    #     if METRICS_FOLDER.exists() is False:
    #         METRICS_FOLDER.mkdir(parents=True, exist_ok=True)
    #     df_2024_predictions.to_parquet(filename_model_metrics)

    #     # Move processed file to 'old' folder
    #     old_folder = PREDICTIONS_FOLDER / 'old'
    #     old_folder.mkdir(exist_ok=True)
    #     shutil.move(parquet_file, old_folder / parquet_file.name)
    #     logger.info(f"Moved {parquet_file.name} to {old_folder}")

    #     df_metrics = pd.concat([df_metrics, df_2024_predictions])

    # Para cada arquivo em METRICS_FOLDER carregue o arquivo.
    df_metrics = pd.DataFrame()
    for parquet_file in tqdm(METRICS_FOLDER.glob('*.parquet'), desc='Carregando arquivos de métricas:'):
        df_metric = pd.read_parquet(parquet_file)
        if df_metric.shape[0] != df_reference_2024.shape[0]:
            raise ValueError(f"Shape mismatch: {df_metric.shape[0]} != {df_reference_2024.shape[0]}")
        df_metrics = pd.concat([df_metrics, df_metric])

    metric_name = 'rmse'
    for metric_name in ['rmse', 'smape', 'rmsse', 'mase']:
        logger.info('Criando dataset de ranking de performance dos modelos por semana')
        df_metrics_pivot = df_metrics.pivot_table(
            index=['id_pdv', 'id_sku', 'anio_semana'],
            values=[metric_name],
            columns=['model_name']       
            )
        new_cols_name = ['_'.join(i) for i in df_metrics_pivot.columns]
        df_metrics_pivot.columns = new_cols_name

        # TODO: Retirar só para teste
        # df_metrics_pivot = df_metrics_pivot.iloc[:, :3]

        # =====================================================================================
        # 1. CALCULO DO EMPÍRICO
        # =====================================================================================
        # Rankeia os modelos por menor métrica (em caso de empate, pega o mais à esquerda)
        logger.info("Calculando o modelo empírico")
        df_empirico = df_metrics_pivot.rank(axis=1, method='first')
        df_empirico_best = df_empirico == 1
        # Identifica o nome do modelo vencedor
        df_empirico['model_name'] = df_empirico_best.idxmax(axis=1)
        # Corrige o nome do modelo
        df_empirico = df_empirico['model_name'].str.replace(f'{metric_name}_', '')
        # Agrega os dados de predições do modelo vencedor
        df_empirico = df_empirico.reset_index().merge(
            df_metrics[['anio_semana', 'id_pdv', 'id_sku', 'model_name', 'cantidad_vendida', 'prediction_int']],
            on=['anio_semana', 'id_pdv', 'id_sku', 'model_name'], 
            how='left'
        )
        # Calcula os WMAPEs do modelo empírico
        # TODO: Pensar se é uma boa estratégia zerar os nulos
        df_empirico = df_empirico.fillna(0)
        df_empirico.to_parquet(f'df_empirico_{metric_name}.parquet')
        wmape_min_global = wmape(df_empirico['cantidad_vendida'].to_numpy(), df_empirico['prediction_int'].to_numpy())
        wmape_min_weekly = df_empirico.groupby('anio_semana').apply(lambda x: wmape(x['cantidad_vendida'].to_numpy(), x['prediction_int'].to_numpy()), include_groups=False).reset_index(name='wmape_w')
        logger.info(f"WMAPE mínimo global usando {metric_name}: {wmape_min_global}")
        logger.info(f"WMAPE mínimo semanal usando {metric_name}: {wmape_min_weekly['wmape_w'].median():.4f}")

        # # =====================================================================================
        # # 2. SELETOR DE MODELOS
        # # =====================================================================================
        # logger.info("Calculando o seletor de modelos")
        # df_ranked = df_metrics_pivot.rank(axis=1, method='min')
        # # Somente os modelos ganhadores (em caso de empate, ambos se mantem como melhor)
        # df_ranked_best = df_ranked == 1

        # # Melhor modelo numa janela móvel de 10 dias
        # metric_cols = df_ranked_best.columns.tolist()
        # df_ranked_best = df_ranked_best.reset_index()

        # # Ordena os dados para garantir que a janela móvel e o shift funcionem corretamente
        # df_ranked_best = df_ranked_best.sort_values(['id_pdv', 'id_sku', 'anio_semana'])

        # # Agrupa, aplica o shift e a janela móvel, e encontra o melhor modelo
        # rolled = df_ranked_best.groupby(['id_pdv', 'id_sku'])[metric_cols].transform(
        #     lambda x: x.shift(1).rolling(window=10, min_periods=1).sum()
        #     # lambda x: x.shift(1).rolling(window=2, min_periods=1).sum()
        # )

        # df_best_model_in_window = df_ranked_best[['id_pdv', 'id_sku', 'anio_semana']].copy()
        # # As primeiras semanas são NA por causa do shift, então são preenchidas com 0
        # df_best_model_in_window['selected_model'] = rolled.fillna(0).idxmax(axis=1)    
        # #TODO: Arrumar depois essa questão do nome da coluna
        # df_best_model_in_window['selected_model'] = df_best_model_in_window['selected_model'].str.replace(f'{metric_name}_', '')

        # # Simulação dos resultados com o seletor de modelos
        # # Junta o df_metrics com a coluna de modelo selecionado
        # df_simulated = pd.merge(
        #     df_metrics, 
        #     df_best_model_in_window, 
        #     on=['id_pdv', 'id_sku', 'anio_semana'],
        #     how='left'
        # )

        # # Filtra para manter apenas as linhas onde o modelo é o selecionado
        # df_simulated = df_simulated.query('model_name == selected_model').copy()
        # df_simulated.to_parquet(f'df_simulated_{metric_name}.parquet')

        # # Agrupa por semana e calcula o wmape modelo selecionado
        # #TODO: Retirar n semanas que não tinham histórico para seleção
        # wmape_selecionado_weekly = df_simulated.dropna(subset=['prediction_int']).groupby(['anio_semana']).apply(
        #     lambda g: wmape(g['cantidad_vendida'].to_numpy(), g['prediction_int'].to_numpy()), include_groups=False
        # )
        # wmape_selecionado_global = wmape(df_simulated['cantidad_vendida'].to_numpy(), df_simulated['prediction_int'].to_numpy())
        # logger.info(f"WMAPE global do modelo selecionado usando {metric_name}: {wmape_selecionado_global:.4f}")
        # logger.info(f"WMAPE mediano do modelo selecionado usando {metric_name}: {wmape_selecionado_weekly.median():.4f}")
        # # df_selection_metrics.reset_index().rename(columns={0:'wmape'}).to_parquet(f'df_selection_metrics_{metric_name}.parquet')

