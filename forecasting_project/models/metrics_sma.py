# generate_sma_cv_forecast.py
# ------------------------------------------------------------
# SMA com seleção de janela (order) via Rolling-Origin CV (estilo R OptimizeSMA)
# Gera forecast one-step-ahead por (id_pdv, id_sku) e métricas.
# ------------------------------------------------------------
import time
import numpy as np
import pandas as pd
import dask.dataframe as dd
from dask.distributed import Client, LocalCluster
from dask.diagnostics import ProgressBar

# ================== Parâmetros ==================
FILENAME_SELLOUT       = "df_sellout_filtered.parquet"          
FILENAME_FORECAST_OUT  = "df_sellout_per_week_with_sma_cv.parquet"
FILENAME_METRICS_OUT   = "metrics_sma_pdv_sku_cv.parquet"

KEY_COLUMNS      = ["anio_semana", "id_pdv", "id_sku"]
TRUE_COLUMN      = "cantidad_vendida"
FORECAST_COLUMN  = "sma"

# Lógica do R: OptimizeSMA
TEST_PERIOD = 1           # tamanho do bloco de teste em cada dobra
ITERATIONS  = 5           # número de dobras (origens rolantes)
STEP        = 4           # quanto cresce a janela de treino a cada dobra
MIN_ORDER   = 3           # ordem mínima (janela)
MAX_ORDER   = None        # se None, é definido de forma adaptativa como no R

# Forecast final
MIN_TRAIN_SIZE_FOR_OUTPUT = 12  

ERROR = "RMSE"            

# ================== Métricas ==================
def rmse(y, yhat):
    return float(np.sqrt(np.mean((y - yhat) ** 2))) if len(y) else np.nan

def mae(y, yhat):
    return float(np.mean(np.abs(y - yhat))) if len(y) else np.nan

def mse(y, yhat):
    return float(np.mean((y - yhat) ** 2)) if len(y) else np.nan

def mape(y, yhat):
    with np.errstate(divide="ignore", invalid="ignore"):
        perc = np.where(y == 0, np.nan, np.abs((y - yhat) / y))
    return float(np.nanmean(perc)) if len(y) else np.nan

def smape(y, yhat):
    denom = (np.abs(y) + np.abs(yhat))
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = np.where(denom == 0, 0.0, np.abs(y - yhat) / denom)
    return float(2.0 * np.mean(frac)) if len(y) else np.nan

def wmape(y, yhat):
    s = np.sum(np.abs(y))
    return float(np.sum(np.abs(y - yhat)) / s) if s != 0 else 0.0

def metric_value(name, y, yhat):
    name = name.upper()
    if name == "RMSE":
        return rmse(y, yhat)
    if name == "WMAPE":
        return wmape(y, yhat)
    if name == "MAE":
        return mae(y, yhat)
    if name == "MSE":
        return mse(y, yhat)
    if name == "MAPE":
        return mape(y, yhat)
    if name == "SMAPE":
        return smape(y, yhat)
    raise ValueError(f"Métrica não suportada: {name}")

# ================== Funções SMA ==================
def _adaptive_max_order(n, test_period, iterations, step, min_order):
    """
    Replica a lógica do R para definir MAX_ORDER adaptativo.
    """
    min_train_for_iter = test_period + (iterations * step)
    train_size_initial = n - min_train_for_iter
    if train_size_initial <= 0:
        # Série curta demais; devolve min_order para não quebrar
        return min_order

    if n > 104:
        max_order = min(int(round(train_size_initial * 0.80)), 84)
    else:
        max_order = int(round(train_size_initial * 0.80))

    if max_order < min_order:
        max_order = min_order
    return max_order

def _sma_predict_h(y_train, order, h):
    """
    Previsão multi-step SMA: média dos últimos 'order' pontos do treino,
    repetida h vezes (constante). Se treino < order, retorna NaN.
    """
    if len(y_train) < order:
        return np.full(h, np.nan, dtype=float)
    val = np.mean(y_train[-order:])
    return np.full(h, float(val), dtype=float)

def optimize_sma_numpy(y, test_period=TEST_PERIOD, min_order=MIN_ORDER, max_order=None,
                       error=ERROR, iterations=ITERATIONS, step=STEP):
    """
    Tradução do OptimizeSMA do R para NumPy:
      - define train_size_initial
      - avalia ordens min_order..max_order
      - em cada ordem, roda 'iterations' dobras com origem rolante e avalia a métrica
      - retorna (best_order, best_error)
    """
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n == 0:
        return min_order, np.inf

    if max_order is None:
        max_order = _adaptive_max_order(n, test_period, iterations, step, min_order)

    # tamanho mínimo de treino para a primeira dobra
    min_train_for_iter = test_period + (iterations * step)
    train_size_initial = n - min_train_for_iter

    if train_size_initial < min_order:
        # muito curto para CV; devolve min_order por falta de evidência
        return min_order, np.inf

    orders = range(min_order, max_order + 1)
    best_order = min_order
    best_err = np.inf

    for order in orders:
        if order > train_size_initial:
            # como no R, pula ordens maiores que o menor treino
            continue

        errs = []
        for i in range(1, iterations + 1):
            train_end = train_size_initial + (i - 1) * step
            y_train = y[:train_end]
            y_test = y[train_end:train_end + test_period]

            # se não houver teste completo, pula (como no R)
            if len(y_test) != test_period or len(y_train) < order:
                continue

            y_pred = _sma_predict_h(y_train, order, test_period)
            if np.all(np.isnan(y_pred)):
                continue

            err = metric_value(error, y_test, y_pred)
            if np.isfinite(err):
                errs.append(err)

        if len(errs) == 0:
            continue

        avg_err = float(np.mean(errs))
        if avg_err < best_err:
            best_err = avg_err
            best_order = order

    if not np.isfinite(best_err):
        best_err = np.inf

    return best_order, best_err

def compute_sma_cv_and_forecast(pdf: pd.DataFrame) -> pd.DataFrame:
    """
    Recebe um grupo (id_pdv, id_sku) e:
      1) escolhe o melhor order via rolling-origin CV (estilo R)
      2) gera forecast one-step-ahead (rolling mean shift(1)) com esse order
      3) aplica "queima" inicial (MIN_TRAIN_SIZE_FOR_OUTPUT) para manter compatível com seu output anterior
      4) retorna linhas do grupo + colunas 'sma' e 'best_order'
    """
    if pdf.empty:
        pdf[FORECAST_COLUMN] = np.nan
        pdf["best_order"] = np.nan
        return pdf

    pdf = pdf.sort_values("anio_semana").copy()
    y = pdf[TRUE_COLUMN].astype(float).values

    # 1) otimizar ordem
    best_order, _ = optimize_sma_numpy(
        y,
        test_period=TEST_PERIOD,
        min_order=MIN_ORDER,
        max_order=MAX_ORDER,
        error=ERROR,
        iterations=ITERATIONS,
        step=STEP,
    )

    # 2) forecast one-step-ahead da série completa com a ordem escolhida
    sma = pd.Series(y).rolling(window=best_order, min_periods=best_order).mean().shift(1)

    # 3) queima inicial opcional (coerente com seu pipeline anterior)
    k = min(MIN_TRAIN_SIZE_FOR_OUTPUT, len(sma))
    if k > 0:
        sma.iloc[:k] = np.nan

    pdf[FORECAST_COLUMN] = sma.values.astype(float)
    pdf["best_order"] = float(best_order)
    return pdf

# ================== Métricas finais (por série) ==================
def _rmsse(y, yhat, y_train, m=1):
    if y_train is None or len(y_train) <= m:
        return np.nan
    denom = np.mean((np.diff(y_train, n=m)) ** 2)
    if denom == 0 or not np.isfinite(denom):
        return np.nan
    return float(np.sqrt(np.mean((y - yhat) ** 2) / denom))

def _mase(y, yhat, y_train, m=1):
    if y_train is None or len(y_train) <= m:
        return np.nan
    denom = np.mean(np.abs(np.diff(y_train, n=m)))
    if denom == 0 or not np.isfinite(denom):
        return np.nan
    return float(np.mean(np.abs(y - yhat)) / denom)

def metrics_per_group(pdf: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula métricas por (id_pdv, id_sku) usando apenas linhas onde há previsão.
    y_train = linhas onde a previsão é NaN.
    """
    if pdf.empty:
        return pd.DataFrame([{
            "id_pdv": None, "id_sku": None, "best_order": np.nan,
            "rmse": np.nan, "smape": np.nan, "wmape": np.nan, "rmsse": np.nan, "mase": np.nan
        }])

    pdf = pdf.sort_values("anio_semana").copy()
    mask_pred = pdf[FORECAST_COLUMN].notna()

    y_true = pdf.loc[mask_pred, TRUE_COLUMN].astype(float).values
    y_pred = pdf.loc[mask_pred, FORECAST_COLUMN].astype(float).values

    y_train = pdf.loc[~mask_pred, TRUE_COLUMN].astype(float).values
    if len(y_train) < 2:
        y_train = None

    out = {
        "id_pdv": str(pdf["id_pdv"].iloc[0]),
        "id_sku": str(pdf["id_sku"].iloc[0]),
        "best_order": float(pdf["best_order"].iloc[0]) if "best_order" in pdf.columns else np.nan,
        "rmse": rmse(y_true, y_pred),
        "smape": smape(y_true, y_pred),
        "wmape": wmape(y_true, y_pred),
        "rmsse": _rmsse(y_true, y_pred, y_train, m=1),
        "mase": _mase(y_true, y_pred, y_train, m=1),
    }
    return pd.DataFrame([out])

# ================== Pipeline ==================
def main():
    start = time.time()
    cluster = LocalCluster(processes=False)  # threads only (bom no Windows)
    client = Client(cluster)
    ProgressBar().register()

    try:
        print("Lendo e agregando sellout por semana…")
        ddf = dd.read_parquet(
            FILENAME_SELLOUT,
            engine="pyarrow",
            columns=KEY_COLUMNS + [TRUE_COLUMN],
        )

        # Filtrar PDVs desejados
        PDVS_FILTRADOS = ["0027", "0028", "0035", "0050", "0053"]
        ddf["id_pdv"] = ddf["id_pdv"].astype(str)
        ddf = ddf[ddf["id_pdv"].isin(PDVS_FILTRADOS)]

        ddf = ddf.groupby(KEY_COLUMNS)[TRUE_COLUMN].sum().reset_index()
        ddf["id_sku"] = ddf["id_sku"].astype(str)

        # --------- Forecast com seleção de janela ---------
        print("Otimizando ordem (CV) e calculando SMA one-step-ahead…")
        meta_fc = pd.DataFrame({
            "anio_semana": pd.Series(dtype="int64"),
            "id_pdv": pd.Series(dtype="object"),
            "id_sku": pd.Series(dtype="object"),
            TRUE_COLUMN: pd.Series(dtype="float64"),
            FORECAST_COLUMN: pd.Series(dtype="float64"),
            "best_order": pd.Series(dtype="float64"),
        }).iloc[0:0]

        ddf_fcst = (
            ddf.groupby(["id_pdv", "id_sku"], group_keys=False)
               .apply(compute_sma_cv_and_forecast, meta=meta_fc)
        )

        print(f"Salvando forecast em: {FILENAME_FORECAST_OUT}")
        ddf_fcst[KEY_COLUMNS + [FORECAST_COLUMN, "best_order"]].to_parquet(
            FILENAME_FORECAST_OUT, engine="pyarrow", write_index=False
        )

        # --------- Métricas por série ---------
        print("Calculando métricas por (id_pdv, id_sku)…")
        meta_mt = pd.DataFrame({
            "id_pdv": pd.Series(dtype="object"),
            "id_sku": pd.Series(dtype="object"),
            "best_order": pd.Series(dtype="float64"),
            "rmse": pd.Series(dtype="float64"),
            "smape": pd.Series(dtype="float64"),
            "wmape": pd.Series(dtype="float64"),
            "rmsse": pd.Series(dtype="float64"),
            "mase": pd.Series(dtype="float64"),
        }).iloc[0:0]

        ddf_metrics = (
            ddf_fcst.groupby(["id_pdv", "id_sku"], group_keys=False)
                    .apply(metrics_per_group, meta=meta_mt)
        )

        print(f"Salvando métricas em: {FILENAME_METRICS_OUT}")
        ddf_metrics.to_parquet(FILENAME_METRICS_OUT, engine="pyarrow", write_index=False)

        print(f"Finalizado em {time.time()-start:.1f}s.")
    finally:
        client.close()
        cluster.close()

if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()
    main()
