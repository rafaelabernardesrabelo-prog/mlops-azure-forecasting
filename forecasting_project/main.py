# -*- coding: utf-8 -*-
"""
Main orchestration script for the forecasting framework (MLflow integrated)
"""
import os
import sys  # <-- ADICIONE ESTA LINHA
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import yaml
import pandas as pd
from loguru import logger
from tqdm import tqdm
from sktime.split import ExpandingWindowSplitter
import mlflow

from utils.data_loader import load_and_prepare_data
from models import ThetaModel
from models.metrics import rmse, smape, wmape


# ===============================================
# 1️⃣ Função principal
# ===============================================
def main():
    """
    Main orchestration script for the forecasting framework.
    """
    # ----------------------------
    # 1.1 Carrega configuração YAML
    # ----------------------------
    logger.info("Loading configuration.")
    with open("config/config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # ----------------------------
    # 1.2 Configuração do MLflow
    # ----------------------------
    from azure.ai.ml import MLClient
    from azure.identity import DefaultAzureCredential
    import mlflow

    credential = DefaultAzureCredential()

    ml_client = MLClient(
        credential=credential,
        subscription_id=os.getenv("AZ_SUBSCRIPTION_ID"),
        resource_group_name=os.getenv("AZ_RESOURCE_GROUP"),
        workspace_name=os.getenv("AZ_ML_WORKSPACE"),
    )

    mlflow.set_tracking_uri(ml_client.workspaces.get(os.getenv("AZ_ML_WORKSPACE")).mlflow_tracking_uri)
    mlflow.set_experiment("forecasting-experiment")

    logger.info("📡 Connected to Azure ML MLflow tracking server.")

    # ----------------------------
    # 1.3 Carregamento dos dados
    # ----------------------------
    logger.info("Loading and preparing data.")
    data_dir = os.path.dirname(config["data"]["data_path"])
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    target_series_list = load_and_prepare_data(config["data"])

    if not target_series_list:
        logger.warning("No time series data found. Exiting.")
        return

    # ----------------------------
    # 1.4 Configuração dos modelos
    # ----------------------------
    models_to_run = {
        "theta": ThetaModel
    }

    ts = target_series_list[0]  # usando apenas a primeira série
    logger.info(f"Time Series ID: {ts.series_id}")

    is_seasonal = ts.is_seasonal
    period = ts.seasonality_period
    target_df = ts.data
    covariate_cols = ts.covariates_cols
    date_col = ts.date_column
    target_col = ts.target_column
    prediction_window = config["models"]["prediction_window"]

    # ----------------------------
    # 1.5 Split de treino e teste
    # ----------------------------
    train_start_date = pd.to_datetime(config["split"]["train"]["start"])
    train_end_date = pd.to_datetime(config["split"]["train"]["end"])
    test_start_date = pd.to_datetime(config["split"]["test"]["start"])
    test_end_date = pd.to_datetime(config["split"]["test"]["end"])

    df_train = target_df.loc[
        (target_df[date_col] >= train_start_date)
        & (target_df[date_col] <= train_end_date)
    ]

    initial_window = df_train.shape[0]
    splitter = ExpandingWindowSplitter(initial_window=initial_window, step_length=prediction_window)
    df_pred = pd.DataFrame()

    # ===============================================
    # 2️⃣ Loop principal de previsão e logging MLflow
    # ===============================================
    for _, (idxs_train, idxs_test) in tqdm(enumerate(splitter.split(target_df))):
        df_train = target_df.iloc[idxs_train].set_index(date_col)
        df_test = target_df.iloc[idxs_test].set_index(date_col)

        # Loop por modelo
        for model_name, model_class in models_to_run.items():
            logger.info(f"--- Running {model_name.replace('_', ' ').title()} Model ---")

            try:
                model_params = config["models"]["model_hyperparams"].get(model_name, {})

                # Ajustes automáticos de sazonalidade
                if model_name == "theta" and is_seasonal:
                    model_params["season_length"] = period

                elif model_name in ["seasonal_naive", "auto_arima"] and is_seasonal:
                    model_params["season_length"] = period
                    model_params["seasonal"] = True


                model_instance = model_class(model_params=model_params)

                # ----------------------------
                # 2.1 Inicia execução no MLflow
                # ----------------------------
                with mlflow.start_run(run_name=model_name):
                    mlflow.log_param("series_id", ts.series_id)
                    mlflow.log_param("model_name", model_name)
                    mlflow.log_params(model_params)

                    # ----------------------------
                    # 2.2 Treinamento
                    # ----------------------------
                    logger.info(f"Fitting {model_name} model.")
                    model_instance.fit(
                        df_train[target_col],
                        covariates=df_train[covariate_cols] if covariate_cols else None,
                    )

                    # ----------------------------
                    # 2.3 Previsão
                    # ----------------------------
                    logger.info("Making predictions.")
                    predictions_df = model_instance.predict(
                        df_test.index,
                        covariates=df_test[covariate_cols] if covariate_cols else None,
                    )

                    y_true = df_test[target_col].values
                    y_pred = predictions_df[target_col].values if target_col in predictions_df else predictions_df.values

                    # ----------------------------
                    # 2.4 Cálculo de métricas
                    # ----------------------------
                    metrics_dict = {
                        "rmse": rmse(y_true, y_pred),
                        "smape": smape(y_true, y_pred),
                        "wmape": wmape(y_true, y_pred),
                    }
                    mlflow.log_metrics(metrics_dict)

                    # ----------------------------
                    # 2.5 Salva previsões e modelo
                    # ----------------------------
                    model_save_dir = config["models"]["model_save_dir"]
                    if not os.path.exists(model_save_dir):
                        os.makedirs(model_save_dir)

                    pred_date_ini = df_test.index[0].date().strftime("%Y%m%d")
                    pred_date_end = df_test.index[-1].date().strftime("%Y%m%d")
                    model_path = os.path.join(
                        model_save_dir,
                        f"{model_name}_{ts.series_id}_{pred_date_ini}_{pred_date_end}.pkl",
                    )

                    model_instance.save_model(model_path)
                    mlflow.log_artifact(model_path)

                    # Salva previsões CSV
                    predictions_df["y_true"] = y_true
                    predictions_df["series_id"] = ts.series_id
                    predictions_df["model"] = model_name
                    predictions_csv = f"predictions_{model_name}_{ts.series_id}.csv"
                    predictions_df.to_csv(predictions_csv, index=False)
                    mlflow.log_artifact(predictions_csv)

                    df_pred = pd.concat([df_pred, predictions_df])

                    logger.info(f"✅ {model_name} logged successfully in MLflow.")

            except Exception as e:
                logger.error(f"❌ Error during {model_name} execution: {e}")

            logger.info(f"--- Finished {model_name.replace('_', ' ').title()} Model ---")

    logger.info("🎯 All models finished.")
    print("FIM")


# ===============================================
# 3️⃣ Execução direta
# ===============================================
if __name__ == "__main__":
    # Ajusta diretório de execução
    if os.path.basename(os.getcwd()) != "forecasting_project":
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "."))
        os.chdir(project_root)
        logger.info(f"Changed working directory to: {os.getcwd()}")

    main()
