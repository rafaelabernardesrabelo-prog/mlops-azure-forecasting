# -*- coding: utf-8 -*-
"""
Main orchestration script for the forecasting framework (MLflow integrated)
"""

############################################################
# 0) DEVE SER A PRIMEIRA COISA DO ARQUIVO
############################################################
import os
os.environ["MLFLOW_TRACKING_DISABLE_REGISTRY"] = "true"   # <-- ESSA FLAG PRECISA VIR ANTES DE TUDO

############################################################
# 1) IMPORTS
############################################################
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import yaml
import pandas as pd
from loguru import logger
from tqdm import tqdm
from sktime.split import ExpandingWindowSplitter
import mlflow   # <-- AGORA PODE

from utils.data_loader import load_and_prepare_data
from models import ThetaModel
from models.metrics import rmse, smape, wmape


############################################################
# 2) FUNÇÃO PRINCIPAL
############################################################
def main():

    logger.info("Loading configuration.")
    with open("config/config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    ############################################################
    # 2.1 Conectar no Azure ML
    ############################################################
    logger.info("🔌 Connecting to Azure ML workspace...")
    from azure.ai.ml import MLClient
    from azure.identity import DefaultAzureCredential

    credential = DefaultAzureCredential()

    ml_client = MLClient(
        credential=credential,
        subscription_id=os.getenv("AZ_SUBSCRIPTION_ID"),
        resource_group_name=os.getenv("AZ_RESOURCE_GROUP"),
        workspace_name=os.getenv("AZ_ML_WORKSPACE"),
    )

    workspace = ml_client.workspaces.get(os.getenv("AZ_ML_WORKSPACE"))
    tracking_uri = workspace.mlflow_tracking_uri

    ############################################################
    # 2.2 CONFIGURA MLflow Tracking
    ############################################################
    logger.info(f"📡 Tracking URI: {tracking_uri}")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("forecasting-experiment")

    ############################################################
    # 2.3 Carregar os dados
    ############################################################
    logger.info("📥 Loading and preparing data.")
    target_series_list = load_and_prepare_data(config["data"])

    if not target_series_list:
        logger.warning("⚠️ No time series found. Exiting.")
        return

    ts = target_series_list[0]
    logger.info(f"Using Time Series: {ts.series_id}")

    ############################################################
    # 2.4 Preparação do split
    ############################################################
    target_df = ts.data
    date_col = ts.date_column
    target_col = ts.target_column
    covariate_cols = ts.covariates_cols

    train_start = pd.to_datetime(config["split"]["train"]["start"])
    train_end   = pd.to_datetime(config["split"]["train"]["end"])
    df_train = target_df[(target_df[date_col] >= train_start) &
                         (target_df[date_col] <= train_end)]

    initial_window = df_train.shape[0]
    splitter = ExpandingWindowSplitter(
        initial_window=initial_window,
        step_length=config["models"]["prediction_window"]
    )

    ############################################################
    # 2.5 Modelos
    ############################################################
    models_to_run = {"theta": ThetaModel}

    ############################################################
    # 3) LOOP PRINCIPAL DE TREINO + LOGGING
    ############################################################
    for _, (idx_train, idx_test) in tqdm(enumerate(splitter.split(target_df))):

        df_train = target_df.iloc[idx_train].set_index(date_col)
        df_test  = target_df.iloc[idx_test].set_index(date_col)

        for model_name, model_class in models_to_run.items():

            logger.info(f"🚀 Running model: {model_name}")

            model_params = config["models"]["model_hyperparams"].get(model_name, {})
            if ts.is_seasonal:
                model_params["season_length"] = ts.seasonality_period

            model_instance = model_class(model_params=model_params)

            try:
                with mlflow.start_run(run_name=model_name):

                    mlflow.log_param("series_id", ts.series_id)
                    mlflow.log_param("model_name", model_name)
                    mlflow.log_params(model_params)

                    # ---- Train
                    logger.info("🧠 Fitting...")
                    model_instance.fit(
                        df_train[target_col],
                        covariates=df_train[covariate_cols] if covariate_cols else None,
                    )

                    # ---- Predict
                    logger.info("🔮 Predicting...")
                    pred_df = model_instance.predict(
                        df_test.index,
                        covariates=df_test[covariate_cols] if covariate_cols else None,
                    )

                    y_true = df_test[target_col].values
                    y_pred = pred_df[target_col].values

                    # ---- Metrics
                    metrics_dict = {
                        "rmse": rmse(y_true, y_pred),
                        "smape": smape(y_true, y_pred),
                        "wmape": wmape(y_true, y_pred),
                    }
                    mlflow.log_metrics(metrics_dict)

                    # ---- Save artifacts
                    os.makedirs(config["models"]["model_save_dir"], exist_ok=True)
                    model_path = f"{config['models']['model_save_dir']}/{model_name}_{ts.series_id}.pkl"
                    model_instance.save_model(model_path)
                    mlflow.log_artifact(model_path)

                    pred_df["y_true"] = y_true
                    pred_file = f"predictions_{model_name}_{ts.series_id}.csv"
                    pred_df.to_csv(pred_file, index=False)
                    mlflow.log_artifact(pred_file)

                    logger.info("✅ Logged!")

            except Exception as e:
                logger.error(f"❌ Error running {model_name}: {e}")

    logger.info("🎯 Finished all models.")
    print("FIM")


############################################################
# 4) EXECUÇÃO DIRETA
############################################################
if __name__ == "__main__":
    if os.path.basename(os.getcwd()) != "forecasting_project":
        new_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "."))
        os.chdir(new_dir)
        logger.info(f"Changed working directory to: {os.getcwd()}")

    main()
