# -*- coding: utf-8 -*-
"""
Main orchestration script for the forecasting framework (Azure ML + MLflow)
"""

############################################################
# 0) ESTA FLAG DEVE SER DEFINIDA ANTES DE IMPORTAR MLflow
############################################################
import os
os.environ["MLFLOW_TRACKING_DISABLE_REGISTRY"] = "true"

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
import mlflow   # <- Agora é seguro importar

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
    # 2.1 Conectar ao Azure ML
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

    logger.info("✔ Connected to Azure ML.")

    workspace = ml_client.workspaces.get(os.getenv("AZ_ML_WORKSPACE"))
    tracking_uri = workspace.mlflow_tracking_uri

    ############################################################
    # 2.2 CONFIGURAR MLflow Tracking (SEM MODEL REGISTRY)
    ############################################################
    logger.info(f"📡 Tracking URI: {tracking_uri}")

    mlflow.set_tracking_uri(tracking_uri)

    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", None)
    logger.info(f"🧪 Experiment (Azure Managed): {experiment_name}")

    # ⭐ IMPORTANTE ⭐
    # Nunca chamar mlflow.set_experiment() com azureml://
    # O experimento é definido automaticamente pelo Azure ML.

    ############################################################
    # 2.3 Carregar dados
    ############################################################
    logger.info("📥 Loading and preparing data.")
    target_series_list = load_and_prepare_data(config["data"])

    if not target_series_list:
        logger.warning("⚠️ No time series found. Exiting.")
        return

    ts = target_series_list[0]
    logger.info(f"Using Time Series: {ts.series_id}")

    ############################################################
    # 2.4 Preparar splits
    ############################################################
    target_df = ts.data
    date_col = ts.date_column
    target_col = ts.target_column
    covariate_cols = ts.covariates_cols

    split_cfg = config["split"]
    train_start = pd.to_datetime(split_cfg["train"]["start"])
    train_end   = pd.to_datetime(split_cfg["train"]["end"])

    df_train = target_df[
        (target_df[date_col] >= train_start) &
        (target_df[date_col] <= train_end)
    ]

    initial_window = df_train.shape[0]
    prediction_window = config["models"]["prediction_window"]

    splitter = ExpandingWindowSplitter(
        initial_window=initial_window,
        step_length=prediction_window
    )

    ############################################################
    # 2.5 Setup dos modelos
    ############################################################
    models_to_run = {"theta": ThetaModel}

    ############################################################
    # 3) LOOP PRINCIPAL
    ############################################################
    for _, (idx_train, idx_test) in tqdm(enumerate(splitter.split(target_df))):

        df_train_split = target_df.iloc[idx_train].set_index(date_col)
        df_test_split  = target_df.iloc[idx_test].set_index(date_col)

        for model_name, model_class in models_to_run.items():

            logger.info(f"🚀 Running model: {model_name}")

            model_params = config["models"]["model_hyperparams"].get(model_name, {})
            if ts.is_seasonal:
                model_params["season_length"] = ts.seasonality_period

            model_instance = model_class(model_params=model_params)

            try:
                # ⭐ Sem set_experiment() — Azure ML cuida disso
                with mlflow.start_run(run_name=model_name, nested=True):

                    mlflow.log_param("series_id", ts.series_id)
                    mlflow.log_param("model_name", model_name)
                    mlflow.log_params(model_params)

                    # ---- Train
                    logger.info("🧠 Fitting model...")
                    model_instance.fit(
                        df_train_split[target_col],
                        covariates=df_train_split[covariate_cols] if covariate_cols else None,
                    )

                    # ---- Predict
                    logger.info("🔮 Predicting...")
                    pred_df = model_instance.predict(
                        df_test_split.index,
                        covariates=df_test_split[covariate_cols] if covariate_cols else None,
                    )

                    # Ajuste obrigatório
                    pred_df["y_true"] = df_test_split[target_col].values

                    # ---- Métricas
                    metrics_dict = {
                        "rmse": rmse(df_test_split[target_col].values, pred_df["y_pred"].values),
                        "smape": smape(df_test_split[target_col].values, pred_df["y_pred"].values),
                        "wmape": wmape(df_test_split[target_col].values, pred_df["y_pred"].values),
                    }
                    mlflow.log_metrics(metrics_dict)

                    # ---- Save model
                    save_dir = config["models"]["model_save_dir"]
                    os.makedirs(save_dir, exist_ok=True)

                    model_path = f"{save_dir}/{model_name}_{ts.series_id}.pkl"
                    model_instance.save_model(model_path)
                    mlflow.log_artifact(model_path)

                    # ---- Save predictions
                    pred_file = f"predictions_{model_name}_{ts.series_id}.csv"
                    pred_df.to_csv(pred_file, index=False)
                    mlflow.log_artifact(pred_file)

                    logger.info("✅ Logged successfully!")

            except Exception as e:
                logger.error(f"❌ Error running {model_name}: {e}")

    logger.info("🎯 Finished all models.")
    print("FIM")


############################################################
# 4) ENTRYPOINT
############################################################
if __name__ == "__main__":
    if os.path.basename(os.getcwd()) != "forecasting_project":
        new_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "."))
        os.chdir(new_dir)
        logger.info(f"Changed working directory to: {os.getcwd()}")

    main()
