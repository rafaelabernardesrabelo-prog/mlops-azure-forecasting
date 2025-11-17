# -*- coding: utf-8 -*-
"""
Main orchestration script for the forecasting framework
Now using SELF-HOSTED MLflow (VM Azure + Docker + MinIO)
"""

############################################################
# 0) IMPORTS E CONFIGURAÇÃO MLflow REMOTO
############################################################
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import yaml
import pandas as pd
from loguru import logger
from tqdm import tqdm
from sktime.split import ExpandingWindowSplitter

import mlflow

# 🔥 CONFIG MLflow remoto (VM Azure)
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
mlflow.set_registry_uri(os.getenv("MLFLOW_TRACKING_URI"))

# 🔥 Configurar MinIO (artefatos)
os.environ["MLFLOW_S3_ENDPOINT_URL"] = os.getenv("MLFLOW_S3_ENDPOINT_URL")
os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("AWS_ACCESS_KEY_ID")
os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("AWS_SECRET_ACCESS_KEY")


############################################################
# 1) IMPORTS DO PROJETO
############################################################
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
    # 2.1 Configurar Experimento (AGORA VOCÊ PODE USAR SET_EXPERIMENT)
    ############################################################
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "forecasting")
    mlflow.set_experiment(experiment_name)
    logger.info(f"🧪 Using experiment: {experiment_name}")

    ############################################################
    # 2.2 Carregar dados
    ############################################################
    logger.info("📥 Loading and preparing data.")
    target_series_list = load_and_prepare_data(config["data"])

    if not target_series_list:
        logger.warning("⚠️ No time series found. Exiting.")
        return

    ts = target_series_list[0]
    logger.info(f"Using Time Series: {ts.series_id}")

    ############################################################
    # 2.3 Preparar splits
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
    # 2.4 Setup dos modelos
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
                # ⭐ Agora você pode startar runs normalmente
                with mlflow.start_run(run_name=f"{model_name}_{ts.series_id}", nested=False):

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
