# -*- coding: utf-8 -*-
"""
MAIN – Versão 100% LOCAL (MLflow + MinIO na VM)
"""

import os
import sys
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


############################################################
# 1) MAIN
############################################################
def main():

    logger.info("Loading configuration.")
    with open("config/config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    ############################################################
    # 1.1 Configurar MLflow LOCAL
    ############################################################
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)

    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "local-forecasting-2")
    mlflow.set_experiment(experiment_name)

    logger.info(f"📡 MLflow Tracking URI: {tracking_uri}")
    logger.info(f"🔬 Experiment: {experiment_name}")

    ############################################################
    # 2) Carregar dados
    ############################################################
    logger.info("📥 Loading and preparing data.")
    ts_list = load_and_prepare_data(config["data"])

    if not ts_list:
        logger.error("Nenhuma série encontrada.")
        return

    ts = ts_list[0]  # primeira série
    df = ts.data

    ############################################################
    # 3) Criar splits
    ############################################################
    date_col = ts.date_column
    target_col = ts.target_column
    cov_cols = ts.covariates_cols

    split_cfg = config["split"]
    train_start = pd.to_datetime(split_cfg["train"]["start"])
    train_end   = pd.to_datetime(split_cfg["train"]["end"])

    df_train = df[(df[date_col] >= train_start) & (df[date_col] <= train_end)]

    initial_window = df_train.shape[0]
    prediction_window = config["models"]["prediction_window"]

    splitter = ExpandingWindowSplitter(
        initial_window=initial_window,
        step_length=prediction_window
    )

    ############################################################
    # 4) Configurar modelos
    ############################################################
    models_to_run = {
        "theta": ThetaModel
    }

    ############################################################
    # 5) LOOP DE TREINO E PREDIÇÃO
    ############################################################
    for split_id, (idx_train, idx_test) in enumerate(splitter.split(df)):

        df_train_split = df.iloc[idx_train].set_index(date_col)
        df_test_split  = df.iloc[idx_test].set_index(date_col)

        for model_name, model_class in models_to_run.items():

            logger.info(f"🚀 Running model: {model_name}")

            params = config["models"]["model_hyperparams"].get(model_name, {})
            if ts.is_seasonal:
                params["season_length"] = ts.seasonality_period

            model = model_class(model_params=params)

            try:
                with mlflow.start_run(run_name=f"{model_name}_split_{split_id}", nested=True):

                    # PARAMS
                    mlflow.log_param("series_id", ts.series_id)
                    mlflow.log_param("model", model_name)
                    mlflow.log_params(params)

                    # TRAIN
                    logger.info("🧠 Training...")
                    model.fit(
                        df_train_split[target_col],
                        covariates=df_train_split[cov_cols] if cov_cols else None
                    )

                    # PREDICT
                    logger.info("🔮 Predicting...")
                    pred_df = model.predict(
                        df_test_split.index,
                        covariates=df_test_split[cov_cols] if cov_cols else None
                    )
                    pred_df["y_true"] = df_test_split[target_col].values

                    # METRICS
                    metrics = {
                        "rmse": rmse(df_test_split[target_col], pred_df["y_pred"]),
                        "smape": smape(df_test_split[target_col], pred_df["y_pred"]),
                        "wmape": wmape(df_test_split[target_col], pred_df["y_pred"]),
                    }
                    mlflow.log_metrics(metrics)

                    # SAVE MODEL
                    save_dir = config["models"]["model_save_dir"]
                    os.makedirs(save_dir, exist_ok=True)

                    model_path = f"{save_dir}/{model_name}_{ts.series_id}_split{split_id}.pkl"
                    model.save_model(model_path)
                    mlflow.log_artifact(model_path)

                    # SAVE PREDICTIONS
                    pred_file = f"preds_{model_name}_{ts.series_id}_split{split_id}.csv"
                    pred_df.to_csv(pred_file, index=False)
                    mlflow.log_artifact(pred_file)

                    logger.info(f"✅ Run logged successfully.")

            except Exception as e:
                logger.error(f"❌ Error in {model_name}: {e}")

    logger.info("🎯 Finished all models!")


############################################################
# 2) ENTRYPOINT
############################################################
if __name__ == "__main__":
    main()
