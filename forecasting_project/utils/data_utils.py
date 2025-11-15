from pathlib import Path
import pandas as pd
from loguru import logger
from pyspark.sql import SparkSession, DataFrame
import numpy as np

def list_parquet_files(path: Path, filename_stem: str) -> list[Path]:
    """
    List all parquet files in the specified directory.
    """
    all_files = list(path.glob("*.parquet"))
    filtered_files = [file for file in all_files if filename_stem in file.name]
    logger.info(f"Found {len(filtered_files)} files matching '{filename_stem}' in {path}")
    return filtered_files


def get_spark_session() -> SparkSession:
    """
    Initializes and returns a Spark session.
    """
    spark = (
        SparkSession.builder.appName("ParquetReader")
        .master("local[*]")  # Use all available local cores
        .config("spark.driver.memory", "4g") # Adjust memory as needed
        .getOrCreate()
    )
    logger.info("Spark session created.")
    return spark

def read_parquet_spark(spark: SparkSession, files: list[Path]) -> DataFrame:
    """
    Reads a list of Parquet files into a Spark DataFrame.
    """
    if not files:
        logger.warning("No files to read.")
        return spark.createDataFrame([], schema=None) # Return empty DataFrame
    
    file_paths = [str(f) for f in files]
    df = spark.read.parquet(*file_paths)
    logger.info(f"Read {len(files)} files into a Spark DataFrame.")
    return df

if __name__ == "__main__":

    # Initialize Spark session
    spark = get_spark_session()

    # Variables
    ID_PDV = ['0027', '0028', '0035', '0050', '0053']
    FILEPATH_FORECAST = "/home/danilo/python/artificial_dynamics/modeling/data/data/farmaciasp/processed/forecast/"
    FILEPATH_SELLOUT = "/home/danilo/python/artificial_dynamics/modeling/data/data/farmaciasp/processed/sellout/"
    FILENAME_FORECAST = "Forecast_Bebés"

    # Read sellout files
    # Folder path where the sellou parquet files are stored
    # List all parquet files in the directory that match the filename stem
    files_sellout = list_parquet_files(Path(FILEPATH_SELLOUT), "SP_SellOut")
    df_sellout = read_parquet_spark(spark, files_sellout)

    # Read forecast files
    # List all parquet files in the directory that match the filename stem
    files_forecast = list_parquet_files(Path(FILEPATH_FORECAST), "Forecast_Bebés")
    # Read the forecast files into a Spark DataFrame
    df_forecast = read_parquet_spark(spark, files_forecast)

    # Ensure 'anio_semana' is cast to int
    df_sellout = df_sellout.withColumn("anio_semana", df_sellout["anio_semana"].cast("int"))
    df_forecast = df_forecast.withColumn("anio_semana", df_forecast["anio_semana"].cast("int"))
    
    # Just id_sku and id_pdv that have a forecast
    logger.info("Filtering IDs with forecast...")
    df_ids_forecast = df_forecast.select(['id_sku', 'id_pdv']).distinct()

    # Select selltout data for the ids that have a forecast
    logger.info("Filtering sellout data for IDs with forecast...")
    df_sellout_filtered = df_sellout.join(df_ids_forecast, on=["id_sku", "id_pdv"], how="inner")

    # Filter sellout data for specific pdv IDs
    df_sellout_filtered = df_sellout_filtered.filter(df_sellout_filtered["id_pdv"].isin(ID_PDV))

    # Save the filtered DataFrame to a Parquet file
    logger.info("Saving filtered sellout data to Parquet file...")
    df_sellout_filtered.coalesce(1).write.format("parquet").mode("append").parquet('df_sellout_filtered')
    logger.info("Filtered sellout data saved successfully.")


    # # Dados do Catálogo
    # filename_catalog = 'data/data/farmaciasp/processed/catalogo/catalogo_sku.parquet'
    # df_catalog = spark.read.parquet(filename_catalog)

    
    # df_sellout_joined = df_sellout.join(df_catalog, on='id_sku', how='left')
    # df_sellout_filtered = df_sellout_joined.filter((df_sellout_joined['linea'] == 'Bebés') & (df_sellout_joined['anio_semana'] >= 202401) & (df_sellout_joined['anio_semana'] <= 202412))

    # Conferencia para validar com AD
    # df_sum = df_sellout_filtered.groupBy('id_pdv').agg({'cantidad_vendida': 'sum'}).withColumnRenamed('sum(cantidad_vendida)', 'cantidad_vendida')
    # df_sum = df_sum.sort('cantidad_vendida', ascending=False)
    
    # Stop the Spark session
    spark.stop()