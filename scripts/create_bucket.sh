#!/bin/sh
# Exit immediately if a command exits with a non-zero status.
set -e

# Function to check if MinIO is ready
minio_ready() {
  mc alias set myminio http://minio:9000 minio minio123 >/dev/null 2>&1
}

# Wait for MinIO to be ready
echo "Waiting for MinIO to be ready..."
until minio_ready; do
  echo "MinIO is unavailable - sleeping"
  sleep 1
done
echo "MinIO is ready."

# Create the bucket if it doesn't exist
BUCKET_NAME="mlflow"
echo "Checking for bucket: $BUCKET_NAME"

# Check if the bucket exists by attempting to list it.
# If `mc ls myminio/$BUCKET_NAME` exits with 0, the bucket exists.
# If it exits with non-zero, it means the bucket does not exist.
if mc ls myminio/"$BUCKET_NAME" >/dev/null 2>&1; then
  echo "Bucket '$BUCKET_NAME' already exists."
else
  echo "Bucket '$BUCKET_NAME' does not exist. Creating now..."
  mc mb myminio/"$BUCKET_NAME" # Create the bucket
  echo "Bucket '$BUCKET_NAME' created successfully."
  mc policy set public myminio/"$BUCKET_NAME"
  mc anonymous set download myminio/"$BUCKET_NAME"
  mc anonymous set upload myminio/"$BUCKET_NAME"
  echo "Bucket '$BUCKET_NAME' policy set to PUBLIC."
fi

# Create the datasets bucket if it doesn't exist
DATASETS_BUCKET_NAME="datasets"
echo "Checking for bucket: $DATASETS_BUCKET_NAME"

if mc ls myminio/"$DATASETS_BUCKET_NAME" >/dev/null 2>&1; then
  echo "Bucket '$DATASETS_BUCKET_NAME' already exists."
else
  echo "Bucket '$DATASETS_BUCKET_NAME' does not exist. Creating now..."
  mc mb myminio/"$DATASETS_BUCKET_NAME"
  echo "Bucket '$DATASETS_BUCKET_NAME' created successfully."
  mc policy set public myminio/"$DATASETS_BUCKET_NAME"
  mc anonymous set download myminio/"$DATASETS_BUCKET_NAME"
  mc anonymous set upload myminio/"$DATASETS_BUCKET_NAME"
  echo "Bucket '$DATASETS_BUCKET_NAME' policy set to PUBLIC."
fi

# # Upload the parquet file to the datasets bucket
# echo "Uploading parquet file to $DATASETS_BUCKET_NAME..."
# mc cp /app/experiments/df_sellout_filtered.parquet myminio/"$DATASETS_BUCKET_NAME"
# echo "Parquet file uploaded successfully."

exit 0
