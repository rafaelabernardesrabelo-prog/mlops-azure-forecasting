# Use the official MinIO Client image
FROM minio/mc

# Copy the bucket creation script into the container
COPY scripts/create_bucket.sh /create_bucket.sh

# Make the script executable
RUN chmod +x /create_bucket.sh

# Set the script as the entrypoint
ENTRYPOINT ["/create_bucket.sh"]
