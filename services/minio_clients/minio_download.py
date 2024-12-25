import boto3
import os
import logging

logger = logging.getLogger(__name__)

# MinIO client configuration
minio_client = boto3.client(
    's3',
    endpoint_url='http://localhost:9000',  # MinIO endpoint
    aws_access_key_id='your-access-key',
    aws_secret_access_key='your-secret-key',
    region_name='us-east-1'
)

def download_image(bucket_name, object_name, download_path):
    try:
        # Download the image from MinIO
        with open(download_path, 'wb') as file_data:
            minio_client.download_fileobj(bucket_name, object_name, file_data)
        print(f"Image {object_name} downloaded successfully to {download_path}.")
    except Exception as e:
        print(f"Error downloading image: {e}")