import os
import boto3
from botocore.exceptions import NoCredentialsError
import dotenv
import logging

dotenv.load_dotenv(override=True)
URL_MINIO = str(os.getenv('URL_MINIO'))
ACCESS_KEY = str(os.getenv('ACCESS_KEY', None))
SECRET_KEY = str(os.getenv('SECRET_KEY', None))

# MinIO client configuration
minio_client = boto3.client(
    's3',
    endpoint_url=URL_MINIO,  # MinIO endpoint
    aws_access_key_id=ACCESS_KEY,
    aws_secret_access_key=SECRET_KEY,
    region_name='us-east-1',
)

logger = logging.getLogger(__name__)

class Minio:
    @staticmethod
    def upload_image(file_path, bucket_name, object_name):
        try:
            # Upload the image to MinIO
            with open(file_path, 'rb') as file_data:
                minio_client.upload_fileobj(file_data, bucket_name, object_name)
            logger.info(f"Image {object_name} uploaded successfully.")
        except NoCredentialsError:
            logger.info("Credentials not available.")
        except Exception as e:
            logger.info(f"Error uploading image: {e}")
    
    @staticmethod
    def download_image(bucket_name, object_name, download_path):
        try:
            # Download the image from MinIO
            with open(download_path, 'wb') as file_data:
                minio_client.download_fileobj(bucket_name, object_name, file_data)
            logger(f"Image {object_name} downloaded successfully to {download_path}.")
        except Exception as e:
            logger(f"Error downloading image: {e}")