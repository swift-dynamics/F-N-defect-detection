import os
import boto3
from botocore.exceptions import NoCredentialsError
import dotenv
import logging

logger = logging.getLogger(__name__)

dotenv.load_dotenv(override=True)
URL_MINIO = str(os.getenv('URL_MINIO', "http://localhost:9000"))
ACCESS_KEY = str(os.getenv('ACCESS_KEY', None))
SECRET_KEY = str(os.getenv('SECRET_KEY', None))

# MinIO client configuration
minio_client = boto3.client(
    's3',
    endpoint_url=URL_MINIO,  # MinIO endpoint
    aws_access_key_id=ACCESS_KEY,
    aws_secret_access_key=SECRET_KEY,
    region_name='us-east-1'
)

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
