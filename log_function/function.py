import os
import time
from typing import Dict

import json
from dotenv import load_dotenv

from azure.storage.blob import BlobServiceClient, BlobClient

def load_data(generated_data: Dict):
    container_name = os.environ.get('BLOB_STORAGE_STORAGE_ACCOUNT', '')
    connect_str = os.environ.get('BLOB_STORAGE_CONNECTION_STRING', '')
    try:
        blob_service_client: BlobServiceClient = BlobServiceClient.from_connection_string(connect_str)
        blob_client: BlobClient = blob_service_client.get_blob_client(
            container=container_name,
            blob=f'log-{time.time()}-{generated_data.get("_id", None)}.json'
        )
        blob_client.upload_blob(json.dumps(generated_data))

    except Exception as ex:
        print('Exception: \n', ex)