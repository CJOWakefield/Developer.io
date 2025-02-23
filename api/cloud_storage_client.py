import os
import sys
from pathlib import Path

# Add the project root directory to Python path
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

from configs.config import CloudConfig
from google.cloud import storage
from google.cloud import logging as cloud_logging
from typing import Optional, List, Dict, Any, Union
from datetime import datetime
from dotenv import load_dotenv
import mimetypes

class Config:
    """Configuration class for managing environment variables and settings."""
    
    def __init__(self):
        """Initialize configuration with environment variables."""
        load_dotenv()
        
        # GCP Configuration
        self.project_id = os.getenv('GCP_PROJECT_ID')
        self.bucket_name = os.getenv('GCP_BUCKET_NAME')
        self.region = os.getenv('GCP_REGION')
        
        # Validate required settings
        if not all([self.project_id, self.bucket_name, self.region]):
            raise ValueError("Missing required GCP configuration in environment variables")

class CloudStorageClient:
    def __init__(self):
        CloudConfig.validate()
        self.config = Config()
        self.client = storage.Client.from_service_account_json(str(CloudConfig.SERVICE_ACCOUNT_PATH), project=CloudConfig.PROJECT_ID)
        self.bucket = self.client.bucket(CloudConfig.BUCKET_NAME)
        self.logger = cloud_logging.Client().logger('CloudStorageClient')

    def upload_file(self,
                    source_file: Union[str, Path],
                    destination_path: Optional[str] = None,
                    content_type: Optional[str] = None) -> str:

        source_path = Path(source_file)
        if not destination_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            destination_path = f"uploads/{timestamp}_{source_path.name}"

        # Create blob
        blob = self.bucket.blob(destination_path)
        if not content_type: content_type, _ = mimetypes.guess_type(str(source_path))
        if content_type: blob.content_type = content_type

        blob.upload_from_filename(str(source_path))

        return blob.public_url

    def upload_directory(self,
                         source_dir: Union[str, Path],
                         destination_path: Optional[str] = None) -> List[str]:
        
       uploaded_files = []
       base_path = Path(source_dir)

       for local_file in base_path.glob('**/*'):
           if local_file.is_file():
               relative_path = local_file.relative_to(base_path)
               gcs_path = f"{destination_path}/{relative_path}"
               self.upload_file(local_file, gcs_path)
               uploaded_files.append(gcs_path)

       return uploaded_files

    def download_file(self, blob_name: str, destination_path: Union[str, Path]) -> Path:
        blob = self.bucket.blob(blob_name)
        destination = Path(destination_path)
        blob.download_to_filename(str(destination))
        return destination

    def download_directory(self, blob_name: str, destination_path: Union[str, Path]) -> List[str]:
        downloaded_files = []
        blobs = self.bucket.list_blobs(prefix=blob_name)
        
        for blob in blobs:
            relative_path = blob.name.replace(blob_name, '', 1).lstrip('/')
            local_file_path = os.path.join(destination_path, relative_path)
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
            
            blob.download_to_filename(local_file_path)
            downloaded_files.append(local_file_path)
            
        return downloaded_files

    def delete_file(self, blob_name: str) -> None:
        blob = self.bucket.blob(blob_name)
        blob.delete()

    def list_files(self, prefix: Optional[str] = None, max_results: Optional[int] = None) -> List[str]:
        blobs = self.bucket.list_blobs(prefix=prefix, max_results=max_results)
        return [blob.name for blob in blobs]

if __name__ == '__main__':
    config = Config()
    cloud = CloudStorageClient()
    files = cloud.list_files()
    for file in files: print(file)