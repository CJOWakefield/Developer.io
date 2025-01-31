from google.cloud import storage
from google.cloud.storage.blob import Blob
from datetime import datetime
from pathlib import Path
from typing import Union, List, Optional
import mimetypes
from config import CloudConfig

from ..config import PROJECT_ID, BUCKET_NAME, SERVICE_ACCOUNT_PATH

class CloudStorageClient:
    def __init__(self):
        # Validate configuration before initializing
        CloudConfig.validate()

        self.client = storage.Client.from_service_account_json(
            str(CloudConfig.SERVICE_ACCOUNT_PATH),
            project=CloudConfig.PROJECT_ID
        )
        self.bucket = self.client.bucket(CloudConfig.BUCKET_NAME)

    def upload_file(
        self,
        source_file: Union[str, Path],
        destination_path: Optional[str] = None,
        content_type: Optional[str] = None
    ) -> str:
        """
        Upload a file to Google Cloud Storage.

        Args:
            source_file: Path to the file to upload
            destination_path: Optional custom path in bucket. If None, generates one
            content_type: Optional content type. If None, tries to detect

        Returns:
            Public URL of the uploaded file
        """
        source_path = Path(source_file)

        # Generate destination path if not provided
        if not destination_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            destination_path = f"uploads/{timestamp}_{source_path.name}"

        # Create blob
        blob = self.bucket.blob(destination_path)

        # Detect content type if not provided
        if not content_type:
            content_type, _ = mimetypes.guess_type(str(source_path))
        if content_type:
            blob.content_type = content_type

        # Upload file
        blob.upload_from_filename(str(source_path))

        return blob.public_url

    def download_file(
        self,
        blob_name: str,
        destination_path: Union[str, Path]
    ) -> Path:
        """Download a file from Google Cloud Storage."""
        blob = self.bucket.blob(blob_name)
        destination = Path(destination_path)
        blob.download_to_filename(str(destination))
        return destination

    def delete_file(self, blob_name: str) -> None:
        """Delete a file from the bucket."""
        blob = self.bucket.blob(blob_name)
        blob.delete()

    def list_files(
        self,
        prefix: Optional[str] = None,
        max_results: Optional[int] = None
    ) -> List[str]:
        """List files in the bucket with optional prefix."""
        blobs = self.bucket.list_blobs(prefix=prefix, max_results=max_results)
        return [blob.name for blob in blobs]
