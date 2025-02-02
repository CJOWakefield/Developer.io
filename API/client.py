from google.cloud import storage
from datetime import datetime
from pathlib import Path
from typing import Union, List, Optional
import mimetypes
from config import CloudConfig

class CloudStorageClient:
    def __init__(self):
        CloudConfig.validate()
        self.client = storage.Client.from_service_account_json(str(CloudConfig.SERVICE_ACCOUNT_PATH), project=CloudConfig.PROJECT_ID)
        self.bucket = self.client.bucket(CloudConfig.BUCKET_NAME)

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

    def download_file(self, blob_name: str, destination_path: Union[str, Path]) -> Path:
        blob = self.bucket.blob(blob_name)
        destination = Path(destination_path)
        blob.download_to_filename(str(destination))
        return destination

    def delete_file(self, blob_name: str) -> None:
        blob = self.bucket.blob(blob_name)
        blob.delete()

    def list_files(self, prefix: Optional[str] = None, max_results: Optional[int] = None) -> List[str]:
        blobs = self.bucket.list_blobs(prefix=prefix, max_results=max_results)
        return [blob.name for blob in blobs]

if __name__ == '__main__':
    cloud = CloudStorageClient()
    files = cloud.list_files()
    for file in files: print(file)
