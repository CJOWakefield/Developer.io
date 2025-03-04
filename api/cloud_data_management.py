import os
from pathlib import Path
import sys
import time
import datetime
from api.cloud_storage_client import CloudStorageClient
from typing import List, Set

project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

class FileManager:
    def __init__(self):
        """
        Initialize the FileManager with a cloud storage client.
        """
        self.cloud_client = CloudStorageClient()
        
    def get_local_files(self, directory: str) -> Set[str]:
        """
        Get a set of all files in a local directory, with paths relative to the directory.
        
        Args:
            directory: Path to the local directory
            
        Returns:
            Set of relative file paths
        """
        local_files = set()
        base_path = Path(directory)
        
        if base_path.exists():
            for file_path in base_path.rglob('*'):
                if file_path.is_file():
                    relative_path = str(file_path.relative_to(base_path))
                    local_files.add(relative_path)
                    
        return local_files
     
    def get_cloud_files(self, prefix: str) -> Set[str]:
        """
        Get a set of all files in cloud storage with the given prefix.
        
        Args:
            prefix: Cloud storage prefix to list files under
            
        Returns:
            Set of cloud file paths
        """
        return set(self.cloud_client.list_files(prefix=prefix))
    
    def sync_directory(self, local_dir: str, cloud_prefix: str) -> List[str]:
        """
        Synchronize a local directory with cloud storage.
        
        Args:
            local_dir: Path to the local directory
            cloud_prefix: Cloud storage prefix to sync with
            
        Returns:
            List of successfully uploaded file paths
        """
        print(f"Syncing directory: {local_dir} to {cloud_prefix}")
        
        local_files = self.get_local_files(local_dir)
        cloud_files = self.get_cloud_files(cloud_prefix)
        
        files_to_upload = local_files - cloud_files
        total_files = len(files_to_upload)
        
        if not files_to_upload:
            print("All files are already synced!")
            return []
        
        print(f"Found {total_files} files to upload")
        
        try:
            uploaded_files = self.cloud_client.upload_directory(local_dir, cloud_prefix)
            print(f"Successfully uploaded {len(uploaded_files)} files")
            return uploaded_files
        except Exception as e:
            print(f"Error during upload: {str(e)}")
            return []

def sync_directories(directories: list[str]):
    """
    Synchronize multiple directories with cloud storage.
    
    Args:
        directories: List of directory paths to synchronize
    """
    file_manager = FileManager()
    sync_dirs = [(dir, dir) for dir in directories]

    upload_count = 0
    start_time = time.time()

    for local_dir, cloud_dir in sync_dirs:
        try:
            print(f'\nInitialising directory sync. {datetime.datetime.now()}')
            curr_uploads = file_manager.sync_directory(local_dir, cloud_dir)
            upload_count += len(curr_uploads)
            print(f'\nSync completed for {local_dir} to {cloud_dir}: {len(curr_uploads)} files uploaded.')
        except Exception as e:
            print(f'Syncing error for {local_dir}: {str(e)}')

    print(f'\nUpload complete. {time.time() - start_time:.2f} seconds elapsed.')


if __name__ == "__main__":
    sync_directories([("data/train", "data/train"),
                      ("data/test", "data/test"),
                      ("data/val", "data/val"),
                      ("models", "models")])