import os
import cv2
import re
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import threading
from concurrent.futures import ThreadPoolExecutor
from torch.nn import functional as F
import yaml
import sys
import json
from pathlib import Path
from io import BytesIO
import tempfile

# Add project root to path to import cloud modules
project_root = str(Path(__file__).parent.parent.parent)
sys.path.append(project_root)

from api.cloud_storage_client import CloudStorageClient

base_directory = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

with open(os.path.join(base_directory, 'configs', 'default_config.yaml'), 'r') as file:
    config = yaml.safe_load(file)

''' ----- Loader file summary -----

> SatelliteImages - Torch dataset class to load images from specified directory as a batch, for training or augmentation.
    >> __init__ -> Initializes dataset with directory path, transformations, and caching.
        >>> Inputs: directory (str), transform (callable), num_threads (int), use_cloud (bool), cloud_prefix (str)
    >> _initialize_from_local -> Initialize dataset from local directory.
        >>> Inputs: None
    >> _initialize_from_cloud -> Initialize dataset from cloud storage.
        >>> Inputs: None
    >> _load_image -> Loads and returns an image from the given file path.
        >>> Inputs: path (str)
    >> __getdata__ -> Returns json style dictionary of image data for a given image ID.
        >>> Inputs: image_id (int)
    >> __getitem__ -> Returns the image, mask and image_id at a specified index position. For testing and visualisation purposes mainly.
        >>> Inputs: image_id (int)
    >> _process_mask -> Converts RGB mask to class indices tensor and resizes to model input size.
        >>> Inputs: mask (np.ndarray)
    >> __len__ -> Returns the number of images contained within the resulting dataset.
        >>> Inputs: None
    >> __del__ -> Clean up temporary directory when done.
        >>> Inputs: None

> ImagePreview - Visualisation function from the SatelliteImages dataset. For frontend and bug fixing use.
    >> __init__ -> Initialize with directory path.
        >>> Inputs: directory (str)
    >> preview -> Returns a plt preview of a specified image (by saved id) and corresponding mask.
        >>> Inputs: image_id (int), figsize (tuple)

    >>> Inputs: directory hard coded. Transformer not needed here.

'''

class SatelliteImages(Dataset):
    # Initializes dataset with directory path, transformations, and caching.
    def __init__(self, directory: str, transform: callable = None, num_threads: int = 4, 
                 use_cloud: bool = False, cloud_prefix: str = None) -> None:
        self.directory = directory
        self.transform = transform
        self.num_threads = num_threads
        self.use_cloud = use_cloud
        self.cloud_prefix = cloud_prefix or os.path.basename(directory)
        
        self.cloud_client = None
        if self.use_cloud:
            try:
                self.cloud_client = CloudStorageClient()
                self.temp_dir = tempfile.mkdtemp(prefix="satellite_images_")
            except Exception as e:
                print(f"Warning: Could not initialize cloud storage: {e}")
                self.use_cloud = False
        
        if self.use_cloud and self.cloud_client:
            self._initialize_from_cloud()
        else:
            self._initialize_from_local()
        
        self.class_to_rgb = {
            0: (0, 255, 255),     # urban - Cyan
            1: (255, 255, 0),     # agriculture - Yellow
            2: (255, 0, 255),     # rangeland - Magenta
            3: (0, 255, 0),       # forest - Green
            4: (0, 0, 255),       # water - Blue
            5: (255, 255, 255),   # barren - White
            6: (0, 0, 0)          # unknown - Black
        }
        self.rgb_to_class = {v: k for k, v in self.class_to_rgb.items()}
    
    def _initialize_from_local(self):
        image_files = sorted([f for f in os.listdir(self.directory) if f.endswith('_sat.jpg')])
        
        self.image_data = {
            "images": [
                {
                    "id": int(re.search(r'(\d+)_sat\.jpg$', f).group(1)),
                    "satellite_path": os.path.join(self.directory, f),
                    "mask_path": os.path.join(self.directory, f.replace('_sat.jpg', '_mask.png')),
                    "metadata": {
                        "resolution": None,
                        "coordinates": None
                    }
                }
                for f in image_files
            ]
        }
        
        self.image_id = [img["id"] for img in self.image_data["images"]]
    
    def _initialize_from_cloud(self):
        cloud_files = self.cloud_client.list_files(prefix=self.cloud_prefix)
        
        sat_files = [f for f in cloud_files if f.endswith('_sat.jpg')]
        
        self.image_data = {
            "images": [
                {
                    "id": int(re.search(r'(\d+)_sat\.jpg$', os.path.basename(f)).group(1)),
                    "satellite_path": f,
                    "mask_path": f.replace('_sat.jpg', '_mask.png'),
                    "cloud_path": True,
                    "local_path": None,
                    "metadata": {
                        "resolution": None,
                        "coordinates": None
                    }
                }
                for f in sat_files
            ]
        }
        
        self.image_id = [img["id"] for img in self.image_data["images"]]
        
        metadata_path = f"{self.cloud_prefix}/metadata.json"
        if metadata_path in cloud_files:
            local_metadata_path = os.path.join(self.temp_dir, "metadata.json")
            self.cloud_client.download_file(metadata_path, local_metadata_path)
            
            with open(local_metadata_path, 'r') as f:
                metadata = json.load(f)
                
            if "images" in metadata:
                for img in self.image_data["images"]:
                    img_id = img["id"]
                    img_filename = f"{img_id}_sat.jpg"
                    if img_filename in metadata["images"]:
                        img_metadata = metadata["images"][img_filename]
                        if "position" in img_metadata:
                            img["metadata"]["coordinates"] = img_metadata["position"]

    def _load_image(self, path: str) -> np.ndarray:
        if not path.startswith(self.cloud_prefix) and os.path.exists(path):
            img = cv2.imread(path)
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img is not None else None
        elif self.use_cloud and self.cloud_client:
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(path)[1]) as temp_file:
                    temp_path = temp_file.name
                self.cloud_client.download_file(path, temp_path)
                img = cv2.imread(temp_path)
                os.unlink(temp_path)
                return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img is not None else None
            
            except Exception as e:
                print(f"Error loading image from cloud: {e}")
                return None
        return None
    
    def __getdata__(self, image_id: int) -> dict:
        return next((img for img in self.image_data["images"] if img["id"] == image_id), None)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        image_id = self.image_id[idx]
        image_info = next((img for img in self.image_data["images"] if img["id"] == image_id), None)
        if image_info is None:
            raise KeyError(f"Image ID {image_id} not found in dataset")

        sat = self._load_image(image_info["satellite_path"])
        mask = self._load_image(image_info["mask_path"])

        if self.transform:
            sat = self.transform(sat)
        else:
            sat = torch.from_numpy(sat) if isinstance(sat, np.ndarray) else sat

        mask_tensor = self._process_mask(mask)
        
        return sat, mask_tensor, image_id
    
    # Converts RGB mask to class indices tensor and resizes to model input size
    def _process_mask(self, mask: np.ndarray) -> torch.Tensor:
        """
        Converts RGB mask to class indices tensor and resizes to model input size.
        
        Args:
            mask: RGB mask as numpy array
            
        Returns:
            Tensor with class indices
        """
        mask_tensor = torch.zeros((mask.shape[0], mask.shape[1]), dtype=torch.long)
        mask_torch = torch.from_numpy(mask)

        for rgb, class_idx in self.rgb_to_class.items():
            rgb_tensor = torch.tensor(rgb)
            mask_match = torch.all(mask_torch == rgb_tensor.view(1, 1, 3), dim=2)
            mask_tensor[mask_match] = class_idx

        mask_tensor = F.interpolate(
            mask_tensor.unsqueeze(0).unsqueeze(0).float(),
            size=config['model']['input_size'], 
            mode='nearest'
        ).squeeze().long()
        
        return mask_tensor

    def __len__(self) -> int:
        return len(self.image_data["images"])
    
    def __del__(self):
        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
            try:
                import shutil
                shutil.rmtree(self.temp_dir)
            except ImportError: pass


class ImagePreview:
    def __init__(self, directory: str = None):
        self.directory = directory or os.path.join(base_directory, 'data', 'train')
        self.dataset = SatelliteImages(self.directory)
    
    def preview(self, image_id: int, figsize: tuple = (10, 5)):
        idx = self.dataset.image_id.index(image_id)
        image, mask, _ = self.dataset[idx]
        
        if isinstance(image, torch.Tensor):
            image = image.permute(1, 2, 0).cpu().numpy()
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy()
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        axes[0].imshow(image)
        axes[0].set_title(f"Satellite Image (ID: {image_id})")
        axes[0].axis('off')
        
        axes[1].imshow(mask)
        axes[1].set_title("Mask")
        axes[1].axis('off')
        
        plt.tight_layout()
        return fig


if __name__ == '__main__':
    base_directory = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    print("Loading local dataset...")
    
    images = SatelliteImages(os.path.join(base_directory, 'data', 'train'))
    print(f"Local dataset contains {len(images)} images")

    print(images.__getdata__(855))
    print("\nLoading cloud dataset...")

    cloud_images = SatelliteImages(
        directory=os.path.join(base_directory, 'data', 'train'),
        use_cloud=True,
        cloud_prefix="data/train"
    )
    print(f"Cloud dataset contains {len(cloud_images)} images")
    
    if len(cloud_images) > 0:
        first_id = cloud_images.image_id[0]
        idx = cloud_images.image_id.index(first_id)
        print(f"Getting image with ID {first_id} from cloud...")
        image, mask, _ = cloud_images[idx]
        print(f"Successfully loaded image with shape {image.shape} and mask with shape {mask.shape}")