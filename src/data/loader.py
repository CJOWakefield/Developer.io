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

base_directory = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

with open(os.path.join(base_directory, 'configs', 'default_config.yaml'), 'r') as file:
    config = yaml.safe_load(file)

''' ----- Loader file summary -----

> SatelliteImages - Torch dataset class to load images from specified directory as a batch, for training or augmentation.
    >> __getitem__ -> Returns the image, mask and image_id at a specified index position. For testing and visualisation purposes mainly.
    >> __len__ -> Returns the number of images cotained within the resulting dataset.

    >>> Inputs: base_directory hard coded in file. Transformation can be used but needs to be reimplemented later on.

> ImagePreview - Visualisation function from the SatelliteImages dataset. For frontend and bug fixing use.
    > Dependencies - SatelliteImages dataset function.

    >> preview -> Returns a plt preview of a specified image (by saved id) and corresponding mask.

    >>> Inputs: directory hard coded. Transformer not needed here.

'''

class SatelliteImages(Dataset):
    # Initializes dataset with directory path, transformations, and caching.
    def __init__(self, directory: str, transform: callable = None, num_threads: int = 4, cache_size: int = 100) -> None:
        self.directory = directory
        self.transform = transform
        self.num_threads = num_threads
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        image_files = sorted([f for f in os.listdir(directory) if f.endswith('_sat.jpg')])
        
        self.image_data = {
            "images": [
                {
                    "id": int(re.search(r'(\d+)_sat\.jpg$', f).group(1)),
                    "satellite_path": os.path.join(directory, f),
                    "mask_path": os.path.join(directory, f.replace('_sat.jpg', '_mask.png')),
                    "metadata": {
                        "date_captured": None,
                        "resolution": None,
                        "coordinates": None
                    }
                }
                for f in image_files
            ]
        }
        
        self.image_id = [img["id"] for img in self.image_data["images"]]
        self.rgb_to_class = {
            0: (0, 255, 255), 
            1: (255, 255, 0), 
            2: (255, 0, 255),
            3: (0, 255, 0), 
            4: (0, 0, 255), 
            5: (255, 255, 255), 
            6: (0, 0, 0)
            }

        self.cache = {}
        self.cache_size = cache_size
        self.cache_lock = threading.Lock()
        self.executor = ThreadPoolExecutor(max_workers=num_threads)

    # Loads and returns an image from the given file path
    def _load_image(self, path: str) -> np.ndarray:
        img = cv2.imread(path)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img is not None else None
    
    # Returns json style dictionary of image data for a given image ID
    def __getdata__(self, image_id: int) -> dict:
        return next((img for img in self.image_data["images"] if img["id"] == image_id), None)

    # Returns a tuple of (image, mask, image_id) for the given image ID
    def __getitem__(self, image_id: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        with self.cache_lock:
            if image_id in self.cache:
                return self.cache[image_id]

        # Find the image info by ID
        image_info = next((img for img in self.image_data["images"] if img["id"] == image_id), None)
        if image_info is None:
            raise KeyError(f"Image ID {image_id} not found in dataset")

        sat = self._load_image(image_info["satellite_path"])
        mask = self._load_image(image_info["mask_path"])

        if self.transform:
            sat = self.transform(sat)
            mask_tensor = self._process_mask(mask)
        else:
            mask_tensor = torch.from_numpy(mask).to(self.device)

        result = (torch.from_numpy(sat).to(self.device) if isinstance(sat, np.ndarray) else sat,
                 mask_tensor, image_id)

        with self.cache_lock:
            if len(self.cache) >= self.cache_size:
                self.cache.pop(next(iter(self.cache)))
            self.cache[image_id] = result

        return result
    
    # Converts RGB mask to class indices tensor and resizes to model input size
    def _process_mask(self, mask: np.ndarray) -> torch.Tensor:
        mask_tensor = torch.zeros((mask.shape[0], mask.shape[1]), dtype=torch.long, device=self.device)
        mask_torch = torch.from_numpy(mask).to(self.device)

        for rgb, class_idx in self.rgb_to_class.items():
            mask_tensor[torch.all(mask_torch == torch.tensor(rgb, device=self.device).view(1, 1, 3), dim=2)] = class_idx

        return F.interpolate(mask_tensor.unsqueeze(0).unsqueeze(0).float(),
                           size=config['model']['input_size'], mode='nearest').squeeze().long()

    # Returns the total number of images in the dataset
    def __len__(self) -> int:
        return len(self.image_data["images"])

    # Pre-loads specified images into cache using multiple threads
    def prefetch(self, indices: list[int] = None) -> list[tuple[torch.Tensor, torch.Tensor, int]]:
        if indices is None:
            indices = range(len(self))
        futures = [self.executor.submit(self.__getitem__, i) for i in indices]
        return [f.result() for f in futures]

class ImagePreview:
    # Initializes preview class with directory path and testing flag
    def __init__(self, directory: str, testing: bool = False) -> None:
        self.directory = directory
        self.testing = testing
        self.class_to_rgb = {
            0: (0, 255, 255), 
            1: (255, 255, 0), 
            2: (255, 0, 255),
            3: (0, 255, 0), 
            4: (0, 0, 255), 
            5: (255, 255, 255), 
            6: (0, 0, 0)
            }

    # Displays satellite image and optionally its mask using matplotlib
    def preview(self, img_id: int, show_mask: bool = False) -> None:
        images = SatelliteImages(self.directory)

        if self.testing: print(images.__len__())
        idx = images.image_id.index(img_id)
        sat, mask, _ = images[idx]

        sat = sat.cpu().numpy() if torch.is_tensor(sat) else sat
        mask = mask.cpu().numpy() if torch.is_tensor(mask) else mask

        if show_mask:
            plt.subplot(1, 2, 1)
            plt.imshow(sat)
            plt.subplot(1, 2, 2)
            plt.imshow(mask)
            plt.show()
        else:
            plt.imshow(sat)
            plt.title('Satellite Image')
            plt.show()

if __name__ == '__main__':
    base_directory = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    images = SatelliteImages(os.path.join(base_directory, 'data', 'train'))
    # print(images.__getitem__(855))
    print(images.__getdata__(855))
    print(images.__len__())
