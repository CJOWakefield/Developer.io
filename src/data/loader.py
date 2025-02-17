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

# Load the configuration file
with open('configs/default_config.yaml', 'r') as file:
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

base_directory = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class SatelliteImages(Dataset):
    def __init__(self, directory, transform=None, num_threads=4, cache_size=100):
        self.directory = directory
        self.transform = transform
        self.num_threads = num_threads
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Mapping
        self.image_sat = sorted([f for f in os.listdir(directory) if f.endswith('_sat.jpg')])
        self.image_mask = sorted([f for f in os.listdir(directory) if f.endswith('_mask.png')])
        self.image_id = sorted([int(re.match(r'(\d+)', f).group(1)) for f in os.listdir(directory) if f.endswith('_sat.jpg')])
        self.rgb_to_class = {
            (0, 255, 255): 0, (255, 255, 0): 1, (255, 0, 255): 2,
            (0, 255, 0): 3, (0, 0, 255): 4, (255, 255, 255): 5, (0, 0, 0): 6
        }

        # Threading/caching
        self.cache = {}
        self.cache_size = cache_size
        self.cache_lock = threading.Lock()
        self.executor = ThreadPoolExecutor(max_workers=num_threads)

        # Pathing
        self.sat_paths = [os.path.join(directory, f) for f in self.image_sat]
        self.mask_paths = [os.path.join(directory, f) for f in self.image_mask]

    def _load_image(self, path):
        img = cv2.imread(path)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img is not None else None

    def _process_mask(self, mask):
        mask_tensor = torch.zeros((mask.shape[0], mask.shape[1]), dtype=torch.long, device=self.device)
        mask_torch = torch.from_numpy(mask).to(self.device)

        for rgb, class_idx in self.rgb_to_class.items():
            mask_tensor[torch.all(mask_torch == torch.tensor(rgb, device=self.device).view(1, 1, 3), dim=2)] = class_idx

        return F.interpolate(mask_tensor.unsqueeze(0).unsqueeze(0).float(),
                           size=config['model']['input_size'], mode='nearest').squeeze().long()

    def __getitem__(self, i):
        with self.cache_lock:
            if i in self.cache:
                return self.cache[i]

        sat = self._load_image(self.sat_paths[i])
        mask = self._load_image(self.mask_paths[i])

        if self.transform:
            sat = self.transform(sat)
            mask_tensor = self._process_mask(mask)
        else:
            mask_tensor = torch.from_numpy(mask).to(self.device)

        result = (torch.from_numpy(sat).to(self.device) if isinstance(sat, np.ndarray) else sat,
                 mask_tensor, self.image_id[i])

        with self.cache_lock:
            if len(self.cache) >= self.cache_size:
                self.cache.pop(next(iter(self.cache)))
            self.cache[i] = result

        return result

    def __len__(self):
        return len(self.image_id)

    def prefetch(self, indices=None):
        if indices is None:
            indices = range(len(self))
        futures = [self.executor.submit(self.__getitem__, i) for i in indices]
        return [f.result() for f in futures]

class ImagePreview:
    def __init__(self, directory, testing=False):
        self.directory = directory
        self.class_to_rgb = {
            0: (0, 255, 255), 1: (255, 255, 0), 2: (255, 0, 255),
            3: (0, 255, 0), 4: (0, 0, 255), 5: (255, 255, 255), 6: (0, 0, 0)
        }
        self.testing = testing

    def preview(self, img_id):
        images = SatelliteImages(self.directory)

        if self.testing: print(images.__len__())
        idx = images.image_id.index(img_id)
        sat, mask, _ = images[idx]

        # Move tensors to CPU for plotting if needed
        sat = sat.cpu().numpy() if torch.is_tensor(sat) else sat
        mask = mask.cpu().numpy() if torch.is_tensor(mask) else mask

        plt.subplot(1, 2, 1)
        plt.imshow(sat)
        plt.title('Satellite Image')

        plt.subplot(1, 2, 2)
        plt.imshow(mask)
        plt.title('Mask')
        plt.show()

if __name__ == '__main__':
    base_directory = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    images = ImagePreview(os.path.join(base_directory, 'data', 'train'), testing=False)
    images.preview(855)
