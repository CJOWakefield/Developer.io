import os
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torchvision.utils import make_grid
from loader import *

## To-do notes: Add variable file for hardcoded pathing of relative directories. If needed, adjust manipulability of image split to further increase training set.

''' ----- Augment file summary -----

> SatelliteAugmentation - Class to augment a specific image by separating into four quarters, resizing and then rotating to enhance training data size.
    >> get_quarter -> Converts image pixels into four quarters
    >> save_pair -> Saves resulting augmentations for both the original satellite image and the corresponding mask.
    >> augment -> Executes the 'get_quarter' function, rotating pixel matrix to generate new augmented images from quartered original images.

    >>> Inputs: save_directory hard coded in file. Can be adjusted but made using OS for robust file navigation.


> AugmentedSatelliteImages - Class to load dataset images and output resulting augmentations. Designed for execution on one image alone.
    > Dependencies - SatelliteImages dataset function, from .loader.py

    >> __len__ -> Simply returns the number of images contained within the dataset.
    >> save_augmentations -> Execute augmentation class and save image to specified directory (in this case, save_directory below)
    >> visualise_augmentations -> Simple helper function for visualisation, plan to implement in resulting front-end for fun but useful for bug fixing from tensor manipulation.

    >>> Inputs: directory hard coded. Labels not needed currently and transformer needs to be implemented in conjunction with the model file.

'''

base_directory = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
save_directory = os.path.join(base_directory, 'data', 'processed')

class SatelliteAugmentation:
    def __init__(self, target_size=(2448, 2448)):
        self.target_size = target_size
        self.rotation_angles = [90, 180, 270]
        self.visualisation_size = (512, 512)
        os.makedirs(save_directory, exist_ok=True)

    def get_quarter(self, image, quarter_idx):
        h, w = image.shape[-2:] if torch.is_tensor(image) else image.shape[:2]
        h_half, w_half = h // 2, w // 2
        quarters = [(slice(None, h_half), slice(None, w_half)),
                   (slice(None, h_half), slice(w_half, None)),
                   (slice(h_half, None), slice(None, w_half)),
                   (slice(h_half, None), slice(w_half, None))]
        return quarters[quarter_idx]

    def save_pair(self, image, mask, prefix):
        cv2.imwrite(os.path.join(save_directory, f'{prefix}_sat.jpg'),
                   cv2.cvtColor(image if isinstance(image, np.ndarray) else image.numpy(), cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(save_directory, f'{prefix}_mask.png'),
                   cv2.cvtColor(mask if isinstance(mask, np.ndarray) else mask.numpy(), cv2.COLOR_RGB2BGR))

    def augment(self, image, mask, base_prefix):
        for i in range(4):
            quarter = self.get_quarter(image, i)
            q_img = image[quarter] if torch.is_tensor(image) else image[quarter[0], quarter[1]]
            q_mask = mask[quarter] if torch.is_tensor(mask) else mask[quarter[0], quarter[1]]

            self.save_pair(q_img, q_mask, f"{base_prefix}_q{i}")

            for angle in self.rotation_angles:
                M = cv2.getRotationMatrix2D((q_img.shape[1]/2, q_img.shape[0]/2), angle, 1)
                rot_img = cv2.warpAffine(q_img if isinstance(q_img, np.ndarray) else q_img.numpy(), M, q_img.shape[:2][::-1])
                rot_mask = cv2.warpAffine(q_mask if isinstance(q_mask, np.ndarray) else q_mask.numpy(), M, q_mask.shape[:2][::-1])
                self.save_pair(rot_img, rot_mask, f"{base_prefix}_q{i}_r{angle}")

class AugmentedSatelliteImages(Dataset):
    def __init__(self, directory, labels=None, transform=None):
        self.base_dataset = SatelliteImages(directory, labels, transform)
        self.augmentor = SatelliteAugmentation()

    def __len__(self): return len(self.base_dataset)

    def save_augmentations(self, index):
        sat, _, image_id = self.base_dataset[index]
        mask = cv2.cvtColor(cv2.imread(os.path.join(self.base_dataset.directory, self.base_dataset.image_mask[index])), cv2.COLOR_BGR2RGB)
        self.augmentor.augment(sat, mask, str(image_id))

    def visualise_augmentations(self, index, num_examples=8):
        sat, _, _ = self.base_dataset[index]
        mask = cv2.cvtColor(cv2.imread(os.path.join(self.base_dataset.directory, self.base_dataset.image_mask[index])), cv2.COLOR_BGR2RGB)

        sat_resized = cv2.resize(sat, self.augmentor.visualisation_size)
        mask_resized = cv2.resize(mask, self.augmentor.visualisation_size)

        aug_images = [torch.from_numpy(sat_resized).permute(2, 0, 1).float() / 255.0]
        aug_masks = [torch.from_numpy(mask_resized).permute(2, 0, 1).float() / 255.0]

        for _ in range(num_examples - 1):
            quarter = self.augmentor.get_quarter(sat, random.randint(0, 3))
            q_img = cv2.resize(sat[quarter[0], quarter[1]], self.augmentor.visualisation_size)
            q_mask = cv2.resize(mask[quarter[0], quarter[1]], self.augmentor.visualisation_size)

            if random.choice([True, False]):
                angle = random.choice(self.augmentor.rotation_angles)
                M = cv2.getRotationMatrix2D((q_img.shape[1]/2, q_img.shape[0]/2), angle, 1)
                q_img = cv2.warpAffine(q_img, M, self.augmentor.visualisation_size)
                q_mask = cv2.warpAffine(q_mask, M, self.augmentor.visualisation_size)

            aug_images.append(torch.from_numpy(q_img).permute(2, 0, 1).float() / 255.0)
            aug_masks.append(torch.from_numpy(q_mask).permute(2, 0, 1).float() / 255.0)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8))
        for ax, grid in zip((ax1, ax2), [make_grid(torch.stack(aug_images), nrow=4, normalize=True), make_grid(torch.stack(aug_masks), nrow=4, normalize=True)]):
            ax.imshow(grid.permute(1, 2, 0).cpu().numpy())
            ax.axis('off')

        ax1.set_title('Original w/ Augmentations')
        ax2.set_title('Corresponding Masks')
        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    dataset = AugmentedSatelliteImages(os.path.join(base_directory, 'data', 'train_data'))
    dataset.visualise_augmentations(100)
