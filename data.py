import os
import cv2
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from model import SatelliteImages
import torchvision.transforms.functional as tf
from torch.utils.data import Dataset
from torchvision.utils import make_grid

class SatelliteAugmentation:
    def __init__(self, target_size=(2448, 2448)):
        self.target_size = target_size
        self.rotation_angles = [90, 180, 270]

    def get_quarter(self, image, mask, quarter_idx):
        h, w = image.shape[-2:]
        h_half, w_half = h // 2, w // 2
        slices = [(slice(None), slice(None, h_half), slice(None, w_half)),
                 (slice(None), slice(None, h_half), slice(w_half, None)),
                 (slice(None), slice(h_half, None), slice(None, w_half)),
                 (slice(None), slice(h_half, None), slice(w_half, None))]
        mask_slices = tuple(s[1:] for s in slices)
        return (image[slices[quarter_idx]],
                mask[mask_slices[quarter_idx]] if mask is not None else None)

    def __call__(self, image, mask=None):
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        quarter_idx = random.randint(0, 3)
        rotation = random.choice([0] + self.rotation_angles)
        img_quarter, mask_quarter = self.get_quarter(image, mask, quarter_idx)

        img_quarter = tf.resize(img_quarter, self.target_size)
        if rotation:
            img_quarter = tf.rotate(img_quarter, rotation)

        if mask_quarter is not None:
            mask_quarter = tf.resize(mask_quarter.unsqueeze(0).float(),
                                  self.target_size,
                                  interpolation=tf.InterpolationMode.NEAREST)
            if rotation:
                mask_quarter = tf.rotate(mask_quarter, rotation,
                                      interpolation=tf.InterpolationMode.NEAREST)
            return img_quarter, mask_quarter.squeeze(0).long()
        return img_quarter

class AugmentedSatelliteImages(Dataset):
    def __init__(self, directory, labels=None, transform=None):
        self.base_dataset = SatelliteImages(directory, labels, transform)
        self.augmentor = SatelliteAugmentation()

    def __getitem__(self, i):
        sat, mask, image_id = self.base_dataset[i]
        aug_sat, aug_mask = self.augmentor(sat, mask)
        return aug_sat, aug_mask, image_id

    def __len__(self):
        return len(self.base_dataset)

    def visualize_augmentations(self, index, num_examples=8):
        sat, mask, _ = self.base_dataset[index]
        orig_tensor = torch.from_numpy(sat).permute(2, 0, 1).float() / 255.0

        augmented_images = [orig_tensor]
        augmented_masks = [mask]

        for _ in range(num_examples - 1):
            aug_sat, aug_mask = self.augmentor(sat.copy(), mask)
            augmented_images.append(aug_sat)
            augmented_masks.append(aug_mask)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8))

        image_grid = make_grid(torch.stack(augmented_images), nrow=4, padding=20, normalize=True)
        mask_grid = make_grid(torch.stack([m.unsqueeze(0).float() for m in augmented_masks]),
                            nrow=4, padding=20, normalize=True)

        ax1.imshow(image_grid.permute(1, 2, 0).cpu().numpy())
        ax2.imshow(mask_grid.permute(1, 2, 0).squeeze().cpu().numpy(), cmap='tab10', vmin=0, vmax=6)

        ax1.set_title('Original + Random Quarters with Rotations')
        ax2.set_title('Corresponding Masks')
        ax1.axis('off')
        ax2.axis('off')
        plt.tight_layout()
        plt.show()

import os
import cv2
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from model import SatelliteImages
import torchvision.transforms.functional as tf
from torch.utils.data import Dataset
from torchvision.utils import make_grid

class SatelliteAugmentation:
    def __init__(self, target_size=(2448, 2448)):
        self.target_size = target_size
        self.rotation_angles = [90, 180, 270]

    def get_quarter(self, image, mask, quarter_idx):
        h, w = image.shape[-2:]
        h_half, w_half = h // 2, w // 2
        slices = [(slice(None), slice(None, h_half), slice(None, w_half)),
                 (slice(None), slice(None, h_half), slice(w_half, None)),
                 (slice(None), slice(h_half, None), slice(None, w_half)),
                 (slice(None), slice(h_half, None), slice(w_half, None))]
        mask_slices = tuple(s[1:] for s in slices)
        return (image[slices[quarter_idx]],
                mask[mask_slices[quarter_idx]] if mask is not None else None)

    def __call__(self, image, mask=None):
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        quarter_idx = random.randint(0, 3)
        rotation = random.choice([0] + self.rotation_angles)
        img_quarter, mask_quarter = self.get_quarter(image, mask, quarter_idx)

        img_quarter = tf.resize(img_quarter, self.target_size)
        if rotation:
            img_quarter = tf.rotate(img_quarter, rotation)

        if mask_quarter is not None:
            mask_quarter = tf.resize(mask_quarter.unsqueeze(0).float(),
                                  self.target_size,
                                  interpolation=tf.InterpolationMode.NEAREST)
            if rotation:
                mask_quarter = tf.rotate(mask_quarter, rotation,
                                      interpolation=tf.InterpolationMode.NEAREST)
            return img_quarter, mask_quarter.squeeze(0).long()
        return img_quarter

class AugmentedSatelliteImages(Dataset):
    def __init__(self, directory, labels=None, transform=None):
        self.base_dataset = SatelliteImages(directory, labels, transform)
        self.augmentor = SatelliteAugmentation()

    def __getitem__(self, i):
        sat, mask, image_id = self.base_dataset[i]
        aug_sat, aug_mask = self.augmentor(sat, mask)
        return aug_sat, aug_mask, image_id

    def __len__(self):
        return len(self.base_dataset)

    def visualize_augmentations(self, index, num_examples=8):
        sat, mask, _ = self.base_dataset[index]
        orig_tensor = torch.from_numpy(sat).permute(2, 0, 1).float() / 255.0

        augmented_images = [orig_tensor]
        augmented_masks = [mask]

        for _ in range(num_examples - 1):
            aug_sat, aug_mask = self.augmentor(sat.copy(), mask)
            augmented_images.append(aug_sat)
            augmented_masks.append(aug_mask)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8))

        image_grid = make_grid(torch.stack(augmented_images), nrow=4, padding=20, normalize=True)
        mask_grid = make_grid(torch.stack([m.unsqueeze(0).float() for m in augmented_masks]),
                            nrow=4, padding=20, normalize=True)

        ax1.imshow(image_grid.permute(1, 2, 0).cpu().numpy())
        ax2.imshow(mask_grid.permute(1, 2, 0).squeeze().cpu().numpy(), cmap='tab10', vmin=0, vmax=6)

        ax1.set_title('Original + Random Quarters with Rotations')
        ax2.set_title('Corresponding Masks')
        ax1.axis('off')
        ax2.axis('off')
        plt.tight_layout()
        plt.show()


def augment_and_save_image(image_index, source_dir='_data/train', target_dir='_data/train_aug'):
    """
    Create and save all augmented versions of a single image
    Args:
        image_index (int): Index of image in dataset to augment
        source_dir (str): Directory containing original images
        target_dir (str): Directory to save augmented images
    """
    # Create target directory if it doesn't exist
    os.makedirs(target_dir, exist_ok=True)

    # Load dataset and get specified image
    dataset = SatelliteImages(source_dir)
    sat, mask, image_id = dataset[image_index]

    # Convert original image to tensor
    image = torch.from_numpy(sat).permute(2, 0, 1).float()

    # Get all quarters
    h, w = image.shape[-2:]
    h_half, w_half = h // 2, w // 2
    quarters = [
        (slice(None), slice(None, h_half), slice(None, w_half)),
        (slice(None), slice(None, h_half), slice(w_half, None)),
        (slice(None), slice(h_half, None), slice(None, w_half)),
        (slice(None), slice(h_half, None), slice(w_half, None))
    ]
    mask_quarters = [s[1:] for s in quarters]

    # Process each quarter
    for q_idx, (q_slice, m_slice) in enumerate(zip(quarters, mask_quarters)):
        # Extract quarter
        img_quarter = image[q_slice]
        mask_quarter = mask[m_slice]

        # Save original quarter
        quarter_name = f"{image_id}_q{q_idx}"
        img_path = os.path.join(target_dir, f"{quarter_name}_sat.jpg")
        mask_path = os.path.join(target_dir, f"{quarter_name}_mask.png")

        # Convert to numpy and save
        img_save = (img_quarter.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        cv2.imwrite(img_path, cv2.cvtColor(img_save, cv2.COLOR_RGB2BGR))
        cv2.imwrite(mask_path, mask_quarter.numpy())

        # Create and save rotated versions
        for angle in [90, 180, 270]:
            # Rotate image and mask
            img_rot = tf.rotate(img_quarter, angle)
            mask_rot = tf.rotate(mask_quarter.unsqueeze(0).float(), angle,
                               interpolation=tf.InterpolationMode.NEAREST).squeeze(0).long()

            # Save rotated versions
            rot_name = f"{image_id}_q{q_idx}_r{angle}"
            img_rot_path = os.path.join(target_dir, f"{rot_name}_sat.jpg")
            mask_rot_path = os.path.join(target_dir, f"{rot_name}_mask.png")

            # Convert to numpy and save
            img_rot_save = (img_rot.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            cv2.imwrite(img_rot_path, cv2.cvtColor(img_rot_save, cv2.COLOR_RGB2BGR))
            cv2.imwrite(mask_rot_path, mask_rot.numpy())

    print(f"Saved all augmented versions of image {image_id} to {target_dir}")

if __name__ == '__main__':
    # data = AugmentedSatelliteImages('_data/train')
    # data.visualize_augmentations(index=0)
    augment_and_save_image(0)
