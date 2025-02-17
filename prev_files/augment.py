import os
import cv2
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from prev_files.model import SatelliteImages
import torchvision.transforms.functional as tf
from torch.utils.data import Dataset
from torchvision.utils import make_grid

class SatelliteAugmentation:
    def __init__(self, target_size=(2448, 2448)):
        self.target_size = target_size
        self.rotation_angles = [90, 180, 270]

    def get_quarter(self, image, quarter_idx):
        h, w = image.shape[-2:] if torch.is_tensor(image) else image.shape[:2]
        h_half, w_half = h // 2, w // 2
        if quarter_idx == 0: return (slice(None, h_half), slice(None, w_half))
        elif quarter_idx == 1: return (slice(None, h_half), slice(w_half, None))
        elif quarter_idx == 2: return (slice(h_half, None), slice(None, w_half))
        else: return (slice(h_half, None), slice(w_half, None))

class AugmentedSatelliteImages(Dataset):
    def __init__(self, directory, labels=None, transform=None):
        self.base_dataset = SatelliteImages(directory, labels, transform)
        self.augmentor = SatelliteAugmentation()

    def __len__(self):
        return len(self.base_dataset)

    def read_mask(self, index):
        mask_path = os.path.join(self.base_dataset.directory, self.base_dataset.image_mask[index])
        return cv2.cvtColor(cv2.imread(mask_path), cv2.COLOR_BGR2RGB)

    def process_quarter(self, image, mask, quarter_idx, angle=0):
        h_idx, w_idx = self.augmentor.get_quarter(image, quarter_idx)
        img_quarter = image[h_idx, w_idx] if isinstance(image, np.ndarray) else image[:, h_idx, w_idx]
        mask_quarter = mask[h_idx, w_idx]

        # Resize
        if torch.is_tensor(img_quarter):
            img_quarter = tf.resize(img_quarter, self.augmentor.target_size)
        else:
            img_quarter = cv2.resize(img_quarter, self.augmentor.target_size[::-1])
        mask_quarter = cv2.resize(mask_quarter, self.augmentor.target_size[::-1], interpolation=cv2.INTER_NEAREST)

        # Rotate if needed
        if angle:
            if torch.is_tensor(img_quarter):
                img_quarter = tf.rotate(img_quarter, angle)
            M = cv2.getRotationMatrix2D((mask_quarter.shape[1]/2, mask_quarter.shape[0]/2), angle, 1.0)
            mask_quarter = cv2.warpAffine(mask_quarter, M, mask_quarter.shape[1::-1], flags=cv2.INTER_NEAREST)

        return img_quarter, mask_quarter

    def save_augmentations(self, index, target_dir='_data/train_aug'):
        os.makedirs(target_dir, exist_ok=True)
        sat, _, image_id = self.base_dataset[index]
        mask = self.read_mask(index)

        for q_idx in range(4):
            # Process and save original quarter
            img_quarter, mask_quarter = self.process_quarter(sat, mask, q_idx)
            base_name = f"{image_id}_q{q_idx}"
            cv2.imwrite(os.path.join(target_dir, f"{base_name}_sat.jpg"), img_quarter)
            cv2.imwrite(os.path.join(target_dir, f"{base_name}_mask.png"), cv2.cvtColor(mask_quarter, cv2.COLOR_RGB2BGR))

            # Process and save rotations
            for angle in self.augmentor.rotation_angles:
                img_rot, mask_rot = self.process_quarter(sat, mask, q_idx, angle)
                rot_name = f"{image_id}_q{q_idx}_r{angle}"
                cv2.imwrite(os.path.join(target_dir, f"{rot_name}_sat.jpg"), img_rot)
                cv2.imwrite(os.path.join(target_dir, f"{rot_name}_mask.png"), cv2.cvtColor(mask_rot, cv2.COLOR_RGB2BGR))

    def visualize_augmentations(self, index, num_examples=8):
        sat, _, _ = self.base_dataset[index]
        mask = self.read_mask(index)

        # Convert original to tensor format
        orig_tensor = torch.from_numpy(sat).permute(2, 0, 1).float() / 255.0
        mask_tensor = torch.from_numpy(mask).permute(2, 0, 1).float() / 255.0

        # Generate augmentations
        aug_images = [orig_tensor]
        aug_masks = [mask_tensor]

        for _ in range(num_examples - 1):
            q_idx = random.randint(0, 3)
            angle = random.choice([0] + self.augmentor.rotation_angles)
            img_aug, mask_aug = self.process_quarter(orig_tensor, mask, q_idx, angle)
            if not torch.is_tensor(img_aug):
                img_aug = torch.from_numpy(img_aug).permute(2, 0, 1).float() / 255.0
            mask_aug = torch.from_numpy(mask_aug).permute(2, 0, 1).float() / 255.0
            aug_images.append(img_aug)
            aug_masks.append(mask_aug)

        # Create visualization
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8))
        image_grid = make_grid(torch.stack(aug_images), nrow=4, padding=20, normalize=True)
        mask_grid = make_grid(torch.stack(aug_masks), nrow=4, padding=20, normalize=True)

        ax1.imshow(image_grid.permute(1, 2, 0).cpu().numpy())
        ax2.imshow(mask_grid.permute(1, 2, 0).cpu().numpy())
        ax1.set_title('Original w/ Augmentations')
        ax2.set_title('Corresponding Masks')
        [ax.axis('off') for ax in (ax1, ax2)]
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    dataset = AugmentedSatelliteImages('_data/train')
    dataset.save_augmentations(0)
