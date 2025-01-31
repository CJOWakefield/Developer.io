import os
import cv2
import re
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torch.nn import functional as F

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
    # Class for data loading from 'https://www.kaggle.com/datasets/balraj98/deepglobe-land-cover-classification-dataset?select=class_dict.csv'
    def __init__(self, directory, transform=None):
        self.directory = directory
        self.transform = transform
        self.image_sat = sorted([f for f in os.listdir(directory) if f.endswith('_sat.jpg')])
        self.image_mask = sorted([f for f in os.listdir(directory) if f.endswith('_mask.png')])
        self.image_id = sorted([int(re.match(r'(\d+)', f).group(1)) for f in os.listdir(directory) if f.endswith('_sat.jpg')])
        self.rgb_to_class = {(0, 255, 255): 0, (255, 255, 0): 1, (255, 0, 255): 2,
                            (0, 255, 0): 3, (0, 0, 255): 4, (255, 255, 255): 5, (0, 0, 0): 6}

    def __getitem__(self, i):
        sat = cv2.cvtColor(cv2.imread(os.path.join(self.directory, self.image_sat[i])), cv2.COLOR_BGR2RGB)
        mask = cv2.cvtColor(cv2.imread(os.path.join(self.directory, self.image_mask[i])), cv2.COLOR_BGR2RGB)
        mask_tensor = torch.zeros((mask.shape[0], mask.shape[1]), dtype=torch.long)

        if self.transform:
            sat = self.transform(sat)
            for rgb, class_idx in self.rgb_to_class.items(): mask_tensor[torch.all(mask == torch.tensor(rgb).view(1, 1, 3), dim=2)] = class_idx
            mask_tensor = F.interpolate(mask_tensor.unsqueeze(0).unsqueeze(0).float(), size=(256, 256), mode='nearest').squeeze().long()

        return sat, mask_tensor, self.image_id[i]

    def __len__(self): return len(self.image_id)


class ImagePreview:
    def __init__(self, directory):
        self.directory = directory
        self.class_to_rgb = {0: (0, 255, 255), 1: (255, 255, 0), 2: (255, 0, 255),
                            3: (0, 255, 0), 4: (0, 0, 255), 5: (255, 255, 255), 6: (0, 0, 0)}

    def preview(self, img_id):
        images = SatelliteImages(self.directory, transform=None)
        idx = images.image_id.index(img_id)

        sat_path = os.path.join(self.directory, images.image_sat[idx])
        mask_path = os.path.join(self.directory, images.image_mask[idx])

        sat = cv2.cvtColor(cv2.imread(sat_path), cv2.COLOR_BGR2RGB)
        mask = cv2.cvtColor(cv2.imread(mask_path), cv2.COLOR_BGR2RGB)

        plt.subplot(1, 2, 1)
        plt.imshow(sat)
        plt.title('Satellite Image')

        plt.subplot(1, 2, 2)
        plt.imshow(mask)
        plt.title('Mask')
        plt.show()

if __name__ == '__main__':
    images = ImagePreview(os.path.join(base_directory, 'data', 'train_data'))
    images.preview(855)
