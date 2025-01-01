# Initial notes: Train/test/val data spec: 2448 x 2448 pixels, RGB, 50cm/pixel -> ~1.224km img width
# For implementation purposes: Data augmentation needed, augment data with different zoom, width specs,
#

import os
import re
import cv2
from torch.utils.data import Dataset
from torchvision import transforms
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Transformer
transformer = transforms.Compose([transforms.ToTensor(),
                                  transforms.Resize((256, 256))])

class SatelliteImages(Dataset):
    # Class for data loading from 'https://www.kaggle.com/datasets/balraj98/deepglobe-land-cover-classification-dataset?select=class_dict.csv'
    def __init__(self, directory, labels=None, transform=None):
        self.directory = directory
        self.labels = labels
        self.transform = transform
        self.image_sat = sorted([file for file in os.listdir(self.directory) if file.endswith('_sat.jpg')])
        self.image_mask = sorted([file for file in os.listdir(self.directory) if file.endswith('_mask.png')])
        self.image_id = sorted([int(re.match(r'(\d+)', file).group(1)) for file in os.listdir(self.directory) if file.endswith('_sat.jpg')])

    def __len__(self):
        return len(self.image_id)

    def __getitem__(self, i):
        id = self.image_id[i]
        sat_path = os.path.join(self.directory, self.image_sat[i])
        mask_path = os.path.join(self.directory, self.image_mask[i])
        # Convert item to RGB cv2 image.
        sat = cv2.cvtColor(cv2.imread(sat_path), cv2.COLOR_BGR2RGB)
        mask = cv2.cvtColor(cv2.imread(mask_path), cv2.COLOR_BGR2RGB)
        # Apply transformer if True
        if self.transform:
            sat = self.transform(sat)
            mask = self.transform(mask)
        return sat, mask, id

class ImagePreview():
    # Preview sat image alongside mask
    def __init__(self, directory, label_directory, id):
        self.directory = directory
        self.label_directory = label_directory
        self.image_pos = id

    def preview(self):
        images = SatelliteImages(self.directory, self.label_directory)
        idx = images.image_id.index(self.image_pos)
        sat, mask, _ = images.__getitem__(idx)
        for i in range(len([sat, mask])):
            plt.subplot(1, 2, i+1)
            plt.imshow([sat, mask][i])
            # plt.axis('off')
        plt.show()

class ResidualBlock(nn.Module):
    def __init__(self, input_dim, output_dim, dropout):
        super().__init__()

        # Init convolution
        self.conv = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(output_dim),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Dropout2d(p=dropout),
            nn.Conv2d(output_dim, output_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(output_dim),
            nn.LeakyReLU(negative_slope=0.1, inplace=True))

        # Init skip
        self.skip = (nn.Sequential(nn.Conv2d(input_dim, output_dim, kernel_size=1, stride=1, padding=0, bias=False),
                                   nn.BatchNorm2d(output_dim),
                                   nn.LeakyReLU(negative_slope=0.1, inplace=True)) if input_dim != output_dim else nn.Identity())

    def forward(self, x):
        return self.conv(x) + self.skip(x)

class UNetConfig():
    def __init__(self, n_classes=7, input_dim=3, output_dim=1, n_features=64, dropout=0.2):
        self.n_classes = n_classes
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_features = n_features
        self.dropout = dropout

class UNet(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config if config != None else UNetConfig()
        input, features, dropout = self.config.input_dim, self.config.n_features, self.config.dropout

        # Encoder
        self.b1 = ResidualBlock(input, features, dropout)
        self.b2 = ResidualBlock(features, features * 2, dropout)
        self.b3 = ResidualBlock(features * 2, features * 4, dropout)
        self.b4 = ResidualBlock(features * 4, features * 8, dropout)

        # Pooling
        self.maxpool = nn.MaxPool2d(2)
        self.dropout_layer = nn.Dropout2d(p=dropout)

        # Decoder - transposed
        self.dconv1 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=2, stride=2)
        self.dconv2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
        self.dconv3 = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)

        # Decoder - residual
        self.dres1 = ResidualBlock(features * 8, features * 4, dropout)
        self.dres2 = ResidualBlock(features * 4, features * 2, dropout)
        self.dres3 = ResidualBlock(features * 2, features, dropout)

        # Final layers
        self.segmentation = nn.Conv2d(features, self.config.n_classes, kernel_size=1)
        self.apply(self.__weights__)

    def __weights__(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Encoder path
        c1 = self.b1(x)
        x1 = self.maxpool(self.dropout_layer(c1))
        c2 = self.b2(x1)
        x2 = self.maxpool(self.dropout_layer(c2))
        c3 = self.b3(x2)
        x3 = self.maxpool(self.dropout_layer(c3))
        c_final = self.b4(x3)

        # Decoder path
        layer = self.dconv1(c_final)
        layer = torch.cat([layer, c3], dim=1)
        layer = self.dres1(layer)

        layer = self.dconv2(layer)
        layer = torch.cat([layer, c2], dim=1)
        layer = self.dres2(layer)

        layer = self.dconv3(layer)
        layer = torch.cat([layer, c1], dim=1)
        layer = self.dres3(layer)

        segmentation = self.segmentation(layer)

        return {'segmentation': segmentation,
                'features': layer,
                'encoded': c_final}

    def log_metrics(self, metrics):
        if self.metric_logger:
            self.metric_logger.update(metrics)

    def params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

def Model():
    config = UNetConfig()
    model = UNet(config)
    x = torch.randn(1, 3, 256, 256)
    output = model(x)

    print(f'Trainable params: {model.params():,}')
    print(f"Segmentation shape: {output['segmentation'].shape}")
    print(f"Feature shape: {output['features'].shape}")
    print(f"Encoded shape: {output['encoded'].shape}")


if __name__ == '__main__':
    # ImagePreview('_data/train', '_data/class_dict.csv', 266).preview()
    Model()
