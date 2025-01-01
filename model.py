# Initial notes: Train/test/val data spec: 2448 x 2448 pixels, RGB, 50cm/pixel -> ~1.224km img width
# For implementation purposes: Data augmentation needed, augment data with different zoom, width specs,
#
import time
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
from tqdm import tqdm
from torch.optim.lr_scheduler import OneCycleLR

# Transformer
transformer = transforms.Compose([transforms.ToTensor(),
                                  transforms.Resize((256, 256), antialias=False),
                                  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

class SatelliteImages(Dataset):
    # Class for data loading from 'https://www.kaggle.com/datasets/balraj98/deepglobe-land-cover-classification-dataset?select=class_dict.csv'
    def __init__(self, directory, labels=None, transform=None):
        self.directory = directory
        self.labels = labels
        self.transform = transform
        self.image_sat = sorted([file for file in os.listdir(self.directory) if file.endswith('_sat.jpg')])
        self.image_mask = sorted([file for file in os.listdir(self.directory) if file.endswith('_mask.png')])
        self.image_id = sorted([int(re.match(r'(\d+)', file).group(1)) for file in os.listdir(self.directory) if file.endswith('_sat.jpg')])

    # def __len__(self):
    #     return len(self.image_id)

    # def __getitem__(self, i):
    #     id = self.image_id[i]
    #     sat_path = os.path.join(self.directory, self.image_sat[i])
    #     mask_path = os.path.join(self.directory, self.image_mask[i])
    #     # Convert item to RGB cv2 image.
    #     sat = cv2.cvtColor(cv2.imread(sat_path), cv2.COLOR_BGR2RGB)
    #     mask = cv2.cvtColor(cv2.imread(mask_path), cv2.COLOR_BGR2RGB)
    #     # Apply transformer if True
    #     if self.transform:
    #         sat = self.transform(sat)
    #         mask = self.transform(mask)
    #     return sat, mask, id

        self.rgb_to_class = {(0, 255, 255): 0,    # urban_land
                             (255, 255, 0): 1,    # agriculture_land
                             (255, 0, 255): 2,    # rangeland
                             (0, 255, 0): 3,      # forest_land
                             (0, 0, 255): 4,      # water
                             (255, 255, 255): 5,  # barren_land
                             (0, 0, 0): 6}

    def __len__(self):
        return len(self.image_id)

    def __getitem__(self, i):
        id = self.image_id[i]
        sat_path = os.path.join(self.directory, self.image_sat[i])
        mask_path = os.path.join(self.directory, self.image_mask[i])

        # Load satellite image
        sat = cv2.cvtColor(cv2.imread(sat_path), cv2.COLOR_BGR2RGB)
        if self.transform:
            sat = self.transform(sat)

        # Load and convert mask to class indices
        mask = cv2.cvtColor(cv2.imread(mask_path), cv2.COLOR_BGR2RGB)
        mask_tensor = torch.zeros((mask.shape[0], mask.shape[1]), dtype=torch.long)

        for rgb, class_idx in self.rgb_to_class.items():
            mask_match = (mask == torch.tensor(rgb).view(1, 1, 3))
            mask_match = mask_match.all(dim=2)
            mask_tensor[mask_match] = class_idx

        if self.transform:
            # For mask, we need a custom transform that maintains the class indices
            mask_tensor = torch.nn.functional.interpolate(
                mask_tensor.unsqueeze(0).unsqueeze(0).float(),
                size=(256, 256),
                mode='nearest'
            ).squeeze(0).squeeze(0).long()

        return sat, mask_tensor, id

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

class Train():
    def __init__(self, data, model, optimiser, loss, epochs, batch_size, val_data=None):
        self.data = data
        self.model = model
        self.optimiser = optimiser
        self.loss = loss
        self.epochs = epochs
        self.batch_size = batch_size
        self.val_data = val_data
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



        # Initialise data
        self.load_train = DataLoader(data,
                                     batch_size=batch_size,
                                     shuffle=True,
                                     pin_memory=True if self.device == 'cuda' else False,
                                     num_workers=4,
                                     prefetch_factor=2,
                                     persistent_workers=True)

        if val_data: self.load_val = DataLoader(val_data,
                                                batch_size=batch_size,
                                                shuffle=False,
                                                pin_memory=True if self.device =='cuda' else False)

        # Scheduler for adjustable LR
        self.scheduler = OneCycleLR(self.optimiser,
                                    max_lr=0.01,
                                    epochs=epochs,
                                    steps_per_epoch=len(self.load_train))

        # Initialise model
        self.model = self.model.to(self.device)
        self.train_loss, self.val_loss, self.best_loss  = [], [], float('inf')

    def train_epoch(self, e):
        self.model.train()
        total_loss = 0
        steps = len(self.load_train)
        pbar = tqdm(self.load_train, desc=f'Epoch {e+1}/{self.epochs}')

        for i, (images, masks, _) in enumerate(pbar):
            images = images.to(self.device, dtype=torch.float32)
            masks = masks.to(self.device, dtype=torch.long)
            self.optimiser.zero_grad()

            # Forward pass, calc loss, backward pass
            output = self.model(images)
            seg = output['segmentation']
            loss = self.loss(seg, masks)
            loss.backward()
            self.optimiser.step()
            self.scheduler.step()

            total_loss += loss.item()
            curr_loss = total_loss / (i+1)

            # Illustrate loss
            pbar.set_postfix({'loss': f'{curr_loss:.3f}'})

        # Individual epoch loss
        self.train_loss.append(total_loss / steps)
        return total_loss / steps

    def validate(self):
        if not self.val_data:
            return None

        self.model.eval()
        total_loss = 0
        steps = len(self.load_val)
        with torch.no_grad():
            for images, masks, _ in tqdm(self.load_val, unit='Validating...'):
                images = images.to(self.device, dtype=torch.float32)
                masks = masks.to(self.device, dtype=torch.long)

                output = self.model(images)
                seg = output['segmentation']

                loss = self.loss(seg, masks)
                total_loss += loss.item()

        val_loss = total_loss / steps
        self.val_loss.append(val_loss)
        return val_loss

    def train(self, checkpoint_dir='_data/models'):
        start_time = time.time()
        print(f" Initialised ------- Training {self.epochs} epochs.")
        for epoch in range(self.epochs):
            # Train epoch
            train_loss = self.train_epoch(epoch)

            # Validate if validation set exists
            val_loss = self.validate()

            # Print epoch summary
            print(f"\nEpoch {epoch+1}/{self.epochs} Summary:")
            print(f"Train Loss: {train_loss:.4f}")
            if val_loss is not None:
                print(f"Validation Loss: {val_loss:.4f}")
                # Save best model
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    torch.save({'epoch': epoch,
                                'model_state_dict': self.model.state_dict(),
                                'optimizer_state_dict': self.optimiser.state_dict(),
                                'train_loss': train_loss,
                                'val_loss': val_loss,}, f"{checkpoint_dir}/model.pt")

            print("-" * 50)
        training_time = time.time() - start_time
        print(f"\nTraining completed in {training_time/60:.2f} minutes")

        torch.save(model.state_dict(), 'model.pth')

        return {'train_losses': self.train_loss,
                'val_losses': self.val_loss,
                'training_time': training_time}

def train_model():
    # Init params & hyperparams
    model = UNet()
    training_data = SatelliteImages('_data/train', transform=transformer)
    validation_data = SatelliteImages('_data/valid', transform=transformer)
    optimiser = torch.optim.Adam(model.parameters(), lr=0.01)
    loss = nn.CrossEntropyLoss()

    trainer = Train(model=model,
                  data=training_data,
                  optimiser=optimiser,
                  loss=loss,
                  epochs=2,
                  batch_size=16,
                  val_data=None)

    print(trainer.device)

    res = trainer.train()
    # print(f"Best validation loss: {min(res['val_losses']):.4f}")
    print(f"Final training loss: {res['train_losses'][-1]:.4f}")
    return res

if __name__ == '__main__':
    model = UNet()
    model.load_state_dict(torch.load('model.pth'))
    model.to(torch.device('cuda'))
    model.eval()
