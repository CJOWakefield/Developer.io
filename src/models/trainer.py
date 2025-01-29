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
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable

# Transformer
transformer = transforms.Compose([transforms.ToTensor(),
                                  transforms.Resize((256, 256), antialias=False),
                                  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

class SatelliteImages(Dataset):
    # Class for data loading from 'https://www.kaggle.com/datasets/balraj98/deepglobe-land-cover-classification-dataset?select=class_dict.csv'
    def __init__(self, directory, labels=None, transform=None):
        self.directory = directory
        self.transform = transform
        self.image_sat = sorted([f for f in os.listdir(directory) if f.endswith('_sat.jpg')])
        self.image_mask = sorted([f for f in os.listdir(directory) if f.endswith('_mask.png')])
        self.image_id = sorted([int(re.match(r'(\d+)', f).group(1)) for f in os.listdir(directory) if f.endswith('_sat.jpg')])
        self.rgb_to_class = {(0, 255, 255): 0,
                             (255, 255, 0): 1,
                             (255, 0, 255): 2,
                             (0, 255, 0): 3,
                             (0, 0, 255): 4,
                             (255, 255, 255): 5,
                             (0, 0, 0): 6}

    def __getitem__(self, i):
        sat = cv2.cvtColor(cv2.imread(os.path.join(self.directory, self.image_sat[i])), cv2.COLOR_BGR2RGB)
        mask = cv2.cvtColor(cv2.imread(os.path.join(self.directory, self.image_mask[i])), cv2.COLOR_BGR2RGB)
        mask_tensor = torch.zeros((mask.shape[0], mask.shape[1]), dtype=torch.long)

        if self.transform:
            sat = self.transform(sat)
            for rgb, class_idx in self.rgb_to_class.items():
                mask_tensor[torch.all(mask == torch.tensor(rgb).view(1, 1, 3), dim=2)] = class_idx
            mask_tensor = F.interpolate(mask_tensor.unsqueeze(0).unsqueeze(0).float(),
                                      size=(256, 256), mode='nearest').squeeze().long()

        return sat, mask_tensor, self.image_id[i]

    def __len__(self): return len(self.image_id)

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
    def __init__(self, n_classes=7, input_dim=3, output_dim=1, n_features=16, dropout=0.2):
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
        self.b1 = ResidualBlock(input, 64, dropout)
        self.b2 = ResidualBlock(64, 128, dropout)
        self.b3 = ResidualBlock(128, 256, dropout)
        self.b4 = ResidualBlock(256, 512, dropout)

        # Pooling
        self.maxpool = nn.MaxPool2d(2)
        self.dropout_layer = nn.Dropout2d(p=dropout)

        # Decoder - transposed
        self.dconv1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2, padding=0)
        self.dconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2, padding=0)
        self.dconv3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, padding=0)

        # Decoder - residual
        self.dres1 = ResidualBlock(512, 256, dropout)  # After concatenation
        self.dres2 = ResidualBlock(256, 128, dropout)  # After concatenation
        self.dres3 = ResidualBlock(128, 64, dropout)      # After concatenation

        # Final layers
        self.segmentation = nn.Conv2d(64, self.config.n_classes, kernel_size=1)
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
        # Ensure input dimensions are correct
        if x.dim() != 4:
            raise ValueError(f"Expected 4D input (batch, channels, height, width), got {x.dim()}D")

        # Encoder path with dimension tracking
        c1 = self.b1(x)
        x1 = self.maxpool(self.dropout_layer(c1))

        c2 = self.b2(x1)
        x2 = self.maxpool(self.dropout_layer(c2))

        c3 = self.b3(x2)
        x3 = self.maxpool(self.dropout_layer(c3))

        c_final = self.b4(x3)

        # Decoder path with explicit dimension handling
        layer = self.dconv1(c_final)
        # Ensure dimensions match before concatenation
        if layer.shape != c3.shape:
            layer = F.interpolate(layer, size=c3.shape[2:])
        layer = torch.cat([layer, c3], dim=1)
        layer = self.dres1(layer)

        layer = self.dconv2(layer)
        if layer.shape != c2.shape:
            layer = F.interpolate(layer, size=c2.shape[2:])
        layer = torch.cat([layer, c2], dim=1)
        layer = self.dres2(layer)

        layer = self.dconv3(layer)
        if layer.shape != c1.shape:
            layer = F.interpolate(layer, size=c1.shape[2:])
        layer = torch.cat([layer, c1], dim=1)
        layer = self.dres3(layer)

        segmentation = self.segmentation(layer)

        return {'segmentation': segmentation,
                'features': layer,
                'encoded': c_final}

def Model():
    config = UNetConfig()
    model = UNet(config)
    x = torch.randn(1, 3, 256, 256)
    output = model(x)

    print(f'Trainable params: {model.params():,}')
    print(f"Segmentation shape: {output['segmentation'].shape}")
    print(f"Feature shape: {output['features'].shape}")
    print(f"Encoded shape: {output['encoded'].shape}")

class Train:
    def __init__(self, data, model, optimiser, loss, epochs, batch_size, val_data=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.optimiser, self.loss, self.epochs = optimiser, loss, epochs
        self.train_loss, self.val_loss = [], []
        self.best_val_loss = float('inf')

        self.load_train = DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=4,
                                     pin_memory=bool(self.device=='cuda'), prefetch_factor=2,
                                     persistent_workers=True)

        self.load_val = DataLoader(val_data, batch_size=batch_size, shuffle=False,
                                   pin_memory=bool(self.device=='cuda')) if val_data else None

        self.scheduler = OneCycleLR(optimiser, max_lr=0.01, epochs=epochs,
                                    steps_per_epoch=len(self.load_train))

    def train_epoch(self, e):
        self.model.train()
        total_loss = 0
        pbar = tqdm(self.load_train, desc=f'Epoch {e+1}/{self.epochs}')

        for i, (images, masks, _) in enumerate(pbar):
            self.optimiser.zero_grad()
            loss = self.loss(self.model(images.to(self.device, dtype=torch.float32))['segmentation'],
                             masks.to(self.device, dtype=torch.long))

            loss.backward()
            self.optimiser.step()
            self.scheduler.step()
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{total_loss/(i+1):.3f}'})

        epoch_loss = total_loss/len(self.load_train)
        self.train_loss.append(epoch_loss)
        return epoch_loss

    def get_next_version(self, base_dir):
        """Generate next available version number for model directory"""
        existing_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and d.startswith('v_')]
        if not existing_dirs:
            return 'v_0_01'

        versions = []
        for d in existing_dirs:
            try:
                # Convert version strings to tuple of integers
                nums = d.replace('v_', '').split('_')
                versions.append((int(nums[0]), int(nums[1])))
            except (ValueError, IndexError):
                continue

        if not versions:
            return 'v_0_01'

        # Get max version and create next version
        latest_major, latest_minor = max(versions)  # Now comparing tuples, not map objects
        next_minor = str(latest_minor + 1).zfill(2)
        return f'v_{latest_major}_{next_minor}'

    def train(self, base_dir='_data/models'):
        version = self.get_next_version(base_dir)
        checkpoint_dir = os.path.join(base_dir, version)
        os.makedirs(checkpoint_dir, exist_ok=True)

        start_time = time.time()
        for epoch in range(self.epochs):
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate() if self.load_val else None

            print(f"\nEpoch {epoch+1}/{self.epochs} Summary:")
            print(f"Train Loss: {train_loss:.4f}")
            if val_loss:
                print(f"Validation Loss: {val_loss:.4f}")

            checkpoint = {'epoch': epoch,
                          'model_state_dict': self.model.state_dict(),
                          'optimizer_state_dict': self.optimiser.state_dict(),
                          'train_loss': train_loss,
                          'val_loss': val_loss}

            torch.save(checkpoint, f"{checkpoint_dir}/epoch_{epoch+1}.pt")
            if val_loss and val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                torch.save(checkpoint, f"{checkpoint_dir}/best_model.pt")

        torch.save(checkpoint, f"{checkpoint_dir}/model.pt")
        training_time = time.time() - start_time

        return {'train_losses': self.train_loss,
                'val_losses': self.val_loss,
                'training_time': training_time,
                'version': version}


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, reduction='mean'):
        """
        Focal Loss for semantic segmentation
        Args:
            gamma: Focusing parameter for modulating factor (1-p)
            reduction: 'none' | 'mean' | 'sum'
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input, target):
        """
        Forward pass
        Args:
            input: Tensor of shape (N, C, H, W) where C is number of classes
            target: Tensor of shape (N, H, W) with class indices
        """
        # Reshape input and target for focal loss calculation
        if input.dim() > 2:
            input = input.permute(0, 2, 3, 1).contiguous()
            input = input.view(-1, input.size(-1))

        target = target.view(-1)

        # Calculate log softmax
        logpt = F.log_softmax(input, dim=-1)
        logpt = logpt.gather(1, target.view(-1, 1))
        logpt = logpt.view(-1)
        pt = torch.exp(logpt)

        # Calculate focal loss
        focal_loss = -1 * ((1 - pt) ** self.gamma) * logpt

        # Apply reduction
        if self.reduction == "none":
            return focal_loss
        elif self.reduction == "mean":
            return focal_loss.mean()
        else:  # sum
            return focal_loss.sum()

def train_model(epochs=5):
    # Init params & hyperparams
    model = UNet()
    training_data = SatelliteImages('data/train', transform=transformer)
    # validation_data = SatelliteImages('_data/valid', transform=transformer)

    optimiser = torch.optim.Adam(model.parameters(), lr=0.01)
    loss = FocalLoss(gamma=2.0)

    trainer = Train(model=model,
                    data=training_data,
                    optimiser=optimiser,
                    loss=loss,
                    epochs=epochs,
                    batch_size=16,
                    val_data=None)

    res = trainer.train()

def visualise_pred(model_path=None, data_dir='_data/train/', n_samples=3, image_ids=None):
    ## Model setup - Load, initiate CUDA
    if not model_path:
        base_dir = '_data/models'
        versions = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and d.startswith('v_')]
        if not versions:
            raise ValueError("No model versions found in _data/models")

        latest_version = sorted(versions, key=lambda x: (int(x.split('_')[1]), int(x.split('_')[2])))[-1]
        model_path = os.path.join(base_dir, latest_version, 'model.pt')

    model = UNet()
    model.load_state_dict(torch.load(model_path)['model_state_dict'])
    model.eval().to(device := torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    ## Load image sample or specified index
    if image_ids:
        image_ids = [image_ids] if isinstance(image_ids, (int, str)) else image_ids
        sat_files = [f for id in image_ids for f in os.listdir(data_dir) if f.endswith(f"_sat.jpg")]
        if not sat_files:
            raise ValueError("No matching images found")
        n_samples = len(sat_files)
    else: sat_files = np.random.choice([f for f in os.listdir(data_dir) if f.endswith('_sat.jpg')], size=n_samples, replace=False)

    class_colors = {0: (0, 255, 255), # urban land - Cyan
                    1: (255, 255, 0), # agricultural land - Yellow
                    2: (255, 0, 255), # rangeland - Magenta
                    3: (0, 255, 0), # forest land - Green
                    4: (0, 0, 255), # water - Dark blue
                    5: (255, 255, 255), # barren land - White
                    6: (0, 0, 0)} # unknown - Black

    rows = 1 if n_samples <= 3 else (n_samples + 2) // 3
    cols = min(n_samples, 3)
    plt.figure(figsize=(6*cols, 8*rows))

    ## Generate image mask, converting to RGB and plotting against _sat.jpg image.
    results = []
    for idx, file in enumerate(sat_files):
        # Load and predict
        sat_img = cv2.cvtColor(cv2.imread(os.path.join(data_dir, file)), cv2.COLOR_BGR2RGB)
        input_tensor = transformer(sat_img).unsqueeze(0).to(device)
        pred = torch.argmax(model(input_tensor)['segmentation'].squeeze(), dim=0).cpu().numpy()

        pred_rgb = np.zeros((*pred.shape, 3), dtype=np.uint8)
        for class_idx, color in class_colors.items():
            pred_rgb[pred == class_idx] = color

        image_id = re.match(r'(\d+)', file).group(1)
        results.append((sat_img, pred_rgb, image_id))

        plt.subplot(rows*2, cols, idx+1)
        plt.title(f'Original Image (ID: {image_id})')
        plt.imshow(sat_img)
        plt.axis('off')

        plt.subplot(rows*2, cols, idx+1+cols)
        plt.title(f'Predicted Mask (ID: {image_id})')
        plt.imshow(pred_rgb)
        plt.axis('off')

    plt.tight_layout()
    plt.show()
    return results

if __name__ == '__main__':
    # train_model(epochs=5)
    print(visualise_pred(n_samples=6))
