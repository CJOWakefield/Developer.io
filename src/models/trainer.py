# Initial notes: Train/test/val data spec: 2448 x 2448 pixels, RGB, 50cm/pixel -> ~1.224km img width
# For implementation purposes: Data augmentation needed, augment data with different zoom, width specs,

import time
import os
from torchvision import transforms
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.optim.lr_scheduler import OneCycleLR
import torch.nn as nn
from torch.nn import functional as F
from src.data.loader import SatelliteImages

base_directory = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
model_directory = os.path.join(base_directory, 'models')

# Transformer
transformer = transforms.Compose([transforms.ToTensor(),
                                  transforms.Resize((256, 256), antialias=False),
                                  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

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
        input, dropout = self.config.input_dim, self.config.dropout

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
        self.dres1 = ResidualBlock(512, 256, dropout)
        self.dres2 = ResidualBlock(256, 128, dropout)
        self.dres3 = ResidualBlock(128, 64, dropout)

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
        if x.dim() != 4: raise ValueError(f"Expected 4D input (batch, channels, height, width), got {x.dim()}D")

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
        if layer.shape != c3.shape: layer = F.interpolate(layer, size=c3.shape[2:])
        layer = torch.cat([layer, c3], dim=1)
        layer = self.dres1(layer)

        layer = self.dconv2(layer)
        if layer.shape != c2.shape: layer = F.interpolate(layer, size=c2.shape[2:])
        layer = torch.cat([layer, c2], dim=1)
        layer = self.dres2(layer)

        layer = self.dconv3(layer)
        if layer.shape != c1.shape: layer = F.interpolate(layer, size=c1.shape[2:])
        layer = torch.cat([layer, c1], dim=1)
        layer = self.dres3(layer)

        segmentation = self.segmentation(layer)

        return {'segmentation': segmentation,
                'features': layer,
                'encoded': c_final}

class ModelInit:
    def __init__(self, config=None):
        self.config = config if config else UNetConfig()
        self.model = UNet(self.config)

    def model_params(self, input_dim=(1, 3, 256, 256)):
        x = torch.randn(*input_dim)
        output = self.model(x)
        print(f'Trainable params: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}')
        print(f"Segmentation shape: {output['segmentation'].shape}")
        print(f"Feature shape: {output['features'].shape}")
        print(f"Encoded shape: {output['encoded'].shape}")

    def get_model(self):
        return self.model

class Train:
    def __init__(self, data, model, optimiser, loss, epochs, batch_size, model_directory=model_directory, val_data=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.optimiser, self.loss, self.epochs = optimiser, loss, epochs
        self.train_loss, self.val_loss = [], []
        self.best_val_loss = float('inf')
        self.model_directory = model_directory

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

    def get_next_version(self):
        """Generate next available version number for model directory"""
        existing_dirs = [d for d in os.listdir(model_directory) if d.startswith('v_')]
        if not existing_dirs: return 'v_0_01'

        versions = []
        for d in existing_dirs:
            nums = d.replace('v_', '').split('_')
            versions.append((int(nums[0]), int(nums[1])))
        if not versions: return 'v_0_01'

        latest_major, latest_minor = max(versions)
        next_minor = str(latest_minor + 1).zfill(2)
        return f'v_{latest_major}_{next_minor}'

    def train(self):
        version = self.get_next_version()
        checkpoint_dir = os.path.join(self.model_directory, version)
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
    model = ModelInit().get_model()
    training_data = SatelliteImages(os.path.join(base_directory, 'data', 'train_data'), transform=transformer)
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

    trainer.train()


if __name__ == '__main__':
    # model = ModelInit()
    # model.model_params()
    train_model(epochs=5)
