# Initial notes: Train/test/val data spec: 2448 x 2448 pixels, RGB, 50cm/pixel -> ~1.224km img width
# For implementation purposes: Data augmentation needed, augment data with different zoom, width specs,

import time
import os
from torchvision import transforms
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.optim.lr_scheduler import OneCycleLR
from torch.nn import functional as F
import sys
from torch.amp import autocast, GradScaler
import multiprocessing
import yaml
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple, Callable

# Set up paths
base_directory = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(base_directory)
model_directory = os.path.join(base_directory, 'data', 'models')

from src.data.loader import SatelliteImages

# Load configuration
with open(os.path.join(base_directory, 'configs', 'default_config.yaml'), 'r') as file:
    config = yaml.safe_load(file)

# Set multiprocessing start method
multiprocessing.set_start_method('spawn', force=True)

# Default image transformation
transformer = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((256, 256), antialias=False),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

''' ----- Trainer file summary -----

> ResidualBlock - Basic building block for UNet with skip connections.
> UNetConfig - Configuration class for UNet model parameters.
> UNet - U-Net architecture for semantic segmentation of satellite imagery.
> ModelInit - Helper class for initializing and examining model parameters.
> Train - Training loop implementation with validation and checkpointing.
> FocalLoss - Loss function that focuses on hard examples.

'''

class SpatialAttention(nn.Module):
    """
    Multi-scale spatial attention module to help focus on different spatial regions.
    """
    def __init__(self, kernel_sizes: List[int] = [3, 5, 7], dropout_rate: float = 0.1, min_scale: float = 0.3, max_scale: float = 0.7) -> None:
        super().__init__()
        self.kernel_sizes = kernel_sizes
        self.convs = nn.ModuleList([
            nn.Conv2d(2, 1, kernel_size=k, padding=k//2)
            for k in kernel_sizes
        ])
        self.sigmoid = nn.Sigmoid()
        self.fusion = nn.Conv2d(len(kernel_sizes), 1, kernel_size=1)
        self.batch_norm = nn.BatchNorm2d(1)
        self.dropout = nn.Dropout2d(p=dropout_rate)
        self.min_scale = min_scale
        self.max_scale = max_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attention_maps = []
        for conv in self.convs:
            avg_out = torch.mean(x, dim=1, keepdim=True)
            max_out, _ = torch.max(x, dim=1, keepdim=True)
            x_cat = torch.cat([avg_out, max_out], dim=1)
            attention = self.sigmoid(conv(x_cat))
            attention_maps.append(attention)
        
        fused_attention = self.fusion(torch.cat(attention_maps, dim=1))
        fused_attention = self.batch_norm(fused_attention)
        fused_attention = self.dropout(fused_attention)
        fused_attention = self.min_scale + (self.max_scale - self.min_scale) * fused_attention
        return x * fused_attention

class ResidualBlock(nn.Module):
    """
    Residual block with two convolutional layers and a skip connection.
    
    Args:
        input_dim: Number of input channels
        output_dim: Number of output channels
        dropout: Dropout probability
    """
    def __init__(self, input_dim: int, output_dim: int, dropout: float) -> None:
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(output_dim),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Dropout2d(p=dropout),
            nn.Conv2d(output_dim, output_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(output_dim),
            nn.LeakyReLU(negative_slope=0.1, inplace=True))

        self.attention = SpatialAttention()

        self.skip = (nn.Sequential(nn.Conv2d(input_dim, output_dim, kernel_size=1, stride=1, padding=0, bias=False),
                                   nn.BatchNorm2d(output_dim),
                                   nn.LeakyReLU(negative_slope=0.1, inplace=True)) if input_dim != output_dim else nn.Identity())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        out = self.attention(out)
        return out + self.skip(x)

class UNetConfig():
    """
    Configuration class for UNet model parameters.
    
    Args:
        n_classes: Number of output classes for segmentation
        input_dim: Number of input channels
        output_dim: Number of output channels
        n_features: Base number of features
        dropout: Dropout probability
    """
    def __init__(self, n_classes: int = 7, input_dim: int = 3, output_dim: int = 1, n_features: int = 16, dropout: float = 0.2) -> None:
        self.n_classes = n_classes
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_features = n_features
        self.dropout = dropout

class UNet(nn.Module):
    """
    U-Net architecture for semantic segmentation of satellite imagery.
    
    Args:
        config: UNetConfig object with model parameters
    """
    def __init__(self, config: UNetConfig = None, confidence_threshold: float = 0.5) -> None:
        super().__init__()
        self.config = config if config != None else UNetConfig()
        input, dropout = self.config.input_dim, self.config.dropout

        self.b1 = ResidualBlock(input, 64, dropout)
        self.b2 = ResidualBlock(64, 128, dropout)
        self.b3 = ResidualBlock(128, 256, dropout)
        self.b4 = ResidualBlock(256, 512, dropout)

        self.maxpool = nn.MaxPool2d(2)
        self.dropout_layer = nn.Dropout2d(p=dropout)

        self.dconv1 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1)
        self.dconv2 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1)
        self.dconv3 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1)

        self.dres1 = ResidualBlock(512, 256, dropout)
        self.dres2 = ResidualBlock(256, 128, dropout)
        self.dres3 = ResidualBlock(128, 64, dropout)

        self.segmentation = nn.Conv2d(64, self.config.n_classes, kernel_size=3, padding=1)
        self.apply(self.__weights__)
        
        self.confidence_threshold = confidence_threshold

    def __weights__(self, m: nn.Module) -> None:
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        if x.dim() != 4: raise ValueError(f"Expected 4D input (batch, channels, height, width), got {x.dim()}D")

        c1 = self.b1(x)
        x1 = self.maxpool(self.dropout_layer(c1))

        c2 = self.b2(x1)
        x2 = self.maxpool(self.dropout_layer(c2))

        c3 = self.b3(x2)
        x3 = self.maxpool(self.dropout_layer(c3))

        c_final = self.b4(x3)

        layer = self.dconv1(c_final)
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

        logits = self.segmentation(layer)
        
        if not self.training:
            probs = F.softmax(logits, dim=1)
            
            max_probs, _ = torch.max(probs, dim=1, keepdim=True)
            
            confident_mask = (max_probs > self.confidence_threshold).float()
            
            if confident_mask.sum() < confident_mask.numel():
                sorted_probs, sorted_indices = torch.sort(probs, dim=1, descending=True)
                
                adjusted_logits = logits.clone()
                for i in range(logits.size(1)):
                    is_not_top_class = (sorted_indices[:, 0:1, :, :] != i).float()
                    is_low_confidence = (1 - confident_mask).float()
                    
                    mask = (is_not_top_class * is_low_confidence) > 0.5
                    
                    adjusted_logits[:, i:i+1, :, :][mask] = -10.0
                
                logits = adjusted_logits

        return {'segmentation': logits,
                'features': layer,
                'encoded': c_final}

class ModelInit:
    """
    Helper class for initializing and examining model parameters.
    
    Args:
        config: UNetConfig object with model parameters
    """
    def __init__(self, config: UNetConfig = None, confidence_threshold: float = 0.5) -> None:
        self.config = config if config else UNetConfig()
        self.model = UNet(self.config, confidence_threshold)

    def model_params(self, input_dim: Tuple[int, int, int, int] = (1, 3, 256, 256)) -> None:
        x = torch.randn(*input_dim)
        output = self.model(x)
        print(f'Trainable params: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}')
        print(f"Segmentation shape: {output['segmentation'].shape}")
        print(f"Feature shape: {output['features'].shape}")
        print(f"Encoded shape: {output['encoded'].shape}")

    def get_model(self) -> UNet:
        return self.model

class Train:
    """
    Training loop implementation with validation and checkpointing.
    
    Args:
        data: Training dataset
        model: Model to train
        optimiser: Optimizer for training
        loss: Loss function
        epochs: Number of training epochs
        batch_size: Batch size for training
        model_directory: Directory to save model checkpoints
        val_data: Validation dataset (optional)
        patience: Number of epochs to wait for improvement before early stopping
        min_delta: Minimum change in validation loss to qualify as an improvement
        distributed: Whether to use distributed training
        local_rank: Local rank for distributed training
    """
    def __init__(self, data: SatelliteImages, model: nn.Module, optimiser: torch.optim.Optimizer, 
                 loss: nn.Module, epochs: int, batch_size: int, model_directory: str = model_directory, 
                 val_data: Optional[SatelliteImages] = None, patience: int = 4, min_delta: float = 0.001,
                 distributed: bool = False, local_rank: int = -1) -> None:
        self.distributed = distributed
        self.local_rank = local_rank
        
        if distributed:
            self.device = torch.device(f'cuda:{local_rank}')
            torch.cuda.set_device(self.device)
            dist.init_process_group(backend='nccl')
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.world_size = 1
            self.rank = 0
            
        self.model = model.to(self.device)
        
        if distributed:
            self.model = DDP(self.model, device_ids=[local_rank])
        elif torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)

        self.optimiser = optimiser
        self.loss = loss
        self.epochs = epochs
        self.train_loss, self.val_loss = [], []
        self.best_val_loss = float('inf')
        self.model_directory = model_directory
        
        self.patience = patience
        self.min_delta = min_delta
        self.patience_counter = 0
        self.best_epoch = 0
        
        if torch.cuda.is_available():
            self.scaler = GradScaler('cuda')
        else:
            self.scaler = None
            
        self.checkpoint = {'epoch': 0, 'model_state_dict': self.model.state_dict()}

        train_sampler = DistributedSampler(data) if distributed else None
        
        self.load_train = DataLoader(
            data,
            batch_size=batch_size,
            shuffle=(train_sampler is None),
            num_workers=config['data']['num_workers'],
            pin_memory=True,
            sampler=train_sampler
        )

        val_sampler = DistributedSampler(val_data) if distributed and val_data else None
        
        self.load_val = DataLoader(
            val_data,
            batch_size=batch_size,
            shuffle=False,
            num_workers=config['data']['num_workers'],
            pin_memory=True,
            sampler=val_sampler
        ) if val_data else None

        self.scheduler = OneCycleLR(optimiser, max_lr=0.01, epochs=epochs,
                                steps_per_epoch=len(self.load_train))
                                
        self.class_distribution = torch.zeros(7, device=self.device)
        self.update_class_distribution = True

    def update_class_weights(self, masks: torch.Tensor) -> None:
        """
        Update class weights based on the distribution of classes in the current batch.
        
        Args:
            masks: Tensor of class masks
        """
        if not self.update_class_distribution:
            return
            
        for i in range(7):
            self.class_distribution[i] += (masks == i).sum().item()
            
        if self.rank == 0 and len(self.train_loss) % 10 == 0:
            total = self.class_distribution.sum()
            if total > 0:
                weights = total / (7 * self.class_distribution + 1e-6)
                weights = weights / weights.sum() * 7
                
                if hasattr(self.loss, 'focal_loss') and hasattr(self.loss.focal_loss, 'class_weights'):
                    self.loss.focal_loss.class_weights = weights.to(self.device)

    def train_epoch(self, e: int) -> float:
        """
        Train for one epoch.
        
        Args:
            e: Current epoch number
            
        Returns:
            Average loss for the epoch
        """
        if self.distributed:
            self.load_train.sampler.set_epoch(e)
            
        self.model.train()
        total_loss = 0
        
        if self.rank == 0:
            pbar = tqdm(self.load_train, desc=f'Epoch {e+1}/{self.epochs}')
        else:
            pbar = self.load_train

        for i, (images, masks, _) in enumerate(pbar):
            images = images.to(self.device, non_blocking=True)
            masks = masks.to(self.device, non_blocking=True)
            
            self.update_class_weights(masks)

            if torch.cuda.is_available():
                with autocast('cuda'):
                    predictions = self.model(images)['segmentation']
                    loss = self.loss(predictions, masks)

                self.optimiser.zero_grad(set_to_none=True)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimiser)
                self.scaler.update()
            else:
                predictions = self.model(images)['segmentation']
                loss = self.loss(predictions, masks)

                self.optimiser.zero_grad(set_to_none=True)
                loss.backward()
                self.optimiser.step()
                
            self.scheduler.step()

            total_loss += loss.item()
            if self.rank == 0 and isinstance(pbar, tqdm):
                pbar.set_postfix({'loss': f'{total_loss/(i+1):.3f}'})

        if self.distributed:
            dist.all_reduce(torch.tensor(total_loss).to(self.device))
            total_loss = total_loss / self.world_size

        return total_loss/len(self.load_train)

    @torch.no_grad()
    def validate(self) -> Optional[float]:
        """
        Validate the model on the validation set.
        
        Returns:
            Average validation loss or None if no validation data
        """
        if not self.load_val:
            return None

        self.model.eval()
        total_loss = 0

        for images, masks, _ in self.load_val:
            images = images.to(self.device, non_blocking=True)
            masks = masks.to(self.device, non_blocking=True)

            if torch.cuda.is_available():
                with autocast():
                    predictions = self.model(images)['segmentation']
                    loss = self.loss(predictions, masks)
            else:
                predictions = self.model(images)['segmentation']
                loss = self.loss(predictions, masks)
                
            total_loss += loss.item()
            
        if self.distributed:
            dist.all_reduce(torch.tensor(total_loss).to(self.device))
            total_loss = total_loss / self.world_size

        return total_loss/len(self.load_val)

    def train(self) -> Dict[str, Any]:
        """
        Train the model with early stopping.
        
        Returns:
            Dictionary with training results
        """
        if self.rank == 0:
            os.makedirs(self.model_directory, exist_ok=True)
            versions = sorted([d for d in os.listdir(self.model_directory) if d.startswith('v_')])
            if not versions:version = 'v_0_01'
            else:
                latest = versions[-1].split('_')
                if len(latest) == 3:
                    major, minor = int(latest[1]), int(latest[2])
                    if self.epochs >= 30:
                        major += 1
                        minor = 1
                    else:
                        minor += 1
                        if minor > 99:
                            major += 1
                            minor = 1
                    version = f"v_{major}_{minor:02d}"
                else: version = 'v_0_01'
            
            checkpoint_dir = os.path.join(self.model_directory, version)
            os.makedirs(checkpoint_dir, exist_ok=True)
        else:
            version = None
            checkpoint_dir = None
            
        if self.distributed:
            version_tensor = torch.zeros(1, dtype=torch.long, device=self.device)
            if self.rank == 0:
                version_tensor[0] = int(version.split('_')[1]) * 100 + int(version.split('_')[2])
            dist.broadcast(version_tensor, 0)
            version = f"v_{version_tensor[0]//100}_{version_tensor[0]%100:02d}"
            checkpoint_dir = os.path.join(self.model_directory, version)
            
        start_time = time.time()
        
        checkpoint = {
            'epoch': 0,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimiser.state_dict(),
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            'train_loss': 0,
            'val_loss': None
        }

        best_checkpoint = checkpoint.copy()
        best_checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pt')
        
        if self.rank == 0:
            torch.save(best_checkpoint, best_checkpoint_path)

        try:
            for epoch in range(self.epochs):
                train_loss = self.train_epoch(epoch)
                val_loss = self.validate()
                
                if self.rank == 0:
                    self.train_loss.append(train_loss)
                    if val_loss: self.val_loss.append(val_loss)

                    checkpoint = {
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimiser.state_dict(),
                        'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
                        'train_loss': train_loss,
                        'val_loss': val_loss
                    }

                    if epoch % 5 == 0 or epoch == self.epochs - 1:
                        torch.save(checkpoint, f"{checkpoint_dir}/epoch_{epoch+1}.pt")
                    
                    if val_loss and val_loss < self.best_val_loss - self.min_delta:
                        self.best_val_loss = val_loss
                        self.best_epoch = epoch
                        self.patience_counter = 0
                        best_checkpoint = checkpoint.copy()
                        torch.save(best_checkpoint, best_checkpoint_path)
                    else:
                        self.patience_counter += 1
                    
                    if self.patience_counter >= self.patience:
                        print(f"Early stopping triggered after {epoch+1} epochs. No improvement in validation loss for {self.patience} epochs.")
                        print(f"Best validation loss: {self.best_val_loss:.6f} at epoch {self.best_epoch+1}")
                        
                        self.model.load_state_dict(best_checkpoint['model_state_dict'])
                        
                        torch.save(best_checkpoint, os.path.join(checkpoint_dir, 'model.pt'))
                        break
                
                if self.distributed:
                    early_stop_tensor = torch.zeros(1, dtype=torch.long, device=self.device)
                    if self.rank == 0:
                        early_stop_tensor[0] = 1 if self.patience_counter >= self.patience else 0
                    dist.broadcast(early_stop_tensor, 0)
                    if early_stop_tensor[0] == 1:
                        break

        except Exception as e:
            print(f"Training interrupted: {str(e)}")
            if self.rank == 0:
                torch.save(checkpoint, f"{checkpoint_dir}/emergency_checkpoint.pt")
            raise e

        if self.rank == 0:
            if self.patience_counter < self.patience:
                torch.save(checkpoint, os.path.join(checkpoint_dir, 'model.pt'))
                
            best_val_loss_str = 'Infinity' if self.best_val_loss == float('inf') else self.best_val_loss
                
            return {'train_losses': self.train_loss, 'val_losses': self.val_loss,
                    'training_time': time.time() - start_time, 'version': version,
                    'best_epoch': self.best_epoch, 'best_val_loss': best_val_loss_str,
                    'early_stopped': self.patience_counter >= self.patience}
        else:
            return None

class FocalLoss(nn.Module):
    """
    Focal Loss implementation that focuses on hard examples with class weighting.
    
    Args:
        gamma: Focusing parameter (higher values focus more on hard examples)
        reduction: Reduction method ('mean', 'sum', or 'none')
        class_weights: Optional tensor of class weights to handle class imbalance
        temperature: Temperature scaling factor for logits
        entropy_weight: Weight for entropy regularization to encourage more decisive predictions
    """
    def __init__(self, gamma: float = 2.0, reduction: str = 'mean', class_weights: Optional[torch.Tensor] = None, 
                 temperature: float = 1.2, entropy_weight: float = 0.1) -> None:
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.class_weights = class_weights
        self.temperature = temperature
        self.entropy_weight = entropy_weight

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if input.dim() > 2:
            input = input.permute(0, 2, 3, 1).contiguous()
            input = input.view(-1, input.size(-1))

        input = input / self.temperature

        target = target.view(-1)
        logpt = F.log_softmax(input, dim=-1)
        logpt = logpt.gather(1, target.view(-1, 1))
        logpt = logpt.view(-1)
        pt = torch.exp(logpt)

        if self.class_weights is not None:
            weights = self.class_weights[target]
            focal_loss = -1 * weights * ((1 - pt) ** self.gamma) * logpt
        else:
            focal_loss = -1 * ((1 - pt) ** self.gamma) * logpt
            
        if self.entropy_weight > 0:
            probs = F.softmax(input, dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
            entropy_loss = self.entropy_weight * entropy.mean()
            focal_loss = focal_loss.mean() + entropy_loss
            return focal_loss

        if self.reduction == "none":
            return focal_loss
        elif self.reduction == "mean":
            return focal_loss.mean()
        else:
            return focal_loss.sum()

class SpatialConsistencyLoss(nn.Module):
    """
    Simplified spatial consistency loss that balances between smoothing and detail preservation.
    """
    def __init__(self, weight: float = 0.1, focal_loss: Optional[FocalLoss] = None) -> None:
        super().__init__()
        self.weight = weight
        self.focal_loss = focal_loss if focal_loss is not None else FocalLoss()

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        focal_loss = self.focal_loss(input, target)
        
        pred = F.softmax(input, dim=1)
        
        if pred.shape[2:] != target.shape[1:]:
            target = F.interpolate(
                target.unsqueeze(1).float(), 
                size=pred.shape[2:],
                mode='nearest'
            ).squeeze(1).long()
        
        target = target.float()
        
        batch_spatial_loss = 0
        
        shifts = [(i, j) for i in range(-1, 2) for j in range(-1, 2) if not (i == 0 and j == 0)]
        
        batch_size, _, height, width = pred.shape
        
        for shift_h, shift_w in shifts:
            pred_shifted = torch.roll(pred, shifts=(shift_h, shift_w), dims=(2, 3))
            target_shifted = torch.roll(target, shifts=(shift_h, shift_w), dims=(1, 2))
            
            mask = (target == target_shifted).float()
            mask = mask.unsqueeze(1)
            
            loss = F.mse_loss(pred, pred_shifted, reduction='none')
            loss = loss.mean(dim=1, keepdim=True)
            
            batch_spatial_loss += (loss * mask).mean()
        
        spatial_loss = batch_spatial_loss / 8
        
        return focal_loss + self.weight * spatial_loss

def train_model(epochs: int = 5, 
                distributed: bool = False, 
                local_rank: int = -1, 
                model_params: Optional[Dict[str, Any]] = None, 
                loss_params: Optional[Dict[str, Any]] = None, 
                optimizer_params: Optional[Dict[str, Any]] = None, 
                training_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Convenience function to train a model with customizable parameters.
    
    Args:
        epochs: Number of training epochs
        distributed: Whether to use distributed training
        local_rank: Local rank for distributed training
        model_params: Dictionary of model parameters
        loss_params: Dictionary of loss function parameters
        optimizer_params: Dictionary of optimizer parameters
        training_params: Dictionary of training parameters
        
    Returns:
        Dictionary with training results
    """
    model_params = model_params or {}
    loss_params = loss_params or {}
    optimizer_params = optimizer_params or {}
    training_params = training_params or {}
    
    n_classes = model_params.get('n_classes', 7)
    input_dim = model_params.get('input_dim', 3)
    dropout = model_params.get('dropout', 0.2)
    confidence_threshold = model_params.get('confidence_threshold', 0.5)
    
    gamma = loss_params.get('gamma', 3.0)
    class_weights = loss_params.get('class_weights', None)
    temperature = loss_params.get('temperature', 1.5)
    entropy_weight = loss_params.get('entropy_weight', 0.2)
    spatial_weight = loss_params.get('spatial_weight', 0.15)
    
    lr = optimizer_params.get('lr', 0.003)
    weight_decay = optimizer_params.get('weight_decay', 0.03)
    
    batch_size = training_params.get('batch_size', 16)
    patience = training_params.get('patience', 6)
    min_delta = training_params.get('min_delta', 0.0005)
    use_validation = training_params.get('use_validation', False)
    
    config = UNetConfig(n_classes=n_classes, input_dim=input_dim, dropout=dropout)
    model = ModelInit(config, confidence_threshold).get_model()
    
    training_data = SatelliteImages(os.path.join(base_directory, 'data', 'train'), transform=transformer)
    
    if class_weights is None:
        class_weights = torch.ones(7)
        class_weights[1] = 0.3
        class_weights[0] = 1.2
        class_weights[2] = 1.0
        class_weights[3] = 1.2
    
    focal_loss = FocalLoss(
        gamma=gamma, 
        class_weights=class_weights, 
        temperature=temperature,
        entropy_weight=entropy_weight
    )
    
    spatial_loss = SpatialConsistencyLoss(weight=spatial_weight, focal_loss=focal_loss)
    
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    trainer = Train(
        model=model, 
        data=training_data, 
        optimiser=optimiser, 
        loss=spatial_loss,
        epochs=epochs, 
        batch_size=batch_size,
        distributed=distributed,
        local_rank=local_rank,
        patience=patience,
        min_delta=min_delta
    )
    
    return trainer.train()

def grid_search(test_image_path: str = 'data/train/119_sat.jpg', epochs: int = 10) -> List[Dict[str, Any]]:
    """
    Perform grid search over key model parameters and visualize results.
    
    Args:
        test_image_path: Path to test image for visualization
        epochs: Number of epochs to train each model
        
    Returns:
        List of dictionaries containing results for each parameter combination
    """
    param_grid = {
        'confidence_threshold': [0.3, 0.5],
        'entropy_weight': [0.05, 0.1],
        'class_weights': [
            {'agricultural': 0.7, 'urban': 1.2, 'rangeland': 1.0, 'forest': 1.2},
            {'agricultural': 0.3, 'urban': 1.5, 'rangeland': 1.2, 'forest': 1.5},
        ]
    }
    
    results = []
    
    n_combinations = (len(param_grid['confidence_threshold']) * 
                     len(param_grid['entropy_weight']) * 
                     len(param_grid['class_weights']))
    
    fig_rows = int(np.ceil(np.sqrt(n_combinations)))
    fig_cols = int(np.ceil(n_combinations / fig_rows))
    fig = plt.figure(figsize=(5*fig_cols, 5*fig_rows))
    
    combination_idx = 0
    
    for conf_thresh in param_grid['confidence_threshold']:
        for ent_weight in param_grid['entropy_weight']:
            for class_weight_dict in param_grid['class_weights']:
                class_weights = torch.ones(7)
                class_weights[1] = class_weight_dict['agricultural']
                class_weights[0] = class_weight_dict['urban']
                class_weights[2] = class_weight_dict['rangeland']
                class_weights[3] = class_weight_dict['forest']
                
                model_params = {
                    'n_classes': 7,
                    'input_dim': 3,
                    'dropout': 0.2,
                    'confidence_threshold': conf_thresh
                }
                
                loss_params = {
                    'gamma': 3.0,
                    'temperature': 1.5,
                    'entropy_weight': ent_weight,
                    'spatial_weight': 0.15,
                    'class_weights': class_weights
                }
                
                optimizer_params = {
                    'lr': 0.003,
                    'weight_decay': 0.03
                }
                
                training_params = {
                    'batch_size': 16,
                    'patience': 6,
                    'min_delta': 0.0005,
                    'use_validation': True
                }
                
                training_results = train_model(
                    epochs=epochs,
                    model_params=model_params,
                    loss_params=loss_params,
                    optimizer_params=optimizer_params,
                    training_params=training_params
                )
                
                predictor = RegionPredictor(model_version=training_results['version'])
                prediction_result = predictor.predict_from_tensor(
                    predictor.tensor_from_file(test_image_path)
                )
                proportions = predictor.get_land_proportions(prediction_result)
                
                results.append({
                    'params': {
                        'confidence_threshold': conf_thresh,
                        'entropy_weight': ent_weight,
                        'class_weights': class_weight_dict
                    },
                    'training_results': training_results,
                    'land_proportions': proportions,
                    'model_version': training_results['version']
                })
                
                plt.subplot(fig_rows, fig_cols, combination_idx + 1)
                plt.imshow(prediction_result['colored_mask'])
                plt.title(f"CT:{conf_thresh}\nEW:{ent_weight}\nAg:{class_weight_dict['agricultural']}")
                plt.axis('off')
                
                combination_idx += 1
    
    plt.tight_layout()
    plt.savefig(os.path.join(base_directory, 'data', 'grid_search_results.png'))
    plt.close()
    
    import json
    results_path = os.path.join(base_directory, 'data', 'grid_search_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    return results

if __name__ == "__main__":
    grid_search(test_image_path='data/train/119_sat.jpg', 
                epochs=3)