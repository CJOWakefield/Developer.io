# Experimental trainer with enhanced model architecture and loss functions
# Based on trainer.py but with experimental improvements

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
import signal
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
import gc

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

# Default image transformation with enhanced augmentation
transformer_exp = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((256, 256), antialias=False),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define signal handler to catch termination signals
def signal_handler(sig, frame):
    print(f"Received signal {sig}, exiting gracefully...")
    sys.exit(0)

# Register the signal handler
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

''' ----- Experimental Trainer file summary -----

> MultiScaleAttention - Enhanced attention mechanism with multi-scale feature fusion
> DenseResidualBlock - Residual block with dense connections for better gradient flow
> UNetConfigExp - Configuration class for experimental UNet model parameters
> UNetExp - Enhanced U-Net architecture with multi-scale feature fusion
> ModelInitExp - Helper class for initializing and examining experimental model parameters
> TrainExp - Training loop implementation with validation and checkpointing
> FocalLossExp - Enhanced loss function with adaptive class weighting
> SemanticBoundaryLoss - Loss function that focuses on semantic boundaries
> TopologyAwareLoss - Loss function that considers topological relationships
> ContrastiveLoss - Contrastive learning component for better feature discrimination

'''

class MultiScaleAttention(nn.Module):
    """
    Enhanced multi-scale spatial attention module with feature pyramid fusion.
    
    Parameters:
        kernel_sizes: List of kernel sizes for multi-scale convolutions
        dropout_rate: Probability of dropout for regularization
        min_scale: Minimum scale value for attention scaling
        max_scale: Maximum scale value for attention scaling
        use_channel_attention: Whether to apply channel-wise attention
        in_channels: Number of input channels
    """
    def __init__(self, kernel_sizes: List[int] = [3, 5], dropout_rate: float = 0.1, 
                 min_scale: float = 0.2, max_scale: float = 1.0, use_channel_attention: bool = False,
                 in_channels: int = 64) -> None:
        super().__init__()
        self.kernel_sizes = kernel_sizes
        self.convs = nn.ModuleList([
            nn.Conv2d(2, 1, kernel_size=k, padding=k//2)
            for k in kernel_sizes
        ])
        self.sigmoid = nn.Sigmoid()
        self.fusion = nn.Conv2d(len(kernel_sizes), 1, kernel_size=1)
        self.dropout = nn.Dropout2d(p=dropout_rate)
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.residual_weight = 0.7
        self.in_channels = in_channels
        
        self.use_channel_attention = use_channel_attention
        if use_channel_attention:
            hidden_dim = max(1, in_channels // 2)
            self.channel_attention = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(1, hidden_dim, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dim, in_channels, kernel_size=1),
                nn.Sigmoid()
            )
            self.channel_reduction = nn.Conv2d(in_channels, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the multi-scale attention module.
        
        Parameters:
            x: Input tensor of shape [B, C, H, W]
            
        Returns:
            Tensor with applied spatial and optional channel attention
        """
        attention_maps = []
        for conv in self.convs:
            avg_out = torch.mean(x, dim=1, keepdim=True)
            max_out, _ = torch.max(x, dim=1, keepdim=True)
            x_cat = torch.cat([avg_out, max_out], dim=1)
            attention = self.sigmoid(conv(x_cat))
            attention_maps.append(attention)
        
        fused_attention = self.fusion(torch.cat(attention_maps, dim=1))
        fused_attention = self.dropout(fused_attention)
        fused_attention = self.min_scale + (self.max_scale - self.min_scale) * fused_attention
        
        spatial_output = x * fused_attention
        
        if self.use_channel_attention:
            if x.size(1) != self.in_channels:
                self.channel_reduction = nn.Conv2d(x.size(1), 1, kernel_size=1).to(x.device)
            
            reduced_channels = self.channel_reduction(spatial_output)
            channel_weights = self.channel_attention(reduced_channels)
            channel_weights = channel_weights.expand_as(spatial_output)
            spatial_output = spatial_output * channel_weights
        
        return spatial_output + self.residual_weight * x

class DenseResidualBlock(nn.Module):
    """
    Residual block with dense connections for better gradient flow.
    
    Parameters:
        input_dim: Number of input channels
        output_dim: Number of output channels
        dropout: Dropout probability
        use_attention: Whether to apply spatial attention
        growth_rate: Number of channels to add in each dense connection
        concat_channels: Number of additional channels from concatenation
    """
    def __init__(self, input_dim: int, output_dim: int, dropout: float, use_attention: bool = True, 
                 growth_rate: int = 32, concat_channels: int = 0) -> None:
        super().__init__()
        
        self.growth_rate = growth_rate
        self.use_attention = use_attention
        
        actual_input_dim = input_dim + concat_channels
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(actual_input_dim, growth_rate, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(growth_rate),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Dropout2d(p=dropout)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(actual_input_dim + growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(growth_rate),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Dropout2d(p=dropout)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(actual_input_dim + 2 * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(growth_rate),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Dropout2d(p=dropout)
        )
        
        self.conv_final = nn.Sequential(
            nn.Conv2d(actual_input_dim + 3 * growth_rate, output_dim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(output_dim),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )
        
        if use_attention:
            self.attention = MultiScaleAttention(in_channels=output_dim)
        else:
            self.attention = nn.Identity()
            
        self.skip = (nn.Sequential(
            nn.Conv2d(actual_input_dim, output_dim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(output_dim),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        ) if actual_input_dim != output_dim else nn.Identity())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the dense residual block.
        
        Parameters:
            x: Input tensor
            
        Returns:
            Processed feature map with dense connections and skip connection
        """
        out1 = self.conv1(x)
        cat1 = torch.cat([x, out1], dim=1)
        
        out2 = self.conv2(cat1)
        cat2 = torch.cat([cat1, out2], dim=1)
        
        out3 = self.conv3(cat2)
        cat3 = torch.cat([cat2, out3], dim=1)
        
        out = self.conv_final(cat3)
        out = self.attention(out)
        
        return out + self.skip(x)

class UNetConfigExp():
    """
    Configuration class for experimental UNet model parameters.
    
    Parameters:
        n_classes: Number of output classes for segmentation
        input_dim: Number of input channels
        output_dim: Number of output channels
        n_features: Base number of features
        dropout: Dropout probability
        use_dense_connections: Whether to use dense connections
        use_multi_scale_attention: Whether to use multi-scale attention
        use_channel_attention: Whether to use channel attention
        use_attention: Whether to use attention in residual blocks
        confidence_threshold: Threshold for confidence in predictions
    """
    def __init__(self, n_classes: int = 7, input_dim: int = 3, output_dim: int = 1, 
                 n_features: int = 16, dropout: float = 0.2, use_dense_connections: bool = True,
                 use_multi_scale_attention: bool = True, use_channel_attention: bool = True,
                 use_attention: bool = True, confidence_threshold: float = 0.3) -> None:
        self.n_classes = n_classes
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_features = n_features
        self.dropout = dropout
        self.use_dense_connections = use_dense_connections
        self.use_multi_scale_attention = use_multi_scale_attention
        self.use_channel_attention = use_channel_attention
        self.use_attention = use_attention
        self.confidence_threshold = confidence_threshold

class UNetExp(nn.Module):
    """
    Enhanced U-Net architecture with multi-scale feature fusion for semantic segmentation.
    
    Parameters:
        config: UNetConfigExp object with model parameters
    """
    def __init__(self, config: UNetConfigExp) -> None:
        super().__init__()
        
        self.encoder_channels = [config.input_dim, 64, 128, 256, 512]
        self.decoder_channels = [256, 128, 64, 32, config.n_classes]
        self.fpn_channels = 256
        
        self.enc1 = ResidualBlock(self.encoder_channels[0], self.encoder_channels[1], config.dropout, config.use_attention)
        self.enc2 = ResidualBlock(self.encoder_channels[1], self.encoder_channels[2], config.dropout, config.use_attention)
        self.enc3 = ResidualBlock(self.encoder_channels[2], self.encoder_channels[3], config.dropout, config.use_attention)
        self.enc4 = ResidualBlock(self.encoder_channels[3], self.encoder_channels[4], config.dropout, config.use_attention)
        
        self.fpn = FeaturePyramidNetwork(self.encoder_channels[4], self.fpn_channels, config.dropout)
        
        self.dec4 = DenseResidualBlock(self.encoder_channels[4], self.decoder_channels[0], config.dropout, config.use_attention, concat_channels=self.encoder_channels[3])
        self.dec3 = DenseResidualBlock(self.decoder_channels[0], self.decoder_channels[1], config.dropout, config.use_attention, concat_channels=self.encoder_channels[2])
        self.dec2 = DenseResidualBlock(self.decoder_channels[1], self.decoder_channels[2], config.dropout, config.use_attention, concat_channels=self.encoder_channels[1])
        
        self.dec1_input_adjust = nn.Conv2d(self.decoder_channels[2], self.decoder_channels[2], kernel_size=1, stride=1, padding=0, bias=False)
        self.dec1 = ResidualBlock(self.decoder_channels[2], self.decoder_channels[3], config.dropout, config.use_attention, concat_channels=0)
        
        self.final = nn.Conv2d(self.decoder_channels[3], self.decoder_channels[4], kernel_size=1, stride=1, padding=0)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        self.attention = MultiScaleAttention(
            kernel_sizes=[3, 5, 7],
            dropout_rate=config.dropout,
            in_channels=self.decoder_channels[3]
        )
        
        self.confidence_threshold = config.confidence_threshold

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the UNet model.
        
        Parameters:
            x: Input tensor of shape [B, C, H, W]
            
        Returns:
            Dictionary containing logits and feature maps at different stages
        """
        if x.dim() != 4:
            raise ValueError(f"Expected 4D input (batch, channels, height, width), got {x.dim()}D")

        e1 = self.enc1(x)
        x1 = self.pool(e1)

        e2 = self.enc2(x1)
        x2 = self.pool(e2)

        e3 = self.enc3(x2)
        x3 = self.pool(e3)

        c_final = self.enc4(x3)

        fpn_features = self.fpn(c_final)

        d4 = self.dec4(torch.cat([self.upsample(c_final), e3], dim=1))
        d3 = self.dec3(torch.cat([self.upsample(d4), e2], dim=1))
        d2 = self.dec2(torch.cat([self.upsample(d3), e1], dim=1))
        
        d2_adjusted = self.dec1_input_adjust(d2)
        d1 = self.dec1(d2_adjusted)

        attended_features = self.attention(d1)

        logits = self.final(attended_features)

        if not self.training and self.confidence_threshold > 0:
            probs = torch.sigmoid(logits)
            # Apply a softer threshold to allow more diverse predictions
            logits = (probs > self.confidence_threshold).float() * probs

        return {
            'logits': logits,
            'features': {
                'encoder': [e1, e2, e3, c_final],
                'decoder': [d1, d2, d3, d4],
                'fpn': fpn_features
            }
        }

class ModelInitExp:
    """
    Helper class for initializing and examining experimental model parameters.
    
    Parameters:
        config: UNetConfigExp object with model parameters
        confidence_threshold: Threshold for confidence in predictions
    """
    def __init__(self, config: UNetConfigExp = None, confidence_threshold: float = 0.5) -> None:
        self.config = config if config else UNetConfigExp()
        self.model = UNetExp(self.config)

    def model_params(self, input_dim: Tuple[int, int, int, int] = (1, 3, 256, 256)) -> None:
        """
        Print model parameter statistics and output shapes.
        
        Parameters:
            input_dim: Input dimensions [batch_size, channels, height, width]
        """
        x = torch.randn(*input_dim)
        output = self.model(x)
        print(f'Trainable params: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}')
        print(f"Segmentation shape: {output['segmentation'].shape}")
        print(f"Feature shape: {output['features'].shape}")
        print(f"Encoded shape: {output['encoded'].shape}")
        print(f"Pyramid features shapes: {[f.shape for f in output['pyramid_features']]}")

    def get_model(self) -> UNetExp:
        """
        Return the initialized model.
        
        Returns:
            Initialized UNetExp model
        """
        return self.model

class FocalLossExp(nn.Module):
    """
    Enhanced Focal Loss implementation with adaptive class weighting.
    
    Parameters:
        gamma: Focusing parameter (higher values focus more on hard examples)
        reduction: Reduction method ('mean', 'sum', or 'none')
        class_weights: Optional tensor of class weights to handle class imbalance
        temperature: Temperature scaling factor for logits
        entropy_weight: Weight for entropy regularization to encourage decisive predictions
        adaptive_gamma: Whether to use adaptive gamma values for different classes
    """
    def __init__(self, gamma: float = 2.0, reduction: str = 'mean', class_weights: Optional[torch.Tensor] = None, 
                 temperature: float = 1.5, entropy_weight: float = 0.1, adaptive_gamma: bool = True) -> None:
        super(FocalLossExp, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.class_weights = class_weights
        self.temperature = temperature
        self.entropy_weight = entropy_weight
        self.adaptive_gamma = adaptive_gamma
        
        if adaptive_gamma and class_weights is not None:
            self.adaptive_gamma_values = gamma * (class_weights / class_weights.mean())
        else:
            self.adaptive_gamma_values = None

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the focal loss.
        
        Parameters:
            input: Predicted logits
            target: Ground truth labels
            
        Returns:
            Computed loss value
        """
        if input.dim() > 2:
            batch_size, channels, height, width = input.shape
            
            if target.shape[-2:] != (height, width):
                target = F.interpolate(
                    target.unsqueeze(1).float(), 
                    size=(height, width),
                    mode='nearest'
                ).squeeze(1).long()
            
            input = input.permute(0, 2, 3, 1).contiguous()
            input = input.view(-1, channels)
            
            if target.dim() == 3:
                target = target.view(-1)
            else:
                target = target.contiguous()

        input = input / self.temperature
        target = target.long()
        
        n_classes = input.size(1)
        if target.max() >= n_classes or target.min() < 0:
            target = torch.clamp(target, 0, n_classes - 1)
        
        logpt = F.log_softmax(input, dim=-1)
        logpt = logpt.gather(1, target.view(-1, 1))
        logpt = logpt.view(-1)
        pt = torch.exp(logpt)

        if self.class_weights is not None:
            weights = self.class_weights.to(target.device)
            
            if self.adaptive_gamma and self.adaptive_gamma_values is not None:
                gamma_values = self.adaptive_gamma_values.to(target.device)
                focal_loss = -1 * weights[target] * ((1 - pt) ** gamma_values[target]) * logpt
            else:
                focal_loss = -1 * weights[target] * ((1 - pt) ** self.gamma) * logpt
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

class SemanticBoundaryLoss(nn.Module):
    """
    Loss function that focuses on semantic boundaries between land types.
    
    Parameters:
        weight: Weight for the boundary loss component
        focal_loss: Base focal loss function
        boundary_dilation: Number of pixels to dilate the boundary
    """
    def __init__(self, weight: float = 0.2, focal_loss: Optional[FocalLossExp] = None, boundary_dilation: int = 1) -> None:
        super().__init__()
        self.weight = weight
        self.focal_loss = focal_loss if focal_loss is not None else FocalLossExp()
        self.boundary_dilation = boundary_dilation
        
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the semantic boundary loss.
        
        Parameters:
            input: Predicted logits
            target: Ground truth labels
            
        Returns:
            Combined focal loss and boundary-aware loss
        """
        n_classes = input.size(1)
        if target.max() >= n_classes or target.min() < 0:
            target = torch.clamp(target, 0, n_classes - 1)
            
        focal_loss = self.focal_loss(input, target)
        
        pred = F.softmax(input, dim=1)
        
        if pred.shape[2:] != target.shape[1:]:
            target = F.interpolate(
                target.unsqueeze(1).float(), 
                size=pred.shape[2:],
                mode='nearest'
            ).squeeze(1).long()
        
        target_boundary = torch.zeros_like(target, dtype=torch.float)
        
        h, w = target.shape[1], target.shape[2]
        
        target_up = torch.roll(target, shifts=-1, dims=1)
        target_down = torch.roll(target, shifts=1, dims=1)
        target_left = torch.roll(target, shifts=-1, dims=2)
        target_right = torch.roll(target, shifts=1, dims=2)
        
        target_up[:, -1, :] = target[:, -1, :]
        target_down[:, 0, :] = target[:, 0, :]
        target_left[:, :, -1] = target[:, :, -1]
        target_right[:, :, 0] = target[:, :, 0]
        
        diff_up = (target != target_up).float()
        diff_down = (target != target_down).float()
        diff_left = (target != target_left).float()
        diff_right = (target != target_right).float()
        
        boundary_mask = (diff_up + diff_down + diff_left + diff_right) > 0
        
        if self.boundary_dilation > 0:
            kernel_size = 2 * self.boundary_dilation + 1
            kernel = torch.ones(1, 1, kernel_size, kernel_size, device=target.device)
            
            boundary_mask = boundary_mask.unsqueeze(1).float()
            dilated_boundary = F.max_pool2d(
                boundary_mask, 
                kernel_size=kernel_size, 
                stride=1, 
                padding=self.boundary_dilation
            )
            dilated_boundary = dilated_boundary.squeeze(1)
        
        boundary_weight = 1.0 + self.weight * dilated_boundary.unsqueeze(1)
        
        ce_loss = F.cross_entropy(input, target, reduction='none')
        ce_loss = ce_loss * boundary_weight.squeeze(1)
        
        return focal_loss + ce_loss.mean()

class TopologyAwareLoss(nn.Module):
    """
    Loss function that considers topological relationships between land types.
    
    Parameters:
        weight: Weight for the topology loss component
        focal_loss: Base focal loss function
        neighborhood_size: Size of the neighborhood to consider for topology
    """
    def __init__(self, weight: float = 0.15, focal_loss: Optional[FocalLossExp] = None, neighborhood_size: int = 3) -> None:
        super().__init__()
        self.weight = weight
        self.focal_loss = focal_loss if focal_loss is not None else FocalLossExp()
        self.neighborhood_size = neighborhood_size
        
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the topology-aware loss.
        
        Parameters:
            input: Predicted logits
            target: Ground truth labels
            
        Returns:
            Combined focal loss and topology-aware loss
        """
        n_classes = input.size(1)
        if target.max() >= n_classes or target.min() < 0:
            target = torch.clamp(target, 0, n_classes - 1)
        
        focal_loss = self.focal_loss(input, target)
        
        pred = F.softmax(input, dim=1)
        
        if pred.shape[2:] != target.shape[1:]:
            target = F.interpolate(
                target.unsqueeze(1).float(), 
                size=pred.shape[2:],
                mode='nearest'
            ).squeeze(1).long()
        
        topology_loss = 0
        
        half_size = self.neighborhood_size // 2
        
        h, w = target.shape[1], target.shape[2]
        if h <= 2*half_size or w <= 2*half_size:
            return focal_loss
        
        target_up = torch.roll(target, shifts=-1, dims=1)
        target_down = torch.roll(target, shifts=1, dims=1)
        target_left = torch.roll(target, shifts=-1, dims=2)
        target_right = torch.roll(target, shifts=1, dims=2)
        
        target_up[:, -1, :] = target[:, -1, :]
        target_down[:, 0, :] = target[:, 0, :]
        target_left[:, :, -1] = target[:, :, -1]
        target_right[:, :, 0] = target[:, :, 0]
        
        neighborhood = torch.stack([target_up, target_down, target_left, target_right], dim=1)
        
        class_counts = torch.zeros((target.shape[0], target.shape[1], target.shape[2], n_classes), 
                                  device=target.device)
        
        for b in range(target.shape[0]):
            for c in range(n_classes):
                class_counts[b, :, :, c] = (neighborhood[b] == c).sum(dim=0)
        
        class_counts = class_counts / 4.0
        
        class_counts = class_counts.view(-1, n_classes)
        center_probs = pred.permute(0, 2, 3, 1).contiguous().view(-1, n_classes)
        
        batch_size = 10000
        topology_loss = 0
        
        for i in range(0, center_probs.size(0), batch_size):
            end_idx = min(i + batch_size, center_probs.size(0))
            batch_probs = center_probs[i:end_idx]
            batch_counts = class_counts[i:end_idx]
            
            batch_loss = F.kl_div(
                F.log_softmax(batch_probs, dim=1),
                batch_counts,
                reduction='batchmean'
            )
            
            topology_loss += batch_loss * (end_idx - i)
        
        topology_loss = topology_loss / center_probs.size(0)
        
        return focal_loss + self.weight * topology_loss

class ContrastiveLoss(nn.Module):
    """
    Contrastive loss to improve feature representation quality.
    
    Parameters:
        weight: Weight for the contrastive loss component
        temperature: Temperature scaling factor for similarity calculation
        margin: Margin for contrastive loss calculation
    """
    def __init__(self, weight: float = 0.1, temperature: float = 0.07, margin: float = 1.0) -> None:
        super().__init__()
        self.weight = weight
        self.temperature = temperature
        self.margin = margin
        
    def forward(self, features: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the contrastive loss.
        
        Parameters:
            features: Feature embeddings from the network
            target: Ground truth labels
            
        Returns:
            Computed contrastive loss value
        """
        if features.numel() > 10_000_000:
            return self.weight * torch.tensor(0.1, device=features.device)
            
        if features.size(1) > 0:
            n_classes = features.size(1)
            if target.max() >= n_classes or target.min() < 0:
                target = torch.clamp(target, 0, n_classes - 1)
        
        batch_size = min(features.size(0), 4)
        if features.size(0) > batch_size:
            indices = torch.randperm(features.size(0), device=features.device)[:batch_size]
            features = features[indices]
            target = target[indices]
        
        if features.dim() > 3 and features.size(2) * features.size(3) > 1024:
            max_size = 32
            if features.size(2) > max_size or features.size(3) > max_size:
                features = F.interpolate(
                    features, 
                    size=(max_size, max_size),
                    mode='bilinear',
                    align_corners=False
                )
                if target.dim() == 3:
                    target = F.interpolate(
                        target.float().unsqueeze(1),
                        size=(max_size, max_size),
                        mode='nearest'
                    ).squeeze(1).long()
                elif target.dim() == 4:
                    target = F.interpolate(
                        target.float(),
                        size=(max_size, max_size),
                        mode='nearest'
                    ).long()
        
        if features.dim() > 3:
            batch_size, channels, height, width = features.shape
            features = features.view(batch_size, channels, -1).permute(0, 2, 1)
            
            if target.dim() == 3:
                target = target.view(batch_size, -1)
            elif target.dim() == 4:
                target = target.argmax(dim=1).view(batch_size, -1)
        
        total_loss = 0
        num_pairs = 0
        
        max_pixels = 1000
        
        for b in range(features.size(0)):
            if features.size(1) > max_pixels:
                indices = torch.randperm(features.size(1), device=features.device)[:max_pixels]
                curr_features = features[b, indices]
                curr_targets = target[b, indices]
            else:
                curr_features = features[b]
                curr_targets = target[b]
            
            curr_features = F.normalize(curr_features, dim=1)
            
            similarities = torch.mm(curr_features, curr_features.t()) / self.temperature
            
            pos_mask = (curr_targets.unsqueeze(0) == curr_targets.unsqueeze(1)).float()
            identity_mask = torch.eye(pos_mask.shape[0], device=pos_mask.device)
            pos_mask = pos_mask * (1 - identity_mask)
            neg_mask = 1 - pos_mask - identity_mask
            
            if pos_mask.sum() > 0:
                pos_similarities = similarities * pos_mask
                pos_loss = -torch.log(torch.exp(pos_similarities) / 
                          (torch.exp(pos_similarities) + 
                           torch.sum(torch.exp(similarities) * neg_mask, dim=1, keepdim=True)))
                pos_loss = pos_loss.sum() / (pos_mask.sum() + 1e-6)
                total_loss += pos_loss
                num_pairs += 1
        
        if num_pairs > 0:
            return self.weight * (total_loss / num_pairs)
        else:
            return self.weight * torch.tensor(0.1, device=features.device)

class ResidualBlock(nn.Module):
    """
    Residual block with skip connections for better gradient flow.
    
    Parameters:
        input_dim: Number of input channels
        output_dim: Number of output channels
        dropout: Dropout probability
        use_attention: Whether to apply spatial attention
        concat_channels: Number of additional channels from concatenation
    """
    def __init__(self, input_dim: int, output_dim: int, dropout: float, use_attention: bool = True, concat_channels: int = 0) -> None:
        super().__init__()
        
        self.use_attention = use_attention
        
        actual_input_dim = input_dim + concat_channels
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(actual_input_dim, output_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(output_dim),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Dropout2d(p=dropout)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(output_dim, output_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(output_dim),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Dropout2d(p=dropout)
        )
        
        if use_attention:
            self.attention = None
        else:
            self.attention = nn.Identity()
            
        self.skip = (nn.Sequential(
            nn.Conv2d(actual_input_dim, output_dim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(output_dim),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        ) if actual_input_dim != output_dim else nn.Identity())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the residual block.
        
        Parameters:
            x: Input tensor
            
        Returns:
            Processed feature map with residual connection
        """
        out = self.conv1(x)
        out = self.conv2(out)
        
        if self.use_attention and self.attention is None:
            self.attention = MultiScaleAttention(in_channels=out.size(1)).to(out.device)
        
        if self.use_attention:
            out = self.attention(out)
        
        return out + self.skip(x)

class FeaturePyramidNetwork(nn.Module):
    """
    Feature Pyramid Network for multi-scale feature extraction.
    
    Parameters:
        in_channels: Number of input channels
        out_channels: Number of output channels
        dropout: Dropout probability
    """
    def __init__(self, in_channels: int, out_channels: int, dropout: float) -> None:
        super().__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Dropout2d(p=dropout)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Dropout2d(p=dropout)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Dropout2d(p=dropout)
        )
        
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward pass of the feature pyramid network.
        
        Parameters:
            x: Input tensor
            
        Returns:
            List of feature maps at different scales
        """
        p1 = self.conv1(x)
        p2 = self.conv2(self.upsample(p1))
        p3 = self.conv3(self.upsample(p2))
        
        return [p1, p2, p3]

class CombinedLoss(nn.Module):
    """
    Combines multiple loss functions into a single loss function.
    
    Parameters:
        loss_functions: List of loss functions to combine
        weights: Optional list of weights for each loss function
    """
    def __init__(self, loss_functions: List[nn.Module], weights: Optional[List[float]] = None) -> None:
        super().__init__()
        self.loss_functions = nn.ModuleList(loss_functions)
        
        if weights is None:
            weights = [1.0] * len(loss_functions)
        
        if len(weights) != len(loss_functions):
            print(f"Warning: Number of weights ({len(weights)}) does not match number of loss functions ({len(loss_functions)}). Adjusting weights.")
            # Adjust weights to match the number of loss functions
            if len(weights) < len(loss_functions):
                # If we have fewer weights than loss functions, pad with 1.0
                weights = weights + [1.0] * (len(loss_functions) - len(weights))
            else:
                # If we have more weights than loss functions, truncate
                weights = weights[:len(loss_functions)]
        
        # Store weights as a tensor to ensure proper indexing
        self.weights = torch.tensor(weights, dtype=torch.float32)
    
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the combined loss function.
        
        Parameters:
            input: Predicted logits
            target: Ground truth labels
            
        Returns:
            Weighted sum of all loss components
        """
        total_loss = 0
        
        for i, loss_fn in enumerate(self.loss_functions):
            loss = loss_fn(input, target)
            total_loss += self.weights[i] * loss
        
        return total_loss

class TrainExp:
    """
    Training loop implementation with validation and checkpointing.
    
    Parameters:
        model: Model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        optimizer: Optimizer for training
        scheduler: Learning rate scheduler
        loss_fn: Loss function
        device: Device to train on
        num_epochs: Number of epochs to train for
        early_stopping_patience: Number of epochs to wait before early stopping
        checkpoint_dir: Directory to save checkpoints
        use_amp: Whether to use automatic mixed precision
        use_ddp: Whether to use distributed data parallel
        rank: Rank of the current process
        world_size: Total number of processes
        gradient_accumulation_steps: Number of batches to accumulate gradients over
    """
    def __init__(self, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, 
                 optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler._LRScheduler,
                 loss_fn: nn.Module, device: torch.device, num_epochs: int = 100,
                 early_stopping_patience: int = 10, checkpoint_dir: str = None,
                 use_amp: bool = True, use_ddp: bool = False, rank: int = 0, world_size: int = 1,
                 gradient_accumulation_steps: int = 4) -> None:
        
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.device = device
        self.num_epochs = num_epochs
        self.early_stopping_patience = early_stopping_patience
        self.checkpoint_dir = checkpoint_dir
        self.use_amp = use_amp
        self.use_ddp = use_ddp
        self.rank = rank
        self.world_size = world_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        
        self.train_losses = []
        self.val_losses = []
        self.metrics_history = []  # New list to store epoch metrics
        
        self.scaler = GradScaler() if use_amp else None
        
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        self.best_model_state = None
        
        self.progress = {'train': [], 'val': []}
        
        # Remove folder creation from __init__
        self.version = None
        self.version_dir = None
        self.metrics_file = None
        self.epochs_dir = None
        
        # Store model configuration for JSON
        self.model_config = {}
        if hasattr(model, 'config'):
            self.model_config = {
                'n_classes': model.config.n_classes,
                'input_dim': model.config.input_dim,
                'output_dim': model.config.output_dim,
                'n_features': model.config.n_features,
                'dropout': model.config.dropout,
                'use_dense_connections': model.config.use_dense_connections,
                'use_multi_scale_attention': model.config.use_multi_scale_attention,
                'use_channel_attention': model.config.use_channel_attention,
                'use_attention': model.config.use_attention,
                'confidence_threshold': model.config.confidence_threshold
            }
        
        # Store optimizer configuration
        self.optimizer_config = {
            'type': optimizer.__class__.__name__,
            'params': {k: v for k, v in optimizer.param_groups[0].items() if k != 'params'}
        }
        
        # Store scheduler configuration
        self.scheduler_config = {
            'type': scheduler.__class__.__name__,
            'params': {k: v for k, v in scheduler.__dict__.items() 
                      if not k.startswith('_') and not callable(v)}
        }
        
        # Store loss function configuration
        self.loss_config = {
            'type': loss_fn.__class__.__name__,
            'params': {k: v for k, v in loss_fn.__dict__.items() 
                      if not k.startswith('_') and not callable(v)}
        }
        
        # Store training configuration
        self.training_config = {
            'num_epochs': num_epochs,
            'early_stopping_patience': early_stopping_patience,
            'use_amp': use_amp,
            'use_ddp': use_ddp,
            'rank': rank,
            'world_size': world_size,
            'gradient_accumulation_steps': gradient_accumulation_steps,
            'device': str(device),
            'train_dataset_size': len(train_loader.dataset),
            'val_dataset_size': len(val_loader.dataset),
            'batch_size': train_loader.batch_size,
            'num_workers': train_loader.num_workers
        }
        
        # Store model parameters count
        self.model_params = {
            'total_params': sum(p.numel() for p in model.parameters()),
            'trainable_params': sum(p.numel() for p in model.parameters() if p.requires_grad)
        }

    def train_epoch(self) -> float:
        """
        Train for one epoch.
        
        Returns:
            Average training loss for the epoch
        """
        self.model.train()
        total_loss = 0
        
        if torch.cuda.is_available() and self.device.type != 'cuda':
            print("WARNING: CUDA is available but not being used. Switching to GPU.")
            self.device = torch.device('cuda:0')
            self.model = self.model.to(self.device)
        
        pbar = tqdm(self.train_loader, desc=f'Training (Epoch {len(self.train_losses)+1})', 
                   disable=False,
                   dynamic_ncols=True,
                   mininterval=0.5)
        
        if torch.cuda.is_available():
            before_loop_mem = torch.cuda.memory_allocated() / 1e9
        
        running_loss = 0
        
        for batch_idx, (data, target, image_ids) in enumerate(pbar):
            try:
                if batch_idx % self.gradient_accumulation_steps == 0:
                    self.optimizer.zero_grad(set_to_none=True)
                
                data, target = data.to(self.device, non_blocking=True), target.to(self.device, non_blocking=True)
                
                if self.use_amp:
                    with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                        output = self.model(data)
                        loss = self.loss_fn(output['logits'], target) / self.gradient_accumulation_steps
                    
                    self.scaler.scale(loss).backward()
                    
                    if (batch_idx + 1) % self.gradient_accumulation_steps == 0 or (batch_idx + 1) == len(self.train_loader):
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        if isinstance(self.scheduler, OneCycleLR):
                            self.scheduler.step()
                else:
                    output = self.model(data)
                    loss = self.loss_fn(output['logits'], target) / self.gradient_accumulation_steps
                    loss.backward()
                    
                    if (batch_idx + 1) % self.gradient_accumulation_steps == 0 or (batch_idx + 1) == len(self.train_loader):
                        self.optimizer.step()
                        if isinstance(self.scheduler, OneCycleLR):
                            self.scheduler.step()
                
                running_loss = loss.item() * self.gradient_accumulation_steps
                total_loss += running_loss
                
                del output
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                avg_loss = total_loss / (batch_idx + 1)
                
                pbar.set_postfix({
                    'loss': f'{avg_loss:.4f}',
                    'lr': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
                })
                
                if batch_idx % 5 == 0 and torch.cuda.is_available():
                    current_mem = torch.cuda.memory_allocated() / 1e9
                    peak_mem = torch.cuda.max_memory_allocated() / 1e9
                    pbar.set_postfix({
                        'loss': f'{avg_loss:.4f}',
                        'mem': f'{current_mem:.1f}GB',
                        'peak': f'{peak_mem:.1f}GB',
                        'lr': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
                    })
                    
                    if peak_mem > torch.cuda.get_device_properties(0).total_memory * 0.8 / 1e9:
                        torch.cuda.empty_cache()
                        gc.collect()
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"WARNING: Out of memory error in batch {batch_idx}. Cleaning up and continuing...")
                    cleanup_semaphores()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
                else:
                    raise e
        
        if torch.cuda.is_available():
            after_loop_mem = torch.cuda.memory_allocated() / 1e9
            print(f"Memory before loop: {before_loop_mem:.2f} GB, after loop: {after_loop_mem:.2f} GB")
            torch.cuda.reset_peak_memory_stats()
        
        return total_loss / len(self.train_loader)

    def validate(self) -> float:
        """
        Validate the model.
        
        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0
        
        pbar = tqdm(self.val_loader, desc='Validation', 
                  disable=False,
                  dynamic_ncols=True,
                  mininterval=0.5)
        
        with torch.no_grad():
            for data, target, image_ids in pbar:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.loss_fn(output['logits'], target)
                total_loss += loss.item()
                
                pbar.set_postfix({
                    'val_loss': f'{total_loss/(pbar.n+1):.4f}'
                })
        
        return total_loss / len(self.val_loader)

    def train(self) -> Dict[str, List[float]]:
        """
        Train the model for the specified number of epochs.
        
        Returns:
            Dictionary containing training and validation losses
        """
        # Create version directory only after first epoch
        version_created = False
        
        for epoch in range(self.num_epochs):
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)
            
            val_loss = self.validate()
            self.val_losses.append(val_loss)
            
            if not isinstance(self.scheduler, OneCycleLR):
                self.scheduler.step(val_loss)
            
            self.progress['train'].append(train_loss)
            self.progress['val'].append(val_loss)
            
            # Create epoch metrics dictionary
            epoch_metrics = {
                'epoch': epoch + 1,
                'train_loss': float(train_loss),
                'val_loss': float(val_loss),
                'learning_rate': float(self.optimizer.param_groups[0]['lr']),
                'early_stopping_counter': self.epochs_without_improvement
            }
            self.metrics_history.append(epoch_metrics)
            
            # Create version directory after first epoch
            if not version_created and self.rank == 0 and self.checkpoint_dir:
                os.makedirs(self.checkpoint_dir, exist_ok=True)
                versions = sorted([d for d in os.listdir(self.checkpoint_dir) if d.startswith('v_')])
                
                if not versions:
                    self.version = 'v_1_01'
                else:
                    latest = versions[-1].split('_')
                    if len(latest) == 3:
                        major, minor = int(latest[1]), int(latest[2])
                        if self.num_epochs > 25:
                            major += 1
                            minor = 1
                        else:
                            minor += 1
                            if minor > 99:
                                major += 1
                                minor = 1
                        self.version = f"v_{major}_{minor:02d}"
                    else:
                        self.version = 'v_1_01'
                
                self.version_dir = os.path.join(self.checkpoint_dir, self.version)
                
                while os.path.exists(self.version_dir):
                    if len(latest) == 3:
                        major, minor = int(self.version.split('_')[1]), int(self.version.split('_')[2])
                        minor += 1
                        if minor > 99:
                            major += 1
                            minor = 1
                        self.version = f"v_{major}_{minor:02d}"
                        self.version_dir = os.path.join(self.checkpoint_dir, self.version)
                    else:
                        break
                        
                os.makedirs(self.version_dir, exist_ok=True)
                
                # Create epochs subdirectory
                self.epochs_dir = os.path.join(self.version_dir, 'epochs')
                os.makedirs(self.epochs_dir, exist_ok=True)
                
                print(f"Model will be saved as version: {self.version}")
                
                # Create metrics file path
                self.metrics_file = os.path.join(self.version_dir, 'training_metrics.json')
                version_created = True
            
            # Save metrics to JSON file after each epoch
            if self.rank == 0 and self.metrics_file:
                import json
                import datetime
                
                # Helper function to make values JSON serializable
                def make_serializable(obj):
                    if isinstance(obj, (int, float, str, bool, type(None))):
                        return obj
                    elif isinstance(obj, (list, tuple)):
                        return [make_serializable(item) for item in obj]
                    elif isinstance(obj, dict):
                        return {k: make_serializable(v) for k, v in obj.items()}
                    elif isinstance(obj, torch.Tensor):
                        return float(obj.item()) if obj.numel() == 1 else obj.tolist()
                    elif hasattr(obj, '__dict__'):
                        return make_serializable(obj.__dict__)
                    else:
                        return str(obj)
                
                try:
                    # Create comprehensive metrics dictionary
                    metrics_data = {
                        'version': self.version,
                        'timestamp': datetime.datetime.now().isoformat(),
                        'model_config': make_serializable(self.model_config),
                        'optimizer_config': make_serializable(self.optimizer_config),
                        'scheduler_config': make_serializable(self.scheduler_config),
                        'loss_config': make_serializable(self.loss_config),
                        'training_config': make_serializable(self.training_config),
                        'model_params': make_serializable(self.model_params),
                        'total_epochs': self.num_epochs,
                        'current_epoch': epoch + 1,
                        'best_val_loss': float(self.best_val_loss),
                        'epochs_without_improvement': self.epochs_without_improvement,
                        'metrics': make_serializable(self.metrics_history)
                    }
                    
                    # Save metrics to both the version directory and the base folder
                    with open(self.metrics_file, 'w') as f:
                        json.dump(metrics_data, f, indent=4)
                    print(f"Saved comprehensive metrics to {self.metrics_file}")
                    
                    # Also save to the base folder
                    base_metrics_file = os.path.join(self.checkpoint_dir, 'training_metrics.json')
                    with open(base_metrics_file, 'w') as f:
                        json.dump(metrics_data, f, indent=4)
                    print(f"Saved comprehensive metrics to base folder: {base_metrics_file}")
                except Exception as e:
                    print(f"Warning: Failed to save metrics to JSON: {str(e)}")
                    print("This is not a critical error, training will continue.")
            
            print(f'Epoch {epoch+1}/{self.num_epochs}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
            
            # Save model after each epoch in the epochs subdirectory
            if self.checkpoint_dir and self.rank == 0 and self.epochs_dir:
                epoch_path = os.path.join(self.epochs_dir, f'epoch_{epoch+1}.pt')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                }, epoch_path)
                print(f"Saved model for epoch {epoch+1} at {epoch_path}")
            
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.epochs_without_improvement = 0
                self.best_model_state = self.model.state_dict()
                
                if self.checkpoint_dir and self.rank == 0 and self.version_dir:
                    checkpoint_path = os.path.join(self.version_dir, f'best_model.pt')
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'scheduler_state_dict': self.scheduler.state_dict(),
                        'train_loss': train_loss,
                        'val_loss': val_loss,
                    }, checkpoint_path)
                    print(f"Saved best model at {checkpoint_path}")
            else:
                self.epochs_without_improvement += 1
            
            if self.epochs_without_improvement >= self.early_stopping_patience:
                print(f'Early stopping triggered after {epoch+1} epochs')
                if self.checkpoint_dir and self.rank == 0 and self.version_dir:
                    final_path = os.path.join(self.version_dir, 'model.pt')
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.best_model_state,
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'scheduler_state_dict': self.scheduler.state_dict(),
                        'train_loss': train_loss,
                        'val_loss': val_loss,
                    }, final_path)
                break
        
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        
        result = {'train_losses': self.train_losses, 'val_losses': self.val_losses}
        if self.rank == 0 and self.checkpoint_dir:
            result['version'] = self.version
            
        return result

def cleanup_semaphores():
    """
    Clean up leaked semaphore objects to prevent resource leaks.
    This should be called at the start and end of training.
    """
    gc.collect()
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    try:
        import torch.multiprocessing as mp
        if hasattr(mp, '_queue_map'):
            mp._queue_map.clear()
    except (ImportError, AttributeError):
        pass
    
    try:
        import resource
        resource.setrlimit(resource.RLIMIT_NOFILE, (4096, 4096))
    except (ImportError, ValueError):
        pass
    
    gc.collect()

def train_model_exp(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                   optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler._LRScheduler,
                   loss_fn: nn.Module, device: torch.device, num_epochs: int = 100,
                   early_stopping_patience: int = 10, checkpoint_dir: str = None,
                   use_amp: bool = True, use_ddp: bool = False, rank: int = 0, world_size: int = 1,
                   gradient_accumulation_steps: int = 4, test_visualization: bool = False) -> Dict[str, List[float]]:
    """
    Train the model using the experimental trainer.
    
    Parameters:
        model: Model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        optimizer: Optimizer for training
        scheduler: Learning rate scheduler
        loss_fn: Loss function
        device: Device to train on
        num_epochs: Number of epochs to train for
        early_stopping_patience: Number of epochs to wait before early stopping
        checkpoint_dir: Directory to save checkpoints
        use_amp: Whether to use automatic mixed precision
        use_ddp: Whether to use distributed data parallel
        rank: Rank of the current process
        world_size: Total number of processes
        gradient_accumulation_steps: Number of batches to accumulate gradients over
        test_visualization: Whether to generate test visualizations after each epoch
        
    Returns:
        Dictionary containing training and validation losses
    """
    cleanup_semaphores()
    
    # Prepare test visualization if needed
    test_img_path = None
    if test_visualization:
        test_img_path = os.path.join(base_directory, 'data', 'train', '119_sat.jpg')
        if not os.path.exists(test_img_path):
            print(f"Warning: Test image {test_img_path} not found. Disabling test visualization.")
            test_visualization = False
        else:
            try:
                from PIL import Image
                import matplotlib.pyplot as plt
                import matplotlib.patches as mpatches
                print(f"Will visualize predictions on {test_img_path} after each epoch")
            except ImportError:
                print("Warning: PIL or matplotlib not available. Disabling test visualization.")
                test_visualization = False
    
    try:
        # Adjust hyperparameters for less restricted predictions
        if isinstance(model, UNetExp):
            model.confidence_threshold = 0.2  # Lower threshold for more diverse predictions
        
        # Adjust loss function weights for better generalization
        if isinstance(loss_fn, CombinedLoss):
            # Update the weights tensor instead of trying to assign a new list
            if len(loss_fn.weights) == 4:  # If we have 4 loss functions
                loss_fn.weights = torch.tensor([0.6, 0.2, 0.15, 0.05], dtype=torch.float32)
            elif len(loss_fn.weights) == 2:  # If we have 2 loss functions
                loss_fn.weights = torch.tensor([0.6, 0.4], dtype=torch.float32)
            else:
                print(f"Warning: Unexpected number of loss functions: {len(loss_fn.weights)}")
        
        trainer = TrainExp(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_fn=loss_fn,
            device=device,
            num_epochs=num_epochs,
            early_stopping_patience=early_stopping_patience,
            checkpoint_dir=checkpoint_dir,
            use_amp=use_amp,
            use_ddp=use_ddp,
            rank=rank,
            world_size=world_size,
            gradient_accumulation_steps=gradient_accumulation_steps
        )
        
        if test_visualization and rank == 0:
            def visualize_prediction(model, img_paths, save_path, device, is_test=False):
                try:
                    from PIL import Image
                    import matplotlib.pyplot as plt
                    import matplotlib.patches as mpatches
                    import numpy as np
                    import torch.nn.functional as F
                    import os
                    
                    # Define class colors
                    class_colors = {
                        0: (0, 255, 255),    # urban land - Cyan
                        1: (255, 255, 0),    # agricultural land - Yellow
                        2: (255, 0, 255),    # rangeland - Magenta
                        3: (0, 255, 0),      # forest land - Green
                        4: (0, 0, 255),      # water - Dark blue
                        5: (255, 255, 255),  # barren land - White
                        6: (0, 0, 0)         # unknown - Black
                    }
                    
                    # Create appropriate figure based on whether this is a test or training visualization
                    if is_test:
                        # For test images: 2 rows (original, prediction) x 5 columns (one for each image)
                        fig, axes = plt.subplots(2, 5, figsize=(25, 10))
                    else:
                        # For training images: 3 rows (one for each image) x 3 columns (original, true mask, predicted mask)
                        fig, axes = plt.subplots(3, 3, figsize=(18, 18))
                    
                    for img_idx, img_path in enumerate(img_paths):
                        print(f"Processing image {img_idx+1}: {img_path}")
                        
                        # Load the original image
                        img = Image.open(img_path).convert('RGB')
                        img_array = np.array(img)
                        
                        # Create model input tensor
                        img_tensor = transformer_exp(img).unsqueeze(0).to(device)
                        
                        # Load true mask if it exists (only for training images)
                        mask_path = img_path.replace('_sat.jpg', '_mask.png')
                        print(f"Looking for mask at {mask_path}")
                        
                        has_true_mask = False
                        if os.path.exists(mask_path) and not is_test:
                            # Load true mask as RGB and convert to class index
                            true_mask_rgb = np.array(Image.open(mask_path).convert('RGB'))
                            print(f"Loaded true mask with shape {true_mask_rgb.shape}")
                            
                            # Resize true mask to match prediction size
                            true_mask_rgb = np.array(Image.fromarray(true_mask_rgb).resize((256, 256), Image.NEAREST))
                            
                            # Convert RGB mask to class indices
                            true_mask = np.zeros((true_mask_rgb.shape[0], true_mask_rgb.shape[1]), dtype=np.uint8)
                            for class_idx, color in class_colors.items():
                                mask = np.all(true_mask_rgb == color, axis=2)
                                true_mask[mask] = class_idx
                            
                            has_true_mask = True
                        else:
                            print(f"True mask not found at {mask_path}, creating dummy mask")
                            true_mask = np.zeros((256, 256), dtype=np.uint8)
                            has_true_mask = False
                        
                        # Run prediction
                        model.eval()
                        with torch.no_grad():
                            with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                                output = model(img_tensor)
                                pred = output['logits']
                                pred = F.softmax(pred, dim=1)
                                pred = pred.argmax(dim=1).squeeze().cpu().numpy()
                        
                        print(f"Generated prediction with shape {pred.shape}")
                        
                        # Create the colored prediction mask
                        colored_pred = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
                        class_counts = {}
                        
                        for i in range(pred.shape[0]):
                            for j in range(pred.shape[1]):
                                try:
                                    class_idx = int(pred[i, j])
                                    if class_idx not in class_counts:
                                        class_counts[class_idx] = 0
                                    class_counts[class_idx] += 1
                                    
                                    if class_idx in class_colors:
                                        colored_pred[i, j] = class_colors[class_idx]
                                    else:
                                        colored_pred[i, j] = class_colors[6]  # Unknown
                                except Exception as e:
                                    print(f"Error processing prediction at ({i}, {j}): {e}")
                                    colored_pred[i, j] = class_colors[6]  # Unknown
                        
                        print(f"Class counts in prediction: {class_counts}")
                        
                        # Create the colored true mask (only for training images)
                        colored_true = np.zeros((true_mask.shape[0], true_mask.shape[1], 3), dtype=np.uint8)
                        true_class_counts = {}
                        
                        for i in range(true_mask.shape[0]):
                            for j in range(true_mask.shape[1]):
                                try:
                                    class_idx = int(true_mask[i, j])
                                    if class_idx not in true_class_counts:
                                        true_class_counts[class_idx] = 0
                                    true_class_counts[class_idx] += 1
                                    
                                    if class_idx in class_colors:
                                        colored_true[i, j] = class_colors[class_idx]
                                    else:
                                        colored_true[i, j] = class_colors[6]  # Unknown
                                except Exception as e:
                                    # If conversion fails or class_idx not found, use black (unknown)
                                    colored_true[i, j] = class_colors[6]  # Unknown
                        
                        if has_true_mask:
                            print(f"Class counts in true mask: {true_class_counts}")
                        
                        # Display images based on whether this is a test or training visualization
                        if is_test:
                            # For test images: original image in first row, prediction in second row
                            axes[0, img_idx].imshow(img_array)
                            axes[0, img_idx].set_title(f'Original Image ({os.path.basename(img_path)})')
                            axes[0, img_idx].axis('off')
                            
                            axes[1, img_idx].imshow(colored_pred)
                            axes[1, img_idx].set_title('Predicted Mask')
                            axes[1, img_idx].axis('off')
                        else:
                            # For training images: original, true mask, predicted mask in each row
                            axes[img_idx, 0].imshow(img_array)
                            axes[img_idx, 0].set_title(f'Original Image ({os.path.basename(img_path)})')
                            axes[img_idx, 0].axis('off')
                            
                            axes[img_idx, 1].imshow(colored_true)
                            axes[img_idx, 1].set_title('True Mask')
                            axes[img_idx, 1].axis('off')
                            
                            axes[img_idx, 2].imshow(colored_pred)
                            axes[img_idx, 2].set_title('Predicted Mask')
                            axes[img_idx, 2].axis('off')
                    
                    # Define class labels
                    classes = ['Urban', 'Agriculture', 'Rangeland', 'Forest', 'Water', 'Barren', 'Unknown']
                    patches = [mpatches.Patch(color=np.array(class_colors[i])/255, label=classes[i]) for i in range(len(classes))]
                    fig.legend(handles=patches, bbox_to_anchor=(1.05, 0.5), loc='center left')
                    
                    plt.tight_layout()
                    print(f"Saving visualization to {save_path}")
                    plt.savefig(save_path, dpi=150, bbox_inches='tight')
                    plt.close()
                    print(f"Successfully saved visualization to {save_path}")
                except Exception as e:
                    print(f"Error generating visualization: {e}")
                    import traceback
                    traceback.print_exc()  # Print detailed error information
            
            original_train = trainer.train
            viz_dir = None
            
            def train_with_viz(self):
                if self.rank == 0 and self.checkpoint_dir:
                    os.makedirs(self.checkpoint_dir, exist_ok=True)
                    versions = sorted([d for d in os.listdir(self.checkpoint_dir) if d.startswith('v_')])
                    
                    if not versions:
                        version = 'v_1_01'
                    else:
                        latest = versions[-1].split('_')
                        if len(latest) == 3:
                            major, minor = int(latest[1]), int(latest[2])
                            if self.num_epochs > 25:
                                major += 1
                                minor = 1
                            else:
                                minor += 1
                                if minor > 99:
                                    major += 1
                                    minor = 1
                            version = f"v_{major}_{minor:02d}"
                        else:
                            version = 'v_1_01'
                    
                    version_dir = os.path.join(self.checkpoint_dir, version)
                    
                    while os.path.exists(version_dir):
                        if len(latest) == 3:
                            major, minor = int(version.split('_')[1]), int(version.split('_')[2])
                            minor += 1
                            if minor > 99:
                                major += 1
                                minor = 1
                            version = f"v_{major}_{minor:02d}"
                            version_dir = os.path.join(self.checkpoint_dir, version)
                        else:
                            break
                            
                    os.makedirs(version_dir, exist_ok=True)
                    print(f"Model will be saved as version: {version}")
                    
                    # Create epochs subdirectory
                    epochs_dir = os.path.join(version_dir, 'epochs')
                    os.makedirs(epochs_dir, exist_ok=True)
                    
                    nonlocal viz_dir
                    viz_dir = os.path.join(version_dir, 'visualizations')
                    os.makedirs(viz_dir, exist_ok=True)
                    
                    # Create train_vis and test_vis subdirectories
                    train_vis_dir = os.path.join(viz_dir, 'train_vis')
                    test_vis_dir = os.path.join(viz_dir, 'test_vis')
                    os.makedirs(train_vis_dir, exist_ok=True)
                    os.makedirs(test_vis_dir, exist_ok=True)
                else:
                    version = None
                    version_dir = None
                    epochs_dir = None
                    train_vis_dir = None
                    test_vis_dir = None
                    
                if self.use_ddp:
                    version_tensor = torch.zeros(1, dtype=torch.long, device=self.device)
                    if self.rank == 0:
                        version_tensor[0] = int(version.split('_')[1]) * 100 + int(version.split('_')[2])
                    dist.broadcast(version_tensor, 0)
                    version = f"v_{version_tensor[0]//100}_{version_tensor[0]%100:02d}"
                    version_dir = os.path.join(self.checkpoint_dir, version)
                    epochs_dir = os.path.join(version_dir, 'epochs')
                    
                for epoch in range(self.num_epochs):
                    train_loss = self.train_epoch()
                    self.train_losses.append(train_loss)
                    
                    val_loss = self.validate()
                    self.val_losses.append(val_loss)
                    
                    if not isinstance(self.scheduler, OneCycleLR):
                        self.scheduler.step(val_loss)
                    
                    self.progress['train'].append(train_loss)
                    self.progress['val'].append(val_loss)
                    
                    print(f'Epoch {epoch+1}/{self.num_epochs}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
                    
                    if self.rank == 0 and viz_dir:
                        # Define training and test image paths
                        train_img_paths = [
                            os.path.join(base_directory, 'data', 'train', '119_sat.jpg'),
                            os.path.join(base_directory, 'data', 'train', '855_sat.jpg'),
                            os.path.join(base_directory, 'data', 'train', '40350_sat.jpg')
                        ]
                        
                        test_img_paths = [
                            os.path.join(base_directory, 'data', 'test', '6783_sat.jpg'),
                            os.path.join(base_directory, 'data', 'test', '14397_sat.jpg'),
                            os.path.join(base_directory, 'data', 'test', '23458_sat.jpg'),
                            os.path.join(base_directory, 'data', 'test', '41687_sat.jpg'),
                            os.path.join(base_directory, 'data', 'test', '100877_sat.jpg')
                        ]
                        
                        # Generate and save training visualizations
                        visualize_prediction(
                            self.model, 
                            train_img_paths, 
                            os.path.join(train_vis_dir, f'epoch_{epoch+1}.png'),
                            self.device,
                            is_test=False
                        )
                        
                        # Generate and save test visualizations
                        visualize_prediction(
                            self.model, 
                            test_img_paths, 
                            os.path.join(test_vis_dir, f'epoch_{epoch+1}.png'),
                            self.device,
                            is_test=True
                        )
                    
                    if self.checkpoint_dir and self.rank == 0 and epochs_dir:
                        epoch_path = os.path.join(epochs_dir, f'epoch_{epoch+1}.pt')
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            'scheduler_state_dict': self.scheduler.state_dict(),
                            'train_loss': train_loss,
                            'val_loss': val_loss,
                        }, epoch_path)
                        print(f"Saved model for epoch {epoch+1} at {epoch_path}")
                    
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self.epochs_without_improvement = 0
                        self.best_model_state = self.model.state_dict()
                        
                        if self.checkpoint_dir and self.rank == 0:
                            checkpoint_path = os.path.join(version_dir, f'best_model.pt')
                            torch.save({
                                'epoch': epoch,
                                'model_state_dict': self.model.state_dict(),
                                'optimizer_state_dict': self.optimizer.state_dict(),
                                'scheduler_state_dict': self.scheduler.state_dict(),
                                'train_loss': train_loss,
                                'val_loss': val_loss,
                            }, checkpoint_path)
                            print(f"Saved best model at {checkpoint_path}")
                    else:
                        self.epochs_without_improvement += 1
                    
                    if self.epochs_without_improvement >= self.early_stopping_patience:
                        print(f'Early stopping triggered after {epoch+1} epochs')
                        if self.checkpoint_dir and self.rank == 0:
                            final_path = os.path.join(version_dir, 'model.pt')
                            torch.save({
                                'epoch': epoch,
                                'model_state_dict': self.best_model_state,
                                'optimizer_state_dict': self.optimizer.state_dict(),
                                'scheduler_state_dict': self.scheduler.state_dict(),
                                'train_loss': train_loss,
                                'val_loss': val_loss,
                            }, final_path)
                        break
                
                if self.best_model_state is not None:
                    self.model.load_state_dict(self.best_model_state)
                
                result = {'train_losses': self.train_losses, 'val_losses': self.val_losses}
                if self.rank == 0 and self.checkpoint_dir:
                    result['version'] = version
                    
                return result
            
            trainer.train = train_with_viz.__get__(trainer)
        
        result = trainer.train()
        
    finally:
        cleanup_semaphores()
    
    return result

def compile_model(
    model_config: UNetConfigExp,
    loss_functions: Optional[List[nn.Module]] = None,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    learning_rate: float = 0.001,
    weight_decay: float = 1e-4,
    optimizer_class: str = 'adam',
    scheduler_class: str = 'cosine',
    scheduler_params: Optional[Dict[str, Any]] = None
) -> Tuple[nn.Module, torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
    """
    Compile a model with the given configuration and return the model, optimizer, and scheduler.
    
    Args:
        model_config: UNetConfigExp object with model parameters
        loss_functions: List of loss functions to use
        device: Device to use for training ('cuda' or 'cpu')
        learning_rate: Learning rate for the optimizer
        weight_decay: Weight decay for the optimizer
        optimizer_class: Optimizer class to use ('adam', 'adamw', 'sgd')
        scheduler_class: Scheduler class to use ('cosine', 'step', 'plateau')
        scheduler_params: Additional parameters for the scheduler
        
    Returns:
        Tuple containing the model, optimizer, and scheduler
    """
    # Create model
    model = UNetExp(model_config)
    model = model.to(device)
    
    # Create loss functions if not provided
    if loss_functions is None:
        focal_loss = FocalLossExp()
        boundary_loss = SemanticBoundaryLoss(focal_loss=focal_loss)
        topology_loss = TopologyAwareLoss(focal_loss=focal_loss)
        contrastive_loss = ContrastiveLoss()
        
        loss_functions = [
            CombinedLoss(
                [focal_loss, boundary_loss, topology_loss, contrastive_loss],
                [1.0, 0.2, 0.15, 0.1]
            )
        ]
    
    # Create optimizer
    if optimizer_class.lower() == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
    elif optimizer_class.lower() == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
    elif optimizer_class.lower() == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=0.9,
            weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer class: {optimizer_class}")
    
    # Create scheduler
    if scheduler_class.lower() == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=100,  # Will be adjusted in the trainer
            eta_min=learning_rate * 0.01
        )
    elif scheduler_class.lower() == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=30,  # Will be adjusted in the trainer
            gamma=0.1
        )
    elif scheduler_class.lower() == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.1,
            patience=5,
            verbose=True
        )
    else:
        raise ValueError(f"Unknown scheduler class: {scheduler_class}")
    
    # Apply additional scheduler parameters if provided
    if scheduler_params is not None:
        for key, value in scheduler_params.items():
            if hasattr(scheduler, key):
                setattr(scheduler, key, value)
    
    return model, optimizer, scheduler

if __name__ == '__main__':
    try:
        cleanup_semaphores()
        print("===== GPU DETECTION =====")
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            print(f"CUDA is available. Found {device_count} GPU device(s).")
            for i in range(device_count):
                print(f"Device {i}: {torch.cuda.get_device_name(i)}")
                print(f"  - Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")
            
            device = torch.device('cuda:0')
            print(f"Using device: {device}")
            
            test_tensor = torch.rand(1000, 1000, device=device)
            _ = torch.matmul(test_tensor, test_tensor)
            print("✓ GPU test passed: Matrix multiplication successful on GPU")
        else:
            print("CUDA is not available. Training will proceed on CPU, which may be very slow.")
            device = torch.device('cpu')
        print("=========================")
        
        def print_memory_status():
            if torch.cuda.is_available():
                gpu_memory_allocated = torch.cuda.memory_allocated() / 1e9
                gpu_memory_reserved = torch.cuda.memory_reserved() / 1e9
                print(f"GPU Memory: {gpu_memory_allocated:.2f} GB allocated, {gpu_memory_reserved:.2f} GB reserved")
            
            import psutil
            memory = psutil.virtual_memory()
            print(f"System Memory: {memory.used/1e9:.2f} GB used out of {memory.total/1e9:.2f} GB total ({memory.percent}%)")
        
        print_memory_status()
        
        model_config = UNetConfigExp(
            n_classes=7,
            input_dim=3,
            output_dim=1,
            n_features=16,
            dropout=0.2,
            use_dense_connections=True,
            use_multi_scale_attention=True,
            use_channel_attention=True,
            use_attention=True,
            confidence_threshold=0.25
        )
        model = UNetExp(model_config).to(device)
        
        print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
        print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
        
        train_data = SatelliteImages(
            os.path.join(base_directory, 'data', 'train'),
            transform=transformer_exp
        )
        val_data = SatelliteImages(
            os.path.join(base_directory, 'data', 'val'),
            transform=transformer_exp
        )
        
        # Reduced batch size and enabled gradient accumulation
        batch_size = 2  # Increase from 1
        gradient_accumulation_steps = 16  # Increase from 8
        print(f"Using batch size: {batch_size} with gradient accumulation steps: {gradient_accumulation_steps}")
        print(f"Effective batch size: {batch_size * gradient_accumulation_steps}")
        
        # Reduce number of worker threads
        num_workers = min(2, config['data']['num_workers'])
        
        train_loader = DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_data,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=0.001,  # Lower base learning rate
            weight_decay=0.01  # Slightly lower weight decay
        )
        
        scheduler = OneCycleLR(
            optimizer,
            max_lr=0.003,  # Lower max learning rate
            epochs=35,  # Match the actual number of epochs
            steps_per_epoch=len(train_loader) // gradient_accumulation_steps
        )

        class_colors = {
        0: (0, 255, 255),    # urban land - Cyan
        1: (255, 255, 0),    # agricultural land - Yellow
        2: (255, 0, 255),    # rangeland - Magenta
        3: (0, 255, 0),      # forest land - Green
        4: (0, 0, 255),      # water - Dark blue
        5: (255, 255, 255),  # barren land - White
        6: (0, 0, 0)         # unknown - Black
        }
        
        class_weights = torch.ones(7)
        class_weights[1] = 0.75  # Increase from 0.5
        class_weights[0] = 1.3  # Increase from 1.0
        class_weights[2] = 1.2  # Increase from 1.0
        class_weights[3] = 1.3  # Increase from 1.0
        
        focal_loss = FocalLossExp(
            gamma=3.0,
            class_weights=class_weights,
            temperature=1.5,
            entropy_weight=0.05,
            adaptive_gamma=True
        )
        
        print("\nMemory after model creation:")
        print_memory_status()
        
        semantic_loss = SemanticBoundaryLoss(
            weight=0.2,
            focal_loss=focal_loss,
            boundary_dilation=1
        )
        
        topology_loss = TopologyAwareLoss(
            weight=0.15,
            focal_loss=focal_loss,
            neighborhood_size=3
        )
        
        contrastive_loss = ContrastiveLoss(
            weight=0.1,
            temperature=0.07,
            margin=1.0
        )
        
        loss_fn = CombinedLoss(
            loss_functions=[focal_loss, semantic_loss, topology_loss, contrastive_loss],
            weights=[0.5, 0.25, 0.15, 0.1]
        )

        # loss_fn = CombinedLoss(
        #     loss_functions=[focal_loss, semantic_loss],
        #     weights=[0.6, 0.4]
        # )
        
        checkpoint_dir = os.path.join(base_directory, 'data', 'models', 'exp_checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        result = train_model_exp(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_fn=loss_fn,
            device=device,
            num_epochs=50,
            early_stopping_patience=7,
            checkpoint_dir=checkpoint_dir,
            use_amp=True,
            use_ddp=False,
            rank=0,
            world_size=1,
            gradient_accumulation_steps=gradient_accumulation_steps,
            test_visualization=True
        )
        print(f"Training completed with result: {result}")
        if 'version' in result:
            print(f"Model saved with version: {result['version']}")
            
        print("\nFinal memory status:")
        print_memory_status()
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cleanup_semaphores()
        print("Training process completed.")