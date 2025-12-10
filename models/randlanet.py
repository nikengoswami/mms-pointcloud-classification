"""
RandLA-Net: Efficient Semantic Segmentation of Large-Scale Point Clouds

Reference:
Hu et al., "RandLA-Net: Efficient Semantic Segmentation of Large-Scale Point Clouds"
CVPR 2020

This implementation is optimized for MMS point cloud classification with classes:
- Road (0)
- Snow (1)
- Vehicle (2)
- Vegetation (3)
- Others (4)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List
import numpy as np


class SharedMLP(nn.Module):
    """Shared MLP layer"""

    def __init__(self, in_channels: int, out_channels: List[int], bn: bool = True, activation: bool = True):
        super().__init__()

        layers = []
        for out_ch in out_channels:
            layers.append(nn.Conv2d(in_channels, out_ch, 1, bias=not bn))
            if bn:
                layers.append(nn.BatchNorm2d(out_ch))
            if activation:
                layers.append(nn.ReLU(inplace=True))
            in_channels = out_ch

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class LocalSpatialEncoding(nn.Module):
    """Relative Point Position Encoding"""

    def __init__(self, d: int = 10):
        super().__init__()
        self.d = d
        self.mlp = SharedMLP(d, [d, 2 * d])

    def forward(self, coords: torch.Tensor, neighbor_coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coords: (B, N, 3) coordinates
            neighbor_coords: (B, N, K, 3) neighbor coordinates

        Returns:
            (B, N, K, out_dim) encoded features
        """
        # Center coordinates
        centered_coords = neighbor_coords - coords.unsqueeze(2)  # (B, N, K, 3)

        # Compute distances
        dist = torch.norm(centered_coords, dim=-1, keepdim=True)  # (B, N, K, 1)

        # Concatenate relative position and distance
        encoding = torch.cat([centered_coords, dist], dim=-1)  # (B, N, K, 4)

        # Permute for Conv2d: (B, N, K, 4) -> (B, 4, N, K)
        encoding = encoding.permute(0, 3, 1, 2)

        # Apply MLP
        encoding = self.mlp(encoding)  # (B, out_dim, N, K)

        # Permute back: (B, out_dim, N, K) -> (B, N, K, out_dim)
        encoding = encoding.permute(0, 2, 3, 1)

        return encoding


class AttentivePooling(nn.Module):
    """Attentive pooling mechanism"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.score_fn = nn.Sequential(
            nn.Linear(in_channels, in_channels, bias=False),
            nn.Softmax(dim=-2)
        )

        self.mlp = SharedMLP(in_channels, [out_channels])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, K, in_channels) features

        Returns:
            (B, N, out_channels) pooled features
        """
        # Compute attention scores
        scores = self.score_fn(x)  # (B, N, K, in_channels)

        # Weighted sum
        features = torch.sum(scores * x, dim=2)  # (B, N, in_channels)

        # Permute for Conv2d
        features = features.permute(0, 2, 1).unsqueeze(-1)  # (B, in_channels, N, 1)

        # Apply MLP
        features = self.mlp(features)  # (B, out_channels, N, 1)

        # Permute back
        features = features.squeeze(-1).permute(0, 2, 1)  # (B, N, out_channels)

        return features


class LocalFeatureAggregation(nn.Module):
    """Local Feature Aggregation module"""

    def __init__(self, in_channels: int, out_channels: int, num_neighbors: int = 16):
        super().__init__()

        self.num_neighbors = num_neighbors

        # Local Spatial Encoding
        self.lse = LocalSpatialEncoding(d=in_channels)

        # MLPs
        self.mlp1 = SharedMLP(in_channels, [out_channels // 2])
        self.mlp2 = SharedMLP(out_channels // 2, [out_channels])

        # Attentive pooling
        self.pool = AttentivePooling(out_channels, out_channels)

    def forward(self, coords: torch.Tensor, features: torch.Tensor,
                neighbor_indices: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coords: (B, N, 3) point coordinates
            features: (B, N, in_channels) point features
            neighbor_indices: (B, N, K) indices of neighbors

        Returns:
            (B, N, out_channels) aggregated features
        """
        B, N, K = neighbor_indices.shape

        # Gather neighbor features
        # Expand indices for gathering
        idx_expanded = neighbor_indices.unsqueeze(-1).expand(-1, -1, -1, features.shape[-1])
        neighbor_features = torch.gather(features, 1, idx_expanded)  # (B, N, K, C)

        # Get neighbor coordinates
        idx_coords = neighbor_indices.unsqueeze(-1).expand(-1, -1, -1, 3)
        neighbor_coords = torch.gather(coords, 1, idx_coords)  # (B, N, K, 3)

        # Relative position encoding
        encoding = self.lse(coords, neighbor_coords)  # (B, N, K, out_dim)

        # Concatenate features
        # First permute for conv: (B, N, K, C) -> (B, C, N, K)
        neighbor_features = neighbor_features.permute(0, 3, 1, 2)
        encoding = encoding.permute(0, 3, 1, 2)

        concat_features = torch.cat([neighbor_features, encoding], dim=1)  # (B, C+out_dim, N, K)

        # Apply MLPs
        features = self.mlp1(concat_features)  # (B, out_channels//2, N, K)
        features = self.mlp2(features)  # (B, out_channels, N, K)

        # Permute back for pooling
        features = features.permute(0, 2, 3, 1)  # (B, N, K, out_channels)

        # Attentive pooling
        features = self.pool(features)  # (B, N, out_channels)

        return features


class DilatedResidualBlock(nn.Module):
    """Dilated Residual Block"""

    def __init__(self, in_channels: int, out_channels: int, num_neighbors: int = 16):
        super().__init__()

        self.lfa1 = LocalFeatureAggregation(in_channels, out_channels, num_neighbors)
        self.lfa2 = LocalFeatureAggregation(out_channels, out_channels, num_neighbors)

        # Shortcut
        if in_channels != out_channels:
            self.shortcut = nn.Linear(in_channels, out_channels)
        else:
            self.shortcut = nn.Identity()

        self.relu = nn.ReLU(inplace=True)

    def forward(self, coords: torch.Tensor, features: torch.Tensor,
                neighbor_indices: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coords: (B, N, 3)
            features: (B, N, in_channels)
            neighbor_indices: (B, N, K)

        Returns:
            (B, N, out_channels)
        """
        identity = self.shortcut(features)

        out = self.lfa1(coords, features, neighbor_indices)
        out = self.lfa2(coords, out, neighbor_indices)

        out = out + identity
        out = self.relu(out)

        return out


class RandLANet(nn.Module):
    """RandLA-Net for semantic segmentation"""

    def __init__(self, num_classes: int = 5, num_features: int = 6,
                 num_neighbors: int = 16, decimation: int = 4):
        """
        Args:
            num_classes: Number of output classes
            num_features: Number of input features (default: 6 for XYZ + RGB)
            num_neighbors: Number of neighbors for local aggregation
            decimation: Decimation factor for downsampling
        """
        super().__init__()

        self.num_classes = num_classes
        self.num_neighbors = num_neighbors
        self.decimation = decimation

        # Feature dimensions at each layer
        d_out = [32, 64, 128, 256, 512]

        # Input MLP
        self.fc_start = nn.Linear(num_features, d_out[0])

        # Encoder (downsampling path)
        self.encoder1 = DilatedResidualBlock(d_out[0], d_out[1], num_neighbors)
        self.encoder2 = DilatedResidualBlock(d_out[1], d_out[2], num_neighbors)
        self.encoder3 = DilatedResidualBlock(d_out[2], d_out[3], num_neighbors)
        self.encoder4 = DilatedResidualBlock(d_out[3], d_out[4], num_neighbors)

        # Decoder (upsampling path)
        self.decoder4 = SharedMLP(d_out[4] + d_out[3], [d_out[3]])
        self.decoder3 = SharedMLP(d_out[3] + d_out[2], [d_out[2]])
        self.decoder2 = SharedMLP(d_out[2] + d_out[1], [d_out[1]])
        self.decoder1 = SharedMLP(d_out[1] + d_out[0], [d_out[0]])

        # Final classifier
        self.fc_end = nn.Sequential(
            nn.Linear(d_out[0], d_out[0]),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(d_out[0], num_classes)
        )

    def random_sample(self, coords: torch.Tensor, features: torch.Tensor,
                     num_samples: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Random sampling for downsampling

        Args:
            coords: (B, N, 3)
            features: (B, N, C)
            num_samples: Number of points to sample

        Returns:
            Tuple of (sampled_coords, sampled_features, sample_indices)
        """
        B, N, _ = coords.shape

        # Random sampling indices
        sample_indices = torch.randperm(N)[:num_samples].to(coords.device)
        sample_indices = sample_indices.unsqueeze(0).expand(B, -1)  # (B, num_samples)

        # Gather sampled points
        idx_3d = sample_indices.unsqueeze(-1).expand(-1, -1, 3)
        sampled_coords = torch.gather(coords, 1, idx_3d)

        idx_features = sample_indices.unsqueeze(-1).expand(-1, -1, features.shape[-1])
        sampled_features = torch.gather(features, 1, idx_features)

        return sampled_coords, sampled_features, sample_indices

    def nearest_interpolation(self, target_coords: torch.Tensor, source_coords: torch.Tensor,
                              source_features: torch.Tensor) -> torch.Tensor:
        """
        Nearest neighbor interpolation for upsampling

        Args:
            target_coords: (B, N_target, 3) coordinates to interpolate to
            source_coords: (B, N_source, 3) source coordinates
            source_features: (B, N_source, C) source features

        Returns:
            (B, N_target, C) interpolated features
        """
        # Find nearest neighbor for each target point
        dist = torch.cdist(target_coords, source_coords)  # (B, N_target, N_source)
        nearest_idx = torch.argmin(dist, dim=-1)  # (B, N_target)

        # Gather features
        idx_expanded = nearest_idx.unsqueeze(-1).expand(-1, -1, source_features.shape[-1])
        interpolated = torch.gather(source_features, 1, idx_expanded)

        return interpolated

    def knn(self, query_coords: torch.Tensor, support_coords: torch.Tensor,
           k: int) -> torch.Tensor:
        """
        K-nearest neighbors

        Args:
            query_coords: (B, N_query, 3)
            support_coords: (B, N_support, 3)
            k: Number of neighbors

        Returns:
            (B, N_query, k) indices of nearest neighbors
        """
        dist = torch.cdist(query_coords, support_coords)  # (B, N_query, N_support)
        _, indices = torch.topk(dist, k, dim=-1, largest=False)  # (B, N_query, k)
        return indices

    def forward(self, coords: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            coords: (B, N, 3) point coordinates
            features: (B, N, F) point features

        Returns:
            (B, N, num_classes) per-point logits
        """
        B, N, _ = coords.shape

        # Store for skip connections
        coords_list = [coords]
        features_list = []

        # Initial feature transform
        features = self.fc_start(features)  # (B, N, d_out[0])
        features_list.append(features)

        # Encoding path
        for i, encoder in enumerate([self.encoder1, self.encoder2, self.encoder3, self.encoder4]):
            # Compute neighbors
            neighbor_indices = self.knn(coords, coords, self.num_neighbors)

            # Apply encoder
            features = encoder(coords, features, neighbor_indices)

            # Downsample (except last layer)
            if i < 3:
                num_samples = coords.shape[1] // self.decimation
                coords, features, _ = self.random_sample(coords, features, num_samples)
                coords_list.append(coords)
                features_list.append(features)

        # Decoding path
        for i, decoder in enumerate([self.decoder4, self.decoder3, self.decoder2, self.decoder1]):
            # Upsample
            target_coords = coords_list[-(i + 2)]
            features = self.nearest_interpolation(target_coords, coords, features)

            # Concatenate skip connection
            skip_features = features_list[-(i + 2)]
            features = torch.cat([features, skip_features], dim=-1)

            # Permute for conv
            features = features.permute(0, 2, 1).unsqueeze(-1)  # (B, C, N, 1)
            features = decoder(features)  # (B, C_out, N, 1)
            features = features.squeeze(-1).permute(0, 2, 1)  # (B, N, C_out)

            coords = target_coords

        # Final classification
        logits = self.fc_end(features)  # (B, N, num_classes)

        return logits


if __name__ == "__main__":
    # Test RandLA-Net
    print("Testing RandLA-Net...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create model
    model = RandLANet(num_classes=5, num_features=6).to(device)

    # Test input
    B, N = 2, 4096
    coords = torch.randn(B, N, 3).to(device)
    features = torch.randn(B, N, 6).to(device)  # XYZ + RGB

    # Forward pass
    with torch.no_grad():
        output = model(coords, features)

    print(f"Input coords shape: {coords.shape}")
    print(f"Input features shape: {features.shape}")
    print(f"Output shape: {output.shape}")

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {num_params:,}")

    print("RandLA-Net test passed!")
