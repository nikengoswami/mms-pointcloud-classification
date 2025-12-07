"""
PointNet++ Implementation for Point Cloud Classification
Simpler and more stable than RandLA-Net for initial training

Reference: Qi et al., "PointNet++: Deep Hierarchical Feature Learning on Point Sets"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class PointNetSetAbstraction(nn.Module):
    """Set Abstraction layer for PointNet++"""

    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all=False):
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.group_all = group_all

        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()

        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

    def forward(self, xyz, points):
        """
        Args:
            xyz: (B, N, 3) coordinates
            points: (B, N, C) features
        Returns:
            new_xyz: (B, npoint, 3)
            new_points: (B, C', npoint)
        """
        if self.group_all:
            new_xyz = xyz[:, 0:1, :]
            grouped_xyz = xyz.unsqueeze(2)
            if points is not None:
                grouped_points = points.unsqueeze(2)
                new_points = torch.cat([grouped_xyz, grouped_points], dim=-1)
            else:
                new_points = grouped_xyz
        else:
            # Farthest point sampling
            new_xyz = self.farthest_point_sample(xyz, self.npoint)

            # Group points
            grouped_xyz, grouped_points = self.query_ball_point(
                self.radius, self.nsample, xyz, new_xyz, points
            )

            # Relative coordinates
            grouped_xyz -= new_xyz.unsqueeze(2)

            if grouped_points is not None:
                new_points = torch.cat([grouped_xyz, grouped_points], dim=-1)
            else:
                new_points = grouped_xyz

        # (B, N, K, C) -> (B, C, N, K)
        new_points = new_points.permute(0, 3, 1, 2)

        # Apply MLPs
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))

        # Max pooling
        new_points = torch.max(new_points, dim=3)[0]  # (B, C, N)

        return new_xyz, new_points

    @staticmethod
    def farthest_point_sample(xyz, npoint):
        """
        Farthest point sampling
        Args:
            xyz: (B, N, 3)
            npoint: number of points to sample
        Returns:
            centroids: (B, npoint, 3)
        """
        device = xyz.device
        B, N, _ = xyz.shape

        centroids = torch.zeros(B, npoint, 3, device=device)
        distance = torch.ones(B, N, device=device) * 1e10
        farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
        batch_indices = torch.arange(B, dtype=torch.long, device=device)

        for i in range(npoint):
            centroids[:, i, :] = xyz[batch_indices, farthest, :]
            centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
            dist = torch.sum((xyz - centroid) ** 2, dim=-1)
            mask = dist < distance
            distance[mask] = dist[mask]
            farthest = torch.max(distance, dim=-1)[1]

        return centroids

    @staticmethod
    def query_ball_point(radius, nsample, xyz, new_xyz, points):
        """
        Query ball point
        Args:
            radius: search radius
            nsample: max number of samples
            xyz: (B, N, 3) all points
            new_xyz: (B, npoint, 3) query points
            points: (B, N, C) features
        Returns:
            grouped_xyz: (B, npoint, nsample, 3)
            grouped_points: (B, npoint, nsample, C) or None
        """
        device = xyz.device
        B, N, _ = xyz.shape
        _, npoint, _ = new_xyz.shape

        # Compute squared distances
        dists = torch.cdist(new_xyz, xyz)  # (B, npoint, N)

        # Find points within radius
        idx = torch.argsort(dists, dim=-1)[:, :, :nsample]  # (B, npoint, nsample)

        # Gather points
        grouped_xyz = torch.gather(
            xyz.unsqueeze(1).expand(-1, npoint, -1, -1),
            2,
            idx.unsqueeze(-1).expand(-1, -1, -1, 3)
        )

        if points is not None:
            grouped_points = torch.gather(
                points.unsqueeze(1).expand(-1, npoint, -1, -1),
                2,
                idx.unsqueeze(-1).expand(-1, -1, -1, points.shape[-1])
            )
        else:
            grouped_points = None

        return grouped_xyz, grouped_points


class PointNetFeaturePropagation(nn.Module):
    """Feature propagation layer"""

    def __init__(self, in_channel, mlp):
        super().__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()

        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Args:
            xyz1: (B, N, 3) coordinates of target points
            xyz2: (B, M, 3) coordinates of source points
            points1: (B, C1, N) features of target points
            points2: (B, C2, M) features of source points
        Returns:
            new_points: (B, mlp[-1], N)
        """
        if points2 is None:
            points2 = xyz2.permute(0, 2, 1)

        B, N, _ = xyz1.shape
        _, M, _ = xyz2.shape

        if M == 1:
            # Broadcast
            interpolated_points = points2.expand(-1, -1, N)
        else:
            # Find 3 nearest neighbors
            dists = torch.cdist(xyz1, xyz2)  # (B, N, M)
            idx = torch.topk(dists, 3, dim=-1, largest=False, sorted=True)[1]  # (B, N, 3)

            dist_recip = 1.0 / (dists.gather(2, idx) + 1e-8)  # (B, N, 3)
            norm = torch.sum(dist_recip, dim=-1, keepdim=True)
            weight = dist_recip / norm  # (B, N, 3)

            # Gather neighbor features
            neighbor_points = torch.gather(
                points2.unsqueeze(2).expand(-1, -1, N, -1),
                3,
                idx.unsqueeze(1).expand(-1, points2.shape[1], -1, -1)
            )  # (B, C, N, 3)

            # Weighted sum
            interpolated_points = torch.sum(
                neighbor_points * weight.unsqueeze(1),
                dim=-1
            )  # (B, C, N)

        # Concatenate
        if points1 is not None:
            new_points = torch.cat([points1, interpolated_points], dim=1)
        else:
            new_points = interpolated_points

        # Apply MLPs
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))

        return new_points


class PointNet2(nn.Module):
    """PointNet++ for semantic segmentation"""

    def __init__(self, num_classes=5, num_features=6):
        super().__init__()

        self.num_classes = num_classes
        self.num_features = num_features

        # Set Abstraction layers (encoder)
        self.sa1 = PointNetSetAbstraction(
            npoint=1024, radius=0.1, nsample=32,
            in_channel=num_features + 3, mlp=[32, 32, 64]  # +3 for concatenated XYZ coords
        )
        self.sa2 = PointNetSetAbstraction(
            npoint=256, radius=0.2, nsample=32,
            in_channel=64 + 3, mlp=[64, 64, 128]
        )
        self.sa3 = PointNetSetAbstraction(
            npoint=64, radius=0.4, nsample=32,
            in_channel=128 + 3, mlp=[128, 128, 256]
        )
        self.sa4 = PointNetSetAbstraction(
            npoint=16, radius=0.8, nsample=32,
            in_channel=256 + 3, mlp=[256, 256, 512]
        )

        # Feature Propagation layers (decoder)
        self.fp4 = PointNetFeaturePropagation(768, [256, 256])
        self.fp3 = PointNetFeaturePropagation(384, [256, 256])
        self.fp2 = PointNetFeaturePropagation(320, [256, 128])
        self.fp1 = PointNetFeaturePropagation(128 + num_features, [128, 128, 128])

        # Final classifier
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_classes, 1)

    def forward(self, coords, features):
        """
        Args:
            coords: (B, N, 3)
            features: (B, N, F)
        Returns:
            logits: (B, N, num_classes)
        """
        # Encoder
        l0_xyz = coords
        l0_points = features

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l1_points = l1_points.permute(0, 2, 1)  # (B, C, N) -> (B, N, C)

        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l2_points = l2_points.permute(0, 2, 1)  # (B, C, N) -> (B, N, C)

        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l3_points = l3_points.permute(0, 2, 1)  # (B, C, N) -> (B, N, C)

        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)
        l4_points = l4_points.permute(0, 2, 1)  # (B, C, N) -> (B, N, C)

        # Decoder - fp layers expect (B, C, N) format
        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points.permute(0, 2, 1), l4_points.permute(0, 2, 1))
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points.permute(0, 2, 1), l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points.permute(0, 2, 1), l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, l0_points.permute(0, 2, 1), l1_points)

        # Classifier
        x = self.drop1(F.relu(self.bn1(self.conv1(l0_points))))
        x = self.conv2(x)
        x = x.permute(0, 2, 1)  # (B, N, C)

        return x


if __name__ == "__main__":
    # Test
    print("Testing PointNet++...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PointNet2(num_classes=5, num_features=6).to(device)

    # Test input
    B, N = 2, 4096
    coords = torch.randn(B, N, 3).to(device)
    features = torch.randn(B, N, 6).to(device)

    # Forward
    with torch.no_grad():
        output = model(coords, features)

    print(f"Input: {coords.shape}, {features.shape}")
    print(f"Output: {output.shape}")

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {num_params:,}")

    print("PointNet++ test passed!")
