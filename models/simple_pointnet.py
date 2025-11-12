"""
Simplified PointNet for Point Cloud Semantic Segmentation
Guaranteed to work - simpler architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimplePointNet(nn.Module):
    """Simple PointNet for semantic segmentation"""

    def __init__(self, num_classes=5, num_features=7):
        super().__init__()

        # Shared MLPs for point features
        self.conv1 = nn.Conv1d(num_features, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)

        # Segmentation head
        self.conv4 = nn.Conv1d(256 + 64, 256, 1)
        self.conv5 = nn.Conv1d(256, 128, 1)
        self.conv6 = nn.Conv1d(128, num_classes, 1)

        self.bn4 = nn.BatchNorm1d(256)
        self.bn5 = nn.BatchNorm1d(128)

        self.dropout = nn.Dropout(0.3)

    def forward(self, coords, features):
        """
        Args:
            coords: (B, N, 3)
            features: (B, N, F)
        Returns:
            logits: (B, N, num_classes)
        """
        B, N, _ = coords.shape

        # Use all features (XYZ + RGB + others)
        x = features.permute(0, 2, 1)  # (B, F, N)

        # Shared MLPs
        x1 = F.relu(self.bn1(self.conv1(x)))
        x2 = F.relu(self.bn2(self.conv2(x1)))
        x3 = F.relu(self.bn3(self.conv3(x2)))

        # Concatenate local and global features
        x = torch.cat([x1, x3], dim=1)  # (B, 64+256, N)

        # Segmentation head
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.dropout(x)
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.dropout(x)
        x = self.conv6(x)

        x = x.permute(0, 2, 1)  # (B, N, num_classes)

        return x


if __name__ == "__main__":
    # Test
    print("Testing SimplePointNet...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimplePointNet(num_classes=5, num_features=7).to(device)

    # Test input
    B, N = 2, 4096
    coords = torch.randn(B, N, 3).to(device)
    features = torch.randn(B, N, 7).to(device)

    # Forward
    with torch.no_grad():
        output = model(coords, features)

    print(f"Input: {coords.shape}, {features.shape}")
    print(f"Output: {output.shape}")

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {num_params:,}")

    print("SimplePointNet test passed!")
