"""
PyTorch Dataset for Point Cloud Classification
Handles loading, batching, and augmentation of point cloud data
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, List
import logging
import sys

sys.path.append(str(Path(__file__).parent.parent))
from utils.las_io import LASProcessor
from utils.preprocessing import DataAugmentation, PointCloudPreprocessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PointCloudDataset(Dataset):
    """Dataset for point cloud semantic segmentation"""

    def __init__(self, las_files: List[str], num_points: int = 4096,
                 augment: bool = False, normalize: bool = True,
                 class_mapping: Optional[Dict] = None):
        """
        Args:
            las_files: List of paths to LAS files
            num_points: Number of points per sample
            augment: Whether to apply data augmentation
            normalize: Whether to normalize coordinates
            class_mapping: Optional mapping from original classes to target classes
        """
        self.las_files = las_files
        self.num_points = num_points
        self.augment = augment
        self.normalize = normalize
        self.class_mapping = class_mapping or {}

        # Load all point clouds
        self.data = []
        self.load_data()

    def load_data(self):
        """Load all point cloud data from LAS files"""
        logger.info(f"Loading {len(self.las_files)} LAS files...")

        for las_file in self.las_files:
            processor = LASProcessor(las_file)
            features_dict = processor.read_las()

            # Extract features and labels
            xyz = features_dict['xyz']

            # Build feature matrix
            feature_list = [xyz]

            # Add RGB if available
            if 'rgb' in features_dict:
                feature_list.append(features_dict['rgb'])

            # Add intensity if available
            if 'intensity' in features_dict:
                intensity = features_dict['intensity'].reshape(-1, 1)
                # Normalize intensity
                intensity = (intensity - np.min(intensity)) / (np.max(intensity) - np.min(intensity) + 1e-8)
                feature_list.append(intensity)

            features = np.hstack(feature_list).astype(np.float32)

            # Get labels
            labels = None
            if 'classification' in features_dict:
                labels = features_dict['classification'].astype(np.int64)

                # Apply class mapping if provided
                if self.class_mapping:
                    labels_mapped = np.zeros_like(labels)
                    for orig_class, target_class in self.class_mapping.items():
                        labels_mapped[labels == orig_class] = target_class
                    labels = labels_mapped

            # Store data
            self.data.append({
                'features': features,
                'labels': labels,
                'xyz': xyz,
                'file': las_file
            })

        logger.info(f"Loaded {len(self.data)} point clouds")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict:
        """
        Get a sample from the dataset

        Returns:
            Dictionary with:
                - coords: (num_points, 3) coordinates
                - features: (num_points, F) features
                - labels: (num_points,) labels (if available)
        """
        data_item = self.data[idx]

        features = data_item['features']
        xyz = data_item['xyz']
        labels = data_item['labels']

        # Sample points if more than num_points
        num_total_points = len(features)

        if num_total_points >= self.num_points:
            # Random sampling
            choice = np.random.choice(num_total_points, self.num_points, replace=False)
        else:
            # Oversample if not enough points
            choice = np.random.choice(num_total_points, self.num_points, replace=True)

        sampled_xyz = xyz[choice]
        sampled_features = features[choice]
        sampled_labels = labels[choice] if labels is not None else None

        # Data augmentation
        if self.augment:
            # Apply rotation
            sampled_xyz = DataAugmentation.random_rotation(sampled_xyz)

            # Apply jittering
            sampled_xyz = DataAugmentation.random_jitter(sampled_xyz, sigma=0.01)

            # Apply scaling
            sampled_xyz = DataAugmentation.random_scale(sampled_xyz, scale_range=(0.9, 1.1))

        # Normalize coordinates
        if self.normalize:
            sampled_xyz, _ = PointCloudPreprocessor.normalize_xyz(sampled_xyz, method='center')

        # Update features with normalized XYZ
        sampled_features[:, :3] = sampled_xyz

        # Convert to tensors
        coords = torch.from_numpy(sampled_xyz).float()
        features = torch.from_numpy(sampled_features).float()

        result = {
            'coords': coords,
            'features': features,
            'file_idx': idx
        }

        if sampled_labels is not None:
            labels_tensor = torch.from_numpy(sampled_labels).long()
            result['labels'] = labels_tensor

        return result


class SpatialDataset(Dataset):
    """
    Dataset that partitions point cloud spatially into blocks
    Better for large-scale point clouds
    """

    def __init__(self, las_file: str, block_size: float = 50.0,
                 stride: Optional[float] = None, min_points: int = 100,
                 augment: bool = False, normalize: bool = True,
                 class_mapping: Optional[Dict] = None):
        """
        Args:
            las_file: Path to LAS file
            block_size: Size of spatial blocks
            stride: Stride for overlapping blocks (default: block_size)
            min_points: Minimum points per block
            augment: Whether to apply augmentation
            normalize: Whether to normalize coordinates
            class_mapping: Mapping from original to target classes
        """
        self.las_file = las_file
        self.block_size = block_size
        self.stride = stride or block_size
        self.min_points = min_points
        self.augment = augment
        self.normalize = normalize
        self.class_mapping = class_mapping or {}

        # Load and partition data
        self.blocks = []
        self.load_and_partition()

    def load_and_partition(self):
        """Load point cloud and partition into spatial blocks"""
        logger.info(f"Loading and partitioning {self.las_file}...")

        # Load data
        processor = LASProcessor(self.las_file)
        features_dict = processor.read_las()

        xyz = features_dict['xyz']

        # Build feature matrix
        feature_list = [xyz]

        if 'rgb' in features_dict:
            feature_list.append(features_dict['rgb'])

        if 'intensity' in features_dict:
            intensity = features_dict['intensity'].reshape(-1, 1)
            intensity = (intensity - np.min(intensity)) / (np.max(intensity) - np.min(intensity) + 1e-8)
            feature_list.append(intensity)

        features = np.hstack(feature_list).astype(np.float32)

        # Get labels
        labels = None
        if 'classification' in features_dict:
            labels = features_dict['classification'].astype(np.int64)

            if self.class_mapping:
                labels_mapped = np.zeros_like(labels)
                for orig_class, target_class in self.class_mapping.items():
                    labels_mapped[labels == orig_class] = target_class
                labels = labels_mapped

        # Partition into blocks
        min_xyz = np.min(xyz, axis=0)
        max_xyz = np.max(xyz, axis=0)

        x_min, y_min = min_xyz[0], min_xyz[1]
        x_max, y_max = max_xyz[0], max_xyz[1]

        x = x_min
        while x < x_max:
            y = y_min
            while y < y_max:
                # Define block bounds
                block_min = np.array([x, y, min_xyz[2]])
                block_max = np.array([x + self.block_size, y + self.block_size, max_xyz[2]])

                # Find points in block
                mask = (
                    (xyz[:, 0] >= block_min[0]) & (xyz[:, 0] < block_max[0]) &
                    (xyz[:, 1] >= block_min[1]) & (xyz[:, 1] < block_max[1])
                )

                if np.sum(mask) >= self.min_points:
                    block_data = {
                        'xyz': xyz[mask],
                        'features': features[mask],
                        'labels': labels[mask] if labels is not None else None,
                        'bounds': (block_min, block_max)
                    }
                    self.blocks.append(block_data)

                y += self.stride
            x += self.stride

        logger.info(f"Created {len(self.blocks)} spatial blocks")

    def __len__(self) -> int:
        return len(self.blocks)

    def __getitem__(self, idx: int) -> Dict:
        """Get a spatial block"""
        block = self.blocks[idx]

        xyz = block['xyz'].copy()
        features = block['features'].copy()
        labels = block['labels'].copy() if block['labels'] is not None else None

        # Data augmentation
        if self.augment:
            xyz = DataAugmentation.random_rotation(xyz)
            xyz = DataAugmentation.random_jitter(xyz, sigma=0.01)
            xyz = DataAugmentation.random_scale(xyz, scale_range=(0.9, 1.1))

        # Normalize
        if self.normalize:
            xyz, _ = PointCloudPreprocessor.normalize_xyz(xyz, method='center')

        # Update features
        features[:, :3] = xyz

        # Convert to tensors
        coords = torch.from_numpy(xyz).float()
        features = torch.from_numpy(features).float()

        result = {
            'coords': coords,
            'features': features,
            'block_idx': idx,
            'num_points': len(coords)
        }

        if labels is not None:
            labels_tensor = torch.from_numpy(labels).long()
            result['labels'] = labels_tensor

        return result


def collate_fn(batch: List[Dict]) -> Dict:
    """
    Custom collate function for batching point clouds of different sizes

    Args:
        batch: List of data dictionaries

    Returns:
        Batched dictionary
    """
    # Get max number of points in batch
    max_points = max([item['coords'].shape[0] for item in batch])

    # Initialize batch tensors
    batch_size = len(batch)
    coords_batch = torch.zeros(batch_size, max_points, 3)
    features_batch = torch.zeros(batch_size, max_points, batch[0]['features'].shape[1])
    masks = torch.zeros(batch_size, max_points, dtype=torch.bool)

    labels_batch = None
    if 'labels' in batch[0]:
        labels_batch = torch.zeros(batch_size, max_points, dtype=torch.long)

    # Fill tensors
    for i, item in enumerate(batch):
        num_points = item['coords'].shape[0]

        coords_batch[i, :num_points] = item['coords']
        features_batch[i, :num_points] = item['features']
        masks[i, :num_points] = True

        if labels_batch is not None:
            labels_batch[i, :num_points] = item['labels']

    result = {
        'coords': coords_batch,
        'features': features_batch,
        'mask': masks
    }

    if labels_batch is not None:
        result['labels'] = labels_batch

    return result


def create_dataloaders(train_files: List[str], val_files: List[str],
                       batch_size: int = 8, num_points: int = 4096,
                       num_workers: int = 4, class_mapping: Optional[Dict] = None) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders

    Args:
        train_files: List of training LAS files
        val_files: List of validation LAS files
        batch_size: Batch size
        num_points: Points per sample
        num_workers: Number of worker threads
        class_mapping: Class mapping dictionary

    Returns:
        Tuple of (train_loader, val_loader)
    """
    train_dataset = PointCloudDataset(
        train_files,
        num_points=num_points,
        augment=True,
        normalize=True,
        class_mapping=class_mapping
    )

    val_dataset = PointCloudDataset(
        val_files,
        num_points=num_points,
        augment=False,
        normalize=True,
        class_mapping=class_mapping
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )

    return train_loader, val_loader


if __name__ == "__main__":
    # Test dataset
    print("Testing dataset...")

    # Create dummy dataset
    las_files = ["../Classified.las"]

    # Example class mapping
    # Map original LAS classes to our classes: Road, Snow, Vehicle, Vegetation, Others
    class_mapping = {
        11: 0,  # Road Surface -> Road
        2: 0,   # Ground -> Road
        3: 3,   # Low Vegetation -> Vegetation
        4: 3,   # Medium Vegetation -> Vegetation
        5: 3,   # High Vegetation -> Vegetation
        6: 4,   # Building -> Others
        # Add more mappings as needed
    }

    dataset = PointCloudDataset(las_files, num_points=2048, augment=True, class_mapping=class_mapping)

    print(f"Dataset size: {len(dataset)}")

    # Get sample
    sample = dataset[0]
    print(f"Coords shape: {sample['coords'].shape}")
    print(f"Features shape: {sample['features'].shape}")
    if 'labels' in sample:
        print(f"Labels shape: {sample['labels'].shape}")
        print(f"Unique labels: {torch.unique(sample['labels'])}")

    print("Dataset test passed!")
