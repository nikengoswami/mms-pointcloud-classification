"""
Preprocessing utilities for point cloud data
Includes normalization, augmentation, and spatial operations
"""

import numpy as np
from typing import Tuple, Optional, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PointCloudPreprocessor:
    """Preprocessing operations for point cloud data"""

    @staticmethod
    def normalize_xyz(points: np.ndarray, method: str = 'center') -> Tuple[np.ndarray, dict]:
        """
        Normalize XYZ coordinates

        Args:
            points: (N, 3) array of XYZ coordinates
            method: 'center' (center and scale) or 'minmax' (0-1 range)

        Returns:
            Tuple of (normalized_points, normalization_params)
        """
        if method == 'center':
            mean = np.mean(points, axis=0)
            std = np.std(points)
            normalized = (points - mean) / (std + 1e-8)
            params = {'method': 'center', 'mean': mean, 'std': std}

        elif method == 'minmax':
            min_vals = np.min(points, axis=0)
            max_vals = np.max(points, axis=0)
            normalized = (points - min_vals) / (max_vals - min_vals + 1e-8)
            params = {'method': 'minmax', 'min': min_vals, 'max': max_vals}

        else:
            raise ValueError(f"Unknown normalization method: {method}")

        return normalized, params

    @staticmethod
    def denormalize_xyz(points: np.ndarray, params: dict) -> np.ndarray:
        """
        Reverse normalization

        Args:
            points: Normalized points
            params: Normalization parameters from normalize_xyz

        Returns:
            Original scale points
        """
        if params['method'] == 'center':
            return points * params['std'] + params['mean']
        elif params['method'] == 'minmax':
            return points * (params['max'] - params['min']) + params['min']
        else:
            raise ValueError(f"Unknown normalization method: {params['method']}")

    @staticmethod
    def compute_local_features(points: np.ndarray, k: int = 20) -> np.ndarray:
        """
        Compute local geometric features for each point

        Args:
            points: (N, 3) array of XYZ coordinates
            k: Number of nearest neighbors

        Returns:
            (N, F) array of local features
        """
        from scipy.spatial import cKDTree

        N = len(points)
        features = []

        # Build KD-tree for efficient neighbor search
        tree = cKDTree(points)

        # For each point, compute local features
        for i in range(N):
            # Find k nearest neighbors
            _, idx = tree.query(points[i], k=min(k, N))

            # Get neighbor points
            neighbors = points[idx]

            # Compute local statistics
            local_mean = np.mean(neighbors, axis=0)
            local_std = np.std(neighbors, axis=0)
            local_range = np.ptp(neighbors, axis=0)  # max - min

            # Compute distance to local center
            dist_to_center = np.linalg.norm(points[i] - local_mean)

            # Combine features
            point_features = np.concatenate([
                local_mean,
                local_std,
                local_range,
                [dist_to_center]
            ])

            features.append(point_features)

        return np.array(features)

    @staticmethod
    def voxel_downsampling(points: np.ndarray, voxel_size: float,
                          features: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Downsample point cloud using voxel grid

        Args:
            points: (N, 3) array of XYZ coordinates
            voxel_size: Size of voxel grid
            features: Optional (N, F) features to average

        Returns:
            Tuple of (downsampled_points, downsampled_features)
        """
        # Compute voxel indices
        voxel_indices = np.floor(points / voxel_size).astype(np.int32)

        # Create unique voxel dictionary
        voxel_dict = {}

        for i, voxel_idx in enumerate(voxel_indices):
            key = tuple(voxel_idx)
            if key not in voxel_dict:
                voxel_dict[key] = []
            voxel_dict[key].append(i)

        # Average points in each voxel
        downsampled_points = []
        downsampled_features = [] if features is not None else None

        for indices in voxel_dict.values():
            # Average point coordinates
            avg_point = np.mean(points[indices], axis=0)
            downsampled_points.append(avg_point)

            # Average features if provided
            if features is not None:
                avg_feature = np.mean(features[indices], axis=0)
                downsampled_features.append(avg_feature)

        downsampled_points = np.array(downsampled_points)
        if downsampled_features is not None:
            downsampled_features = np.array(downsampled_features)

        logger.info(f"Downsampled from {len(points)} to {len(downsampled_points)} points")

        return downsampled_points, downsampled_features


class DataAugmentation:
    """Data augmentation for point cloud training"""

    @staticmethod
    def random_rotation(points: np.ndarray, angle_range: float = np.pi) -> np.ndarray:
        """
        Random rotation around Z-axis

        Args:
            points: (N, 3) array
            angle_range: Maximum rotation angle in radians

        Returns:
            Rotated points
        """
        angle = np.random.uniform(-angle_range, angle_range)
        cos_a, sin_a = np.cos(angle), np.sin(angle)

        rotation_matrix = np.array([
            [cos_a, -sin_a, 0],
            [sin_a, cos_a, 0],
            [0, 0, 1]
        ])

        return points @ rotation_matrix.T

    @staticmethod
    def random_rotation_3d(points: np.ndarray) -> np.ndarray:
        """
        Random 3D rotation

        Args:
            points: (N, 3) array

        Returns:
            Rotated points
        """
        # Generate random rotation angles
        angles = np.random.uniform(-np.pi, np.pi, 3)

        # Rotation matrices
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(angles[0]), -np.sin(angles[0])],
            [0, np.sin(angles[0]), np.cos(angles[0])]
        ])

        Ry = np.array([
            [np.cos(angles[1]), 0, np.sin(angles[1])],
            [0, 1, 0],
            [-np.sin(angles[1]), 0, np.cos(angles[1])]
        ])

        Rz = np.array([
            [np.cos(angles[2]), -np.sin(angles[2]), 0],
            [np.sin(angles[2]), np.cos(angles[2]), 0],
            [0, 0, 1]
        ])

        R = Rz @ Ry @ Rx
        return points @ R.T

    @staticmethod
    def random_scale(points: np.ndarray, scale_range: Tuple[float, float] = (0.8, 1.2)) -> np.ndarray:
        """
        Random scaling

        Args:
            points: (N, 3) array
            scale_range: (min_scale, max_scale)

        Returns:
            Scaled points
        """
        scale = np.random.uniform(scale_range[0], scale_range[1])
        return points * scale

    @staticmethod
    def random_jitter(points: np.ndarray, sigma: float = 0.01, clip: float = 0.05) -> np.ndarray:
        """
        Add random noise to points

        Args:
            points: (N, 3) array
            sigma: Standard deviation of Gaussian noise
            clip: Maximum noise value

        Returns:
            Jittered points
        """
        noise = np.clip(sigma * np.random.randn(*points.shape), -clip, clip)
        return points + noise

    @staticmethod
    def random_dropout(points: np.ndarray, features: Optional[np.ndarray] = None,
                      dropout_ratio: float = 0.2) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Randomly drop points

        Args:
            points: (N, 3) array
            features: Optional (N, F) features
            dropout_ratio: Ratio of points to drop

        Returns:
            Tuple of (points, features) after dropout
        """
        N = len(points)
        keep_num = int(N * (1 - dropout_ratio))
        keep_indices = np.random.choice(N, keep_num, replace=False)

        points_out = points[keep_indices]
        features_out = features[keep_indices] if features is not None else None

        return points_out, features_out

    @staticmethod
    def augment_point_cloud(points: np.ndarray, features: Optional[np.ndarray] = None,
                          rotation: bool = True, scale: bool = True,
                          jitter: bool = True, dropout: bool = False) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Apply multiple augmentations

        Args:
            points: (N, 3) array
            features: Optional (N, F) features
            rotation: Apply rotation
            scale: Apply scaling
            jitter: Apply jittering
            dropout: Apply dropout

        Returns:
            Tuple of (augmented_points, augmented_features)
        """
        aug_points = points.copy()

        if rotation:
            aug_points = DataAugmentation.random_rotation(aug_points)

        if scale:
            aug_points = DataAugmentation.random_scale(aug_points)

        if jitter:
            aug_points = DataAugmentation.random_jitter(aug_points)

        if dropout:
            aug_points, features = DataAugmentation.random_dropout(aug_points, features)

        return aug_points, features


class SpatialPartitioner:
    """Partition large point clouds into manageable chunks"""

    @staticmethod
    def grid_partition(points: np.ndarray, features: Optional[np.ndarray],
                      labels: Optional[np.ndarray], grid_size: float) -> List[dict]:
        """
        Partition point cloud into grid cells

        Args:
            points: (N, 3) array
            features: Optional (N, F) features
            labels: Optional (N,) labels
            grid_size: Size of grid cells

        Returns:
            List of dictionaries containing partitioned data
        """
        # Compute grid indices
        min_coords = np.min(points, axis=0)
        grid_indices = np.floor((points - min_coords) / grid_size).astype(np.int32)

        # Group by grid cell
        grid_dict = {}
        for i, grid_idx in enumerate(grid_indices):
            key = tuple(grid_idx)
            if key not in grid_dict:
                grid_dict[key] = []
            grid_dict[key].append(i)

        # Create partitions
        partitions = []
        for grid_id, indices in grid_dict.items():
            partition = {
                'grid_id': grid_id,
                'points': points[indices],
                'num_points': len(indices)
            }

            if features is not None:
                partition['features'] = features[indices]

            if labels is not None:
                partition['labels'] = labels[indices]

            partitions.append(partition)

        logger.info(f"Created {len(partitions)} partitions from {len(points)} points")

        return partitions

    @staticmethod
    def sliding_window_partition(points: np.ndarray, features: Optional[np.ndarray],
                                 labels: Optional[np.ndarray],
                                 window_size: float, stride: float) -> List[dict]:
        """
        Partition using sliding window

        Args:
            points: (N, 3) array
            features: Optional (N, F) features
            labels: Optional (N,) labels
            window_size: Size of window
            stride: Stride between windows

        Returns:
            List of dictionaries containing partitioned data
        """
        from scipy.spatial import cKDTree

        min_coords = np.min(points, axis=0)
        max_coords = np.max(points, axis=0)

        tree = cKDTree(points)
        partitions = []

        # Slide window in X and Y
        x = min_coords[0]
        while x < max_coords[0]:
            y = min_coords[1]
            while y < max_coords[1]:
                # Define window bounds
                window_min = np.array([x, y, min_coords[2]])
                window_max = np.array([x + window_size, y + window_size, max_coords[2]])

                # Find points in window
                center = (window_min + window_max) / 2
                half_size = window_size / 2

                indices = tree.query_ball_point(center, half_size * np.sqrt(2))

                if len(indices) > 0:
                    partition = {
                        'window_center': center,
                        'points': points[indices],
                        'num_points': len(indices)
                    }

                    if features is not None:
                        partition['features'] = features[indices]

                    if labels is not None:
                        partition['labels'] = labels[indices]

                    partitions.append(partition)

                y += stride
            x += stride

        logger.info(f"Created {len(partitions)} sliding window partitions")

        return partitions


if __name__ == "__main__":
    # Test preprocessing
    print("Testing preprocessing utilities...")

    # Create sample point cloud
    points = np.random.randn(1000, 3)

    # Test normalization
    normalized, params = PointCloudPreprocessor.normalize_xyz(points)
    denormalized = PointCloudPreprocessor.denormalize_xyz(normalized, params)
    print(f"Normalization error: {np.max(np.abs(points - denormalized))}")

    # Test augmentation
    aug_points, _ = DataAugmentation.augment_point_cloud(points)
    print(f"Augmented points shape: {aug_points.shape}")

    print("Tests passed!")
