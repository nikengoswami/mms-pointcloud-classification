"""
LAS File I/O Utilities for Point Cloud Processing
Handles reading, writing, and basic manipulation of LAS files
"""

import numpy as np
import laspy
from pathlib import Path
from typing import Tuple, Optional, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LASProcessor:
    """Process LAS point cloud files for classification tasks"""

    def __init__(self, las_path: str):
        """
        Initialize LAS processor

        Args:
            las_path: Path to LAS file
        """
        self.las_path = Path(las_path)
        self.las_data = None
        self.points = None
        self.features = {}

    def read_las(self) -> Dict:
        """
        Read LAS file and extract all available features

        Returns:
            Dictionary containing point cloud data and metadata
        """
        try:
            logger.info(f"Reading LAS file: {self.las_path}")
            self.las_data = laspy.read(self.las_path)

            # Extract XYZ coordinates
            xyz = np.vstack((
                self.las_data.x,
                self.las_data.y,
                self.las_data.z
            )).T

            self.points = xyz

            # Extract available features
            self.features = {
                'xyz': xyz,
                'num_points': len(xyz)
            }

            # Check for RGB colors
            if hasattr(self.las_data, 'red'):
                rgb = np.vstack((
                    self.las_data.red,
                    self.las_data.green,
                    self.las_data.blue
                )).T
                # Normalize to 0-1 range (LAS stores as 16-bit)
                rgb = rgb / 65535.0
                self.features['rgb'] = rgb
                logger.info("RGB colors found")

            # Check for intensity
            if hasattr(self.las_data, 'intensity'):
                intensity = np.array(self.las_data.intensity)
                self.features['intensity'] = intensity
                logger.info("Intensity values found")

            # Check for classification
            if hasattr(self.las_data, 'classification'):
                classification = np.array(self.las_data.classification)
                self.features['classification'] = classification
                unique_classes = np.unique(classification)
                logger.info(f"Classification found. Unique classes: {unique_classes}")

            # Check for return number
            if hasattr(self.las_data, 'return_number'):
                self.features['return_number'] = np.array(self.las_data.return_number)
                logger.info("Return number found")

            # Check for number of returns
            if hasattr(self.las_data, 'number_of_returns'):
                self.features['number_of_returns'] = np.array(self.las_data.number_of_returns)
                logger.info("Number of returns found")

            # Get bounding box
            self.features['bbox'] = {
                'min_x': self.las_data.header.mins[0],
                'min_y': self.las_data.header.mins[1],
                'min_z': self.las_data.header.mins[2],
                'max_x': self.las_data.header.maxs[0],
                'max_y': self.las_data.header.maxs[1],
                'max_z': self.las_data.header.maxs[2]
            }

            logger.info(f"Successfully loaded {self.features['num_points']:,} points")
            logger.info(f"Bounding box: {self.features['bbox']}")

            return self.features

        except Exception as e:
            logger.error(f"Error reading LAS file: {e}")
            raise

    def get_statistics(self) -> Dict:
        """
        Calculate statistics of the point cloud

        Returns:
            Dictionary with various statistics
        """
        if self.features is None or len(self.features) == 0:
            raise ValueError("No data loaded. Call read_las() first.")

        stats = {
            'num_points': self.features['num_points'],
            'xyz_mean': np.mean(self.features['xyz'], axis=0).tolist(),
            'xyz_std': np.std(self.features['xyz'], axis=0).tolist(),
            'xyz_min': np.min(self.features['xyz'], axis=0).tolist(),
            'xyz_max': np.max(self.features['xyz'], axis=0).tolist(),
        }

        if 'rgb' in self.features:
            stats['rgb_mean'] = np.mean(self.features['rgb'], axis=0).tolist()
            stats['rgb_std'] = np.std(self.features['rgb'], axis=0).tolist()

        if 'intensity' in self.features:
            stats['intensity_mean'] = float(np.mean(self.features['intensity']))
            stats['intensity_std'] = float(np.std(self.features['intensity']))
            stats['intensity_min'] = float(np.min(self.features['intensity']))
            stats['intensity_max'] = float(np.max(self.features['intensity']))

        if 'classification' in self.features:
            unique, counts = np.unique(self.features['classification'], return_counts=True)
            stats['class_distribution'] = {
                int(cls): int(count) for cls, count in zip(unique, counts)
            }

        return stats

    def extract_features_for_training(self, normalize: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Extract and combine features for model training

        Args:
            normalize: Whether to normalize features

        Returns:
            Tuple of (features, labels)
            features: (N, F) array where F is number of features
            labels: (N,) array of classification labels (or None if not available)
        """
        if self.features is None or len(self.features) == 0:
            raise ValueError("No data loaded. Call read_las() first.")

        feature_list = []

        # Always include XYZ
        xyz = self.features['xyz'].copy()
        if normalize:
            # Center and normalize XYZ
            xyz_mean = np.mean(xyz, axis=0)
            xyz = xyz - xyz_mean
            xyz_std = np.std(xyz)
            xyz = xyz / xyz_std
        feature_list.append(xyz)

        # Add RGB if available
        if 'rgb' in self.features:
            feature_list.append(self.features['rgb'])

        # Add intensity if available
        if 'intensity' in self.features:
            intensity = self.features['intensity'].reshape(-1, 1)
            if normalize:
                intensity = (intensity - np.min(intensity)) / (np.max(intensity) - np.min(intensity) + 1e-8)
            feature_list.append(intensity)

        # Add return number features if available
        if 'return_number' in self.features:
            return_num = self.features['return_number'].reshape(-1, 1)
            feature_list.append(return_num / 5.0)  # Normalize (usually max 5 returns)

        if 'number_of_returns' in self.features:
            num_returns = self.features['number_of_returns'].reshape(-1, 1)
            feature_list.append(num_returns / 5.0)

        # Combine all features
        features = np.hstack(feature_list)

        # Extract labels if available
        labels = self.features.get('classification', None)

        logger.info(f"Extracted features shape: {features.shape}")
        if labels is not None:
            logger.info(f"Labels shape: {labels.shape}")

        return features, labels

    def write_classified_las(self, output_path: str, classifications: np.ndarray):
        """
        Write point cloud with new classification to LAS file

        Args:
            output_path: Path for output LAS file
            classifications: (N,) array of classification labels
        """
        if self.las_data is None:
            raise ValueError("No LAS data loaded. Call read_las() first.")

        try:
            # Create a copy of the original LAS data
            output_las = laspy.LasData(self.las_data.header)
            output_las.points = self.las_data.points.copy()

            # Update classification
            output_las.classification = classifications.astype(np.uint8)

            # Write to file
            output_las.write(output_path)
            logger.info(f"Classified point cloud saved to: {output_path}")

        except Exception as e:
            logger.error(f"Error writing LAS file: {e}")
            raise

    def sample_points(self, num_samples: int, method: str = 'random') -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample points from the point cloud

        Args:
            num_samples: Number of points to sample
            method: Sampling method ('random' or 'fps' for farthest point sampling)

        Returns:
            Tuple of (sampled_indices, sampled_points)
        """
        if self.points is None:
            raise ValueError("No points loaded. Call read_las() first.")

        num_points = len(self.points)

        if method == 'random':
            indices = np.random.choice(num_points, size=min(num_samples, num_points), replace=False)
        elif method == 'fps':
            # Simplified FPS (for large clouds, this is slow)
            indices = self._farthest_point_sampling(self.points, num_samples)
        else:
            raise ValueError(f"Unknown sampling method: {method}")

        return indices, self.points[indices]

    def _farthest_point_sampling(self, points: np.ndarray, num_samples: int) -> np.ndarray:
        """
        Farthest point sampling algorithm

        Args:
            points: (N, 3) array of points
            num_samples: Number of points to sample

        Returns:
            (num_samples,) array of indices
        """
        num_points = len(points)
        num_samples = min(num_samples, num_points)

        selected_indices = np.zeros(num_samples, dtype=np.int32)
        distances = np.ones(num_points) * 1e10

        # Start with a random point
        current_idx = np.random.randint(0, num_points)
        selected_indices[0] = current_idx

        for i in range(1, num_samples):
            # Update distances
            current_point = points[current_idx]
            dist = np.sum((points - current_point) ** 2, axis=1)
            distances = np.minimum(distances, dist)

            # Select farthest point
            current_idx = np.argmax(distances)
            selected_indices[i] = current_idx

        return selected_indices


def visualize_las_statistics(las_path: str, save_path: Optional[str] = None):
    """
    Visualize statistics of a LAS file

    Args:
        las_path: Path to LAS file
        save_path: Optional path to save visualization
    """
    import matplotlib.pyplot as plt

    processor = LASProcessor(las_path)
    processor.read_las()
    stats = processor.get_statistics()

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Plot 1: XYZ distribution
    ax = axes[0, 0]
    xyz = processor.features['xyz']
    ax.scatter(xyz[:, 0], xyz[:, 1], c=xyz[:, 2], s=0.1, cmap='viridis')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('XY Distribution (colored by Z)')
    ax.axis('equal')

    # Plot 2: Height distribution
    ax = axes[0, 1]
    ax.hist(xyz[:, 2], bins=100, edgecolor='black')
    ax.set_xlabel('Z (Height)')
    ax.set_ylabel('Count')
    ax.set_title('Height Distribution')
    ax.grid(True, alpha=0.3)

    # Plot 3: RGB distribution (if available)
    ax = axes[1, 0]
    if 'rgb' in processor.features:
        rgb = processor.features['rgb']
        colors = ['red', 'green', 'blue']
        for i, color in enumerate(colors):
            ax.hist(rgb[:, i], bins=50, alpha=0.5, label=color.upper(), color=color)
        ax.set_xlabel('Color Value')
        ax.set_ylabel('Count')
        ax.set_title('RGB Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No RGB data', ha='center', va='center')

    # Plot 4: Classification distribution (if available)
    ax = axes[1, 1]
    if 'classification' in processor.features and 'class_distribution' in stats:
        class_dist = stats['class_distribution']
        classes = list(class_dist.keys())
        counts = list(class_dist.values())
        ax.bar(classes, counts, edgecolor='black')
        ax.set_xlabel('Class')
        ax.set_ylabel('Count')
        ax.set_title('Classification Distribution')
        ax.grid(True, alpha=0.3, axis='y')
    else:
        ax.text(0.5, 0.5, 'No classification data', ha='center', va='center')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Visualization saved to: {save_path}")

    plt.show()


if __name__ == "__main__":
    # Example usage
    las_path = "../Classified.las"

    processor = LASProcessor(las_path)
    features_dict = processor.read_las()
    stats = processor.get_statistics()

    print("\n=== LAS File Statistics ===")
    for key, value in stats.items():
        print(f"{key}: {value}")

    # Extract features for training
    features, labels = processor.extract_features_for_training(normalize=True)
    print(f"\nFeatures shape: {features.shape}")
    print(f"Labels shape: {labels.shape if labels is not None else 'None'}")
