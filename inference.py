"""
Inference script for RandLA-Net
Apply trained model to classify MMS point cloud data
"""

import torch
import numpy as np
from pathlib import Path
import argparse
import logging
from tqdm import tqdm

from models.randlanet import RandLANet
from utils.las_io import LASProcessor
from utils.preprocessing import PointCloudPreprocessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PointCloudClassifier:
    """Classify point clouds using trained RandLA-Net"""

    def __init__(self, model_path: str, num_classes: int = 5, num_features: int = 6,
                 device: torch.device = None):
        """
        Args:
            model_path: Path to trained model checkpoint
            num_classes: Number of classes
            num_features: Number of input features
            device: Device to run inference on
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_classes = num_classes
        self.num_features = num_features

        # Load model
        self.model = RandLANet(
            num_classes=num_classes,
            num_features=num_features
        ).to(self.device)

        logger.info(f"Loading model from {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device)

        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)

        self.model.eval()
        logger.info("Model loaded successfully")

    @torch.no_grad()
    def classify_points(self, coords: np.ndarray, features: np.ndarray,
                       batch_size: int = 4096, overlap: float = 0.0) -> np.ndarray:
        """
        Classify points in batches

        Args:
            coords: (N, 3) coordinates
            features: (N, F) features
            batch_size: Number of points per batch
            overlap: Overlap ratio between batches (0-1)

        Returns:
            (N,) array of predicted class labels
        """
        num_points = len(coords)
        predictions = np.zeros(num_points, dtype=np.int64)
        vote_counts = np.zeros(num_points, dtype=np.int32)

        # Calculate stride
        stride = int(batch_size * (1 - overlap))
        if stride <= 0:
            stride = batch_size

        # Process in batches
        num_batches = (num_points + stride - 1) // stride
        logger.info(f"Processing {num_points:,} points in {num_batches} batches")

        for i in tqdm(range(0, num_points, stride), desc="Classifying"):
            end_idx = min(i + batch_size, num_points)
            batch_coords = coords[i:end_idx]
            batch_features = features[i:end_idx]

            # Normalize coordinates
            norm_coords, _ = PointCloudPreprocessor.normalize_xyz(batch_coords, method='center')
            batch_features[:, :3] = norm_coords

            # Pad if necessary
            actual_size = len(batch_coords)
            if actual_size < batch_size:
                pad_size = batch_size - actual_size
                batch_coords = np.vstack([batch_coords, np.zeros((pad_size, 3))])
                batch_features = np.vstack([batch_features, np.zeros((pad_size, self.num_features))])

            # Convert to tensor
            coords_tensor = torch.from_numpy(batch_coords).unsqueeze(0).float().to(self.device)
            features_tensor = torch.from_numpy(batch_features).unsqueeze(0).float().to(self.device)

            # Forward pass
            logits = self.model(coords_tensor, features_tensor)  # (1, batch_size, num_classes)

            # Get predictions
            batch_preds = torch.argmax(logits[0], dim=1).cpu().numpy()

            # Store predictions (only for actual points, not padding)
            batch_preds = batch_preds[:actual_size]
            predictions[i:end_idx] += batch_preds
            vote_counts[i:end_idx] += 1

        # Average votes
        predictions = (predictions / np.maximum(vote_counts, 1)).astype(np.int64)

        return predictions

    def classify_las_file(self, input_las: str, output_las: str, batch_size: int = 4096):
        """
        Classify a LAS file and save results

        Args:
            input_las: Path to input LAS file
            output_las: Path to output classified LAS file
            batch_size: Batch size for processing
        """
        logger.info(f"Classifying {input_las}")

        # Load LAS file
        processor = LASProcessor(input_las)
        features_dict = processor.read_las()

        # Extract features
        xyz = features_dict['xyz']
        feature_list = [xyz]

        # Add RGB if available
        if 'rgb' in features_dict:
            feature_list.append(features_dict['rgb'])

        # Add intensity if available
        if 'intensity' in features_dict:
            intensity = features_dict['intensity'].reshape(-1, 1)
            intensity = (intensity - np.min(intensity)) / (np.max(intensity) - np.min(intensity) + 1e-8)
            feature_list.append(intensity)

        features = np.hstack(feature_list).astype(np.float32)

        logger.info(f"Point cloud has {len(xyz):,} points with {features.shape[1]} features")

        # Classify
        predictions = self.classify_points(xyz, features, batch_size=batch_size, overlap=0.2)

        # Save classified point cloud
        logger.info(f"Saving classified point cloud to {output_las}")
        processor.write_classified_las(output_las, predictions)

        # Print classification statistics
        unique, counts = np.unique(predictions, return_counts=True)
        logger.info("\nClassification Results:")
        class_names = ['Road', 'Snow', 'Vehicle', 'Vegetation', 'Others']
        for cls, count in zip(unique, counts):
            percentage = (count / len(predictions)) * 100
            class_name = class_names[cls] if cls < len(class_names) else f"Unknown ({cls})"
            logger.info(f"  Class {cls} ({class_name}): {count:,} points ({percentage:.2f}%)")

        return predictions

    def classify_spatial_blocks(self, input_las: str, output_las: str,
                               block_size: float = 50.0, batch_size: int = 4096):
        """
        Classify point cloud by spatial blocks (for very large files)

        Args:
            input_las: Path to input LAS file
            output_las: Path to output classified LAS file
            block_size: Size of spatial blocks in meters
            batch_size: Batch size for processing
        """
        logger.info(f"Classifying {input_las} using spatial blocks")
        logger.info(f"Block size: {block_size}m, Batch size: {batch_size}")

        # Load LAS file
        processor = LASProcessor(input_las)
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

        # Initialize predictions
        predictions = np.zeros(len(xyz), dtype=np.int64)

        # Partition into spatial blocks
        min_xyz = np.min(xyz, axis=0)
        max_xyz = np.max(xyz, axis=0)

        x_min, y_min = min_xyz[0], min_xyz[1]
        x_max, y_max = max_xyz[0], max_xyz[1]

        num_blocks_x = int(np.ceil((x_max - x_min) / block_size))
        num_blocks_y = int(np.ceil((y_max - y_min) / block_size))
        total_blocks = num_blocks_x * num_blocks_y

        logger.info(f"Processing {total_blocks} spatial blocks ({num_blocks_x}x{num_blocks_y})")

        block_count = 0
        for i in range(num_blocks_x):
            for j in range(num_blocks_y):
                # Define block bounds
                x_start = x_min + i * block_size
                y_start = y_min + j * block_size
                x_end = x_start + block_size
                y_end = y_start + block_size

                # Find points in block
                mask = (
                    (xyz[:, 0] >= x_start) & (xyz[:, 0] < x_end) &
                    (xyz[:, 1] >= y_start) & (xyz[:, 1] < y_end)
                )

                if np.sum(mask) == 0:
                    continue

                block_count += 1
                logger.info(f"Processing block {block_count}/{total_blocks} "
                          f"({np.sum(mask):,} points)")

                # Extract block data
                block_xyz = xyz[mask]
                block_features = features[mask]

                # Classify block
                block_predictions = self.classify_points(
                    block_xyz, block_features,
                    batch_size=batch_size,
                    overlap=0.0
                )

                # Store predictions
                predictions[mask] = block_predictions

        # Save classified point cloud
        logger.info(f"Saving classified point cloud to {output_las}")
        processor.write_classified_las(output_las, predictions)

        # Print statistics
        unique, counts = np.unique(predictions, return_counts=True)
        logger.info("\nClassification Results:")
        class_names = ['Road', 'Snow', 'Vehicle', 'Vegetation', 'Others']
        for cls, count in zip(unique, counts):
            percentage = (count / len(predictions)) * 100
            class_name = class_names[cls] if cls < len(class_names) else f"Unknown ({cls})"
            logger.info(f"  Class {cls} ({class_name}): {count:,} points ({percentage:.2f}%)")

        return predictions


def main():
    parser = argparse.ArgumentParser(description='Classify point cloud using trained RandLA-Net')
    parser.add_argument('--input', type=str, required=True,
                       help='Path to input LAS file')
    parser.add_argument('--output', type=str, required=True,
                       help='Path to output classified LAS file')
    parser.add_argument('--model', type=str, default='checkpoints/best_model.pth',
                       help='Path to trained model checkpoint')
    parser.add_argument('--batch_size', type=int, default=4096,
                       help='Batch size for processing')
    parser.add_argument('--num_classes', type=int, default=5,
                       help='Number of classes')
    parser.add_argument('--num_features', type=int, default=6,
                       help='Number of input features (XYZ+RGB=6, XYZ+RGB+I=7)')
    parser.add_argument('--spatial', action='store_true',
                       help='Use spatial block processing for large files')
    parser.add_argument('--block_size', type=float, default=50.0,
                       help='Spatial block size in meters (if --spatial is used)')

    args = parser.parse_args()

    # Create classifier
    classifier = PointCloudClassifier(
        model_path=args.model,
        num_classes=args.num_classes,
        num_features=args.num_features
    )

    # Classify
    if args.spatial:
        classifier.classify_spatial_blocks(
            input_las=args.input,
            output_las=args.output,
            block_size=args.block_size,
            batch_size=args.batch_size
        )
    else:
        classifier.classify_las_file(
            input_las=args.input,
            output_las=args.output,
            batch_size=args.batch_size
        )

    logger.info("Classification complete!")


if __name__ == "__main__":
    main()
