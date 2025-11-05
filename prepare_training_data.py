"""
Prepare training data with custom class mapping
Combines multiple labeled files and splits into train/val/test
"""

import numpy as np
from pathlib import Path
import logging
from utils.las_io import LASProcessor
from class_mapping_config import CLASS_MAPPING, TARGET_CLASSES
import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_and_map_classes(las_file: str):
    """Load LAS file and map classes to target categories"""

    logger.info(f"Loading: {las_file}")
    processor = LASProcessor(las_file)
    features_dict = processor.read_las()

    # Extract data
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

    # Get original classifications
    if 'classification' not in features_dict:
        raise ValueError(f"No classification found in {las_file}")

    original_classes = features_dict['classification']

    # Map to target classes
    mapped_classes = np.zeros_like(original_classes)
    for orig_cls, target_cls in CLASS_MAPPING.items():
        mask = original_classes == orig_cls
        mapped_classes[mask] = target_cls
        if np.sum(mask) > 0:
            logger.info(f"  Mapped class {orig_cls} -> {target_cls}: {np.sum(mask):,} points")

    # Check for unmapped classes
    unique_orig = np.unique(original_classes)
    unmapped = [c for c in unique_orig if c not in CLASS_MAPPING]
    if unmapped:
        logger.warning(f"Unmapped classes (will be set to Others): {unmapped}")
        for c in unmapped:
            mask = original_classes == c
            mapped_classes[mask] = 4  # Others

    logger.info(f"Loaded {len(xyz):,} points from {Path(las_file).name}")

    return xyz, features, mapped_classes


def combine_and_split_data(las_files: list, train_ratio=0.7, val_ratio=0.15):
    """
    Combine multiple LAS files and split into train/val/test

    Args:
        las_files: List of LAS file paths
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set (rest goes to test)
    """

    all_xyz = []
    all_features = []
    all_labels = []

    # Load all files
    for las_file in las_files:
        xyz, features, labels = load_and_map_classes(las_file)
        all_xyz.append(xyz)
        all_features.append(features)
        all_labels.append(labels)

    # Combine
    combined_xyz = np.vstack(all_xyz)
    combined_features = np.vstack(all_features)
    combined_labels = np.concatenate(all_labels)

    logger.info(f"\nCombined total: {len(combined_xyz):,} points")

    # Show combined class distribution
    logger.info("\nTarget class distribution (after mapping):")
    for target_id, target_name in TARGET_CLASSES.items():
        count = np.sum(combined_labels == target_id)
        pct = (count / len(combined_labels)) * 100
        logger.info(f"  {target_id}. {target_name:12s}: {count:7,} points ({pct:5.1f}%)")

    # Random shuffle
    indices = np.random.permutation(len(combined_xyz))
    combined_xyz = combined_xyz[indices]
    combined_features = combined_features[indices]
    combined_labels = combined_labels[indices]

    # Split
    n_total = len(combined_xyz)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    n_test = n_total - n_train - n_val

    train_xyz = combined_xyz[:n_train]
    train_features = combined_features[:n_train]
    train_labels = combined_labels[:n_train]

    val_xyz = combined_xyz[n_train:n_train+n_val]
    val_features = combined_features[n_train:n_train+n_val]
    val_labels = combined_labels[n_train:n_train+n_val]

    test_xyz = combined_xyz[n_train+n_val:]
    test_features = combined_features[n_train+n_val:]
    test_labels = combined_labels[n_train+n_val:]

    logger.info(f"\nData split:")
    logger.info(f"  Train: {len(train_xyz):,} points ({train_ratio*100:.0f}%)")
    logger.info(f"  Val:   {len(val_xyz):,} points ({val_ratio*100:.0f}%)")
    logger.info(f"  Test:  {len(test_xyz):,} points ({(1-train_ratio-val_ratio)*100:.0f}%)")

    # Save to numpy files
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    np.savez(output_dir / "train_data.npz",
             xyz=train_xyz, features=train_features, labels=train_labels)
    np.savez(output_dir / "val_data.npz",
             xyz=val_xyz, features=val_features, labels=val_labels)
    np.savez(output_dir / "test_data.npz",
             xyz=test_xyz, features=test_features, labels=test_labels)

    logger.info(f"\nSaved processed data to: {output_dir}")
    logger.info("  - train_data.npz")
    logger.info("  - val_data.npz")
    logger.info("  - test_data.npz")

    return {
        'train': (train_xyz, train_features, train_labels),
        'val': (val_xyz, val_features, val_labels),
        'test': (test_xyz, test_features, test_labels)
    }


def main():
    parser = argparse.ArgumentParser(description='Prepare training data')
    parser.add_argument('--input_dir', type=str, default='data/labeled',
                       help='Directory containing labeled LAS files')
    parser.add_argument('--train_ratio', type=float, default=0.7,
                       help='Training set ratio')
    parser.add_argument('--val_ratio', type=float, default=0.15,
                       help='Validation set ratio')

    args = parser.parse_args()

    # Find all LAS files in input directory
    input_path = Path(args.input_dir)
    las_files = list(input_path.glob("*.las"))

    if len(las_files) == 0:
        logger.error(f"No .las files found in {input_path}")
        return

    logger.info(f"Found {len(las_files)} LAS files:")
    for f in las_files:
        logger.info(f"  - {f.name}")

    # Process and split data
    combine_and_split_data(las_files, args.train_ratio, args.val_ratio)

    logger.info("\nData preparation complete!")


if __name__ == "__main__":
    main()
