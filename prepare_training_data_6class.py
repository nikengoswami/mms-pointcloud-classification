"""
Prepare training data with 6 classes (including Building)
Processes all labeled LAS files and creates train/val/test splits
"""

import numpy as np
import laspy
from pathlib import Path
from sklearn.model_selection import train_test_split
import logging

from class_mapping_config import CLASS_MAPPING, TARGET_CLASSES

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_and_map_las(las_path):
    """Load LAS file and map classes to 6-class scheme"""
    logger.info(f"Loading: {las_path}")

    las = laspy.read(las_path)

    # Extract XYZ
    xyz = np.vstack([las.x, las.y, las.z]).T.astype(np.float32)

    # Extract RGB (normalize to 0-1)
    if hasattr(las, 'red'):
        max_val = max(las.red.max(), las.green.max(), las.blue.max())
        if max_val > 255:
            max_val = 65535  # 16-bit color
        else:
            max_val = 255  # 8-bit color
        rgb = np.vstack([
            las.red / max_val,
            las.green / max_val,
            las.blue / max_val
        ]).T.astype(np.float32)
    else:
        rgb = np.zeros((len(xyz), 3), dtype=np.float32)

    # Extract intensity (normalize to 0-1)
    if hasattr(las, 'intensity'):
        intensity = las.intensity.astype(np.float32)
        if intensity.max() > 0:
            intensity = intensity / intensity.max()
        intensity = intensity.reshape(-1, 1)
    else:
        intensity = np.zeros((len(xyz), 1), dtype=np.float32)

    # Get original classification
    original_labels = las.classification.astype(np.int64)

    # Map to 6 classes
    mapped_labels = np.full_like(original_labels, 5)  # Default to Others (5)
    for orig_class, target_class in CLASS_MAPPING.items():
        mapped_labels[original_labels == orig_class] = target_class

    # ============ NEW: Calculate height above ground ============
    # Use Road points (class 0) to estimate ground level, or use 5th percentile of Z
    road_mask = mapped_labels == 0  # Road class
    if np.any(road_mask):
        ground_level = np.percentile(xyz[road_mask, 2], 5)  # 5th percentile of road Z
    else:
        ground_level = np.percentile(xyz[:, 2], 5)  # 5th percentile of all Z

    # Calculate height above ground (normalized to 0-1 range, cap at 15m)
    height_above_ground = (xyz[:, 2] - ground_level).astype(np.float32)
    height_above_ground = np.clip(height_above_ground, 0, 15) / 15.0  # Normalize to 0-1
    height_above_ground = height_above_ground.reshape(-1, 1)

    logger.info(f"  Ground level: {ground_level:.2f}m, Height range: 0-{(xyz[:, 2] - ground_level).max():.1f}m")

    # Combine features: XYZ (3) + RGB (3) + Intensity (1) + Height (1) = 8 features
    features = np.hstack([xyz, rgb, intensity, height_above_ground])

    # Print class distribution
    unique, counts = np.unique(mapped_labels, return_counts=True)
    logger.info(f"  Points: {len(xyz):,}")
    for cls_id, count in zip(unique, counts):
        cls_name = TARGET_CLASSES.get(cls_id, f"Unknown({cls_id})")
        pct = count / len(mapped_labels) * 100
        logger.info(f"    {cls_id} ({cls_name}): {count:,} ({pct:.2f}%)")

    return xyz, features, mapped_labels


def main():
    # Input files
    labeled_dir = Path("data/labeled")
    output_dir = Path("data/processed")
    output_dir.mkdir(exist_ok=True)

    # Find all labeled LAS files
    las_files = list(labeled_dir.glob("*.las"))
    logger.info(f"Found {len(las_files)} labeled LAS files")

    # Load all data
    all_xyz = []
    all_features = []
    all_labels = []

    for las_file in las_files:
        xyz, features, labels = load_and_map_las(las_file)
        all_xyz.append(xyz)
        all_features.append(features)
        all_labels.append(labels)

    # Concatenate all data
    all_xyz = np.vstack(all_xyz)
    all_features = np.vstack(all_features)
    all_labels = np.concatenate(all_labels)

    logger.info(f"\nTotal points: {len(all_xyz):,}")

    # Print overall class distribution
    logger.info("\nOverall class distribution:")
    unique, counts = np.unique(all_labels, return_counts=True)
    for cls_id, count in zip(unique, counts):
        cls_name = TARGET_CLASSES.get(cls_id, f"Unknown({cls_id})")
        pct = count / len(all_labels) * 100
        logger.info(f"  {cls_id} ({cls_name}): {count:,} ({pct:.2f}%)")

    # Shuffle data
    indices = np.random.permutation(len(all_xyz))
    all_xyz = all_xyz[indices]
    all_features = all_features[indices]
    all_labels = all_labels[indices]

    # Split: 70% train, 15% val, 15% test
    n_total = len(all_xyz)
    n_train = int(0.7 * n_total)
    n_val = int(0.15 * n_total)

    train_xyz = all_xyz[:n_train]
    train_features = all_features[:n_train]
    train_labels = all_labels[:n_train]

    val_xyz = all_xyz[n_train:n_train+n_val]
    val_features = all_features[n_train:n_train+n_val]
    val_labels = all_labels[n_train:n_train+n_val]

    test_xyz = all_xyz[n_train+n_val:]
    test_features = all_features[n_train+n_val:]
    test_labels = all_labels[n_train+n_val:]

    logger.info(f"\nSplit sizes:")
    logger.info(f"  Train: {len(train_xyz):,} points")
    logger.info(f"  Val:   {len(val_xyz):,} points")
    logger.info(f"  Test:  {len(test_xyz):,} points")

    # Save
    logger.info("\nSaving processed data...")

    np.savez(output_dir / "train_data.npz",
             xyz=train_xyz, features=train_features, labels=train_labels)
    np.savez(output_dir / "val_data.npz",
             xyz=val_xyz, features=val_features, labels=val_labels)
    np.savez(output_dir / "test_data.npz",
             xyz=test_xyz, features=test_features, labels=test_labels)

    logger.info("Done!")
    logger.info(f"\nNew 6-class mapping:")
    for cls_id, cls_name in TARGET_CLASSES.items():
        logger.info(f"  {cls_id}: {cls_name}")


if __name__ == "__main__":
    main()
