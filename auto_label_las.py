"""
Automatic Point Cloud Labeling using Trained PointNet++ Model

This script takes a raw, unlabeled LAS file and automatically classifies
all points using the trained PointNet++ model.

Usage:
    python auto_label_las.py --input raw_data.las --output labeled_data.las

Options:
    --input, -i     : Input LAS file (unlabeled)
    --output, -o    : Output LAS file (will be labeled)
    --model, -m     : Path to trained model checkpoint (default: checkpoints/pointnet2_best_model.pth)
    --batch-size, -b: Batch size for inference (default: 8192)
    --device, -d    : Device to use (cuda/cpu, default: auto-detect)

Author: Niken Goswami
Date: December 2025
"""

import argparse
import os
import sys
import time
import numpy as np
import torch
from tqdm import tqdm
import laspy

# Add models directory to path
sys.path.append(os.path.dirname(__file__))
from models.pointnet2 import PointNet2


# Class mapping (6 classes including Building)
CLASS_NAMES = {
    0: "Road",
    1: "Snow",
    2: "Vehicle",
    3: "Vegetation",
    4: "Building",
    5: "Others"
}


def load_model(checkpoint_path, device):
    """
    Load trained PointNet++ model from checkpoint

    Args:
        checkpoint_path: Path to model checkpoint file
        device: torch device (cuda or cpu)

    Returns:
        Loaded model in eval mode
    """
    print(f"\n[1/5] Loading model from: {checkpoint_path}")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")

    # Initialize model (6 classes, 8 features: XYZ + RGB + Intensity + Height)
    num_classes = 6
    model = PointNet2(num_classes=num_classes, num_features=8).to(device)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Print model info
    if 'epoch' in checkpoint:
        print(f"  Model trained for {checkpoint['epoch']} epochs")
    if 'val_iou' in checkpoint:
        print(f"  Validation IoU: {checkpoint['val_iou']:.4f}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {total_params:,}")
    print(f"  Device: {device}")

    return model


def load_las_file(las_path):
    """
    Load LAS file and extract features

    Args:
        las_path: Path to input LAS file

    Returns:
        tuple: (las_file, xyz, features)
    """
    print(f"\n[2/5] Loading LAS file: {las_path}")

    if not os.path.exists(las_path):
        raise FileNotFoundError(f"Input LAS file not found: {las_path}")

    # Read LAS file
    las = laspy.read(las_path)
    num_points = len(las.x)
    print(f"  Total points: {num_points:,}")

    # Extract coordinates
    xyz = np.vstack([las.x, las.y, las.z]).T
    print(f"  XYZ range: X({xyz[:, 0].min():.2f}, {xyz[:, 0].max():.2f}), "
          f"Y({xyz[:, 1].min():.2f}, {xyz[:, 1].max():.2f}), "
          f"Z({xyz[:, 2].min():.2f}, {xyz[:, 2].max():.2f})")

    # Extract RGB colors (handle both 8-bit and 16-bit)
    if hasattr(las, 'red'):
        max_val = las.red.max()
        scale = 65535.0 if max_val > 255 else 255.0
        rgb = np.vstack([las.red, las.green, las.blue]).T / scale
        print(f"  RGB colors: Available (max value: {max_val})")
    else:
        print("  RGB colors: Not available, using default values (0.5)")
        rgb = np.ones((num_points, 3)) * 0.5

    # Extract intensity (handle different scales)
    if hasattr(las, 'intensity'):
        max_intensity = las.intensity.max()
        scale = 255.0 if max_intensity > 1.0 else 1.0
        intensity = las.intensity.reshape(-1, 1) / scale
        print(f"  Intensity: Available (max value: {max_intensity})")
    else:
        print("  Intensity: Not available, using default values (0.5)")
        intensity = np.ones((num_points, 1)) * 0.5

    # Calculate height above ground (estimate ground as 5th percentile of Z)
    ground_level = np.percentile(xyz[:, 2], 5)
    height_above_ground = (xyz[:, 2] - ground_level).astype(np.float32)
    height_above_ground = np.clip(height_above_ground, 0, 15) / 15.0  # Normalize to 0-1
    height_above_ground = height_above_ground.reshape(-1, 1)
    print(f"  Height above ground: Computed (ground level: {ground_level:.2f}m)")

    # Combine features: [X, Y, Z, R, G, B, Intensity, Height]
    features = np.hstack([xyz, rgb, intensity, height_above_ground])
    print(f"  Feature shape: {features.shape} (8 features)")

    return las, xyz, features


def normalize_coordinates(xyz):
    """
    Normalize XYZ coordinates (center and scale)

    Args:
        xyz: numpy array of shape (N, 3)

    Returns:
        Normalized xyz coordinates
    """
    xyz_mean = xyz.mean(axis=0)
    xyz_centered = xyz - xyz_mean
    xyz_std = xyz_centered.std()

    if xyz_std > 0:
        xyz_norm = xyz_centered / xyz_std
    else:
        xyz_norm = xyz_centered

    return xyz_norm


def run_inference(model, features, xyz, batch_size, device):
    """
    Run inference on point cloud data

    Args:
        model: Trained PointNet++ model
        features: Feature array (N, 7)
        xyz: Coordinate array (N, 3)
        batch_size: Batch size for processing
        device: torch device

    Returns:
        numpy array of predictions
    """
    print(f"\n[3/5] Running inference...")
    print(f"  Batch size: {batch_size}")

    # Normalize coordinates
    xyz_norm = normalize_coordinates(xyz)

    # Update features with normalized coordinates
    features_normalized = features.copy()
    features_normalized[:, :3] = xyz_norm

    num_points = len(features)
    num_batches = (num_points + batch_size - 1) // batch_size

    predictions = []

    # Process in batches with progress bar
    with torch.no_grad():
        for i in tqdm(range(num_batches), desc="  Processing batches"):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, num_points)

            # Get batch
            batch_xyz = xyz_norm[start_idx:end_idx]
            batch_features = features_normalized[start_idx:end_idx]

            # Convert to tensors and add batch dimension
            coords_tensor = torch.from_numpy(batch_xyz).float().unsqueeze(0).to(device)
            features_tensor = torch.from_numpy(batch_features).float().unsqueeze(0).to(device)

            # Run model
            logits = model(coords_tensor, features_tensor)

            # Get predictions
            batch_preds = torch.argmax(logits, dim=2).cpu().numpy().flatten()
            predictions.append(batch_preds)

    # Concatenate all predictions
    all_predictions = np.concatenate(predictions)

    return all_predictions


def save_labeled_las(las, predictions, output_path):
    """
    Save predictions to LAS file

    Args:
        las: Original LAS file object
        predictions: Predicted class labels (0-5)
        output_path: Output file path
    """
    print(f"\n[4/5] Saving labeled LAS file: {output_path}")

    # Map predictions back to original LAS classification codes
    # Our model predicts: 0=Road, 1=Snow, 2=Vehicle, 3=Vegetation, 4=Building, 5=Others
    # Need to map to LAS standard codes that CloudCompare understands
    REVERSE_MAPPING = {
        0: 2,   # Road -> Ground (LAS code 2)
        1: 32,  # Snow -> Snow (LAS code 32)
        2: 54,  # Vehicle -> Vehicle (LAS code 54)
        3: 5,   # Vegetation -> High vegetation (LAS code 5)
        4: 6,   # Building -> Building (LAS code 6)
        5: 1    # Others -> Unclassified (LAS code 1)
    }

    # Convert predictions to LAS codes
    las_codes = np.array([REVERSE_MAPPING.get(pred, 1) for pred in predictions], dtype=np.uint8)

    # Update classification field
    las.classification = las_codes

    # Save to new file
    las.write(output_path)

    file_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  File saved successfully")
    print(f"  File size: {file_size:.2f} MB")


def print_statistics(predictions):
    """
    Print classification statistics

    Args:
        predictions: Array of predicted labels
    """
    print(f"\n[5/5] Classification Statistics:")
    print(f"  {'Class':<15} {'Points':<12} {'Percentage':<10}")
    print(f"  {'-'*40}")

    total_points = len(predictions)

    for class_id, class_name in CLASS_NAMES.items():
        count = (predictions == class_id).sum()
        percentage = (count / total_points) * 100
        print(f"  {class_name:<15} {count:<12,} {percentage:>6.2f}%")

    print(f"  {'-'*40}")
    print(f"  {'TOTAL':<15} {total_points:<12,} {100.0:>6.2f}%")


def main():
    """
    Main function
    """
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Automatic Point Cloud Labeling using PointNet++",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('-i', '--input', required=True,
                        help='Input LAS file (unlabeled)')
    parser.add_argument('-o', '--output', required=True,
                        help='Output LAS file (will be labeled)')
    parser.add_argument('-m', '--model', default='checkpoints/pointnet2_best_model.pth',
                        help='Path to trained model checkpoint (default: checkpoints/pointnet2_best_model.pth)')
    parser.add_argument('-b', '--batch-size', type=int, default=8192,
                        help='Batch size for inference (default: 8192)')
    parser.add_argument('-d', '--device', choices=['cuda', 'cpu', 'auto'], default='auto',
                        help='Device to use (default: auto)')

    args = parser.parse_args()

    # Print header
    print("="*60)
    print("  AUTOMATIC POINT CLOUD LABELING WITH POINTNET++")
    print("="*60)
    print(f"\nInput:  {args.input}")
    print(f"Output: {args.output}")
    print(f"Model:  {args.model}")

    # Determine device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    # Start timer
    start_time = time.time()

    try:
        # Load model
        model = load_model(args.model, device)

        # Load LAS file
        las, xyz, features = load_las_file(args.input)

        # Run inference
        predictions = run_inference(model, features, xyz, args.batch_size, device)

        # Save results
        save_labeled_las(las, predictions, args.output)

        # Print statistics
        print_statistics(predictions)

        # Print summary
        elapsed_time = time.time() - start_time
        points_per_second = len(predictions) / elapsed_time

        print(f"\n{'='*60}")
        print(f"  LABELING COMPLETE!")
        print(f"{'='*60}")
        print(f"  Total time: {elapsed_time:.2f} seconds")
        print(f"  Speed: {points_per_second:,.0f} points/second")
        print(f"  Output: {args.output}")
        print(f"{'='*60}\n")

    except Exception as e:
        print(f"\n[ERROR] {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
