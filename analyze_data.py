"""
Analyze the Classified.las file to understand its structure and features
This script provides comprehensive analysis of the point cloud data
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from utils.las_io import LASProcessor, visualize_las_statistics
import json


def main():
    # Path to the LAS file
    las_file = "Classified.las"

    print("=" * 80)
    print("MMS POINT CLOUD DATA ANALYSIS")
    print("=" * 80)
    print(f"\nAnalyzing: {las_file}\n")

    # Initialize processor
    processor = LASProcessor(las_file)

    # Read LAS file
    features = processor.read_las()

    # Get statistics
    stats = processor.get_statistics()

    print("\n" + "=" * 80)
    print("POINT CLOUD STATISTICS")
    print("=" * 80)

    # Print general info
    print(f"\nTotal Points: {stats['num_points']:,}")

    # Print bounding box
    print("\nBounding Box:")
    bbox = features['bbox']
    print(f"  X: [{bbox['min_x']:.2f}, {bbox['max_x']:.2f}] (range: {bbox['max_x']-bbox['min_x']:.2f}m)")
    print(f"  Y: [{bbox['min_y']:.2f}, {bbox['max_y']:.2f}] (range: {bbox['max_y']-bbox['min_y']:.2f}m)")
    print(f"  Z: [{bbox['min_z']:.2f}, {bbox['max_z']:.2f}] (range: {bbox['max_z']-bbox['min_z']:.2f}m)")

    # Print XYZ statistics
    print("\nXYZ Statistics:")
    print(f"  Mean: [{stats['xyz_mean'][0]:.2f}, {stats['xyz_mean'][1]:.2f}, {stats['xyz_mean'][2]:.2f}]")
    print(f"  Std:  [{stats['xyz_std'][0]:.2f}, {stats['xyz_std'][1]:.2f}, {stats['xyz_std'][2]:.2f}]")

    # Print RGB statistics if available
    if 'rgb_mean' in stats:
        print("\nRGB Statistics:")
        print(f"  Mean: [{stats['rgb_mean'][0]:.3f}, {stats['rgb_mean'][1]:.3f}, {stats['rgb_mean'][2]:.3f}]")
        print(f"  Std:  [{stats['rgb_std'][0]:.3f}, {stats['rgb_std'][1]:.3f}, {stats['rgb_std'][2]:.3f}]")

    # Print Intensity statistics if available
    if 'intensity_mean' in stats:
        print("\nIntensity Statistics:")
        print(f"  Mean: {stats['intensity_mean']:.2f}")
        print(f"  Std:  {stats['intensity_std']:.2f}")
        print(f"  Min:  {stats['intensity_min']:.2f}")
        print(f"  Max:  {stats['intensity_max']:.2f}")

    # Print classification distribution if available
    if 'class_distribution' in stats:
        print("\nClassification Distribution:")
        class_dist = stats['class_distribution']
        total = sum(class_dist.values())

        # Standard LAS classification codes
        class_names = {
            0: "Never classified",
            1: "Unclassified",
            2: "Ground",
            3: "Low Vegetation",
            4: "Medium Vegetation",
            5: "High Vegetation",
            6: "Building",
            7: "Low Point (noise)",
            8: "Reserved",
            9: "Water",
            10: "Rail",
            11: "Road Surface",
            12: "Reserved",
            13: "Wire - Guard (Shield)",
            14: "Wire - Conductor (Phase)",
            15: "Transmission Tower",
            16: "Wire-structure Connector",
            17: "Bridge Deck",
            18: "High Noise"
        }

        for cls, count in sorted(class_dist.items()):
            percentage = (count / total) * 100
            class_name = class_names.get(cls, f"Unknown ({cls})")
            print(f"  Class {cls:2d} ({class_name:20s}): {count:10,} points ({percentage:5.2f}%)")

    # Print available features
    print("\nAvailable Features:")
    feature_names = [key for key in features.keys() if key not in ['num_points', 'bbox']]
    for feature in feature_names:
        if feature == 'xyz':
            print(f"  - XYZ coordinates: shape {features[feature].shape}")
        elif feature == 'rgb':
            print(f"  - RGB colors: shape {features[feature].shape}")
        elif feature == 'intensity':
            print(f"  - Intensity: shape {features[feature].shape}")
        elif feature == 'classification':
            print(f"  - Classification: shape {features[feature].shape}")
        elif feature == 'return_number':
            print(f"  - Return number: shape {features[feature].shape}")
        elif feature == 'number_of_returns':
            print(f"  - Number of returns: shape {features[feature].shape}")

    # Extract features for training
    print("\n" + "=" * 80)
    print("FEATURE EXTRACTION FOR TRAINING")
    print("=" * 80)

    train_features, labels = processor.extract_features_for_training(normalize=True)
    print(f"\nExtracted feature matrix shape: {train_features.shape}")
    print(f"  - Number of points: {train_features.shape[0]:,}")
    print(f"  - Number of features per point: {train_features.shape[1]}")

    if labels is not None:
        print(f"\nLabels shape: {labels.shape}")
        unique_labels = len(set(labels))
        print(f"Number of unique classes: {unique_labels}")
    else:
        print("\nNo classification labels found in the LAS file.")

    # Save statistics to JSON
    stats_file = "results/data_statistics.json"
    Path("results").mkdir(exist_ok=True)
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"\nStatistics saved to: {stats_file}")

    # Generate visualization
    print("\n" + "=" * 80)
    print("GENERATING VISUALIZATIONS")
    print("=" * 80)

    try:
        viz_file = "results/data_visualization.png"
        visualize_las_statistics(las_file, save_path=viz_file)
        print(f"Visualization saved to: {viz_file}")
    except Exception as e:
        print(f"Could not generate visualization: {e}")
        print("(This is normal if matplotlib is not installed yet)")

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
