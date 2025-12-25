"""
Visualization tools for point cloud classification results
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import open3d as o3d
from pathlib import Path
import argparse
import logging

from utils.las_io import LASProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PointCloudVisualizer:
    """Visualize classified point clouds"""

    # Color scheme for classes
    CLASS_COLORS = {
        0: [0.5, 0.5, 0.5],    # Road - Gray
        1: [1.0, 1.0, 1.0],    # Snow - White
        2: [1.0, 0.0, 0.0],    # Vehicle - Red
        3: [0.0, 1.0, 0.0],    # Vegetation - Green
        4: [0.5, 0.5, 0.0],    # Others - Brown
    }

    CLASS_NAMES = ['Road', 'Snow', 'Vehicle', 'Vegetation', 'Others']

    @staticmethod
    def visualize_2d(las_file: str, save_path: str = None, figsize=(15, 12)):
        """
        Create 2D visualization of classified point cloud

        Args:
            las_file: Path to LAS file
            save_path: Optional path to save figure
            figsize: Figure size
        """
        logger.info(f"Creating 2D visualization of {las_file}")

        # Load LAS file
        processor = LASProcessor(las_file)
        features_dict = processor.read_las()

        xyz = features_dict['xyz']
        classification = features_dict.get('classification', None)

        if classification is None:
            logger.error("No classification found in LAS file")
            return

        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=figsize)

        # Plot 1: XY view colored by classification
        ax = axes[0, 0]
        colors = np.array([PointCloudVisualizer.CLASS_COLORS.get(c, [0, 0, 0])
                          for c in classification])
        ax.scatter(xyz[:, 0], xyz[:, 1], c=colors, s=0.1, alpha=0.8)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title('XY View - Classified')
        ax.axis('equal')

        # Plot 2: XZ view (side view)
        ax = axes[0, 1]
        ax.scatter(xyz[:, 0], xyz[:, 2], c=colors, s=0.1, alpha=0.8)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Z (m)')
        ax.set_title('XZ View - Classified')

        # Plot 3: YZ view (side view)
        ax = axes[1, 0]
        ax.scatter(xyz[:, 1], xyz[:, 2], c=colors, s=0.1, alpha=0.8)
        ax.set_xlabel('Y (m)')
        ax.set_ylabel('Z (m)')
        ax.set_title('YZ View - Classified')

        # Plot 4: Class distribution
        ax = axes[1, 1]
        unique, counts = np.unique(classification, return_counts=True)
        class_names = [PointCloudVisualizer.CLASS_NAMES[c] if c < len(PointCloudVisualizer.CLASS_NAMES)
                      else f"Class {c}" for c in unique]
        class_colors = [PointCloudVisualizer.CLASS_COLORS.get(c, [0.5, 0.5, 0.5]) for c in unique]

        bars = ax.bar(class_names, counts, color=class_colors, edgecolor='black', alpha=0.8)
        ax.set_ylabel('Number of Points')
        ax.set_title('Classification Distribution')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3, axis='y')

        # Add percentage labels on bars
        total_points = len(classification)
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            percentage = (count / total_points) * 100
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{percentage:.1f}%',
                   ha='center', va='bottom', fontsize=9)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"2D visualization saved to {save_path}")

        plt.show()

    @staticmethod
    def visualize_3d_open3d(las_file: str, point_size: float = 1.0, background: str = 'white'):
        """
        Interactive 3D visualization using Open3D

        Args:
            las_file: Path to LAS file
            point_size: Size of points in visualization
            background: Background color ('white' or 'black')
        """
        logger.info(f"Creating 3D visualization of {las_file}")

        # Load LAS file
        processor = LASProcessor(las_file)
        features_dict = processor.read_las()

        xyz = features_dict['xyz']
        classification = features_dict.get('classification', None)

        if classification is None:
            logger.error("No classification found in LAS file")
            return

        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)

        # Assign colors based on classification
        colors = np.array([PointCloudVisualizer.CLASS_COLORS.get(c, [0.5, 0.5, 0.5])
                          for c in classification])
        pcd.colors = o3d.utility.Vector3dVector(colors)

        # Visualize
        logger.info("Opening 3D viewer...")
        logger.info("Controls: Mouse to rotate, scroll to zoom, Shift+mouse to pan")

        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name='Point Cloud Classification')
        vis.add_geometry(pcd)

        # Set rendering options
        opt = vis.get_render_option()
        opt.point_size = point_size
        if background == 'white':
            opt.background_color = np.array([1, 1, 1])
        else:
            opt.background_color = np.array([0, 0, 0])

        vis.run()
        vis.destroy_window()

    @staticmethod
    def compare_classifications(ground_truth_las: str, predicted_las: str, save_path: str = None):
        """
        Compare ground truth and predicted classifications

        Args:
            ground_truth_las: Path to ground truth LAS file
            predicted_las: Path to predicted LAS file
            save_path: Optional path to save figure
        """
        logger.info("Comparing classifications...")

        # Load ground truth
        gt_processor = LASProcessor(ground_truth_las)
        gt_features = gt_processor.read_las()
        gt_xyz = gt_features['xyz']
        gt_labels = gt_features.get('classification', None)

        # Load predictions
        pred_processor = LASProcessor(predicted_las)
        pred_features = pred_processor.read_las()
        pred_labels = pred_features.get('classification', None)

        if gt_labels is None or pred_labels is None:
            logger.error("Classification not found in one or both files")
            return

        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))

        # Ground truth
        ax = axes[0]
        gt_colors = np.array([PointCloudVisualizer.CLASS_COLORS.get(c, [0, 0, 0])
                             for c in gt_labels])
        ax.scatter(gt_xyz[:, 0], gt_xyz[:, 1], c=gt_colors, s=0.1, alpha=0.8)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title('Ground Truth')
        ax.axis('equal')

        # Predictions
        ax = axes[1]
        pred_colors = np.array([PointCloudVisualizer.CLASS_COLORS.get(c, [0, 0, 0])
                               for c in pred_labels])
        ax.scatter(gt_xyz[:, 0], gt_xyz[:, 1], c=pred_colors, s=0.1, alpha=0.8)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title('Predictions')
        ax.axis('equal')

        # Errors (red where wrong, green where correct)
        ax = axes[2]
        correct = gt_labels == pred_labels
        error_colors = np.where(correct.reshape(-1, 1),
                               np.array([[0, 1, 0]]),  # Green for correct
                               np.array([[1, 0, 0]]))  # Red for errors
        ax.scatter(gt_xyz[:, 0], gt_xyz[:, 1], c=error_colors, s=0.1, alpha=0.8)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        accuracy = np.mean(correct) * 100
        ax.set_title(f'Errors (Accuracy: {accuracy:.2f}%)')
        ax.axis('equal')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Comparison visualization saved to {save_path}")

        plt.show()

    @staticmethod
    def create_legend(save_path: str = None):
        """
        Create a legend for the classification colors

        Args:
            save_path: Optional path to save legend
        """
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.axis('off')

        # Create legend patches
        from matplotlib.patches import Rectangle

        y_pos = 0.9
        for class_id, class_name in enumerate(PointCloudVisualizer.CLASS_NAMES):
            color = PointCloudVisualizer.CLASS_COLORS[class_id]
            rect = Rectangle((0.1, y_pos - 0.05), 0.1, 0.08,
                           facecolor=color, edgecolor='black')
            ax.add_patch(rect)
            ax.text(0.25, y_pos, f"{class_name}", fontsize=14, va='center')
            y_pos -= 0.15

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title('Point Cloud Classification Legend', fontsize=16, fontweight='bold')

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Legend saved to {save_path}")

        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Visualize classified point clouds')
    parser.add_argument('--input', type=str, required=True,
                       help='Path to classified LAS file')
    parser.add_argument('--mode', type=str, default='2d', choices=['2d', '3d', 'legend'],
                       help='Visualization mode')
    parser.add_argument('--save', type=str, default=None,
                       help='Path to save visualization')
    parser.add_argument('--compare', type=str, default=None,
                       help='Path to ground truth LAS file for comparison')
    parser.add_argument('--point_size', type=float, default=1.0,
                       help='Point size for 3D visualization')

    args = parser.parse_args()

    visualizer = PointCloudVisualizer()

    if args.mode == '2d':
        if args.compare:
            visualizer.compare_classifications(args.compare, args.input, args.save)
        else:
            visualizer.visualize_2d(args.input, args.save)

    elif args.mode == '3d':
        visualizer.visualize_3d_open3d(args.input, point_size=args.point_size)

    elif args.mode == 'legend':
        visualizer.create_legend(args.save)


if __name__ == "__main__":
    main()
