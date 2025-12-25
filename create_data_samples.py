"""
Create visual samples of point cloud data for presentation
Shows raw data, preprocessing steps, and intermediate results
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches
from pathlib import Path

# Load sample data
data_path = Path("data/processed/train_data.npz")
data = np.load(data_path)

xyz = data['xyz'][:50000]  # Sample 50K points for visualization
features = data['features'][:50000]
labels = data['labels'][:50000]

# Class information
class_names = ['Road', 'Snow', 'Vehicle', 'Vegetation', 'Others']
class_colors = {
    0: '#E74C3C',  # Road - Red
    1: '#3498DB',  # Snow - Blue
    2: '#F39C12',  # Vehicle - Orange
    3: '#27AE60',  # Vegetation - Green
    4: '#95A5A6'   # Others - Gray
}

output_dir = Path("results")
output_dir.mkdir(exist_ok=True)


# ============================================================================
# Figure 1: Point Cloud Sample (Top View - XY Plane)
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left: Colored by RGB
ax = axes[0]
rgb = features[:, 3:6]  # Extract RGB
rgb_normalized = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)

scatter = ax.scatter(xyz[:, 0], xyz[:, 1], c=rgb_normalized, s=0.5, alpha=0.6)
ax.set_xlabel('X Coordinate (m)', fontsize=12, fontweight='bold')
ax.set_ylabel('Y Coordinate (m)', fontsize=12, fontweight='bold')
ax.set_title('Point Cloud Sample - RGB Colors (Top View)', fontsize=14, fontweight='bold')
ax.grid(alpha=0.3)
ax.set_aspect('equal')

# Right: Colored by Class
ax = axes[1]
for class_id, class_name in enumerate(class_names):
    mask = labels == class_id
    if np.any(mask):
        ax.scatter(xyz[mask, 0], xyz[mask, 1],
                  c=class_colors[class_id], label=class_name,
                  s=0.5, alpha=0.7)

ax.set_xlabel('X Coordinate (m)', fontsize=12, fontweight='bold')
ax.set_ylabel('Y Coordinate (m)', fontsize=12, fontweight='bold')
ax.set_title('Point Cloud Sample - Classification Labels', fontsize=14, fontweight='bold')
ax.legend(markerscale=10, fontsize=10, loc='upper right')
ax.grid(alpha=0.3)
ax.set_aspect('equal')

plt.tight_layout()
plt.savefig(output_dir / 'sample_point_cloud_top_view.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {output_dir / 'sample_point_cloud_top_view.png'}")


# ============================================================================
# Figure 2: Point Cloud Sample (Side View - XZ Plane)
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left: Colored by Intensity
ax = axes[0]
intensity = features[:, 6]  # Extract intensity
scatter = ax.scatter(xyz[:, 0], xyz[:, 2], c=intensity, cmap='viridis', s=0.5, alpha=0.6)
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Intensity', fontsize=11, fontweight='bold')
ax.set_xlabel('X Coordinate (m)', fontsize=12, fontweight='bold')
ax.set_ylabel('Z Coordinate (Height, m)', fontsize=12, fontweight='bold')
ax.set_title('Point Cloud Sample - Intensity (Side View)', fontsize=14, fontweight='bold')
ax.grid(alpha=0.3)

# Right: Colored by Class
ax = axes[1]
for class_id, class_name in enumerate(class_names):
    mask = labels == class_id
    if np.any(mask):
        ax.scatter(xyz[mask, 0], xyz[mask, 2],
                  c=class_colors[class_id], label=class_name,
                  s=0.5, alpha=0.7)

ax.set_xlabel('X Coordinate (m)', fontsize=12, fontweight='bold')
ax.set_ylabel('Z Coordinate (Height, m)', fontsize=12, fontweight='bold')
ax.set_title('Point Cloud Sample - Classification Labels (Side View)', fontsize=14, fontweight='bold')
ax.legend(markerscale=10, fontsize=10, loc='upper right')
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'sample_point_cloud_side_view.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {output_dir / 'sample_point_cloud_side_view.png'}")


# ============================================================================
# Figure 3: Feature Distribution Analysis
# ============================================================================
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# XYZ distributions
for i, coord_name in enumerate(['X', 'Y', 'Z']):
    ax = axes[0, i]
    coord_data = xyz[:, i]

    for class_id, class_name in enumerate(class_names):
        mask = labels == class_id
        if np.any(mask):
            ax.hist(coord_data[mask], bins=50, alpha=0.5,
                   label=class_name, color=class_colors[class_id])

    ax.set_xlabel(f'{coord_name} Coordinate (m)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax.set_title(f'{coord_name} Coordinate Distribution by Class', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

# RGB distributions
for i, color_name in enumerate(['Red', 'Green', 'Blue']):
    ax = axes[1, i]
    color_data = features[:, 3 + i]

    for class_id, class_name in enumerate(class_names):
        mask = labels == class_id
        if np.any(mask):
            ax.hist(color_data[mask], bins=50, alpha=0.5,
                   label=class_name, color=class_colors[class_id])

    ax.set_xlabel(f'{color_name} Value', fontsize=11, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax.set_title(f'{color_name} Channel Distribution by Class', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'feature_distributions.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {output_dir / 'feature_distributions.png'}")


# ============================================================================
# Figure 4: Preprocessing Steps Visualization
# ============================================================================
fig = plt.figure(figsize=(16, 10))

# Original Data
ax1 = plt.subplot(2, 3, 1)
sample_points = xyz[:2048]
sample_labels = labels[:2048]
for class_id in range(5):
    mask = sample_labels == class_id
    if np.any(mask):
        ax1.scatter(sample_points[mask, 0], sample_points[mask, 1],
                   c=class_colors[class_id], s=1, alpha=0.7)
ax1.set_title('1. Original Data\n(2048 points)', fontsize=12, fontweight='bold')
ax1.set_xlabel('X (m)', fontsize=10)
ax1.set_ylabel('Y (m)', fontsize=10)
ax1.grid(alpha=0.3)
ax1.set_aspect('equal')

# Random Sampling
ax2 = plt.subplot(2, 3, 2)
random_indices = np.random.choice(len(sample_points), 2048, replace=False)
sampled_points = sample_points[random_indices]
sampled_labels = sample_labels[random_indices]
for class_id in range(5):
    mask = sampled_labels == class_id
    if np.any(mask):
        ax2.scatter(sampled_points[mask, 0], sampled_points[mask, 1],
                   c=class_colors[class_id], s=1, alpha=0.7)
ax2.set_title('2. Random Sampling\n(2048 points)', fontsize=12, fontweight='bold')
ax2.set_xlabel('X (m)', fontsize=10)
ax2.set_ylabel('Y (m)', fontsize=10)
ax2.grid(alpha=0.3)
ax2.set_aspect('equal')

# Centering (Zero-mean)
ax3 = plt.subplot(2, 3, 3)
centered_points = sampled_points - np.mean(sampled_points, axis=0)
for class_id in range(5):
    mask = sampled_labels == class_id
    if np.any(mask):
        ax3.scatter(centered_points[mask, 0], centered_points[mask, 1],
                   c=class_colors[class_id], s=1, alpha=0.7)
ax3.set_title('3. Centering\n(Zero-mean)', fontsize=12, fontweight='bold')
ax3.set_xlabel('X (normalized)', fontsize=10)
ax3.set_ylabel('Y (normalized)', fontsize=10)
ax3.grid(alpha=0.3)
ax3.set_aspect('equal')

# Normalization (Unit variance)
ax4 = plt.subplot(2, 3, 4)
std = np.std(centered_points)
normalized_points = centered_points / (std + 1e-8)
for class_id in range(5):
    mask = sampled_labels == class_id
    if np.any(mask):
        ax4.scatter(normalized_points[mask, 0], normalized_points[mask, 1],
                   c=class_colors[class_id], s=1, alpha=0.7)
ax4.set_title('4. Normalization\n(Unit variance)', fontsize=12, fontweight='bold')
ax4.set_xlabel('X (normalized)', fontsize=10)
ax4.set_ylabel('Y (normalized)', fontsize=10)
ax4.grid(alpha=0.3)
ax4.set_aspect('equal')

# Rotation Augmentation
ax5 = plt.subplot(2, 3, 5)
angle = np.pi / 4  # 45 degrees
cos_a, sin_a = np.cos(angle), np.sin(angle)
rot_matrix = np.array([[cos_a, -sin_a, 0],
                       [sin_a, cos_a, 0],
                       [0, 0, 1]])
rotated_points = normalized_points @ rot_matrix.T
for class_id in range(5):
    mask = sampled_labels == class_id
    if np.any(mask):
        ax5.scatter(rotated_points[mask, 0], rotated_points[mask, 1],
                   c=class_colors[class_id], s=1, alpha=0.7)
ax5.set_title('5. Rotation Augmentation\n(45° rotation)', fontsize=12, fontweight='bold')
ax5.set_xlabel('X (normalized)', fontsize=10)
ax5.set_ylabel('Y (normalized)', fontsize=10)
ax5.grid(alpha=0.3)
ax5.set_aspect('equal')

# Scaling Augmentation
ax6 = plt.subplot(2, 3, 6)
scale = 1.05
scaled_points = normalized_points * scale
for class_id in range(5):
    mask = sampled_labels == class_id
    if np.any(mask):
        ax6.scatter(scaled_points[mask, 0], scaled_points[mask, 1],
                   c=class_colors[class_id], s=1, alpha=0.7, label=class_names[class_id])
ax6.set_title('6. Scaling Augmentation\n(1.05x scale)', fontsize=12, fontweight='bold')
ax6.set_xlabel('X (normalized)', fontsize=10)
ax6.set_ylabel('Y (normalized)', fontsize=10)
ax6.legend(markerscale=5, fontsize=8, loc='upper right')
ax6.grid(alpha=0.3)
ax6.set_aspect('equal')

plt.suptitle('Data Preprocessing Pipeline - Step by Step', fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(output_dir / 'preprocessing_steps.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {output_dir / 'preprocessing_steps.png'}")


# ============================================================================
# Figure 5: Data Statistics Summary
# ============================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Per-class point counts
ax = axes[0, 0]
class_counts = [np.sum(labels == i) for i in range(5)]
bars = ax.bar(class_names, class_counts, color=[class_colors[i] for i in range(5)],
              alpha=0.7, edgecolor='black', linewidth=1.5)
ax.set_ylabel('Number of Points', fontsize=12, fontweight='bold')
ax.set_title('Class Distribution in Sample', fontsize=14, fontweight='bold')
ax.grid(axis='y', alpha=0.3)
for bar, count in zip(bars, class_counts):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{count:,}',
            ha='center', va='bottom', fontsize=10, fontweight='bold')
plt.setp(ax.xaxis.get_majorticklabels(), rotation=15, ha='right')

# XYZ range statistics
ax = axes[0, 1]
coord_names = ['X', 'Y', 'Z']
coord_ranges = [(xyz[:, i].min(), xyz[:, i].max()) for i in range(3)]
coord_spans = [r[1] - r[0] for r in coord_ranges]

bars = ax.bar(coord_names, coord_spans, color=['#E74C3C', '#3498DB', '#27AE60'],
              alpha=0.7, edgecolor='black', linewidth=1.5)
ax.set_ylabel('Range (m)', fontsize=12, fontweight='bold')
ax.set_title('Coordinate Range Statistics', fontsize=14, fontweight='bold')
ax.grid(axis='y', alpha=0.3)
for bar, span, coord_range in zip(bars, coord_spans, coord_ranges):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{span:.2f}m\n({coord_range[0]:.1f} to {coord_range[1]:.1f})',
            ha='center', va='bottom', fontsize=9, fontweight='bold')

# RGB statistics
ax = axes[1, 0]
rgb_names = ['Red', 'Green', 'Blue']
rgb_means = [features[:, 3+i].mean() for i in range(3)]

bars = ax.bar(rgb_names, rgb_means, color=['#E74C3C', '#27AE60', '#3498DB'],
              alpha=0.7, edgecolor='black', linewidth=1.5)
ax.set_ylabel('Mean Value', fontsize=12, fontweight='bold')
ax.set_title('RGB Channel Statistics', fontsize=14, fontweight='bold')
ax.grid(axis='y', alpha=0.3)
for bar, mean in zip(bars, rgb_means):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{mean:.1f}',
            ha='center', va='bottom', fontsize=10, fontweight='bold')

# Intensity statistics
ax = axes[1, 1]
intensity = features[:, 6]
ax.hist(intensity, bins=50, color='#9B59B6', alpha=0.7, edgecolor='black', linewidth=1)
ax.set_xlabel('Intensity Value', fontsize=12, fontweight='bold')
ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
ax.set_title(f'Intensity Distribution\n(Mean: {intensity.mean():.2f}, Std: {intensity.std():.2f})',
             fontsize=14, fontweight='bold')
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'data_statistics_summary.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {output_dir / 'data_statistics_summary.png'}")


print("\n" + "="*80)
print("ALL DATA SAMPLE VISUALIZATIONS CREATED SUCCESSFULLY!")
print("="*80)
print("\nGenerated files:")
print("  1. sample_point_cloud_top_view.png - Top view (XY plane) with RGB and labels")
print("  2. sample_point_cloud_side_view.png - Side view (XZ plane) with intensity and labels")
print("  3. feature_distributions.png - XYZ and RGB distributions by class")
print("  4. preprocessing_steps.png - Step-by-step preprocessing visualization")
print("  5. data_statistics_summary.png - Statistical summary of data")
print("="*80)
