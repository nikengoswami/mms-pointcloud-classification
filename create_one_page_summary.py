"""
Create a 1-page PDF summary of the MMS Point Cloud Classification project
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import matplotlib.gridspec as gridspec
from pathlib import Path
from PIL import Image
import json

# Load results
with open('results/pointnet2_test_results.json', 'r') as f:
    results = json.load(f)

# Create figure
fig = plt.figure(figsize=(11, 8.5))  # Letter size
fig.patch.set_facecolor('white')

# Create grid layout
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.3,
                       left=0.08, right=0.95, top=0.92, bottom=0.05)

# ============================================================================
# HEADER
# ============================================================================
ax_header = fig.add_subplot(gs[0, :])
ax_header.axis('off')

ax_header.text(0.5, 0.85, 'MMS Point Cloud Classification Project',
               ha='center', va='top', fontsize=24, fontweight='bold',
               transform=ax_header.transAxes)

ax_header.text(0.5, 0.55, 'Deep Learning for Semantic Segmentation | December 2025',
               ha='center', va='top', fontsize=14,
               transform=ax_header.transAxes, style='italic')

# Status box
status_box = FancyBboxPatch((0.35, 0.05), 0.3, 0.35,
                            boxstyle="round,pad=0.02",
                            edgecolor='green', facecolor='lightgreen',
                            linewidth=2, transform=ax_header.transAxes)
ax_header.add_patch(status_box)

ax_header.text(0.5, 0.22, '✓ PROJECT COMPLETE',
               ha='center', va='center', fontsize=12, fontweight='bold',
               transform=ax_header.transAxes)


# ============================================================================
# LEFT COLUMN - Sample Data
# ============================================================================
ax1 = fig.add_subplot(gs[1, 0])
ax1.axis('off')
ax1.text(0.5, 1.05, 'Sample Point Cloud Data',
         ha='center', va='bottom', fontsize=11, fontweight='bold',
         transform=ax1.transAxes)

# Load and display top view
img = Image.open('results/sample_point_cloud_top_view.png')
ax1.imshow(img)
ax1.set_xlim(0, img.width)
ax1.set_ylim(img.height, 0)


# ============================================================================
# MIDDLE COLUMN - Workflow
# ============================================================================
ax2 = fig.add_subplot(gs[1, 1])
ax2.axis('off')
ax2.text(0.5, 1.05, 'Data Processing Workflow',
         ha='center', va='bottom', fontsize=11, fontweight='bold',
         transform=ax2.transAxes)

# Load and display workflow
img = Image.open('results/data_flow_diagram.png')
ax2.imshow(img)
ax2.set_xlim(0, img.width)
ax2.set_ylim(img.height, 0)


# ============================================================================
# RIGHT COLUMN - Results Summary
# ============================================================================
ax3 = fig.add_subplot(gs[1, 2])
ax3.axis('off')
ax3.text(0.5, 1.05, 'Final Results',
         ha='center', va='bottom', fontsize=11, fontweight='bold',
         transform=ax3.transAxes)

# Results text
results_text = f"""Overall Performance:
• Accuracy: {results['overall_accuracy']*100:.2f}%
• Mean IoU: {results['mean_iou']*100:.2f}%
• Kappa: {results['kappa']:.4f}

Per-Class IoU:
• Road: {results['per_class_metrics']['Road']['iou']*100:.1f}%
• Snow: {results['per_class_metrics']['Snow']['iou']*100:.1f}%
• Vehicle: {results['per_class_metrics']['Vehicle']['iou']*100:.1f}%
• Vegetation: {results['per_class_metrics']['Vegetation']['iou']*100:.1f}%
• Others: {results['per_class_metrics']['Others']['iou']*100:.1f}%

Dataset:
• Total: 1,461,189 labeled points
• Train/Val/Test: 70/15/15 split
• Features: XYZ, RGB, Intensity

Model: PointNet++
• Parameters: 968,069
• Training: 30 epochs (~4 hours)
• Hardware: RTX 4050 GPU"""

ax3.text(0.05, 0.95, results_text,
         ha='left', va='top', fontsize=9,
         transform=ax3.transAxes, family='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))


# ============================================================================
# BOTTOM LEFT - Preprocessing Steps
# ============================================================================
ax4 = fig.add_subplot(gs[2, 0])
ax4.axis('off')
ax4.text(0.5, 1.05, 'Preprocessing Pipeline',
         ha='center', va='bottom', fontsize=11, fontweight='bold',
         transform=ax4.transAxes)

# Load and display preprocessing
img = Image.open('results/preprocessing_steps.png')
ax4.imshow(img)
ax4.set_xlim(0, img.width)
ax4.set_ylim(img.height, 0)


# ============================================================================
# BOTTOM MIDDLE - Confusion Matrix
# ============================================================================
ax5 = fig.add_subplot(gs[2, 1])
ax5.axis('off')
ax5.text(0.5, 1.05, 'Confusion Matrix',
         ha='center', va='bottom', fontsize=11, fontweight='bold',
         transform=ax5.transAxes)

# Load and display confusion matrix
img = Image.open('results/pointnet2_confusion_matrix_normalized.png')
ax5.imshow(img)
ax5.set_xlim(0, img.width)
ax5.set_ylim(img.height, 0)


# ============================================================================
# BOTTOM RIGHT - Per-Class Performance
# ============================================================================
ax6 = fig.add_subplot(gs[2, 2])
ax6.axis('off')
ax6.text(0.5, 1.05, 'Per-Class Performance',
         ha='center', va='bottom', fontsize=11, fontweight='bold',
         transform=ax6.transAxes)

# Load and display per-class comparison
img = Image.open('results/per_class_iou_comparison.png')
ax6.imshow(img)
ax6.set_xlim(0, img.width)
ax6.set_ylim(img.height, 0)


# Save as PDF
output_path = Path("results/Project_Summary_One_Page.pdf")
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print("="*80)
print("ONE-PAGE PDF SUMMARY CREATED!")
print("="*80)
print(f"\nSaved to: {output_path}")
print("\nContents:")
print("  • Header with project title and status")
print("  • Sample point cloud data (top view)")
print("  • Data processing workflow diagram")
print("  • Final results summary (94.78% accuracy)")
print("  • Preprocessing pipeline (6 steps)")
print("  • Confusion matrix (normalized)")
print("  • Per-class performance comparison")
print("\nReady to send to professor!")
print("="*80)

plt.close()
