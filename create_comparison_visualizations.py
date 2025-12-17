"""
Generate comparison visualizations between SimplePointNet and PointNet++
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# Load results
with open('results/test_results.json', 'r') as f:
    simple_results = json.load(f)

with open('results/pointnet2_test_results.json', 'r') as f:
    pointnet2_results = json.load(f)

output_dir = Path('results')
output_dir.mkdir(exist_ok=True)

# 1. Overall Metrics Comparison
fig, ax = plt.subplots(1, 1, figsize=(10, 6))

metrics = ['Overall Accuracy', 'Mean IoU', 'Kappa']
simple_values = [
    simple_results['overall_accuracy'] * 100,
    simple_results['mean_iou'] * 100,
    simple_results['kappa'] * 100
]
pointnet2_values = [
    pointnet2_results['overall_accuracy'] * 100,
    pointnet2_results['mean_iou'] * 100,
    pointnet2_results['kappa'] * 100
]

x = np.arange(len(metrics))
width = 0.35

bars1 = ax.bar(x - width/2, simple_values, width, label='SimplePointNet', color='#5DADE2', alpha=0.8)
bars2 = ax.bar(x + width/2, pointnet2_values, width, label='PointNet++', color='#48C9B0', alpha=0.8)

ax.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
ax.set_title('Overall Metrics Comparison: SimplePointNet vs PointNet++', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=11)
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}%',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / 'overall_metrics_comparison.png', dpi=300, bbox_inches='tight')
print(f"Saved: {output_dir / 'overall_metrics_comparison.png'}")
plt.close()


# 2. Per-Class IoU Comparison
fig, ax = plt.subplots(1, 1, figsize=(12, 6))

classes = ['Road', 'Snow', 'Vehicle', 'Vegetation', 'Others']
simple_ious = [simple_results['per_class_metrics'][c]['iou'] * 100 for c in classes]
pointnet2_ious = [pointnet2_results['per_class_metrics'][c]['iou'] * 100 for c in classes]

x = np.arange(len(classes))
width = 0.35

bars1 = ax.bar(x - width/2, simple_ious, width, label='SimplePointNet', color='#E74C3C', alpha=0.8)
bars2 = ax.bar(x + width/2, pointnet2_ious, width, label='PointNet++', color='#27AE60', alpha=0.8)

ax.set_ylabel('IoU (%)', fontsize=12, fontweight='bold')
ax.set_title('Per-Class IoU Comparison: SimplePointNet vs PointNet++', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(classes, fontsize=11)
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

# Add improvement arrows
for i, (s, p) in enumerate(zip(simple_ious, pointnet2_ious)):
    improvement = p - s
    if improvement > 0:
        ax.annotate('', xy=(i, p), xytext=(i, s),
                    arrowprops=dict(arrowstyle='->', color='green', lw=2, alpha=0.5))
        ax.text(i, (s + p) / 2, f'+{improvement:.1f}%',
                ha='center', va='center', fontsize=8,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7))

plt.tight_layout()
plt.savefig(output_dir / 'per_class_iou_comparison.png', dpi=300, bbox_inches='tight')
print(f"Saved: {output_dir / 'per_class_iou_comparison.png'}")
plt.close()


# 3. Per-Class F1-Score Comparison
fig, ax = plt.subplots(1, 1, figsize=(12, 6))

simple_f1s = [simple_results['per_class_metrics'][c]['f1'] * 100 for c in classes]
pointnet2_f1s = [pointnet2_results['per_class_metrics'][c]['f1'] * 100 for c in classes]

x = np.arange(len(classes))
width = 0.35

bars1 = ax.bar(x - width/2, simple_f1s, width, label='SimplePointNet', color='#9B59B6', alpha=0.8)
bars2 = ax.bar(x + width/2, pointnet2_f1s, width, label='PointNet++', color='#F39C12', alpha=0.8)

ax.set_ylabel('F1-Score (%)', fontsize=12, fontweight='bold')
ax.set_title('Per-Class F1-Score Comparison: SimplePointNet vs PointNet++', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(classes, fontsize=11)
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / 'per_class_f1_comparison.png', dpi=300, bbox_inches='tight')
print(f"Saved: {output_dir / 'per_class_f1_comparison.png'}")
plt.close()


# 4. Precision vs Recall Scatter Plot
fig, ax = plt.subplots(1, 1, figsize=(10, 10))

# SimplePointNet points
simple_precisions = [simple_results['per_class_metrics'][c]['precision'] * 100 for c in classes]
simple_recalls = [simple_results['per_class_metrics'][c]['recall'] * 100 for c in classes]

# PointNet++ points
pointnet2_precisions = [pointnet2_results['per_class_metrics'][c]['precision'] * 100 for c in classes]
pointnet2_recalls = [pointnet2_results['per_class_metrics'][c]['recall'] * 100 for c in classes]

# Plot points
for i, cls in enumerate(classes):
    ax.scatter(simple_recalls[i], simple_precisions[i],
               s=200, alpha=0.6, label=f'{cls} (Simple)', marker='o', edgecolors='black', linewidth=1.5)
    ax.scatter(pointnet2_recalls[i], pointnet2_precisions[i],
               s=200, alpha=0.6, label=f'{cls} (PN++)', marker='s', edgecolors='black', linewidth=1.5)

    # Draw arrow showing improvement
    ax.annotate('', xy=(pointnet2_recalls[i], pointnet2_precisions[i]),
                xytext=(simple_recalls[i], simple_precisions[i]),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.5, alpha=0.5))

ax.set_xlabel('Recall (%)', fontsize=12, fontweight='bold')
ax.set_ylabel('Precision (%)', fontsize=12, fontweight='bold')
ax.set_title('Precision vs Recall: SimplePointNet (circles) vs PointNet++ (squares)',
             fontsize=14, fontweight='bold')
ax.legend(fontsize=9, loc='lower left', ncol=2)
ax.grid(alpha=0.3)
ax.set_xlim([70, 100])
ax.set_ylim([70, 100])

# Add diagonal reference line (F1=0.9)
x_diag = np.linspace(70, 100, 100)
y_diag = x_diag
ax.plot(x_diag, y_diag, 'k--', alpha=0.3, linewidth=1, label='Precision = Recall')

plt.tight_layout()
plt.savefig(output_dir / 'precision_recall_scatter.png', dpi=300, bbox_inches='tight')
print(f"Saved: {output_dir / 'precision_recall_scatter.png'}")
plt.close()


# 5. Improvement Heatmap
fig, ax = plt.subplots(1, 1, figsize=(10, 6))

metrics_names = ['IoU', 'Precision', 'Recall', 'F1-Score']
improvements = []

for cls in classes:
    cls_improvements = []
    for metric in ['iou', 'precision', 'recall', 'f1']:
        simple_val = simple_results['per_class_metrics'][cls][metric]
        pointnet2_val = pointnet2_results['per_class_metrics'][cls][metric]
        improvement = (pointnet2_val - simple_val) * 100  # percentage points
        cls_improvements.append(improvement)
    improvements.append(cls_improvements)

improvements = np.array(improvements)

sns.heatmap(improvements, annot=True, fmt='.2f', cmap='RdYlGn', center=0,
            xticklabels=metrics_names, yticklabels=classes,
            cbar_kws={'label': 'Improvement (percentage points)'},
            linewidths=0.5, linecolor='gray', ax=ax)

ax.set_title('PointNet++ Improvement over SimplePointNet (percentage points)',
             fontsize=14, fontweight='bold')
ax.set_xlabel('Metrics', fontsize=12, fontweight='bold')
ax.set_ylabel('Classes', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / 'improvement_heatmap.png', dpi=300, bbox_inches='tight')
print(f"Saved: {output_dir / 'improvement_heatmap.png'}")
plt.close()


# 6. Model Summary Table (as image)
fig, ax = plt.subplots(1, 1, figsize=(12, 4))
ax.axis('tight')
ax.axis('off')

table_data = [
    ['Metric', 'SimplePointNet', 'PointNet++', 'Improvement'],
    ['Overall Accuracy', f"{simple_results['overall_accuracy']*100:.2f}%",
     f"{pointnet2_results['overall_accuracy']*100:.2f}%",
     f"+{(pointnet2_results['overall_accuracy'] - simple_results['overall_accuracy'])*100:.2f}%"],
    ['Mean IoU', f"{simple_results['mean_iou']*100:.2f}%",
     f"{pointnet2_results['mean_iou']*100:.2f}%",
     f"+{(pointnet2_results['mean_iou'] - simple_results['mean_iou'])*100:.2f}%"],
    ['Kappa Coefficient', f"{simple_results['kappa']:.4f}",
     f"{pointnet2_results['kappa']:.4f}",
     f"+{pointnet2_results['kappa'] - simple_results['kappa']:.4f}"],
    ['Parameters', '192,517', '968,069', '+775,552'],
    ['Training Time', '~2.5 hours', '~4 hours', '+1.5 hours'],
]

table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                 colWidths=[0.3, 0.25, 0.25, 0.2])
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2.5)

# Style header row
for i in range(4):
    table[(0, i)].set_facecolor('#34495E')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Style data rows
for i in range(1, len(table_data)):
    for j in range(4):
        if i % 2 == 0:
            table[(i, j)].set_facecolor('#ECF0F1')
        else:
            table[(i, j)].set_facecolor('white')

plt.title('Model Comparison Summary', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(output_dir / 'model_summary_table.png', dpi=300, bbox_inches='tight')
print(f"Saved: {output_dir / 'model_summary_table.png'}")
plt.close()

print("\n" + "="*80)
print("ALL COMPARISON VISUALIZATIONS GENERATED SUCCESSFULLY!")
print("="*80)
print("\nGenerated files:")
print("  1. overall_metrics_comparison.png - Overall metrics comparison")
print("  2. per_class_iou_comparison.png - Per-class IoU comparison")
print("  3. per_class_f1_comparison.png - Per-class F1-Score comparison")
print("  4. precision_recall_scatter.png - Precision vs Recall scatter plot")
print("  5. improvement_heatmap.png - Improvement heatmap across all metrics")
print("  6. model_summary_table.png - Summary table")
print("="*80)
