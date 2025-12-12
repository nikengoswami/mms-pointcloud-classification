"""
Evaluate trained PointNet++ model on test set
Generate confusion matrix, accuracy, F1-score, Kappa coefficient
"""

import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import json

from models.pointnet2 import PointNet2
from evaluation.metrics import SegmentationMetrics
from class_mapping_config import TARGET_CLASSES


def evaluate_model(checkpoint_path: str, test_data_path: str):
    """
    Evaluate model on test set

    Args:
        checkpoint_path: Path to model checkpoint
        test_data_path: Path to test data .npz file
    """

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load test data
    print(f"\nLoading test data from {test_data_path}...")
    data = np.load(test_data_path)
    xyz = data['xyz']
    features = data['features']
    labels = data['labels']

    print(f"Test data: {len(xyz):,} points")
    print(f"Features shape: {features.shape}")

    # Load model
    print(f"\nLoading model from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    num_features = features.shape[1]
    model = PointNet2(num_classes=5, num_features=num_features).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"Model loaded successfully")
    if 'epoch' in checkpoint:
        print(f"Checkpoint from epoch: {checkpoint['epoch']}")
    if 'metrics' in checkpoint:
        print(f"Checkpoint validation IoU: {checkpoint['metrics']['mean_iou']:.4f}")

    # Run inference on all points
    print(f"\nRunning inference on {len(xyz):,} points...")

    batch_size = 8192  # Process in batches
    all_predictions = []

    with torch.no_grad():
        for i in range(0, len(xyz), batch_size):
            end_idx = min(i + batch_size, len(xyz))

            # Get batch
            batch_xyz = xyz[i:end_idx]
            batch_features = features[i:end_idx]

            # Normalize XYZ
            xyz_mean = np.mean(batch_xyz, axis=0)
            batch_xyz_norm = batch_xyz - xyz_mean
            xyz_std = np.std(batch_xyz_norm)
            if xyz_std > 0:
                batch_xyz_norm = batch_xyz_norm / xyz_std

            # Update features with normalized XYZ
            batch_features = batch_features.copy()
            batch_features[:, :3] = batch_xyz_norm

            # Convert to tensors and add batch dimension
            coords_tensor = torch.from_numpy(batch_xyz_norm).float().unsqueeze(0).to(device)
            features_tensor = torch.from_numpy(batch_features).float().unsqueeze(0).to(device)

            # Forward pass
            logits = model(coords_tensor, features_tensor)
            predictions = torch.argmax(logits, dim=2).cpu().numpy().flatten()

            all_predictions.append(predictions)

            if (i // batch_size) % 10 == 0:
                print(f"  Processed {end_idx:,}/{len(xyz):,} points...")

    # Combine predictions
    all_predictions = np.concatenate(all_predictions)

    print(f"\nInference complete!")

    # Calculate metrics
    print(f"\n{'='*80}")
    print("EVALUATION RESULTS - POINTNET++")
    print('='*80)

    # Overall accuracy
    overall_acc = np.mean(all_predictions == labels)
    print(f"\nOverall Accuracy: {overall_acc:.4f} ({overall_acc*100:.2f}%)")

    # Use SegmentationMetrics for detailed metrics
    metrics_calc = SegmentationMetrics(5, list(TARGET_CLASSES.values()))
    metrics_calc.update(all_predictions, labels)
    metrics = metrics_calc.get_all_metrics()

    print(f"Mean IoU: {metrics['mean_iou']:.4f} ({metrics['mean_iou']*100:.2f}%)")
    print(f"Kappa Coefficient: {metrics['kappa']:.4f}")

    # Per-class metrics
    print(f"\n{'='*80}")
    print("PER-CLASS METRICS")
    print('='*80)

    for cls_id, cls_name in TARGET_CLASSES.items():
        cls_metrics = metrics['per_class_metrics'][cls_name]
        print(f"\n{cls_id}. {cls_name}:")
        print(f"   IoU:       {cls_metrics['iou']:.4f} ({cls_metrics['iou']*100:.2f}%)")
        print(f"   Precision: {cls_metrics['precision']:.4f}")
        print(f"   Recall:    {cls_metrics['recall']:.4f}")
        print(f"   F1-Score:  {cls_metrics['f1']:.4f}")

    # Confusion Matrix
    print(f"\n{'='*80}")
    print("CONFUSION MATRIX")
    print('='*80)

    cm = confusion_matrix(labels, all_predictions, labels=[0, 1, 2, 3, 4])

    print("\n     Predicted:")
    print("       ", end="")
    for cls_name in TARGET_CLASSES.values():
        print(f"{cls_name[:8]:>10s}", end="")
    print()

    print("True:")
    for i, cls_name in enumerate(TARGET_CLASSES.values()):
        print(f"{cls_name[:8]:>8s}:", end="")
        for j in range(5):
            print(f"{cm[i, j]:10,}", end="")
        print()

    # Save confusion matrix as figure
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=list(TARGET_CLASSES.values()),
                yticklabels=list(TARGET_CLASSES.values()))
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('PointNet++ Confusion Matrix - Test Set')
    plt.tight_layout()
    plt.savefig(output_dir / 'pointnet2_confusion_matrix.png', dpi=300, bbox_inches='tight')
    print(f"\nConfusion matrix saved to: {output_dir / 'pointnet2_confusion_matrix.png'}")

    # Save normalized confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues',
                xticklabels=list(TARGET_CLASSES.values()),
                yticklabels=list(TARGET_CLASSES.values()))
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('PointNet++ Normalized Confusion Matrix - Test Set')
    plt.tight_layout()
    plt.savefig(output_dir / 'pointnet2_confusion_matrix_normalized.png', dpi=300, bbox_inches='tight')
    print(f"Normalized confusion matrix saved to: {output_dir / 'pointnet2_confusion_matrix_normalized.png'}")

    # Classification report
    print(f"\n{'='*80}")
    print("CLASSIFICATION REPORT")
    print('='*80)
    print()

    report = classification_report(labels, all_predictions,
                                   target_names=list(TARGET_CLASSES.values()),
                                   digits=4)
    print(report)

    # Save results to JSON
    results = {
        'model': 'PointNet++',
        'overall_accuracy': float(overall_acc),
        'mean_iou': float(metrics['mean_iou']),
        'kappa': float(metrics['kappa']),
        'per_class_metrics': metrics['per_class_metrics'],
        'macro_avg': metrics['macro_avg'],
        'confusion_matrix': cm.tolist(),
        'confusion_matrix_normalized': cm_normalized.tolist()
    }

    if 'weighted_avg' in metrics:
        results['weighted_avg'] = metrics['weighted_avg']

    with open(output_dir / 'pointnet2_test_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_dir / 'pointnet2_test_results.json'}")

    print(f"\n{'='*80}")
    print("EVALUATION COMPLETE!")
    print('='*80)

    return results


if __name__ == "__main__":
    checkpoint_path = "checkpoints/pointnet2_best_model.pth"
    test_data_path = "data/processed/test_data.npz"

    results = evaluate_model(checkpoint_path, test_data_path)
