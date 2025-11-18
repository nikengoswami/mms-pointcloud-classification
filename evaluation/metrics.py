"""
Evaluation metrics for point cloud classification
Includes confusion matrix, accuracy, F1-score, and Kappa coefficient
"""

import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, cohen_kappa_score
from sklearn.metrics import precision_recall_fscore_support, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SegmentationMetrics:
    """Compute metrics for semantic segmentation"""

    def __init__(self, num_classes: int, class_names: Optional[List[str]] = None,
                 ignore_index: int = -1):
        """
        Args:
            num_classes: Number of classes
            class_names: Optional list of class names
            ignore_index: Index to ignore in evaluation (default: -1)
        """
        self.num_classes = num_classes
        self.class_names = class_names or [f"Class {i}" for i in range(num_classes)]
        self.ignore_index = ignore_index

        # Initialize accumulators
        self.reset()

    def reset(self):
        """Reset all metrics"""
        self.confusion_mat = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
        self.total_seen = 0
        self.total_correct = 0

    def update(self, predictions: np.ndarray, targets: np.ndarray):
        """
        Update metrics with new predictions

        Args:
            predictions: (N,) predicted class indices
            targets: (N,) ground truth class indices
        """
        # Filter out ignore index
        if self.ignore_index >= 0:
            valid_mask = targets != self.ignore_index
            predictions = predictions[valid_mask]
            targets = targets[valid_mask]

        # Update confusion matrix
        self.confusion_mat += confusion_matrix(
            targets, predictions,
            labels=list(range(self.num_classes))
        )

        # Update accuracy counters
        self.total_seen += len(targets)
        self.total_correct += np.sum(predictions == targets)

    def get_confusion_matrix(self) -> np.ndarray:
        """Get confusion matrix"""
        return self.confusion_mat

    def get_overall_accuracy(self) -> float:
        """Get overall accuracy"""
        if self.total_seen == 0:
            return 0.0
        return self.total_correct / self.total_seen

    def get_mean_accuracy(self) -> float:
        """Get mean class accuracy"""
        accuracies = []
        for i in range(self.num_classes):
            total = self.confusion_mat[i, :].sum()
            if total > 0:
                acc = self.confusion_mat[i, i] / total
                accuracies.append(acc)
        return np.mean(accuracies) if accuracies else 0.0

    def get_iou(self) -> Tuple[np.ndarray, float]:
        """
        Get Intersection over Union (IoU) per class and mean IoU

        Returns:
            Tuple of (per_class_iou, mean_iou)
        """
        iou_list = []

        for i in range(self.num_classes):
            tp = self.confusion_mat[i, i]
            fp = self.confusion_mat[:, i].sum() - tp
            fn = self.confusion_mat[i, :].sum() - tp

            denominator = tp + fp + fn
            if denominator > 0:
                iou = tp / denominator
            else:
                iou = 0.0

            iou_list.append(iou)

        iou_array = np.array(iou_list)
        mean_iou = np.mean(iou_array)

        return iou_array, mean_iou

    def get_precision_recall_f1(self) -> Dict:
        """
        Get precision, recall, and F1-score per class

        Returns:
            Dictionary with per-class and average metrics
        """
        metrics = {}

        for i in range(self.num_classes):
            tp = self.confusion_mat[i, i]
            fp = self.confusion_mat[:, i].sum() - tp
            fn = self.confusion_mat[i, :].sum() - tp

            # Precision
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

            # Recall
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

            # F1-score
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            metrics[self.class_names[i]] = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'support': self.confusion_mat[i, :].sum()
            }

        # Compute averages
        precisions = [m['precision'] for m in metrics.values() if m['support'] > 0]
        recalls = [m['recall'] for m in metrics.values() if m['support'] > 0]
        f1s = [m['f1'] for m in metrics.values() if m['support'] > 0]

        metrics['macro_avg'] = {
            'precision': np.mean(precisions) if precisions else 0.0,
            'recall': np.mean(recalls) if recalls else 0.0,
            'f1': np.mean(f1s) if f1s else 0.0
        }

        # Weighted average
        total_support = sum([m['support'] for m in metrics.values() if isinstance(m, dict) and 'support' in m])
        if total_support > 0:
            weighted_precision = sum([m['precision'] * m['support'] for m in metrics.values()
                                     if isinstance(m, dict) and 'support' in m]) / total_support
            weighted_recall = sum([m['recall'] * m['support'] for m in metrics.values()
                                  if isinstance(m, dict) and 'support' in m]) / total_support
            weighted_f1 = sum([m['f1'] * m['support'] for m in metrics.values()
                              if isinstance(m, dict) and 'support' in m]) / total_support

            metrics['weighted_avg'] = {
                'precision': weighted_precision,
                'recall': weighted_recall,
                'f1': weighted_f1
            }

        return metrics

    def get_kappa_coefficient(self) -> float:
        """
        Get Cohen's Kappa coefficient

        Returns:
            Kappa coefficient
        """
        # Convert confusion matrix to predictions and targets
        predictions = []
        targets = []

        for i in range(self.num_classes):
            for j in range(self.num_classes):
                count = self.confusion_mat[i, j]
                predictions.extend([j] * count)
                targets.extend([i] * count)

        if len(predictions) == 0:
            return 0.0

        return cohen_kappa_score(targets, predictions)

    def get_all_metrics(self) -> Dict:
        """
        Get all metrics in a single dictionary

        Returns:
            Dictionary with all metrics
        """
        iou_per_class, mean_iou = self.get_iou()
        prf_metrics = self.get_precision_recall_f1()

        metrics = {
            'overall_accuracy': self.get_overall_accuracy(),
            'mean_accuracy': self.get_mean_accuracy(),
            'mean_iou': mean_iou,
            'kappa': self.get_kappa_coefficient(),
            'confusion_matrix': self.confusion_mat.tolist(),
            'per_class_metrics': {}
        }

        # Add per-class metrics
        for i, class_name in enumerate(self.class_names):
            metrics['per_class_metrics'][class_name] = {
                'iou': float(iou_per_class[i]),
                'precision': prf_metrics[class_name]['precision'],
                'recall': prf_metrics[class_name]['recall'],
                'f1': prf_metrics[class_name]['f1'],
                'support': int(prf_metrics[class_name]['support'])
            }

        # Add average metrics
        metrics['macro_avg'] = prf_metrics['macro_avg']
        if 'weighted_avg' in prf_metrics:
            metrics['weighted_avg'] = prf_metrics['weighted_avg']

        return metrics

    def print_metrics(self):
        """Print all metrics in a readable format"""
        metrics = self.get_all_metrics()

        print("\n" + "=" * 80)
        print("EVALUATION METRICS")
        print("=" * 80)

        print(f"\nOverall Accuracy: {metrics['overall_accuracy']:.4f}")
        print(f"Mean Accuracy:    {metrics['mean_accuracy']:.4f}")
        print(f"Mean IoU:         {metrics['mean_iou']:.4f}")
        print(f"Kappa Coefficient: {metrics['kappa']:.4f}")

        print("\n" + "-" * 80)
        print("PER-CLASS METRICS")
        print("-" * 80)
        print(f"{'Class':<20} {'IoU':>8} {'Prec':>8} {'Recall':>8} {'F1':>8} {'Support':>10}")
        print("-" * 80)

        for class_name, class_metrics in metrics['per_class_metrics'].items():
            print(f"{class_name:<20} "
                  f"{class_metrics['iou']:>8.4f} "
                  f"{class_metrics['precision']:>8.4f} "
                  f"{class_metrics['recall']:>8.4f} "
                  f"{class_metrics['f1']:>8.4f} "
                  f"{class_metrics['support']:>10}")

        print("-" * 80)
        print(f"{'Macro Average':<20} "
              f"{'':>8} "
              f"{metrics['macro_avg']['precision']:>8.4f} "
              f"{metrics['macro_avg']['recall']:>8.4f} "
              f"{metrics['macro_avg']['f1']:>8.4f}")

        if 'weighted_avg' in metrics:
            print(f"{'Weighted Average':<20} "
                  f"{'':>8} "
                  f"{metrics['weighted_avg']['precision']:>8.4f} "
                  f"{metrics['weighted_avg']['recall']:>8.4f} "
                  f"{metrics['weighted_avg']['f1']:>8.4f}")

        print("=" * 80)

    def plot_confusion_matrix(self, save_path: Optional[str] = None, normalize: bool = False):
        """
        Plot confusion matrix

        Args:
            save_path: Path to save the plot
            normalize: Whether to normalize the confusion matrix
        """
        cm = self.confusion_mat.copy()

        if normalize:
            # Normalize by row (true labels)
            row_sums = cm.sum(axis=1, keepdims=True)
            cm = cm.astype(np.float32) / (row_sums + 1e-10)

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt='.2f' if normalize else 'd',
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            cbar_kws={'label': 'Proportion' if normalize else 'Count'}
        )

        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix' + (' (Normalized)' if normalize else ''))
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix saved to: {save_path}")

        plt.show()

    def plot_per_class_metrics(self, save_path: Optional[str] = None):
        """
        Plot per-class metrics (IoU, Precision, Recall, F1)

        Args:
            save_path: Path to save the plot
        """
        metrics = self.get_all_metrics()
        class_names = list(metrics['per_class_metrics'].keys())

        # Extract metrics
        ious = [metrics['per_class_metrics'][c]['iou'] for c in class_names]
        precisions = [metrics['per_class_metrics'][c]['precision'] for c in class_names]
        recalls = [metrics['per_class_metrics'][c]['recall'] for c in class_names]
        f1s = [metrics['per_class_metrics'][c]['f1'] for c in class_names]

        # Plot
        x = np.arange(len(class_names))
        width = 0.2

        fig, ax = plt.subplots(figsize=(12, 6))

        ax.bar(x - 1.5*width, ious, width, label='IoU', alpha=0.8)
        ax.bar(x - 0.5*width, precisions, width, label='Precision', alpha=0.8)
        ax.bar(x + 0.5*width, recalls, width, label='Recall', alpha=0.8)
        ax.bar(x + 1.5*width, f1s, width, label='F1-Score', alpha=0.8)

        ax.set_xlabel('Class')
        ax.set_ylabel('Score')
        ax.set_title('Per-Class Metrics')
        ax.set_xticks(x)
        ax.set_xticklabels(class_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim([0, 1.0])

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Per-class metrics plot saved to: {save_path}")

        plt.show()


def compute_metrics_from_arrays(predictions: np.ndarray, targets: np.ndarray,
                                num_classes: int, class_names: Optional[List[str]] = None) -> Dict:
    """
    Compute all metrics from prediction and target arrays

    Args:
        predictions: (N,) predicted labels
        targets: (N,) ground truth labels
        num_classes: Number of classes
        class_names: Optional class names

    Returns:
        Dictionary with all metrics
    """
    metrics_calculator = SegmentationMetrics(num_classes, class_names)
    metrics_calculator.update(predictions, targets)
    return metrics_calculator.get_all_metrics()


if __name__ == "__main__":
    # Test metrics
    print("Testing evaluation metrics...")

    # Create dummy predictions and targets
    num_classes = 5
    class_names = ['Road', 'Snow', 'Vehicle', 'Vegetation', 'Others']

    np.random.seed(42)
    targets = np.random.randint(0, num_classes, 10000)
    predictions = targets.copy()

    # Add some errors
    error_indices = np.random.choice(len(predictions), 2000, replace=False)
    predictions[error_indices] = np.random.randint(0, num_classes, len(error_indices))

    # Compute metrics
    metrics_calc = SegmentationMetrics(num_classes, class_names)
    metrics_calc.update(predictions, targets)

    # Print metrics
    metrics_calc.print_metrics()

    # Plot confusion matrix
    metrics_calc.plot_confusion_matrix(normalize=True)

    # Plot per-class metrics
    metrics_calc.plot_per_class_metrics()

    print("\nMetrics test passed!")
