"""
Train PointNet++ with 6 classes (including Building)
Optimized for best performance
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import json
import logging
from tqdm import tqdm

from models.pointnet2 import PointNet2
from class_mapping_config import TARGET_CLASSES

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

NUM_CLASSES = 6


class PointCloudDataset(Dataset):
    """Dataset from preprocessed numpy files"""

    def __init__(self, data_path: str, num_points: int = 2048, augment: bool = False):
        self.num_points = num_points
        self.augment = augment

        # Load data
        data = np.load(data_path)
        self.xyz = data['xyz']
        self.features = data['features']
        self.labels = data['labels']

        logger.info(f"Loaded {len(self.xyz):,} points from {data_path}")

    def __len__(self):
        return len(self.xyz) // (self.num_points // 2)

    def __getitem__(self, idx):
        # Random sampling
        indices = np.random.choice(len(self.xyz), self.num_points, replace=False)

        xyz = self.xyz[indices].copy()
        features = self.features[indices].copy()
        labels = self.labels[indices].copy()

        # Normalize XYZ
        xyz_mean = np.mean(xyz, axis=0)
        xyz = xyz - xyz_mean
        xyz_std = np.std(xyz)
        if xyz_std > 0:
            xyz = xyz / xyz_std

        # Update features with normalized XYZ
        features[:, :3] = xyz

        # Augmentation
        if self.augment:
            # Random rotation around Z-axis
            angle = np.random.uniform(0, 2 * np.pi)
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            rot_matrix = np.array([[cos_a, -sin_a, 0],
                                   [sin_a, cos_a, 0],
                                   [0, 0, 1]])
            xyz = xyz @ rot_matrix.T
            features[:, :3] = xyz

            # Random scaling
            scale = np.random.uniform(0.9, 1.1)
            xyz *= scale
            features[:, :3] = xyz

            # Random jitter
            jitter = np.random.normal(0, 0.01, xyz.shape)
            xyz += jitter
            features[:, :3] = xyz

            # Random flip
            if np.random.random() > 0.5:
                xyz[:, 0] = -xyz[:, 0]
                features[:, 0] = -features[:, 0]

        return (torch.from_numpy(xyz).float(),
                torch.from_numpy(features).float(),
                torch.from_numpy(labels).long())


class SegmentationMetrics:
    """Calculate segmentation metrics"""

    def __init__(self, num_classes, class_names):
        self.num_classes = num_classes
        self.class_names = class_names
        self.reset()

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

    def update(self, preds, labels):
        for p, l in zip(preds.flatten(), labels.flatten()):
            if 0 <= p < self.num_classes and 0 <= l < self.num_classes:
                self.confusion_matrix[l, p] += 1

    def get_all_metrics(self):
        cm = self.confusion_matrix

        # Overall accuracy
        total_correct = np.diag(cm).sum()
        total_points = cm.sum()
        overall_accuracy = total_correct / (total_points + 1e-10)

        # Per-class IoU
        ious = []
        for i in range(self.num_classes):
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp
            iou = tp / (tp + fp + fn + 1e-10)
            ious.append(iou)

        mean_iou = np.mean(ious)

        # Kappa coefficient
        po = overall_accuracy
        pe = sum((cm[i, :].sum() * cm[:, i].sum()) for i in range(self.num_classes)) / (total_points ** 2 + 1e-10)
        kappa = (po - pe) / (1 - pe + 1e-10)

        return {
            'overall_accuracy': overall_accuracy,
            'mean_iou': mean_iou,
            'per_class_iou': {self.class_names[i]: ious[i] for i in range(self.num_classes)},
            'kappa': kappa
        }


def compute_class_weights(labels):
    """Compute class weights for imbalanced data"""
    unique, counts = np.unique(labels, return_counts=True)
    total = counts.sum()

    # Inverse frequency weighting
    weights = np.zeros(NUM_CLASSES)
    for cls, count in zip(unique, counts):
        if cls < NUM_CLASSES:
            weights[cls] = total / (NUM_CLASSES * count + 1e-10)

    # Normalize
    weights = weights / weights.sum() * NUM_CLASSES

    # Clip extreme weights
    weights = np.clip(weights, 0.5, 5.0)

    return torch.FloatTensor(weights)


def train_epoch(model, loader, criterion, optimizer, device, metrics):
    """Train for one epoch"""
    model.train()
    metrics.reset()
    total_loss = 0
    total_correct = 0
    total_points = 0

    pbar = tqdm(loader, desc="Train")
    for coords, features, labels in pbar:
        coords = coords.to(device)
        features = features.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Forward pass
        logits = model(coords, features)

        # Compute loss
        loss = criterion(logits.reshape(-1, NUM_CLASSES), labels.reshape(-1))

        # Backward pass
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # Metrics
        preds = torch.argmax(logits, dim=2).cpu().numpy().flatten()
        labels_np = labels.cpu().numpy().flatten()
        metrics.update(preds, labels_np)

        total_loss += loss.item()
        total_correct += np.sum(preds == labels_np)
        total_points += len(labels_np)

        pbar.set_postfix({'loss': f'{loss.item():.4f}',
                          'acc': f'{total_correct/total_points:.4f}'})

    avg_loss = total_loss / len(loader)
    avg_acc = total_correct / total_points

    return avg_loss, avg_acc


def validate(model, loader, criterion, device, metrics):
    """Validate model"""
    model.eval()
    metrics.reset()
    total_loss = 0

    with torch.no_grad():
        pbar = tqdm(loader, desc="Val")
        for coords, features, labels in pbar:
            coords = coords.to(device)
            features = features.to(device)
            labels = labels.to(device)

            # Forward pass
            logits = model(coords, features)

            # Compute loss
            loss = criterion(logits.reshape(-1, NUM_CLASSES), labels.reshape(-1))

            # Metrics
            preds = torch.argmax(logits, dim=2).cpu().numpy().flatten()
            labels_np = labels.cpu().numpy().flatten()
            metrics.update(preds, labels_np)

            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    avg_loss = total_loss / len(loader)
    all_metrics = metrics.get_all_metrics()

    return avg_loss, all_metrics


def main():
    print("=" * 70)
    print("  POINTNET++ TRAINING - 6 CLASSES (WITH BUILDING)")
    print("=" * 70)

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    if device.type == 'cuda':
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    # Paths
    train_path = "data/processed/train_data.npz"
    val_path = "data/processed/val_data.npz"
    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)

    # Hyperparameters
    num_points = 2048
    batch_size = 8
    num_epochs = 10  # Optimized: 10 epochs for 6-class model (~4.5 hours)
    learning_rate = 0.001

    # Create datasets
    logger.info("Loading datasets...")
    train_dataset = PointCloudDataset(train_path, num_points=num_points, augment=True)
    val_dataset = PointCloudDataset(val_path, num_points=num_points, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=0, pin_memory=True)

    # Compute class weights for imbalanced data
    logger.info("Computing class weights...")
    train_data = np.load(train_path)
    class_weights = compute_class_weights(train_data['labels'])
    logger.info(f"Class weights: {class_weights.numpy()}")

    # Create model (8 features: XYZ + RGB + Intensity + Height)
    logger.info("Creating PointNet++ model with 6 classes and 8 features...")
    model = PointNet2(num_classes=NUM_CLASSES, num_features=8).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {num_params:,}")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3, verbose=True
    )

    # Metrics
    class_names = list(TARGET_CLASSES.values())
    train_metrics = SegmentationMetrics(NUM_CLASSES, class_names)
    val_metrics = SegmentationMetrics(NUM_CLASSES, class_names)

    # Training loop
    best_iou = 0.0
    best_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [],
               'val_acc': [], 'val_iou': [], 'val_kappa': []}

    print("\n" + "=" * 70)
    print(f"  Starting training for {num_epochs} epochs")
    print(f"  Batch size: {batch_size}, Points per sample: {num_points}")
    print(f"  Classes: {', '.join(class_names)}")
    print("=" * 70 + "\n")

    for epoch in range(num_epochs):
        logger.info(f"\n{'='*50}")
        logger.info(f"Epoch {epoch+1}/{num_epochs}")
        logger.info(f"{'='*50}")

        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion,
                                            optimizer, device, train_metrics)
        logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")

        # Validate
        val_loss, val_metrics_dict = validate(model, val_loader, criterion,
                                              device, val_metrics)

        val_acc = val_metrics_dict['overall_accuracy']
        val_iou = val_metrics_dict['mean_iou']
        val_kappa = val_metrics_dict['kappa']

        logger.info(f"Val Loss: {val_loss:.4f}")
        logger.info(f"Val Accuracy: {val_acc:.4f}")
        logger.info(f"Val Mean IoU: {val_iou:.4f}")
        logger.info(f"Val Kappa: {val_kappa:.4f}")

        # Per-class IoU
        logger.info("Per-class IoU:")
        for cls_name, iou in val_metrics_dict['per_class_iou'].items():
            logger.info(f"  {cls_name}: {iou:.4f}")

        # Update scheduler
        scheduler.step(val_iou)

        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_iou'].append(val_iou)
        history['val_kappa'].append(val_kappa)

        # Save best model
        if val_iou > best_iou:
            best_iou = val_iou
            best_acc = val_acc
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': val_metrics_dict,
                'num_classes': NUM_CLASSES,
                'class_names': class_names
            }
            torch.save(checkpoint, checkpoint_dir / 'pointnet2_best_model.pth')
            logger.info(f"[SAVED] Best model with IoU: {best_iou:.4f}, Acc: {best_acc:.4f}")

    # Save final model
    final_checkpoint = {
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': history,
        'num_classes': NUM_CLASSES,
        'class_names': class_names
    }
    torch.save(final_checkpoint, checkpoint_dir / 'pointnet2_final_model.pth')

    # Save history
    with open(checkpoint_dir / 'pointnet2_training_history.json', 'w') as f:
        json.dump(history, f, indent=2)

    print("\n" + "=" * 70)
    print("  TRAINING COMPLETE!")
    print("=" * 70)
    print(f"  Best Validation Accuracy: {best_acc:.4f} ({best_acc*100:.2f}%)")
    print(f"  Best Validation Mean IoU: {best_iou:.4f} ({best_iou*100:.2f}%)")
    print(f"  Model saved to: checkpoints/pointnet2_best_model.pth")
    print("=" * 70)


if __name__ == "__main__":
    main()
