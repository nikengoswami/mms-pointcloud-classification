"""
Train PointNet++ from preprocessed numpy data
Optimized for MMS point cloud classification
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
from evaluation.metrics import SegmentationMetrics
from class_mapping_config import TARGET_CLASSES

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PreprocessedDataset(Dataset):
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
        start_idx = np.random.randint(0, max(1, len(self.xyz) - self.num_points))
        end_idx = start_idx + self.num_points

        if end_idx > len(self.xyz):
            indices = np.random.choice(len(self.xyz), self.num_points, replace=True)
        else:
            indices = np.arange(start_idx, end_idx)

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
            scale = np.random.uniform(0.95, 1.05)
            xyz *= scale
            features[:, :3] = xyz

        return (torch.from_numpy(xyz).float(),
                torch.from_numpy(features).float(),
                torch.from_numpy(labels).long())


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
        loss = criterion(logits.reshape(-1, 5), labels.reshape(-1))

        # Backward pass
        loss.backward()
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
            loss = criterion(logits.reshape(-1, 5), labels.reshape(-1))

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
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Paths
    train_path = "data/processed/train_data.npz"
    val_path = "data/processed/val_data.npz"
    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)

    # Hyperparameters
    num_points = 2048
    batch_size = 8
    num_epochs = 30
    learning_rate = 0.001

    # Create datasets
    logger.info("Loading datasets...")
    train_dataset = PreprocessedDataset(train_path, num_points=num_points, augment=True)
    val_dataset = PreprocessedDataset(val_path, num_points=num_points, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                               num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=0, pin_memory=True)

    # Create model
    logger.info("Creating PointNet++ model...")
    model = PointNet2(num_classes=5, num_features=7).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {num_params:,}")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )

    # Metrics
    train_metrics = SegmentationMetrics(5, list(TARGET_CLASSES.values()))
    val_metrics = SegmentationMetrics(5, list(TARGET_CLASSES.values()))

    # Training loop
    best_iou = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [],
               'val_acc': [], 'val_iou': [], 'val_kappa': []}

    logger.info(f"\nStarting training for {num_epochs} epochs...")
    logger.info(f"Batch size: {batch_size}, Num points: {num_points}")

    for epoch in range(num_epochs):
        logger.info(f"\nEpoch {epoch+1}/{num_epochs}")

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
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': val_metrics_dict
            }
            torch.save(checkpoint, checkpoint_dir / 'pointnet2_best_model.pth')
            logger.info(f"Saved best model with IoU: {best_iou:.4f}")

    # Save final model
    final_checkpoint = {
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': history
    }
    torch.save(final_checkpoint, checkpoint_dir / 'pointnet2_final_model.pth')

    # Save history
    with open(checkpoint_dir / 'pointnet2_training_history.json', 'w') as f:
        json.dump(history, f, indent=2)

    logger.info("\nTraining complete!")
    logger.info(f"Best validation IoU: {best_iou:.4f}")


if __name__ == "__main__":
    main()
