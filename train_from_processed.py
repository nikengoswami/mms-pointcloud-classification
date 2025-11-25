"""
Train RandLA-Net from preprocessed numpy data
Optimized for MMS point cloud classification
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import argparse
import json
import logging
from tqdm import tqdm
from datetime import datetime

from models.simple_pointnet import SimplePointNet
from evaluation.metrics import SegmentationMetrics
from class_mapping_config import TARGET_CLASSES

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PreprocessedDataset(Dataset):
    """Dataset from preprocessed numpy files"""

    def __init__(self, data_path: str, num_points: int = 4096, augment: bool = False):
        self.num_points = num_points
        self.augment = augment

        # Load data
        data = np.load(data_path)
        self.xyz = data['xyz']
        self.features = data['features']
        self.labels = data['labels']

        logger.info(f"Loaded {len(self.xyz):,} points from {data_path}")

    def __len__(self):
        # Return number of samples based on point sampling
        return len(self.xyz) // (self.num_points // 2)  # Overlapping samples

    def __getitem__(self, idx):
        # Random sampling with overlap
        start_idx = np.random.randint(0, max(1, len(self.xyz) - self.num_points))
        end_idx = start_idx + self.num_points

        if end_idx > len(self.xyz):
            # Wrap around or pad
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
            # Random rotation around Z
            angle = np.random.uniform(0, 2 * np.pi)
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            rot_matrix = np.array([[cos_a, -sin_a, 0],
                                   [sin_a, cos_a, 0],
                                   [0, 0, 1]])
            xyz = xyz @ rot_matrix.T
            features[:, :3] = xyz

            # Random scaling
            scale = np.random.uniform(0.9, 1.1)
            xyz = xyz * scale
            features[:, :3] = xyz

            # Jittering
            noise = np.random.normal(0, 0.01, xyz.shape)
            xyz = xyz + noise
            features[:, :3] = xyz

        return {
            'coords': torch.from_numpy(xyz).float(),
            'features': torch.from_numpy(features).float(),
            'labels': torch.from_numpy(labels).long()
        }


def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_points = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")

    for batch in pbar:
        coords = batch['coords'].to(device)
        features = batch['features'].to(device)
        labels = batch['labels'].to(device)

        # Forward
        optimizer.zero_grad()
        logits = model(coords, features)

        # Loss
        loss = criterion(logits.reshape(-1, 5), labels.reshape(-1))

        # Backward
        loss.backward()
        optimizer.step()

        # Metrics
        predictions = torch.argmax(logits, dim=2)
        correct = (predictions == labels).sum().item()
        points = labels.numel()

        total_loss += loss.item()
        total_correct += correct
        total_points += points

        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'acc': f"{correct/points:.4f}"
        })

    avg_loss = total_loss / len(dataloader)
    avg_acc = total_correct / total_points

    return avg_loss, avg_acc


@torch.no_grad()
def validate(model, dataloader, criterion, device, epoch):
    """Validate model"""
    model.eval()

    total_loss = 0.0
    metrics_calc = SegmentationMetrics(5, list(TARGET_CLASSES.values()))

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Val]")

    for batch in pbar:
        coords = batch['coords'].to(device)
        features = batch['features'].to(device)
        labels = batch['labels'].to(device)

        # Forward
        logits = model(coords, features)
        loss = criterion(logits.reshape(-1, 5), labels.reshape(-1))

        total_loss += loss.item()

        # Metrics
        predictions = torch.argmax(logits, dim=2).cpu().numpy().flatten()
        targets = labels.cpu().numpy().flatten()

        metrics_calc.update(predictions, targets)

        pbar.set_postfix({'loss': f"{loss.item():.4f}"})

    avg_loss = total_loss / len(dataloader)
    metrics = metrics_calc.get_all_metrics()

    return avg_loss, metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_points', type=int, default=4096)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--save_dir', type=str, default='checkpoints')

    args = parser.parse_args()

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Create datasets
    train_dataset = PreprocessedDataset('data/processed/train_data.npz',
                                       num_points=args.num_points, augment=True)
    val_dataset = PreprocessedDataset('data/processed/val_data.npz',
                                     num_points=args.num_points, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                             shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                           shuffle=False, num_workers=0, pin_memory=True)

    logger.info(f"Train batches: {len(train_loader)}")
    logger.info(f"Val batches: {len(val_loader)}")

    # Model
    model = SimplePointNet(num_classes=5, num_features=train_dataset.features.shape[1]).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {num_params:,}")

    # Loss & Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max',
                                                     factor=0.5, patience=5)

    # Training
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    best_iou = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'val_iou': []}

    logger.info(f"\nStarting training for {args.num_epochs} epochs")
    logger.info("=" * 80)

    for epoch in range(1, args.num_epochs + 1):
        logger.info(f"\nEpoch {epoch}/{args.num_epochs}")

        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")

        # Validate
        val_loss, val_metrics = validate(model, val_loader, criterion, device, epoch)
        logger.info(f"Val Loss: {val_loss:.4f}")
        logger.info(f"Val Accuracy: {val_metrics['overall_accuracy']:.4f}")
        logger.info(f"Val Mean IoU: {val_metrics['mean_iou']:.4f}")
        logger.info(f"Val Kappa: {val_metrics['kappa']:.4f}")

        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_metrics['overall_accuracy'])
        history['val_iou'].append(val_metrics['mean_iou'])

        # LR scheduling
        scheduler.step(val_metrics['mean_iou'])

        # Save best model
        if val_metrics['mean_iou'] > best_iou:
            best_iou = val_metrics['mean_iou']
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': val_metrics,
                'history': history
            }
            torch.save(checkpoint, Path(args.save_dir) / 'best_model.pth')
            logger.info(f"Saved best model (IoU: {best_iou:.4f})")

        # Save checkpoint every 10 epochs
        if epoch % 10 == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': val_metrics,
                'history': history
            }
            torch.save(checkpoint, Path(args.save_dir) / f'checkpoint_epoch_{epoch}.pth')

    # Save final
    checkpoint = {
        'epoch': args.num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': history
    }
    torch.save(checkpoint, Path(args.save_dir) / 'final_model.pth')

    # Save history
    with open(Path(args.save_dir) / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)

    logger.info(f"\nTraining complete!")
    logger.info(f"Best validation IoU: {best_iou:.4f}")


if __name__ == "__main__":
    main()
