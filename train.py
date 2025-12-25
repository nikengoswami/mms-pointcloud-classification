"""
Training script for RandLA-Net point cloud classification
Trains model on labeled MMS point cloud data
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import argparse
import json
import logging
from tqdm import tqdm
from datetime import datetime

from models.randlanet import RandLANet
from models.dataset import PointCloudDataset, SpatialDataset, collate_fn, create_dataloaders
from evaluation.metrics import SegmentationMetrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Trainer:
    """Training manager for RandLA-Net"""

    def __init__(self, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                 num_classes: int, class_names: list, device: torch.device,
                 learning_rate: float = 0.001, weight_decay: float = 1e-4,
                 class_weights: torch.Tensor = None):
        """
        Args:
            model: RandLA-Net model
            train_loader: Training data loader
            val_loader: Validation data loader
            num_classes: Number of classes
            class_names: List of class names
            device: Device to train on
            learning_rate: Initial learning rate
            weight_decay: Weight decay for optimizer
            class_weights: Optional class weights for loss function
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_classes = num_classes
        self.class_names = class_names
        self.device = device

        # Loss function
        if class_weights is not None:
            class_weights = class_weights.to(device)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-1)

        # Optimizer
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=5,
            verbose=True
        )

        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'val_iou': []
        }

        # Best model tracking
        self.best_val_iou = 0.0
        self.best_epoch = 0

    def train_epoch(self, epoch: int) -> tuple:
        """
        Train for one epoch

        Args:
            epoch: Current epoch number

        Returns:
            Tuple of (average_loss, average_accuracy)
        """
        self.model.train()

        total_loss = 0.0
        total_correct = 0
        total_points = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]")

        for batch_idx, batch in enumerate(pbar):
            coords = batch['coords'].to(self.device)
            features = batch['features'].to(self.device)
            labels = batch['labels'].to(self.device)
            mask = batch['mask'].to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            logits = self.model(coords, features)  # (B, N, num_classes)

            # Compute loss (only on valid points)
            # Reshape for loss computation
            logits_flat = logits.view(-1, self.num_classes)
            labels_flat = labels.view(-1)
            mask_flat = mask.view(-1)

            # Apply mask
            logits_masked = logits_flat[mask_flat]
            labels_masked = labels_flat[mask_flat]

            loss = self.criterion(logits_masked, labels_masked)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Compute accuracy
            predictions = torch.argmax(logits_masked, dim=1)
            correct = (predictions == labels_masked).sum().item()
            num_points = len(labels_masked)

            total_loss += loss.item()
            total_correct += correct
            total_points += num_points

            # Update progress bar
            current_acc = correct / num_points if num_points > 0 else 0
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{current_acc:.4f}"
            })

        avg_loss = total_loss / len(self.train_loader)
        avg_acc = total_correct / total_points if total_points > 0 else 0

        return avg_loss, avg_acc

    @torch.no_grad()
    def validate(self, epoch: int) -> tuple:
        """
        Validate model

        Args:
            epoch: Current epoch number

        Returns:
            Tuple of (average_loss, metrics_dict)
        """
        self.model.eval()

        total_loss = 0.0
        metrics_calculator = SegmentationMetrics(self.num_classes, self.class_names)

        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch} [Val]")

        for batch in pbar:
            coords = batch['coords'].to(self.device)
            features = batch['features'].to(self.device)
            labels = batch['labels'].to(self.device)
            mask = batch['mask'].to(self.device)

            # Forward pass
            logits = self.model(coords, features)

            # Compute loss
            logits_flat = logits.view(-1, self.num_classes)
            labels_flat = labels.view(-1)
            mask_flat = mask.view(-1)

            logits_masked = logits_flat[mask_flat]
            labels_masked = labels_flat[mask_flat]

            loss = self.criterion(logits_masked, labels_masked)
            total_loss += loss.item()

            # Compute metrics
            predictions = torch.argmax(logits_masked, dim=1).cpu().numpy()
            targets = labels_masked.cpu().numpy()

            metrics_calculator.update(predictions, targets)

            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        avg_loss = total_loss / len(self.val_loader)
        metrics = metrics_calculator.get_all_metrics()

        return avg_loss, metrics

    def train(self, num_epochs: int, save_dir: str = "checkpoints"):
        """
        Train the model

        Args:
            num_epochs: Number of epochs to train
            save_dir: Directory to save checkpoints
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Starting training for {num_epochs} epochs")
        logger.info(f"Device: {self.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        for epoch in range(1, num_epochs + 1):
            logger.info(f"\nEpoch {epoch}/{num_epochs}")
            logger.info("-" * 50)

            # Train
            train_loss, train_acc = self.train_epoch(epoch)
            logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")

            # Validate
            val_loss, val_metrics = self.validate(epoch)
            logger.info(f"Val Loss: {val_loss:.4f}")
            logger.info(f"Val Accuracy: {val_metrics['overall_accuracy']:.4f}")
            logger.info(f"Val Mean IoU: {val_metrics['mean_iou']:.4f}")
            logger.info(f"Val Kappa: {val_metrics['kappa']:.4f}")

            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_metrics['overall_accuracy'])
            self.history['val_iou'].append(val_metrics['mean_iou'])

            # Learning rate scheduling
            self.scheduler.step(val_metrics['mean_iou'])

            # Save best model
            if val_metrics['mean_iou'] > self.best_val_iou:
                self.best_val_iou = val_metrics['mean_iou']
                self.best_epoch = epoch

                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'metrics': val_metrics,
                    'history': self.history
                }

                torch.save(checkpoint, save_path / 'best_model.pth')
                logger.info(f"Saved best model (IoU: {self.best_val_iou:.4f})")

            # Save checkpoint every 10 epochs
            if epoch % 10 == 0:
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'metrics': val_metrics,
                    'history': self.history
                }
                torch.save(checkpoint, save_path / f'checkpoint_epoch_{epoch}.pth')

        logger.info(f"\nTraining complete!")
        logger.info(f"Best validation IoU: {self.best_val_iou:.4f} at epoch {self.best_epoch}")

        # Save final model
        checkpoint = {
            'epoch': num_epochs,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history
        }
        torch.save(checkpoint, save_path / 'final_model.pth')

        # Save training history
        with open(save_path / 'training_history.json', 'w') as f:
            json.dump(self.history, f, indent=2)

    def plot_training_history(self, save_path: str = None):
        """Plot training history"""
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Loss
        axes[0].plot(self.history['train_loss'], label='Train')
        axes[0].plot(self.history['val_loss'], label='Validation')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Accuracy
        axes[1].plot(self.history['train_acc'], label='Train')
        axes[1].plot(self.history['val_acc'], label='Validation')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Training and Validation Accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # IoU
        axes[2].plot(self.history['val_iou'])
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Mean IoU')
        axes[2].set_title('Validation Mean IoU')
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Training history plot saved to: {save_path}")

        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Train RandLA-Net for point cloud classification')
    parser.add_argument('--data_dir', type=str, default='data/labeled',
                       help='Directory containing labeled LAS files')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size')
    parser.add_argument('--num_points', type=int, default=4096,
                       help='Number of points per sample')
    parser.add_argument('--num_epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                       help='Directory to save checkpoints')

    args = parser.parse_args()

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Class configuration
    num_classes = 5
    class_names = ['Road', 'Snow', 'Vehicle', 'Vegetation', 'Others']

    # Class mapping from LAS standard classes to our classes
    class_mapping = {
        0: 4,   # Never classified -> Others
        1: 4,   # Unclassified -> Others
        2: 0,   # Ground -> Road
        3: 3,   # Low Vegetation -> Vegetation
        4: 3,   # Medium Vegetation -> Vegetation
        5: 3,   # High Vegetation -> Vegetation
        6: 4,   # Building -> Others
        7: 4,   # Low Point (noise) -> Others
        9: 4,   # Water -> Others
        10: 4,  # Rail -> Others
        11: 0,  # Road Surface -> Road
        17: 0,  # Bridge Deck -> Road
    }

    # TODO: Split data into train and validation
    # For now, using the same file for both (you should split this!)
    train_files = [str(Path(args.data_dir) / "train.las")]
    val_files = [str(Path(args.data_dir) / "val.las")]

    # If split files don't exist, use Classified.las for demo
    if not Path(train_files[0]).exists():
        logger.warning("Training files not found. Using Classified.las for demonstration.")
        train_files = ["Classified.las"]
        val_files = ["Classified.las"]
        logger.warning("WARNING: Using same file for train and validation - split your data!")

    # Create data loaders
    train_loader, val_loader = create_dataloaders(
        train_files=train_files,
        val_files=val_files,
        batch_size=args.batch_size,
        num_points=args.num_points,
        num_workers=args.num_workers,
        class_mapping=class_mapping
    )

    logger.info(f"Train batches: {len(train_loader)}")
    logger.info(f"Val batches: {len(val_loader)}")

    # Create model
    # Determine number of features (XYZ + RGB + intensity if available)
    sample_batch = next(iter(train_loader))
    num_features = sample_batch['features'].shape[2]
    logger.info(f"Number of features: {num_features}")

    model = RandLANet(
        num_classes=num_classes,
        num_features=num_features,
        num_neighbors=16,
        decimation=4
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_classes=num_classes,
        class_names=class_names,
        device=device,
        learning_rate=args.lr
    )

    # Train
    trainer.train(num_epochs=args.num_epochs, save_dir=args.save_dir)

    # Plot training history
    trainer.plot_training_history(save_path=Path(args.save_dir) / 'training_history.png')


if __name__ == "__main__":
    main()
