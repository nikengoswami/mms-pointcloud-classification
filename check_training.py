"""
Quick script to check PointNet++ training progress
"""

import time
from pathlib import Path

log_file = Path("pointnet2_training.log")

if not log_file.exists():
    print("Training log not found!")
    exit(1)

# Read last 100 lines
with open(log_file, 'r') as f:
    lines = f.readlines()
    last_lines = lines[-100:]

# Extract key information
print("="*80)
print("POINTNET++ TRAINING PROGRESS")
print("="*80)

for line in last_lines:
    if any(keyword in line for keyword in ["Epoch", "Train Loss", "Val Loss", "Val Accuracy", "Val Mean IoU", "Val Kappa", "Saved best"]):
        print(line.strip())

print("="*80)
print("\nTo monitor in real-time:")
print("  tail -f pointnet2_training.log")
