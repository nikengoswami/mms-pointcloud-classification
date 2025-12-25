"""
Test script to verify installation and basic functionality
Run this after installing dependencies to ensure everything is set up correctly
"""

import sys
from pathlib import Path

# Set UTF-8 encoding for Windows
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print("=" * 80)
print("TESTING MMS POINT CLOUD CLASSIFICATION INSTALLATION")
print("=" * 80)

# Test 1: Python version
print("\n[1/10] Testing Python version...")
import sys
python_version = sys.version_info
print(f"[OK] Python {python_version.major}.{python_version.minor}.{python_version.micro}")
if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
    print("[WARN] Warning: Python 3.8+ recommended")

# Test 2: Core dependencies
print("\n[2/10] Testing core dependencies...")
try:
    import numpy as np
    print(f"[OK] NumPy {np.__version__}")
except ImportError as e:
    print(f"[FAIL] NumPy not found: {e}")
    sys.exit(1)

try:
    import torch
    print(f"[OK] PyTorch {torch.__version__}")
    if torch.cuda.is_available():
        print(f"  [OK] CUDA available: {torch.cuda.get_device_name(0)}")
    else:
        print(f"  [WARN] CUDA not available (CPU only)")
except ImportError as e:
    print(f"[FAIL] PyTorch not found: {e}")
    sys.exit(1)

# Test 3: Point cloud libraries
print("\n[3/10] Testing point cloud libraries...")
try:
    import laspy
    print(f"[OK] laspy {laspy.__version__}")
except ImportError as e:
    print(f"[FAIL] laspy not found: {e}")
    sys.exit(1)

try:
    import open3d as o3d
    print(f"[OK] Open3D {o3d.__version__}")
except ImportError as e:
    print(f"[WARN] Open3D not found (3D visualization will not work): {e}")

# Test 4: Scientific computing
print("\n[4/10] Testing scientific computing libraries...")
try:
    import pandas as pd
    print(f"[OK] Pandas {pd.__version__}")
except ImportError as e:
    print(f"[FAIL] Pandas not found: {e}")

try:
    import scipy
    print(f"[OK] SciPy {scipy.__version__}")
except ImportError as e:
    print(f"[FAIL] SciPy not found: {e}")

try:
    import sklearn
    print(f"[OK] scikit-learn {sklearn.__version__}")
except ImportError as e:
    print(f"[FAIL] scikit-learn not found: {e}")

# Test 5: Visualization
print("\n[5/10] Testing visualization libraries...")
try:
    import matplotlib
    print(f"[OK] Matplotlib {matplotlib.__version__}")
except ImportError as e:
    print(f"[FAIL] Matplotlib not found: {e}")

try:
    import seaborn as sns
    print(f"[OK] Seaborn {sns.__version__}")
except ImportError as e:
    print(f"[FAIL] Seaborn not found: {e}")

# Test 6: Utilities
print("\n[6/10] Testing utility libraries...")
try:
    import tqdm
    print(f"[OK] tqdm {tqdm.__version__}")
except ImportError as e:
    print(f"[WARN] tqdm not found: {e}")

try:
    import yaml
    print(f"[OK] PyYAML available")
except ImportError as e:
    print(f"[WARN] PyYAML not found: {e}")

# Test 7: Project modules
print("\n[7/10] Testing project modules...")
try:
    from models.randlanet import RandLANet
    print("[OK] RandLA-Net model")
except ImportError as e:
    print(f"[FAIL] RandLA-Net import failed: {e}")
    sys.exit(1)

try:
    from utils.las_io import LASProcessor
    print("[OK] LAS I/O utilities")
except ImportError as e:
    print(f"[FAIL] LAS I/O import failed: {e}")
    sys.exit(1)

try:
    from evaluation.metrics import SegmentationMetrics
    print("[OK] Evaluation metrics")
except ImportError as e:
    print(f"[FAIL] Evaluation metrics import failed: {e}")
    sys.exit(1)

# Test 8: Model creation
print("\n[8/10] Testing model creation...")
try:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RandLANet(num_classes=5, num_features=6)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"[OK] Model created successfully ({num_params:,} parameters)")
    print(f"  Device: {device}")
except Exception as e:
    print(f"[FAIL] Model creation failed: {e}")
    sys.exit(1)

# Test 9: Data loading (if Classified.las exists)
print("\n[9/10] Testing data loading...")
try:
    if Path("Classified.las").exists():
        processor = LASProcessor("Classified.las")
        features = processor.read_las()
        print(f"[OK] LAS file loaded: {features['num_points']:,} points")

        # Get statistics
        stats = processor.get_statistics()
        print(f"  Bounding box: [{stats['xyz_min'][0]:.1f}, {stats['xyz_max'][0]:.1f}] x "
              f"[{stats['xyz_min'][1]:.1f}, {stats['xyz_max'][1]:.1f}] x "
              f"[{stats['xyz_min'][2]:.1f}, {stats['xyz_max'][2]:.1f}]")
    else:
        print("[WARN] Classified.las not found (data loading test skipped)")
except Exception as e:
    print(f"[WARN] Data loading failed: {e}")

# Test 10: Forward pass
print("\n[10/10] Testing model forward pass...")
try:
    # Create dummy data
    batch_size = 2
    num_points = 1024
    num_features = 6

    coords = torch.randn(batch_size, num_points, 3).to(device)
    features = torch.randn(batch_size, num_points, num_features).to(device)

    model = model.to(device)
    model.eval()

    with torch.no_grad():
        output = model(coords, features)

    print(f"[OK] Forward pass successful")
    print(f"  Input: {coords.shape} coords, {features.shape} features")
    print(f"  Output: {output.shape} logits")

except Exception as e:
    print(f"[FAIL] Forward pass failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Summary
print("\n" + "=" * 80)
print("INSTALLATION TEST COMPLETE")
print("=" * 80)
print("\n[OK] All critical components are working!")
print("\nYou can now:")
print("  1. Analyze your data:  python analyze_data.py")
print("  2. Train the model:    python train.py --num_epochs 50")
print("  3. Run inference:      python inference.py --input <file> --output <file>")
print("  4. Visualize results:  python visualize.py --input <file> --mode 2d")
print("\nSee README.md and QUICKSTART.md for detailed instructions.")
print("=" * 80)
