# Quick Start Guide

## Complete Workflow for MMS Point Cloud Classification

### Step 1: Install Dependencies (5 minutes)

```bash
# Install basic dependencies
pip install laspy numpy pandas scikit-learn scipy matplotlib seaborn tqdm pyyaml h5py

# Install PyTorch (choose based on your system)
# For CUDA 11.8:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CPU only:
pip install torch torchvision

# Install Open3D for 3D visualization
pip install open3d
```

### Step 2: Analyze Your Data (2 minutes)

```bash
python analyze_data.py
```

**Expected Output:**
- Point cloud statistics
- Feature information (XYZ, RGB, Intensity, etc.)
- Class distribution (if already classified)
- Bounding box dimensions
- Visualizations saved to `results/`

### Step 3: Prepare Training Data (Manual - Variable Time)

#### Option A: Use Existing Classified.las (Demo)

If you want to test the pipeline quickly, the `Classified.las` file can be used directly.

#### Option B: Label Your Own Data (Recommended)

1. **Open CloudCompare**
   ```bash
   # Open your LAS file in CloudCompare
   ```

2. **Select Regions & Classify**
   - Use "Segment" or "Scissors" tool to select points
   - Right-click → Edit → Scalar Fields → Classification
   - Assign class values:
     - Road: 0 or 11
     - Snow: 1
     - Vehicle: 2
     - Vegetation: 3, 4, or 5
     - Others: 6 or any other value

3. **Save Labeled File**
   - Save as LAS format
   - Preserve classification field

4. **Split Data**
   ```bash
   # Manually split or use a script to create:
   data/labeled/train.las  (70% of data)
   data/labeled/val.las    (15% of data)
   data/labeled/test.las   (15% of data)
   ```

### Step 4: Train the Model (1-2 hours on GPU)

```bash
# Basic training
python train.py --num_epochs 50

# Advanced training with custom parameters
python train.py \
    --data_dir data/labeled \
    --batch_size 4 \
    --num_points 4096 \
    --num_epochs 100 \
    --lr 0.001 \
    --save_dir checkpoints
```

**Monitor Training:**
- Watch console for loss and accuracy
- Training curves saved to `checkpoints/training_history.png`
- Best model saved to `checkpoints/best_model.pth`

**Expected Results After 50 Epochs:**
- Training Accuracy: 80-90%
- Validation IoU: 60-75%
- Training time: 5-10 min/epoch (GPU) or 30-60 min/epoch (CPU)

### Step 5: Run Inference (5-10 minutes)

```bash
# Classify a new point cloud
python inference.py \
    --input Classified.las \
    --output results/classified_output.las \
    --model checkpoints/best_model.pth
```

**For large files (>10M points):**
```bash
python inference.py \
    --input large_file.las \
    --output results/classified_large.las \
    --model checkpoints/best_model.pth \
    --spatial \
    --block_size 50.0
```

### Step 6: Evaluate Results (2 minutes)

Create an evaluation script:

```python
# evaluate_model.py
from utils.las_io import LASProcessor
from evaluation.metrics import SegmentationMetrics
import numpy as np

# Load ground truth
gt_processor = LASProcessor("data/labeled/test.las")
gt_features = gt_processor.read_las()
gt_labels = gt_features['classification']

# Load predictions
pred_processor = LASProcessor("results/classified_output.las")
pred_features = pred_processor.read_las()
pred_labels = pred_features['classification']

# Compute metrics
metrics = SegmentationMetrics(
    num_classes=5,
    class_names=['Road', 'Snow', 'Vehicle', 'Vegetation', 'Others']
)
metrics.update(pred_labels, gt_labels)

# Print and save results
metrics.print_metrics()
metrics.plot_confusion_matrix(save_path='results/confusion_matrix.png', normalize=True)
metrics.plot_per_class_metrics(save_path='results/per_class_metrics.png')
```

Run evaluation:
```bash
python evaluate_model.py
```

### Step 7: Visualize Results (1 minute)

```bash
# 2D visualization
python visualize.py \
    --input results/classified_output.las \
    --mode 2d \
    --save results/visualization_2d.png

# 3D interactive viewer
python visualize.py \
    --input results/classified_output.las \
    --mode 3d \
    --point_size 2.0

# Compare with ground truth
python visualize.py \
    --input results/classified_output.las \
    --compare data/labeled/test.las \
    --mode 2d \
    --save results/comparison.png
```

## Complete Example Workflow

```bash
# 1. Install (one-time setup)
pip install -r requirements.txt

# 2. Analyze data
python analyze_data.py

# 3. Train model
python train.py --num_epochs 50 --batch_size 4

# 4. Run inference
python inference.py \
    --input Classified.las \
    --output results/classified_output.las \
    --model checkpoints/best_model.pth

# 5. Visualize
python visualize.py --input results/classified_output.las --mode 2d
```

## Troubleshooting

### GPU Not Detected

```bash
# Check CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Out of Memory

```bash
# Reduce batch size and num_points
python train.py --batch_size 2 --num_points 2048
```

### Import Errors

```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

## Expected File Sizes

- **Model checkpoint**: ~20-50 MB
- **Training history**: ~10 KB
- **Classified LAS output**: Similar to input size
- **Visualizations**: 1-5 MB per image

## Performance Benchmarks

| Operation | GPU (RTX 3090) | CPU (Intel i7) |
|-----------|----------------|----------------|
| Training (per epoch) | 5-10 min | 30-60 min |
| Inference (1M points) | 10-20 sec | 2-5 min |
| Visualization 2D | 5-10 sec | 10-20 sec |
| Visualization 3D | Instant | Instant |

## Next Steps

1. **Improve Model**
   - Label more training data
   - Balance class distribution
   - Tune hyperparameters
   - Experiment with augmentation

2. **Deploy Model**
   - Batch process multiple files
   - Integrate with existing pipeline
   - Create automated workflow

3. **Advanced Features**
   - Add more classes
   - Multi-task learning
   - Temporal analysis (if multi-temporal data)
   - Integration with dashcam video

## Support

For issues or questions:
1. Check the main [README.md](README.md)
2. Review error messages carefully
3. Verify data format and paths
4. Test with small subset first

---

**Happy Classifying!** 🚀
