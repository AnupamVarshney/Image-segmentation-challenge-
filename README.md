# Image Segmentation Challenge — Enhanced U-Net (Simple Pipeline)

This repository provides a clean, reliable pipeline for weakly supervised image segmentation using an enhanced U-Net. The main entry point is `unet_simple.py`, which supports a quick 2‑image overfit sanity check and a full class‑balanced training run.

## Features

- **Enhanced U-Net** with a clear, minimal architecture
- **Weakly supervised** training from scribbles (no pretrained models)
- **Two modes** in one script:
  - `OVERFIT_TWO`: prove learning on two images
  - `FULL_CLASS_BALANCED`: full training with weighted BCE + Dice + extras
- **TTA + simple post‑processing** at inference

## Project structure

```
├── unet_simple.py          # Main training/inference script (use this)
├── util.py                 # Dataset I/O, storage, visualization utilities
├── dataset/
│   ├── train/              # Training set
│   └── test1/              # Test set
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

Other scripts (e.g., `unet.py`, `challenge.py`) are legacy/baseline references.

## Setup

### Option A: Python venv

```bash
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Option B: Conda

```bash
conda create -y -n seg_env python=3.9
conda activate seg_env
pip install -r requirements.txt
```

Notes:
- Python 3.8+ recommended (tested with TF ≥ 2.8).
- GPU support: install a TensorFlow build compatible with your CUDA/cuDNN stack if needed.

## Dataset layout

The script expects the following layout and filename conventions:

```
dataset/
  train/
    images/        # RGB images, filenames like XYZ.jpg
    scribbles/     # scribble masks, filenames XYZ.png (0=bg, 1=fg, 255=unlabeled)
    ground_truth/  # GT masks, filenames XYZ.png (0/1), used for training and metrics
  test1/
    images/        # RGB images XYZ.jpg
    scribbles/     # scribbles XYZ.png
```

- Filenames must match by basename across folders (e.g., `123.jpg` ↔ `123.png`).
- Paths can be changed by editing the top of `unet_simple.py`.

## Quick start

```bash
# Activate your environment first (venv or conda)
python unet_simple.py
```

By default, the script runs with `FULL_CLASS_BALANCED=True` and `OVERFIT_TWO=False`. Toggle these at the top of `unet_simple.py`:

```python
OVERFIT_TWO = False          # set True to quickly overfit two samples
FULL_CLASS_BALANCED = True   # set False to skip the full training
```

### What the script does

- Loads data from `dataset/train` and `dataset/test1`
- Trains an enhanced U-Net (with class‑balanced loss)
- Applies TTA + optional DenseCRF refinement
- Saves predictions to:
  - `dataset/train/unet_balanced/`
  - `dataset/test1/unet_balanced/`
- If `OVERFIT_TWO=True`, also writes `dataset/train/unet_overfit_two/`

## Configuration highlights

- Image sizes: `TRAIN_SIZE=(384, 512)`, final `TARGET_SIZE=(375, 500)`
- Loss: weighted BCE + Dice (+ small focal and edge terms)
- Metrics: IoU
- Simple augmentation: horizontal flip
- TTA: horizontal/vertical flips + multi‑scale at inference

Adjust hyperparameters near the top of `unet_simple.py`:

```python
BATCH_FULL  = 6
EPOCHS_FULL = 150
LR_FULL     = 3e-4
THRESH_FULL = 0.30
```

## Requirements

Install from `requirements.txt`:

```text
tensorflow>=2.8.0
numpy
opencv-python
scikit-image
scikit-learn
matplotlib
Pillow
```

For GPU TensorFlow, install the version compatible with your CUDA/cuDNN.

## Tips

- If filenames/extensions differ, edit `load_dataset` in `util.py` to match your naming.
- Large runs: use `conda` with a CUDA‑enabled TensorFlow build.
- To visualize results, see helpers in `util.py` (e.g., `visualize`).

## Clone

```bash
git clone https://github.com/AnupamVarshney/Image-segmentation-challenge-.git
cd Image-segmentation-challenge-
```
