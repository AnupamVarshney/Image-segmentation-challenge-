# Image Segmentation Challenge - Enhanced U-Net

This repository contains an enhanced U-Net implementation for weakly supervised image segmentation using sparse scribbles.

## 🚀 Features

- **Enhanced U-Net Architecture** with:
  - Residual connections
  - Squeeze-and-Excitation blocks
  - Attention gates for skip connections
  - ASPP (Atrous Spatial Pyramid Pooling) bottleneck
  - Deep supervision

- **Weakly Supervised Training**:
  - Scribble-only training (no pre-trained models)
  - Patch-based training for sparse supervision
  - Masked, class-balanced BCE + Dice loss
  - Advanced data augmentation

- **Multiple Training Modes**:
  - Direct scribble supervision
  - Random Walker pseudo-labels (optional)

## 📁 Project Structure

```
├── unet.py              # Main training script
├── util.py              # Utility functions (data loading, visualization)
├── dataset/             # Dataset directory
│   ├── train/          # Training data
│   └── test1/          # Test data
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended)
- Git

### Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/YOUR_USERNAME/image-segmentation-challenge.git
   cd image-segmentation-challenge
   ```

2. **Create virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## 🎯 Usage

### Training

```bash
# Activate virtual environment
source venv/bin/activate

# Run training
python unet.py
```

### Configuration

Edit the configuration section in `unet.py`:

```python
TARGET_SIZE = (375, 500)  # Final output size
PATCH = 256              # Training patch size
BATCH = 4                # Batch size
EPOCHS = 120             # Maximum epochs
USE_PSEUDO = False       # Use Random Walker pseudo-labels
```

## 📊 Model Architecture

The enhanced U-Net includes:

- **Encoder:** 5 levels with residual blocks and SE attention
- **Bottleneck:** ASPP with multiple dilated convolutions
- **Decoder:** Attention gates and skip connections
- **Output:** Single segmentation mask

## 🔧 Training Details

- **Loss Function:** Masked BCE + Dice loss
- **Optimizer:** Adam with learning rate scheduling
- **Metrics:** Masked IoU
- **Augmentation:** Flips, brightness, contrast
- **Early Stopping:** Based on validation IoU

## 📈 Results

The enhanced architecture provides:
- Better boundary detection
- Improved feature learning
- More stable training
- Higher segmentation accuracy

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📝 License

This project is for educational and research purposes.

## 🙏 Acknowledgments

- U-Net architecture (Ronneberger et al.)
- Attention mechanisms in medical image segmentation
- Weakly supervised learning techniques
