# Mask Detection

A computer vision project for detecting face masks in images using deep learning.

## Overview

This repository contains code and models for the face mask detection challenge. The project uses object detection techniques to identify whether people are wearing masks, not wearing masks, or wearing masks incorrectly.

## Datasets

This project leverages two Kaggle datasets for training and evaluation:

1. **Face Mask Detection Dataset**
   - Source: https://www.kaggle.com/datasets/andrewmvd/face-mask-detection
   - Format: Pascal VOC XML annotations
   - Classes: 3 (with_mask, without_mask, mask_weared_incorrect)

2. **Face Mask 12K Images Dataset**
   - Source: https://www.kaggle.com/datasets/ashishjangra27/face-mask-12k-images-dataset
   - Provides additional labeled images for improved model training

## Project Structure

```
mask_detection/
├── README.md              # This file
├── requirements.txt       # Python dependencies
├── Dockerfile             # Docker configuration for RunPod
├── CLAUDE.md             # Project guidelines
├── .gitignore            # Git ignore rules
├── models/               # Model definitions
│   ├── __init__.py
│   └── baseline_cnn.py
├── train/                # Training pipeline
│   ├── __init__.py
│   ├── __main__.py       # CLI entry point
│   ├── utils.py          # Data loading utilities
│   └── trainer.py        # Training functions
├── checkpoints/          # Model checkpoints (generated during training, gitignored)
├── venv/                 # Virtual environment (ignored)
└── face_mask_detection/  # Dataset directory (ignored)
    ├── images/           # Training/validation images
    └── annotations/      # Pascal VOC XML annotations
```

## Setup

### Prerequisites

- Python 3.9.6
- pip

### Installation

1. Clone the repository:
```bash
git clone https://github.com/liang-wei-tan/face-mask-detection
cd face-mask-detection
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### GPU Setup (Optional - for CUDA/NVIDIA GPUs)

If you have an NVIDIA GPU and want TensorFlow to use it:

```bash
pip uninstall tensorflow -y
pip install tensorflow[and-cuda]
```

Verify GPU is detected:
```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

You should see output like:
```
[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

This installs TensorFlow with compatible CUDA/cuDNN libraries for automatic GPU acceleration.

### Docker Setup (Optional - for containerized deployment)

A Dockerfile is provided for containerized deployment (e.g., on RunPod).

**Building the Docker image for RunPod (linux/amd64):**

```bash
# For Apple Silicon/Intel Macs - build for linux/amd64 architecture and push to Docker Hub
docker buildx build --platform linux/amd64 -t tanliangwei/face-mask-detection:latest --push .
```

**Prerequisites for building:**
- Docker Desktop installed
- Logged in to Docker Hub: `docker login`
- `docker buildx` available (usually built-in; if not, run `docker buildx create --use`)

**Note:** Building for `linux/amd64` on Mac takes longer (5-10 minutes) as it runs on a remote builder, but ensures compatibility with RunPod servers.

**Using on RunPod:**
- Use image: `docker.io/tanliangwei/face-mask-detection:latest`
- Ports: 8888 (JupyterLab)
- GPU support: Automatic (CUDA/cuDNN included in base image)

4. Download dataset
```bash
curl -L -o face-mask-12k-images-dataset.zip\
  https://www.kaggle.com/api/v1/datasets/download/ashishjangra27/face-mask-12k-images-dataset

unzip face-mask-12k-images-dataset.zip -d face-mask-12k-images-dataset/
```


## Model Training & Evaluation

### Training the Model

**From command line:**

```bash
# Activate virtual environment
source venv/bin/activate

# Run training with default settings (10 epochs)
python -m train "face-mask-12k-images-dataset/Face Mask Dataset"

# Run training with custom epochs
python -m train "face-mask-12k-images-dataset/Face Mask Dataset" 20
```

**From Jupyter notebook:**

```python
from models import create_baseline_cnn
from train.utils import load_datasets, optimize_datasets
from train.trainer import train_model, evaluate_model

# Load datasets
data_dir = "face-mask-12k-images-dataset/Face Mask Dataset"
train_ds, val_ds, test_ds, class_names = load_datasets(data_dir)

# Optimize for training
train_ds, val_ds, test_ds = optimize_datasets(train_ds, val_ds, test_ds)

# Create model and checkpoint callback
model, checkpoint_callback = create_baseline_cnn(num_classes=len(class_names))

# Train
history = train_model(model, train_ds, val_ds, callbacks=[checkpoint_callback], epochs=10)

# Evaluate
loss, accuracy = evaluate_model(model, test_ds)
```

### Checkpoints

Model checkpoints are automatically saved during training to:
```
checkpoints/baseline_cnn/model-{epoch:02d}-{val_loss:.4f}.h5
```

Example:
```
checkpoints/baseline_cnn/
├── model-01-0.5234.h5
├── model-02-0.4821.h5
├── model-03-0.4612.h5
└── ...
```

Each file is saved after every epoch, named with the epoch number and validation loss.

**Loading a checkpoint:**

```python
import tensorflow as tf

model = tf.keras.models.load_model("checkpoints/baseline_cnn/model-05-0.4612.h5")
```

## Classes

- **with_mask**: Person wearing a mask correctly
- **without_mask**: Person not wearing a mask
- **mask_weared_incorrect**: Person wearing a mask incorrectly

## Notes

- Model weights (`.pt`, `.h5`, `.onnx`) are gitignored
- Checkpoint files in `checkpoints/` are gitignored
- The `face_mask_detection/` directory containing raw data is gitignored
- Python virtual environment is gitignored
- Docker image must be built with `--platform linux/amd64` for RunPod compatibility (not just `docker build` on Mac)

## License

(Add license information as needed)
