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
├── CLAUDE.md             # Project guidelines
├── .gitignore            # Git ignore rules
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

4. Download dataset
```bash
curl -L -o face-mask-detection.zip\
  https://www.kaggle.com/api/v1/datasets/download/andrewmvd/face-mask-detection

unzip face-mask-12k-images-dataset.zip -d face-mask-12k-images-dataset/
```


## Model Training & Evaluation

(Add specific instructions for training and evaluation once implemented)

## Classes

- **with_mask**: Person wearing a mask correctly
- **without_mask**: Person not wearing a mask
- **mask_weared_incorrect**: Person wearing a mask incorrectly

## Notes

- Model weights (`.pt`, `.h5`, `.onnx`) are gitignored
- The `face_mask_detection/` directory containing raw data is gitignored
- Python virtual environment is gitignored

## License

(Add license information as needed)
