"""
CLI entry point for training.

Usage:
    python -m train [model_name] [data_dir] [epochs]

Examples:
    python -m train baseline_cnn
    python -m train baseline_cnn_64 "face-mask-12k-images-dataset/Face Mask Dataset"
    python -m train baseline_cnn_64 "face-mask-12k-images-dataset/Face Mask Dataset" 20
"""

import sys
from models import get_model
from .utils import load_datasets, optimize_datasets
from .trainer import train_model
from .evaluate import evaluate_model

# Configuration
EPOCHS = 20
DEFAULT_MODEL = "baseline_cnn"
DEFAULT_DATA_DIR = "face-mask-12k-images-dataset/Face Mask Dataset"


def main(model_name, data_dir, epochs=EPOCHS):
    """
    Main training pipeline.

    Args:
        model_name: Name of the model to train (e.g., 'baseline_cnn', 'baseline_cnn_64')
        data_dir: Root directory containing Train/, Validation/, Test/ subdirectories
        epochs: Number of training epochs
    """
    print(f"Loading model: {model_name}")
    create_model = get_model(model_name)

    print("Loading datasets...")
    train_ds, val_ds, test_ds, class_names = load_datasets(data_dir)
    print(f"Classes: {class_names}")

    print("Optimizing datasets...")
    train_ds, val_ds, test_ds = optimize_datasets(train_ds, val_ds, test_ds)

    print("Creating model...")
    model, checkpoint_callback = create_model(num_classes=len(class_names))
    model.summary()

    print("Training model...")
    history = train_model(
        model, train_ds, val_ds, callbacks=[checkpoint_callback], epochs=epochs
    )

    print("Evaluating model on test set...")
    evaluate_model(model, test_ds, class_names=class_names)

    return model, history


if __name__ == "__main__":
    model_name = DEFAULT_MODEL
    data_dir = DEFAULT_DATA_DIR
    epochs = EPOCHS

    if len(sys.argv) > 1:
        model_name = sys.argv[1]
    if len(sys.argv) > 2:
        data_dir = sys.argv[2]
    if len(sys.argv) > 3:
        epochs = int(sys.argv[3])

    model, history = main(model_name, data_dir, epochs=epochs)
