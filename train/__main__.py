"""
CLI entry point for training. Run with: python -m train [data_dir] [epochs]
"""
import sys
from models import create_baseline_cnn
from train.utils import load_datasets, optimize_datasets
from train.trainer import train_model

# Configuration
EPOCHS = 10


def main(data_dir, epochs=EPOCHS):
    """
    Main training pipeline.

    Args:
        data_dir: Root directory containing Train/, Validation/, Test/ subdirectories
        epochs: Number of training epochs
    """
    print("Loading datasets...")
    train_ds, val_ds, test_ds, class_names = load_datasets(data_dir)
    print(f"Classes: {class_names}")

    print("Optimizing datasets...")
    train_ds, val_ds, test_ds = optimize_datasets(train_ds, val_ds, test_ds)

    print("Creating model...")
    model = create_baseline_cnn(num_classes=len(class_names))
    model.summary()

    print("Training model...")
    history = train_model(model, train_ds, val_ds, epochs=epochs)

    return model, history


if __name__ == "__main__":
    epochs = EPOCHS
    data_dir = "face-mask-12k-images-dataset/Face Mask Dataset"

    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    if len(sys.argv) > 2:
        epochs = int(sys.argv[2])

    model, history = main(data_dir, epochs=epochs)
