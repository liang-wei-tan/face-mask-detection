import tensorflow as tf
from models import create_baseline_cnn
from train.utils import load_datasets, optimize_datasets

# Configuration
EPOCHS = 10


def train_model(model, train_ds, val_ds, callbacks, epochs=EPOCHS):
    """
    Train the model.

    Args:
        model: Keras model to train
        train_ds: Training dataset
        val_ds: Validation dataset
        epochs: Number of training epochs

    Returns:
        Training history object
    """
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
    )
    return history


def evaluate_model(model, test_ds):
    """
    Evaluate model on test dataset.

    Args:
        model: Keras model to evaluate
        test_ds: Test dataset

    Returns:
        Tuple of (loss, accuracy)
    """
    loss, accuracy = model.evaluate(test_ds)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")
    return loss, accuracy
