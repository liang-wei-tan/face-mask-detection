import os
import tensorflow as tf

INPUT_DIM = (None, 128, 128, 3)
CHECKPOINT_DIR = "checkpoints/baseline_cnn"


def _get_checkpoint_callback():
    """Create and return a checkpoint callback."""
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    checkpoint_path = os.path.join(CHECKPOINT_DIR, "model-{epoch:02d}-{val_loss:.4f}.h5")
    return tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        save_freq="epoch",
        monitor="val_loss",
        mode="min",
        verbose=1,
    )


def create_baseline_cnn(num_classes=2):
    """
    Creates a baseline CNN model for mask detection.

    Args:
        num_classes: Number of output classes (default: 2 for with_mask/without_mask)

    Returns:
        Tuple of (model, checkpoint_callback)
    """
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Rescaling(1.0 / 255),
            tf.keras.layers.Conv2D(64, 3, activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(64, 3, activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(64, 3, activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dense(num_classes),
        ]
    )

    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    model.build(INPUT_DIM)

    return model, _get_checkpoint_callback()
