import tensorflow as tf

# Configuration
EPOCHS = 20


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