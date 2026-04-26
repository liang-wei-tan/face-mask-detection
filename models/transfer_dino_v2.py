import os
import tensorflow as tf
import keras_hub
import keras

img_height = 128
img_width = 128
INPUT_DIM = (None, img_height, img_width, 3)
IMG_SHAPE = (img_height, img_width) + (3,)
CHECKPOINT_DIR = "checkpoints/transfer_dino_v2"


def _get_checkpoint_callback():
    """Create and return a checkpoint callback."""
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    checkpoint_path = os.path.join(CHECKPOINT_DIR, "model-{epoch:02d}-{val_loss:.4f}.keras")
    return tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        save_freq="epoch",
        monitor="val_loss",
        mode="min",
        verbose=1,
    )


def create_transfer_dino_v2(num_classes=2):
    """
    Creates a transfer learning model using DINOv2 for mask detection.

    Args:
        num_classes: Number of output classes (default: 2 for with_mask/without_mask)

    Returns:
        Tuple of (model, checkpoint_callback)
    """
    
    backbone = keras_hub.models.DINOV2Backbone.from_preset(
        "dinov2_base", image_shape=IMG_SHAPE
    )
    backbone_model = keras.Model(
        inputs=backbone.inputs,
        outputs=backbone.pyramid_outputs,
    )
    backbone_model.trainable = False
    preprocessor = keras_hub.layers.DINOV2ImageConverter.from_preset("dinov2_base", image_size=(img_height, img_width), interpolation="bilinear")
    inputs = tf.keras.Input(shape=(img_height, img_width, 3))
    x = preprocessor(inputs)
    x = backbone_model(x, training=False)
    final_stage = x["stage12"]
    x = keras.layers.Lambda(lambda t: t[:, 0, :])(final_stage)

    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.3)(x)
    x = keras.layers.Dense(256, activation="relu")(x)
    final_outputs = keras.layers.Dense(num_classes)(x)

    model = keras.Model(inputs, final_outputs)
    model.summary()
    
    base_learning_rate = 0.0001
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=base_learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model, _get_checkpoint_callback()
