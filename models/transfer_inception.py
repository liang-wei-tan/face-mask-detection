import os
import tensorflow as tf

img_height = 128
img_width = 128
INPUT_DIM = (None, img_height, img_width, 3)
IMG_SHAPE = (img_height, img_width) + (3,)
CHECKPOINT_DIR = "checkpoints/transfer_inception"


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


def create_transfer_inception(num_classes=2):
    """
    Creates a transfer learning model using InceptionV3 for mask detection.

    Args:
        num_classes: Number of output classes (default: 2 for with_mask/without_mask)

    Returns:
        Tuple of (model, checkpoint_callback)
    """
    base_model = tf.keras.applications.Xception(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')
    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    base_model.trainable = False
    num_classes = 2
    prediction_layer = tf.keras.layers.Dense(num_classes)
    preprocess_input = tf.keras.applications.xception.preprocess_input
    
    inputs = tf.keras.Input(shape=(img_height, img_width, 3))
    x = preprocess_input(inputs)
    x = base_model(x, training=False)
    x = global_average_layer(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = prediction_layer(x)
    model = tf.keras.Model(inputs, outputs)
    
    tf.keras.utils.plot_model(model, show_shapes=True)

    base_learning_rate = 0.0001
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

    model.build(INPUT_DIM)

    return model, _get_checkpoint_callback()
