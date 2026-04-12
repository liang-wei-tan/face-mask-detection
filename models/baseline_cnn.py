import tensorflow as tf

INPUT_DIM = (None, 128, 128, 3)

def create_baseline_cnn(num_classes=2):
    """
    Creates a baseline CNN model for mask detection.

    Args:
        num_classes: Number of output classes (default: 2 for with_mask/without_mask)

    Returns:
        A compiled Keras Sequential model
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Rescaling(1./255),
        tf.keras.layers.Conv2D(64, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(num_classes)
    ])

    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    
    model.build(INPUT_DIM)

    return model
