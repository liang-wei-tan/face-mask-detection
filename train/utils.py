import tensorflow as tf

# Configuration
BATCH_SIZE = 32
IMG_HEIGHT = 128
IMG_WIDTH = 128
AUTOTUNE = tf.data.AUTOTUNE


def load_datasets(data_dir, batch_size=BATCH_SIZE, img_size=(IMG_HEIGHT, IMG_WIDTH)):
    """
    Load train, validation, and test datasets from directory structure.

    Args:
        data_dir: Root directory containing Train/, Validation/, Test/ subdirectories
        batch_size: Batch size for data loading
        img_size: Tuple of (height, width) for image resizing

    Returns:
        Tuple of (train_ds, val_ds, test_ds, class_names)
    """
    train_dir = f"{data_dir}/Train"
    val_dir = f"{data_dir}/Validation"
    test_dir = f"{data_dir}/Test"

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        seed=123,
        image_size=img_size,
        batch_size=batch_size
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        seed=123,
        image_size=img_size,
        batch_size=batch_size
    )

    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        seed=123,
        image_size=img_size,
        batch_size=batch_size
    )

    class_names = train_ds.class_names

    return train_ds, val_ds, test_ds, class_names


def optimize_datasets(train_ds, val_ds, test_ds):
    """
    Apply caching and prefetching optimization to datasets.

    Args:
        train_ds: Training dataset
        val_ds: Validation dataset
        test_ds: Test dataset

    Returns:
        Tuple of optimized (train_ds, val_ds, test_ds)
    """
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds, test_ds
