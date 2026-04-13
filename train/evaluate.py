import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
from .utils import load_datasets

def error_analysis(checkpoint_dir="checkpoints/baseline_cnn", data_dir="face-mask-12k-images-dataset/Face Mask Dataset"):
  print(f"Loading dataset from {data_dir}...")
  _, _, test_ds, class_names = load_datasets(data_dir)

  print("Finding latest checkpoint...")
  latest_checkpoint = get_latest_checkpoint(checkpoint_dir)
  print(f"Loading model from: {latest_checkpoint}")

  model = tf.keras.models.load_model(latest_checkpoint)
  probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
  _, _, test_ds, class_names = load_datasets(data_dir)
  test_batches = list(test_ds.as_numpy_iterator())
  test_predictions = []

  for batch in test_batches:
    batch_image, batch_label = batch[0], batch[1]
    p = probability_model.predict(batch_image)
    test_predictions.append(p)
  
  incorrect_img_tmp = []
  incorrect_label_tmp = []
  incorrect_predictions_tmp = []

  for i in range(0, len(test_batches)):
    batch = test_batches[i]
    predictions = test_predictions[i]
    batch_img, batch_labels = batch[0], batch[1]
    p_temp = np.argmax(predictions, axis = 1)
    correctness = p_temp == batch_labels
    incorrect_data = correctness == False
    if len(batch_img[incorrect_data]) > 0:
      print('some incorrect_data, appending')
      incorrect_img_tmp.append(batch_img[incorrect_data])
      incorrect_label_tmp.append(batch_labels[incorrect_data])
      incorrect_predictions_tmp.append(predictions[incorrect_data])
  incorrect_img = np.concatenate(incorrect_img_tmp)
  incorrect_label = np.concatenate(incorrect_label_tmp)
  incorrect_predictions = np.concatenate(incorrect_predictions_tmp)
  num_rows = 30
  num_cols = 2
  num_images = num_rows*num_cols
  plt.figure(figsize=(2*2*num_cols, 2*num_rows))
  for i in range(num_images):
    if i >= len(incorrect_predictions):
      break
    plt.subplot(num_rows, 2*num_cols, 2*i+1)
    plot_image(i, incorrect_predictions[i], incorrect_label, incorrect_img, class_names)
    plt.subplot(num_rows, 2*num_cols, 2*i+2)
    plot_value_array(i, incorrect_predictions[i], incorrect_label)
  plt.tight_layout()
  plt.show()



def evaluate_model(model, test_ds, class_names=None):
    """
    Evaluate model on test dataset.

    Args:
        model: Keras model to evaluate
        test_ds: Test dataset
        class_names: List of class names (optional, for visualization)

    Returns:
        Tuple of (loss, accuracy)
    """
    loss, accuracy = model.evaluate(test_ds)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")

    if class_names:
        print(class_names)

        # Create a probability model for inference
        probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])

        temp_ds = test_ds.take(1)
        sample = list(temp_ds.as_numpy_iterator())[0]

        test_images = sample[0]
        test_labels = sample[1]

        predictions = probability_model.predict(test_images)

        num_rows = 5
        num_cols = 3
        num_images = num_rows * num_cols
        plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
        for i in range(num_images):
            plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
            plot_image(i, predictions[i], test_labels, test_images, class_names=class_names)
            plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
            plot_value_array(i, predictions[i], test_labels)
        plt.tight_layout()
        plt.show()

    return loss, accuracy


def plot_image(i, predictions_array, true_label, img, class_names):
  true_label, img = true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  img = np.asarray(img, dtype=np.uint8)

  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  true_label = true_label[i]
  plt.grid(False)
  plt.xticks(range(2))
  plt.yticks([])
  thisplot = plt.bar(range(2), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')


def get_latest_checkpoint(checkpoint_dir="checkpoints/baseline_cnn"):
  """
  Find and return the latest checkpoint file in the checkpoint directory.

  Args:
      checkpoint_dir: Path to checkpoint directory

  Returns:
      Path to the latest checkpoint file

  Raises:
      FileNotFoundError: If no checkpoints found
  """
  if not os.path.exists(checkpoint_dir):
    raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

  # Find all .h5 files
  checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "*.keras"))
  checkpoint_files = checkpoint_files + glob.glob(os.path.join(checkpoint_dir, "*.h5")) 

  # print(f"Found checkpoint files: {checkpoint_files}" )
  if not checkpoint_files:
    raise FileNotFoundError(f"No checkpoint files found in {checkpoint_dir}")

  # Sort by modification time and return the latest
  latest_checkpoint = max(checkpoint_files, key=os.path.getmtime)
  return latest_checkpoint


def main(data_dir, checkpoint_dir="checkpoints/baseline_cnn"):
  """
  Load the latest checkpoint model and evaluate it on the test dataset.

  Args:
      data_dir: Path to dataset root directory
      checkpoint_dir: Path to checkpoint directory
  """
  print(f"Loading dataset from {data_dir}...")
  _, _, test_ds, class_names = load_datasets(data_dir)

  print("Finding latest checkpoint...")
  latest_checkpoint = get_latest_checkpoint(checkpoint_dir)
  print(f"Loading model from: {latest_checkpoint}")

  model = tf.keras.models.load_model(latest_checkpoint)

  print("\nEvaluating model...")
  loss, accuracy = evaluate_model(model, test_ds, class_names=class_names)

  return model, loss, accuracy


if __name__ == "__main__":
  import sys

  data_dir = "face-mask-12k-images-dataset/Face Mask Dataset"
  if len(sys.argv) > 1:
    data_dir = sys.argv[1]
  if len(sys.argv) > 2:
    checkpoint_dir = sys.argv[2]

  model, loss, accuracy = main(data_dir)

