import tensorflow as tf
import numpy as np

from config import BATCH_SIZE, BUFFER_SIZE, NUM_EPOCHS

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def write_tfrecord(filename, spectrograms, inst_freq):
    """
    Write a list of spectrograms to a TFRecord file.
    
    Args:
    filename (str): The filename for the TFRecord file.
    spectrograms (list of np.array): List of spectrograms to be written to the TFRecord.
    inst_freq (list of np.array): Corresponding instantaneous frequency to be written.
    """
    with tf.io.TFRecordWriter(filename) as writer:
        # Serialize each pair of spectrogram and corresponding instantaneous frequency
        example = serialize_example(spectrograms, inst_freq)
        # Write the serialized example to the TFRecord file
        writer.write(example)

def serialize_example(spectrograms, inst_freq):
    """
    Serialize spectrograms into a format compatible with TFRecords.
    
    Args:
    spectrograms (np.array): The (N, N, len(NW)) spectrograms to be serialized.
    inst_freq (np.array): The (N, N) corresponding instantaneous frequency to be serialized.

    Returns:
    tf.train.Example: A TF Example ready to be written as a TFRecord.
    """
    # Flatten the 3D array to 1D (required for serialization)
    spectrogram_flat = spectrograms.flatten().astype(np.float32)
    inst_freq_flat = inst_freq.flatten().astype(np.float32)

    # Create a dictionary mapping the feature name to the tf.train.Feature
    feature = {
        'spectrograms': _bytes_feature(spectrogram_flat.tobytes()),
        'shape': _bytes_feature(np.array(spectrograms.shape).astype(np.int32).tobytes()),  # Store shape as integers for parsing
        'inst_freq': _bytes_feature(inst_freq_flat.tobytes()),
        'inst_freq_shape': _bytes_feature(np.array(inst_freq.shape).astype(np.int32).tobytes())  # Store the shape of 

    }
    
    # Create the Features message using the dictionary
    features = tf.train.Features(feature=feature)
    
    # Create an Example message
    example = tf.train.Example(features=features)
    
    return example.SerializeToString()

def generate_tfrecord_filename(index):
    """Generate a filename for TFRecord files."""
    return f"data/{index:03d}.tfrecord"


def parse_example(serialized_example):
    feature_description = {
        'spectrograms': tf.io.FixedLenFeature([], tf.string),
        'shape': tf.io.FixedLenFeature([], tf.string),
        'inst_freq': tf.io.FixedLenFeature([], tf.string),
        'inst_freq_shape': tf.io.FixedLenFeature([], tf.string),
    }

    example = tf.io.parse_single_example(serialized_example, feature_description)

    spectrograms_flat = tf.io.decode_raw(example['spectrograms'], tf.float32)
    spectrogram_shape = tf.io.decode_raw(example['shape'], tf.int32)
    spectrograms = tf.reshape(spectrograms_flat, spectrogram_shape)

    inst_freq_flat = tf.io.decode_raw(example['inst_freq'], tf.float32)
    inst_freq_shape = tf.io.decode_raw(example['inst_freq_shape'], tf.int32)
    inst_freq = tf.reshape(inst_freq_flat, inst_freq_shape)

    return spectrograms, inst_freq

def generator(iterator):
  try:
    while True:
      yield next(iterator)
  except (RuntimeError, StopIteration):
    return
  
def create_dataset_iterator(filenames, shuffle=True, batch_size=BATCH_SIZE, num_epochs=NUM_EPOCHS, buffer_size=BUFFER_SIZE):
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(parse_example)
    dataset = dataset.repeat(count=num_epochs)
    if shuffle is True:
        dataset = dataset.shuffle(buffer_size=buffer_size)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    return iter(dataset)