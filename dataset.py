import tensorflow as tf
import numpy as np

from config import BATCH_SIZE, BUFFER_SIZE, NUM_EPOCHS

def _bytes_feature(value):
    """
    Returns a bytes_list from a string / byte value.
    
    Args:
    value (bytes): The value to be converted to bytes_list.

    Returns:
    tf.train.Feature: A TensorFlow Feature containing a bytes_list.
    """
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def write_tfrecord(filename, spectrograms, inst_freq):
    """
    Write a list of spectrograms and their corresponding instantaneous frequency to a TFRecord file.
    
    Args:
    filename (str): The filename for the TFRecord file.
    spectrograms (np.array): A list of spectrogram arrays to be serialized and written.
    inst_freq (np.array): Corresponding instantaneous frequency arrays to be serialized and written.
    """
    with tf.io.TFRecordWriter(filename) as writer:
        # Serialize each pair of spectrogram and corresponding instantaneous frequency
        example = serialize_example(spectrograms, inst_freq)
        # Write the serialized example to the TFRecord file
        writer.write(example)

def serialize_example(spectrograms, inst_freq):
    """
    Serialize spectrograms and instantaneous frequency into a format compatible with TFRecords.
    
    Args:
    spectrograms (np.array): The 3D array (?, N, len(NW)) of spectrograms to be serialized.
    inst_freq (np.array): The 2D array (N, N) corresponding instantaneous frequency to be serialized.

    Returns:
    str: A serialized TensorFlow Example ready to be written as a TFRecord.
    """
    # Flatten the 3D array to 1D (required for serialization)
    spectrogram_flat = spectrograms.flatten().astype(np.float32)
    inst_freq_flat = inst_freq.flatten().astype(np.float32)

    # Create a dictionary mapping the feature name to the tf.train.Feature
    feature = {
        'spectrograms': _bytes_feature(spectrogram_flat.tobytes()),
        'shape': _bytes_feature(np.array(spectrograms.shape).astype(np.int32).tobytes()),  # Store shape for parsing
        'inst_freq': _bytes_feature(inst_freq_flat.tobytes()),
        'inst_freq_shape': _bytes_feature(np.array(inst_freq.shape).astype(np.int32).tobytes())  # Store the shape for parsing
    }
    
    features = tf.train.Features(feature=feature)
    example = tf.train.Example(features=features)
    
    return example.SerializeToString()

def generate_tfrecord_filename(index):
    """
    Generate a filename for TFRecord files.

    Args:
    index (int): Index used to create the filename.

    Returns:
    str: Generated filename in the format 'data/###.tfrecord'.
    """
    return f"data/{index:03d}.tfrecord"


def parse_example(serialized_example):
    """
    Parse a serialized TFRecord example back into spectrograms and instantaneous frequency.
    
    Args:
    serialized_example (tf.Tensor): Serialized TFRecord example.

    Returns:
    Tuple (spectrograms, inst_freq): Parsed spectrogram and corresponding instantaneous frequency arrays.
    """
    feature_description = {
        'spectrograms': tf.io.FixedLenFeature([], tf.string),
        'shape': tf.io.FixedLenFeature([], tf.string),
        'inst_freq': tf.io.FixedLenFeature([], tf.string),
        'inst_freq_shape': tf.io.FixedLenFeature([], tf.string),
    }

    # Parse the example
    example = tf.io.parse_single_example(serialized_example, feature_description)

    # Decode the spectrogram and reshape
    spectrograms_flat = tf.io.decode_raw(example['spectrograms'], tf.float32)
    spectrogram_shape = tf.io.decode_raw(example['shape'], tf.int32)
    spectrograms = tf.reshape(spectrograms_flat, spectrogram_shape)

    # Decode the instantaneous frequency and reshape
    inst_freq_flat = tf.io.decode_raw(example['inst_freq'], tf.float32)
    inst_freq_shape = tf.io.decode_raw(example['inst_freq_shape'], tf.int32)
    inst_freq = tf.reshape(inst_freq_flat, inst_freq_shape)

    return spectrograms, inst_freq

def generator(iterator):
    """
    A generator function to yield elements from an iterator.
    
    Args:
    iterator (iterator): An iterator to yield elements from.

    Yields:
    Next element from the iterator until StopIteration.
    """
    try:
        while True:
            yield next(iterator)
    except (RuntimeError, StopIteration):
        return
  
def create_dataset_iterator(filenames, shuffle=True, batch_size=BATCH_SIZE, num_epochs=NUM_EPOCHS, buffer_size=BUFFER_SIZE):
    """
    Create an iterator over a dataset of TFRecord files, applying batching, shuffling, and parsing.
    
    Args:
    filenames (list of str): List of filenames of the TFRecord files.
    shuffle (bool): Whether to shuffle the dataset.
    batch_size (int): Number of examples per batch.
    num_epochs (int): Number of times to repeat the dataset.
    buffer_size (int): Buffer size for shuffling.

    Returns:
    iterator: An iterator over the processed dataset.
    """
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(parse_example)
    dataset = dataset.repeat(count=num_epochs)
    
    if shuffle:
        dataset = dataset.shuffle(buffer_size=buffer_size)
    
    dataset = dataset.batch(batch_size, drop_remainder=True)
    return iter(dataset)