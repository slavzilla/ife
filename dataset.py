import tensorflow as tf

def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _int64_feature(value):
    """Returns an int64_list from an int / unsigned int."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

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
    Serialize a 3D spectrogram into a format compatible with TFRecords.
    
    Args:
    spectrogram (np.array): The (N, N, len(NW)) spectrogram to be serialized.

    Returns:
    tf.train.Example: A TF Example ready to be written as a TFRecord.
    """
    # Flatten the 3D array to 1D (required for serialization)
    spectrogram_flat = spectrograms.flatten()
    inst_freq_flat = inst_freq.flatten()

    # Create a dictionary mapping the feature name to the tf.train.Feature
    feature = {
        'spectrograms': _float_feature(spectrogram_flat),
        'shape': _int64_feature(list(spectrograms.shape)),  # Store shape as integers for parsing
        'inst_freq': _float_feature(inst_freq_flat),
        'inst_freq_shape': _int64_feature(list(inst_freq.shape))  # Store the shape of instantaneous frequency for parsing
    }
    
    # Create the Features message using the dictionary
    features = tf.train.Features(feature=feature)
    
    # Create an Example message
    example = tf.train.Example(features=features)
    
    return example.SerializeToString()

def generate_tfrecord_filename(index):
    """Generate a filename for TFRecord files."""
    return f"{index:03d}.tfrecord"