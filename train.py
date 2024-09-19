

from dataset import create_dataset_iterator, generator
from tensorflow.keras.losses import BinaryCrossentropy, MeanAbsoluteError
import numpy as np
import tensorflow as tf
import time



from plotting import plot_instantaneous_frequency, plot_stfts

import tensorflow as tf

from utils import threshold_output

def compute_focal_loss(y_true, y_pred, gamma=3.0, alpha=0.1):
    """
    Computes the Focal Loss for the given true labels and predicted labels.

    Args:
    y_true (tf.Tensor): Ground truth labels (binary, 0 or 1).
    y_pred (tf.Tensor): Model predictions (output of sigmoid, values between 0 and 1).
    gamma (float): Focusing parameter to adjust the importance of hard-to-classify examples.
    alpha (float): Balancing factor for class imbalance.

    Returns:
    tf.Tensor: Computed Focal Loss.
    """
    
    # Clip predictions to prevent log(0) errors
    epsilon = tf.keras.backend.epsilon()
    y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)

    # Compute the binary cross-entropy loss for each sample
    cross_entropy = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)

    # Calculate the probability of true class
    p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)

    # Apply the modulating factor (1 - p_t)^gamma
    modulating_factor = tf.pow(1.0 - p_t, gamma)

    # Apply the alpha balancing factor
    alpha_weight = y_true * alpha + (1 - y_true) * (1 - alpha)

    # Compute the final Focal Loss
    focal_loss = alpha_weight * modulating_factor * cross_entropy
    
    return tf.reduce_mean(focal_loss)

def compute_dice_loss(y_true, y_pred):
    numerator = 2 * tf.reduce_sum(y_true * y_pred)
    denominator = tf.reduce_sum(y_true + y_pred)
    return 1 - numerator / (denominator + tf.keras.backend.epsilon())


def compute_loss(model, input, mask):
  predictions = model(input)
  return compute_focal_loss(mask, predictions) #+ compute_dice_loss(mask, predictions)


def train_step(model, input, mask, optimizer):
  with tf.GradientTape() as tape:
    loss = compute_loss(model, input, mask)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
  return loss


def train(model, optimizer, train_files):
    global step
    step = 0
    it = create_dataset_iterator(train_files)
    losses = []
    min_loss = float('inf')
    for spectrograms, inst_frq in generator(it):
        batch_start_time = time.time()
        loss = train_step(model, spectrograms, inst_frq, optimizer).numpy()
        if step%10==0:
          print("batch_num:", step, "loss:", loss, "elapsed time:", time.time() - batch_start_time)
          losses.append(loss)
        step += 1
        if step%1000==0:
          break

    output = model(spectrograms)
    arr = np.squeeze(output.numpy()[0])
    plot_instantaneous_frequency(arr, path='output.pdf')
    plot_instantaneous_frequency(np.squeeze(inst_frq.numpy()[0]))