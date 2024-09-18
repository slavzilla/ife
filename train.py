

from dataset import create_dataset_iterator, generator
from tensorflow.keras.losses import BinaryCrossentropy
import numpy as np
import tensorflow as tf

from plotting import plot_instantaneous_frequency, plot_stfts


def compute_loss(model, input, mask):
  o = model(input)
  return BinaryCrossentropy()(o, mask)


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
    for spectrograms, inst_frq in generator(it):
        loss = train_step(model, spectrograms, inst_frq, optimizer).numpy()
        losses.append(loss)

        print (loss)

        step += 1

    output = model(spectrograms)
    plot_instantaneous_frequency(np.squeeze(output.numpy()[0]), path='output.pdf')
    plot_instantaneous_frequency(np.squeeze(inst_frq.numpy()[0]))
       


if __name__=="__main__":
    train()