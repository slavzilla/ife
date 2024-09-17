

from dataset import create_dataset_iterator, generator
import numpy as np


def train(train_files=None):
    train_files = ['data/000.tfrecord']
    global step
    step = 0
    it = create_dataset_iterator(train_files)
    for spectrograms, inst_frq in generator(it):
        inst_frq = np.squeeze(inst_frq.numpy())
       


if __name__=="__main__":
    train()