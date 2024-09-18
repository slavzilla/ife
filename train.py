

from dataset import create_dataset_iterator, generator
import numpy as np


def train(train_files=None):
    train_files = ['data/000.tfrecord']
    global step
    step = 0
    it = create_dataset_iterator(train_files, True, 1, 1, 1)
    for spectrograms, inst_frq in generator(it):
        print(inst_frq.numpy().shape)
        print(spectrograms.numpy().shape)
       


if __name__=="__main__":
    train()