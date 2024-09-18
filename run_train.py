from model import create_model
from train import train
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.backend import clear_session
import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import random
random.seed(42)

if __name__=="__main__":
    clear_session()
    model = create_model()
    optimizer = Nadam(learning_rate=1e-4, clipnorm=1.0)
    files = [os.path.join(dirpath, filename) for dirpath, _, filenames in os.walk('data') for filename in filenames]
    train(model, optimizer, files)