import os
import time
import numpy as np
from keras.callbacks import Callback
from keras.models import Sequential, load_model
from keras.layers import Input, Dense, Dropout, Flatten, Activation
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras import regularizers
from keras.layers.advanced_activations import LeakyReLU
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.noise import GaussianNoise


class BaseModel:

    class History(Callback):

        def on_train_begin(self, logs={}):
            self.tr_losses = []
            self.val_losses = []
            self.tr_accs = []
            self.val_accs = []

        def on_epoch_end(self, epoch, logs={}):
            self.tr_losses.append(logs.get('loss'))
            self.val_losses.append(logs.get('val_loss'))
            self.tr_accs.append(logs.get('acc'))
            self.val_accs.append(logs.get('val_acc'))

    def _build_model(self):
        pass
        
    def __init__(self):
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.4
        set_session(tf.Session(config=config))

    def load(self, filename):
        self.model = load_model(filename)

    # def predict(self, X):
    #     return np.argmax(self.model.predict(X.reshape(-1, 48, 48, 1)),
    #                      axis=1)
        
    def save(self, n_iter, path=None):
        if path is None:
            path = self.save_path

        os.makedirs(path, exist_ok=True)
        self.model.save(os.path.join(path, 'model-%d.h5' % n_iter))

    def dump_history(self, n_iter, path=None):
        if path is None:
            path = self.save_path

        with open(os.path.join(path, 'train_loss-%d' % n_iter), 'a') as f:
            for loss in self.history.tr_losses:
                f.write('{}\n'.format(loss))
        with open(os.path.join(path, 'train_accuracy-%d' % n_iter), 'a') as f:
            for acc in self.history.tr_accs:
                f.write('{}\n'.format(acc))
        with open(os.path.join(path, 'valid_loss-%d' % n_iter), 'a') as f:
            for loss in self.history.val_losses:
                f.write('{}\n'.format(loss))
        with open(os.path.join(path, 'valid_accuracy-%d' % n_iter), 'a') as f:
            for acc in self.history.val_accs:
                f.write('{}\n'.format(acc))

    def fit(self, X, y, valid):
        pass
