from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Embedding, Dropout, Flatten, GRU, Input, Merge
from keras.layers.core import Lambda, Activation
from keras.layers.convolutional import Conv1D
from keras.layers.pooling import MaxPooling1D
from keras.optimizers import Adam
from keras.layers.advanced_activations import LeakyReLU
from keras import backend as K
from keras.models import Model
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras.layers.normalization import BatchNormalization
from keras.layers.wrappers import Bidirectional
import numpy as np
from keras.layers.merge import Concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import layers


class MF:

    def _build_model(self, dim1, dim2, d_latent):
        print("dim1 = %d, dim2 = %d" % (dim1, dim2))
        inputs = Input(shape=(2,))
        p = Lambda(lambda x: x[:, 0:1])(inputs)
        p = Embedding(dim1, d_latent, input_length=1)(p)
        p = Flatten()(p)
        
        q = Lambda(lambda x: x[:, 1:2])(inputs)
        q = Embedding(dim2, d_latent, input_length=1)(q)
        q = Flatten()(q)
        pred = layers.Dot(axes=1)([p, q])
        self.model = Model(input=inputs, output=pred)

        # P = Sequential()
        # P.add(Embedding(dim1, d_latent, input_length=1))
        # P.add(Flatten())

        # Q = Sequential()
        # Q.add(Embedding(dim1, d_latent, input_length=1))
        # Q.add(Flatten())

        # self.model = Sequential()
        # self.model.add(Merge([P, Q], mode='dot', dot_axes=-1))
        self.model.summary()

        optimizer = Adam(lr=self.lr, decay=self.lr_decay)

        self.model.compile(loss='mean_squared_error',
                           optimizer=optimizer,
                           metrics=['mean_squared_error'])

    def __init__(self, n_iters=100, lr=0.001,
                 lr_decay=0, batch_size=128,
                 filename=None, ram=0.2, d_latent=50):
        self.n_iters = n_iters
        self.lr = lr
        self.lr_decay = lr_decay
        self.batch_size = batch_size
        self.model = None
        self.filename = filename
        self.d_latent = d_latent

        # set GPU memory limit
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = ram
        set_session(tf.Session(config=config))

    def fit(self, X, y, valid=None):
        if self.model is None:
            self._build_model(np.max(X[:, 0]) + 1,
                              np.max(X[:, 1]) + 1,
                              self.d_latent)

        if valid is not None:
            valid = (valid['x'], valid['y'])

        earlystopping = EarlyStopping(monitor='val_mean_squared_error',
                                      patience=15,
                                      mode='min')

        checkpoint = ModelCheckpoint(filepath=self.filename,
                                     verbose=1,
                                     save_best_only=True,
                                     monitor='val_mean_squared_error',
                                     mode='min')

        self.model.fit(X, y,
                       epochs=self.n_iters,
                       validation_data=valid,
                       batch_size=self.batch_size,
                       callbacks=[earlystopping, checkpoint])

    def load(self, filename):
        self.model = load_model(filename)

    def predict_raw(self, X):
        return self.model.predict(X)

    def predict(self, X, threshold=0.5):
        predict = self.model.predict(X)
        return np.where(predict > threshold, 1, 0)
