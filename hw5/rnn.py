from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Embedding, Dropout, Flatten, GRU
from keras.layers.convolutional import Conv1D
from keras.layers.pooling import MaxPooling1D
from keras.optimizers import Adam
from keras.layers.advanced_activations import LeakyReLU
from kr_base import BaseModel
from keras import backend as K
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras.layers.normalization import BatchNormalization
from keras.layers.wrappers import Bidirectional
import numpy as np
from keras.layers.merge import Concatenate


def _f1score(y_true, y_pred):
    true_positives = K.sum(y_true * y_pred, axis=1)
    precision = true_positives / (K.sum(y_pred, axis=1) + K.epsilon())
    recall = true_positives / (K.sum(y_true, axis=1) + K.epsilon())
    f1 = 2 * precision * recall / (precision + recall + K.epsilon())
    return K.mean(f1)


def _f1loss(y_true, y_pred):
    true_positives = K.sum(y_true * y_pred, axis=1)
    precision = true_positives / (K.sum(y_pred, axis=1) + K.epsilon())
    recall = true_positives / (K.sum(y_true, axis=1) + K.epsilon())
    f1 = 2 * precision * recall / (precision + recall + K.epsilon())
    return -K.mean(f1)


class TextClassifier(BaseModel):

    def _build_model(self, seq_length, n_labels):
        self.model = Sequential()
        self.model.add(Embedding(self.vol_size + 1,
                                 self.embedding_matrix.shape[1],
                                 weights=[self.embedding_matrix],
                                 input_length=seq_length,
                                 trainable=False))

        self.model.add(Conv1D(100, 2))
        self.model.add(BatchNormalization(momentum=0.5))
        self.model.add(LeakyReLU())
        self.model.add(MaxPooling1D(2))
        self.model.add(Dropout(0.2))

        # self.model.add(Conv1D(100, 3))
        # self.model.add(BatchNormalization(momentum=0.5))
        # self.model.add(LeakyReLU())
        # self.model.add(MaxPooling1D(3))
        # self.model.add(Dropout(0.2))
        self.model.add(GRU(512, return_sequences=True))

        self.model.add(Conv1D(100, 2))
        self.model.add(BatchNormalization(momentum=0.5))
        self.model.add(LeakyReLU())
        self.model.add(MaxPooling1D(2))
        self.model.add(Dropout(0.2))

        self.model.add(GRU(256))

        # nns = []
        # for i in range(n_labels):
        #     seq = Sequential()
        #     seq.add(Dense(64, input_shape=(512,)))
        #     seq.add(BatchNormalization(momentum=0.5))
        #     seq.add(LeakyReLU())

        #     seq.add(Dropout(0.4))
        #     nns.append(seq)
            
        # self.model.add(Concatenate(nns))
        # self.model.add(Flatten())
        self.model.add(Dense(n_labels, activation='sigmoid'))
        optimizer = Adam(lr=self.lr, decay=self.lr_decay)

        self.model.compile(loss=_f1loss,
                           optimizer=optimizer,
                           metrics=[_f1score])

    def __init__(self, vol_size=None, embedding_matrix=None, n_iters=100,
                 lr=0.001, lr_decay=0, batch_size=128):
        self.vol_size = vol_size
        self.embedding_matrix = embedding_matrix
        self.n_iters = n_iters
        self.lr = lr
        self.lr_decay = lr_decay
        self.batch_size = batch_size
        self.model = None

        # set GPU memory limit
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.3
        set_session(tf.Session(config=config))

    def fit(self, X, y, valid=None):
        if self.model is None:
            self._build_model(X.shape[1], y.shape[1])

        if valid is not None:
            valid = (valid['x'], valid['y'])

        weights = {}
        for i in range(X.shape[1]):
            total = X.shape[0]
            n_positive = np.sum(X[:, i])
            weights[i] = (total - n_positive) / n_positive

        self.model.fit(X, y,
                       epochs=self.n_iters,
                       validation_data=valid,
                       batch_size=self.batch_size)

    def load(self, filename):
        self.model = load_model(filename,
                                custom_objects={'_f1score': _f1score})
