from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Embedding, Dropout, Flatten, GRU, Input
from keras.layers.core import Lambda, Activation
from keras.layers.convolutional import Conv1D
from keras.layers.pooling import MaxPooling1D
from keras.optimizers import Adam
from keras.layers.advanced_activations import LeakyReLU
from kr_base import BaseModel
from keras import backend as K
from keras.models import Model
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras.layers.normalization import BatchNormalization
from keras.layers.wrappers import Bidirectional
import numpy as np
from keras.layers.merge import Concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint


def _f1score(y_true, y_pred):
    true_positives = K.sum(y_true * K.round(y_pred), axis=1)
    precision = true_positives / (K.sum(K.round(y_pred), axis=1) + K.epsilon())
    recall = true_positives / (K.sum(y_true, axis=1) + K.epsilon())
    f1 = 2 * precision * recall / (precision + recall + K.epsilon())
    return K.mean(f1)

# def _f1score(y_true,y_pred):
#     thresh = 0.4
#     y_pred = K.cast(K.greater(y_pred,thresh),dtype='float32')
#     tp = K.sum(y_true * y_pred,axis=-1)
    
#     precision=tp/(K.sum(y_pred,axis=-1)+K.epsilon())
#     recall=tp/(K.sum(y_true,axis=-1)+K.epsilon())
#     return K.mean(2*((precision*recall)/(precision+recall+K.epsilon())))

# def _f1score(y_true,y_pred):
#     thresh = 0.4
#     y_pred = K.cast(K.greater(y_pred,thresh),dtype='float32')
#     tp = K.sum(y_true * y_pred,axis=-1)

#     precision=tp/(K.sum(y_pred,axis=-1)+K.epsilon())
#     recall=tp/(K.sum(y_true,axis=-1)+K.epsilon())
#     return K.mean(2*((precision*recall)/(precision+recall+K.epsilon())))


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

        # self.model.add(GRU(128, activation='tanh',
        #                    return_sequences=True, dropout=0.3))
        self.model.add(GRU(128, activation='tanh',
                           return_sequences=False, dropout=0.4))

        self.model.add(Dense(256, activation='relu'))
        self.model.add(Dropout(0.2))

        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dropout(0.2))

        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dropout(0.2))

        self.model.add(Dense(n_labels, activation='sigmoid'))
        optimizer = Adam(lr=self.lr, decay=self.lr_decay)

        self.model.compile(loss='categorical_crossentropy',
                           optimizer=optimizer,
                           metrics=[_f1score])

        # self.model.add(Activation('sigmoid'))

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
            # self._build_model_func(X.shape[1], y.shape[1])

        if valid is not None:
            valid = (valid['x'], valid['y'])

        weights = {}
        for i in range(X.shape[1]):
            total = X.shape[0]
            n_positive = np.sum(X[:, i])
            weights[i] = (total - n_positive) / n_positive

        earlystopping = EarlyStopping(monitor='val__f1score',
                                      patience=15,
                                      mode='max')

        checkpoint = ModelCheckpoint(filepath='best.h5',
                                     verbose=1,
                                     save_best_only=True,
                                     monitor='val__f1score',
                                     mode='max')

        self.model.fit(X, y,
                       epochs=self.n_iters,
                       validation_data=valid,
                       batch_size=self.batch_size,
                       callbacks=[earlystopping, checkpoint])

    def load(self, filename):
        self.model = load_model(filename,
                                custom_objects={'_f1score': _f1score})

    def predict_raw(self, X):
        return self.model.predict(X)
