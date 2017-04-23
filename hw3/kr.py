# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import os
import time
import numpy as np
from keras.callbacks import Callback
from keras.models import Sequential
from keras.layers import Input, Dense, Dropout, Flatten, Activation
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import Adam
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session


class CNNModel:

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
        self.model = Sequential()
        # CNN part (you can repeat this part several times)
        self.model.add(Convolution2D(8, 1, 1,
                                     border_mode='valid',
                                     input_shape=(48, 48, 1)))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.8))

        # Fully connected part
        self.model.add(Flatten())
        self.model.add(Dense(16))
        self.model.add(Activation('relu'))
        self.model.add(Dense(self.n_classes))
        self.model.add(Activation('softmax'))
        opt = Adam(lr=self.eta, decay=0.0)
        self.model.compile(loss='categorical_crossentropy',
                           optimizer=opt,
                           metrics=['accuracy'])
        # model.summary()

    def _one_hot_encode(self, labels):
        n_classes = np.max(labels) - np.min(labels) + 1
        one_hot = np.zeros((labels.shape[0], n_classes))
        one_hot[np.arange(one_hot.shape[0]), labels] = 1
        return one_hot
        
    def __init__(self, batch_size=40, n_iter=100,
                 eta=1e-5, save_path=None):
        self.n_classes = 7
        self.batch_size = batch_size
        self.n_iter = n_iter
        self.eta = eta
        if save_path is None:
            save_path = time.ctime()
                    
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.1
        set_session(tf.Session(config=config))

    def fit(self, X, y, valid):
        self._build_model()
        self.history = self.History()
        X = X.reshape(-1, 48, 48, 1)
        y = self._one_hot_encode(y)
        valid = (valid['x'].reshape(-1, 48, 48, 1),
                 self._one_hot_encode(valid['y']))
        # pdb.set_trace()
        self.model.fit(x=X, y=y,
                       batch_size=self.batch_size,
                       epochs=self.n_iter,
                       validation_data=valid,
                       callbacks=[self.history])

    def save(self, path=None):
        if path is None:
            path = self.save_path

        self.model.save(os.path.join(path, 'model.h5'))

    def dump_history(self, path=None):
        if path is None:
            path = self.save_path

        with open(os.path.join(path, 'train_loss'), 'a') as f:
            for loss in self.history.tr_losses:
                f.write('{}\n'.format(loss))
        with open(os.path.join(path, 'train_accuracy'), 'a') as f:
            for acc in self.history.tr_accs:
                f.write('{}\n'.format(acc))
        with open(os.path.join(path, 'valid_loss'), 'a') as f:
            for loss in self.history.val_losses:
                f.write('{}\n'.format(loss))
        with open(os.path.join(path, 'valid_accuracy'), 'a') as f:
            for acc in self.history.val_accs:
                f.write('{}\n'.format(acc))
