# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import os
import time
import numpy as np
from keras.callbacks import Callback
from keras.models import Sequential
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

        # self.model.add(GaussianNoise(0.01, input_shape=(48, 48, 1)))

        # CNN part (you can repeat this part several times)
        self.model.add(Conv2D(32, (3, 3),
                              padding='same',
                              input_shape=(48, 48, 1)))
        # kernel_regularizer=regularizers.l2(1e-9)))
        self.model.add(BatchNormalization(momentum=0.5))
        self.model.add(LeakyReLU())
        self.model.add(MaxPooling2D())
        # self.model.add(Dropout(0.1))

        # 24 x 24

        self.model.add(Conv2D(64, (3, 3),
                              padding='same'))
        # kernel_regularizer=regularizers.l2(1e-9)))
        self.model.add(BatchNormalization(momentum=0.5))
        self.model.add(LeakyReLU())
        self.model.add(MaxPooling2D())
        self.model.add(Dropout(0.1))

        # 12 x 12
        self.model.add(Conv2D(128, (3, 3),
                              padding='same'))
        # kernel_regularizer=regularizers.l2(1e-9)))
        self.model.add(BatchNormalization(momentum=0.5))
        self.model.add(LeakyReLU())
        self.model.add(MaxPooling2D())
        self.model.add(Dropout(0.2))

        # 6 x 6
        self.model.add(Conv2D(256, (3, 3),
                              padding='same'))
        self.model.add(BatchNormalization(momentum=0.5))
        self.model.add(LeakyReLU())
        self.model.add(MaxPooling2D())
        self.model.add(Dropout(0.1))

        # Fully connected part
        self.model.add(Flatten())
        self.model.add(Dense(512))
        self.model.add(BatchNormalization(momentum=0.5))
        self.model.add(LeakyReLU())
        self.model.add(Dropout(0.5))

        self.model.add(Dense(256))
        self.model.add(BatchNormalization(momentum=0.5))
        self.model.add(LeakyReLU())
        self.model.add(Dropout(0.1))

        self.model.add(Dense(self.n_classes))
        self.model.add(Activation('softmax'))
        opt = Adam(lr=self.eta, decay=self.eta_decay)
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
                 eta=1e-5, save_path=None, eta_decay=5e-6):
        self.n_classes = 7
        self.batch_size = batch_size
        self.n_iter = n_iter
        self.eta = eta
        self.eta_decay = eta_decay
        if save_path is None:
            self.save_path = time.ctime()
        else:
            self.save_path = save_path

        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.4
        set_session(tf.Session(config=config))

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
        self._build_model()
        self.history = self.History()
        X = X.reshape(-1, 48, 48, 1)
        y = self._one_hot_encode(y)
        valid = (valid['x'].reshape(-1, 48, 48, 1),
                 self._one_hot_encode(valid['y']))

        # augmentation
        datagen = ImageDataGenerator(
            # featurewise_center=True,
            # featurewise_std_normalization=True,
            # samplewise_center=True,
            rotation_range=30,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='constant',
            vertical_flip=False)

        datagen.fit(X)
        datagen_flow = datagen.flow(X, y, batch_size=self.batch_size)

        for i in range(self.n_iter // 100):
            self.model.fit_generator(datagen_flow,
                                     steps_per_epoch=len(X) / self.batch_size,
                                     initial_epoch=i * 100,
                                     epochs=(i + 1) * 100,
                                     validation_data=valid,
                                     callbacks=[self.history])
            self.save(i * 100)
            self.dump_history(i * 100)

        self.model.fit_generator(datagen_flow,
                                 steps_per_epoch=len(X) / self.batch_size,
                                 initial_epoch=i * 100,
                                 epochs=self.n_iter,
                                 validation_data=valid,
                                 callbacks=[self.history])

        # self.model.fit(x=X, y=y,
        #                batch_size=self.batch_size,
        #                epochs=self.n_iter,
        #                validation_data=valid,
        #                callbacks=[self.history])
