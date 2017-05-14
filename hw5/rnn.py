from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding
from keras.optimizers import Adam
from keras.layers.advanced_activations import LeakyReLU
from kr_base import BaseModel
from keras import backend as K


class TextClassifier(BaseModel):

    def _build_model(self, seq_length, n_labels):
        self.model = Sequential()
        self.model.add(Embedding(self.vol_size + 1,
                                 self.embedding_matrix.shape[1],
                                 weights=[self.embedding_matrix],
                                 input_length=seq_length,
                                 trainable=False))
        self.model.add(LSTM(1024))
        self.model.add(Dense(512))
        self.model.add(LeakyReLU())
        self.model.add(Dense(n_labels, activation='sigmoid'))

        def f1_score(y_true, y_pred):
            n_correct = K.sum(y_true * y_pred)
            if n_correct == 0:
                return 0

            precision = n_correct / K.sum(y_pred)
            recall = n_correct / K.sum(y_true)
            return 2 * precision * recall / (precision + recall)

        optimizer = Adam(lr=self.lr, decay=self.lr_decay)
        self.model.compile(loss='binary_crossentropy',
                           optimizer=optimizer,
                           metrics=[f1_score])

    def __init__(self, vol_size, embedding_matrix, n_iters=100,
                 lr=0.001, lr_decay=0, batch_size=128):
        super()
        self.vol_size = vol_size
        self.embedding_matrix = embedding_matrix
        self.n_iters = n_iters
        self.lr = lr
        self.lr_decay = lr_decay
        self.batch_size = batch_size
        self.model = None

    def fit(self, X, y, valid=None):
        if self.model is None:
            self._build_model(X.shape[1], y.shape[1])

        if valid is not None:
            valid = (valid['x'], valid['y'])

        self.model.fit(X, y,
                       epochs=self.n_iters,
                       validation_data=valid)
