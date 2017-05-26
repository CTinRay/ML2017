import pandas as pd
import numpy as np


def read_train(filename):
    df = pd.read_csv(filename, header=0)
    df = df.as_matrix()
    return {'x': df[:, [1, 2]], 'y': df[:, 3]}


def read_test(filename):
    df = pd.read_csv(filename, header=0)
    df = df.as_matrix()
    return {'x': df[:, [1, 2]]}


def write_predict(filename, predict):
    with open(filename, 'w') as f:
        f.write('TestDataID,Rating\n')
        for i in range(predict.shape[0]):
            f.write('%d,%f\n' % (i + 1, predict[i]))


def split_valid(data, valid_ratio, indices=None):
    if indices is None:
        indices = np.arange(data['x'].shape[0])
        np.random.shuffle(indices)

    data['x'] = data['x'][indices]
    data['y'] = data['y'][indices]

    n_valid = int(data['x'].shape[0] * valid_ratio)
    train = {'x': data['x'][n_valid:], 'y': data['y'][n_valid:]}
    valid = {'x': data['x'][:n_valid], 'y': data['y'][:n_valid]}

    return train, valid
