import pandas as pd
import numpy as np


def read_train(filename):
    df = pd.read_csv(filename, header=1)
    df = df.as_matrix()
    return {'x': df[:, [1, 2]], 'y': df[:, 3]}


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
