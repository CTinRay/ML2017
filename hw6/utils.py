import pandas as pd
import numpy as np
import re


def read_train(filename):
    df = pd.read_csv(filename, header=0)
    df = df.as_matrix()
    return {'x': df[:, [1, 2]], 'y': df[:, 3]}


def read_test(filename):
    df = pd.read_csv(filename, header=0)
    df = df.as_matrix()
    return df[:, 0], {'x': df[:, [1, 2]]}


def write_predict(filename, inds, predict):
    with open(filename, 'w') as f:
        f.write('TestDataID,Rating\n')
        for i in range(predict.shape[0]):
            f.write('%d,%f\n' % (inds[i], predict[i]))


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


def get_user_features(filename):
    df = pd.read_csv(filename, header=0)
    df = df.sort_values(['UserID'])
    data = df['Age'].as_matrix().reshape(-1, 1)
    data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)
    return data


def get_movie_features(filename):
    def encode_category(category):
        all_categories = ['Musical', 'Comedy', 'Romance',
                          'Drama', 'War', 'Fantasy', 'Action',
                          'Adventure', 'Crime', 'Horror',
                          'Thriller', 'Mystery', 'Sci-Fi',
                          'Film-Noir', 'Western', 'Documentary',
                          'Animation', 'Children\'s']
        encoded = np.zeros(len(all_categories))
        for cat in category:
            ind = all_categories.index(category)
            encoded[ind] = 1

        return encoded

    categories = [[] for i in range(3952)]
    years = [2000] * 3952
    with open(filename, encoding='latin_1') as f:
        next(f)
        for row in f:
            cols = row.split(',')
            mid = int(cols[0])
            category = cols[-1].strip().split('|')[0]
            categories[mid - 1] = category
            year = int(re.search('\(([0-9]+)\)', row).groups()[0])
            years[mid - 1] = year

    categories = [encode_category(cat) for cat in categories]
    categories = np.array(categories)
    years = np.array(years).reshape(-1, 1)
    years = (years - np.mean(years)) / np.std(years)

    return np.concatenate([categories, years], axis=1)
