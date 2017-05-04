import numpy as np
import pandas as pd
import argparse
import pdb
import sys
import traceback
import pickle
from kr_semi import CNNModel


def get_X(csv):
    data = pd.read_csv(csv)
    xs = data.as_matrix()
    return xs


def get_Y(y):
    f = open(y)
    ys = list(map(int, f.read().split()))
    ys = np.array(ys).reshape(-1,)
    return ys


def get_XY(csv):
    xs = []
    ys = []
    with open(csv) as f:
        f.readline()
        for l in f:
            cols = l.split(',')
            ys.append(int(cols[0]))
            xs.append(list(map(int, cols[1].split())))

    return {'x': np.array(xs), 'y': np.array(ys)}


def split_valid_from_ind(raw, filename):
    n_rows = raw['x'].shape[0]

    # shuffle data
    with open(filename, 'rb') as f:
        inds = pickle.load(f)

    raw['x'] = raw['x'][inds]
    raw['y'] = raw['y'][inds]

    # split data
    n_valid = int(n_rows * 0.1)
    train_data = {'x': raw['x'][n_valid:], 'y': raw['y'][n_valid:]}
    valid_data = {'x': raw['x'][:n_valid], 'y': raw['y'][:n_valid]}

    return train_data, valid_data


def accuracy(y, y_):
    return np.sum(y == y_) / y.shape[0]


def write_csv(y, filename):
    f = open(filename, 'w')
    f.write('id,label\n')
    for i in range(y.shape[0]):
        f.write('%d,%d\n' % (i + 1, y[i]))

    f.close()


def get_test(csv):
    xs = []
    with open(csv) as f:
        f.readline()
        for l in f:
            cols = l.split(',')
            xs.append(list(map(int, cols[1].split())))

    return {'x': np.array(xs)}


def transform(x):
    # x = x.reshape((-1, 48, 48))
    # x = x[:, 4:44, 4:44]
    # x = x.reshape(-1, 40 * 40)
    x = x - np.mean(x, axis=1).reshape(-1, 1)
    x = x * 100 / np.linalg.norm(x)
    return x


def augmentate(data):
    xs_flip = data['x'].reshape(-1, 48, 48)[:, :, ::-1].reshape(-1, 48 * 48)
    data['x'] = np.concatenate((data['x'], xs_flip), axis=0)
    data['y'] = np.concatenate((data['y'], data['y']), axis=0)


def main():
    parser = argparse.ArgumentParser(description='ML HW2')
    parser.add_argument('model', type=str, help='model.h5')
    parser.add_argument('train', type=str, help='train.csv')
    parser.add_argument('test', type=str, help='test.csv')
    parser.add_argument('out', type=str, help='outcome')
    parser.add_argument('--valid_ratio', type=float,
                        help='ratio of validation data', default=0.2)
    parser.add_argument('--n_iter', type=int,
                        help='ratio of validation data', default=100)
    parser.add_argument('--eta', type=float,
                        help='learning rate', default=1e-5)
    parser.add_argument('--verbose', type=int, help='verbose', default=0)
    parser.add_argument('--batch_size', type=int,
                        help='batch size', default=10)
    parser.add_argument('--valid_file', type=str,
                        help='pickle to store indices used to validate.',
                        default='valid.pickle')
    parser.add_argument('--mean_max_file', type=str,
                        default='mean_max.pickle',
                        help='pickle to store mean and max')
    parser.add_argument('--eta_decay', type=float, default=0.000005,
                        help='pickle to store mean and max')
    args = parser.parse_args()
    raw_train = get_XY(args.train)

    train, valid = split_valid_from_ind(raw_train, args.valid_file)
    test = get_test(args.test)

    # do data augmentation
    # augmentate(train)

    # calculate mean, std
    mean = np.average(train['x'], axis=0)
    abs_max = np.max(np.abs(train['x'] - mean))

    # normalize
    train['x'] = (train['x'] - mean) / abs_max
    valid['x'] = (valid['x'] - mean) / abs_max
    test['x'] = (test['x'] - mean) / abs_max

    with open(args.mean_max_file, 'wb') as f:
        pickle.dump({'mean': mean, 'max': abs_max}, f)

    # transform
    # train['x'] = transform(train['x'])
    # valid['x'] = transform(valid['x'])
    # test['x'] = transform(test['x'])

    classifier = CNNModel(eta=args.eta, eta_decay=args.eta_decay,
                          n_iter=args.n_iter, batch_size=args.batch_size)

    # classifier.load(args.model)

    classifier.fit(train['x'], train['y'], valid, 0, 500)
    
    valid['y_'] = classifier.predict(valid['x'])
    print('accuracy:', accuracy(valid['y_'], valid['y']))
    
    for i in range(10):
        # get confidence of prediction on test
        test['y_prob'] = classifier.predict_prob(test['x'])
        test['y_'] = np.argmax(test['y_prob'], axis=1)
        test['confidence'] = np.max(test['y_prob'], axis=1)

        # sort test data by confidence
        inds = np.argsort(test['confidence'])
        test['x'] = test['x'][inds]
        test['y_'] = test['y_'][inds]

        # pdb.set_trace()
        # add those of top 10 confidence to train
        add_size = test['x'].shape[0] // 10
        train['x'] = np.concatenate([train['x'], test['x'][add_size:]],
                                    axis=0)
        train['y'] = np.concatenate([train['y'], test['y_'][add_size:]],
                                    axis=0)

        # remove added from test data
        test['x'] = test['x'][:add_size]

        # shuffle training data again
        inds = np.arange(train['x'].shape[0])
        np.random.shuffle(inds)
        train['x'] = train['x'][inds]
        train['y'] = train['y'][inds]

        classifier.fit(train['x'], train['y'], valid, 1000 + i * 100)
        classifier.dump_history(1000 + i * 100)


    # classifier.save(args.n_iter)
    classifier.dump_history()
    # train['y_'] = classifier.predict(train['x'])
    # print('accuracy train:', accuracy(train['y_'], train['y']))

    # valid['y_'] = classifier.predict(valid['x'])
    # print('accuracy valid:', accuracy(valid['y_'], valid['y']))

    # pdb.set_trace()
    # test['y_'] = classifier.predict(test['x'])
    # write_csv(test['y_'], args.out)


if __name__ == '__main__':
    try:
        main()
    except:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
