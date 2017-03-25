import numpy as np
import pandas as pd
import argparse 
from linear_model import LogisticRegression
import pdb
import sys
import traceback

def get_X(csv):
    data = pd.read_csv(csv)
    xs = data.as_matrix()
    return xs


def get_Y(y):
    f = open(y)
    ys = list(map(int, f.read().split()))
    ys = np.array(ys).reshape(-1, 1)
    return ys


def split_valid(raw, valid_ratio):
    n_rows = raw['x'].shape[0]

    # shuffle data
    inds = np.arange(n_rows)
    np.random.shuffle(inds)
    raw['x'] = raw['x'][inds]
    raw['y'] = raw['y'][inds]

    # split data
    n_valid = int(n_rows * valid_ratio)
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

    
def main():
    parser = argparse.ArgumentParser(description='ML HW2')    
    parser.add_argument('x_train', type=str, help='X_train')
    parser.add_argument('y_train', type=str, help='Y_train')
    parser.add_argument('x_test', type=str, help='testing data')
    parser.add_argument('out', type=str, help='outcome')
    parser.add_argument('--valid_ratio', type=float, help='ratio of validation data', default=0.2)
    parser.add_argument('--n_iter', type=int, help='ratio of validation data', default=100)
    parser.add_argument('--eta', type=float, help='learning rate', default=1e-5)
    parser.add_argument('--alpha', type=float, help='regularization', default=1e-4)
    parser.add_argument('--verbose', type=int, help='verbose', default=0)
    parser.add_argument('--batch_size', type=int, help='batch size', default=10)    
    args = parser.parse_args()
    raw_train = {'x': get_X(args.x_train), 'y': get_Y(args.y_train)}
    train, valid = split_valid(raw_train, args.valid_ratio)    
    test = {'x': get_X(args.x_test)}

    # calculate mean, std
    mean = np.average(train['x'], axis=0)
    std = np.std(train['x'], axis=0) + 1e-10
        
    # normalize
    train['x'] = (train['x'] - mean) / std
    valid['x'] = (valid['x'] - mean) / std
    test['x'] = (test['x'] - mean) / std

    
    regressor = LogisticRegression(alpha=args.alpha, eta=args.eta,
                                       n_iter=args.n_iter, batch_size=args.batch_size,
                                       verbose=args.verbose)
    regressor.fit(train['x'], train['y'])
    train['y_'] = regressor.predict(train['x'])
    print('accuracy train:', accuracy(train['y_'], train['y']))

    valid['y_'] = regressor.predict(valid['x'])
    print('accuracy valid:', accuracy(valid['y_'], valid['y']))

    test['y_'] = regressor.predict(test['x'])
    write_csv(test['y_'], args.out)
    
    
if __name__ == '__main__':
    try:
        main()
    except:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
