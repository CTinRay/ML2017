import pickle
import numpy as np
import pandas as pd
import argparse
# import pdb
# import sys
# import traceback
from kr import CNNModel


def write_csv(y, filename):
    f = open(filename, 'w')
    f.write('id,label\n')
    for i in range(y.shape[0]):
        f.write('%d,%d\n' % (i, y[i]))

    f.close()


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


def get_test(csv):
    xs = []
    with open(csv) as f:
        f.readline()
        for l in f:
            cols = l.split(',')
            xs.append(list(map(int, cols[1].split())))

    return {'x': np.array(xs)}


def main():
    parser = argparse.ArgumentParser(description='ML HW3')
    parser.add_argument('model', type=str, help='model name')
    parser.add_argument('test', type=str, help='test.csv')
    parser.add_argument('out', type=str, help='outcome')
    parser.add_argument('--mean_max', type=str, help='mean_max',
                        default='mean_max.pickle')
    args = parser.parse_args()

    # train = get_XY(args.train)
    test = get_test(args.test)

    # transform
    # train['x'] = transform(train['x'])
    # valid['x'] = transform(valid['x'])
    # test['x'] = transform(test['x'])

    # calculate mean, std
    mean_max = pickle.load(open(args.mean_max, 'rb'))
    mean = mean_max['mean']
    abs_max = mean_max['max']

    # normalize
    # train['x'] = (train['x'] - mean) / abs_max
    test['x'] = (test['x'] - mean) / abs_max

    classifier = CNNModel()

    classifier.load(args.model)
    test['y_'] = classifier.predict(test['x'])
    write_csv(test['y_'], args.out)


if __name__ == '__main__':
    try:
        main()
    except:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
