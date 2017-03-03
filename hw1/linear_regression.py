import argparse
import numpy as np
import pandas as pd

class LinearRegressor:
    def __init__(self, l=0.1, rate=1e-5, stop=0.7):
        self.l = l
        self.rate = rate
        self.stop = stop
        pass

    def fit(self, x, y):
        self.w = np.zeros((x.shape[1] + 1, 1))
        x = np.append(x, np.ones((x.shape[0], 1)), axis=1)
        y = y.reshape(-1, 1)
        print(x)
        np.dot(x, self.w)
        dw = 2 * np.dot(x.T, np.dot(x, self.w) - y) / x.shape[0]
        while np.linalg.norm(dw) > self.stop:
            self.w -= dw * self.rate
            print('err:', np.linalg.norm(np.dot(x, self.w) - y) / x.shape[0])
            print('|dw|:', np.linalg.norm(dw))
            # print('dw:' ,dw)
            dw = 2 * np.dot(x.T, (np.dot(x, self.w) - y)) / x.shape[0]
            # print(dw)
            
    def predict(self, x):
        x = np.append(x, np.ones((x.shape[0], 1)), axis=1)
        return np.dot(x , self.w)

def get_train_data(csv):
    data = pd.read_csv(csv, encoding='big5', index_col=[0,2], parse_dates=True)
    data = data.replace('NR', 0)
    del data['測站']
    data = data.unstack(level=0).swaplevel(0, 1, 1).sort_index(1, 0)
    data = data.convert_objects(convert_numeric=True)
    x = data.drop('PM2.5').as_matrix().T    
    y = data.loc['PM2.5'].as_matrix()
    # print('x:', x)
    # print('y:', y)
    return x, y


def split_valid(x, y, ratio):
    inds = np.arange(x.shape[0])
    np.random.shuffle(inds)
    x[inds] = x
    y[inds] = y
    n_valid = int(x.shape[0] * ratio)
    train = {'x': x[n_valid:], 'y': y[n_valid:]}
    valid = {'x': x[:n_valid], 'y': y[:n_valid]}
    return train, valid


def rmse(y_, y):
    return (np.average((y_ - y) ** 2)) ** 0.5


def main():
    parser = argparse.ArgumentParser(description='ML HW1')    
    parser.add_argument('train', type=str, help='training data')
    parser.add_argument('--valid_ratio', type=float, help='ratio of validation data', default=0.2)
    args = parser.parse_args()
    x, y = get_train_data(args.train)
    train, valid = split_valid(x, y, args.valid_ratio)
    regressor = LinearRegressor()
    regressor.fit(train['x'], train['y'])
    y_ = regressor.predict(valid['x'])
    print('valid rmse:', rmse(y_, valid['y']))

    
if __name__ == '__main__':
    main()
