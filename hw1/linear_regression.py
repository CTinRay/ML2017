import argparse
import numpy as np
import pandas as pd
import math

def rmse(y_, y):
    return (np.average((y_ - y) ** 2)) ** 0.5


class LinearRegressor:
    def __init__(self, l=0.1, rate=1e-5, stop=0.7):
        self.l = l
        self.rate = rate
        self.stop = stop

    def fit_analytics(self, x, y):
        x = np.append(x, np.ones((x.shape[0], 1)), axis=1)
        self.w = np.dot(np.dot(np.linalg.inv(np.dot(x.T, x) + np.identity(x.shape[1]) * self.l), x.T), y)
        # print(self.w)
        
    def fit(self, x, y, x_, y_):        
        x = np.append(x, np.ones((x.shape[0], 1)), axis=1)
        # y = y.reshape(-1, 1)
        self.w = np.random.random((x.shape[1],))
        # self.w = np.dot(np.dot(np.linalg.inv(np.dot(x.T, x)), x.T), y)
        dw = 2 * np.dot(x.T, np.dot(x, self.w) - y) / x.shape[0] + 2 * self.l * self.w
        while np.linalg.norm(dw) > self.stop:
            self.w -= dw * self.rate
            predict = np.dot(x, self.w)
            # print('err:', np.linalg.norm(predict - y) / x.shape[0], rmse(predict, y))
            print('err valid:', rmse(self.predict(x_), y_))
            print('|dw|:', np.linalg.norm(dw))
            # print('dw:' ,dw)
            dw = 2 * np.dot(x.T, (np.dot(x, self.w) - y)) / x.shape[0] + self.l * self.w
            # print(dw)
            
    def predict(self, x):
        x = np.append(x, np.ones((x.shape[0], 1)), axis=1)
        return np.dot(x , self.w)



def get_raw(csv):
    data = pd.read_csv(csv, encoding='big5', index_col=[0,2], parse_dates=True)
    data = data.replace('NR', 0)
    del data['測站']
    data = data.unstack(level=0).swaplevel(0, 1, 1).sort_index(1, 0)
    data = data.apply(pd.to_numeric)
    data.to_csv('tmp.csv')
    pm25 = data.loc['PM2.5'].as_matrix()
    # data = data.loc['PM2.5'].as_matrix().reshape(-1, 1)
    data = data.as_matrix().T

    return pm25, data


def split_valid(pm25, data, ratio):
    print('raw data.shape:', data.shape)
    days = int(20 * 24 * (1 - ratio))
    train_inds = np.where(np.arange(data.shape[0]) % (20 * 24) <= days)[0]
    valid_inds = np.where(np.arange(data.shape[0]) % (20 * 24) > days)[0]
    train = {'x': data[train_inds], 'y': pm25[train_inds]}
    valid = {'x': data[valid_inds], 'y': pm25[valid_inds]}
    return train, valid


def scan(n_prev, data):
    n_hours = int(data['x'].shape[0] / 12)
    x = np.zeros((12 * (n_hours - n_prev), data['x'].shape[1] * n_prev))
    y = np.zeros(12 * (n_hours - n_prev))
    for m in range(12):
        for h in range(0, n_hours - n_prev):
            start = m * (n_hours - n_prev) + h
            end = m * (n_hours - n_prev) + h + n_prev
            x[start] = data['x'][start:end].reshape(1, -1)
            y[start] = data['y'][end]            
            
    # print('x', x)
    return {'x': x, 'y': y}


def angle2abs(angle):
    x = np.cos(2 * math.pi * angle / 360)
    y = np.sin(2 * math.pi * angle / 360)
    return x, y


def transform(x):
    n_features = 18
    ind_base = n_features * np.arange(int(x.shape[1] / n_features))
    # std_pm25 = np.std(x[:,ind_pm25s], axis=1).reshape(-1, 1)
    # x_ = np.append(x, std_pm25, axis=1)
    # ind_wind = np.append(ind_base + 15, ind_base + 16)
    ind_wind = ind_base + 15
    xs, ys = angle2abs(x[:,ind_wind])
    x_ = np.array(x)
    x_ = np.concatenate((x, xs, ys), axis=1)
    ind_wind = np.append(ind_base + 15, ind_base + 16)    
    np.delete(x_, ind_wind, axis=1)
    
    # return x[:,np.concatenate((ind_base + 9, ind_base + 8, ind_base + 10))]
    return x_

def get_test_data(csv):
    data = pd.read_csv(csv, encoding='big5', parse_dates=True, header=None)
    data = data.replace('NR', 0)
    data = data.replace('id_', '', regex=True)
    data[0] = data[0].apply(pd.to_numeric)
    data = data.set_index([0, 1])    
    data = data.apply(pd.to_numeric)
    data = data.unstack(level=0).swaplevel(0, 1, 1).sort_index(1, 0)
    x = data.as_matrix().T
    x = x.reshape(int(x.shape[0] / 9), -1)
    return x
        

def write_csv(path, data):
    f = open(path, 'w')
    f.write('id,value\n')
    for i in range(data.shape[0]):
        f.write('id_%d,%f\n' % (i, data[i]))
    
    f.close()

        
def main():
    parser = argparse.ArgumentParser(description='ML HW1')    
    parser.add_argument('train', type=str, help='training data')
    parser.add_argument('test', type=str, help='testing data')
    parser.add_argument('out', type=str, help='outcome')
    parser.add_argument('--valid_ratio', type=float, help='ratio of validation data', default=0.2)
    parser.add_argument('--l', type=float, help='ratio of validation data', default=0.1)
    parser.add_argument('--stop', type=float, help='ratio of validation data', default=1)
    parser.add_argument('--rate', type=float, help='ratio of validation data', default=1e-5)
    parser.add_argument('--n_prev', type=int, help='', default=9)
    args = parser.parse_args()
    
    pm25, raw_data = get_raw(args.train)
    # raw_data = (raw_data - np.average(raw_data, axis=0)) / np.std(raw_data, axis=0)
    train_raw, valid_raw = split_valid(pm25, raw_data, args.valid_ratio)
    train = scan(9, train_raw)
    valid = scan(9, valid_raw)
    # valid['x'] = transform(valid['x'])
    n_features = 18

    regressors = [0] * 9
    train['ys'] = np.zeros((train['x'].shape[0], 9))
    for n_prev in range(1, 10):
        print('n_prev:', n_prev)
        train_ = scan(n_prev, train_raw)
        regressors[n_prev - 1] = LinearRegressor(l=args.l, stop=args.stop, rate=args.rate)
        regressors[n_prev - 1].fit_analytics(train_['x'], train_['y'])
        train['ys'][:,n_prev - 1] = regressors[n_prev - 1].predict(train['x'][:,-n_prev * n_features:])
    
    regressor = LinearRegressor(l=args.l, stop=args.stop, rate=args.rate)
    # regressor.fit_analytics(train['ys'], train['y'])
    # print('w', regressor.w)
    # train['y_'] = regressor.predict(train['ys'])
    # print('ein =', rmse(train['y_'], train['y']))
    
    if args.valid_ratio > 0:
        valid['ys'] = np.zeros((valid['x'].shape[0], 9))
        for n_prev in range(1, 10):
            valid['ys'][:,n_prev - 1] = regressors[n_prev - 1].predict(valid['x'][:,-n_prev * n_features:])

        regressor.fit_analytics(valid['ys'], valid['y'])
        print('valid rmse:', rmse(regressor.predict(valid['ys']), valid['y']))

        
    test_x = get_test_data(args.test)
    test_y_ = np.zeros((test_x.shape[0], 9))
    for n_prev in range(1, 10):
        test_y_[:,n_prev - 1] = regressors[n_prev - 1].predict(test_x[:,-n_prev * n_features:])    
        
    test_y = regressor.predict(test_y_)
    write_csv(args.out, test_y)
    
if __name__ == '__main__':
    main()
