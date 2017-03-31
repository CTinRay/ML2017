import numpy as np
import copy

class TreeClassifier:
    def __init__(self, base_estimator):
        self.base_estimator = base_estimator

    def fit(self, X, y):
        self.estimators = [{}] * 3
        self.estimators[0] = copy.deepcopy(self.base_estimator)
        self.estimators[1] = copy.deepcopy(self.base_estimator)
        self.estimators[2] = copy.deepcopy(self.base_estimator)

        self.estimators[0].fit(X, y)
        predict0 = self.estimators[0].predict(X)

        ind00 = np.where(predict0 == 0)[0]
        ind01 = np.where(predict0 == 1)[0]
        self.estimators[1].fit(X[ind00], y[ind00])
        self.estimators[2].fit(X[ind01], y[ind01])


    def predict(self, X):
        predict = self.estimators[0].predict(X)
        
        ind00 = np.where(predict == 0)[0]
        ind01 = np.where(predict == 1)[0]
        
        predict[ind00] = self.estimators[1].predict(X[ind00])
        predict[ind01] = self.estimators[2].predict(X[ind01])

        return predict


    
