import numpy as np
import pdb
import math

class LogisticRegression:
    """ 
    Parameters
    ----------
    alpha : float
        Coefficient of regularization term.

    eta: float
        Learning rate.
    """
    def __init__(self, alpha=1e-4, eta=1e-5, n_iter=100, batch_size=10, verbose=0):
        self.alpha = alpha
        self.eta = eta
        self.n_iter = n_iter
        self.verbose = verbose
        self.batch_size = batch_size
        
        
    def fit(self, X, y):
        # padding
        n_rows = X.shape[0]
        X = np.concatenate((X, np.ones((n_rows, 1))), axis=1)

        self.w = np.zeros((X.shape[1], 1))
        if self.batch_size > 0:
            batch_size = self.batch_size            
        else:
            batch_size = n_rows
            
        for i in range(self.n_iter):
            # shuffle
            inds = np.arange(n_rows)
            np.random.shuffle(inds)
            X = X[inds]
            y = y[inds]
            
            for b in range(0, n_rows, batch_size):
                batch_X = X[b: b + batch_size]
                batch_y = y[b: b + batch_size]
                z = - np.dot(batch_X, self.w)
                sigz = 1 / (1 + np.exp(z))
                gradient = np.dot(batch_X.T, sigz - batch_y)
                gradient = np.average(gradient, axis=1).reshape(-1, 1)
                gradient += self.alpha * self.w

                if np.any(np.isnan(gradient)):
                    pdb.set_trace()

                self.w -= self.eta * gradient

                if self.verbose >= 2:
                    l = np.log(1 + sigz) + (1 - batch_y) * z                    
                    print("likelihood =", l.sum(), l.shape)
                    print("|gradient| =", np.linalg.norm(gradient))
                                    
            if self.verbose:
                print("Iter %d" % i)

        if self.verbose >= 2:
            print('w', self.w)
                

    def predict(self, X):
        # padding
        n_rows = X.shape[0]
        X = np.concatenate((X, np.ones((n_rows, 1))), axis=1)

        z = - np.dot(X, self.w)
        sigz = 1 / (1 + np.exp(z))
        y = np.zeros((n_rows, 1))
        y[np.where(sigz > 0.5)] = 1
        return y


    def predict_proba(self, X):
        # padding
        n_rows = X.shape[0]
        X = np.concatenate((X, np.ones((n_rows, 1))), axis=1)

        z = - np.dot(X, self.w)
        sigz = 1 / (1 + np.exp(z))
        return sigz

    

class ProbabilisticGenenerative:
    def __init__(self):
        pass

    
    def fit(self, X, y):
        n_rows = X.shape[0]
        X1 = X[np.where(y == 1)[0],:]
        X0 = X[np.where(y == 0)[0],:]
        covar = np.dot(X.T, X) / n_rows
        covar_inv = np.linalg.inv(covar)
        mu1 = np.average(X1, axis=0).reshape(-1, 1)
        mu0 = np.average(X0, axis=0).reshape(-1, 1)
        self.w = np.dot((mu1 - mu0).T, covar_inv).T
        self.b = - 0.5 * np.dot(np.dot(mu1.T, covar_inv), mu1) 
        self.b +=  0.5 * np.dot(np.dot(mu0.T, covar_inv), mu0)
        self.b += math.log(X1.shape[0] / X0.shape[0])

        
    def predict(self, X):
        z = np.dot(X, self.w) + self.b
        sigz = 1 / (1 + np.exp(z))
        y = np.zeros(X.shape[0])
        y[np.where(sigz > 0.5)[0]] = 1
        return y
