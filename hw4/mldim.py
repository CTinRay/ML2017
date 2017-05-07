"""
Reference: https://www.stat.berkeley.edu/~bickel/mldim.pdf
"""
import numpy as np
import multiprocessing
import argparse
from sklearn.neighbors import NearestNeighbors
from gen import gen
import pdb
import sys
import traceback


def estimate_id(data, k1, k2, n_samples):
    n_cpu = multiprocessing.cpu_count()
    nbrs = NearestNeighbors(n_neighbors=k2 + 1,
                            algorithm='ball_tree',
                            n_jobs=n_cpu).fit(data)

    ind_samples = np.random.choice(data.shape[0],
                                   n_samples, False)
    samples = data[ind_samples]

    distances, indices = nbrs.kneighbors(samples)

    # print('distances =', np.mean(distances, axis=0))
    mks = np.zeros(k2 - k1 + 1)
    for k in range(k1, k2 + 1):
        mk = 1 / (np.log(distances[:, k])
                  - np.mean(np.log(distances[:, 1:k]), axis=1))
        mks[k - k1] = np.mean(mk, axis=0)

    # pdb.set_trace()
    return np.mean(mks)


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--n_test', type=int, default=200)
    parser.add_argument('--k1', type=int, default=3)
    parser.add_argument('--k2', type=int, default=10)
    parser.add_argument('--n_samples', type=int, default=100)
    args = parser.parse_args()

    print('using argument k1 = %d, k2 = %d, n_samples = %d'
          % (args.k1, args.k2, args.n_samples))
    
    abs_errs = np.zeros(args.n_test)    
    for i in range(args.n_test):
        dim = np.random.randint(1, 61)
        N = np.random.randint(10000, 100000)
        data = gen(dim, N)
        # data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)
        # print('i = %d, N = %d, dim = %f'
        #       % (i, N, dim))
        eid = estimate_id(data, args.k1, args.k2, args.n_samples)
        abs_err = np.abs(np.log(dim) - np.log(eid))
        abs_errs[i] = abs_err
        print('i = %d, N = %d, dim = %f, eid = %f, abs_err = %f, mean_abs_err = %f, std = %f'
              % (i, N, dim, eid, abs_err, np.mean(abs_errs[:i + 1]), np.std(abs_errs[:i + 1])))

    print('mean absolute error =', np.mean(abs_errs))


if __name__ == '__main__':
    try:
        main()
    except:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
