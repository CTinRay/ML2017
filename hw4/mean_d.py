import numpy as np
import multiprocessing
import argparse
from sklearn.neighbors import NearestNeighbors
# from gen import gen
# import pdb
# import sys
# import traceback


mean_d_k1 = [0.03154246, 1.63759781, 7.06542885, 15.25761895, 23.85304995, 33.46429235, 44.75832246, 50.34473004, 61.10834587, 68.5134748, 78.59144788, 85.12539014, 90.0070838, 95.07347339, 101.6702367, 106.97151439, 112.64446119, 119.74353551, 122.79621082, 125.82636676, 129.56227525, 132.24704497, 137.88769494, 146.85696729, 146.14020392, 148.7139694, 153.68614343, 159.75358424, 161.10616516, 168.38942663, 167.10334305, 172.32425614, 173.67191126, 175.22464796, 174.15414996, 188.56208549, 185.83475252, 190.25582291, 191.69992115, 194.06569949, 200.5492479, 199.99238781, 198.97477479, 206.77735276, 207.58104413, 204.74021747, 215.51551191, 218.76286685, 216.54216127, 217.706258, 218.13942039, 222.11372363, 228.0045562, 224.84555733, 228.04948842, 232.94908247, 234.55664031, 241.87698998, 233.9463051, 235.75382078]


def estimate_id(data, n_samples=100):
    n_cpu = multiprocessing.cpu_count()
    nbrs = NearestNeighbors(n_neighbors=2,
                            algorithm='ball_tree',
                            n_jobs=n_cpu).fit(data)

    ind_samples = np.random.choice(data.shape[0],
                                   n_samples, False)
    samples = data[ind_samples]

    distances, indices = nbrs.kneighbors(samples)

    mean_d = np.mean(distances[:,1])
    # print(mean_d)
    
    if mean_d < mean_d_k1[0]:
        return 1
    
    for i in range(59):
        if mean_d > mean_d_k1[i] and mean_d < mean_d_k1[i + 1]:
            break

    return i + 2


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--n_test', type=int, default=200)
    # parser.add_argument('--k1', type=int, default=3)
    # parser.add_argument('--k2', type=int, default=10)
    # parser.add_argument('--n_samples', type=int, default=100)
    args = parser.parse_args()

    # print('using argument n_samples = %d'
    #       % (args.n_samples))
    
    abs_errs = np.zeros(args.n_test)    
    for i in range(args.n_test):
        dim = np.random.randint(1, 61)
        N = np.random.randint(10000, 100000)
        data = gen(dim, N)
        # data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)
        # print('i = %d, N = %d, dim = %f'
        #       % (i, N, dim))
        eid = estimate_id(data)
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
