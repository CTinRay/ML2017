from mean_d import estimate_id
import numpy as np
import argparse
import pdb
import sys
import traceback


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('npz', type=str)
    parser.add_argument('out', type=str)
    args = parser.parse_args()

    with open(args.out, 'w') as f:
        f.write('SetId,LogDim\n')
        data = np.load(args.npz)
        for i in range(200):
            x = data[str(i)]
            print('estimating %d' % i)
            eid = estimate_id(x)
            f.write('%d,%f\n' % (i, np.log(eid)))


if __name__ == '__main__':
    try:
        main()
    except:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
