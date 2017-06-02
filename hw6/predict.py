import pdb
import sys
import traceback
import argparse
from utils import read_test, write_predict
from mf import MF


def main():
    parser = argparse.ArgumentParser(description='ML HW6')
    parser.add_argument('test', type=str, help='test.csv')
    parser.add_argument('model', type=str, help='model to save')
    parser.add_argument('out', type=str, help='out.csv')
    args = parser.parse_args()

    test = read_test(args.test)

    # Fix index offset
    test['x'][:, 0] -= 1
    test['x'][:, 1] -= 1

    # start training
    mf = MF()
    mf.load(args.model)
    test['y_'] = mf.predict_raw(test['x'])
    write_predict(args.out, test['y_'])


if __name__ == '__main__':
    try:
        main()
    except:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
