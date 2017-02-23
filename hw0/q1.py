import numpy as np
import argparse


def main():
    parser = argparse.ArgumentParser(description='ML HW0')
    parser.add_argument('mA', type=str, help='file1')
    parser.add_argument('mB', type=str, help='file2')
    parser.add_argument('out', type=str, help='out')
    args = parser.parse_args()

    mA = np.array(list(map(int, open(args.mA).read().strip().split(',')))).reshape(1, -1)
    mB = np.array([list(map(int, line.strip().split(','))) for line in open(args.mB).read().strip().split('\n')])
    mC = sorted(list((np.dot(mA, mB)).reshape(1, -1)[0]))
    open(args.out, 'w').write('\n'.join(map(str, mC)))

    
if __name__ == '__main__':
    main()
    
