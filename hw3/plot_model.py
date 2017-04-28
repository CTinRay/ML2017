import os
# from termcolor import colored, cprint
import argparse
from keras.utils import plot_model
from keras.models import load_model
import h5py


base_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
exp_dir = os.path.join(base_dir, 'exp')


def main():
    parser = argparse.ArgumentParser(prog='plot_model.py',
                                     description='Plot the model.')
    parser.add_argument('path', type=str)
    parser.add_argument('out', type=str)
    args = parser.parse_args()

    emotion_classifier = load_model(os.path.join(args.path, 'model.h5'))
    emotion_classifier.summary()
    plot_model(emotion_classifier, to_file=args.out)


if __name__ == '__main__':
    main()
