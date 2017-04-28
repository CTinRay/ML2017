import argparse
import matplotlib.pyplot as plt
import os
import numpy as np


def load_log(filename):
    with open(filename) as f:
        content = f.read()
        content = list(map(float, content.split()))
        return np.array(content)


def main():
    parser = argparse.ArgumentParser(prog='plot_model.py',
                                     description='Plot the model.')
    parser.add_argument('path', type=str)
    args = parser.parse_args()

    # plot loss
    plt.figure(1)
    plt.xlabel('epochs', fontsize=18)
    plt.ylabel('loss', fontsize=16)
    arr_train_loss = load_log(os.path.join(args.path, 'train_loss'))
    arr_valid_loss = load_log(os.path.join(args.path, 'valid_loss'))
    plt.plot(np.arange(arr_train_loss.shape[0]),
             arr_train_loss,
             label='train')
    plt.plot(np.arange(arr_valid_loss.shape[0]),
             arr_valid_loss,
             label='valid')
    plt.legend()
    plt.savefig(os.path.join(args.path, 'loss.png'), dpi=300)

    # plot accuracy
    plt.figure(2)
    plt.xlabel('epochs', fontsize=18)
    plt.ylabel('accuracy', fontsize=16)
    arr_train_loss = load_log(os.path.join(args.path, 'train_accuracy'))
    arr_valid_loss = load_log(os.path.join(args.path, 'valid_accuracy'))
    plt.plot(np.arange(arr_train_loss.shape[0]),
             arr_train_loss,
             label='train')
    plt.plot(np.arange(arr_valid_loss.shape[0]),
             arr_valid_loss,
             label='valid')
    plt.legend()
    plt.savefig(os.path.join(args.path, 'accuracy.png'), dpi=300)


if __name__ == '__main__':
    main()
