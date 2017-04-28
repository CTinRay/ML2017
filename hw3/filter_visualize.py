import argparse
import numpy as np
import os
from keras import backend as K
from keras.models import load_model
import pdb
import sys
import traceback
import matplotlib.pyplot as plt


# dimensions of the generated pictures for each filter.
img_width = 48
img_height = 48


def get_test(csv):
    xs = []
    with open(csv) as f:
        f.readline()
        for l in f:
            cols = l.split(',')
            xs.append(list(map(int, cols[1].split())))

    return {'x': np.array(xs)}


def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-7)


def main():
    parser = argparse.ArgumentParser(prog='',
                                     description='')
    parser.add_argument('path', type=str)
    parser.add_argument('test', type=str)
    parser.add_argument('out', type=str)
    parser.add_argument('--layer', type=int, default=1)
    parser.add_argument('--n_iter', type=int, default=100)
    args = parser.parse_args()

    model = load_model(os.path.join(args.path, 'model.h5'))
    layer_dict = dict([layer.name, layer] for layer in model.layers)
    layer = layer_dict["conv2d_%d" % args.layer]

    n_filters = layer.filters

    filter_imgs = [[] for i in range(n_filters)]
    for ind_filter in range(n_filters):
        filter_imgs[ind_filter] = np.random.random((1, 48, 48, 1))
        activation = K.mean(layer.output[:, :, :, ind_filter])
        grads = normalize(K.gradients(activation, model.input)[0])
        iterate = K.function([model.input, K.learning_phase()],
                             [activation, grads])

        print('processing filter %d' % ind_filter)
        for i in range(args.n_iter):
            act, g = iterate([filter_imgs[ind_filter], 0])
            filter_imgs[ind_filter] += g

    fig = plt.figure(figsize=(14, 2 * ind_filter / 16))
    for ind_filter in range(n_filters):
        ax = fig.add_subplot(n_filters / 16 + 1, 16, ind_filter + 1)
        ax.imshow(filter_imgs[ind_filter].reshape(48, 48),
                  cmap='BuGn')
        plt.xticks(np.array([]))
        plt.yticks(np.array([]))
        plt.xlabel('filter %d' % ind_filter)
        plt.tight_layout()

    fig.suptitle('Filters of layer %d' % args.layer)
    fig.savefig(args.out)


if __name__ == '__main__':
    try:
        main()
    except:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
