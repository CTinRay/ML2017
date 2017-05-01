#!/usr/bin/env python
# -- coding: utf-8 --

import os
import matplotlib.pyplot as plt
import argparse
from keras.models import load_model
from keras import backend as K
# from utils import *
# from marcos import *
import numpy as np


def get_test(csv):
    xs = []
    cnt = 0
    with open(csv) as f:
        f.readline()
        for l in f:
            cols = l.split(',')
            xs.append(list(map(int, cols[1].split())))

    return {'x': np.array(xs)}


def main():
    parser = argparse.ArgumentParser(prog='Visualize output of a layer.',
                                     description='')
    parser.add_argument('path', type=str, help='Path of the model.')
    parser.add_argument('test', type=str, help='Path of the test data.')
    parser.add_argument('out', type=str, help='Path of the output picture.')
    parser.add_argument('--id', type=int, default=0,
                        help='Id of image in test data to use.')
    parser.add_argument('--layer', type=int, default=1,
                        help='Layer of which output will be visualized.')
    args = parser.parse_args()

    emotion_classifier = load_model(os.path.join(args.path, 'model.h5'))

    layer_dict = dict([layer.name, layer]
                      for layer in emotion_classifier.layers[:])

    input_img = emotion_classifier.input
    name_ls = ["conv2d_%d" % args.layer]
    collect_layers = [K.function([input_img, K.learning_phase()], [
                                 layer_dict[name].output]) for name in name_ls]

    test = get_test(args.test)['x']
    test = (test - np.mean(test, axis=0)) / np.max(np.abs(test))
    photo = test[args.id].reshape(1, 48, 48, 1)
    for cnt, fn in enumerate(collect_layers):
        im = fn([photo, 0])  # get the output of that layer
        fig = plt.figure(figsize=(14, 8))
        nb_filter = im[0].shape[3]
        for i in range(nb_filter):
            ax = fig.add_subplot(nb_filter / 16, 16, i + 1)
            ax.imshow(im[0][0, :, :, i], cmap='BuGn')
            plt.xticks(np.array([]))
            plt.yticks(np.array([]))
            plt.tight_layout()
        fig.suptitle('Output of layer{} (Given image{})'.format(cnt, 17))
        fig.savefig(args.out, dpi=300)


main()
