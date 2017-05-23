import numpy as np
import pandas as pd
import argparse
import pickle
from utils import *
from rnn import TextClassifier
import os
import pdb
import sys
import traceback
from sklearn.metrics import f1_score


def main():
    parser = argparse.ArgumentParser(description='ML HW4')
    parser.add_argument('data', type=str, help='data.pickle')
    parser.add_argument('embedding', type=str, help='embedding.pickle')
    parser.add_argument('model', type=str, help='model to save')
    parser.add_argument('--preprocess_args', type=str,
                        default='preprocess_args.pickle',
                        help='pickle to store preprocess arguments')
    parser.add_argument('--valid_ratio', type=float,
                        help='ratio of validation data.', default=0.1)
    parser.add_argument('--glove', type=str,
                        help='glove wordvec file',
                        default='./data/glove.6B.100d.txt')
    parser.add_argument('--n_iters', type=int, default=100,
                        help='number of epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='batch size')
    args = parser.parse_args()

    # load preprocess args
    with open(args.preprocess_args, 'rb') as f:
        preprocess_args = pickle.load(f)
        tokenizer = preprocess_args['tokenizer']

    with open(args.embedding, 'rb') as f:
        embedding_matrix = pickle.load(f)

    # load data
    with open(args.data, 'rb') as f:
        data = pickle.load(f)
        train = data['train']
        valid = data['valid']

    # start training
    classifier = TextClassifier(len(tokenizer.word_index),
                                embedding_matrix,
                                n_iters=args.n_iters,
                                lr=args.lr, batch_size=args.batch_size)
    classifier.fit(train['x'], train['y'], valid)
    valid['y_'] = classifier.predict(valid['x'], 0.4)
    print('f1 score =',
          f1_score(valid['y'], valid['y_'],
                   average='samples'))
    classifier.save(args.model)


if __name__ == '__main__':
    try:
        main()
    except:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
