import numpy as np
import pandas as pd
import argparse
import pickle
from utils import *
from rnn import TextClassifier


def main():
    parser = argparse.ArgumentParser(description='ML HW4')
    parser.add_argument('train', type=str, help='train.csv')
    parser.add_argument('test', type=str, help='train.csv')
    parser.add_argument('out', type=str, help='outcome')
    parser.add_argument('--preprocess_args', type=str,
                        default='preprocss_args.pickle',
                        help='pickle to store preprocess arguments')
    parser.add_argument('--valid_ratio', type=float,
                        help='ratio of validation data.', default=float)
    parser.add_argument('--glove', type=str,
                        help='glove wordvec file',
                        default='./data/glove.6B.100d.txt')
    args = parser.parse_args()

    # read data
    train_data = read_train(args.train)
    test_data = read_test(args.test)

    # process tags
    tag_table = make_tag_table(train_data['tags'])
    encode_tags(train, tag_table)

    # preprocess text
    tokenizer = make_tokenizer(train_data['text'] + test_data['text'])
    encode_text(train, tokenizer)

    # load GloVe and make embedding matrix
    glove_dict = load_glove(args.glove)
    embedding_matrix = make_embedding_matrix(tokenizer, golve_dict)

    # save preprocess args
    with open(args.preprocess_args, 'wb') as f:
        preprocess_args = {'tag_table': tag_table,
                           'tokenizer': tokenizer,
                           'max_len': train['xs'].shape[1]}
        pickle.dump(preprocess_args, f)

    # split data
    train, valid = split_valid(train_data, args.valid_ratio)

    # start training
    classifier = TextClassifier(embedding_matrix=embedding_matrix)
    classifier.fit(train['text'], train['ys'], valid)

    
