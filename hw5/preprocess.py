from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
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
    parser.add_argument('train', type=str, help='train.csv')
    parser.add_argument('test', type=str, help='train.csv')
    parser.add_argument('data', type=str, help='data.pickle')
    parser.add_argument('embedding_matrix', type=str, help='embedding.pickle')
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

    # read data
    train_data = read_train(args.train)
    test_data = read_test(args.test)

    # process tags
    tag_table = make_tag_table(train_data['tags'])
    encode_tags(train_data, tag_table)

    # preprocess text
    tokenizer = make_tokenizer(train_data['text'] + test_data['text'])
    encode_text(train_data, tokenizer)

    # preprocess text
    vectorizer = CountVectorizer(min_df=1, ngram_range=(1, 3),
                                 stop_words='english', max_features=40000)
    vectorizer.fit(train_data['text'] + test_data['text'])
    train_data['count'] = vectorizer.transform(train_data['text']).toarray()
    test_data['count'] = vectorizer.transform(test_data['text']).toarray()

    transformer = TfidfTransformer(sublinear_tf=True)
    transformer.fit(np.concatenate(
        [train_data['count'], test_data['count']], axis=0))
    train_data['tfidf'] = transformer.transform(train_data['count']).toarray()
    test_data['tfidf'] = transformer.transform(test_data['count']).toarray()

    # load GloVe and make embedding matrix
    glove_dict = load_glove(args.glove)
    embedding_matrix = make_embedding_matrix(tokenizer, glove_dict)

    # split data
    train, valid, rand_indices = split_valid(train_data, args.valid_ratio)

    # save preprocess args
    with open(args.preprocess_args, 'wb') as f:
        preprocess_args = {'tag_table': tag_table,
                           'tokenizer': tokenizer,
                           'rand_indices': rand_indices,
                           'max_len': train_data['x'].shape[1],
                           'vectorizer': vectorizer,
                           'transformer': transformer}

        # 'embedding_matrix': embedding_matrix}
        pickle.dump(preprocess_args, f)

    # save embedding matrix
    with open(args.embedding_matrix, 'wb') as f:
        pickle.dump(embedding_matrix, f)

    with open(args.data, 'wb') as f:
        data = {'train': train,
                'valid': valid}
        pickle.dump(data, f)


if __name__ == '__main__':
    try:
        main()
    except:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
