import pickle
import argparse
import numpy as np
from utils import *
import os
import pdb
import sys
import traceback
from sklearn.metrics import f1_score
from rnn import TextClassifier


def tfidf_predict(model, x):
    y_ = np.zeros((x.shape[0], 38))
    for i in range(len(model)):
        y_[:, i] = model[i].predict(x)

    return y_


def main():
    parser = argparse.ArgumentParser(description='ML HW4')
    parser.add_argument('train', type=str, help='train.csv')
    parser.add_argument('test', type=str, help='test.csv')
    parser.add_argument('out', type=str, help='out.csv')
    parser.add_argument('--preprocess_args', type=str,
                        default='preprocess_args.pickle',
                        help='pickle to store preprocess arguments')
    args = parser.parse_args()
    # parser.add_argument('model', type=str, help='out.csv')

    # read data
    with open(args.preprocess_args, 'rb') as f:
        preprocess_args = pickle.load(f)
        tag_table = preprocess_args['tag_table']
        rand_indices = preprocess_args['rand_indices']
        vectorizer = preprocess_args['vectorizer']
        transformer = preprocess_args['transformer']
        tokenizer = preprocess_args['tokenizer']

    train_data = read_train(args.train)
    test = read_test(args.test)
    encode_tags(train_data, tag_table)
    train_data['count'] = vectorizer.transform(train_data['text']).toarray()
    test['count'] = vectorizer.transform(test['text']).toarray()
    train_data['x'] = transformer.transform(train_data['count']).toarray()
    test['x'] = transformer.transform(test['count']).toarray()

    train, valid, _ = split_valid(train_data, 0.1, rand_indices)

    tfidf_model_files = ['svm-0002.pickle', 'svm-0003.pickle', 'svm-00025.pickle']
    tfidf_model_files = []
    tfidf_models = []
    for filename in tfidf_model_files:
        with open(filename, 'rb') as f:
            tfidf_models.append(pickle.load(f))

    train['y_'] = np.zeros((train['x'].shape[0], 38))
    valid['y_'] = np.zeros((valid['x'].shape[0], 38))
    test['y_'] = np.zeros((test['x'].shape[0], 38))

    for model in tfidf_models:
        # train['x_']
        valid['y_'] += (tfidf_predict(model, valid['x']))
        test['y_'] += (tfidf_predict(model, test['x']))

    valid_y = valid['y_']
    test_y = test['y_']
    
    train_data = read_train(args.train)
    encode_tags(train_data, tag_table)
    encode_text(train_data, tokenizer)
    encode_text(test, tokenizer, train_data['x'].shape[1])
    train, valid, _ = split_valid(train_data, 0.1, rand_indices)
    valid['y_'] = valid_y
    test['y_'] = test_y

    rnn_models = ['1GRU-SIG-2NN-52.h5', '1GRU-SIG-3NN-TA-52.h5',
                  '2GRU-SIG-52.h5']
    # rnn_models = []
    for model in rnn_models:
        classifier = TextClassifier()
        classifier.load(model)
        valid['y_'] += (classifier.predict_raw(valid['x']))
        test['y_'] += (classifier.predict_raw(test['x']))

    
    threshold = (len(tfidf_model_files) + len(rnn_models)) / 2
    # threshold = (len(rnn_models)) / 2

    all_zero_inds = np.where(np.sum(valid['y_'], axis=1) == 0)[0]
    valid['y_'] = np.where(valid['y_'] > threshold, 1, 0)
    test['y_'] = np.where(test['y_'] > threshold, 1, 0)

    all_zero_inds = np.where(np.sum(test['y_'], axis=1) == 0)[0]
    max_inds = np.argmax(test['y_'], axis=1)
    test['y_'][all_zero_inds, max_inds[all_zero_inds]] = 1
    
    print('valid f1 score =', f1_score(valid['y'], valid['y_'],
                                       average='samples'))

    pdb.set_trace()
    test['tags'] = decode_tags(test['y_'], tag_table)
    write_predict(test['tags'], args.out)


if __name__ == '__main__':
    try:
        main()
    except:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
