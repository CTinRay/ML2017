import numpy as np
import pandas as pd
import argparse
import pickle
from utils import *
import os
import pdb
import sys
import traceback
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score


def text2seq(texts, tokenizer):
    sequences = tokenizer.texts_to_sequences(texts)
    stopwords_seq = tokenizer.texts_to_sequences(stopwords)
    for i in range(len(sequences)):
        sequences[i] = list(filter(lambda x: x not in stopwords_seq,
                                   sequences[i]))

    return sequences


def main():
    parser = argparse.ArgumentParser(description='ML HW4')
    parser.add_argument('train', type=str, help='train.csv')
    parser.add_argument('model', type=str, help='out.csv')
    parser.add_argument('--preprocess_args', type=str,
                        default='preprocess_args.pickle',
                        help='pickle to store preprocess arguments')
    parser.add_argument('--c', type=float, default=0.0002)
    args = parser.parse_args()

    with open(args.preprocess_args, 'rb') as f:
        preprocess_args = pickle.load(f)
        tag_table = preprocess_args['tag_table']
        rand_indices = preprocess_args['rand_indices']
        vectorizer = preprocess_args['vectorizer']
        transformer = preprocess_args['transformer']
    
    # read data
    train_data = read_train(args.train)
    encode_tags(train_data, tag_table)
    train_data['count'] = vectorizer.transform(train_data['text']).toarray()
    train_data['x'] = transformer.transform(train_data['count']).toarray()
    train, valid, _ = split_valid(train_data, 0.1, rand_indices)

    # start training
    # classifier = RandomForestClassifier(n_estimators=100, n_jobs=-1)
    cs = [args.c] * 38
    classifiers = [LinearSVC(class_weight='balanced', C=cs[i])
                   for i in range(train['y'].shape[1])]
    # classifiers = [RandomForestClassifier(n_estimators=1000, n_jobs=-1) for i in range(train['y'].shape[1])]
    for i in range(train['y'].shape[1]):
        classifiers[i].fit(train['x'], train['y'][:, i])
        # y_ = classifiers[i].predict(valid['x'])
        # y_train = classifiers[i].predict(train['x'])
        # print('label %d, train f1 = %f valid f1 = %f'
        #       % (i, f1_score(train['y'][:, i], y_train),
        #          f1_score(valid['y'][:, i], y_)))

    train['y_'] = np.zeros(train['y'].shape)
    valid['y_'] = np.zeros(valid['y'].shape)

    for i in range(valid['y'].shape[1]):
        valid['y_'][:, i] = classifiers[i].predict(valid['x'])
        train['y_'][:, i] = classifiers[i].predict(train['x'])

    # valid['y_'] = classifier.predict(valid['x'])
    print('train F1 score =', f1_score(
        train['y'], train['y_'], average='samples'))
    print('valid F1 score =', f1_score(
        valid['y'], valid['y_'], average='samples'))

    with open(args.model, 'wb') as f:
        pickle.dump(classifiers, f)

    pdb.set_trace()


if __name__ == '__main__':
    try:
        main()
    except:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
