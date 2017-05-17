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
from stopwords import stopwords
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score


def f1score(y_true, y_pred):
    n_correct = np.sum(y_true * y_pred)
    if n_correct == 0:
        return 0

    precision = n_correct / np.sum(y_pred)
    recall = n_correct / np.sum(y_true)

    return 2 * precision * recall / (precision + recall)


def text2seq(texts, tokenizer):
    sequences = tokenizer.texts_to_sequences(texts)
    stopwords_seq = tokenizer.texts_to_sequences(stopwords)
    for i in range(len(sequences)):
        sequences[i] = list(filter(lambda x: x not in stopwords_seq,
                                   sequences[i]))

    return sequences


def count_word(sequences, volcab_size):
    word_count = np.zeros((len(sequences), volcab_size))
    for i in range(len(sequences)):
        for n in sequences[i]:
            word_count[i, n - 1] += 1

    return word_count


def calc_tfidf(train_word_count, test_word_count):
    train_size = train_word_count.shape[0]
    # test_size = test_word_count.shape[0]
    word_count = np.concatenate([train_word_count, test_word_count],
                                axis=0)
    df = np.log(np.sum(word_count > 0, axis=0) / word_count.shape[0])
    tf = word_count / np.sum(word_count.shape[1])
    tfidf = tf / df
    return tfidf[:train_size], tfidf[train_size:]


def main():
    parser = argparse.ArgumentParser(description='ML HW4')
    parser.add_argument('train', type=str, help='train.csv')
    parser.add_argument('test', type=str, help='train.csv')
    parser.add_argument('out', type=str, help='out.csv')
    parser.add_argument('--preprocess_args', type=str,
                        default='preprocss_args.pickle',
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
    vectorizer = CountVectorizer(min_df=1, stop_words='english')
    vectorizer.fit(train_data['text'] + test_data['text'])
    train_data['count'] = vectorizer.transform(train_data['text']).toarray()
    test_data['count'] = vectorizer.transform(test_data['text']).toarray()

    # pdb.set_trace()
    
    transformer = TfidfTransformer(smooth_idf=False)
    transformer.fit(np.concatenate([train_data['count'], test_data['count']], axis=0))
    train_data['x'] = transformer.transform(train_data['count']).toarray()
    test_data['x'] = transformer.transform(test_data['count']).toarray()
    
    # save preprocess args
    with open(args.preprocess_args, 'wb') as f:
        preprocess_args = {'tag_table': tag_table,
                           'vectorizer': vectorizer,
                           'transformer': transformer}
        # 'embedding_matrix': embedding_matrix}
        pickle.dump(preprocess_args, f)

    # split data
    train, valid = split_valid(train_data, args.valid_ratio)

    # start training
    # classifier = RandomForestClassifier(n_estimators=100, n_jobs=-1)
    classifiers = [LinearSVC(class_weight='balanced', C=0.0002) for i in range(train['y'].shape[1])]
    # classifiers = [RandomForestClassifier(n_estimators=100, n_jobs=-1) for i in range(train['y'].shape[1])]
    for i in range(train['y'].shape[1]):
        classifiers[i].fit(train['x'], train['y'][:, i])

    train['y_'] = np.zeros(train['y'].shape)
    valid['y_'] = np.zeros(valid['y'].shape)
    test_data['y_'] = np.zeros((test_data['x'].shape[0], valid['y'].shape[1]))

    for i in range(valid['y'].shape[1]):
        valid['y_'][:, i] = classifiers[i].predict(valid['x'])
        train['y_'][:, i] = classifiers[i].predict(train['x'])
        test_data['y_'][:, i] = classifiers[i].predict(test_data['x'])

    # valid['y_'] = classifier.predict(valid['x'])
    print('train F1 score =', f1_score(train['y'], train['y_'], average='samples'))
    print('valid F1 score =', f1_score(valid['y'], valid['y_'], average='samples'))

    test_data['tags'] = decode_tags(test_data['y_'], tag_table)
    write_predict(test_data['tags'], args.out)

    pdb.set_trace()

    
    

if __name__ == '__main__':
    try:
        main()
    except:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
