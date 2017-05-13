import numpy as np
import csv
import random
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


def read_train(filename):
    raw_data = {'text': [], 'tags': []}
    with open(filename, errors='ignore') as f:
        csv_reader = csv.reader(f)
        for row in csv_reader:
            tags = row[1].split(',')
            raw_data['tags'].append(tags)
            raw_data['text'].append(row[2])

    return raw_data


def read_test(filename):
    raw_data = {'text': []}
    with open(filename, errors='ignore') as f:
        csv_reader = csv.reader(f)
        for row in csv_reader:
            raw_data['text'].append(row[1])

    return raw_data


def make_tag_table(data_tags):
    all_tags = set()
    for tags in data_tags:
        for tag in tags:
            all_tags.add(tag)

    tag_table = list(all_tags)
    return tag_table


def make_tokenizer(texts):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts)
    return tokenizer


def encode_text(data, tokenizer, max_len=None):
    sequences = tokenizer.texts_to_sequences(data['text'])
    data['xs'] = pad_sequences(sequences)


def encode_tags(data, tag_table):
    n_tags = len(tag_table)
    data['ys'] = np.zeros((len(data['text']), n_tags))
    for i in range(len(data['tags'])):
        for tag in data['tags'][i]:
            ind = tag_table.index(tag)
            data['ys'][i][ind] = 1

    data['ys'] = np.array(data['ys'])


def decode_tags(data, tag_table):
    for i in range(len(data['tags'])):
        data['tags'][i] = [tag_table[i]
                           for i in np.where(data['ys'][i] == 1)[0]]


def split_valid(data, valid_ratio):
    indices = np.arange(data['xs'].shape[0])
    np.random.shuffle(indices)
    data['xs'] = data['xs'][indices]
    data['ys'] = data['ys'][indices]

    n_valid = int(data['xs'].shape[0] * valid_ratio)
    train = {'xs': data['xs'][n_valid:], 'ys': data['ys'][n_valid:]}
    valid = {'xs': data['xs'][:n_valid], 'ys': data['ys'][:n_valid]}

    return train, valid


def load_glove(filename):
    glove_dict = {}
    with open(filename) as f:
        for row in f:
            cols = row.split()
            word = cols[0]
            vec = np.asarray(cols[1:], dtype='float32')
            glove_dict[word] = vec

    return glove_dict


def make_embedding_matrix(tokenizer, glove_dict):
    n_words = len(tokenizer.word_index)
    wv_dim = glove_dict[','].shape[0]
    embedding_matrix = np.zeros((n_words + 1, wv_dim))
    for word, i in tokenizer.word_index.items():
        if word in glove_dict:
            embedding_matrix[i] = glove_dict[word]

    return embedding_matrix
