import matplotlib
matplotlib.use('Agg')
import word2vec
import argparse
import os
import pdb
import sys
import traceback
from sklearn.manifold import TSNE
from adjustText import adjust_text
import matplotlib.pyplot as plt
import nltk
import re
import numpy as np


def main():
    parser = argparse.ArgumentParser(description='ML HW4: Word2Vec')
    parser.add_argument('raw', type=str, default=None)
    parser.add_argument('tsne', type=str)
    parser.add_argument('--wordvec', type=str, help='directory of the images')
    parser.add_argument('--n_words', type=int, default=1000,
                        help='number of words to show')
    args = parser.parse_args()

    if args.raw is not None:
        word2vec.word2phrase(args.raw, './phrases.txt')
        word2vec.word2vec('./phrases.txt', args.wordvec, size=100)

    model = word2vec.load(args.wordvec)

    with open(args.raw) as f:
        # text = f.read().lower()
        text = f.read()
        tokens = nltk.word_tokenize(text)
        fdist = nltk.FreqDist(tokens)
        most_common = fdist.most_common()[::-1]
    
    tags_to_show = ['JJ', 'NNP', 'NN', 'NNS']
    vocab = []
    vecs = []
    tags = []
                        
    while len(vocab) < args.n_words:
        word, _ = most_common.pop()
        tag = nltk.pos_tag([word])[0][1]
        if tag not in tags_to_show:
            continue

        if len(word) <= 1:
            continue

        if re.search('[(“,.:;’!?”"_)]', word) is not None:
            continue

        if word not in model:
            continue
        
        vocab.append(word)
        vecs.append(model[word])
        tags.append(tag)

    vocab = np.array(vocab)
    vecs = np.array(vecs)
    tags = np.array(tags)

    vocab = np.array(vocab)
    vecs = np.array(vecs)
    tags = np.array(tags)
    
    tsne = TSNE()
    d2 = tsne.fit_transform(vecs)
    d2 *= 100
    print('finish tsne')    
    
    fig = plt.figure(figsize=(20, 20), dpi=300)
    ax = fig.add_subplot(1, 1, 1)
    for tag in tags_to_show:
        inds = np.where(tags == tag)[0]
        ax.plot(d2[inds, 0], d2[inds, 1], 'o', label=tag)

        texts = []
        for ind in inds:
            texts.append(plt.text(d2[ind][0], d2[ind][1], vocab[ind]))

        adjust_text(texts, d2[inds, 0], d2[inds, 1])
        
    ax.legend()
    fig.savefig(args.tsne)
    
    

if __name__ == '__main__':
    try:
        main()
    except:
        type, value, tb=sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
