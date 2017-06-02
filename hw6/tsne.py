import argparse
import pdb
import sys
import traceback
import numpy as np
from sklearn.manifold import TSNE
from keras.models import load_model
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def get_big_category(category):
    big_categories = ['Musical/Comedy/Romance',
                      'Drama/War',
                      'Fantacy/Action/Adventure',
                      'Crime/Horror/Thriller/Mystry/Sci-Fi',
                      'Film-Noir/Western', 'Documentary',
                      'Animation/Children\'s']
    
    for bc in big_categories:
        if category in bc:
            return bc

    return ''


def read_movies(filename):
    categories = [''] * 3952
    with open(filename, encoding='latin_1') as f:
        next(f)
        for row in f:
            cols = row.split(',')
            mid = int(cols[0])
            category = cols[-1].strip().split('|')[0]
            categories[mid - 1] = get_big_category(category)

    return np.array(categories)


def get_embedding(filename):
    model = load_model(filename)
    return np.array(model.layers[4].get_weights())


def main():
    parser = argparse.ArgumentParser(description='ML HW6 TSNE')
    parser.add_argument('movies', type=str, help='movies.csv')
    parser.add_argument('model', type=str, help='model to save')
    parser.add_argument('--tsne', type=str,
                        help='Output image', default='tsne.png')
    args = parser.parse_args()

    categories = read_movies(args.movies)

    embedding = get_embedding(args.model)
    embedding = embedding.reshape(embedding.shape[1], -1)

    tsne = TSNE()
    d2 = tsne.fit_transform(embedding)

    fig = plt.figure(figsize=(10, 10), dpi=300)
    ax = fig.add_subplot(1, 1, 1)
    tags_to_show = np.unique(categories)
    pdb.set_trace()
    for tag in tags_to_show:
        if tag != '':
            inds = np.where((categories == tag) &
                            (np.sum(embedding, axis=1) != 0))[0]
            ax.plot(d2[inds, 0], d2[inds, 1], 'o', label=tag)

    ax.legend()
    fig.savefig(args.tsne)


if __name__ == '__main__':
    try:
        main()
    except:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
