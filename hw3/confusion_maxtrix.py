import pdb
import argparse
import pickle
import itertools
import numpy as np
from keras.models import load_model
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.jet):
    """
    This function prints and plots the confusion matrix.
    """
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, '{:.2f}'.format(cm[i, j]), horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def get_XY(csv):
    xs = []
    ys = []
    with open(csv) as f:
        f.readline()
        for l in f:
            cols = l.split(',')
            ys.append(int(cols[0]))
            xs.append(list(map(int, cols[1].split())))

    return {'x': np.array(xs), 'y': np.array(ys)}


def main():
    parser = argparse.ArgumentParser(prog='plot_saliency.py',
                                     description='ML-Assignment3 confusion matrix.')
    parser.add_argument('model', type=str)
    parser.add_argument('train', type=str)
    parser.add_argument('valid', type=str)
    parser.add_argument('out', type=str)
    args = parser.parse_args()

    model = load_model(args.model)
    np.set_printoptions(precision=2)

    train = get_XY(args.train)
    valid_inds = pickle.load(open(args.valid, 'rb'))
    train['x'] = train['x'][valid_inds]
    train['y'] = train['y'][valid_inds]
    mean = np.mean(train['x'], axis=0)
    abs_max = np.max(np.abs(train['x']))
    n_valid = int(train['x'].shape[0] * 0.1)
    valid = {'x': (train['x'][:n_valid] - mean) / abs_max,
             'y': train['y'][:n_valid]}

    valid['y_'] = model.predict_classes(valid['x'].reshape(-1, 48, 48, 1))
    conf_mat = confusion_matrix(valid['y'], valid['y_'])

    plt.figure(dpi=600)
    plot_confusion_matrix(conf_mat, classes=[
                          "Angry", "Disgust", "Fear",
                          "Happy", "Sad", "Surprise", "Neutral"])
    plt.savefig(args.out)


if __name__ == '__main__':
    main()
