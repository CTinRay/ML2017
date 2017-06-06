import pdb
import sys
import traceback
import numpy as np
import argparse
from utils import read_train, split_valid, get_user_features, get_movie_features
from mf import MF


def main():
    parser = argparse.ArgumentParser(description='ML HW6')
    parser.add_argument('train', type=str, help='train.csv')
    parser.add_argument('model', type=str, help='model to save')
    parser.add_argument('--user', type=str, help='user.csv')
    parser.add_argument('--movie', type=str, help='movie.csv')
    parser.add_argument('--valid_ratio', type=float,
                        help='ratio of validation data.', default=0.1)
    parser.add_argument('--n_iters', type=int, default=100,
                        help='number of epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='batch size')
    parser.add_argument('--d_latent', type=int, default=50,
                        help='dimension of latent feature')
    args = parser.parse_args()

    train_data = read_train(args.train)

    # Fix index offset
    train_data['x'][:, 0] -= 1
    train_data['x'][:, 1] -= 1

    train, valid = split_valid(train_data, args.valid_ratio)

    # mean = np.mean(train['y'])
    # std = np.std(train['y'])

    # train['y'] = (train['y'] - mean) / std
    # valid['y'] = (valid['y'] - mean) / std
    user_features = None
    if args.user:
        user_features = get_user_features(args.user)

    movie_features = None
    if args.movie:
        movie_features = get_movie_features(args.movie)

    # start training
    mf = MF(n_iters=args.n_iters,
            lr=args.lr, batch_size=args.batch_size,
            filename=args.model, d_latent=args.d_latent,
            user_features=user_features,
            movie_features=movie_features)
    mf.fit(train['x'], train['y'], valid)


if __name__ == '__main__':
    try:
        main()
    except:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
