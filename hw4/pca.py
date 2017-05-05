import pickle
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import argparse
import os
import pdb
import sys
import traceback


def load_imgs(dir):
    img_subjects = []
    for s in 'ABCDEFGHIJKLM':
        img_subject = []
        for i in range(75):
            filename = os.path.join(dir, "%c%02d.bmp" % (s, i))
            img = matplotlib.image.imread(filename)
            img = img.reshape(-1,)
            img_subject.append(img)

        img_subject = np.array(img_subject)
        img_subjects.append(img_subject)

    return img_subjects


def merge_imgs(imgs_subject):
    all_imgs = np.concatenate(imgs_subjects,
                              axis=0)
    return all_imgs


def pca(imgs):
    imgs = imgs - np.mean(imgs, axis=0)
    u, s, v = np.linalg.svd(imgs)
    eigen_val = s ** 2
    eigen_vec = v.T
    inds = np.argsort(eigen_val)
    eigen_val = eigen_val[inds]
    eigen_vec = eigen_vec[:, inds]
    return eigen_val, eigen_vec


def rmse(x_, x):
    n = x.shape[0]
    return np.sqrt(np.sum((x_ - x) ** 2) / n)


def p1(img_subjects,
       filename_avg_face, filename_eigen_faces,
       filename_projected_faces):
    first_10 = []
    for s in range(10):
        first_10.append(img_subjects[s][:10])

    first_10 = np.concatenate(first_10, axis=0)
    mean = np.mean(first_10)
    # std = np.std(first_10)
    # first_10 = (first_10 - mean) / std
    first_10 = (first_10 - mean)

    _, eigen_faces = pca(first_10)

    avg_face = np.mean(first_10, axis=0)
    plt.imsave(filename_avg_face,
               avg_face.reshape(64, 64),
               cmap=plt.cm.gray)

    fig = plt.figure(dpi=300)
    for i in range(9):
        ax = fig.add_subplot(3, 3, i + 1)
        img = eigen_faces[:, -1 - i].reshape(64, 64)
        # img = img * std + mean
        img = img + mean
        ax.imshow(img, cmap=plt.cm.gray)
        plt.xticks(np.array([]))
        plt.yticks(np.array([]))

    fig.savefig(filename_eigen_faces)

    # project imgs to eigen faces
    projected_imgs = np.zeros([100, 64 * 64])
    for e in range(5):
        inner = first_10 @ eigen_faces[:, -1 - e]
        projected_imgs += inner.reshape(-1, 1) * eigen_faces[:, -1 - e]

    fig = plt.figure(dpi=300)
    for i in range(100):
        ax = fig.add_subplot(10, 10, i + 1)
        ax.imshow(projected_imgs[i].reshape(64, 64), cmap=plt.cm.gray)
        plt.xticks(np.array([]))
        plt.yticks(np.array([]))

    fig.savefig(filename_projected_faces)

    # rmse =


def main():
    parser = argparse.ArgumentParser(description='ML HW4: PCA')
    parser.add_argument('images', type=str, help='directory of the images')
    parser.add_argument('path', type=str,
                        help='directory of the output images')
    args = parser.parse_args()
    img_subjects = load_imgs(args.images)
    # imgs = merge_imgs(imgs_subject)

    p1(img_subjects,
       os.path.join(args.path, 'avg-face.png'),
       os.path.join(args.path, 'eigen-faces.png'),
       os.path.join(args.path, 'projected-faces.png'))


if __name__ == '__main__':
    try:
        main()
    except:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
