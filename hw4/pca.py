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
    covar = (imgs.T @ imgs) / imgs.shape[0]
    eigen_val, eigen_vec = np.linalg.eigh(covar)
    return eigen_val, eigen_vec


def p1(img_subjects, filename_avg_face, filename_eigen_faces):
    first_10 = []
    for s in range(10):
        first_10.append(img_subjects[s][:10])

    first_10 = np.concatenate(first_10, axis=0)
    _, eigen_faces = pca(first_10)

    avg_face = np.mean(first_10, axis=0)
    plt.imsave(filename_avg_face,
               avg_face.reshape(64, 64),
               cmap=plt.cm.gray)

    fig = plt.figure(dpi=300)
    for i in range(9):
        ax = fig.add_subplot(3, 3, i + 1)
        ax.imshow(eigen_faces[:, -1 - i].reshape(64, 64), cmap=plt.cm.gray)
        plt.xticks(np.array([]))
        plt.yticks(np.array([]))

    fig.savefig(filename_eigen_faces)
    

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
       os.path.join(args.path, 'eigen-faces.png'))

    


if __name__ == '__main__':
    try:
        main()
    except:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
