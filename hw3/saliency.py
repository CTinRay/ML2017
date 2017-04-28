import argparse
import numpy as np
import os
from keras.models import load_model
import keras.backend as K
import numpy as np
import matplotlib.pyplot as plt
import pdb
import sys
import traceback


base_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
img_dir = os.path.join(base_dir, 'image')
if not os.path.exists(img_dir):
    os.makedirs(img_dir)
cmap_dir = os.path.join(img_dir, 'cmap')
if not os.path.exists(cmap_dir):
    os.makedirs(cmap_dir)
partial_see_dir = os.path.join(img_dir,'partial_see')
if not os.path.exists(partial_see_dir):
    os.makedirs(partial_see_dir)
model_dir = os.path.join(base_dir, 'model')

def get_test(csv):
    xs = []
    cnt = 0
    with open(csv) as f:
        f.readline()
        for l in f:
            cols = l.split(',')
            xs.append(list(map(int, cols[1].split())))

    return {'x': np.array(xs)}




def main():
    parser = argparse.ArgumentParser(prog='plot_saliency.py',
            description='ML-Assignment3 visualize attention heat map.')
    parser.add_argument('path', type=str)
    parser.add_argument('test', type=str)
    args = parser.parse_args()
    emotion_classifier = load_model(os.path.join(args.path, 'model.h5'))

    input_img = emotion_classifier.input
    # img_ids = [args.id]

    private_pixels = get_test(args.test)['x'].reshape(-1, 48, 48, 1)
    private_pixels = (private_pixels - np.mean(private_pixels, axis=0)) / np.max(np.abs(private_pixels))

    # get max
    target = emotion_classifier.output
    fn = K.function([input_img, K.learning_phase()], [target])
    preds = fn([private_pixels[:1000], 0])[0]
    preds = np.max(preds, axis=1)
        
    img_ids = np.argsort(preds)[-20:]
    for idx in img_ids:
        img = private_pixels[idx].reshape(-1, 48, 48, 1)
        val_proba = emotion_classifier.predict(img)
        pred = val_proba.argmax(axis=-1)
        target = K.mean(emotion_classifier.output[:, pred])
        grads = K.gradients(target, input_img)[0]
        fn = K.function([input_img, K.learning_phase()], [grads])

        heatmap = np.array(fn([img, 0]))
        # pdb.set_trace()

        heatmap = heatmap.reshape(48, 48)
        heatmap = np.abs(heatmap)
        heatmap = (heatmap - np.mean(heatmap)) / (np.std(heatmap) + 1e-5)
        '''
        Implement your heatmap processing here!
        hint: Do some normalization or smoothening on grads
        '''
        thres = 0.5
        see = np.copy(private_pixels[idx].reshape(48, 48))
        see[np.where(heatmap <= thres)] = np.mean(see)

        plt.figure()
        plt.imshow(heatmap, cmap=plt.cm.jet)
        plt.colorbar()
        plt.tight_layout()
        fig = plt.gcf()
        plt.draw()
        fig.savefig('sa-%d.png' % idx, dpi=100)

        plt.figure()
        plt.imshow(see, cmap='gray')
        plt.colorbar()
        plt.tight_layout()
        fig = plt.gcf()
        plt.draw()
        fig.savefig('sa-grey-%d.png' % idx, dpi=100)

        plt.figure()
        plt.imshow(img.reshape(48, 48), cmap='gray')
        plt.colorbar()
        plt.tight_layout()
        fig = plt.gcf()
        plt.draw()
        fig.savefig('%d.png' % idx, dpi=100)



if __name__ == '__main__':
    try:
        main()
    except:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
        
