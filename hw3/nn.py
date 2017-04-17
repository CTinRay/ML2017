import tensorflow as tf
import math
import numpy as np
import time
import pdb
import random
from scipy import ndimage


class NNModel:

    def _get_var_normal(self, name, shape):
        intializer = tf.truncated_normal_initializer(stddev=0.1)
        return tf.get_variable(name, shape,
                               initializer=intializer)

    def _get_var_const(self, name, shape, val=0.2):
        value = np.ones(shape) * val
        intializer = tf.constant_initializer(value)
        return tf.get_variable(name, shape,
                               initializer=intializer)

    def _conv2d(self, x, w):
        return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')

    def _max_pool(self, x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1], padding='SAME')

    def _avg_pool(self, x):
        return tf.nn.avg_pool(x, ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1], padding='VALID')

    def _conv(self, name, x, shape):
        with tf.variable_scope(name):
            conv_w = self._get_var_normal(name + '_w', shape)
            conv_b = self._get_var_const(name + '_b', shape[-1])
            return tf.nn.relu(self._conv2d(x, conv_w) + conv_b)

    def _fc(self, name, x, shape):
        with tf.variable_scope(name):
            fc_w = self._get_var_normal(name + '_w', shape)
            fc_b = self._get_var_const(name + '_b', shape[-1])
            n_in = tf.matmul(x, fc_w) + fc_b
            return tf.nn.relu(n_in) * 0.9 + 0.1 * n_in

    def _fc_linear(self, name, x, shape):
        with tf.variable_scope(name):
            fc_w = self._get_var_normal(name + '_w', shape)
            fc_b = self._get_var_const(name + '_b', shape[-1])
            return tf.matmul(x, fc_w) + fc_b

    def _augmentation(self, x, aug_params):
        x = tf.reshape(x, (-1, self.img_shape[0], self.img_shape[1], 1))

        # rotate
        a0 = tf.cos(aug_params['angle'])
        a1 = - tf.sin(aug_params['angle'])
        b0 = tf.sin(aug_params['angle'])
        b1 = tf.cos(aug_params['angle'])

        # shift
        a2 = aug_params['shift1']
        a2 -= 24 * (tf.cos(aug_params['angle']) - tf.sin(aug_params['angle']))
        a2 += 24
        b2 = aug_params['shift2']
        b2 -= 24 * (tf.cos(aug_params['angle']) + tf.sin(aug_params['angle']))
        b2 += 24

        # scale
        a0 *= aug_params['scale1']
        a1 *= aug_params['scale1']
        a2 *= aug_params['scale1']
        b0 *= aug_params['scale2']
        b1 *= aug_params['scale2']
        b2 *= aug_params['scale2']

        x = tf.contrib.image.transform(x, [a0, a1, a2, b0, b1, b2, 0, 0])
        return tf.reshape(x, (-1, self.img_shape[0] * self.img_shape[1]))

    def _inference(self, X, keep_prob):
        x_image = tf.reshape(X, [-1, self.img_shape[0], self.img_shape[1], 1])

        def crop(img):
            return tf.image.resize_image_with_crop_or_pad(img, 40, 40)
        x_image = tf.map_fn(crop, x_image)
        tf.summary.image('augged_img', x_image[0:1])

        self.conv_shape = [[5, 5, 1, 32], [4, 4, 32, 32], [5, 5, 32, 64]]

        conv0_h = self._conv('conv0', x_image, self.conv_shape[0])
        pool0_h = self._max_pool(conv0_h)
        # 20 x 20 x 32

        conv1_h = self._conv('conv1', pool0_h, self.conv_shape[1])
        pool1_h = self._avg_pool(conv1_h)
        # 10 x 10 x 32

        # pool1_drop = tf.nn.dropout(pool1_h, keep_prob)
        conv2_h = self._conv('conv2', pool1_h, self.conv_shape[2])
        pool2_h = self._avg_pool(conv2_h)
        # 5 x 5 x 64

        # conv_last = tf.nn.dropout(pool1_h, keep_prob)
        conv_last = pool2_h

        self.fc_widths = [3072, 10]
        last_shape = [-1, 5, 5, 64]  # computed manually...
        flattern_width = last_shape[1] * last_shape[2] * last_shape[3]
        flat_h = tf.reshape(conv_last, [-1, flattern_width])

        fc0_h = self._fc('fc0', flat_h,
                         [flattern_width, self.fc_widths[0]])
        fc0_drop = tf.nn.dropout(fc0_h, keep_prob)
        fc1_h = self._fc(
            'fc1', fc0_drop, [self.fc_widths[0], self.fc_widths[1]])  #

        return fc1_h

    def _loss(self, logits, labels):
        labels = tf.to_int64(labels)
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels, logits=logits, name='xentropy')
        return tf.reduce_mean(cross_entropy, name='xentropy_mean')

    def _train(self, loss, learning_rate):
        tf.summary.scalar('loss', loss)
        optimizer = tf.train.AdamOptimizer(learning_rate)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        train_op = optimizer.minimize(loss, global_step=global_step)
        return train_op

    def _evaluate(self, logits, labels):
        correct = tf.equal(tf.argmax(logits, 1), labels)
        return tf.reduce_sum(tf.cast(correct, tf.int32))

    def __init__(self, eta=1e-5, nh1=2, batch_size=1,
                 n_iter=100, img_shape=[48, 48], savefile=None):
        self.eta = eta
        self.nh1 = nh1
        self.batch_size = batch_size
        self.n_iter = n_iter
        self.img_shape = img_shape
        if savefile is None:
            self.savefile = time.ctime()
        else:
            self.savefile = savefile
            
        tf.GPUOptions(per_process_gpu_memory_fraction=0.05)

    def fit(self, X, y, valid=None):
        # calc batch size
        if self.batch_size > 0:
            batch_size = self.batch_size
        else:
            batch_size = X.shape[0]

        # connect NN
        raw_X = tf.placeholder(tf.float32, shape=(None, 48 * 48), name='raw_X')
        raw_y = tf.placeholder(tf.int64, shape=(None), name='label')
        aug_params = {'angle': tf.placeholder(tf.float32, shape=(None)),
                      'scale1': tf.placeholder(tf.float32, shape=(None)),
                      'scale2': tf.placeholder(tf.float32, shape=(None)),
                      'shift1': tf.placeholder(tf.float32, shape=(None)),
                      'shift2': tf.placeholder(tf.float32, shape=(None))}
        augged_X_tensor = self._augmentation(raw_X, aug_params)
        augged_X = tf.placeholder_with_default(augged_X_tensor,
                                               shape=(None, 48 * 48),
                                               name='augged_X')
        keep_prob = tf.placeholder(tf.float32, shape=(None))
        with tf.variable_scope('nn'):
            logits = self._inference(augged_X, keep_prob)
            loss = self._loss(logits, raw_y)
            train_op = self._train(loss, self.eta)
            n_correct = self._evaluate(logits, raw_y)

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        self.sess = tf.Session()

        # log
        summary = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(
            './' + time.ctime(), self.sess.graph)

        self.sess.run(init)
        # Start the training loop.
        for i in range(self.n_iter):
            start_time = time.time()

            # shuffle
            inds = np.arange(X.shape[0])
            np.random.shuffle(inds)
            X = X[inds]
            y = y[inds]

            for b in range(0, X.shape[0], batch_size):
                batch_X = X[b: b + batch_size]
                batch_y = y[b: b + batch_size]
                feed_dict = {}
                if i > -1:
                    feed_dict = {
                        raw_X: batch_X,
                        raw_y: batch_y,
                        keep_prob: 0.8,
                        aug_params['angle']: random.uniform(-0.4, 0.4),
                        aug_params['scale1']: random.uniform(0.8, 1.2),
                        aug_params['scale2']: random.uniform(0.8, 1.2),
                        aug_params['shift1']: random.uniform(-2, 2),
                        aug_params['shift2']: random.uniform(-2, 2)
                    }
                else:
                    feed_dict = {augged_X: batch_X,
                                 keep_prob: 0.8,
                                 raw_y: batch_y}

                _, summary_str = self.sess.run([train_op, summary],
                                               feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, i)

                duration = time.time() - start_time
            print('iter %d, %d' % (i, duration))
            # self._evaluate(logits, y_placeholder)

            # calc accuracy
            # ain = self.sess.run(accuracy,
            #                    feed_dict={X_placeholder: X, y_placeholder: y})
            # print('a in :', ain)

            # tf summary
            train_correct = 0
            for b in range(0, X.shape[0], 4000):
                batch_X = X[b: b + 4000]
                batch_y = y[b: b + 4000]
                feed_dict = {augged_X: batch_X,
                             raw_y: batch_y, keep_prob: 1.0}
                batch_correct = self.sess.run(n_correct,
                                              feed_dict=feed_dict)
                train_correct += batch_correct

            avg_summary = tf.Summary()
            accuracy = train_correct / X.shape[0]
            avg_summary.value.add(tag="Train Accuracy", simple_value=accuracy)
            print('accuracy', accuracy)

            if valid is not None:
                valid_correct = 0
                for b in range(0, valid['x'].shape[0], 4000):
                    batch_X = valid['x'][b: b + 4000]
                    batch_y = valid['y'][b: b + 4000]
                    feed_dict = {augged_X: batch_X,
                                 raw_y: batch_y, keep_prob: 1.0}
                    valid_correct += self.sess.run(n_correct,
                                                   feed_dict=feed_dict)

                accuracy = valid_correct / valid['x'].shape[0]
                avg_summary.value.add(
                    tag="Valid Accuracy", simple_value=accuracy)
                print('accuracy valid', accuracy)

            summary_writer.add_summary(avg_summary, i)
            summary_writer.flush()

        save_path = saver.save(self.sess, self.savefile)
        print('saved to', save_path)

    def predict(self, X):
        y_ = np.zeros(X.shape[0])
        with tf.variable_scope('nn', reuse=True):
            for b in range(0, X.shape[0], 4000):
                batch_X = X[b: b + 4000]
                X_placeholder = tf.placeholder(
                    tf.float32, shape=(None, X.shape[1]))
                keep_prob = tf.placeholder(tf.float32, shape=(None))
                logits = self._inference(X_placeholder, keep_prob)
                y_[b:b + 4000] = self.sess.run(tf.argmax(logits, axis=1),
                                               feed_dict={X_placeholder: batch_X, keep_prob: 1.0})

        return y_
