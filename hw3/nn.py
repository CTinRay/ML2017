import tensorflow as tf
import math
import numpy as np
import time
import pdb


class NNModel:

    def _get_var_normal(self, name, shape):
        intializer = tf.truncated_normal_initializer(stddev=0.1)
        return tf.get_variable(name, shape,
                               initializer=intializer)

    def _get_var_const(self, name, shape, val=0.1):
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
            return tf.nn.relu(tf.matmul(x, fc_w) + fc_b)

    def _fc_linear(self, name, x, shape):
        with tf.variable_scope(name):
            fc_w = self._get_var_normal(name + '_w', shape)
            fc_b = self._get_var_const(name + '_b', shape[-1])
            return tf.matmul(x, fc_w) + fc_b

    def _inference(self, X, keep_prob):
        x_image = tf.reshape(X, [-1, self.img_shape[0], self.img_shape[1], 1])
        self.conv_shape = [[5, 5, 1, 32], [4, 4, 32, 32], [5, 5, 32, 64]]

        conv0_h = self._conv('conv0', x_image, self.conv_shape[0])
        pool0_h = self._max_pool(conv0_h)
        # 20 x 20 x 32

        conv1_h = self._conv('conv1', pool0_h, self.conv_shape[1])
        pool1_h = self._avg_pool(conv1_h)
        # 10 x 10 x 32

        pool1_drop = tf.nn.dropout(pool1_h, keep_prob)
        conv2_h = self._conv('conv2', pool1_drop, self.conv_shape[2])
        pool2_h = self._avg_pool(conv2_h)
        # 5 x 5 x 64

        # conv_last = tf.nn.dropout(pool1_h, keep_prob)
        conv_last = pool2_h

        self.fc_widths = [3072, 10]
        last_shape = [-1, 5, 5, 64]  # computed manually...
        flattern_width = last_shape[1] * last_shape[2] * last_shape[3]
        flat_h = tf.reshape(conv_last, [-1, flattern_width])

        fc0_h = self._fc('fc0', flat_h, [flattern_width, self.fc_widths[0]])
        fc0_drop = tf.nn.dropout(fc0_h, keep_prob)
        fc1_h = self._fc_linear('fc1', fc0_drop, [self.fc_widths[0], self.fc_widths[1]])  # 

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
                 n_iter=100, img_shape=[40, 40]):
        self.eta = eta
        self.nh1 = nh1
        self.batch_size = batch_size
        self.n_iter = n_iter
        self.img_shape = img_shape
        tf.GPUOptions(per_process_gpu_memory_fraction=0.05)

    def fit(self, X, y, valid=None):
        # calc batch size
        if self.batch_size > 0:
            batch_size = self.batch_size
        else:
            batch_size = X.shape[0]

        # connect NN
        X_placeholder = tf.placeholder(tf.float32, shape=(None, X.shape[1]))
        y_placeholder = tf.placeholder(tf.int64, shape=(None))
        keep_prob = tf.placeholder(tf.float32, shape=(None))
        with tf.variable_scope('nn') as scope:
            logits = self._inference(X_placeholder, keep_prob)
            loss = self._loss(logits, y_placeholder)
            train_op = self._train(loss, self.eta)
            n_correct = self._evaluate(logits, y_placeholder)

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        self.sess = tf.Session()

        # log
        summary = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter('./' + time.ctime(), self.sess.graph)

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
                feed_dict = {X_placeholder: batch_X, y_placeholder: batch_y, keep_prob: 0.8}
                self.sess.run(train_op, feed_dict=feed_dict)

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
                feed_dict = {X_placeholder: batch_X, y_placeholder: batch_y, keep_prob: 1.0}
                batch_correct, summary_str = self.sess.run([n_correct, summary],
                                                            feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, i)
                train_correct += batch_correct

            avg_summary = tf.Summary()
            accuracy = train_correct / X.shape[0]
            avg_summary.value.add(tag="Train Accuracy", simple_value = accuracy)
            print('accuracy', accuracy)

            if valid is not None:
                valid_correct = 0
                for b in range(0, valid['x'].shape[0], 4000):
                    batch_X = valid['x'][b: b + 4000]
                    batch_y = valid['y'][b: b + 4000]
                    feed_dict = {X_placeholder: batch_X, y_placeholder: batch_y, keep_prob: 1.0}
                    valid_correct += self.sess.run(n_correct, feed_dict=feed_dict)

                accuracy = valid_correct / valid['x'].shape[0]
                avg_summary.value.add(tag="Valid Accuracy", simple_value = accuracy)
                print('accuracy valid', accuracy)

            summary_writer.add_summary(avg_summary, i)
            summary_writer.flush()

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
