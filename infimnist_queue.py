# Tensorflow queue-based producer of infinite MNIST training data

import _infimnist as infimnist
import numpy as np
import tensorflow as tf
import threading

NUM_TRANSFORMS = 200  # number of random digit transformations to consider


class InfimnistProducer(object):
    def __init__(self, gen_batch_size=256, dequeue_batch_size=256, capacity=512):
        self.mnist = infimnist.InfimnistGenerator()
        self.gen_batch_size = gen_batch_size
        self.dequeue_batch_size = dequeue_batch_size

        self.indexes_placeholder = tf.placeholder(tf.int64, shape=(gen_batch_size))
        self.digits_placeholder = tf.placeholder(tf.uint8, shape=[gen_batch_size, 28 * 28])
        self.labels_placeholder = tf.placeholder(tf.uint8, shape=(gen_batch_size))

        q = tf.FIFOQueue(capacity=capacity,
                         dtypes=[tf.int64, tf.uint8, tf.uint8],
                         shapes=[[], [28 * 28], []])

        self.enqueue_op = q.enqueue_many([self.indexes_placeholder,
                                          self.digits_placeholder,
                                          self.labels_placeholder])

        indexes, digits, labels = q.dequeue_many(dequeue_batch_size)
        self.indexes = tf.mod(indexes - 10000, 60000)

        # basic preprocessing of X into [0, 1] range
        self.X = tf.cast(digits, tf.float32)
        self.X = tf.reshape(self.X, [dequeue_batch_size, 28, 28, 1])
        self.X = self.X / 255

        self.y = tf.cast(labels, tf.int32)

    def gen_digits(self, sess):
        while True:
            idxs = np.random.randint(10000, 10000 + 60000 * NUM_TRANSFORMS,
                                     size=self.gen_batch_size, dtype=np.int64)
            digits, labels = self.mnist.gen(idxs)

            print('enqueuing')
            sess.run(self.enqueue_op,
                     feed_dict={self.indexes_placeholder: idxs,
                                self.digits_placeholder: digits.reshape(self.gen_batch_size, 28 * 28),
                                self.labels_placeholder: labels})

    def start_queue(self, sess):
        t = threading.Thread(target=self.gen_digits, args=(sess,))
        t.start()


if __name__ == '__main__':
    data = InfimnistProducer(gen_batch_size=2000)

    sess = tf.Session()

    data.start_queue(sess)

    for _ in range(10):
        indexes, X, y = sess.run([data.indexes, data.X, data.y])
        print(indexes.shape, X.shape, y.shape)
