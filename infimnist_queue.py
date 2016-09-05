# Tensorflow queue-based producer of infinite MNIST training data

import _infimnist as infimnist
import numpy as np
import tensorflow as tf
import threading

NUM_TRANSFORMS = 200  # number of random digit transformations to consider


class InfimnistProducer(object):
    def __init__(self, batch_size=256, gen_batch_size=256, capacity=512):
        '''
        batch_size: size of batches obtained after dequeuing
        gen_batch_size: size of batches produced during generation
        '''
        self.mnist = infimnist.InfimnistGenerator()
        self.gen_batch_size = gen_batch_size
        self.batch_size = batch_size

        self.gen_thread = None

        self.indexes_placeholder = tf.placeholder(tf.int64, shape=(gen_batch_size))
        self.digits_placeholder = tf.placeholder(tf.uint8, shape=[gen_batch_size, 28 * 28])
        self.labels_placeholder = tf.placeholder(tf.uint8, shape=(gen_batch_size))

        self.q = tf.FIFOQueue(capacity=capacity,
                              dtypes=[tf.int64, tf.float32, tf.int32],
                              shapes=[[], [28, 28, 1], []])

        indexes, digits, labels = self.process_raw_op(self.indexes_placeholder,
                                                      self.digits_placeholder,
                                                      self.labels_placeholder)
        self.enqueue_op = self.q.enqueue_many([indexes, digits, labels])

        # dequeue in batches of size batch_size
        self.indexes, self.digits, self.labels = self.q.dequeue_many(batch_size)

    def process_raw_op(self, indexes_raw, digits_raw, labels_raw):
        indexes = tf.mod(indexes_raw - 10000, 60000)

        # basic preprocessing of digits into [0, 1] range
        digits = tf.cast(digits_raw, tf.float32)
        digits = tf.reshape(digits, [self.gen_batch_size, 28, 28, 1])
        digits = digits / 255

        labels = tf.cast(labels_raw, tf.int32)

        return indexes, digits, labels

    def gen_digits(self, sess, coord):
        while coord is None or not coord.should_stop():
            idxs = np.random.randint(10000, 10000 + 60000 * NUM_TRANSFORMS,
                                     size=self.gen_batch_size, dtype=np.int64)
            digits, labels = self.mnist.gen(idxs)

            if coord is not None and coord.should_stop():
                break
            try:
                sess.run(self.enqueue_op,
                         feed_dict={self.indexes_placeholder: idxs,
                                    self.digits_placeholder: digits.reshape(self.gen_batch_size, 28 * 28),
                                    self.labels_placeholder: labels})
            except tf.errors.CancelledError:
                break

    def start_queue(self, sess, coord=None):
        self.gen_thread = threading.Thread(target=self.gen_digits, args=(sess, coord))
        self.gen_thread.daemon = True
        self.gen_thread.start()

    def join(self, sess, coord=None):
        sess.run(self.q.close(cancel_pending_enqueues=True))
        if self.gen_thread:
            if coord is None:
                self.gen_thread.join()
            else:
                coord.join([self.gen_thread])


if __name__ == '__main__':
    data = InfimnistProducer(gen_batch_size=2000, capacity=4000)

    sess = tf.Session()
    coord = tf.train.Coordinator()  # optional
    data.start_queue(sess, coord)

    for _ in range(10):
        indexes, digits, labels = sess.run([data.indexes, data.digits, data.labels])
        print(indexes.shape, digits.shape, labels.shape)

    coord.request_stop()
    data.join(sess, coord)
