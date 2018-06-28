"""Use Bayesian optimization to choose hyperparameters for a CNN which classifiers MNIST digits."""

import tensorflow as tf
import GPyOpt
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

PARAMS = ["filter_1_width", "filter_1_height", "filter_2_width", "filter_2_height"]
INT_PARAMS = ["filter_1_width", "filter_1_height", "filter_2_width", "filter_2_height"]


class MNISTClassifier(object):
    """Classifies MNSIT data set using a CNN. Based on the tutorial at:
     https://www.tensorflow.org/versions/r1.2/get_started/mnist/pros"""
    def __init__(self, filter_1_width=5, filter_1_height=5, filter_2_width=5, filter_2_height=5):
        """Initialize computational graph for CNN.
        args:
            filter_1_width: width of filter for first conv layer
            filter_1_height: height of filter for first conv layer
            filter_2_width: width of filter for second conv layer
            filter_2_height: height of filter for second conv layer

        """
        self.best_error = np.inf
        self.x = tf.placeholder(tf.float32, shape=[None, 784])
        self.y_ = tf.placeholder(tf.float32, shape=[None, 10])

        self.W_conv1 = self.weight_variable([filter_1_height, filter_1_width, 1, 32])
        self.b_conv1 = self.bias_variable([32])

        self.x_image = tf.reshape(self.x, [-1, 28, 28, 1])

        self.h_conv1 = tf.nn.relu(self.conv2d(self.x_image, self.W_conv1) + self.b_conv1)
        self.h_pool1 = self.max_pool_2x2(self.h_conv1)

        self.W_conv2 = self.weight_variable([filter_2_height, filter_2_width, 32, 64])
        self.b_conv2 = self.bias_variable([64])

        self.h_conv2 = tf.nn.relu(self.conv2d(self.h_pool1, self.W_conv2) + self.b_conv2)
        self.h_pool2 = self.max_pool_2x2(self.h_conv2)

        self.W_fc1 = self.weight_variable([7 * 7 * 64, 1024])
        self.b_fc1 = self.bias_variable([1024])

        self.h_pool2_flat = tf.reshape(self.h_pool2, [-1, 7 * 7 * 64])
        self.h_fc1 = tf.nn.relu(tf.matmul(self.h_pool2_flat, self.W_fc1) + self.b_fc1)

        self.keep_prob = tf.placeholder(tf.float32)
        self.h_fc1_drop = tf.nn.dropout(self.h_fc1, self.keep_prob)

        self.W_fc2 = self.weight_variable([1024, 10])
        self.b_fc2 = self.bias_variable([10])

        self.y_conv = tf.matmul(self.h_fc1_drop, self.W_fc2) + self.b_fc2

        self.cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=self.y_, logits=self.y_conv))
        self.train_step = tf.train.AdamOptimizer(1e-4).minimize(self.cross_entropy)
        self.correct_prediction = tf.equal(tf.argmax(self.y_conv, 1), tf.argmax(self.y_, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

    def weight_variable(self, shape):
        """Helper function which returns a truncated normal tf variable of a given shape.
        args:
            shape: the shape of the variable tensor to be returned
        """
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(self, shape, bias=0.1):
        """Helper function which returns a tensor of a given value and shape.
            args:
                shape: the shape of the variable tensor to be returned
        """
        initial = tf.constant(bias, shape=shape)
        return tf.Variable(initial)

    def conv2d(self, x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(self, x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1], padding='SAME')

    def train(self, iters):
        """Helper function"""
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(iters):
                batch = mnist.train.next_batch(50)
                if i % 1 == 0:
                    train_accuracy = self.accuracy.eval(feed_dict={
                        self.x: batch[0], self.y_: batch[1], self.keep_prob: 1.0})
                    print('step %d, training accuracy %g' % (i, train_accuracy))
                self.train_step.run(feed_dict={self.x: batch[0], self.y_: batch[1], self.keep_prob: 0.5})
                test_error = self.accuracy.eval(
                    feed_dict={self.x: mnist.test.images, self.y_: mnist.test.labels, self.keep_prob: 1.0})
                if test_error < self.best_error:
                    self.best_error = test_error


def gpyopt_helper(x):
    params = {}
    for param, value in zip(PARAMS, x[0]):
        if param in INT_PARAMS:
            value = int(value)
        params[param] = value
    mc = MNISTClassifier(**params)
    mc.train(5)
    print "Params: {}".format(params)
    print "Error: {}".format(mc.best_error)
    return np.array([[mc.best_error]])


def bayes_opt():
    bounds = [{'name': 'filter_1_width', 'type': 'discrete', 'domain': (3, 7)},
              {'name': 'filter_1_height', 'type': 'discrete', 'domain': (3, 7)},
              {'name': 'filter_2_width', 'type': 'discrete', 'domain': (3, 7)},
              {'name': 'filter_2_height', 'type': 'discrete', 'domain': (3, 7)},
              ]
    myProblem = GPyOpt.methods.BayesianOptimization(gpyopt_helper, bounds)
    myProblem.run_optimization(10)



if __name__ == "__main__":
    bayes_opt()
