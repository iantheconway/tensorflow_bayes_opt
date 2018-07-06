"""Use Bayesian optimization to choose hyperparameters for a CNN which classifiers MNIST digits."""

import tensorflow as tf
import GPyOpt
import numpy as np
import os

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

PARAMS = ["filter_1_width", "filter_1_height", "filter_2_width", "filter_2_height", "learning_rate", "n_filters_conv1",
          "n_filters_conv2", "n_hidden_dense_1", "batch_size"]
INT_PARAMS = ["filter_1_width", "filter_1_height", "filter_2_width", "filter_2_height", "n_filters_conv1",
              "n_filters_conv2", "n_hidden_dense_1", "batch_size"]


class MNISTClassifier(object):
    """Classifies MNSIT data set using a CNN. Based on the tutorial at:
     https://www.tensorflow.org/tutorials/layers"""

    def __init__(self, filter_1_width=5, filter_1_height=5, filter_2_width=5, filter_2_height=5, learning_rate=1e-4,
                 n_filters_conv1=32, n_filters_conv2=64, n_hidden_dense_1=1024, batch_size=50):
        """Initialize computational graph for CNN.
        args:
            filter_1_width: width of filter for first conv layer
            filter_1_height: height of filter for first conv layer
            filter_2_width: width of filter for second conv layer
            filter_2_height: height of filter for second conv layer

        """
        # dynamically create experiment name from function arguments
        self.experiment_name = ""
        for key, value in locals().iteritems():
            if key != "self":
                self.experiment_name += "{}_{}_".format(key, value)
        self.experiment_name = self.experiment_name[:-1]
        print self.experiment_name
        # keep experiments in separate graphs.
        tf.reset_default_graph()
        self.sess = tf.Session()
        self.batch_size = batch_size
        self.best_accuracy = np.inf

        self.x = tf.placeholder(tf.float32, shape=[None, 784], name="x")
        self.y_ = tf.placeholder(tf.float32, shape=[None, 10], name="y_")
        self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")
        self.is_training = tf.placeholder(tf.bool, name="keep_prob")
        self.x_image = tf.reshape(self.x, [-1, 28, 28, 1], name="x_image")

        # Convolutional Layer #1
        self.conv1 = tf.layers.conv2d(
            inputs=self.x_image,
            filters=n_filters_conv1,
            kernel_size=[filter_1_width, filter_1_height],
            padding="same",
            activation=tf.nn.relu)

        # Pooling Layer #1
        self.pool1 = tf.layers.max_pooling2d(inputs=self.conv1, pool_size=[2, 2], strides=2)

        # Convolutional Layer #2 and Pooling Layer #2
        self.conv2 = tf.layers.conv2d(
            inputs=self.pool1,
            filters=n_filters_conv2,
            kernel_size=[filter_2_width, filter_2_height],
            padding="same",
            activation=tf.nn.relu)
        self.pool2 = tf.layers.max_pooling2d(inputs=self.conv2, pool_size=[2, 2], strides=2)

        # Dense Layer
        self.pool2_flat = tf.reshape(self.pool2, [-1, 7 * 7 * n_filters_conv2])
        self.dense = tf.layers.dense(inputs=self.pool2_flat, units=n_hidden_dense_1, activation=tf.nn.relu)
        self.dropout = tf.layers.dropout(
            inputs=self.dense, rate=self.keep_prob, training=self.is_training == tf.estimator.ModeKeys.TRAIN)

        # Logits Layer
        self.logits = tf.layers.dense(inputs=self.dropout, units=10)

        predictions = {
            # Generate predictions (for PREDICT and EVAL mode)
            "classes": tf.argmax(input=self.logits, axis=1),
            # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
            # `logging_hook`.
            "probabilities": tf.nn.softmax(self.logits, name="softmax_tensor")
        }

        self.cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=self.y_, logits=self.logits), name="cross_entropy")
        self.train_step = tf.train.AdamOptimizer(learning_rate).minimize(self.cross_entropy)
        self.correct_prediction = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.y_, 1),
                                           name="correct_prediction")
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32), name="accuracy")
        # for Tensorboard
        self.train_acc_val = np.array([0])
        self.test_acc_val = np.array([0])
        self.train_accuracy = tf.placeholder("float", shape=[1], name="train_accuracy")
        self.test_accuracy = tf.placeholder("float", shape=[1], name="test_accuracy")
        tf.summary.scalar('Train_accuracy', tf.reduce_sum(self.train_accuracy))
        tf.summary.scalar('Test_Accuracy', tf.reduce_sum(self.test_accuracy))
        self.merged_summary_op = tf.summary.merge_all()

        self.summary_writer = tf.summary.FileWriter(os.path.join(os.getcwd(), 'logs', self.experiment_name),
                                                    graph=self.sess.graph)
        # initialize variables
        self.sess.run(tf.global_variables_initializer())

    def train(self, iters):
        """Training function"""
        for i in range(iters):
            batch = mnist.train.next_batch(self.batch_size)


            _, summary, train_accuracy = self.sess.run([self.train_step, self.merged_summary_op, self.accuracy],
                                       feed_dict={self.x: batch[0], self.y_: batch[1],
                                                  self.keep_prob: 0.5, self.test_accuracy: self.test_acc_val,
                                                  self.train_accuracy: self.test_acc_val, self.is_training: True
                                                  })
            self.train_acc_val = np.array([train_accuracy])
            self.summary_writer.add_summary(summary, i)
            if i % 5 == 0:
                print('step %d, training accuracy %g' % (i, train_accuracy))
            test_accuracy = self.sess.run(self.accuracy,
                                          feed_dict={self.x: mnist.test.images, self.y_: mnist.test.labels,
                                                     self.keep_prob: 1.0, self.is_training: False})
            self.test_acc_val = np.array([test_accuracy])
            if test_accuracy > self.best_accuracy:
                self.best_accuracy = test_accuracy


def gpyopt_helper(x):
    """Objective function for GPyOpt.
    args:
        x: a 2D numpy array containing hyperparameters for the current acquisition
    returns:
        Error: The best test error for the training run."""
    params = {}
    for param, value in zip(PARAMS, x[0]):
        if param in INT_PARAMS:
            value = int(value)
        params[param] = value
    mc = MNISTClassifier(**params)
    mc.train(5000)
    # Convert accuracy to error
    error = 1 - mc.best_accuracy
    return np.array([[error]])


def bayes_opt():
    """Run bayesian optimization on the MNIST Classifier using GPyOpt"""
    bounds = [{'name': 'filter_1_width', 'type': 'discrete', 'domain': range(3, 7)},
              {'name': 'filter_1_height', 'type': 'discrete', 'domain': range(3, 7)},
              {'name': 'filter_2_width', 'type': 'discrete', 'domain': range(3, 7)},
              {'name': 'filter_2_height', 'type': 'discrete', 'domain': range(3, 7)},
              {'name': 'learning_rate', 'type': 'continuous', 'domain': (0.000001, 0.02)},
              {'name': 'n_filters_conv1', 'type': 'discrete', 'domain': range(32, 128)},
              {'name': 'n_filters_conv2', 'type': 'discrete', 'domain': range(32, 128)},
              {'name': 'n_hidden_dense_1', 'type': 'discrete', 'domain': range(512, 1024)},
              {'name': 'batch_size', 'type': 'discrete', 'domain': range(32, 512)},
              ]
    myProblem = GPyOpt.methods.BayesianOptimization(gpyopt_helper, bounds)
    myProblem.run_optimization(50)
    myProblem.save_evaluations("ev_file")


if __name__ == "__main__":
    bayes_opt()
