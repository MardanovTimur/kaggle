import os
import warnings

import pickle

import numpy as np
import tensorflow as tf
# Import MINST data
from tensorflow.examples.tutorials.mnist import input_data

import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

mnist = input_data.read_data_sets("MNIST_data/", seed=69, one_hot=True)

NOISY_ARR = 'data/noisy_data_mnist_noisy.dump'
NOISY_ARR_TRAIN = 'data/noisy_data_mnist_noisy_train.dump'


# Parameters
learning_rate = 0.01
training_epochs = 20
batch_size = 256
display_step = 1
examples_to_show = 10

# Network Parameters
n_hidden_1 = 256  # 1st layer num features
n_hidden_2 = 128  # 2nd layer num features
n_input = 784  # MNIST data input (img shape: 28*28)

# tf Graph input (only pictures)
X = tf.placeholder("float", [None, n_input])

weights = {
    'encoder_h1': tf.Variable
    (tf.random_normal([n_input, n_hidden_1])),
    'encoder_h2': tf.Variable
    (tf.random_normal([n_hidden_1, n_hidden_2])),
    'decoder_h1': tf.Variable
    (tf.random_normal([n_hidden_2, n_hidden_1])),
    'decoder_h2': tf.Variable
    (tf.random_normal([n_hidden_1, n_input])),
}
biases = {
    'encoder_b1': tf.Variable
    (tf.random_normal([n_hidden_1])),
    'encoder_b2': tf.Variable
    (tf.random_normal([n_hidden_2])),
    'decoder_b1': tf.Variable
    (tf.random_normal([n_hidden_1])),
    'decoder_b2': tf.Variable
    (tf.random_normal([n_input])),
}


# Encoder Hidden layer with sigmoid activation #1
encoder_in = tf.nn.sigmoid(tf.add
                           (tf.matmul(X,
                                      weights['encoder_h1']),
                            biases['encoder_b1']))

# Decoder Hidden layer with sigmoid activation #2
encoder_out = tf.nn.sigmoid(tf.add
                            (tf.matmul(encoder_in,
                                       weights['encoder_h2']),
                             biases['encoder_b2']))


# Encoder Hidden layer with sigmoid activation #1
decoder_in = tf.nn.sigmoid(tf.add
                           (tf.matmul(encoder_out,
                                      weights['decoder_h1']),
                            biases['decoder_b1']))

# Decoder Hidden layer with sigmoid activation #2
decoder_out = tf.nn.sigmoid(tf.add
                            (tf.matmul(decoder_in,
                                       weights['decoder_h2']),
                             biases['decoder_b2']))


# Prediction
y_pred = decoder_out
# Targets (Labels) are the input data.
y_true = X

# Define loss and optimizer, minimize the squared error
cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)

# Initializing the variables

# Launch the graph
if __name__ == '__main__':
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        total_batch = int(mnist.train.num_examples/batch_size)
        # Training cycle
        for epoch in range(training_epochs):
            # Loop over all batches
            for i in range(total_batch):
                batch_xs = mnist.train.images[i*batch_size: (i+1) * batch_size]
                # Run optimization op (backprop) and cost op (to get loss value
                _, c = sess.run([optimizer, cost],
                                feed_dict={X: batch_xs})
            # Display logs per epoch step
            if epoch % display_step == 0:
                print("Epoch:", '%04d' % (epoch+1),
                      "cost=", "{:.9f}".format(c))

        print("Optimization Finished!")

        # Applying encode and decode over test set
        encode_decode = sess.run(
            y_pred, feed_dict={X: mnist.test.images})

        print(encode_decode.shape)

        # save noisy data
        with open(NOISY_ARR, 'wb') as file:
            pickle.dump(np.array(encode_decode), file)

        encode_decode = sess.run(
            y_pred, feed_dict={X: mnist.train.images})

        print(encode_decode.shape)
        # save noisy data
        with open(NOISY_ARR_TRAIN, 'wb') as file:
            pickle.dump(np.array(encode_decode), file)

        #  encode_decode_train = sess.run(
        #      y_pred, feed_dict={X: mnist.train.images})

        #  with open(NOISY_ARR, 'wb') as file:
        #      pickle.dump(np.array(encode_decode), file)
        # Compare original images with their reconstructions
        #  f, a = plt.subplots(2, 10, figsize=(10, 2))
        #  for i in range(examples_to_show):
        #      a[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))
        #      a[1][i].imshow(np.reshape(encode_decode[i], (28, 28)))
