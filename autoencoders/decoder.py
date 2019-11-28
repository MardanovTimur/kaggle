import pickle

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

from autoencoder import (NOISY_ARR, NOISY_ARR_TRAIN, X, batch_size,
                         decoder_out, learning_rate, mnist, n_input, weights)

display_step = 10
training_epochs = 100

true_positives = tf.placeholder('float', [None, n_input])
y_true = X

cost = tf.reduce_mean(tf.pow(decoder_out - true_positives, 2))
optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)


def get_data(file_name):
    data = []
    with open(file_name, 'rb') as file:
        data = pickle.load(file)
    return data


if __name__ == '__main__':
    # read data
    noisy_train = get_data(NOISY_ARR_TRAIN)
    noisy_test = get_data(NOISY_ARR)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        total_batch = int(mnist.train.num_examples/batch_size)
        # Training cycle
        for epoch in range(training_epochs):
            # Loop over all batches
            for i in range(total_batch):
                noisy_batch = noisy_train[i*batch_size: (i+1) * batch_size]
                batch_xs = mnist.train.images[i*batch_size: (i+1) * batch_size]

                if batch_xs.shape[0] != noisy_batch.shape[0]:
                    continue

                _, c = sess.run([optimizer, cost],
                                feed_dict={
                                    X: noisy_batch,
                                    true_positives: batch_xs,
                })
            # Display logs per epoch step
            if epoch % display_step == 0:
                print("Epoch:", '%04d' % (epoch+1),
                      "cost=", "{:.9f}".format(c))

        # Applying encode and decode over test set
        encode_decode = sess.run(
            decoder_out, feed_dict={X: noisy_test[:20]})

        f, a = plt.subplots(3, 10, figsize=(10, 3))
        for i in range(10):
            a[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))
            a[1][i].imshow(np.reshape(noisy_test[i], (28, 28)))
            a[2][i].imshow(np.reshape(encode_decode[i], (28, 28)))

        f.show()
        plt.draw()
        plt.show()
