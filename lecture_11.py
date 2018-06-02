from __future__ import print_function, division
from builtins import range, input
import util
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


class Autoencoder:

    def __init__(self, D, M):
        # D - размерность входных и выходных данных
        # M - размерность скрытого слоя
        # Представление входных данных
        self.X = tf.placeholder(tf.float32, shape=(None, D))
        # Представление выходы -> скрытый слой
        self.W = tf.Variable(tf.random_normal(shape=(D, M))*2/np.sqrt(M))
        self.b = tf.Variable(np.zeros(M).astype(np.float32))
        # Представление скрытый слой -> выходы
        self.V = tf.Variable(tf.random_normal(shape=(M, D))*2/np.sqrt(D))
        self.c = tf.Variable(np.zeros(D).astype(np.float32))
        # Создание реконструкции
        self.Z = tf.nn.relu(tf.matmul(self.X, self.W)+self.b)
        logits = tf.matmul(self.Z, self.V) + self.c
        self.X_hat = tf.nn.sigmoid(logits)

        # Вычисление функции стоимости
        self.cost = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.X, logits = logits))
        # Computes the sum of elements across dimensions of a tensor.
        # Reduces input_tensor along the dimensions given in axis. Unless keepdims is true,
        # the rank of the tensor is reduced by 1 for each entry in axis. If keepdims is true,
        #  the reduced dimensions are retained with length 1.
        # If axis has no entries, all dimensions are reduced, and a tensor with a single element is returned.
        # For example:
        # x = tf.constant([[1, 1, 1], [1, 1, 1]])
        # tf.reduce_sum(x)  # 6
        self.train_op = tf.train.RMSPropOptimizer(learning_rate=0.001).minimize(self.cost)
        self.init_op = tf.global_variables_initializer()
        self.sess = tf.InteractiveSession()
        self.sess.run(self.init_op)

    def fit(self, X, epochs = 30, batch_size = 64):
        costs = []
        n_batches = len(X) // batch_size
        print("Fit function:")
        print("n_batches = %d" % n_batches)
        for i in range(epochs):
            print("-> Epoch #%d" % i)
            np.random.shuffle(X)
            for j in range(n_batches):
                batch = X[j*batch_size:(j+1)*batch_size]
                _, c, = self.sess.run((self.train_op, self.cost), feed_dict={self.X: batch})
                # c - это значение функции стоимости, сохраняется для того, чтобы построить график
                c /= batch_size
                costs.append(c)
                if j % 100 == 0:
                    print("--> Iter %d, cost = %.3f" % (j,c))
        plt.plot(costs)
        plt.show()

    def predict(self, X):
        return self.sess.run(self.X_hat, feed_dict={self.X: X})

if __name__ == "__main__":
    # Пример из лекции
    X, Y = util.get_mnist()
    model = Autoencoder(28*28, 300)
    model.fit(X)
    # Plot reconstruction
    done = False
    # create a new plot
    my_figures = []
    while not done:
        i = np.random.choice(len(X))
        x = X[i]
        im = model.predict([x]).reshape(28, 28)
        x = x.reshape(28, 28)
        plt.subplot(1, 2, 1)
        plt.imshow(x, cmap="gray")
        plt.title("Original")
        plt.subplot(1, 2, 2)
        plt.imshow(im, cmap="gray")
        plt.title("Reconstruction")
        plt.show()

        ans = input("Generate another? [y/n]")
        if ans and ans[0] in ("n" or "N"):
            done = True
