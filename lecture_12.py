from __future__ import print_function, division
from builtins import range, input
import util
import numpy as np
import theano
import theano.tensor as T
import matplotlib.pyplot as plt

class Autoencoder:

    def __init__(self, D, M):
        # D - размерность входных и выходных данных
        # M - размерность скрытого слоя
        # Представление входных данных
        self.X = T.matrix("X")
        # Представление выходы -> скрытый слой
        self.W = theano.shared(np.random.randn(D, M) * 2/np.sqrt(M))
        self.b = theano.shared(np.zeros(M))
        # Представление скрытый слой -> выходы
        self.V = theano.shared(np.random.randn(M,D) * 2/np.sqrt(D))
        self.c = theano.shared(np.zeros(D))
        # Создание реконструкции
        self.Z = T.nnet.relu(self.X.dot(self.W)+self.b)
        self.X_hat = T.nnet.sigmoid(self.Z.dot(self.V)+self.c)

        # Вычисление функции стоимости
        self.cost = T.sum(T.nnet.binary_crossentropy(output=self.X_hat, target = self.X))
        # X и X_hat считаются имеющими распределение Бернулли
        # Define the updates
        params = [self.W, self.b, self.V, self.c]
        grads = T.grad(self.cost, params)
        # Параметры обучения rms_prop
        decay = 0.9
        learning_rate = 0.001

        # Переменные для rms_prop
        cache = [theano.shared(np.ones_like(p.get_value())) for p in params]
        new_cache = [decay*c + (1-decay)*g*g for p, c, g in zip(params, cache, grads)]
        updates = [(c, new_c) for c, new_c in zip(cache, new_cache)] + \
                  [(p, p-learning_rate*g/T.sqrt(new_c + 10e-10)) for p, new_c, g in zip(params, new_cache, grads)]

        self.train_op = theano.function(inputs=[self.X], outputs=self.cost, updates=updates)
        self.predict = theano.function(inputs=[self.X], outputs=self.X_hat)

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
                c = self.train_op(batch)
                # c - это значение функции стоимости, сохраняется для того, чтобы построить график
                c /= batch_size
                costs.append(c)
                if j % 100 == 0:
                    print("--> Iter %d, cost = %.3f" % (j,c))
        plt.plot(costs)
        plt.show()

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

