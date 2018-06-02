from __future__ import print_function, division
from builtins import range, input
import util
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal as mvn

class BayesClassifier:
    K  = int
    gaussian = []
    def fit(self, X, Y):
        # Python also includes a data type for sets. A set is an unordered collection with no duplicate elements.
        self.K = len(set(Y))

        self.gaussian = []
        for k in range(self.K):
            # Тут немного тупо сделано. Если метка класса - не число, то все сломается
            Xk = X[Y == k]
            # Тут сделано тоже тупо. Для многомерного распределения матожидание и дисперсия по каждой размерности вычисляются независимо
            # Что в общем случае может быть не так. Это Naive Bayessian Classifier ? См. конспект, стр. 8
            mean = Xk.mean(axis=0)
            cov = np.cov(Xk.T)
            g = {"m": mean, "c": cov}
            self.gaussian.append(g)

    def sample_given_y(self,y):
        g = self.gaussian[y]
        return mvn.rvs(mean=g["m"], cov=g["c"])

    def sample(self):
        # генерация y на основе P(y)
        # P(y) не было рассчитано. Предполагается, что классы равномерно распредлены в обучающей выборке
        # В действительности - это может быть не так.
        y = np.random.randint(self.K)
        return self.sample_given_y(y)

if __name__ == "__main__":
    # Мой пример со случайными данными
    train_data = np.random.randn(1000)
    target_data = np.round(np.random.rand(1000))
    clf = BayesClassifier()
    clf.fit(train_data, target_data)
    for k in range(clf.K):
        print("Class k = %d, mean = %f, cov = %f" %(k, clf.gaussian[k]["m"], clf.gaussian[k]["c"]))

    # Пример из лекции
    X, Y = util.get_mnist()
    clf = BayesClassifier()
    clf.fit(X,Y)
    for k in range(clf.K):
        mean = clf.gaussian[k]["m"].reshape(28, 28)
        sample = clf.sample_given_y(k).reshape(28, 28)
        plt.subplot(1, 2, 1)
        plt.imshow(mean, cmap="gray")
        plt.title("Computed mean")
        plt.subplot(1, 2, 2)
        plt.imshow(sample, cmap="gray")
        plt.title("Generated sample")
        plt.show()

    #Generating random sample
    random_sample = clf.sample().reshape(28, 28)
    plt.imshow(random_sample, cmap="gray")
    plt.title("Random sample from random class")
    plt.show()
