from __future__ import print_function, division
from builtins import range, input
import util
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal as mvn
from sklearn.mixture import BayesianGaussianMixture



class BayesClassifier:
    K  = int
    gaussian = []
    def fit(self, X, Y):
        # Python also includes a data type for sets. A set is an unordered collection with no duplicate elements.
        self.K = len(set(Y))

        self.gaussian = []
        for k in range(self.K):
            print("BayesClassifier, training class #%d" % k)
            # Тут немного тупо сделано. Если метка класса - не число, то все сломается
            Xk = X[Y == k]
            # Обучение Bayesian c GMM
            gmm = BayesianGaussianMixture(10) # Максимальное количество кластеров 10
            gmm.fit(Xk)
            self.gaussian.append(gmm)

    def sample_given_y(self,y):
        gmm = self.gaussian[y]
        sample = gmm.sample()
        # sample возвращает кортеж:
        # 1) Сам образец
        # 2) К какому кластеру образец принадлежит
        mean_value = gmm.means_[sample[1]]
        # Принадлежность к кластеру используется для определения среднего
        # Среднее находится в переменной means_ (непубличной переменной класса, узнать можно из исходного кода)
        return sample[0].reshape(28, 28), mean_value.reshape(28, 28)

    def sample(self):
        # генерация y на основе P(y)
        # P(y) не было рассчитано. Предполагается, что классы равномерно распредлены в обучающей выборке
        # В действительности - это может быть не так.
        y = np.random.randint(self.K)
        return self.sample_given_y(y)

if __name__ == "__main__":
    # Пример из лекции
    X, Y = util.get_mnist()
    clf = BayesClassifier()
    clf.fit(X, Y)
    for k in range(clf.K):
        # Здесь тоже изменение, поскольку mean теперь возвращается функцией  sample_given_y
        sample, mean = clf.sample_given_y(k)
        plt.subplot(1, 2, 1)
        plt.imshow(mean, cmap="gray")
        plt.title("Computed mean")
        plt.subplot(1, 2, 2)
        plt.imshow(sample, cmap="gray")
        plt.title("Generated sample")
        plt.show()

    #Generating random sample
    random_sample = clf.sample()[0].reshape(28, 28)
    plt.imshow(random_sample, cmap="gray")
    plt.title("Random sample from random class")
    plt.show()
