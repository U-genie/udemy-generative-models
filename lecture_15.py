from __future__ import print_function, division
from builtins import range, input
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal as mvn


def softplus(x):
    # log1p(x) == log (1+x)
    return np.log1p(np.exp(x))

# Создание нейронной сети с размером слоев (4,3,2)
# Это упрощенная версия декодера

W1 = np.random.randn(4, 3)
W2 = np.random.randn(3, 2*2)  # поскольку каждый выход должен содержать mean мат. ожидание) и stdev (стандартное отклонение)

# Прямой проход делается как обычно в нейронных сетях, за исключением необычной функции softplus
def forward(x, W1, W2):
    hidden = np.tanh(x.dot(W1))
    output = hidden.dot(W2)  # Не используем активационную функцию
    mean = output[:2]  # Берутся элементы 0 и 1
    stdev = softplus(output[2:])  # Берутся элементы 2 и 3
    return mean, stdev

# Генерируем случайные входы
X = np.random.randn(4)

# ===== 1) Вычисление параметров гауссианы =====
mean, stdev = forward(X, W1, W2)

# ===== 2) Выход прямого прохода - не значение, а распределние q(z) выраженное mean и stdev ====
print("Gaussian 1: Mean = %f, Stdev = %f" % (mean[0], stdev[0]))
print("Gaussian 2: Mean = %f, Stdev = %f" % (mean[1], stdev[1]))

# ===== 3) Генерация примеров на основе q(z) =====
# RVS принимает ковариацию
# Ковариация случайной величины с собой равна её дисперсии, а дисперсия = квадрату стандартного отклонения
# Квадратный корень из дисперсии, равный  σ\displaystyle \sigma , называется среднеквадрати́ческим отклоне́нием, станда́ртным отклоне́нием или стандартным разбросом.
samples = mvn.rvs(mean=mean, cov=stdev**2, size=10000)
# Нет описания как обучать веса и как реализовать градиентный спуск

# ===== Вывод примеров на график =====
plt.scatter(samples[:, 0], samples[:, 1], alpha=0.5)
plt.show()
