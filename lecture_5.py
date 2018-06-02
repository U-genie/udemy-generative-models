import numpy as np
from scipy.stats import multivariate_normal as mvn
# Генерация данных из Гауссовского распределения
X = np.random.randn()

# Генерация данных из распределения Бернулли
X = np.random.binomial(n=1, p=0.5)

# Байесовский генератор
# Исходные данные
train_data = np.random.randn(1000)
target_data = np.round(np.random.rand(1000))
K = [0, 1]
mean_for_y = np.zeros(len(K))
cov_for_y = np.zeros(len(K))
for k in K:
    class_y = train_data[np.where(target_data == k)]
    mean_for_y[k] = np.mean(class_y)
    cov_for_y[k] = np.cov(class_y)
for k in K:
    print("Class %d : mean %f, covariance %f" % (k, mean_for_y[k], cov_for_y[k]))

# Алгоритм 1. Простой - генерация только на основе P(x/y)
np.random.seed(3)
X = mvn.rvs(mean=mean_for_y[1], cov=cov_for_y[1])
print("Algorithm 1: Generated value %f " % X)

# Алгоритм 2. Генерация на основе совместного распределения P(x,y) = P(y)*P(x/y)
y = np.random.choice(K)
np.random.seed(3)
X = mvn.rvs(mean=mean_for_y[y], cov=cov_for_y[y])
print("Algorithm 2: Generated value %f " % X)
