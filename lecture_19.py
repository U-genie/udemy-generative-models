from __future__ import print_function, division
from builtins import range, input

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

st = tf.contrib.bayesflow.stochastic_tensor
Normal = tf.contrib.distributions.Normal

# Выборка N сэмплов из N(5,3*3)
N = 10000
mean = np.ones(N)*5
scale = np.ones(N)*3


with st.value_type(st.SampleValue()):
    X = st.StochasticTensor(Normal(loc=mean, scale=scale))
# Это потому, что StochasticTensor нельзя напрямую сделать session run
I = tf.Variable(np.ones(N))
Y = I * X # Поэтому считается просто тензор путем уножения X (StochasticTensor) на переменную I (Variable)
init_op = tf.global_variables_initializer()
with tf.Session() as session:
    session.run(init_op)
    Y_val = session.run(Y)
    print("Sample mean : ", Y_val.mean())
    print("Sample stdev : ", Y_val.std())

plt.hist(Y_val, bins=20)
plt.show()
