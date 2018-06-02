from __future__ import print_function, division
from builtins import range, input

import util
import numpy as np
import theano
import theano.tensor as T
import matplotlib.pyplot as plt

from theano.tensor.shared_randomstreams import RandomStreams


class DenseLayer(object):
    def __init__(self, M1, M2, f=T.nnet.relu):
        self.W = theano.shared(np.random.randn(M1, M2) * 2 / np.sqrt(M1))
        self.b = theano.shared(np.zeros(M2))
        self.f = f
        self.params = [self.W, self.b]

    def forward(self, X):
        return self.f(X.dot(self.W) + self.b)


class VariationalAutoencoder(object):
    def __init__(self, D, hidden_layer_sizes):
        # D - размерность входных данных
        # hidden_layer_sizes - содержат размерность входных данных каждого слоя Encoder
        # до скрытого слоя Z - включительно
        # Слои декодера идут в обратном порядке
        self.X = T.matrix("X")
        # ----------------------------------------- Encoder ------------------------------------------
        self.encoder_layers = []
        M_in = D
        for M_out in hidden_layer_sizes[:-1]:
            h = DenseLayer(M_in, M_out)
            self.encoder_layers.append(h)
            M_in = M_out
        M = hidden_layer_sizes[-1]  # Последний слой
        h = DenseLayer(M_in, M * 2, f=lambda x: x)
        self.encoder_layers.append(h)

        # Вычисление mean и variance / stdev переменной Z
        # Внимание! variance должна быть > 0. Мы можем получить sigma (standart dev) > 0 из переменной
        # путем вычисления от нее функции softplus(...) и добавления небольщого положительного значения
        current_layer_value = self.X
        for layer in self.encoder_layers:
            current_layer_value = layer.forward(current_layer_value)
        self.means = current_layer_value[:, :M]  # Первая половина выходных значений
        self.stddev = current_layer_value[:, M:] + 1e-6  # Вторая половина выходных значений
        # Получаем образец из распределения Z
        self.rnd = RandomStreams()
        eps = self.rnd.normal((self.means.shape[0], M))  # Генерируем образце из Normal(0,1)
        self.Z = self.means + self.stddev * eps  # Делаем репараметризацию
        # ----------------------------------------- Decoder ------------------------------------------
        self.decoder_layers = []
        M_in = M
        for M_out in reversed(hidden_layer_sizes[:-1]):
            h = DenseLayer(M_in, M_out)
            self.decoder_layers.append(h)
            M_in = M_out
        h = DenseLayer(M_in, D, f=T.nnet.sigmoid)

        self.decoder_layers.append(h)
        current_layer_value = self.Z
        for layer in self.decoder_layers:
            current_layer_value = layer.forward(current_layer_value)
        self.posterior_predictive_probs = current_layer_value
        self.posterior_predictive = self.rnd.binomial(
            size=self.posterior_predictive_probs.shape,
            n=1,
            p=self.posterior_predictive_probs
        )
        #*** Сгенерировать образец Z ~ Normal(0,1) ***
        # prior_predictive_probs и prior_predictive
        Z_std = self.rnd.normal((1, M))
        current_layer_value = Z_std
        for layer in self.decoder_layers:
            current_layer_value = layer.forward(current_layer_value)
        self.prior_predictive_probs = current_layer_value
        self.prior_predictive = self.rnd.binomial(
            size=self.prior_predictive_probs.shape,
            n=1,
            p=self.prior_predictive_probs
        )
        #*** Сгенерировать образец из входных данных, используется только для визуализации ***
        # prior_predictive_probs_form_Z_input
        Z_input = T.matrix("Z_input")
        current_layer_value = Z_input
        for layer in self.decoder_layers:
            current_layer_value = layer.forward(current_layer_value)
        self.prior_predictive_probs_form_Z_input = current_layer_value

        #*** Вычисление функции стоимости ***
        # now build the cost
        # https://stats.stackexchange.com/questions/7440/kl-divergence-between-two-univariate-gaussians
        # https://stats.stackexchange.com/questions/60680/kl-divergence-between-two-multivariate-gaussians
        kl = -T.log(self.stddev) + 0.5*(self.stddev**2 + self.means**2) - 0.5
        kl = T.sum(kl, axis=1)
        expected_log_likelihood = - T.nnet.binary_crossentropy(
            output=self.posterior_predictive_probs,
            target=self.X,
        )
        expected_log_likelihood = T.sum(expected_log_likelihood, axis=1)
        self.elbo = T.sum(expected_log_likelihood - kl)
        #*** Выстраивание набора параметров ***
        params = []
        for layer in self.encoder_layers:
            params += layer.params
        for layer in self.decoder_layers:
            params += layer.params
        grads = T.grad(-self.elbo, params)

        # RMSProp
        decay = 0.9
        learning_rate = 0.001
        cache = [theano.shared(np.ones_like(p.get_value())) for p in params]
        new_cache = [decay * c + (1 - decay) * g * g for p, c, g in zip(params, cache, grads)]
        updates = [
                      (c, new_c) for c, new_c in zip(cache, new_cache)
                      ] + [
                      (p, p - learning_rate * g / T.sqrt(new_c + 1e-10)) for p, new_c, g in
                      zip(params, new_cache, grads)
                      ]
        self.train_op = theano.function(
            inputs=[self.X],
            outputs=self.elbo,
            updates=updates
        )
        self.posterior_predictive_sample = theano.function(
            inputs=[self.X],
            outputs=self.posterior_predictive
        )
        self.posterior_predictive_sample_with_probs = theano.function(
            inputs=[],
            outputs=[self.posterior_predictive, self.posterior_predictive_sample]
        )
        self.transform = theano.function(
            inputs=[self.X],
            outputs=self.means
        )
        self.prior_predictive_with_input = theano.function(
            inputs=[Z_input],
            outputs=prior_predictive_probs_from_Z_input
        )