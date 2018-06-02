# Generative Adversarial Nets in TensorFlow
[1] https://wiseodd.github.io/techblog/2016/09/17/gan-tensorflow/
[2] https://github.com/wiseodd/generative-models

Ссылка на статью:
[3] Ian J. Goodfellow Generative Adversarial Nets http://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf

## Generative Adversarial Nets

Рассматривается пример с фальшивомонетчиком и полицейским. Выделяется конфликт между преступником и полицеским, поскольку цель преступника - обмануть полицейского, а цель полицейского распознать фальшивые деньги.
Этот тип ситуации моделируется minimax-игрой в теории игр. И это процесс называется Adversarial Process (Соревновательный процесс).

**Generative Adversarial Nets (GAN)** - это специальный случай соревновательного процесса, в котором компоненты ("преступник" и "полицейский") - это нейронные сети:
* *первая сеть*  - генерирует данные
* *вторая сеть* - пытается определить разницу между реальными данным и ложными данными, сгенерированными первой сетью. Выход второй сети - это скаляр [0, 1], который представляет вероятность данных.
В GAN - первая сеть называется генератор (Generator Net) $G(Z)$, а вторая сеть - дискриминатор (Discriminator Net), $D(X)$.

$$
\min\limits_{G} \max\limits_{D} V(D,G) = \mathbf{E}_{x\sim p_{data(x)}}[log\space D(x)] + \mathbf{E}_{z\sim p_{z(z)}}[1 - log\space D(G(z))]
$$

В точке равновесия, которая является оптимальной точкой в минимаксной игре, первая сеть будет моделировать реальные данные, а вторая будет выдавать вероятность 0.5  если первая сеть дает реальные данные.

**Зачем тренировать GAN:** Это необходимо поскольку распределение вероятностей данных $P_{data}$ может быть очень сложным распределением и очень сложным и трудным для вывода. Таким образом, было бы очень перспективно получить генеративную машину,  которая может генерировать образцы из $P_{data}$ без работы со сложным распределением вероятностей.
Если мы получаем такую машину, мы можем использовать ее в другом процессе, который требует образцы из распределения $P_{data}$, поскольку мы можем получить образцы относительно дешево используя обученную Generative Net.

## GAN Implementation
By the definition of GAN, we need two nets. This could be anything, be it a sophisticated net like convnet or just a two layer neural net. Let’s be simple first and use a two layer nets for both of them. We’ll use TensorFlow for this purpose.

```python
# Discriminator Net
X = tf.placeholder(tf.float32, shape=[None, 784], name='X')
D_W1 = tf.Variable(xavier_init([784, 128]), name='D_W1')
D_b1 = tf.Variable(tf.zeros(shape=[128]), name='D_b1')
D_W2 = tf.Variable(xavier_init([128, 1]), name='D_W2')
D_b2 = tf.Variable(tf.zeros(shape=[1]), name='D_b2')
theta_D = [D_W1, D_W2, D_b1, D_b2]
# Generator Net
Z = tf.placeholder(tf.float32, shape=[None, 100], name='Z')
G_W1 = tf.Variable(xavier_init([100, 128]), name='G_W1')
G_b1 = tf.Variable(tf.zeros(shape=[128]), name='G_b1')
G_W2 = tf.Variable(xavier_init([128, 784]), name='G_W2')
G_b2 = tf.Variable(tf.zeros(shape=[784]), name='G_b2')
theta_G = [G_W1, G_W2, G_b1, G_b2]

def generator(z):
    G_h1 = tf.nn.relu(tf.matmul(z, G_W1) + G_b1)
    G_log_prob = tf.matmul(G_h1, G_W2) + G_b2
    G_prob = tf.nn.sigmoid(G_log_prob)
    return G_prob

def discriminator(x):
    D_h1 = tf.nn.relu(tf.matmul(x, D_W1) + D_b1)
    D_logit = tf.matmul(D_h1, D_W2) + D_b2
    D_prob = tf.nn.sigmoid(D_logit)
    return D_prob, D_logit
```

Тут ```generator(z)``` принимает вектор 100-dimensional и возвращает вектор 786-dimensional, который является картинкой MNIST (28x28). Здесь z - это приор для $G(Z)$. In a way it learns a mapping between the prior space to $P_{data}$.

Далее ```discriminator(x)``` принимает изображение MNIST и возвращает скаляр, который представляет вероятность того, что изображение - это реальное MNIST-изображение.
Теперь мы опишем Adversarial Process для обучения этой GAN:
> Алгоритм 1: Алгорим minibatch стохастического градиентного спуска генеративных соревновательных сетей. Число шагов k  - гиперпараметр. В алгоритме используется k=1, как величина делающая алгоритм наименее трудоемким.
```
for number of training iterations do
  for k steps do
     - Сгенерировать minibatch {z(1)...z(m)} из m noise сэмплов из noise prior p_g(z)
     - Выбрать minibatch {x(1)...x(m)} из m реальных сэмплов из распределения данных p_data(x)
     - Обновить дискриминатор путем подъема по стохастическому градиенту (1)
  end for
  - Сгенерировать minibatch {z(1)...z(m)} из m noise сэмплов из noise prior p_g(z)
  - Обновить генератор путем схождения по стохастическому градиенту (2)
end for
```
> Обновления генератора и дискриминатов, основанные на градиенте могут использовать любое стандартное градиентное правило. В экспериментах [3] использовался momentum.

Формула (1):
$$
\nabla_{{\theta}_d} \frac1m \sum \limits_{i=1}^m [log D(x^{(i)}) + log(1 - D(G(z^{(i)})))]
$$

Формула 2:
$$
\nabla_{{\theta}_g} \frac1m \sum \limits_{i=1}^m [log(1 - D(G(z^{(i)})))]
$$

```python
G_sample = generator(Z)
D_real, D_logit_real = discriminator(X)
D_fake, D_logit_fake = discriminator(G_sample)

D_loss = -tf.reduce_mean(tf.log(D_real) + tf.log(1. - D_fake))
G_loss = -tf.reduce_mean(tf.log(D_fake))
```
Тут используется для функции потерь используется -1, поскольку функция потерь должна быть максимизирована, в то время как TensorFlow может делать только минимизацию.
Кроме того, как предлагается в работе [3], лучше не минимизировать  ```tf.reduce_mean(1 - tf.log(D_fake))``` (как в формуле 2), а максимизировать ```tf.reduce_mean(tf.log(D_fake))```.

Мы обучаем сети одну за одной соревновательным обучением с использованием вышеописанной функции.

```python
# Only update D(X)'s parameters, so var_list = theta_D
D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list=theta_D)
# Only update G(X)'s parameters, so var_list = theta_G
G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=theta_G)

def sample_Z(m, n):
    '''Uniform prior for G(Z)'''
    return np.random.uniform(-1., 1., size=[m, n])

for it in range(1000000):
    X_mb, _ = mnist.train.next_batch(mb_size)
    _, D_loss_curr = sess.run([D_solver, D_loss], feed_dict={X: X_mb, Z: sample_Z(mb_size, Z_dim)})
    _, G_loss_curr = sess.run([G_solver, G_loss], feed_dict={Z: sample_Z(mb_size, Z_dim)})
```

# NIPS 2016 Workshop
https://sites.google.com/site/nips2016adversarial/
