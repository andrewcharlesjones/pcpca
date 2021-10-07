import functools
import warnings

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

from tensorflow_probability import bijectors as tfb
from tensorflow_probability import distributions as tfd

tf.enable_v2_behavior()

warnings.filterwarnings('ignore')

def probabilistic_pca(data_dim, latent_dim, num_datapoints, stddv_datapoints, w):
  # w = yield tfd.Normal(loc=tf.zeros([data_dim, latent_dim]),
  #                scale=2.0 * tf.ones([data_dim, latent_dim]),
  #                name="w")
  z = yield tfd.Normal(loc=tf.zeros([latent_dim, num_datapoints]),
                 scale=tf.ones([latent_dim, num_datapoints]),
                 name="z")
  x = yield tfd.Normal(loc=tf.matmul(w, z),
                       scale=stddv_datapoints,
                       name="x")


num_datapoints = 500
data_dim = 2
latent_dim = 1
stddv_datapoints = 0.5

# w = tf.Variable(np.random.normal(size=[data_dim, latent_dim]).astype(np.float32))
w_true = tf.Variable(np.array([[5.], [5.]]).astype(np.float32))
concrete_ppca_model = functools.partial(probabilistic_pca,
    data_dim=data_dim,
    latent_dim=latent_dim,
    num_datapoints=num_datapoints,
    stddv_datapoints=stddv_datapoints,
    w=w_true)

model = tfd.JointDistributionCoroutineAutoBatched(concrete_ppca_model)
actual_z, x_train = model.sample()


w = tf.Variable(tf.random.normal([data_dim, latent_dim]))
print(w)
concrete_ppca_model = functools.partial(probabilistic_pca,
    data_dim=data_dim,
    latent_dim=latent_dim,
    num_datapoints=num_datapoints,
    stddv_datapoints=stddv_datapoints,
    w=w)

model = tfd.JointDistributionCoroutineAutoBatched(concrete_ppca_model)
target_log_prob_fn = lambda z: model.log_prob((z, x_train))

# qw_mean = tf.Variable(tf.random.normal([data_dim, latent_dim]))
qz_mean = tf.Variable(tf.random.normal([latent_dim, num_datapoints]))
# qw_stddv = tfp.util.TransformedVariable(1e-4 * tf.ones([data_dim, latent_dim]),
#                                         bijector=tfb.Softplus())
qz_stddv = tfp.util.TransformedVariable(
    1e-4 * tf.ones([latent_dim, num_datapoints]),
    bijector=tfb.Softplus())
def factored_normal_variational_model():
  # qw = yield tfd.Normal(loc=qw_mean, scale=qw_stddv, name="qw")
  qz = yield tfd.Normal(loc=qz_mean, scale=qz_stddv, name="qz")

surrogate_posterior = tfd.JointDistributionCoroutineAutoBatched(
    factored_normal_variational_model)

losses = tfp.vi.fit_surrogate_posterior(
    target_log_prob_fn,
    surrogate_posterior=surrogate_posterior,
    optimizer=tf.optimizers.Adam(learning_rate=0.05),
    num_steps=1000)
print(w)

plt.scatter(x_train.numpy()[0, :], x_train.numpy()[1, :])
plt.axis([-20, 20, -20, 20])
plt.show()
import ipdb; ipdb.set_trace()




