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


class CLVM():
    def __init__(self, n_bg, n_fg, data_dim, latent_dim_shared=1, latent_dim_fg=1):
        self.latent_dim_shared = latent_dim_shared
        self.latent_dim_fg = latent_dim_fg
        self.n_bg = n_bg
        self.n_fg = n_fg
        self.data_dim = data_dim

        self.init_flag = False

    def model(self, noise_variance_bg, noise_variance_fg, S, W):

        ## Latent variables
        zy = yield tfd.Normal(loc=tf.zeros([self.latent_dim_shared, self.n_bg]),
                     scale=tf.ones([self.latent_dim_shared, self.n_bg]),
                     name="zy")
        zx = yield tfd.Normal(loc=tf.zeros([self.latent_dim_shared, self.n_fg]),
                     scale=tf.ones([self.latent_dim_shared, self.n_fg]),
                     name="zx")
        tx = yield tfd.Normal(loc=tf.zeros([self.latent_dim_fg, self.n_fg]),
                     scale=tf.ones([self.latent_dim_fg, self.n_fg]),
                     name="tx")

        ## Data
        y = yield tfd.Normal(loc=tf.matmul(S, zy),
                           scale=noise_variance_bg,
                           name="y")
        x = yield tfd.Normal(loc=tf.matmul(S, zx) + tf.matmul(W, tx),
                           scale=noise_variance_fg,
                           name="x")

    def init_model(self, S=None, W=None, noise_variance_bg=None, noise_variance_fg=None):

        if S is None:
            S = tf.Variable(tf.random.normal([self.data_dim, self.latent_dim_shared]))
        if W is None:
            W = tf.Variable(tf.random.normal([self.data_dim, self.latent_dim_fg]))
        if noise_variance_bg is None:
            noise_variance_bg = tfp.util.TransformedVariable(
                1e-4 * tf.ones([1]),
                bijector=tfb.Softplus())
        if noise_variance_fg is None:
            noise_variance_fg = tfp.util.TransformedVariable(
                1e-4 * tf.ones([1]),
                bijector=tfb.Softplus())

        self.S = S
        self.W = W
        self.noise_variance_bg = noise_variance_bg
        self.noise_variance_fg = noise_variance_fg
        concrete_ppca_model = functools.partial(self.model,
            noise_variance_bg=self.noise_variance_bg,
            noise_variance_fg=self.noise_variance_fg,
            S=self.S,
            W=self.W,
        )

        self.tf_joint_model = tfd.JointDistributionCoroutineAutoBatched(concrete_ppca_model)
        self.init_flag = True

    def sample_data(self):
        if not self.init_flag:
            raise Exception("Need to initialize model first.")

        zy_sampled, zx_sampled, tx_sampled, Y_sampled, X_sampled = self.tf_joint_model.sample()
        return zy_sampled, zx_sampled, tx_sampled, Y_sampled, X_sampled

    def fit_model(self, Y, X, n_iters=1000, learning_rate=0.05):
        if not self.init_flag:
            raise Exception("Need to initialize model first.")

        target_log_prob_fn = lambda zx, zy, ty: self.tf_joint_model.log_prob((zx, zy, ty, Y, X))

        self.qzy_mean = tf.Variable(tf.random.normal([self.latent_dim_shared, self.n_bg]))
        self.qzy_stddv = tfp.util.TransformedVariable(
            1e-4 * tf.ones([self.latent_dim_shared, self.n_bg]),
            bijector=tfb.Softplus())

        self.qzx_mean = tf.Variable(tf.random.normal([self.latent_dim_shared, self.n_fg]))
        self.qzx_stddv = tfp.util.TransformedVariable(
            1e-4 * tf.ones([self.latent_dim_shared, self.n_fg]),
            bijector=tfb.Softplus())

        self.qtx_mean = tf.Variable(tf.random.normal([self.latent_dim_fg, self.n_fg]))
        self.qtx_stddv = tfp.util.TransformedVariable(
            1e-4 * tf.ones([self.latent_dim_fg, self.n_fg]),
            bijector=tfb.Softplus())
        def factored_normal_variational_model():
          qzy = yield tfd.Normal(loc=self.qzy_mean, scale=self.qzy_stddv, name="qzy")
          qzx = yield tfd.Normal(loc=self.qzx_mean, scale=self.qzx_stddv, name="qzx")
          qtx = yield tfd.Normal(loc=self.qtx_mean, scale=self.qtx_stddv, name="qtx")

        surrogate_posterior = tfd.JointDistributionCoroutineAutoBatched(
            factored_normal_variational_model)

        self.losses = tfp.vi.fit_surrogate_posterior(
            target_log_prob_fn,
            surrogate_posterior=surrogate_posterior,
            optimizer=tf.optimizers.Adam(learning_rate=learning_rate),
            num_steps=n_iters)

        # posterior_samples = surrogate_posterior.sample(50)
        # v_posterior_samples = self.tf_joint_model.sample(value=(posterior_samples))

if __name__ == "__main__":
    latent_dim_shared = 1
    latent_dim_fg = 1
    data_dim = 2

    ## Draw data from true model
    clvm_true_model = CLVM()

    S_true = tf.Variable(np.array([[5.], [5.]]).astype(np.float32))
    W_true = tf.Variable(np.array([[-5.], [5.]]).astype(np.float32))
    noise_variance_true = tfp.util.TransformedVariable(np.ones(1).astype(np.float32), bijector=tfb.Softplus())

    clvm_true_model.init_model(S_true, W_true, noise_variance_true)
    zy_sampled, zx_sampled, tx_sampled, Y, X = clvm_true_model.sample_data()


    ## Fit model
    clvm = CLVM()

    S = tf.Variable(tf.random.normal([data_dim, latent_dim_shared]))
    W = tf.Variable(tf.random.normal([data_dim, latent_dim_fg]))
    noise_variance = tfp.util.TransformedVariable(
        1e-4 * tf.ones([1]),
        bijector=tfb.Softplus())

    clvm.init_model(S, W, noise_variance)

    clvm.fit_model(Y, X, n_iters=1000)

    plt.scatter(Y.numpy()[0, :], Y.numpy()[1, :], label="BG")
    plt.scatter(X.numpy()[0, :], X.numpy()[1, :], label="FG")
    plt.axis([-20, 20, -20, 20])
    plt.show()
    import ipdb; ipdb.set_trace()




