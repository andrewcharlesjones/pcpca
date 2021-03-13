data {
  int<lower=0> n;          // number of foreground samples
  int<lower=0> m;          // number of background samples
  int<lower=0> p;          // number of features
  int<lower=0> k;          // latent dim
  matrix[m, p] Y;          // background data
  matrix[n, p] X;          // foreground data
  real<lower=0> gamma;     // foreground data
}
transformed data {
  matrix[p, p] C;
  C = 1.0 * X' * X - gamma * 1.0 * Y' * Y;
}
parameters {
  matrix[k, p] W;
  real<lower=0.1> sigma2;
}
transformed parameters {
  matrix[p, p] A;
  A = W' * W + sigma2 * diag_matrix(rep_vector(1, p));
}
model {
  target += -(n - gamma * m) * 0.5 * log_determinant(A) - 0.5 * trace(inverse(A) * C);
}