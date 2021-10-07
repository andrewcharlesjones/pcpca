data {
  int<lower=0> n;          // number of samples 
  int<lower=0> p;          // number of features 
  int<lower=0> k;          // latent dim
  matrix[n, p] X;          // data
}
transformed data {
 matrix[p, p] C;
 C = 1.0/n * X' * X;
}
parameters {
  matrix[k, p] W;
  real<lower=0> sigma2;
}
transformed parameters {
  cov_matrix[p] A;
  A = W' * W + sigma2 * diag_matrix(rep_vector(1, p));
}
model {
  target += -n/2.0 * log_determinant(A) - 0.5 * trace(inverse(A) * C);
}
