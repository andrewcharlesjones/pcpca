data {
  int<lower=0> n;          // number of samples 
  int<lower=0> p;          // number of features 
  int<lower=0> k;          // latent dim
  matrix[n, p] X;          // data
}
parameters {
  matrix[n, k] z; 
  matrix[k, p] W;
}
model {
  for (j in 1:k) {
    target += normal_lpdf(z[,j] | 0.0, 1.0);              // prior
  }

  for (j in 1:p) {
    target += normal_lpdf(X[,j] | z * to_vector(W[,j]), 1);      // likelihood
  }
}
