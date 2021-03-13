library(ggplot2)
library(magrittr)
library(rstan)

### PPCA ----------

## Generate data
n <- 20
p <- 2
mu = c(0, 0)
cov = matrix(data = c(2.7, -2.6, -2.6, 2.7), nrow = 2)
x <- MASS::mvrnorm(n = n, mu, cov) %>% 
  as.data.frame() %>% 
  set_colnames(c("x1", "x2"))
C = cov(X)
pca_data <- list(
  n = n,
  p = p,
  k = 1,
  X = x
)

## Fit model
fit1 <- rstan::stan(
  file = "~/Documents/beehive/pcpca/pca_gibbs.stan",
  data = pca_data,
  chains = 1,
  warmup = 1000,
  iter = 4000,
  cores = -1
)

samples <- rstan::extract(fit1)
mean_W <- colMeans(samples$W[,1,])

x %>% ggplot(aes(x1, x2)) +
  geom_point() +
  theme_bw() + 
  geom_abline(slope = samples$W[1:200,1,1] / samples$W[1:200,1,2], alpha = 0.1)

### PCPCA ----------

n_list <- c(20, 50, 200)

for (n in n_list) {
  
  ## Generate data
  m <- n
  p <- 2
  mu_y = c(0, 0)
  mu_x1 = c(-2, 2)
  mu_x2 = c(2, -2)
  cov = matrix(data = c(2.7, 2.6, 2.6, 2.7), nrow = p)
  y <- MASS::mvrnorm(n = m, mu_y, cov)
  x1 <- MASS::mvrnorm(n = n/2, mu_x1, cov)
  x2 <- MASS::mvrnorm(n = n/2, mu_x2, cov)
  
  X <- rbind(x1, x2) %>% 
    as.data.frame() %>% 
    set_colnames(c("x1", "x2"))
  Y <- y %>% 
    as.data.frame() %>% 
    set_colnames(c("x1", "x2"))
  
  pcpca_data <- list(
    n = n,
    m = m,
    p = p,
    k = 1,
    X = X,
    Y = Y,
    gamma = 0.4
  )
  
  ## Fit model
  sm <- rstan::stan_model(file = "~/Documents/beehive/pcpca/experiments/simulation/gibbs_posterior/pcpca_gibbs.stan")
  fit1 <- rstan::sampling(sm, data = pcpca_data, chains = 1, iter = 10000)
  
  samples <- rstan::extract(fit1)
  mean_W <- colMeans(samples$W[,1,])
  
  dat <- rbind(X, Y)
  dat['condition'] <- c(rep("Foreground", n), rep("Background", m))
  dat %>% ggplot(aes(x1, x2, color=condition)) +
    geom_point() +
    theme_bw() +
    geom_abline(slope = samples$W[1:200,1,1] / samples$W[1:200,1,2], alpha = 0.1)
  
  write.csv(x = samples$W[,1,], file = sprintf("~/Documents/beehive/pcpca/experiments/simulation/gibbs_posterior/out/W_samples_%d.csv", n))
  write.csv(x = dat, file = sprintf("~/Documents/beehive/pcpca/experiments/simulation/gibbs_posterior/out/data_%d.csv", n))
  
}

