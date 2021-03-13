library(ggplot2)
library(magrittr)
library(rstan)


frac_exceeding_list <- c()

n_list <- c(10, 100, 1000, 10000, 100000)
for (n in c(n_list)) {
  
  
  ## Generate data
  # n <- 100000
  m <- n
  p <- 2
  mu = c(0, 0)
  bg_population_cov = matrix(data = c(2.7, 2.6, 2.6, 2.7), nrow = p)
  fg_population_cov = matrix(data = c(2.7, 0.6, 0.6, 2.7), nrow = p)
  gamma <- 0.3
  k <- 1
  beta <- 0.5
  
  y <- MASS::mvrnorm(n = m, mu, bg_population_cov)
  x <- MASS::mvrnorm(n = n, mu, fg_population_cov)
  
  X <- x %>% 
    as.data.frame() %>% 
    set_colnames(c("x1", "x2"))
  Y <- y %>% 
    as.data.frame() %>% 
    set_colnames(c("x1", "x2"))
  
  dat <- rbind(X, Y)
  dat['condition'] <- c(rep("Foreground", n), rep("Background", m))
  
  ## Compute risk minimizer W*, sigma^2*
  C <- beta * fg_population_cov - (1 - beta) * gamma * bg_population_cov
  eig_decomp <- eigen(C)
  if ((p == 2) & (k == 1)) {
    sigma2_star <- 1 / (beta - (1 - beta) * gamma) * eig_decomp$values[2]
  } else {
    sigma2_star <- 1 / (beta - (1 - beta) * gamma) * mean(eig_decomp$values[k+1:length(eig_decomp$values)])
  }
  # W_star <- eig_decomp$vectors[,1] * sqrt(eig_decomp$values[1:k] - sigma2_star)
  W_star <- eig_decomp$vectors[,1] * sqrt(1 / (beta - (1 - beta) * gamma) * eig_decomp$values[1:k] - sigma2_star)
  A_star <- outer(W_star, W_star) + sigma2_star * diag(p)
  A_star_inv <- solve(A_star)
  
  compute_risk(W_star, sigma2_star)
  print(sigma2_star)
  
  
  ## Fit Gibbs posterior
  pcpca_data <- list(
    n = n,
    m = m,
    p = p,
    k = k,
    X = X,
    Y = Y,
    gamma = gamma
  )
  
  sm <- rstan::stan_model(file = "~/Documents/beehive/pcpca/experiments/simulation/gibbs_posterior/pcpca_gibbs.stan")
  fit <- rstan::sampling(sm, data = pcpca_data, chains = 1, iter = 10000)
  
  samples <- rstan::extract(fit)
  W_samples <- samples$W[,1,]
  sigma2_samples <- samples$sigma2
  
  ## Compute divergence of sample risk from risk minimizer
  compute_risk <- function(W, sigma2) {
    A <- outer(W, W) + sigma2 * diag(p)
    A_inv <- solve(A)
    risk <- 0.5 * (beta - gamma * (1 - beta)) * msos::logdet(A) + 0.5 * sum(diag(A_inv %*% C))
    return(risk)
  }
  
  compute_divergence <- function(W, sigma2) {
    empirical_risk <- compute_risk(W, sigma2)
    div <- sqrt(empirical_risk - risk_star)
    return(div)
  }
  
  risk_star <- compute_risk(W_star, sigma2_star)
  
  div_list <- c()
  for (ii in seq(length(sigma2_samples))) {
    curr_div <- compute_divergence(W_samples[ii,], sigma2_samples[ii])
    div_list <- c(div_list, curr_div)
  }
  
  risk_list <- c()
  for (ii in seq(length(sigma2_samples))) {
    curr_risk <- compute_risk(W_samples[ii,], sigma2_samples[ii])
    risk_list <- c(risk_list, curr_risk)
  }
  
  ## Find number of samples larger than threshold eps
  epsilon <- (n + m)**(-0.5)
  num_exceeding <- sum(div_list > epsilon)
  fraction_exceeding <- num_exceeding / length(div_list)
  div_list %>% as.data.frame() %>% 
    set_colnames("divergence") %>% 
    ggplot(aes(divergence)) + 
    geom_histogram() + 
    theme_bw() + 
    geom_vline(xintercept = epsilon)
  
  print(fraction_exceeding)
  
  frac_exceeding_list <- c(frac_exceeding_list, fraction_exceeding)
  
  # dat <- rbind(X, Y)
  # dat['condition'] <- c(rep("Foreground", n), rep("Background", m))
  # dat %>% ggplot(aes(x1, x2, color=condition)) +
  #   geom_point() +
  #   theme_bw() +
  #   geom_abline(slope = W_samples[1:500,2] / W_samples[1:500,1], alpha = 0.1)
  
  
  
  
}

plot(n_list, frac_exceeding_list, log='x')




