# Bayesian modeling 

# Install R (v >= 3.5.3) + RStudio (v >= 4.0) 
# + Stan, from RStudio, using package 'rstan' (install info here: https://github.com/stan-dev/rstan/wiki/RStan-Getting-Started.
# + package rethinking (not available on CRAN): 
# devtools::install_github("rmcelreath/rethinking")

# install.packages(c("data.table", "coda", "mvtnorm", "devtools", "MASS", "ellipse", "rstan", "BayesFactor", "BEST", "coda", "LearnBayes", "markdown", "mcmc", "MCMCpack", "MuMIn", "reshape2", "rmarkdown", "brms", "tidyverse", "tidybayes", "bayesplot", "shinystan", "lme4", "patchwork"), dependencies = TRUE)
# install.packages("ks")

# download and install on windows: JAGS-4.2.0-Rtools33
# install.packages("rjags")
# install.packages("BEST")



# my_packages <- c("data.table", "coda", "mvtnorm", "devtools", "MASS", "ellipse", "rstan", "BayesFactor", "coda", "LearnBayes", "mcmc", "MCMCpack", "MuMIn", "reshape2", "tidybayes", "bayesplot", "shinystan", "patchwork")
# lapply(my_packages, require, character.only = TRUE)   



########################################
### Useful functions
## generate a Stan code for the model
# make_stancode(formula, ...)
# stancode(fit)

## define priors
# get_prior(formula, ...)
# set_prior(prior, ...)

## get model predictions
# fitted(fit, ...)
# predict(fit, ...)
# conditional_effects(fit, ...)

# posterior predictive checking
# pp_check(fit, ...)

## model comparisons
# loo(fit1, fit2, ...)
# bayes_factor(fit1, fit2, ...)
# model_weights(fit1, fit2, ...)

## hypothesis testing
# hypothesis(fit, hypothesis, ...)

## fixed effect estimates from model
# fixef(mod)

## to apply an inverse function (ex:inverse link function)
# mutate(p = brms::inv_logit_scaled(Intercept) ) %>% 

## BEST::plotPost() to replace hist()


########################################
## Bayesian with BRMS
library(rethinking)
library(tidyverse)
library(brms)
library(ks)

data(Howell1)
d <- Howell1

# Solution 1: use default priors
# mod <- brm(height ~ 1, data = d2)
# posterior_summary(mod, pars = c("^b_", "sigma"), probs = c(0.025, 0.975) )
# get_prior(height ~ 1, data = d2) 



# Solution 2: use own priors
priors <- c(
  prior(normal(120, 20), class = Intercept),
  prior(normal(0, 10), class = b),
  prior(exponential(0.01), class = sigma)
)

# can center the responses:   d2$weight.c <- d2$weight - mean(d2$weight) 
# can standardize the scores: d2 <- d2 %>% mutate(weight.s = (weight - mean(weight) ) / sd(weight) )

mod <- brm(
  height ~ 1 + weight,
  prior = priors,
  family = gaussian(),
  data = d, 
  sample_prior = "yes", 
  chains = 4, # nombre de MCMCs
  iter = 2000, # nombre total d'itérations (par chaîne)
  warmup = 1000, # nombre d'itérations pour le warm-up
  thin = 1 # thinning (1 = no thinning)
)



## Summary
summary(mod)
post <- posterior_samples(mod) 
fixef(mod) 



## Prior predictive checking (activate "yes" to sample_prior in brms mod)
prior <- prior_samples(mod) # getting the samples from the prior distribution
head(prior)

# Prior distribution (got from model estimation)
data.frame(x = rnorm(4000, prior$Intercept + prior$b, prior$sigma) ) %>%
  ggplot(aes(x) ) +
  geom_histogram() +
  labs(x = "weight", y = "height")

# Prior Predictions
prior %>% 
  sample_n(size = 1e2) %>% 
  rownames_to_column("draw") %>% 
  expand(nesting(draw, Intercept, b), a = c(-2, 2) ) %>%
  mutate(d = Intercept + b * a) %>% 
  ggplot(aes(x = a, y = d)) +
  geom_line(aes(group = draw), color = "steelblue", size = 0.5, alpha = 0.5) +
  labs(x = "weight", y = "height")



## Get and visualize distribution of SAMPLES FROM the posterior, highlighted by density
mutate(density = get_density(b_Intercept, sigma, n = 1e2) ) #PB 1: GET_DENSITY
head(post)

mu <- post$b_Intercept + post$b_weight # Get the mean value
quantile(x = mu, probs = c(0.025, 0.5, 0.975) ) 
t(sapply(post[, 1:2], quantile, probs = c(0.025, 0.5, 0.975) ) ) # Get the median and the 95% credible interval

ggplot(post, aes(x = b_Intercept, y = sigma) ) +
  geom_point(size = 2, , pch = 21, alpha = 0.5, color = "white", fill = "black", show.legend = FALSE) +
  labs(x = expression(mu), y = expression(sigma) ) +
  viridis::scale_color_viridis()
# Values at the center are the values with the greatest density
# if get_density is resolved, can add parameter: ,colour = density in line119 after sigma


## Visualize THE posterior distribution of p(mu, sigma)
H.scv <- Hscv(post[, 1:2])
fhat_post <- kde(x = post[, 1:2], H = H.scv, compute.cont = TRUE)

plot(fhat_post, display = "persp", col = "purple", border = NA,
     xlab = "\nmu", ylab = "\nsigma", zlab = "\np(mu, sigma)",
     shade = 0.8, phi = 30, ticktype = "detailed",
     cex.lab = 1.2, family = "Helvetica")

# Posterior distribution: marginal distributions of mu and sigma
  hist(post$b_Intercept + post$b_weight, breaks = 40, xlab = expression(mu))
  hist(post$sigma, breaks = 40, xlab = expression(mu))


  
## Represent model predictions + Represent uncertainty on µ via fitted() + incorporate sigma (intervals of prediction)
weight.seq <- data.frame(weight = seq(from = 0, to = 70, length.out = 1e2) ) # on crée un vecteur de valeurs possibles pour "weight"

mu <- data.frame( 
  fitted(mod, newdata = weight.seq, probs = c(0.025, 0.975) )) %>%
  bind_cols(weight.seq) # on récupère les prédictions du modèle pour ces valeurs de poids

pred_height <- data.frame(
  predict(mod, newdata = weight.seq, probs = c(0.025, 0.975) )) %>%
  bind_cols(weight.seq)

head(pred_height)

d %>%
  ggplot(aes(x = weight, y = height) ) +
  # Can plot with scores standardized: ggplot(aes(x = weight.s, y = height) ) +
  geom_point(colour = "white", fill = "black", pch = 21, size = 3, alpha = 0.8) +
  geom_ribbon(
    data = pred_height, aes(x = weight, ymin = Q2.5, ymax = Q97.5),
    alpha = 0.2, inherit.aes = FALSE) +
  geom_smooth(
    data = mu, aes(y = Estimate, ymin = Q2.5, ymax = Q97.5),
    stat = "identity", color = "black", alpha = 0.8, size = 1) # PB3: RESOLVE Q2.5

# OR (adapt the code to the data)
nd <- data.frame(weight = seq(from = 0, to = 70, length.out = 1e2) )

posterior_samples(mod, pars = "^b_") %>%
  sample_n(size = 1e2) %>%
  rownames_to_column("draw") %>%
  expand(nesting(draw, b_Intercept, b_weight), a = c(0, 70) ) %>%
  mutate(d = b_Intercept + b_weight * a) %>%
  ggplot(aes(x = a, y = d) ) +
  geom_point(data = d, aes(x = weight, y = height), size = 2) +
  geom_line(aes(group = draw), color = "purple", size = 0.5, alpha = 0.5) +
  labs(x = "weight", y = "height")


## Get effect size of R²
post <- posterior_samples(mod)
beta <- post$b_weight
sigma <- post$sigma

f1 <- beta^2 * var(d$weight)
rho <- f1 / (f1 + sigma^2)

bayes_R2(mod)
hist(bayes_R2(mod, summary = FALSE), showMode = TRUE, xlab = expression(rho))
# OR
hist(rho, showMode = TRUE, xlab = expression(rho))
summary(lm(height ~ weight, data = d2) )$r.squared

## Posterior prediction checking
pp_check(mod, type = "intervals", nsamples = 1e2, prob = 0.5, prob_outer = 0.95) +
  labs(x = "weight", y = "height")

pp_check(mod, nsamples = 1e2) + labs(x = "weight", y = "height")



## MCMC evaluation - implementation via BRMS
# combo can be hist, dens, dens_overlay, trace, trace_highlight...
# cf. https://mc-stan.org/bayesplot/reference/MCMC-overview.html

# Chain convergence
plot(
  x = mod, combo = c("dens_overlay", "trace"),
  theme = theme_bw(base_size = 16, base_family = "Open Sans")
)

# Autocorrelation
post <- posterior_samples(mod, add_chain = TRUE)
post %>% mcmc_acf(pars = vars(b_Intercept:sigma), lags = 10)

post %>% # rank plots
  mcmc_rank_overlay(pars = vars(b_Intercept:sigma) ) +
  labs(x = "Rang", y = "Fréquence") +
  coord_cartesian(ylim = c(25, NA) )


pairs(mod)






########################################################
# Pour prédicteurs catégoriels, c'est une autre paire de manches. A voir plus tard
# Construire un nouveau script ?
# Construire une nouvelle branche pour inclure du catégoriel dans un mod2 ?
# (plot de conditional_effects, etc.)


# Faire 1 script par situation:
# - celui ci pour du continu, avec suite pour inclure interactions + catégoriel
# - 1 beta-binomial
# - 1 régression linéaire avec fonction logit
