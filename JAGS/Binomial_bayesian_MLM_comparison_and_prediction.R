##### VARIABLES #####

# Free memory
rm(list = ls(all.names = TRUE)) # Will clear all objects including hidden objects.
gc() # Free up memory and report the memory usage

## If not installed, uncomment and run the following commands to avoid compilation errors:
# remove.packages(c("StanHeaders", "rstan"))
# install.packages("StanHeaders", repos = c("https://mc-stan.org/r-packages/", getOption("repos")))
# install.packages("rstan", repos = "https://cloud.r-project.org/", dependencies = TRUE)
# verify the installation: example(stan_model, package = "rstan", run.dontrun = TRUE)
# In case of need: install.packages("cmdstanr", repos = c("https://mc-stan.org/r-packages/", getOption("repos")))
# then install the rethinking package: devtools::install_github("rmcelreath/rethinking")

# package BEST has been removed from CRAN, here is how to install it:
# library(devtools)
# install_url('https://cran.r-project.org/src/contrib/Archive/BEST/BEST_0.5.4.tar.gz')

list.of.packages <- c("brms", "rstan", "rjags", "mcmc", "rethinking", "StanHeaders"
                      , "tidyverse", "modelr", "ks", "bayestestR", "logspline"
                      , "tidybayes", "bayesplot", "BEST", "ggplot2", "see", "GGally"
                      , "dplyr", "scales"
                      , "rprojroot")
{
  new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
  if(length(new.packages)) install.packages(new.packages)
  lapply(list.of.packages, require, character.only = TRUE)
}

# Set paths and make necessary directories
set.seed(12345)
project_root = find_rstudio_root_file()

# Generate sample data
set.seed(123)
num_subjects <- 100  # Number of subjects
sample_data <- data.frame(
  Subject = factor(rep(1:num_subjects, each = 10)),
  Outcome = rbinom(10 * num_subjects, 1, 0.5),
  FactorA = factor(sample(c("Level1", "Level2"), 10 * num_subjects, replace = TRUE)),
  Covariate1 = rnorm(10 * num_subjects),
  Covariate2 = rnorm(10 * num_subjects)
)

# Scaled data
sample_data_scaled <- sample_data
sample_data_scaled$Covariate1 <- scales::rescale(sample_data_scaled$Covariate1, to = c(0, 1))
sample_data_scaled$Covariate2 <- scales::rescale(sample_data_scaled$Covariate2, to = c(0, 1))

## Notes:
# Caution: output models in log-odds -> set function to retransform
# When Bulk Effective Samples Size (ESS) is too low, posterior means and medians may be unreliable.
#     Running the chains for more iterations may help.
#     See: http://mc-stan.org/misc/warnings.html#bulk-ess
# Same for tail

########## MODELS OF SUCCESS ##########

##### PRIORS #####

prior_random <- c(
  prior(beta(0.5, 0.5), class = Intercept))

priorSubj <- c(
  prior(normal(0, 1), class = Intercept),
  prior(cauchy(0, 1), class = sd))

priorLM <- c(
  prior(normal(0, 1), class = Intercept),
  prior(normal(0, 1), class = b),
  prior(cauchy(0, 1), class = sd))

##### MODELS #####

mRandom <- brm(
  Outcome | trials(1) ~ (1|Subject),
  family = binomial(link = "logit"),
  prior = prior_random,
  sample_prior = "yes",
  chains = 4,
  iter = 4000,
  warmup = 1500,
  thin = 1,
  data = sample_data,
  control = list(adapt_delta = 0.85))

mSubject <- brm(
  Outcome | trials(1) ~ 1 + (1|Subject),
  family = binomial(link = "logit"),
  prior = priorSubj,
  sample_prior = "yes",
  iter = 4000,
  warmup = 1500,
  thin = 1,
  data = sample_data,
  control = list(adapt_delta = 0.85))

mCovariate1 <- brm(
  Outcome | trials(1) ~ 1 + Covariate1 + (1|Subject),
  family = binomial(link = "logit"),
  prior = priorLM,
  sample_prior = "yes",
  iter = 4000,
  warmup = 1500,
  thin = 1,
  data = sample_data,
  control = list(adapt_delta = 0.85))

mCovariate2 <- brm(
  Outcome | trials(1) ~ 1 + Covariate2 + (1|Subject),
  family = binomial(link = "logit"),
  prior = priorLM,
  sample_prior = "yes",
  iter = 4000,
  warmup = 1500,
  thin = 1,
  data = sample_data,
  control = list(adapt_delta = 0.85))

# Perform k-fold cross validation
kf <- kfold(mCovariate1, save_fits = TRUE, chains = 1)

# Define a loss function
mean_squared_error <- function(y, yrep) {
  mean((y - yrep)^2)
}

log_loss <- function(y, yrep) {
  epsilon <- 1e-15  # A small constant to prevent taking the log of zero
  yrep <- pmax(epsilon, pmin(1 - epsilon, yrep))  # Clip predicted values to (epsilon, 1-epsilon)
  -mean(y * log(yrep) + (1 - y) * log(1 - yrep))
}

# Predict responses and evaluate the loss
kfp <- kfold_predict(kf)
mean_squared_error(y = kfp$y, yrep = kfp$yrep)
log_loss(y = kfp$y, yrep = kfp$yrep)

##### MODELS COMPARISON #####

mRandom <- add_criterion(mRandom, "waic")
mSubject <- add_criterion(mSubject, "waic")
mCovariate1 <- add_criterion(mCovariate1, "waic")
mCovariate2 <- add_criterion(mCovariate2, "waic")

mRandom <- add_criterion(mRandom, "loo")
mSubject <- add_criterion(mSubject, "loo")
mCovariate1 <- add_criterion(mCovariate1, "loo")
mCovariate2 <- add_criterion(mCovariate2, "loo")

## Hypothesis testing
# General models
model_comparison_table <- loo_compare(mRandom, mSubject,
                                              mCovariate1, mCovariate2,
                                              criterion = "waic") %>%
  data.frame %>%
  rownames_to_column(var = "model")

# Akaike weights
weights <- data.frame(weight = model_weights(mRandom, mSubject,
                                                     mCovariate1, mCovariate2,
                                                     weights = "waic") ) %>%
  round(digits = 3) %>%
  rownames_to_column(var = "model")

# waic
comparison <- model_comparison_table %>% data.frame %>% select(waic) %>% rownames_to_column()
waics <- comparison %>% arrange(rowname) %>% pull(waic)

# Ranking
left_join(model_comparison_table, weights, by = "model")

# Summary of the winner model
summary(mCovariate1) # marginal means in odds ratio
