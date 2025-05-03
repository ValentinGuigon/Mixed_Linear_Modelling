##### VARIABLES #####

# Free memory
rm(list = ls(all.names = TRUE)) # Will clear all objects including hidden objects.
gc() # Free up memory and report the memory usage

## If not installed, uncomment and run the following commands to avoid compilation errors:
# remove.packages(c("StanHeaders", "rstan"))
# install.packages("StanHeaders", repos = c("https://mc-stan.org/r-packages/", getOption("repos")))
# install.packages("rstan", repos = "https://cloud.r-project.org/", dependencies = TRUE)
# verify the installation: example(stan_model, package = "rstan", run.dontrun = TRUE)
# devtools::install_github("rmcelreath/rethinking")
# devtools::install_version("BEST", version = "0.5.4", repos = "http://cran.us.r-project.org")
# In case of need: install.packages("cmdstanr", repos = c("https://mc-stan.org/r-packages/", getOption("repos")))

list.of.packages <- c("brms", "rstan", "rjags", "mcmc", "StanHeaders"
                      , "tidyverse", "modelr", "ks", "bayestestR", "logspline"
                      , "tidybayes", "bayesplot", "ggplot2", "see", "GGally"
                      , "dplyr", "scales", "pracma"
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
  Covariate1 = rnorm(10 * num_subjects),
  Covariate2 = rnorm(10 * num_subjects)
)

# Standardize Covariate1
sample_data$Covariate1 = scale(sample_data$Covariate1, center = TRUE, scale = TRUE)

## Notes:
# Caution: output models in log-odds -> set function to retransform
# When Bulk Effective Samples Size (ESS) is too low, posterior means and medians may be unreliable.
#     Running the chains for more iterations may help.
#     See: http://mc-stan.org/misc/warnings.html#bulk-ess
# Same for tail

########## MODELS OF SUCCESS ##########

##### PRIORS #####

priorLM <- c(
  prior(normal(0, 1), class = Intercept),
  prior(normal(0, 1), class = b),
  prior(cauchy(0, 1), class = sd))

##### MODELS #####

mRandom <- brm(
  Outcome | trials(1) ~ (1|Subject),
  family = binomial(link = "logit"),
  prior = priorLM,
  sample_prior = "yes",
  chains = 4,
  iter = 4000,
  warmup = 1500,
  thin = 1,
  data = sample_data,
  control = list(adapt_delta = 0.85),
  save_pars = save_pars(all = TRUE))

mSubject <- brm(
  Outcome | trials(1) ~ 1 + (1|Subject),
  family = binomial(link = "logit"),
  prior = priorLM,
  sample_prior = "yes",
  iter = 4000,
  warmup = 1500,
  thin = 1,
  data = sample_data,
  control = list(adapt_delta = 0.85),
  save_pars = save_pars(all = TRUE))

mCovariate1 <- brm(
  Outcome | trials(1) ~ 1 + Covariate1 + (1|Subject),
  family = binomial(link = "logit"),
  prior = priorLM,
  sample_prior = "yes",
  iter = 4000,
  warmup = 1500,
  thin = 1,
  data = sample_data,
  control = list(adapt_delta = 0.85),
  save_pars = save_pars(all = TRUE))

##### MODELS COMPARISON #####

mRandom <- add_criterion(mRandom, "waic")
mSubject <- add_criterion(mSubject, "waic")
mCovariate1 <- add_criterion(mCovariate1, "waic")

mRandom <- add_criterion(mRandom, "loo")
mSubject <- add_criterion(mSubject, "loo")
mCovariate1 <- add_criterion(mCovariate1, "loo")

## Hypothesis testing
# General models
model_comparison_table <- loo_compare(mRandom, mSubject, mCovariate1,
                                      criterion = "waic") %>%
  data.frame %>%
  rownames_to_column(var = "model")

# Akaike weights
weights <- data.frame(weight = model_weights(mRandom, mSubject, mCovariate1,
                                              weights = "waic") ) %>%
  round(digits = 3) %>%
  rownames_to_column(var = "model")

# waic
comparison <- model_comparison_table %>% data.frame %>% select(waic) %>% rownames_to_column()
waics <- comparison %>% arrange(rowname) %>% pull(waic)

# Ranking
left_join(model_comparison_table, weights, by = "model")

# Looking at the relationship between outcome and covariate1
Covariate1_summary = summary(mCovariate1)
Covariate1_bayesR2 = bayes_R2(mCovariate1)

Covariate1_hyp = hypothesis(mCovariate1, "Covariate1 > 0")
Covariate1_over_int_hyp = hypothesis(mCovariate1, "Intercept - Covariate1 > 0")

bf_covariate1 = bayestestR::bayesfactor(mCovariate1)
bf_covariate1_over_random = bayestestR::bayesfactor(mRandom, mCovariate1)
bf_covariate1_over_intercept = bayestestR::bayesfactor(mSubject, mCovariate1)

parameters_bf_zero = bayesfactor_parameters(mCovariate1, null = 0)
parameters_bf_interval = bayesfactor_parameters(mCovariate1, null = c(-0.1, 0.1))
parameters_bf_right_sided = bayesfactor_parameters(mCovariate1, direction = ">")

Covariate1_si <- si(mCovariate1, BF = 1, verbose = FALSE)

Covariate1_summary
Covariate1_bayesR2
Covariate1_hyp
Covariate1_over_int_hyp
bf_covariate1
bf_covariate1_over_random
bf_covariate1_over_intercept

parameters_bf_zero
effectsize::interpret_bf(exp(parameters_bf_zero$log_BF[2]), include_value = TRUE)
parameters_bf_interval
effectsize::interpret_bf(exp(parameters_bf_interval$log_BF[2]), include_value = TRUE)
parameters_bf_right_sided
effectsize::interpret_bf(exp(parameters_bf_right_sided$log_BF[2]), include_value = TRUE)

rope(mCovariate1, range = c(-0.1, 0.1), parameters = "Covariate1")
Covariate1_si
plot(Covariate1_si)
