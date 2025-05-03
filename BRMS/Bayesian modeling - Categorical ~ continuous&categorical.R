### Predict categorical y with continuous and/or categorical x ###


## Packages

# install.packages(c("data.table", "coda", "mvtnorm", "devtools", "MASS", "ellipse", "rstan", "BayesFactor", "BEST", "coda", "LearnBayes", "markdown", "mcmc", "MCMCpack", "MuMIn", "reshape2", "rmarkdown", "brms", "tidyverse", "tidybayes", "bayesplot", "shinystan", "lme4", "patchwork"), dependencies = TRUE)
# install.packages("ks")

# my_packages <- c("data.table", "coda", "mvtnorm", "devtools", "MASS", "ellipse", "rstan", "BayesFactor", "coda", "LearnBayes", "mcmc", "MCMCpack", "MuMIn", "reshape2", "tidybayes", "bayesplot", "shinystan", "patchwork")
# lapply(my_packages, require, character.only = TRUE)   

# download and install on windows: JAGS-4.2.0-Rtools33
# install.packages("rjags") # In case of trouble, write in C:\Users\vguigon\Documents\.R\Makevars.win the following: JAGS_ROOT=c:/Progra~1/JAGS/JAGS-4.2.0
# install.packages("BEST")

# Be wise in choosing the ROPE (detect right placement with: rope(post$b_variable, range = "default", ci = 1, ci_method = "HDI", verbose = TRUE))

library(brms) 
library(modelr)
library(rethinking)
library(ks)
library(tidyverse)
library(tidybayes)
library(modelr)
library(mcmc)
library(bayesplot)
library(bayestestR)
library(see)
library(logspline)
library(rjags)
library(BEST)
# library(cmdstanr)

# More functions at: https://mc-stan.org/rstanarm/reference/index.html
# Graphical posterior predictive checks: https://mc-stan.org/rstanarm/reference/pp_check.stanreg.html
# More info on ordinal logistic regression at: https://stats.idre.ucla.edu/r/dae/ordinal-logistic-regression/
# Or here: https://mvuorre.github.io/brms-workshop/posts/ordinal/

## Data

d1 <- read.csv("morale.csv")
head(d1, 10)

# IN CASE : rescaling predictors (e.g.)
# d1$gender <- ifelse(d1$gender == "F", -0.5, 0.5)                        # reweight binomial variable 'gender'
# d1$mother <- scale(d1$mother) %>% as.numeric                            # rescale variable: calculate mean+SD of the vector then scale each element by subtracting mean and dividing by SD
# d1 <- d1 %>% mutate(mother.s = (mother - mean(mother) ) / sd(mother) )  # in case of standardization
# d1 <- d1 %>% mutate(mother.c = (mother - mean(mother) )                 # in case of data to center



# create a "gender" variale
d1 <- d1 %>% mutate(gender = male)
d1$gender <- ifelse (d1$gender == 0, "male", "female")

# distribution of responses
d1$response %>% table %>%
  plot(xlab = "response", ylab = "", cex.axis = 2, cex.lab = 2)

d1 %>%
  ggplot(aes(x = response) ) +
  geom_histogram() +
  facet_wrap(~gender) +
  scale_x_continuous(breaks = 1:10, limits = c(1, 10) )

d1 %>%
  ggplot(aes(x = age, y = response) ) +
  geom_point(alpha = 0.2) +
  geom_smooth(method = "lm", colour = "black") +
  facet_wrap(~gender)



## I. Set models 

# backend = "cmdstanr" -> if convergence is too slow or too buggy)

# priors = ~ 1 -> complete pooling (1 common intercept); 
# priors = ~ 0 + factor(cafe) -> no pooling (0 common intercept); 
# priors = ~ 1 + factor(cafe) -> partial pooling (1 common intercept + 1 intercept per cafe): allows shrinkage
# LKJ priors accounts for correlation coefficient; (1 + days || subject) fixes correlation between intercept and days to 0 

# if family = cumulative("logit") -> only intercept is acceptable as prior
# otherwise, can use intercept + sd (if random structure) + sigma

# cumulative logit
m1 <- brm(
  response ~ 1,
  sample_prior = "yes", 
  iter = 2000,
  warmup = 1000,
  thin = 1,
  data = d1,
  family = cumulative("logit"),
  control = list(adapt_delta = 0.8)) # adjusting the delta step size (default=.8) for better samples (slower computing)

m2 <- brm(
  response ~ 1 + action + intention + contact,
  sample_prior = "yes", 
  iter = 2000,
  warmup = 1000,
  thin = 1,
  data = d1,
  family = cumulative("logit"),
  control = list(adapt_delta = 0.8))


# normal (but stupid solution given curve fitting)
d2 <- d1
d2$action <- ifelse(d2$action == "0", -0.5, 0.5) 
d2$intention <- ifelse(d2$intention == "0", -0.5, 0.5) 
d2$contact <- ifelse(d2$contact == "0", -0.5, 0.5) 

priors <- c(
  prior(normal(3.5, 2.5), class = Intercept),
  prior(cauchy(0, 10), class = sd),
  prior(cauchy(0, 10), class = sigma)) # sd 
m3 <- brm(
  response ~ 1 + action + intention + contact + (1 | id),
  prior = priors,
  sample_prior = "yes", 
  iter = 2000,
  warmup = 1000,
  thin = 1,
  data = d2,
  control = list(adapt_delta = 0.8))

pp_check(m3, nsamples = 1e2) 



## Interlude: get model prior and posterior distributions

mod <- m2 # Chose the model you want to assess
prior <- prior_samples(mod)
post <- posterior_samples(mod)
head(prior)
head(post)



## II. Prior predictive checking

prior_summary(mod)



## III. Convergence diagnostics (if fail: go back to I. / modify fitting parameters. Check Prior distributions @end)
mod %>%
  plot(
    pars = "^b_",
    combo = c("dens_overlay", "trace"), widths = c(1, 1.5),
    theme = theme_bw(base_size = 14, base_family = "Open Sans")  )



## IV. Posterior predictive checking (if fail: go back to I. or add data. Check Prior distributions @end)
# Graphical posterior predictive checks: https://mc-stan.org/rstanarm/reference/pp_check.stanreg.html

conditional_effects(mod, categorical = TRUE)

pp_check(mod, nsamples = 1e2)+
  labs(x = "Morality", y = "Proportion")

pp_check(moral2, nsamples = 1e2, type = "bars", prob = 0.95, freq = FALSE) +
  scale_x_continuous(breaks = 1:7) +
  labs(x = "Morality", y = "Proportion")

posterior_summary(mod, pars = c("^b_", "sigma"), probs = c(0.025, 0.975) )



## V. Model comparisons

m1 <- add_criterion(m1, "waic")
m2 <- add_criterion(m2, "waic")
m3 <- add_criterion(m3, "waic")
# computation time is high
# else
brms::waic(m1, m2, m3) # all slopes are negative: meaning that each factor reduces the mean response. These slopes represent changes in cumulative log-odds
# m2 is the best model when comparing waic but m3 is the best when comparing elpd_diff


# 1. WAIC with Leave-One-Out
model_comparison_table <- loo_compare(m1, m2, m3, criterion = "waic") %>%
  data.frame %>%
  rownames_to_column(var = "model")

# 2. Akaike weights
weights <- data.frame(weight = model_weights(m1, m2, m3, weights = "waic") ) %>%
  round(digits = 3) %>%
  rownames_to_column(var = "model")
# weights gives which model generates distributions that fits the best data given other models: Akaike weight

# 3. Bayes factor (high computation time) -> Better use bayes_factor (in-factor) to compare hypotheses about models
bayes_R2(mod) # Predictions are treated as continuous variables in 'bayes_R2' which is likely invalid for ordinal families. 
hist(bayes_R2(mod, summary = FALSE), showMode = TRUE, xlab = expression(rho))
plotPost(bayes_R2(mod, summary = FALSE) - bayes_R2(m2, summary = FALSE) ) # compare R2 between m3 and m2

# 4. Summary
left_join(model_comparison_table, weights, by = "model")
summary(mod) # marginal means in odds ratio

dev.m1 <- mean(-2 * rowSums(log_lik(m1) ) )
dev.m2 <- mean(-2 * rowSums(log_lik(m2) ) )
dev.m3 <- mean(-2 * rowSums(log_lik(m3) ) )
deviances <- c(dev.m1, dev.m2, dev.m3)
comparison <- model_comparison_table %>% data.frame %>% select(waic) %>% rownames_to_column()
waics <- comparison %>% arrange(rowname) %>% pull(waic)

# 5. Interpreting relative effect
fixed_effects <- fixef(mod) # fixed effects extraction
rethinking::logistic(fixed_effects) # == plogis(fixed_effects)
# Proportion of change on the odds induced by predictor
hist(exp(post$b_action), xlab = "Odds ratio") 

res = rope(exp(post$b_action), range = "default", ci = 1, ci_method = "HDI", verbose = TRUE)
plot(res) # check position of ROPE, chose the value as compVal in plotPost
plotPost(exp(post$b_action), ROPE = c(res$ROPE_low, res$ROPE_high), compVal = 0, xlab = "Odds ratio") # similar

# 6. Interpreting absolute effect
intercept_samples <- plogis(post$b_action) 
# Effective impact on probabilities when predictor changes by 1 unit
hist(exp(intercept_samples), xlab = "Effect of action on response") 

res = rope(intercept_samples, range = "default", ci = 1, ci_method = "HDI", verbose = TRUE)
plot(res) # check position of ROPE, chose the value as compVal in plotPost
plotPost(intercept_samples, ROPE = c(res$ROPE_low, res$ROPE_high), compVal = 0, xlab = "Effect of action on response") 



# Optional: R2 for parameter (for y~b_param+sigma)
beta <- post$b_action
sigma <- post$sigma
f1 <- beta^2 * var(d1$height)
rho <- f1 / (f1 + sigma^2)
hist(rho, showMode = TRUE, xlab = expression(rho))



## VI. Visualize posterior predictions

conditional_effects(mod, categorical = TRUE)

intercept_samples <- plogis(post$b_Intercept)

# Represent model predictions
marg1 <- marginal_effects(mod, "action", ordinal = TRUE)
p1 = plot(marg1, theme = theme_bw(base_size = 20, base_family = "Open Sans"), plot = FALSE)[[1]]

marg2 <- marginal_effects(moral2, "intention", ordinal = TRUE)
p2 = plot(marg2, theme = theme_bw(base_size = 20, base_family = "Open Sans"), plot = FALSE)[[1]]

marg3 <- marginal_effects(moral2, "contact", ordinal = TRUE)
p3 = plot(marg3, theme = theme_bw(base_size = 20, base_family = "Open Sans"), plot = FALSE)[[1]]


library(patchwork)
p1 + p2 + p3 + plot_layout(guides = "collect") & theme(legend.position = "right")









## Tools: Prior and Posterior checking

# PRIOR
# Visualize distribution of prior parameters
H.scv <- Hscv(x = prior, verbose = TRUE)
fhat_prior <- kde(x = prior, H = H.scv, compute.cont = TRUE)

# Nothing to see, really






# POSTERIOR
# Visualize distribution of posterior parameters
H.scv <- Hscv(post[, 1:2])
fhat_post <- kde(x = post[, 1:2], H = H.scv, compute.cont = TRUE)

# Nothing to see, really


# Posterior samples check (with Autocorrelation for each parameter)
post <- posterior_samples(mod, add_chain = TRUE)
post %>% mcmc_acf(pars = vars(`b_Intercept[1]`:b_contact), lags = 10)

# Multicolinearity check (pain in the eyes; to check along with MCMC chains convergence)
pairs(mod)






# MODEL

# Continuous predictors
# Represent model predictions (specifically for 1 variable)
age.seq <- data.frame(age = seq(from = min(d1$age), to = max(d1$age), 
                                length.out = 1e2) ) # create a vector of all possible values for y

mu <- data.frame(
  fitted(mod, newdata = age.seq, probs = c(0.025, 0.975)) ) %>% 
  bind_cols(age.seq) # we get the model predictions for those values: we obtain mu

pred_height <- data.frame(
  predict(mod, newdata = age.seq, probs = c(0.025, 0.975) )) %>%
  bind_cols(age.seq) # we obtain sigma for the model

d1 %>%
  ggplot(aes(x = age, y = height) ) +
  geom_point(colour = "white", fill = "black", pch = 21, size = 3, alpha = 0.8) +
  geom_ribbon(
    data = pred_height, aes(x = age, ymin = Q2.5, ymax = Q97.5),
    alpha = 0.2, inherit.aes = FALSE  ) +
  geom_smooth(
    data = mu, aes(y = Estimate, ymin = Q2.5, ymax = Q97.5),
    stat = "identity", color = "black", alpha = 0.8, size = 1  )



# Categorical predictors: absolute effect for all parameters
model_predictions <- fitted(mod) %>%
  data.frame() %>% 
  bind_cols(d1) %>%
  mutate(gender = factor(gender), action = factor(action), intention = factor(intention),
         contact = factor(contact) )



# Model averaging 
new_data <- data.frame(
  age = seq(from = min(d1$age), to = max(d1$age), length.out = 30),
  mass = 4.5) # grid of values for which we will generate predictions

averaged_predictions <- pp_average(
  m1, m2, m3,
  weights = "waic",
  method  = "fitted",
  newdata = new_data
) %>%
  as.data.frame() %>%
  bind_cols(new_data) # predictions averaged on 4 models






# PARAMETER
# Calculate Highest Density Interval (HDI) for the most probable values of the parameter + ROPE

# Example with distribution of 1 parameter taken from samples taken from the posterior
samples <- sample(post, size = 1e3, replace = TRUE) # if need to: sample the posterior distribution

set.seed(666)
p_grid <- seq(from = 0, to = 1, length.out = 1e3) # check that [From To] values are correct
pParam <- dbeta(p_grid, 3, 10) # change parameter according to needs (check that code is working)
massVec <- pParam / sum(pParam)
samples <- sample(p_grid, size = 1e4, replace = TRUE, prob = pParam)

hist(samples, credMass = 0.89, cex = 1.5, xlab = expression(param), xlim = c(0, 1) )

# Visualize ROPE 
res = rope(samples, range = "default", ci = 0.95, ci_method = "HDI", verbose = TRUE)
plot(res) # Ugly as ****


plotPost(
  samples, cex = 2, cex.axis = 1.5, cex.lab = 2,
  xlab = expression(param),
  ROPE = c(res$ROPE_low, res$ROPE_high), compVal = 0)






# HYPOTHESIS TEST
# Difference in response probability between male and female 

# SOLUTION 1
# male is gender as dummy variable

m4 <- brm(
  response ~ 1 + male + age,
  family = cumulative(link = "logit"),
  sample_prior = "yes", 
  iter = 2000,
  warmup = 1000,
  thin = 1,
  data = d1)

# 1. Chose the parameter you want to test
(hyp1 <- hypothesis(m4, "male = 0") )

# 2. Plot posterior vs prior
plot(hyp1, plot = FALSE, theme = theme_bw(base_size = 20, base_family = "Open Sans") )[[1]] +
  geom_vline(xintercept = 0, linetype = 2) +
  coord_cartesian(xlim = c(0, 1) )

# 3. Compare prior and posterior for that parameter
data.frame(prior = hyp1$prior_samples$H1, posterior = hyp1$samples$H1) %>%
  gather(type, value) %>%
  mutate(type = factor(type, levels = c("prior", "posterior") ) ) %>%
  ggplot(aes(x = value) ) +
  geom_histogram(bins = 50, alpha = 0.8, col = "white", fill = "steelblue") +
  geom_vline(xintercept = 0, lty = 2, size = 1) +
  facet_wrap(~type, scales = "free") +
  labs(x = expression(beta[male]), y = "Nombre d'échantillons")

# 4. Histogram for probability of response in males compared to females
post <- posterior_samples(m4)

# below is the version for binomial data; find a solution for categorical (cumulative) ?
p.resp.male <- plogis(post$b_Intercept + post$b_male) # b_Intercept doesn't exist
p.resp.female <- plogis(post$b_Intercept)             # b_Intercept doesn't exist
diff.resp <- p.resp.male - p.resp.female
hist(diff.surviv)



# SOLUTION 2: [bayestestr bayes_factor: HDI+ROPE ; point_estimate (savage-dickey) + plots]

# Point estimate (Savage-Dickey)
BF_param = bayesfactor_parameters(mod, null = 0) # compares prior and posterior samples at 1 point
plot(BF_param)

BFrope_parram=bf_rope(mod)
BFrope_parram # reads as any BF, must be >1 
plot(BF_param)


# SOLUTION 3: compare model 1 with effect of interest against model 2 without effect of interest






## Multi-levels example

prior1 <- c(
  prior(normal(0, 1), class = Intercept, coef = ""),
  prior(lkj(2), class = cor))
mod1 <- brm(
  response ~ 1 + age + (1 + gender | id), 
  family = cumulative(link = "logit"),
  prior = prior1,
  data = d1,
  sample_prior = TRUE,
  warmup = 2000, iter = 1e4,
  control = list(adapt_delta = 0.8))



# Other example of multi-level

d4 <- get(data(UCBadmit) )

d4 %>%
  ggplot(aes(x = dept, y = admit / applications) ) +
  geom_bar(stat = "identity") +
  facet_wrap(~ applicant.gender) +
  labs(x = "Département", y = "Probabilité d'admission")

d4$gender <- ifelse(d4$applicant.gender == "female", -0.5, 0.5)
d4$dept_id <- coerce_index(d4$dept) # create an index for department

p3 <- c(
  prior(normal(0, 10), class = "Intercept"),
  prior(normal(0, 1), class = "b"),
  prior(cauchy(0, 2), class = "sd"),
  prior(lkj(2), class = "cor"))

m3 <- brm(
  admit | trials(applications) ~ 1 + gender + (1 + gender | dept_id),
  data = d4, family = binomial,
  prior = p3,
  warmup = 1000, iter = 5000,
  control = list(adapt_delta = 0.99, max_treedepth = 12),
  backend = "cmdstanr")

d4 %>%
  group_by(dept_id, applications) %>%
  data_grid(gender = seq_range(gender, n = 1e2) ) %>%
  add_fitted_samples(m3, newdata = ., n = 100, scale = "linear") %>%
  mutate(estimate = plogis(estimate) ) %>%
  ggplot(aes(x = gender, y = estimate, group = .iteration) ) +
  geom_hline(yintercept = 0.5, lty = 2) +
  geom_line(aes(y = estimate, group = .iteration), size = 0.5, alpha = 0.2) +
  facet_wrap(~dept_id, nrow = 2)