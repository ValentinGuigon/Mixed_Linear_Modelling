### Predict continuous y with continuous x ###


## Packages

# install JAGS following installation manual: https://sourceforge.net/projects/mcmc-jags/files/Manuals/4.x/
# install.packages(c("data.table", "coda", "mvtnorm", "devtools", "MASS", "ellipse", "rstan", "BayesFactor", "BEST", "coda", "LearnBayes", "markdown", "mcmc", "MCMCpack", "MuMIn", "reshape2", "rmarkdown", "brms", "tidyverse", "tidybayes", "bayesplot", "shinystan", "lme4", "patchwork"), dependencies = TRUE)
# install.packages("ks")

# my_packages <- c("data.table", "coda", "mvtnorm", "devtools", "MASS", "ellipse", "rstan", "BayesFactor", "coda", "LearnBayes", "mcmc", "MCMCpack", "MuMIn", "reshape2", "tidybayes", "bayesplot", "shinystan", "patchwork")
# lapply(my_packages, require, character.only = TRUE)   

# download and install on windows: JAGS-4.2.0-Rtools33
# install.packages("rjags") # In case of trouble, write in C:\Users\vguigon\Documents\.R\Makevars.win the following: JAGS_ROOT=c:/Progra~1/JAGS/JAGS-4.2.0
# install.packages("BEST")

library(rstan)
library(modelr)
library(brms) 
library(rethinking)
library(ks)
library(tidyverse)
library(mcmc)
library(bayesplot)
library(bayestestR)
library(see)
library(logspline)
library(rjags)
library(BEST)

# More functions at: https://mc-stan.org/rstanarm/reference/index.html
# Graphical posterior predictive checks: https://mc-stan.org/rstanarm/reference/pp_check.stanreg.html


## Data

d1 <- read.csv("parents.csv")
head(d1, 10)


# rescaling predictors
d1$mother <- scale(d1$mother) %>% as.numeric     # rescale variable: calculate mean+SD of the vector then scale each element by subtracting mean and dividing by SD
d1$father <- scale(d1$father) %>% as.numeric
# d1 <- d1 %>% mutate(mother.s = (mother - mean(mother) ) / sd(mother) ) # in case of standardization
# d1 <- d1 %>% mutate(mother.c = (mother - mean(mother) )                # in case of data to center


# y ~ x
d1 %>%
  ggplot(aes(x = height, y = mother) ) +
  geom_point(colour = "white", fill = "black", pch = 21, size = 3, alpha = 0.8)

d1 %>%
  ggplot(aes(x = height, y = father) ) +
  geom_point(colour = "white", fill = "black", pch = 21, size = 3, alpha = 0.8)

# y ~ (x + x2 | x3)
d1 %>%
  gather(parent, parent.height, 3:4) %>%
  ggplot(aes(x = parent.height, y = height, colour = parent, fill = parent) ) +
  geom_point(pch = 21, size = 4, color = "white", alpha = 1) +
  stat_smooth(method = "lm", fullrange = TRUE) +
  facet_wrap(~ gender)

# rescaling gender predictors
d1$gender <- ifelse(d1$gender == "F", -0.5, 0.5) # reweight binomial variable 'gender'



## I. Set models 

# backend = "cmdstanr" -> if convergence is too slow or too buggy

# priors = ~ 1 -> complete pooling (1 common intercept); 
# priors = ~ 0 + factor(cafe) -> no pooling (0 common intercept); 
# priors = ~ 1 + factor(cafe) -> partial pooling (1 common intercept + 1 intercept per cafe): allows shrinkage
# LKJ priors accounts for correlation coefficient; (1 + days || subject) fixes correlation between intercept and days to 0 

p1 <- c(
  prior(normal(70, 10), class = Intercept),
  prior(cauchy(0, 10), class = sigma))
m1 <- brm(
  height ~ 1 + mother,
  prior = p1,
  sample_prior = "yes", 
  iter = 2000,
  warmup = 1000,
  thin = 1,
  data = d1,
  cores = parallel::detectCores(),
  control = list(adapt_delta = 0.8)) # adjusting the delta step size (default=.8) for better samples (slower computing)

p2 <- c(
  prior(normal(70, 10), class = Intercept),
  prior(normal(0, 10), class = b),
  prior(cauchy(0, 10), class = sigma))
m2 <- brm(
  height ~ 1 + gender + mother + father,
  prior = p2,
  sample_prior = "yes", 
  iter = 2000,
  warmup = 1000,
  thin = 1,
  data = d1,
  cores = parallel::detectCores(),
  control = list(adapt_delta = 0.8))

p3 <- c(
  prior(normal(70, 10), class = Intercept),
  prior(normal(0, 10), class = b),
  prior(cauchy(0, 10), class = sigma))
m3 <- brm(
  height ~ 1 + gender + mother + father + gender:mother,
  prior = p3,
  sample_prior = "yes", 
  iter = 2000,
  warmup = 1000,
  thin = 1,
  data = d1,
  cores = parallel::detectCores(),
  control = list(adapt_delta = 0.8))

p4 <- c(
  prior(normal(80, 5), class = Intercept),
  prior(normal(0, 10), class = b),
  prior(cauchy(0, 10), class = sigma))
m4 <- brm(
  height ~ 1 + gender + mother + father + gender:father,
  prior = p4,
  sample_prior = "yes", 
  iter = 2000,
  warmup = 1000,
  thin = 1,
  data = d1,
  cores = parallel::detectCores(),
  control = list(adapt_delta = 0.8))



## Interlude: get model prior and posterior distributions

mod <- m4 # Chose the model you want to assess
prior <- prior_samples(mod)
post <- posterior_samples(mod)
head(prior)
head(post)



## II. Prior predictive checking

prior_summary(mod)

# Prior distribution
sample_mu <- rnorm(1e4, 70, 10) # prior on mu
sample_sigma <- runif(1e4, 0, 10) # prior on sigma

data.frame(x = rnorm(1000, sample_mu, sample_sigma) ) %>%
  ggplot(aes(x) ) +
  geom_histogram(col = "white") 


# Prior samples distribution
prior %>% 
  sample_n(size = 1e2) %>% 
  rownames_to_column("draw") %>% 
  expand(nesting(draw, Intercept, b), a = c(-2, 2) ) %>%
  mutate(d = Intercept + b * a) %>% 
  ggplot(aes(x = a, y = d)) +
  geom_line(aes(group = draw), color = "steelblue", size = 0.5, alpha = 0.5) +
  labs(x = "weight", y = "height")
# Visualize n predictions estimated from prior distribution



## III. Convergence diagnostics (if fail: go back to I. / modify fitting parameters. Check Prior distributions @end)
mod %>%
  plot(
    pars = "^b_",
    combo = c("dens_overlay", "trace"), widths = c(1, 1.5),
    theme = theme_bw(base_size = 14, base_family = "Open Sans")  )



## IV. Posterior predictive checking (if fail: go back to I. or add data. Check Prior distributions @end)

pp_check(mod, nsamples = 1e2) + theme_bw(base_size = 20)
pp_check(mod, type = "intervals", nsamples = 1e2, prob = 0.5, prob_outer = 0.95) +
  labs(y = "Height")
posterior_summary(mod, pars = c("^b_", "sigma"), probs = c(0.025, 0.975) )






## V. Model comparisons

m1 <- add_criterion(m1, "waic")
m2 <- add_criterion(m2, "waic")
m3 <- add_criterion(m3, "waic")
m4 <- add_criterion(m4, "waic")

# 1. WAIC with Leave-One-Out
model_comparison_table <- loo_compare(m1, m2, m3, m4, criterion = "waic") %>%
  data.frame %>%
  rownames_to_column(var = "model")

# 2. Akaike weights
weights <- data.frame(weight = model_weights(m1, m2, m3, m4, weights = "waic") ) %>%
  round(digits = 3) %>%
  rownames_to_column(var = "model")
# weights gives which model generates distributions that fits the best data given other models: Akaike weight

# 3. Bayes factor -> Better use bayes_factor (in-factor) to compare hypotheses about models
bayes_R2(mod)
hist(bayes_R2(mod, summary = FALSE), showMode = TRUE, xlab = expression(rho))
plotPost(bayes_R2(mod, summary = FALSE) - bayes_R2(m2, summary = FALSE) ) # compare R2 between m3 and m2

# 4. Summary
left_join(model_comparison_table, weights, by = "model")

summary(mod) # marginal means

dev.m1 <- mean(-2 * rowSums(log_lik(m1) ) )
dev.m2 <- mean(-2 * rowSums(log_lik(m2) ) )
dev.m3 <- mean(-2 * rowSums(log_lik(m3) ) )
dev.m4 <- mean(-2 * rowSums(log_lik(m4) ) )
deviances <- c(dev.m1, dev.m2, dev.m3, dev.m4)
comparison <- model_comparison_table %>% data.frame %>% select(waic) %>% rownames_to_column()
waics <- comparison %>% arrange(rowname) %>% pull(waic)


# Optional: parameter R2 (for y~b_param+sigma)
beta <- post$b_mother
sigma <- post$sigma
f1 <- beta^2 * var(d1$height)
rho <- f1 / (f1 + sigma^2)
hist(rho, showMode = TRUE, xlab = expression(rho))



## VI. Visualize posterior predictions

plot(conditional_effects(mod), effects = "gender:mother")
# Conditional_effects given the model









## Tools: Prior and Posterior checking

# PRIOR
# Visualize distribution of prior parameters
prior <- data.frame(cbind(sample_mu, sample_sigma) ) # multivariate prior
H.scv <- Hscv(x = prior, verbose = TRUE)
fhat_prior <- kde(x = prior, H = H.scv, compute.cont = TRUE)

plot(
  fhat_prior, display = "persp", col = "steelblue", border = NA,
  xlab = "\nmu", ylab = "\nsigma", zlab = "\n\np(mu, sigma)",
  shade = 0.8, phi = 30, ticktype = "detailed",
  cex.lab = 1.2, family = "Helvetica")

# Prior on intercept
data.frame(value1 = rnorm(1e4, 70, 10) ) %>% # 10.000 samples from Normal(70, 10)
  ggplot(aes(value1) ) +
  geom_histogram(col = "white")

# Prior on beta
data.frame(value2 = rnorm(1e4, 0, 10) ) %>% 
  ggplot(aes(value2) ) +
  geom_histogram(col = "white")

# Full prior
data.frame(x = rnorm(100, prior$Intercept + prior$b, prior$sigma) ) %>%
  ggplot(aes(x) ) +
  geom_histogram(col = "white")
# Check the form that prior distribution can take (random process, generated from prior intercept, beta and sigma)






# POSTERIOR
# Visualize distribution of posterior parameters
H.scv <- Hscv(post[, 1:2])
fhat_post <- kde(x = post[, 1:2], H = H.scv, compute.cont = TRUE)

plot(fhat_post, display = "persp", col = "purple", border = NA,
     xlab = "\nmu", ylab = "\nsigma", zlab = "\np(mu, sigma)",
     shade = 0.8, phi = 30, ticktype = "detailed",
     cex.lab = 1.2, family = "Helvetica")

# Posterior samples check (with Autocorrelation for each parameter)
post <- posterior_samples(mod, add_chain = TRUE)
post %>% mcmc_acf(pars = vars(b_Intercept:sigma), lags = 10)

# Multicolinearity check (pain in the eyes; to check along with MCMC chains convergence)
pairs(mod)

# Visualize posterior single parameter distribution
hist(post$b_Intercept + post$b_gender, breaks = 40, xlab = expression(mu))
hist(post$sigma, breaks = 40, xlab = expression(mu))
# Distributions of Intercept, beta, sigma; 2nd line may not work

# Visualize posterior samples (spaghetti plot)
posterior_samples(mod, pars = "^b_") %>%
  sample_n(size = 1e2) %>%
  rownames_to_column("draw") %>%
  expand(nesting(draw, b_Intercept, b_mother), a = c(-2.5, 3.5) ) %>%
  mutate(d = b_Intercept + b_mother * a) %>%
  ggplot(aes(x = a, y = d) ) +
  geom_point(data = d1, aes(x = mother, y = height), size = 2) +
  geom_line(aes(group = draw), color = "purple", size = 0.5, alpha = 0.5) +
  labs(x = "mother height", y = "height")






# MODEL
# Represent model predictions (specifically for 1 variable)
mother.seq <- data.frame(mother = seq(from = min(d1$mother), to = max(d1$mother), 
                                      length.out = 1e2) ) # create a vector of all possible values for y

mu <- data.frame(
  fitted(mod, newdata = mother.seq, probs = c(0.025, 0.975)) ) %>% 
  bind_cols(mother.seq) # we get the model predictions for those values: we obtain mu

pred_height <- data.frame(
  predict(mod, newdata = mother.seq, probs = c(0.025, 0.975) )) %>%
  bind_cols(mother.seq) # we obtain sigma for the model

d1 %>%
  ggplot(aes(x = mother, y = height) ) +
  geom_point(colour = "white", fill = "black", pch = 21, size = 3, alpha = 0.8) +
  geom_ribbon(
    data = pred_height, aes(x = mother, ymin = Q2.5, ymax = Q97.5),
    alpha = 0.2, inherit.aes = FALSE  ) +
  geom_smooth(
    data = mu, aes(y = Estimate, ymin = Q2.5, ymax = Q97.5),
    stat = "identity", color = "black", alpha = 0.8, size = 1 )



# Model averaging 
new_data <- data.frame(
  mother = seq(from = min(d1$mother), to = max(d1$mother), length.out = 30),
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

BEST::plotPost(
  samples, cex = 2, cex.axis = 1.5, cex.lab = 2,
  xlab = expression(param),
  ROPE = c(res$ROPE_low, res$ROPE_high), compVal = 0.5) # Beautiful as dope






# HYPOTHESIS TEST

# SOLUTION 1: [bayestestr bayes_factor: HDI+ROPE ; point_estimate (savage-dickey) + plots]
# Point estimate (Savage-Dickey)
BF_param = bayesfactor_parameters(mod, null = 0) # compares prior and posterior samples at 1 point
plot(BF_param)

BFrope_parram=bf_rope(mod)
BFrope_parram # reads as any BF, must be >1 
plot(BF_param)


# SOLUTION 3: compare model 1 with effect of interest against model 2 without effect of interest









## Multi-levels example

setwd("C:/Users/vguigon/Dropbox (Personnelle)/Cursus/Divers/Mooc/Bayes - IMSB2021")
df <- read.csv("Cours08/data/robot.csv")

mod <- brm(
  wait ~ 1 + afternoon + (1 + afternoon | cafe),
  prior = c(
    set_prior("normal(0, 10)", class = "Intercept"),
    set_prior("normal(0, 10)", class = "b"),
    set_prior("cauchy(0, 2)", class = "sigma"),
    set_prior("cauchy(0, 2)", class = "sd")
  ),
  data = df,
  warmup = 1000, iter = 5000,
  cores = parallel::detectCores()
)

# Check distributions when accounting for correlation in random effects
post <- posterior_samples(mod) # extracts posterior samples
R <- rlkjcorr(16000, K = 2, eta = 2) # samples from prior

data.frame(prior = R[, 1, 2], posterior = post$cor_cafe__Intercept__afternoon) %>%
  gather(type, value, prior:posterior) %>%
  ggplot(aes(x = value, color = type, fill = type) ) +
  geom_histogram(position = "identity", alpha = 0.2) +
  labs(x = expression(rho), y = "Nombre d'�chantillons")



# Other example of multi-level

d3 <- read.csv("Cours10/data/apples.csv")

p5 <- c(
  prior(normal(0, 10), class = Intercept),
  prior(normal(0, 10), class = b),
  prior(cauchy(0, 10), class = sd),
  prior(cauchy(0, 10), class = sigma),
  prior(lkj(2), class = cor))

m5 <- brm(
  diam ~ 1 + time + (1 + time | tree / apple),
  prior = p5,
  data = d3,
  cores = parallel::detectCores(),
  control = list(adapt_delta = 0.99))
  # backend = "cmdstanr")

post <- posterior_samples(m5, "b") # extracts posterior samples

ggplot(data = d3, aes(x = time, y = diam) ) +
  geom_point(alpha = 0.5, shape = 1) +
  geom_abline(
    data = post, aes(intercept = b_Intercept, slope = b_time),
    alpha = 0.01, size = 0.5) +
  labs(x = "Temps", y = "Diam�tre")

d3 %>%
  group_by(tree, apple) %>%
  data_grid(time = seq_range(time, n = 1e2) ) %>%
  add_fitted_samples(m5, n = 1e2) %>%
  ggplot(aes(x = time, y = diam, colour = factor(apple) ) ) +
  geom_line(
    aes(y = estimate, group = paste(apple, .iteration) ),
    alpha = 0.2, show.legend = FALSE) +
  facet_wrap(~tree, ncol = 5) +
  labs(x = "Temps", y = "Diam�tre")








### Predict continuous y with categorical x ###
## Categorical predictors

# create dummy variables, e.g.,
df$clade.NWM <- ifelse(df$clade == "New World Monkey", 1, 0)
df$clade.OWM <- ifelse(df$clade == "Old World Monkey", 1, 0)
df$clade.S <- ifelse(df$clade == "Strepsirrhine", 1, 0)

# model
mod1 <- brm(
  kcal.per.g ~ 1 + clade.NWM + clade.OWM + clade.S,
  prior(normal(0.6, 10), class = Intercept),
  prior(normal(0, 1), class = b),
  prior(exponential(0.01), class = sigma),
  family = gaussian,
  data = df) # model is: �(i) = intercept + b*categ(i)

summary(mod1)
post <- posterior_samples(mod1)

# retrieves posterior samples for each category (k-1 dummy variables)
mu.ape <- post$b_Intercept # ape will be our intercept
mu.NWM <- post$b_Intercept + post$b_clade.NWM
mu.OWM <- post$b_Intercept + post$b_clade.OWM
mu.S <- post$b_Intercept + post$b_clade.S
precis(data.frame(mu.ape, mu.NWM, mu.OWM, mu.S), prob = 0.95) # displays a summary of the posterior samples

# difference between 2 groups
diff.NWM.OWM <- mu.NWM - mu.OWM 
quantile(diff.NWM.OWM, probs = c(0.025, 0.5, 0.975) )
plotPost(diff.NWM.OWM, compVal = 0, ROPE = c(-0.1, 0.1) )


mod2 <- brm(
  kcal.per.g ~ 1 + clade.NWM + clade.OWM + clade.S,
  prior(normal(0.6, 10), class = b),
  prior(exponential(0.01), class = sigma),
  family = gaussian,
  data = df) # model is: �(i) = intercept(categ[i]) = no intercept but a categorical predictor

summary(mod2) # get estimate for each categ